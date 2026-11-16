#!/usr/bin/env python3
"""
Stateful GRU Training Pipeline for GuitarToDIBass (`1 -> 48 GRU -> 1 Dense` with residual skip).

- Loads paired time-aligned WAV files from `ml_training/dataset/shifted_guitar/` (produced by
  `OfflineShifterCLI`) and `ml_training/dataset/target_bass/`.
- Supports style filtering (`--style-filter all|FS|PK|SS|B1_FS|B1_PK|...` or comma-separated lists)
  for all 10 built-in presets and master model.
- Includes `--train-all-presets` to load the full 1,008-clip dataset once and sequentially train and
  export all 10 built-in presets (plus fingerstyle and picked generic presets) in RTX 3070 GPU VRAM.
- Performs a pitch-grouped leakage-free train/validation split (80% train / 20% val)
  stratified across all 28 semitones (MIDI 28..55, E1..G3) before fitting peak normalization.
- Preloads datasets directly into contiguous GPU VRAM tensors, achieving zero CPU-to-GPU transfers
  during epochs and maximum GPU compute utilization.
- Trains `ResidualGRUBassNet` using Truncated Backpropagation Through Time (TBPTT) with
  a numerically stable cross-session composite loss:
  1. 4-Resolution Multi-Resolution STFT Loss (FFTs 512, 1024, 2048, 4096).
  2. Sub-band Log-RMS Envelope Loss (Low/Fundamental 30-250 Hz, Body 250-1800 Hz, multi-scale pooling).
  3. Pre-emphasized Time-Domain Error-to-Signal Ratio (ESR) Loss.
  Composite loss: 0.80 * loss_spec + 0.15 * loss_env + 0.05 * loss_time
- Exports best validation checkpoints directly to RTNeural JSON format via `export_rtneural.py`.
"""

import argparse
import copy
import re
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from export_rtneural import HAS_TORCH, export_state_dict_to_json

if HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from export_rtneural import ResidualGRUBassNet

_GLOBAL_HANN_WINDOWS: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def get_cached_hann_window(win_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (win_length, device, dtype)
    win = _GLOBAL_HANN_WINDOWS.get(key)
    if win is None:
        win = torch.hann_window(win_length, periodic=True, dtype=dtype, device=device)
        _GLOBAL_HANN_WINDOWS[key] = win
    return win


def load_mono_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Lightweight pure-Python binary WAV loader (supports 16-bit PCM and 32-bit float)."""
    with open(path, "rb") as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"Not a valid RIFF/WAVE file: {path}")

        audio_format = None
        num_channels = None
        sample_rate = None
        bits_per_sample = None
        raw_bytes = None

        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[:4]
            chunk_size = struct.unpack("<I", chunk_header[4:8])[0]

            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                audio_format, num_channels, sample_rate = struct.unpack("<HHI", fmt_data[:8])
                bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
                if audio_format == 0xFFFE and chunk_size >= 40:
                    audio_format = struct.unpack("<H", fmt_data[24:26])[0]
            elif chunk_id == b"data":
                raw_bytes = f.read(chunk_size)
            else:
                f.seek(chunk_size, 1)
            if chunk_size & 1:
                f.seek(1, 1)

    if audio_format == 3 and bits_per_sample == 32:
        data = np.frombuffer(raw_bytes, dtype=np.float32)
    elif audio_format == 1 and bits_per_sample == 16:
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported WAV format ({audio_format}, {bits_per_sample} bits) in {path}")

    if num_channels > 1:
        data = data.reshape(-1, num_channels).mean(axis=1)
    return data.astype(np.float32), sample_rate


def parse_filename_info(filename: str) -> Tuple[str, int]:
    """Extracts (category, midi_pitch) from real_{CATEGORY}_M{pitch}_..."""
    m = re.match(r"real_([A-Za-z0-9_]+)_M(\d+)_", filename)
    if m:
        return m.group(1), int(m.group(2))
    return "ALL", 0


def matches_filter(cat: str, style_filter: str) -> bool:
    """Checks if a sound category matches the filter string or comma-separated list."""
    if style_filter == "all":
        return True
    if style_filter == "FS":
        return cat == "FS" or cat.endswith("_FS")
    if style_filter == "PK":
        return cat == "PK" or cat.endswith("_PK")
    target_cats = set(c.strip() for c in style_filter.split(","))
    return cat in target_cats


def load_all_dataset_pairs(
    shifted_dir: Path, target_dir: Path
) -> List[Tuple[np.ndarray, np.ndarray, str, int, str]]:
    """Loads all paired audio from disk into memory once."""
    shifted_files = sorted(shifted_dir.glob("*.wav"))
    if not shifted_files:
        raise FileNotFoundError(f"No WAV files found in {shifted_dir}. Run build_real_paired_dataset.py first.")

    dataset_items = []
    for sh_path in shifted_files:
        tg_path = target_dir / sh_path.name
        if not tg_path.exists():
            continue

        x, sr_x = load_mono_wav(sh_path)
        y, sr_y = load_mono_wav(tg_path)
        if sr_x != sr_y:
            raise ValueError(f"Sample rate mismatch: {sh_path} ({sr_x}) vs {tg_path} ({sr_y})")

        min_len = min(len(x), len(y))
        if min_len < 2048:
            continue
        x = x[:min_len]
        y = y[:min_len]

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        cat, midi_pitch = parse_filename_info(sh_path.name)
        dataset_items.append((x, y, cat, midi_pitch, sh_path.name))

    return dataset_items


def filter_and_split_dataset(
    dataset_items: List[Tuple[np.ndarray, np.ndarray, str, int, str]], style_filter: str = "all"
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Filters items by category and creates a leakage-free pitch-stratified 80/20 train/val split."""
    grouped_pairs: Dict[Tuple[str, int], List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    for x, y, cat, midi_pitch, fname in dataset_items:
        if matches_filter(cat, style_filter):
            grouped_pairs[(cat, midi_pitch)].append((x, y))

    if not grouped_pairs:
        raise RuntimeError(f"No WAV files matched style-filter '{style_filter}'.")

    train_raw: List[Tuple[np.ndarray, np.ndarray]] = []
    val_raw: List[Tuple[np.ndarray, np.ndarray]] = []

    for grp_key in sorted(grouped_pairs.keys()):
        takes = grouped_pairs[grp_key]
        n_takes = len(takes)
        if n_takes >= 2:
            val_count = max(1, int(round(n_takes * 0.20)))
            train_raw.extend(takes[:-val_count])
            val_raw.extend(takes[-val_count:])
        else:
            train_raw.append(takes[0])

    if not train_raw:
        raise RuntimeError(f"No training pairs found after filtering (style_filter={style_filter}).")
    if not val_raw:
        x0, y0 = train_raw.pop()
        split_pt = int(len(x0) * 0.8)
        train_raw.append((x0[:split_pt], y0[:split_pt]))
        val_raw.append((x0[split_pt:], y0[split_pt:]))

    # Normalize strictly using training set peak amplitude (leakage-free)
    train_peak = max(float(np.max(np.abs(x))) for x, _ in train_raw)
    scale = train_peak if train_peak > 1e-4 else 1.0

    train_pairs = [(x / scale, y / scale) for x, y in train_raw]
    val_pairs = [(x / scale, y / scale) for x, y in val_raw]

    print(
        f"[collect_aligned_dataset] Filter '{style_filter}': {len(train_pairs)} train pairs, "
        f"{len(val_pairs)} val pairs across {len(grouped_pairs)} pitch groups. Scale={scale:.4f}"
    )
    return train_pairs, val_pairs


def collect_aligned_dataset(
    shifted_dir: Path, target_dir: Path, style_filter: str = "all"
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Compatibility loader that reads from disk and performs filtered split."""
    dataset_items = load_all_dataset_pairs(shifted_dir, target_dir)
    return filter_and_split_dataset(dataset_items, style_filter=style_filter)


if HAS_TORCH:
    class NativeMultiResolutionSTFTLoss(nn.Module):
        """
        Pure-PyTorch Multi-Resolution STFT Loss (Spectral Convergence + Log STFT Magnitude Loss)
        across FFT sizes [512, 1024, 2048, 4096].
        Stabilized with epsilon in both numerator and denominator to prevent NaN on silent frames.
        """
        def __init__(
            self,
            fft_sizes: Tuple[int, ...] = (512, 1024, 2048, 4096),
            hop_sizes: Tuple[int, ...] = (128, 256, 512, 1024),
            win_lengths: Tuple[int, ...] = (512, 1024, 2048, 4096),
        ):
            super().__init__()
            self.fft_sizes = fft_sizes
            self.hop_sizes = hop_sizes
            self.win_lengths = win_lengths

        def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            if y_pred.ndim == 3:
                y_pred = y_pred.squeeze(-1) if y_pred.shape[-1] == 1 else y_pred.squeeze(1)
            if y_true.ndim == 3:
                y_true = y_true.squeeze(-1) if y_true.shape[-1] == 1 else y_true.squeeze(1)

            total_loss = 0.0
            for n_fft, hop, win_len in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
                if y_pred.shape[-1] < n_fft:
                    continue
                window = get_cached_hann_window(win_len, y_pred.device, y_pred.dtype)
                pred_stft = torch.stft(
                    y_pred, n_fft=n_fft, hop_length=hop, win_length=win_len, window=window, return_complex=True
                )
                true_stft = torch.stft(
                    y_true, n_fft=n_fft, hop_length=hop, win_length=win_len, window=window, return_complex=True
                )

                # +2e-3 magnitude floor suppresses sub--60 dBFS AD/room noise floor so models never self-oscillate to fill silence
                pred_mag = torch.abs(pred_stft) + 2e-3
                true_mag = torch.abs(true_stft) + 2e-3

                sc_loss = torch.norm(true_mag - pred_mag, p="fro") / (torch.norm(true_mag, p="fro") + 1e-6)
                mag_loss = torch.mean(torch.abs(torch.log(true_mag) - torch.log(pred_mag)))
                total_loss = total_loss + (sc_loss + mag_loss)

            return total_loss / float(len(self.fft_sizes))

    def esr_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Pre-emphasized Error-to-Signal Ratio (ESR) loss with mean-energy normalization."""
        pred_emp = y_pred[:, 1:, :] - 0.85 * y_pred[:, :-1, :]
        true_emp = y_true[:, 1:, :] - 0.85 * y_true[:, :-1, :]
        error_energy = torch.mean((true_emp - pred_emp) ** 2, dim=1)
        target_energy = torch.mean(true_emp ** 2, dim=1)
        return torch.mean(error_energy / (target_energy + 1e-3))

    def subband_envelope_loss(y_pred: torch.Tensor, y_true: torch.Tensor, sr: int = 48000) -> torch.Tensor:
        """
        Sub-Band Log-Envelope Loss: compares log-RMS envelopes in critical bass bands
        (Low/Fundamental 30-250 Hz, Upper-Harmonic/Body 250-1800 Hz) plus multi-scale
        time-domain pooled envelopes [128, 512].
        """
        if y_pred.ndim == 3:
            y_pred_2d = y_pred.squeeze(-1) if y_pred.shape[-1] == 1 else y_pred.squeeze(1)
        else:
            y_pred_2d = y_pred

        if y_true.ndim == 3:
            y_true_2d = y_true.squeeze(-1) if y_true.shape[-1] == 1 else y_true.squeeze(1)
        else:
            y_true_2d = y_true

        n_fft = 1024
        hop = 128
        win = get_cached_hann_window(n_fft, y_pred.device, y_pred.dtype)

        p_stft = torch.stft(y_pred_2d, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)
        t_stft = torch.stft(y_true_2d, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)

        p_pwr = torch.abs(p_stft) ** 2
        t_pwr = torch.abs(t_stft) ** 2

        # Low/Fundamental band: 30 - 250 Hz
        bin_low_start = max(1, int(30 * n_fft / sr))
        bin_low_end = max(bin_low_start + 1, int(250 * n_fft / sr))

        # Upper-Harmonic/Body band: 250 - 1800 Hz
        bin_high_start = bin_low_end
        bin_high_end = max(bin_high_start + 1, int(1800 * n_fft / sr))

        # Presence/Air band: 1800 - 4500 Hz (suppresses upper-band synthetic whistle/hiss)
        bin_air_start = bin_high_end
        bin_air_end = max(bin_air_start + 1, int(4500 * n_fft / sr))

        p_low = torch.sqrt(torch.clamp(torch.sum(p_pwr[:, bin_low_start:bin_low_end, :], dim=1), min=1e-6))
        t_low = torch.sqrt(torch.clamp(torch.sum(t_pwr[:, bin_low_start:bin_low_end, :], dim=1), min=1e-6))

        p_high = torch.sqrt(torch.clamp(torch.sum(p_pwr[:, bin_high_start:bin_high_end, :], dim=1), min=1e-6))
        t_high = torch.sqrt(torch.clamp(torch.sum(t_pwr[:, bin_high_start:bin_high_end, :], dim=1), min=1e-6))

        p_air = torch.sqrt(torch.clamp(torch.sum(p_pwr[:, bin_air_start:bin_air_end, :], dim=1), min=1e-6))
        t_air = torch.sqrt(torch.clamp(torch.sum(t_pwr[:, bin_air_start:bin_air_end, :], dim=1), min=1e-6))

        # Multi-scale AC RMS envelopes over windows 128 and 512 (mean-subtracted so constant DC cannot lower loss)
        def _ac_rms(sig: torch.Tensor, kernel: int, stride: int) -> torch.Tensor:
            sig_u = sig.unsqueeze(1)
            m1 = F.avg_pool1d(sig_u, kernel_size=kernel, stride=stride)
            m2 = F.avg_pool1d(sig_u * sig_u, kernel_size=kernel, stride=stride)
            return torch.sqrt(torch.clamp(m2 - m1 * m1, min=1e-6))

        p_env1 = _ac_rms(y_pred_2d, 128, 64)
        t_env1 = _ac_rms(y_true_2d, 128, 64)

        p_env2 = _ac_rms(y_pred_2d, 512, 128)
        t_env2 = _ac_rms(y_true_2d, 512, 128)

        # +2e-3 floor (~ -54 dBFS) prevents quiet room/pre-onset noise from rewarding artificial bias/oscillation
        loss_low = torch.mean(torch.abs(torch.log(t_low + 2e-3) - torch.log(p_low + 2e-3)))
        loss_high = torch.mean(torch.abs(torch.log(t_high + 2e-3) - torch.log(p_high + 2e-3)))
        loss_air = torch.mean(torch.abs(torch.log(t_air + 2e-3) - torch.log(p_air + 2e-3)))
        loss_env1 = torch.mean(torch.abs(torch.log(t_env1 + 2e-3) - torch.log(p_env1 + 2e-3)))
        loss_env2 = torch.mean(torch.abs(torch.log(t_env2 + 2e-3) - torch.log(p_env2 + 2e-3)))

        return 0.30 * loss_low + 0.30 * loss_high + 0.15 * loss_air + 0.125 * loss_env1 + 0.125 * loss_env2

    class AntiFuzzPurityLoss(nn.Module):
        """
        Probes single tones (41..1397 Hz, -32..-9 dBFS), intervals, and a dense 49-bin Schroeder odd multitone
        (56..1856 Hz) so >=4th-harmonic fuzz, chord IMD, and boxy EQ resonance/cliffs are suppressed together.
        """
        def __init__(self, sr: float = 48000.0):
            super().__init__()
            probe_len, ss_start = 4096, 1536
            self.ss_start = ss_start
            fft_len = probe_len - ss_start
            freqs = np.fft.rfftfreq(fft_len, 1.0 / sr)
            t_ax = np.arange(probe_len, dtype=np.float32) / sr

            specs = [
                ([41.2], 0.03), ([41.2], 0.12), ([55.0], 0.04), ([55.0], 0.15), ([55.0], 0.30),
                ([82.4], 0.05), ([123.5], 0.08), ([185.0], 0.08), ([277.2], 0.07),
                ([415.3], 0.06), ([622.3], 0.05), ([932.3], 0.04), ([1396.9], 0.03),
                ([41.2, 61.7], 0.12), ([55.0, 82.4], 0.14), ([110.0, 164.8], 0.12),
            ]
            n_single = len(specs)
            self.n_single = n_single
            x_np = np.zeros((n_single + 1, probe_len, 1), dtype=np.float32)
            m_fund = np.zeros((n_single, len(freqs)), dtype=np.float32)
            m_warm = np.zeros((n_single, len(freqs)), dtype=np.float32)
            m_fuzz = np.zeros((n_single, len(freqs)), dtype=np.float32)
            hann_np = np.hanning(fft_len).astype(np.float32)

            for i, (flist, amp) in enumerate(specs):
                per_tone = amp / np.sqrt(len(flist))
                for f0 in flist:
                    x_np[i, :, 0] += per_tone * np.sin(2.0 * np.pi * f0 * t_ax)
                    m_fund[i] = np.maximum(m_fund[i], (np.abs(freqs - f0) <= 22.0).astype(np.float32))
                    if len(flist) == 1:
                        for h in (2, 3):
                            if h * f0 < 12000.0:
                                m_warm[i] = np.maximum(m_warm[i], (np.abs(freqs - h * f0) <= 22.0).astype(np.float32))
                m_fuzz[i] = ((freqs >= 15.0) & (freqs <= 12000.0)).astype(np.float32) * (1.0 - m_fund[i]) * (1.0 - m_warm[i])

            # Row n_single: dense Schroeder odd-bin multitone (37.5 Hz spacing across 56..1856 Hz; zero even-order IMD overlap)
            mt_bins = np.arange(3, 100, 2, dtype=np.int64)
            per_mt = 0.025 * float(np.sqrt(2.0 / len(mt_bins)))
            for idx_m, k in enumerate(mt_bins):
                fk = float(k) * sr / float(fft_len)
                phi = np.pi * float(idx_m * idx_m) / float(len(mt_bins))
                x_np[n_single, :, 0] += per_mt * np.cos(2.0 * np.pi * fk * t_ax + phi)

            e_in_41 = np.array(
                [float((np.abs(np.fft.rfft(x_np[i, ss_start:, 0] * hann_np)) ** 2 * m_fund[i]).sum() + 1e-8) for i in (0, 1)],
                dtype=np.float32,
            )
            e_in_mt = (np.abs(np.fft.rfft(x_np[n_single, ss_start:, 0])) ** 2)[mt_bins].astype(np.float32) + 1e-12
            mt_freqs = np.concatenate(([41.2], mt_bins * (sr / float(fft_len))))
            # Allow -6 dB below 800 Hz, easing to -10 dB at >=1600 Hz so palm-muted/warm tones stay natural
            lo_bound = np.where(
                mt_freqs <= 800.0,
                -0.69,
                -0.69 - 0.46 * np.clip((mt_freqs - 800.0) / 800.0, 0.0, 1.0),
            ).astype(np.float32)

            self.register_buffer("x_clean", torch.from_numpy(x_np))
            self.register_buffer("m_fund", torch.from_numpy(m_fund))
            self.register_buffer("m_warm", torch.from_numpy(m_warm))
            self.register_buffer("m_fuzz", torch.from_numpy(m_fuzz))
            self.register_buffer("e_in_41", torch.from_numpy(e_in_41))
            self.register_buffer("mt_bins", torch.from_numpy(mt_bins))
            self.register_buffer("e_in_mt", torch.from_numpy(e_in_mt))
            self.register_buffer("lo_bound", torch.from_numpy(lo_bound))
            self.register_buffer("win", torch.from_numpy(hann_np))

        def forward(self, model: nn.Module) -> torch.Tensor:
            y_ss = model(self.x_clean, None)[0][:, self.ss_start :, 0]
            pwr = torch.abs(torch.fft.rfft(y_ss[: self.n_single] * self.win, dim=-1)) ** 2
            e_fund = (pwr * self.m_fund).sum(dim=-1) + 1e-8
            r_warm = torch.sqrt((pwr * self.m_warm).sum(dim=-1) / e_fund + 1e-8)
            r_fuzz = torch.sqrt((pwr * self.m_fuzz).sum(dim=-1) / e_fund + 1e-8)

            log_g_41 = 0.5 * torch.log(((pwr[:2] * self.m_fund[:2]).sum(dim=-1) + 1e-8) / self.e_in_41).mean().reshape(1)
            pwr_mt = torch.abs(torch.fft.rfft(y_ss[self.n_single])) ** 2
            log_g_mt = 0.5 * torch.log((pwr_mt[self.mt_bins] + 1e-12) / self.e_in_mt)
            log_g = torch.cat([log_g_41, log_g_mt])

            eq_dev = torch.mean(F.relu(log_g - 0.58) + F.relu(self.lo_bound - log_g))
            d1 = torch.mean(F.relu(torch.abs(log_g[1:] - log_g[:-1]) - 0.16))
            d3 = torch.mean(F.relu(torch.abs(log_g[3:] - log_g[:-3]) - 0.28))
            d6 = torch.mean(F.relu(torch.abs(log_g[6:] - log_g[:-6]) - 0.42))
            return (
                3.5 * torch.mean(r_fuzz)
                + 1.0 * torch.mean(F.relu(r_warm - 0.12))
                + 2.0 * eq_dev
                + 2.0 * (d1 + d3 + d6)
            )

        @torch.no_grad()
        def measure_thd_imd(self, model: nn.Module) -> Tuple[float, float]:
            dev, dt = self.x_clean.device, self.x_clean.dtype
            t = torch.arange(48000, device=dev, dtype=dt) / 48000.0
            win = torch.hann_window(32000, device=dev, dtype=dt)
            freqs = torch.fft.rfftfreq(32000, 1.0 / 48000.0).to(dev)
            max_thd = 0.0
            for db in (-30.0, -18.0, -9.0):
                x = (10.0 ** (db / 20.0)) * torch.sin(2.0 * np.pi * 55.0 * t)[None, :, None]
                y = model(x, None)[0].squeeze()[16000:]
                spec = torch.abs(torch.fft.rfft(y * win))
                hs = [float(spec[(freqs > 55.0 * k - 3.0) & (freqs < 55.0 * k + 3.0)].max()) for k in range(1, 13)]
                thd = 100.0 * float(np.sqrt(sum(h * h for h in hs[1:]))) / (hs[0] + 1e-12)
                max_thd = max(max_thd, thd)
            x2 = (0.09 * torch.sin(2.0 * np.pi * 41.2 * t) + 0.09 * torch.sin(2.0 * np.pi * 61.7 * t))[None, :, None]
            y2 = model(x2, None)[0].squeeze()[16000:]
            s2 = torch.abs(torch.fft.rfft(y2 * win))
            f1 = float(s2[(freqs > 39.0) & (freqs < 43.0)].max())
            f2 = float(s2[(freqs > 59.0) & (freqs < 64.0)].max())
            imd = 100.0 * float(torch.sqrt(torch.sum(s2[freqs > 150.0] ** 2))) / (float(np.sqrt(f1 * f1 + f2 * f2)) + 1e-9)
            return max_thd, imd

        @torch.no_grad()
        def measure_passband_hf(self, model: nn.Module) -> Tuple[Dict[int, float], float, float, float]:
            dev, dt = self.x_clean.device, self.x_clean.dtype
            gen = torch.Generator(device=dev).manual_seed(0)
            x_wn = 0.04 * torch.randn((1, 48000, 1), generator=gen, device=dev, dtype=dt)
            y_wn = model(x_wn, None)[0].squeeze()[16000:]
            x_ss = x_wn.squeeze()[16000:]
            Fw = torch.fft.rfftfreq(len(x_ss), 1.0 / 48000.0).to(dev)
            Xw = torch.abs(torch.fft.rfft(x_ss))
            Yw = torch.abs(torch.fft.rfft(y_wn))
            bands = (45, 75, 110, 160, 215, 320, 500, 800, 1400)
            hf: Dict[int, float] = {}
            for fc in bands:
                m = (Fw >= fc * 0.85) & (Fw <= fc * 1.15)
                gain = torch.sqrt(torch.mean(Yw[m] ** 2)) / (torch.sqrt(torch.mean(Xw[m] ** 2)) + 1e-9)
                hf[fc] = round(float(20.0 * torch.log10(gain + 1e-9)), 1)
            box_bands = (75, 110, 160, 215, 320, 500, 800)
            max_drop = max(
                max(hf[a] - hf[b] for a, b in zip(box_bands[:-1], box_bands[1:])),
                max(hf[a] - hf[b] for a, b in zip(box_bands[:-2], box_bands[2:])),
            )
            mid_floor = min(hf[fc] for fc in (75, 110, 160, 215, 320, 500))
            peak_db = max(hf.values())
            return hf, max_drop, mid_floor, peak_db


def train_pytorch(
    args,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    val_pairs: List[Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[Path] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
) -> float:
    """Trains ResidualGRUBassNet with resident GPU VRAM tensors and exports to RTNeural JSON format."""
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    out_file = output_path or args.output
    total_epochs = epochs or args.epochs
    learning_rate = lr or args.lr

    print(f"\n[train_gru_bass] Training with PyTorch on {device} ({len(train_pairs)} train, {len(val_pairs)} val) -> {out_file.name}")

    # Preload dataset into contiguous GPU VRAM tensors
    max_train_len = max(len(p[0]) for p in train_pairs)
    x_train_np = np.zeros((len(train_pairs), max_train_len, 1), dtype=np.float32)
    y_train_np = np.zeros((len(train_pairs), max_train_len, 1), dtype=np.float32)
    for i, (x_arr, y_arr) in enumerate(train_pairs):
        x_train_np[i, :len(x_arr), 0] = x_arr
        y_train_np[i, :len(y_arr), 0] = y_arr

    X_train_vram = torch.from_numpy(x_train_np).to(device, non_blocking=True)
    Y_train_vram = torch.from_numpy(y_train_np).to(device, non_blocking=True)

    max_val_len = max(len(p[0]) for p in val_pairs)
    x_val_np = np.zeros((len(val_pairs), max_val_len, 1), dtype=np.float32)
    y_val_np = np.zeros((len(val_pairs), max_val_len, 1), dtype=np.float32)
    for i, (x_arr, y_arr) in enumerate(val_pairs):
        x_val_np[i, :len(x_arr), 0] = x_arr
        y_val_np[i, :len(y_arr), 0] = y_arr

    X_val_vram = torch.from_numpy(x_val_np).to(device, non_blocking=True)
    Y_val_vram = torch.from_numpy(y_val_np).to(device, non_blocking=True)

    if device.type == "cuda":
        vram_mb = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
        gpu_name = torch.cuda.get_device_name(device)
        print(f"  VRAM Resident: {vram_mb:.1f} MB allocated on {gpu_name}")

    model = ResidualGRUBassNet(hidden_size=48).to(device)
    hidden_dim = model.hidden_size
    # Lock exact zero-input fixed point: candidate biases [2*H:] and dense bias stay strictly 0
    with torch.no_grad():
        model.gru.bias_ih_l0[2 * hidden_dim :].zero_()
        model.gru.bias_hh_l0[2 * hidden_dim :].zero_()
        model.dense.bias.zero_()
    model.dense.bias.requires_grad_(False)
    model.gru.bias_ih_l0.register_hook(
        lambda g, cut=2 * hidden_dim: torch.cat([g[:cut], torch.zeros_like(g[cut:])])
    )
    model.gru.bias_hh_l0.register_hook(
        lambda g, cut=2 * hidden_dim: torch.cat([g[:cut], torch.zeros_like(g[cut:])])
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-4)

    fft_sizes = [512, 1024, 2048, 4096]
    hop_sizes = [128, 256, 512, 1024]
    win_lengths = [512, 1024, 2048, 4096]

    mrstft_loss_fn = NativeMultiResolutionSTFTLoss(
        fft_sizes=tuple(fft_sizes),
        hop_sizes=tuple(hop_sizes),
        win_lengths=tuple(win_lengths),
    ).to(device)

    def composite_loss(y_p: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        return (
            0.80 * mrstft_loss_fn(y_p.transpose(1, 2), y_t.transpose(1, 2))
            + 0.15 * subband_envelope_loss(y_p, y_t)
            + 0.05 * esr_loss(y_p, y_t)
        )

    best_val_loss = float("inf")
    best_val_score = float("inf")
    best_state_dict = None

    # Contiguous TBPTT (9000 samples cleanly tiles 72,000-sample clips without skipping initial attacks)
    tbptt_len = args.chunk_size
    batch_size = max(1, args.batch_size)

    def iterate_vram_batches(X_t: torch.Tensor, Y_t: torch.Tensor, b_size: int, shuffle: bool):
        N = X_t.shape[0]
        order = torch.randperm(N, device=X_t.device) if shuffle else torch.arange(N, device=X_t.device)
        for i in range(0, N, b_size):
            idx = order[i : i + b_size]
            yield X_t[idx], Y_t[idx]

    def eval_chunked(x_in: torch.Tensor, h_init: torch.Tensor = None):
        # Chunk over time so cuDNN 9.1 never hits max seq_len limits on long validation clips
        h_cur = h_init
        outs = []
        for s0 in range(0, x_in.shape[1], tbptt_len):
            out_c, h_cur = model(x_in[:, s0 : s0 + tbptt_len, :], h_cur)
            outs.append(out_c)
        return torch.cat(outs, dim=1), h_cur

    with torch.no_grad():
        base_acc = 0.0
        base_cnt = 0
        for x_val, y_val in iterate_vram_batches(X_val_vram, Y_val_vram, batch_size, shuffle=False):
            n_b = x_val.shape[0]
            base_acc += float(composite_loss(x_val, y_val).item()) * n_b
            base_cnt += n_b
        baseline_val_loss = base_acc / max(base_cnt, 1)

    anti_fuzz_fn = AntiFuzzPurityLoss().to(device)
    total_len = X_train_vram.shape[1]
    num_chunks = max(1, total_len // tbptt_len)
    probe_chunks = {min(num_chunks - 1, c) for c in range(2, max(3, num_chunks), 3)}
    fuzz_chunks = {min(num_chunks - 1, 2), min(num_chunks - 1, 8)}

    t0 = time.time()
    for epoch in range(1, total_epochs + 1):
        model.train()
        train_loss_acc = torch.zeros((), device=device)
        train_steps = 0

        for x_tensor, y_tensor in iterate_vram_batches(X_train_vram, Y_train_vram, batch_size, shuffle=True):
            bs_cur = x_tensor.shape[0]
            h_state = None
            chunk_idx = 0
            for start in range(0, total_len, tbptt_len):
                end = min(total_len, start + tbptt_len)
                if end - start < 4096:
                    break
                optimizer.zero_grad(set_to_none=True)

                y_chunk = y_tensor[:, start:end, :]
                y_pred, h_state = model(x_tensor[:, start:end, :], h_state)
                h_state = h_state.detach()

                loss = composite_loss(y_pred, y_chunk)

                # Full-batch silence & -80 dBFS tail/hidden-state damping across 4 chunks/clip kills limit cycles
                if chunk_idx in probe_chunks:
                    x_sil = torch.zeros((bs_cur, 1536, 1), device=device, dtype=x_tensor.dtype)
                    x_sil[bs_cur // 2 :] = 1e-4 * torch.randn(
                        (bs_cur - bs_cur // 2, 1536, 1), device=device, dtype=x_tensor.dtype
                    )
                    y_sil, h_sil = model(x_sil, h_state)
                    tail = (y_sil - x_sil)[:, 256:, :]
                    loss = (
                        loss
                        + 8.0 * F.smooth_l1_loss(tail, torch.zeros_like(tail), beta=2e-4)
                        + 100.0 * torch.mean(tail * tail)
                        + 5.0 * torch.mean(h_sil * h_sil)
                    )
                if chunk_idx in fuzz_chunks:
                    loss = loss + anti_fuzz_fn(model)

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                    train_loss_acc += loss.detach()
                    train_steps += 1
                chunk_idx += 1

        scheduler.step()

        should_validate = epoch == 1 or epoch % 5 == 0 or epoch >= total_epochs - 4
        if not should_validate:
            continue

        avg_train_loss = float(train_loss_acc.item()) / max(train_steps, 1)
        model.eval()
        val_loss_accum = 0.0
        val_cnt = 0

        def measure_silence_db() -> Tuple[float, float, float]:
            cold_in = torch.zeros((1, 96000, 1), device=device, dtype=X_val_vram.dtype)
            cold_out, _ = eval_chunked(cold_in, None)
            cold_res = (cold_out - cold_in).squeeze()[48000:]
            cold_rms = float(torch.sqrt(torch.mean(cold_res ** 2) + 1e-24).item())

            tail_in = torch.zeros((1, 96000, 1), device=device, dtype=X_val_vram.dtype)
            lead_n = min(24000, X_val_vram.shape[1])
            tail_in[:, :lead_n, :] = X_val_vram[:1, :lead_n, :]
            tail_out, _ = eval_chunked(tail_in, None)
            tail_res = (tail_out - tail_in).squeeze()[48000:]
            tail_rms = float(torch.sqrt(torch.mean(tail_res ** 2) + 1e-24).item())
            tail_pk = float(torch.maximum(cold_res.abs().max(), tail_res.abs().max()).item())
            return (
                20.0 * float(np.log10(max(cold_rms, 1e-12))),
                20.0 * float(np.log10(max(tail_rms, 1e-12))),
                tail_pk,
            )

        with torch.no_grad():
            for x_val, y_val in iterate_vram_batches(X_val_vram, Y_val_vram, batch_size, shuffle=False):
                y_pred, _ = eval_chunked(x_val, None)
                n_b = x_val.shape[0]
                val_loss_accum += float(composite_loss(y_pred, y_val).item()) * n_b
                val_cnt += n_b
            ep_cold_db, ep_tail_db, val_sil_peak = measure_silence_db()
            ep_thd, ep_imd = anti_fuzz_fn.measure_thd_imd(model)
            ep_hf, ep_drop, ep_mid_fl, ep_pk_db = anti_fuzz_fn.measure_passband_hf(model)

        avg_val_loss = val_loss_accum / max(val_cnt, 1)
        # Hard-reject self-oscillating, high-distortion, or boxy/brick-walled checkpoints from best_state_dict selection
        if (
            torch.isfinite(torch.tensor(avg_val_loss))
            and ep_cold_db < -90.0
            and ep_tail_db < -80.0
            and ep_thd < 25.0
            and ep_imd < 12.0
            and ep_drop < 9.0
            and ep_mid_fl > -9.0
            and ep_pk_db < 7.0
            and avg_val_loss <= best_val_loss
        ):
            best_val_loss = avg_val_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  [Epoch {epoch:03d}/{total_epochs:03d}] "
            f"train={avg_train_loss:.5f} | val={avg_val_loss:.5f} | sil={val_sil_peak:.1e} | thd={ep_thd:.1f}% imd={ep_imd:.1f}% drop={ep_drop:.1f}dB | best={best_val_loss:.5f}",
            flush=True,
        )

    elapsed = time.time() - t0
    assert best_state_dict is not None, f"No checkpoint passed purity/passband audit for {out_file.name}"
    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    assert float(model.dense.bias.abs().max().item()) == 0.0
    assert float(model.gru.bias_ih_l0[2 * hidden_dim :].abs().max().item()) == 0.0
    assert float(model.gru.bias_hh_l0[2 * hidden_dim :].abs().max().item()) == 0.0
    model.eval()
    with torch.no_grad():
        cold_db, tail_db, _ = measure_silence_db()
        fin_thd, fin_imd = anti_fuzz_fn.measure_thd_imd(model)
        fin_hf, fin_drop, fin_mid_fl, fin_pk_db = anti_fuzz_fn.measure_passband_hf(model)

    delta_pct = 100.0 * (best_val_loss - baseline_val_loss) / max(baseline_val_loss, 1e-8)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    export_state_dict_to_json(best_state_dict, out_file, hidden_size=48)
    print(
        f"  [Audit] BaseVal={baseline_val_loss:.5f} -> ModelVal={best_val_loss:.5f} ({delta_pct:+.1f}%) | "
        f"ColdSil={cold_db:.1f} dBFS | TailSil={tail_db:.1f} dBFS | MaxTHD={fin_thd:.1f}% IMD={fin_imd:.1f}% | "
        f"Drop={fin_drop:.1f}dB MidFloor={fin_mid_fl:.1f}dB H(f)={fin_hf}"
    )
    print(f"  Exported -> {out_file} ({elapsed:.1f}s)", flush=True)
    assert cold_db < -80.0 and tail_db < -70.0, (
        f"Silence stability audit failed for {out_file.name}: Cold={cold_db:.1f} dBFS, Tail={tail_db:.1f} dBFS"
    )
    assert fin_thd < 25.0 and fin_imd < 12.0, (
        f"Distortion audit failed for {out_file.name}: MaxTHD={fin_thd:.1f}%, IMD={fin_imd:.1f}%"
    )
    assert fin_drop < 9.0 and fin_mid_fl > -9.0 and fin_pk_db < 7.0, (
        f"Passband audit failed for {out_file.name}: Drop={fin_drop:.1f}dB, MidFloor={fin_mid_fl:.1f}dB, Peak={fin_pk_db:.1f}dB"
    )
    return best_val_loss


ALL_BUILTIN_PRESETS = [
    {
        "name": "default_di_bass",
        "label": "1. All-Around Studio DI Bass (Master)",
        "filename": "default_di_bass.json",
        "filter": "SS,B1_FS,B1_PK,B2_FS,B2_PK,B3_FS,B3_PK",
        "epochs": 100,
        "lr": 2.0e-3,
    },
    {
        "name": "short_scale_bass",
        "label": "2. Short Scale Vintage Bass (Bridge DI)",
        "filename": "short_scale_bass.json",
        "filter": "SS",
        "epochs": 120,
        "lr": 2.0e-3,
    },
    {
        "name": "fame_baphomet_fs",
        "label": "3. Fame Baphomet 4 NTB — Fingerstyle",
        "filename": "fame_baphomet_fs.json",
        "filter": "B1_FS",
        "epochs": 120,
        "lr": 2.0e-3,
    },
    {
        "name": "fame_baphomet_pk",
        "label": "4. Fame Baphomet 4 NTB — Picked",
        "filename": "fame_baphomet_pk.json",
        "filter": "B1_PK",
        "epochs": 120,
        "lr": 2.0e-3,
    },
    {
        "name": "warwick_fs",
        "label": "5. Warwick Corvette 4 — Fingerstyle",
        "filename": "warwick_fs.json",
        "filter": "B2_FS",
        "epochs": 120,
        "lr": 2.0e-3,
    },
    {
        "name": "warwick_pk",
        "label": "6. Warwick Corvette 4 — Picked",
        "filename": "warwick_pk.json",
        "filter": "B2_PK",
        "epochs": 120,
        "lr": 2.0e-3,
    },
    {
        "name": "yamaha_bb604_fs",
        "label": "7. Yamaha BB604 Jazz — Fingerstyle",
        "filename": "yamaha_bb604_fs.json",
        "filter": "B3_FS",
        "epochs": 120,
        "lr": 2.0e-3,
    },
    {
        "name": "yamaha_bb604_pk",
        "label": "8. Yamaha BB604 Jazz — Picked",
        "filename": "yamaha_bb604_pk.json",
        "filter": "B3_PK",
        "epochs": 120,
        "lr": 2.0e-3,
    },
    {
        "name": "palm_muted_bass",
        "label": "9. Palm-Muted DI Bass (MU)",
        "filename": "palm_muted_bass.json",
        "filter": "MU",
        "epochs": 120,
        "lr": 2.0e-3,
    },
    {
        "name": "slap_bass",
        "label": "10. Slap & Pop DI Bass (SP/ST)",
        "filename": "slap_bass.json",
        "filter": "SLAP",
        "epochs": 120,
        "lr": 2.0e-3,
    },
    {
        "name": "idmt_fingerstyle_bass",
        "label": "11. IDMT Generic Fingerstyle Bass Preset",
        "filename": "idmt_fingerstyle_bass.json",
        "filter": "B1_FS,B2_FS,B3_FS",
        "epochs": 110,
        "lr": 2.0e-3,
    },
    {
        "name": "idmt_picked_bass",
        "label": "12. IDMT Generic Picked Bass Preset",
        "filename": "idmt_picked_bass.json",
        "filter": "B1_PK,B2_PK,B3_PK",
        "epochs": 110,
        "lr": 2.0e-3,
    },
]


def train_all_presets_pipeline(args):
    """Loads the 1,008-clip dataset once and trains all 10 built-in presets plus 2 generic presets."""
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("Multi-Bass Neural Preset Training Pipeline (10 Built-In Presets + 2 Generics)")
    print("================================================================================")
    print(f"Loading full dataset from:\n  Shifted: {args.shifted_dir}\n  Target:  {args.target_dir}")
    t_start = time.time()
    dataset_items = load_all_dataset_pairs(args.shifted_dir, args.target_dir)
    print(f"Loaded {len(dataset_items)} total paired clips in {time.time() - t_start:.2f}s.\n")

    results = []

    for idx, preset in enumerate(ALL_BUILTIN_PRESETS, 1):
        print("--------------------------------------------------------------------------------")
        print(f"[{idx}/{len(ALL_BUILTIN_PRESETS)}] {preset['label']}")
        print(f"  Filter: {preset['filter']} | Epochs: {preset['epochs']} | LR: {preset['lr']}")
        out_path = models_dir / preset["filename"]

        train_pairs, val_pairs = filter_and_split_dataset(dataset_items, style_filter=preset["filter"])
        best_loss = train_pytorch(
            args,
            train_pairs,
            val_pairs,
            output_path=out_path,
            epochs=preset["epochs"],
            lr=preset["lr"],
        )
        results.append(
            {
                "index": idx,
                "label": preset["label"],
                "file": preset["filename"],
                "best_val_loss": best_loss,
                "train_clips": len(train_pairs),
                "val_clips": len(val_pairs),
                "size_kb": out_path.stat().st_size / 1024.0 if out_path.exists() else 0.0,
            }
        )

    total_time = time.time() - t_start
    print("\n================================================================================")
    print(f"All {len(ALL_BUILTIN_PRESETS)} Presets Trained & Exported in {total_time / 60:.1f} minutes!")
    print("================================================================================")
    print(f"{'Idx':<4} {'Preset Name':<42} {'Val Loss':<10} {'Train/Val':<12} {'Size':<8}")
    print("-" * 80)
    for r in results:
        tv_str = f"{r['train_clips']}/{r['val_clips']}"
        print(f"{r['index']:<4} {r['label']:<42} {r['best_val_loss']:<10.5f} {tv_str:<12} {r['size_kb']:<6.1f}KB")
    print("================================================================================\n")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train Stateful GRU (1->48->1) DI Bass Timbre Model in VRAM")
    parser.add_argument(
        "--shifted-dir",
        type=Path,
        default=repo_root / "ml_training" / "dataset" / "shifted_guitar",
        help="Directory of pre-shifted guitar WAVs",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=repo_root / "ml_training" / "dataset" / "target_bass",
        help="Directory of target DI bass WAVs",
    )
    parser.add_argument(
        "--style-filter",
        type=str,
        default="all",
        help="Filter dataset by style/category: 'all', 'FS', 'PK', 'SS', 'B1_FS', 'MU', etc. or comma-separated list",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "models" / "default_di_bass.json",
        help="Output RTNeural JSON model path",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2.0e-3, help="Initial learning rate")
    parser.add_argument("--burn-in", type=int, default=0, help="Deprecated overlap burn-in (contiguous TBPTT starts at t=0)")
    parser.add_argument("--chunk-size", type=int, default=6000, help="Contiguous TBPTT chunk length in samples")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for GPU training")
    parser.add_argument(
        "--train-all-presets",
        action="store_true",
        help="Load full dataset once and sequentially train/export all 10 built-in presets + 2 generics",
    )
    args = parser.parse_args()

    if args.train_all_presets:
        train_all_presets_pipeline(args)
    else:
        train_pairs, val_pairs = collect_aligned_dataset(
            args.shifted_dir, args.target_dir, style_filter=args.style_filter
        )
        if HAS_TORCH:
            train_pytorch(args, train_pairs, val_pairs)
        else:
            raise RuntimeError("PyTorch is required for training.")


if __name__ == "__main__":
    main()
