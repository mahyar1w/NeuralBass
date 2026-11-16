#!/usr/bin/env python3
"""
build_real_paired_dataset.py

Constructs the comprehensive multi-bass paired dataset for GuitarToDIBass:
1. Short Scale Vintage Bass (SS):
   - Curates Short Scale Bass [Decent Sampler] 48kHz 24-bit samples.
   - Audit fixes:
     * GB_051_RR4.wav re-assigned from MIDI 51 to root MIDI 52 (E3, 164.81 Hz).
     * GB_028_RR3.wav pitch-corrected by factor 41.75 / 41.2034 to exact 41.203 Hz (<1 cent error).
   - Strips DC offset, trims leading silence (4ms pre-onset lead-in, 2ms fade-in).
   - Populates all 28 bass semitones (MIDI 28..55, E1..G3) by high-precision resampling of nearest root round-robin takes.
2. IDMT-SMT-BASS (8 Categories):
   - B1_FS: Fame Baphomet 4 NTB — Fingerstyle (BS_1 + FS, PS & ES/VI)
   - B1_PK: Fame Baphomet 4 NTB — Picked (BS_1 + PK, PS & ES/VI)
   - B2_FS: Warwick Corvette 4 — Fingerstyle (BS_2 + FS, PS & ES/VI)
   - B2_PK: Warwick Corvette 4 — Picked (BS_2 + PK, PS & ES/VI)
   - B3_FS: Yamaha BB604 Jazz — Fingerstyle (BS_3 + FS, PS & ES/VI)
   - B3_PK: Yamaha BB604 Jazz — Picked (BS_3 + PK, PS & ES/VI)
   - MU: Palm-Muted DI Bass (PS/MU, all 3 basses)
   - SLAP: Slap & Pop DI Bass (PS/SP & PS/ST, all 3 basses)
3. IDMT-SMT-GUITAR_V2:
   - Extracts all clean single-coil and humbucker notes (dataset1 & dataset2), MIDI 40..67.
   - Interleaves notes across all 7 guitar configurations (FSNeckSC, IbBridgeHU, IbBNegSC, IbNeckHU, Aristides, Strat, LesPaul).
4. Pairing & Shifting:
   - 9 categories * 28 pitches * 4 pairs = 1,008 paired 48kHz clips (25.2 minutes).
   - Filenames: real_{CATEGORY}_M{bass_midi:02d}_G{g_tag}_{pair_idx:04d}.wav.
   - Runs build/OfflineShifterCLI on raw_guitar/ -> shifted_guitar/.
   - Post-shift fundamental phase and slow-decay envelope alignment on target_bass/.
"""

import argparse
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.signal
import soundfile as sf

SR_OUT = 48000
OPEN_BASS_MIDI = {1: 28, 2: 33, 3: 38, 4: 43}  # String 1: E1, 2: A1, 3: D2, 4: G2
BASS_FILENAME_RE = re.compile(r"BS_(\d+)_EQ_(\d+)_(?:PS_)?([A-Z]+)_(?:ES_)?([A-Z]+)_(\d+)_(\d+)\.wav")


def clean_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s)


def load_and_preprocess_guitar(
    wav_path: Path, onset_sec: float, offset_sec: float, is_multi_event: bool = False
) -> np.ndarray:
    """Slices a guitar note event, trims leading silence, resamples 44.1k -> 48k, pads to 72,000 samples."""
    x, sr = sf.read(str(wav_path))
    if x.ndim > 1:
        x = x.mean(axis=1)

    s_start = max(0, int((onset_sec - 0.008) * sr))
    s_end = min(len(x), int((onset_sec + 1.55) * sr))
    if is_multi_event:
        s_end = min(s_end, int(offset_sec * sr))

    clip = x[s_start:s_end].copy()
    clip = clip - np.mean(clip)

    pk = float(np.max(np.abs(clip))) if len(clip) > 0 else 0.0
    if pk > 1e-4:
        on_idx = int(np.argmax(np.abs(clip) > 0.03 * pk))
        lead = max(0, on_idx - int(0.004 * sr))
        trimmed = clip[lead:].copy()
    else:
        trimmed = clip.copy()

    fade_in = min(int(0.002 * sr), len(trimmed))
    if fade_in > 0:
        trimmed[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    fade_out = min(int(0.015 * sr), len(trimmed))
    if fade_out > 0:
        trimmed[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)

    resamp = scipy.signal.resample_poly(trimmed, 160, 147).astype(np.float32)
    target_samples = 72000
    if len(resamp) < target_samples:
        resamp = np.pad(resamp, (0, target_samples - len(resamp)))
    else:
        resamp = resamp[:target_samples]
        fo = int(0.015 * SR_OUT)
        resamp[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)

    max_val = float(np.max(np.abs(resamp)))
    if max_val > 1e-5:
        resamp = (0.80 * resamp / max_val).astype(np.float32)
    return resamp


def load_and_preprocess_idmt_bass(wav_path: Path) -> np.ndarray:
    """Loads an IDMT-SMT-BASS note, trims leading silence, resamples 44.1k -> 48k, pads to 72,000 samples."""
    x, sr = sf.read(str(wav_path))
    if x.ndim > 1:
        x = x.mean(axis=1)

    x = x - np.mean(x)
    pk = float(np.max(np.abs(x))) if len(x) > 0 else 0.0
    if pk > 1e-4:
        on_idx = int(np.argmax(np.abs(x) > 0.03 * pk))
        lead = max(0, on_idx - int(0.004 * sr))
        trimmed = x[lead:].copy()
    else:
        trimmed = x.copy()

    fade_in = min(int(0.002 * sr), len(trimmed))
    if fade_in > 0:
        trimmed[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)

    resamp = scipy.signal.resample_poly(trimmed, 160, 147).astype(np.float32)
    target_samples = 72000
    if len(resamp) < target_samples:
        resamp = np.pad(resamp, (0, target_samples - len(resamp)))
    else:
        resamp = resamp[:target_samples]
        fo = int(0.015 * SR_OUT)
        resamp[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)

    max_val = float(np.max(np.abs(resamp)))
    if max_val > 1e-5:
        resamp = (0.80 * resamp / max_val).astype(np.float32)
    return resamp


def curate_short_scale_bass(ss_samples_dir: Path) -> Dict[int, List[np.ndarray]]:
    """
    Curates Short Scale Bass [Decent Sampler] samples across all 28 semitones (MIDI 28..55):
    - Reassigns GB_051_RR4.wav (f0=164.79 Hz) to MIDI 52 (E3 = 164.81 Hz).
    - Pitch-corrects GB_028_RR3.wav (f0=41.75 Hz) by factor 41.75 / 41.2034.
    - Strips DC, trims silence, applies high-precision resampling to pitch-shift
      nearest root round-robin takes across all semitones 28..55.
    """
    by_root: Dict[int, List[Path]] = defaultdict(list)
    reassigned_count = 0
    pitch_corrected_count = 0

    for w in sorted(ss_samples_dir.glob("*.wav")):
        m = re.match(r"GB_(\d+)_RR(\d+)\.wav", w.name)
        if not m:
            continue
        orig_root = int(m.group(1))

        # Audit Fix 1: GB_051_RR4.wav actual f0 is 128.75 Hz (MIDI 48 C3, not 51/52)
        if w.name == "GB_051_RR4.wav":
            by_root[48].append(w)
            reassigned_count += 1
        else:
            by_root[orig_root].append(w)

    print(f"  [Short Scale Bass Curation] Reassigned {reassigned_count} mislabeled sample (GB_051_RR4 -> MIDI 48 C3).")

    # Load cleaned root samples
    cleaned_root_samples: Dict[int, List[np.ndarray]] = defaultdict(list)
    for root_pitch, wav_list in by_root.items():
        for wav_path in wav_list:
            x, sr = sf.read(str(wav_path))
            if x.ndim > 1:
                x = x.mean(axis=1)
            x = x - np.mean(x)

            # Audit Fix 2: GB_028_RR3.wav pitch correction
            if wav_path.name == "GB_028_RR3.wav":
                resamp_factor = 41.75 / 41.2034
                new_len = int(round(len(x) * resamp_factor))
                x = scipy.signal.resample(x, new_len).astype(np.float32)
                pitch_corrected_count += 1

            # Trim silence
            pk = float(np.max(np.abs(x))) if len(x) > 0 else 0.0
            if pk > 1e-4:
                on_idx = int(np.argmax(np.abs(x) > 0.03 * pk))
                lead = max(0, on_idx - int(0.004 * sr))
                x = x[lead:].copy()

            fade_in = min(int(0.002 * sr), len(x))
            if fade_in > 0:
                x[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)

            cleaned_root_samples[root_pitch].append(x.astype(np.float32))

    print(f"  [Short Scale Bass Curation] Pitch-corrected {pitch_corrected_count} sample (GB_028_RR3 -> 41.203 Hz).")

    avail_roots = sorted(cleaned_root_samples.keys())
    ss_by_pitch: Dict[int, List[np.ndarray]] = defaultdict(list)
    target_samples = 72000

    for target_midi in range(28, 56):
        # Find nearest recorded root
        nearest_root = min(avail_roots, key=lambda r: abs(r - target_midi))
        shift_st = target_midi - nearest_root
        ratio = 2.0 ** (-shift_st / 12.0)  # Playback speed ratio

        for raw_take in cleaned_root_samples[nearest_root]:
            if shift_st == 0:
                y = raw_take.copy()
            else:
                new_l = int(round(len(raw_take) * ratio))
                y = scipy.signal.resample(raw_take, new_l).astype(np.float32)

            if len(y) < target_samples:
                y = np.pad(y, (0, target_samples - len(y)))
            else:
                y = y[:target_samples]
                fo = int(0.015 * SR_OUT)
                y[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)

            max_y = float(np.max(np.abs(y)))
            if max_y > 1e-5:
                y = (0.80 * y / max_y).astype(np.float32)

            ss_by_pitch[target_midi].append(y)

    print(
        f"  [Short Scale Bass Curation] Populated all 28 semitones (MIDI 28..55) with "
        f"{min(len(v) for v in ss_by_pitch.values())}..{max(len(v) for v in ss_by_pitch.values())} round-robin takes per pitch."
    )
    return ss_by_pitch


def extract_guitar_notes(guitar_root: Path) -> Dict[int, List[Dict]]:
    """Extracts valid clean guitar DI note events indexed by guitar MIDI pitch (40..67)."""
    raw_by_pitch = defaultdict(list)

    # 1. Dataset 1 (Isolated single-note clean takes)
    d1_folders = [
        ("Fender Strat Clean Neck SC", "FSNeckSC"),
        ("Ibanez Power Strat Clean Bridge HU", "IbBridgeHU"),
        ("Ibanez Power Strat Clean Bridge+Neck SC", "IbBNegSC"),
        ("Ibanez Power Strat Clean Neck HU", "IbNeckHU"),
    ]
    for folder_name, tag in d1_folders:
        ann_dir = guitar_root / "dataset1" / folder_name / "annotation"
        aud_dir = guitar_root / "dataset1" / folder_name / "audio"
        if not ann_dir.exists():
            continue
        for xml_path in sorted(ann_dir.glob("*.xml")):
            wav_path = aud_dir / (xml_path.stem + ".wav")
            if not wav_path.exists():
                continue
            try:
                tree = ET.parse(xml_path)
                for ev in tree.findall(".//event"):
                    p = int(ev.find("pitch").text)
                    exc = ev.find("excitationStyle").text if ev.find("excitationStyle") is not None else ""
                    expr = ev.find("expressionStyle").text if ev.find("expressionStyle") is not None else ""
                    on = float(ev.find("onsetSec").text)
                    off = float(ev.find("offsetSec").text)
                    if 40 <= p <= 67 and exc in ("PK", "FS") and expr in ("NO", "VI") and (off - on) >= 0.35:
                        raw_by_pitch[p].append(
                            {
                                "source": "d1",
                                "tag": tag,
                                "excitation": exc,
                                "onset": on,
                                "offset": off,
                                "wav_path": wav_path,
                                "is_multi": False,
                            }
                        )
            except Exception:
                pass

    # 2. Dataset 2 (All annotated fretboard, scale, and lick passages)
    d2_ann_dir = guitar_root / "dataset2" / "annotation"
    d2_aud_dir = guitar_root / "dataset2" / "audio"
    d2_tags = {"AR": "Aristides", "FS": "Strat", "LP": "LesPaul"}
    if d2_ann_dir.exists():
        for xml_path in sorted(d2_ann_dir.glob("*.xml")):
            wav_path = d2_aud_dir / (xml_path.stem + ".wav")
            if not wav_path.exists():
                continue
            prefix = xml_path.stem.split("_")[0]
            tag = d2_tags.get(prefix, prefix)
            try:
                tree = ET.parse(xml_path)
                for ev in tree.findall(".//event"):
                    p = int(ev.find("pitch").text)
                    exc = ev.find("excitationStyle").text if ev.find("excitationStyle") is not None else ""
                    expr = ev.find("expressionStyle").text if ev.find("expressionStyle") is not None else ""
                    on = float(ev.find("onsetSec").text)
                    off = float(ev.find("offsetSec").text)
                    if 40 <= p <= 67 and exc in ("PK", "FS") and expr in ("NO", "VI") and (off - on) >= 0.35:
                        raw_by_pitch[p].append(
                            {
                                "source": "d2",
                                "tag": tag,
                                "excitation": exc,
                                "onset": on,
                                "offset": off,
                                "wav_path": wav_path,
                                "is_multi": True,
                            }
                        )
            except Exception:
                pass

    # Interleave notes across the 7 configurations for diverse pickup & neck sampling
    interleaved_by_pitch = {}
    for p, notes_list in raw_by_pitch.items():
        by_tag = defaultdict(list)
        for n in notes_list:
            by_tag[n["tag"]].append(n)

        tags = sorted(by_tag.keys())
        interleaved = []
        max_t = max(len(by_tag[t]) for t in tags)
        for idx in range(max_t):
            for t in tags:
                if idx < len(by_tag[t]):
                    interleaved.append(by_tag[t][idx])
        interleaved_by_pitch[p] = interleaved

    return interleaved_by_pitch


def index_idmt_bass_categories(idmt_root: Path) -> Dict[str, Dict[int, List[Path]]]:
    """
    Indexes IDMT-SMT-BASS recordings into 8 categories across MIDI 28..55:
    - B1_FS: Fame Baphomet 4 NTB — Fingerstyle (BS_1, PS/FS & ES/VI)
    - B1_PK: Fame Baphomet 4 NTB — Picked (BS_1, PS/PK & ES/VI)
    - B2_FS: Warwick Corvette 4 — Fingerstyle (BS_2, PS/FS & ES/VI)
    - B2_PK: Warwick Corvette 4 — Picked (BS_2, PS/PK & ES/VI)
    - B3_FS: Yamaha BB604 Jazz — Fingerstyle (BS_3, PS/FS & ES/VI)
    - B3_PK: Yamaha BB604 Jazz — Picked (BS_3, PS/PK & ES/VI)
    - MU: Palm-Muted DI Bass (PS/MU, all basses)
    - SLAP: Slap & Pop DI Bass (PS/SP & PS/ST, all basses)
    """
    categories = ["B1_FS", "B1_PK", "B2_FS", "B2_PK", "B3_FS", "B3_PK", "MU", "SLAP"]
    cat_by_pitch: Dict[str, Dict[int, List[Path]]] = {c: defaultdict(list) for c in categories}

    # 1. PS directory
    for sub in ("FS", "PK", "MU", "SP", "ST"):
        p_dir = idmt_root / "PS" / sub
        if not p_dir.exists():
            continue
        for wav_path in sorted(p_dir.glob("*.wav")):
            m = BASS_FILENAME_RE.match(wav_path.name)
            if not m:
                continue
            b_id, eq_id, ps, es, str_s, fret_s = m.groups()
            b_num, str_num, fret = int(b_id), int(str_s), int(fret_s)
            if str_num not in OPEN_BASS_MIDI:
                continue
            midi = OPEN_BASS_MIDI[str_num] + fret
            if not (28 <= midi <= 55):
                continue

            if ps == "FS":
                cat_by_pitch[f"B{b_num}_FS"][midi].append(wav_path)
            elif ps == "PK":
                cat_by_pitch[f"B{b_num}_PK"][midi].append(wav_path)
            elif ps == "MU":
                cat_by_pitch["MU"][midi].append(wav_path)
            elif ps in ("SP", "ST"):
                cat_by_pitch["SLAP"][midi].append(wav_path)

    # 2. ES/VI directory (vibrato takes)
    es_vi_dir = idmt_root / "ES" / "VI"
    if es_vi_dir.exists():
        for wav_path in sorted(es_vi_dir.glob("*.wav")):
            m = BASS_FILENAME_RE.match(wav_path.name)
            if not m:
                continue
            b_id, eq_id, ps, es, str_s, fret_s = m.groups()
            b_num, str_num, fret = int(b_id), int(str_s), int(fret_s)
            if str_num not in OPEN_BASS_MIDI:
                continue
            midi = OPEN_BASS_MIDI[str_num] + fret
            if not (28 <= midi <= 55):
                continue

            if ps == "FS":
                cat_by_pitch[f"B{b_num}_FS"][midi].append(wav_path)
            elif ps == "PK":
                cat_by_pitch[f"B{b_num}_PK"][midi].append(wav_path)

    return cat_by_pitch


def align_fundamental_phase_and_envelope(
    shifted_path: Path, target_path: Path, bass_midi: int
) -> Tuple[float, float, int]:
    """
    Fine-tunes the fundamental cycle phase and slow RMS decay envelope of target_bass
    to match shifted_guitar, preventing destructive phase cancellation in training.
    """
    sh, sr = sf.read(str(shifted_path))
    tg, sr_tg = sf.read(str(target_path))
    assert sr == sr_tg == SR_OUT

    f0 = 440.0 * (2.0 ** ((bass_midi - 69) / 12.0))
    cutoff = min(1.4 * f0, 0.45 * sr)
    b, a = scipy.signal.butter(2, cutoff / (0.5 * sr), btype="low")

    filt_s = scipy.signal.filtfilt(b, a, sh)
    filt_t = scipy.signal.filtfilt(b, a, tg)

    w_start = int(0.02 * sr)
    w_end = int(0.22 * sr)
    max_lag = int(round(0.6 * sr / f0))

    seg_s = filt_s[w_start:w_end]
    best_shift = 0
    best_corr = -1.0
    corr_0 = float(np.corrcoef(seg_s, filt_t[w_start:w_end])[0, 1])

    for s in range(-max_lag, max_lag + 1):
        s0 = w_start - s
        s1 = w_end - s
        if s0 >= 0 and s1 <= len(filt_t):
            c = float(np.corrcoef(seg_s, filt_t[s0:s1])[0, 1])
            if c > best_corr:
                best_corr = c
                best_shift = s

    # Phase-align target
    if best_shift > 0:
        aligned_t = np.pad(tg, (best_shift, 0))[: len(tg)]
        fade = min(int(0.002 * sr), len(aligned_t))
        aligned_t[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    elif best_shift < 0:
        aligned_t = np.pad(tg[-best_shift:], (0, -best_shift))
        fade = min(int(0.002 * sr), len(aligned_t))
        aligned_t[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    else:
        aligned_t = tg.copy()

    # Slow RMS decay envelope matching (~35 ms low-pass)
    b_env, a_env = scipy.signal.butter(2, 28.5 / (0.5 * sr), btype="low")
    env_s = np.maximum(scipy.signal.filtfilt(b_env, a_env, np.abs(sh)), 1e-4)
    env_t = np.maximum(scipy.signal.filtfilt(b_env, a_env, np.abs(aligned_t)), 1e-4)

    raw_gain = np.clip(env_s / env_t, 0.25, 2.5)
    smooth_gain = np.clip(scipy.signal.filtfilt(b_env, a_env, raw_gain), 0.25, 2.5)

    # Gently band-limit target DI bass (< 3.5 kHz) to suppress ultrasonic room hiss while preserving pick bite
    sos_lpf = scipy.signal.butter(4, 3500.0 / (0.5 * sr), btype="low", output="sos")
    final_t = scipy.signal.sosfiltfilt(sos_lpf, aligned_t * smooth_gain).astype(np.float32)

    # Match active RMS to shifted_guitar so the GRU learns timbre residual rather than session gain offsets
    active_mask = (np.abs(sh) > 1e-3) | (np.abs(final_t) > 1e-3)
    if np.any(active_mask):
        rms_s = float(np.sqrt(np.mean(sh[active_mask] ** 2)))
        rms_t = float(np.sqrt(np.mean(final_t[active_mask] ** 2)))
        if rms_t > 1e-5:
            final_t = (final_t * float(np.clip(rms_s / rms_t, 0.5, 2.0))).astype(np.float32)

    max_t = float(np.max(np.abs(final_t)))
    if max_t > 0.90:
        final_t = (0.90 * final_t / max_t).astype(np.float32)

    # Overwrite target_bass WAV with aligned audio
    sf.write(str(target_path), final_t, sr)
    return corr_0, best_corr, best_shift


def main():
    repo_root = Path(__file__).resolve().parent.parent
    workspace_parent = repo_root.parent
    parser = argparse.ArgumentParser(description="Build Comprehensive Multi-Bass Paired Dataset")
    parser.add_argument(
        "--guitar-root",
        type=Path,
        default=workspace_parent / "IDMT-SMT-GUITAR_V2",
        help="Root directory of IDMT-SMT-GUITAR_V2",
    )
    parser.add_argument(
        "--idmt-bass-root",
        type=Path,
        default=workspace_parent / "IDMT-SMT-BASS",
        help="Root directory of IDMT-SMT-BASS",
    )
    parser.add_argument(
        "--ss-samples-dir",
        type=Path,
        default=workspace_parent / "Short Scale Bass [Decent Sampler]" / "Short Scale Bass.dsbundle" / "Samples",
        help="Samples directory of Short Scale Bass [Decent Sampler]",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "ml_training" / "dataset",
        help="Output dataset root directory",
    )
    parser.add_argument(
        "--pairs-per-pitch",
        type=int,
        default=8,
        help="Number of paired clips per pitch (MIDI 28..55) for each of the 9 bass categories",
    )
    args = parser.parse_args()

    raw_dir = args.output_dir / "raw_guitar"
    shifted_dir = args.output_dir / "shifted_guitar"
    target_dir = args.output_dir / "target_bass"

    for d in (raw_dir, shifted_dir, target_dir):
        d.mkdir(parents=True, exist_ok=True)
        # Clear existing WAV files
        for old_wav in d.glob("*.wav"):
            old_wav.unlink()

    print("================================================================================")
    print("Building Multi-Bass Paired Dataset (Short Scale + IDMT-SMT-BASS + Real Guitars)")
    print("================================================================================")

    # 1. Curate Short Scale Bass
    print("[1/5] Curating Short Scale Vintage Bass samples...")
    ss_bass_by_pitch = curate_short_scale_bass(args.ss_samples_dir)

    # 2. Index IDMT-SMT-BASS 8 Categories
    print("[2/5] Indexing IDMT-SMT-BASS across 8 sound categories...")
    idmt_by_cat = index_idmt_bass_categories(args.idmt_bass_root)
    for cat, p_map in sorted(idmt_by_cat.items()):
        print(f"  {cat:8s}: {len(p_map)}/28 pitches indexed.")

    # 3. Extract & Interleave Guitar Notes
    print("[3/5] Extracting & interleaving IDMT-SMT-GUITAR_V2 note events (dataset1 & dataset2)...")
    guitar_notes = extract_guitar_notes(args.guitar_root)
    print(f"  Extracted guitar notes across {len(guitar_notes)} pitches (MIDI 40..67).")

    # Combine all 9 bass categories
    # SS stores numpy arrays; IDMT categories store Path objects
    all_categories = ["SS", "B1_FS", "B1_PK", "B2_FS", "B2_PK", "B3_FS", "B3_PK", "MU", "SLAP"]

    pair_idx = 1
    paired_metadata = []

    print(f"[4/5] Constructing {len(all_categories)} * 28 * {args.pairs_per_pitch} = "
          f"{len(all_categories) * 28 * args.pairs_per_pitch} paired clips...")

    for cat_idx, cat in enumerate(all_categories):
        for bass_midi in range(28, 56):
            guitar_midi = bass_midi + 12
            avail_g = guitar_notes.get(guitar_midi, [])
            if not avail_g:
                raise RuntimeError(f"No guitar notes available for guitar MIDI {guitar_midi}")

            if cat == "SS":
                avail_b = ss_bass_by_pitch.get(bass_midi, [])
                is_ss = True
            else:
                avail_b = idmt_by_cat[cat].get(bass_midi, [])
                is_ss = False

            if not avail_b:
                raise RuntimeError(f"No bass notes available for {cat} bass MIDI {bass_midi}")

            len_g = len(avail_g)
            len_b = len(avail_b)

            for k in range(args.pairs_per_pitch):
                gi = (cat_idx * args.pairs_per_pitch + k) % len_g
                bi = k % len_b

                g_info = avail_g[gi]
                g_tag = clean_tag(g_info["tag"])

                filename = f"real_{cat}_M{bass_midi:02d}_G{g_tag}_{pair_idx:04d}.wav"

                # Preprocess guitar note
                g_audio = load_and_preprocess_guitar(
                    g_info["wav_path"], g_info["onset"], g_info["offset"], is_multi_event=g_info["is_multi"]
                )
                sf.write(str(raw_dir / filename), g_audio, SR_OUT)

                # Preprocess target bass note
                if is_ss:
                    b_audio = avail_b[bi]
                else:
                    b_audio = load_and_preprocess_idmt_bass(avail_b[bi])
                sf.write(str(target_dir / filename), b_audio, SR_OUT)

                paired_metadata.append(
                    {
                        "pair_idx": pair_idx,
                        "cat": cat,
                        "bass_midi": bass_midi,
                        "filename": filename,
                    }
                )
                pair_idx += 1

    total_pairs = len(paired_metadata)
    print(f"  Wrote {total_pairs} paired clips to raw_guitar/ and target_bass/.")

    # 4. Run OfflineShifterCLI on raw_guitar/ -> shifted_guitar/
    shifter_bin = repo_root / "build" / "OfflineShifterCLI"
    if not shifter_bin.exists():
        raise FileNotFoundError(f"OfflineShifterCLI binary not found at {shifter_bin}")

    print(f"[5/5] Invoking OfflineShifterCLI on {total_pairs} files...")
    cmd = [str(shifter_bin), str(raw_dir), str(shifted_dir)]
    subprocess.run(cmd, check=True)

    # Post-shift fundamental phase & slow-envelope alignment
    print("  Aligning fundamental phase and decay envelopes on target_bass/...")
    corrs_before = []
    corrs_after = []
    shifts = []

    for item in paired_metadata:
        sh_file = shifted_dir / item["filename"]
        tg_file = target_dir / item["filename"]
        if not sh_file.exists() or not tg_file.exists():
            continue

        c_before, c_after, s_samp = align_fundamental_phase_and_envelope(sh_file, tg_file, item["bass_midi"])
        corrs_before.append(c_before)
        corrs_after.append(c_after)
        shifts.append(s_samp)

    median_before = float(np.median(corrs_before))
    median_after = float(np.median(corrs_after))
    mean_shift_ms = float(np.mean(np.abs(shifts))) / (SR_OUT / 1000.0)

    print("\n=== Comprehensive Multi-Bass Dataset Alignment Diagnostics ===")
    print(f"  Total Paired Clips:             {len(corrs_after)} ({len(corrs_after) * 1.5 / 60:.1f} minutes of 48kHz audio)")
    print(f"  Median Fundamental Corr Before: {median_before:.4f}")
    print(f"  Median Fundamental Corr After:  {median_after:.4f}")
    print(f"  Min Fundamental Corr After:     {float(np.min(corrs_after)):.4f}")
    print(f"  Mean Absolute Shift:            {mean_shift_ms:.2f} ms")
    print("===============================================================\n")
    print("[build_real_paired_dataset] Dataset generation complete!")


if __name__ == "__main__":
    main()
