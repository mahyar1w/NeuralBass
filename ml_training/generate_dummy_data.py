#!/usr/bin/env python3
"""
Generates:
1. Valid initialized fallback/warm-saturation RTNeural weights at `models/default_di_bass.json`
   so the C++ plugin and CLI process audio immediately out of the box (Rule 5).
2. Synthetic paired 48 kHz test WAV files in:
   - `ml_training/dataset/raw_guitar/`
   - `ml_training/dataset/shifted_guitar/`
   - `ml_training/dataset/target_bass/`
"""

import struct
import wave
from pathlib import Path
import numpy as np

from export_rtneural import build_rtneural_json_dict
import json


def write_wav_float32_or_pcm16(path: Path, samples: np.ndarray, sample_rate: int = 48000) -> None:
    """Writes a standard 16-bit PCM mono WAV file using only Python's standard library + numpy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -0.999, 0.999)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def generate_default_di_bass_json(output_path: Path, hidden_size: int = 48) -> None:
    """
    Creates a deterministic warm-saturation / sub-harmonic thickening weight initialization
    for the 1 -> 48 GRU -> 1 Dense residual network (`y = x + net(x)`).
    """
    rng = np.random.default_rng(seed=42)

    w_ih = rng.normal(0.0, 0.08, size=(3 * hidden_size, 1)).astype(np.float32)
    w_hh = rng.normal(0.0, 0.04, size=(3 * hidden_size, hidden_size)).astype(np.float32)
    b_ih = np.zeros((3 * hidden_size,), dtype=np.float32)
    b_hh = np.zeros((3 * hidden_size,), dtype=np.float32)

    # Bias update gate (z, indices H..2H in PyTorch) slightly positive for smooth bass sustain memory
    b_ih[hidden_size:2 * hidden_size] = np.linspace(-0.2, 0.6, hidden_size, dtype=np.float32)

    # Configure first 4 hidden units as clean low-order harmonic & envelope follower units
    w_ih[2 * hidden_size + 0, 0] = 0.65
    w_ih[2 * hidden_size + 1, 0] = -0.45
    w_ih[2 * hidden_size + 2, 0] = 0.30
    w_hh[2 * hidden_size + 0, 0] = 0.78
    w_hh[2 * hidden_size + 1, 1] = 0.68
    w_hh[2 * hidden_size + 2, 2] = 0.55

    dense_w = np.zeros((1, hidden_size), dtype=np.float32)
    dense_w[0, 0] = 0.22
    dense_w[0, 1] = 0.14
    dense_w[0, 2] = 0.09
    dense_w[0, 3:] = rng.normal(0.0, 0.015, size=(hidden_size - 3,)).astype(np.float32)
    dense_b = np.zeros((1,), dtype=np.float32)

    payload = build_rtneural_json_dict(w_ih, w_hh, b_ih, b_hh, dense_w, dense_b, hidden_size=hidden_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[generate_dummy_data] Generated initialized RTNeural weights -> {output_path}")


def synthesize_plucked_guitar_and_target_bass(
    fundamental_guitar_hz: float,
    duration_sec: float = 1.5,
    sample_rate: int = 48000,
):
    """
    Synthesizes a dry electric guitar pluck at `fundamental_guitar_hz`,
    an octave-shifted guitar approximation at `0.5 * fundamental_guitar_hz`,
    and a thick .105-gauge DI bass target waveform at `0.5 * fundamental_guitar_hz`.
    """
    num_samples = int(duration_sec * sample_rate)
    t = np.arange(num_samples, dtype=np.float32) / float(sample_rate)

    # 1. Dry Electric Guitar (bright upper harmonics + sharp 12ms pick transient)
    env_body = np.exp(-t * 3.2) * (1.0 - np.exp(-t * 450.0))
    env_pick = np.exp(-t * 95.0)

    guitar_sig = np.zeros_like(t)
    harmonic_amps_guitar = [1.0, 0.82, 0.60, 0.42, 0.28, 0.18, 0.12, 0.08]
    for h_idx, amp in enumerate(harmonic_amps_guitar, start=1):
        freq = fundamental_guitar_hz * h_idx
        if freq < sample_rate * 0.45:
            decay = np.exp(-t * (2.5 + 0.65 * h_idx))
            guitar_sig += amp * decay * np.sin(2.0 * np.pi * freq * t)

    # Add high-frequency pick transient clank (>600 Hz burst in first 12 ms)
    pick_clank = (
        0.35 * np.sin(2.0 * np.pi * 1150.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 2380.0 * t)
    ) * env_pick
    guitar_sig = 0.55 * (guitar_sig * env_body + pick_clank)
    guitar_sig = guitar_sig / (np.max(np.abs(guitar_sig)) + 1e-6) * 0.8

    # 2. Pre-shifted Guitar fallback (-12 semitones fundamental, +4 semitones formant tilt)
    bass_f0 = 0.5 * fundamental_guitar_hz
    shifted_sig = np.zeros_like(t)
    for h_idx, amp in enumerate(harmonic_amps_guitar, start=1):
        freq = bass_f0 * h_idx
        if freq < sample_rate * 0.45:
            decay = np.exp(-t * (2.5 + 0.65 * h_idx))
            shifted_sig += amp * decay * np.sin(2.0 * np.pi * freq * t)
    shifted_sig = 0.55 * shifted_sig * env_body
    shifted_sig = shifted_sig / (np.max(np.abs(shifted_sig)) + 1e-6) * 0.8

    # 3. Target .105-gauge Studio DI Bass (dominant fundamental + warm 2nd/3rd transformer harmonics)
    di_env = np.exp(-t * 2.1) * (1.0 - np.exp(-t * 380.0))
    bass_raw = (
        1.25 * np.sin(2.0 * np.pi * bass_f0 * t)
        + 0.55 * np.sin(2.0 * np.pi * (2.0 * bass_f0) * t + 0.15)
        + 0.25 * np.sin(2.0 * np.pi * (3.0 * bass_f0) * t)
        + 0.10 * np.sin(2.0 * np.pi * (4.0 * bass_f0) * t)
    ) * di_env

    # Apply asymmetrical DI transformer saturation tanh(drive*x + 0.12*(drive*x)^2)
    drive = 1.35
    dx = drive * bass_raw
    target_bass = np.tanh(dx + 0.12 * (dx ** 2))
    target_bass = target_bass - np.mean(target_bass)
    target_bass = target_bass / (np.max(np.abs(target_bass)) + 1e-6) * 0.85

    return guitar_sig.astype(np.float32), shifted_sig.astype(np.float32), target_bass.astype(np.float32)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    model_json_path = repo_root / "models" / "default_di_bass.json"
    dataset_root = repo_root / "ml_training" / "dataset"

    # 1. Generate default_di_bass.json
    generate_default_di_bass_json(model_json_path, hidden_size=48)

    # 2. Generate synthetic guitar & target DI bass WAV dataset across standard guitar open/fretted notes
    notes = {
        "note_01_E2_82Hz.wav": 82.41,
        "note_02_G2_98Hz.wav": 98.00,
        "note_03_A2_110Hz.wav": 110.00,
        "note_04_C3_131Hz.wav": 130.81,
        "note_05_D3_147Hz.wav": 146.83,
        "note_06_E3_165Hz.wav": 164.81,
    }

    for filename, f0_hz in notes.items():
        raw_g, shifted_g, target_b = synthesize_plucked_guitar_and_target_bass(f0_hz, duration_sec=1.5)
        write_wav_float32_or_pcm16(dataset_root / "raw_guitar" / filename, raw_g)
        write_wav_float32_or_pcm16(dataset_root / "shifted_guitar" / filename, shifted_g)
        write_wav_float32_or_pcm16(dataset_root / "target_bass" / filename, target_b)

    print(f"[generate_dummy_data] Wrote {len(notes)} synthetic WAV pairs into {dataset_root}")


if __name__ == "__main__":
    main()
