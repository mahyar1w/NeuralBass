#!/usr/bin/env python3
"""
Analyzes the Fraunhofer IDMT-SMT-BASS dataset and builds time-aligned training pairs
in `ml_training/dataset/` from real studio-recorded DI bass notes:
  - Target (`target_bass/`): Real 4-string electric bass DI recordings (Fame Baphomet NTB,
    Warwick 4-string, Yamaha BB604 Jazz Bass; Finger-style `FS` & Picked `PK`;
    Neck / Neck+Bridge / Bridge pickups), trimmed of leading silence and resampled 44.1k -> 48k.
  - Input (`raw_guitar/`): Phase- and envelope-matched electric guitar proxy 1 octave above
    (`2 * f0`), including realistic high-frequency pick transient, so that after
    `OfflineShifterCLI` applies 75 Hz HPF + Signalsmith Stretch (-12st pitch, +4st formant),
    `shifted_guitar/` is sample-aligned with `target_bass/`.

Also writes measured spectral/envelope statistics to `models/idmt_bass_profile.json`.
"""

import argparse
import json
import os
import re
import struct
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SR_OUT = 48000
OPEN_MIDI = {1: 28, 2: 33, 3: 38, 4: 43}  # E1, A1, D2, G2
FILENAME_RE = re.compile(r"BS_(\d+)_EQ_(\d+)_(?:PS_)?([A-Z]+)_(?:ES_)?([A-Z]+)_(\d+)_(\d+)\.wav")


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    with open(path, "rb") as f:
        f.read(12)
        fmt, ch, sr, bits, raw = 1, 1, 44100, 24, b""
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            cid, csz = struct.unpack("<4sI", h)
            if cid == b"fmt ":
                d = f.read(csz)
                fmt, ch, sr, _, _, bits = struct.unpack("<HHIIHH", d[:16])
                if fmt == 0xFFFE and csz >= 40:
                    fmt = struct.unpack("<H", d[24:26])[0]
            elif cid == b"data":
                raw = f.read(csz)
            else:
                f.seek(csz, 1)
            if csz & 1:
                f.seek(1, 1)

    if fmt == 1 and bits == 24:
        b = np.frombuffer(raw[: len(raw) // 3 * 3], dtype=np.uint8).reshape(-1, 3)
        v = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int8).astype(np.int32) << 16)
        )
        x = v.astype(np.float32) / 8388608.0
    elif fmt == 1 and bits == 16:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif fmt == 3 and bits == 32:
        x = np.frombuffer(raw, dtype=np.float32)
    else:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0

    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), int(sr)


def write_wav16(path: Path, samples: np.ndarray, sr: int = SR_OUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -0.999, 0.999) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def resample_audio(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    try:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except ImportError:
        n_out = int(round(len(x) * sr_out / sr_in))
        t_in = np.linspace(0.0, 1.0, len(x), endpoint=False)
        t_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.interp(t_out, t_in, x).astype(np.float32)


def estimate_f0(x: np.ndarray, sr: int, nominal: float) -> float:
    n = 1 << 16
    start = min(int(0.05 * sr), max(0, len(x) - n))
    seg = x[start : start + n]
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    spec = np.abs(np.fft.rfft(seg * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    lo, hi = nominal * 2 ** (-0.75 / 12.0), nominal * 2 ** (0.75 / 12.0)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(freqs[mask][np.argmax(spec[mask])]) if mask.any() else nominal


def one_pole_env(x_abs: np.ndarray, fc: float, sr: int) -> np.ndarray:
    a = np.exp(-2.0 * np.pi * fc / sr)
    y = np.empty_like(x_abs)
    s = 0.0
    for i, v in enumerate(x_abs):
        s = a * s + (1.0 - a) * v
        y[i] = s
    return y


def trim_leading_silence(x: np.ndarray, sr: int) -> np.ndarray:
    pk = float(np.max(np.abs(x)))
    if pk < 1e-5:
        return x
    onset = int(np.argmax(np.abs(x) > 0.03 * pk))
    lead = max(0, onset - int(0.004 * sr))
    trimmed = x[lead:].copy()
    fade = min(int(0.002 * sr), len(trimmed))
    if fade > 1:
        trimmed[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    return trimmed


def analyze_sample(x: np.ndarray, sr: int, midi: int) -> Dict:
    nominal = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    f0 = estimate_f0(x, sr, nominal)
    env = np.abs(x)
    pk_idx = int(np.argmax(env[: int(0.25 * sr)]))
    pk_val = float(env[pk_idx]) + 1e-9
    on_idx = int(np.argmax(env > 0.05 * pk_val))
    att_ms = (pk_idx - on_idx) * 1000.0 / sr

    att = x[on_idx : on_idx + int(0.015 * sr)]
    hi_att = 0.0
    if len(att) >= 64:
        sp = np.abs(np.fft.rfft(att * np.hanning(len(att)))) ** 2
        fr = np.fft.rfftfreq(len(att), 1.0 / sr)
        hi_att = float(sp[fr > 600.0].sum() / (sp.sum() + 1e-12))

    n = 16384
    s0 = pk_idx + int(0.06 * sr)
    if s0 + n > len(x):
        return {}
    seg = x[s0 : s0 + n] * np.hanning(n)
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    cs = np.cumsum(spec ** 2)
    roll95 = float(freqs[np.searchsorted(cs, 0.95 * cs[-1])])
    roll99 = float(freqs[np.searchsorted(cs, 0.99 * cs[-1])])

    harms = []
    for k in range(1, 9):
        idx = int(round(f0 * k * n / sr))
        w = max(2, int(round(f0 * 0.12 * n / sr)))
        harms.append(float(spec[max(1, idx - w) : idx + w + 1].max()))
    harms = np.array(harms)
    harm_db = (20.0 * np.log10(harms / (harms[0] + 1e-12) + 1e-12)).tolist()

    pos_pk = float(np.max(x[pk_idx : pk_idx + int(0.2 * sr)]))
    neg_pk = float(np.min(x[pk_idx : pk_idx + int(0.2 * sr)]))
    asym = pos_pk / (abs(neg_pk) + 1e-9)
    return dict(
        midi=midi,
        f0=f0,
        attack_ms=att_ms,
        hi_att=hi_att,
        roll95=roll95,
        roll99=roll99,
        asym=asym,
        harm_db=harm_db,
    )


def make_guitar_proxy(bass: np.ndarray, sr: int, f0: float, rng: np.random.Generator) -> np.ndarray:
    """
    Synthesizes an electric guitar waveform 1 octave above `f0` whose envelope and
    transient attack follow the real IDMT-SMT-BASS note.
    """
    n = len(bass)
    env = one_pole_env(np.abs(bass), 28.0, sr)
    env = env / (env.max() + 1e-9)
    t = np.arange(n, dtype=np.float32) / float(sr)

    f_guitar = 2.0 * f0
    guitar = np.zeros(n, dtype=np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=14)
    for k in range(1, 15):
        fk = f_guitar * k
        if fk > 9500.0:
            break
        # Electric guitar upper harmonics decay faster over time than fundamental
        harmonic_decay = np.exp(-t * (0.4 * (k - 1)))
        guitar += (1.0 / (k ** 0.75)) * harmonic_decay * np.sin(2.0 * np.pi * fk * t + phases[k - 1])

    guitar *= env

    # Inject dry guitar pick transient (>600 Hz burst) aligned with bass onset
    onset = int(np.argmax(np.abs(bass) > 0.05 * np.max(np.abs(bass))))
    clank_len = min(int(0.014 * sr), n - onset)
    if clank_len > 16:
        tc = np.arange(clank_len, dtype=np.float32) / float(sr)
        pick_burst = (
            0.35 * np.sin(2.0 * np.pi * 1250.0 * tc)
            + 0.25 * np.sin(2.0 * np.pi * 2450.0 * tc)
            + 0.20 * rng.standard_normal(clank_len).astype(np.float32)
        ) * np.exp(-tc / 0.0035)
        guitar[onset : onset + clank_len] += 0.45 * pick_burst

    return (guitar / (np.max(np.abs(guitar)) + 1e-9) * 0.8).astype(np.float32)


def main():
    workspace_parent = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Analyze IDMT-SMT-BASS and export aligned training dataset")
    parser.add_argument(
        "idmt_root",
        nargs="?",
        type=Path,
        default=workspace_parent / "IDMT-SMT-BASS",
        help="Path to IDMT-SMT-BASS root folder",
    )
    parser.add_argument("--seconds", type=float, default=2.0, help="Clip duration in seconds")
    parser.add_argument("--max-per-style", type=int, default=36, help="Max clips per plucking style (FS, PK)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    ds_root = repo_root / "ml_training" / "dataset"

    # Remove old dummy / short-scale files so only IDMT-SMT-BASS data remains
    for sub in ("raw_guitar", "shifted_guitar", "target_bass"):
        sub_dir = ds_root / sub
        if sub_dir.exists():
            for old in sub_dir.glob("*.wav"):
                old.unlink()

    old_profile = repo_root / "models" / "bass_pack_profile.json"
    if old_profile.exists():
        old_profile.unlink()

    rng = np.random.default_rng(seed=2026)
    summary_by_style: Dict[str, Dict] = {}
    exported = 0

    for style in ("FS", "PK", "MU", "SP", "ST"):
        style_dir = args.idmt_root / "PS" / style
        if not style_dir.exists():
            continue

        all_files = sorted(style_dir.glob("*.wav"))
        # Sample evenly across basses (1..3), pickups (1..3), strings (1..4), and frets
        step = max(1, len(all_files) // 150)
        stats: List[Dict] = []

        # Export balanced training pairs from Finger-style (FS) and Picked (PK)
        export_candidates = []
        for fn in all_files:
            m = FILENAME_RE.match(fn.name)
            if not m:
                continue
            bass_id, eq_id, ps, es, str_s, fret_s = m.groups()
            str_num, fret = int(str_s), int(fret_s)
            if str_num not in OPEN_MIDI:
                continue
            export_candidates.append((int(bass_id), int(eq_id), str_num, fret, fn))

        export_step = max(1, len(export_candidates) // args.max_per_style)
        export_set = set(x[4] for x in export_candidates[::export_step][: args.max_per_style]) if style in ("FS", "PK") else set()

        for idx, (bass_id, eq_id, str_num, fret, fn) in enumerate(export_candidates):
            is_stat = (idx % step == 0)
            is_export = fn in export_set
            if not is_stat and not is_export:
                continue

            midi = OPEN_MIDI[str_num] + fret
            raw_x, sr_in = read_wav(fn)
            trimmed = trim_leading_silence(raw_x, sr_in)
            if len(trimmed) < int(0.8 * sr_in):
                continue

            if is_stat:
                r = analyze_sample(trimmed, sr_in, midi)
                if r:
                    r["bass_id"] = bass_id
                    r["eq_id"] = eq_id
                    stats.append(r)

            if is_export:
                x48 = resample_audio(trimmed, sr_in, SR_OUT)
                target_len = int(args.seconds * SR_OUT)
                if len(x48) >= target_len:
                    seg = x48[:target_len]
                else:
                    seg = np.pad(x48, (0, target_len - len(x48)))
                seg = seg / (np.max(np.abs(seg)) + 1e-9) * 0.85

                nominal = 440.0 * (2.0 ** ((midi - 69) / 12.0))
                f0 = estimate_f0(seg, SR_OUT, nominal)
                guitar_proxy = make_guitar_proxy(seg, SR_OUT, f0, rng)

                out_name = f"idmt_{style}_B{bass_id}_EQ{eq_id}_S{str_num}_F{fret:02d}.wav"
                write_wav16(ds_root / "target_bass" / out_name, seg, SR_OUT)
                write_wav16(ds_root / "raw_guitar" / out_name, guitar_proxy, SR_OUT)
                exported += 1

        if stats:
            harms = np.array([s["harm_db"] for s in stats])
            summary_by_style[style] = {
                "samples_analyzed": len(stats),
                "attack_ms_median": float(np.median([s["attack_ms"] for s in stats])),
                "hi_att_gt600hz_median": float(np.median([s["hi_att"] for s in stats])),
                "hi_att_gt600hz_p90": float(np.percentile([s["hi_att"] for s in stats], 90)),
                "rolloff95_hz_median": float(np.median([s["roll95"] for s in stats])),
                "rolloff99_hz_median": float(np.median([s["roll99"] for s in stats])),
                "rolloff99_hz_p90": float(np.percentile([s["roll99"] for s in stats], 90)),
                "asym_median": float(np.median([s["asym"] for s in stats])),
                "harmonic_db_median": np.median(harms, axis=0).tolist(),
            }
            st = summary_by_style[style]
            print(
                f"[IDMT {style}] n={st['samples_analyzed']} | att={st['attack_ms_median']:.1f}ms | "
                f">600Hz_att={st['hi_att_gt600hz_median']*100:.1f}% | "
                f"roll99_med={st['rolloff99_hz_median']:.0f}Hz (p90={st['rolloff99_hz_p90']:.0f}Hz) | "
                f"H1..H6={[round(v, 1) for v in st['harmonic_db_median'][:6]]}"
            )

    profile_path = repo_root / "models" / "idmt_bass_profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps({"dataset": "IDMT-SMT-BASS", "styles": summary_by_style}, indent=2))
    print(f"[analyze_idmt_bass] Wrote {profile_path} and exported {exported} 48kHz training pairs to {ds_root}")


if __name__ == "__main__":
    main()
