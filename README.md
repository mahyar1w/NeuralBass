# GuitarToDIBass — Low-Latency Hybrid DSP + AVX2 Neural Guitar-to-DI-Bass Plugin

`GuitarToDIBass` is a low-latency (`~11.8 ms` @ 48 kHz) VST3/Standalone audio plugin and PyTorch neural training pipeline that transforms an electric guitar DI signal into a `.105`-gauge Direct Injection (DI) bass guitar.

---

## Key Features

- **Low-Latency Pitch-Synchronous Octave Down (`PitchSyncOctaveShifter`)**: Shifts `-12` semitones at `~11.8 ms` mean latency (`567` samples @ 48 kHz) using time-domain WSOLA with Hermite interpolation, coherent/equal-power adaptive crossfading, and `-54 dBFS` idle lag recentering.
- **Analog-Style Clean Sub-Octave (`SubOctaveTracker`)**: Schmitt-trigger zero-crossing octave divider with hysteresis (`0.012`), `-42 dBFS` smooth noise gate, and `38 Hz` HPF + `110 Hz` 4th-order LPF blended at `12%` (`88%` shifted + `12%` sub).
- **Transient Pick-Clank Reinjection (`TransientExtractor`)**: Dual-RMS envelope detector isolating `600–1900 Hz` pick attack scaled by `0.09` with a `-44 dBFS` idle noise gate.
- **Lock-Free Worker Processing (`HybridBassEngine`)**: Offloads concurrent sub-octave and transient extraction to a real-time worker thread via an atomic claim state (`jobCtrl_`) when multi-core concurrency is available.
- **AVX2/FMA Neural DI Timbre Transfer (`RTNeural`)**: Stateful `GRU(48)` + `Dense(48, 1)` residual skip network with zero-allocation double-buffered JSON switching across 12 built-in bass profiles.
- **DI Output Conditioning (`FETCompressor` + `BiquadFilter`)**: Gentle transformer saturation (`15 Hz` DC blocker, default unity small-signal gain), fast 4:1 FET compressor (`5 ms` attack, `40 ms` release, makeup faded below `-42 dBFS`), `32 Hz` sub-rumble HPF, and `2.1 kHz` studio DI LPF.

---

## 12 Built-In Bass Sound Models

Select any preset inside the plugin UI or load a custom RTNeural `.json` model:

1. **`1. All-Around Studio DI Bass (Master)`** (`models/default_di_bass.json`)
2. **`2. Short Scale Vintage Bass (Bridge DI)`** (`models/short_scale_bass.json`)
3. **`3. Fame Baphomet 4 NTB — Fingerstyle`** (`models/fame_baphomet_fs.json`)
4. **`4. Fame Baphomet 4 NTB — Picked`** (`models/fame_baphomet_pk.json`)
5. **`5. Warwick Corvette 4 — Fingerstyle`** (`models/warwick_fs.json`)
6. **`6. Warwick Corvette 4 — Picked`** (`models/warwick_pk.json`)
7. **`7. Yamaha BB604 Jazz — Fingerstyle`** (`models/yamaha_bb604_fs.json`)
8. **`8. Yamaha BB604 Jazz — Picked`** (`models/yamaha_bb604_pk.json`)
9. **`9. Palm-Muted DI Bass (MU)`** (`models/palm_muted_bass.json`)
10. **`10. Slap & Pop DI Bass (SP/ST)`** (`models/slap_bass.json`)
11. **`11. IDMT Generic Fingerstyle Bass`** (`models/idmt_fingerstyle_bass.json`)
12. **`12. IDMT Generic Picked Bass`** (`models/idmt_picked_bass.json`)

---

## Quick Start

### 1. Build the C++ CLI & JUCE Plugin (VST3 + Standalone)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

Built artifacts:
- Offline CLI pre-shifter & full-engine renderer: `build/OfflineShifterCLI`
- Standalone app: `build/GuitarToDIBass_artefacts/Release/Standalone/GuitarToDIBass`
- VST3 bundle: `build/GuitarToDIBass_artefacts/Release/VST3/GuitarToDIBass.vst3`

### 2. Install VST3 on Linux

```bash
mkdir -p ~/.vst3
cp -a build/GuitarToDIBass_artefacts/Release/VST3/GuitarToDIBass.vst3 ~/.vst3/
```

### 3. Offline CLI Usage

```bash
# Shift paired training guitar WAVs (-12st, Stage 1)
./build/OfflineShifterCLI ml_training/dataset/raw_guitar ml_training/dataset/shifted_guitar

# Full 5-stage HybridBassEngine offline render
./build/OfflineShifterCLI --full-engine input_guitar.wav output_bass.wav models/default_di_bass.json
```

### 4. Retrain Neural Presets (PyTorch + CUDA)

```bash
python3 -m venv ml_training/.venv
ml_training/.venv/bin/pip install -r ml_training/requirements.txt

# Build real paired training data (pass dataset paths via CLI flags if stored externally)
ml_training/.venv/bin/python ml_training/build_real_paired_dataset.py

# Train & audit all 12 built-in RTNeural presets
ml_training/.venv/bin/python -u ml_training/train_gru_bass.py --train-all-presets

# Verify end-to-end rendered audio & silence floors
ml_training/.venv/bin/python -u ml_training/verify_end_to_end.py
```
