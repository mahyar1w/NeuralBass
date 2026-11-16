#pragma once

#include "BiquadFilter.h"
#include "TransientExtractor.h"
#include "SubOctaveTracker.h"
#include "PitchSyncOctaveShifter.h"
#include "FETCompressor.h"

#include <RTNeural/RTNeural.h>

#include <atomic>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace dsp {

/**
 * Complete 5-Stage Hybrid DSP + Neural Guitar-to-DI-Bass Engine.
 *
 * Stage 1: Input Conditioning (2nd-order Butterworth HPF @ 60 Hz) + Concurrent Stage 1b
 *          (Dual-RMS Band-Passed Transient Extractor + 4th-order LPF Sub-Octave Fundamental Tracker).
 * Stage 2: Low-Latency Pitch Shifting (AVX2 Pitch-Synchronous WSOLA: -12st pitch, ~11.8 ms mean latency).
 *          Concurrent with Stage 1b via real-time lock-free worker thread on multi-core systems.
 * Stage 3: Neural Timbre Transfer (Compile-Time AVX2 RTNeural GRU48 with soft-clipped residual skip).
 * Stage 4: Transient Reinjection (Band-passed 600–1900 Hz pick clank scaled by transient mask).
 * Stage 5: DI Output Conditioning (Transparent transformer waveshaper + FET 4:1 compressor +
 *          32 Hz sub-rumble HPF + 2.1 kHz studio DI LPF).
 */
class HybridBassEngine {
public:
    // Calibrated studio DI acoustic constants
    static constexpr float kOutputLpfHz = 2100.0f;
    static constexpr float kSubRumbleHpfHz = 32.0f;
    static constexpr float kClankReinjectScale = 0.09f;

    using NeuralModel = RTNeural::ModelT<float, 1, 1,
        RTNeural::GRULayerT<float, 1, 48>,
        RTNeural::DenseT<float, 48, 1>>;

    HybridBassEngine() {
        initializeFallbackIdentityWeights(models_[0]);
        initializeFallbackIdentityWeights(models_[1]);
    }

    ~HybridBassEngine() {
        stopWorkerThread();
    }

    void prepare(double sampleRate, int maxBlockSize) {
        sampleRate_ = (sampleRate > 0.0) ? sampleRate : 48000.0;
        maxBlockSize_ = std::max(maxBlockSize, 64);

        // 1. Input Conditioning: 2nd-order Butterworth HPF @ 60 Hz (preserves full E2 82.4 Hz fundamental)
        inputHpf_.setHighPass(sampleRate_, 60.0f, 0.70710678f);
        inputHpf_.reset();

        // 2. Low-Latency Pitch Shifting: AVX2 Pitch-Synchronous WSOLA (-12st, ~11.8 ms mean latency)
        pitchShifter_.prepare(sampleRate_);
        const int totalStretchLatency = pitchShifter_.getLatencySamples();

        // Scratch buffers for Pitch Shifter, Sub-Octave, and Transient reinjection
        stretchInBuf_.assign(static_cast<size_t>(maxBlockSize_), 0.0f);
        stretchOutBuf_.assign(static_cast<size_t>(maxBlockSize_), 0.0f);
        clankBuf_.assign(static_cast<size_t>(maxBlockSize_), 0.0f);
        subOctBuf_.assign(static_cast<size_t>(maxBlockSize_), 0.0f);
        blendedBuf_.assign(static_cast<size_t>(maxBlockSize_), 0.0f);

        // Prepare Dual-RMS Transient Extractor (600–1900 Hz band-passed clank)
        transientExtractor_.prepare(sampleRate_, maxBlockSize_, totalStretchLatency + 1024);
        transientExtractor_.setLatencyCompensation(totalStretchLatency);

        // Prepare Sub-Octave Fundamental Tracker (38 Hz – 130 Hz clean phase-locked fundamental)
        subOctaveTracker_.prepare(sampleRate_, maxBlockSize_, totalStretchLatency + 1024);
        subOctaveTracker_.setLatencyCompensation(totalStretchLatency);

        // 3. Reset compile-time RTNeural models
        models_[0].reset();
        models_[1].reset();

        // 5. DI Output Conditioning: FET Compressor + 32 Hz sub-rumble HPF + 2.1 kHz LPF
        fetCompressor_.prepare(sampleRate_);
        subRumbleHpf_.setHighPass(sampleRate_, kSubRumbleHpfHz, 0.70710678f);
        subRumbleHpf_.reset();
        outputLpf_.setLowPass(sampleRate_, kOutputLpfHz, 0.70710678f);
        outputLpf_.reset();

        // Multi-threading management: enable worker thread when hardware has >= 4 logical cores
        const unsigned int hwCores = std::thread::hardware_concurrency();
        const bool shouldBeMultithreaded = (hwCores >= 4);

        if (shouldBeMultithreaded && !workerThread_.joinable()) {
            workerRunning_.store(true, std::memory_order_release);
            workerThread_ = std::thread([this]() { workerLoop(); });
            isMultithreaded_ = true;
        } else if (!shouldBeMultithreaded) {
            stopWorkerThread();
            isMultithreaded_ = false;
        }
    }

    void reset() noexcept {
        inputHpf_.reset();
        pitchShifter_.reset();
        transientExtractor_.reset();
        subOctaveTracker_.reset();
        models_[0].reset();
        models_[1].reset();
        fetCompressor_.reset();
        subRumbleHpf_.reset();
        outputLpf_.reset();
    }

    int getLatencySamples() const noexcept {
        return pitchShifter_.getLatencySamples();
    }

    bool isMultithreaded() const noexcept {
        return isMultithreaded_;
    }

    int getActiveWorkerThreads() const noexcept {
        return isMultithreaded_ ? 1 : 0;
    }

    /**
     * Loads RTNeural weights from a JSON file on the non-real-time (UI/init) thread
     * into the inactive model buffer, then atomically swaps the active model index.
     */
    bool loadModelFromJsonFile(const std::string& jsonPath) {
        std::ifstream jsonStream(jsonPath, std::ifstream::binary);
        if (!jsonStream.is_open()) {
            return false;
        }

        try {
            nlohmann::json modelJson;
            jsonStream >> modelJson;

            const int currentIdx = activeModelIdx_.load(std::memory_order_relaxed);
            const int nextIdx = 1 - currentIdx;
            auto& targetModel = models_[nextIdx];

            if (modelJson.contains("layers") && modelJson.contains("in_shape")) {
                targetModel.parseJson(modelJson);
            } else if (modelJson.contains("gru.weight_ih_l0")) {
                RTNeural::torch_helpers::loadGRU<float>(modelJson, "gru.", targetModel.template get<0>());
                RTNeural::torch_helpers::loadDense<float>(modelJson, "dense.", targetModel.template get<1>());
            } else {
                return false;
            }

            targetModel.reset();
            activeModelIdx_.store(nextIdx, std::memory_order_release);
            return true;
        } catch (...) {
            return false;
        }
    }

    /**
     * Real-time audio processing block.
     * Zero memory allocations, zero locks, zero I/O.
     */
    void processBlock(float* channelData,
                      int numSamples,
                      float subHarmonicsGain,
                      float pickAttackGain,
                      float diDrive,
                      float outputLevelLinear) noexcept {
        if (numSamples <= 0 || channelData == nullptr) {
            return;
        }

        int offset = 0;
        while (offset < numSamples) {
            const int chunk = std::min(numSamples - offset, maxBlockSize_);
            processChunk(channelData + offset,
                         chunk,
                         subHarmonicsGain,
                         pickAttackGain,
                         diDrive,
                         outputLevelLinear);
            offset += chunk;
        }
    }

private:
    void stopWorkerThread() {
        if (workerThread_.joinable()) {
            workerRunning_.store(false, std::memory_order_release);
            const uint32_t seq = jobSeq_.fetch_add(1, std::memory_order_relaxed) + 1;
            jobCtrl_.store((seq << 2) | 1u, std::memory_order_release);
            workerThread_.join();
        }
        isMultithreaded_ = false;
    }

    void workerLoop() {
        uint32_t lastClaimedSeq = 0;
        int spinCount = 0;
        while (workerRunning_.load(std::memory_order_relaxed)) {
            uint32_t ctrl = jobCtrl_.load(std::memory_order_acquire);
            const uint32_t seq = ctrl >> 2;
            const uint32_t st  = ctrl & 3u;
            if (st == 1u && seq != lastClaimedSeq) {
                uint32_t expected = (seq << 2) | 1u;
                // Atomically claim job before executing so audio timeout fallback cannot race
                if (jobCtrl_.compare_exchange_strong(expected, (seq << 2) | 2u, std::memory_order_acq_rel)) {
                    lastClaimedSeq = seq;
                    if (!workerRunning_.load(std::memory_order_acquire)) {
                        jobCtrl_.store((seq << 2) | 0u, std::memory_order_release);
                        break;
                    }
                    const int numSamples = workerNumSamples_.load(std::memory_order_relaxed);
                    processWorkerStage1b(numSamples);
                    jobCtrl_.store((seq << 2) | 0u, std::memory_order_release);
                    spinCount = 0;
                    continue;
                }
            }
            if (++spinCount < 1500) {
                #if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
                #else
                std::this_thread::yield();
                #endif
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(40));
            }
        }
    }

    void processWorkerStage1b(int numSamples) noexcept {
        for (int i = 0; i < numSamples; ++i) {
            const float in = stretchInBuf_[static_cast<size_t>(i)];
            clankBuf_[static_cast<size_t>(i)]  = transientExtractor_.processSample(in);
            subOctBuf_[static_cast<size_t>(i)] = subOctaveTracker_.processSample(in);
        }
    }

    void processChunk(float* data,
                      int numSamples,
                      float subHarmonicsGain,
                      float pickAttackGain,
                      float diDrive,
                      float outputLevelLinear) noexcept {
        // Stage 1a: Input Conditioning (60 Hz 2nd-order Butterworth HPF)
        for (int i = 0; i < numSamples; ++i) {
            stretchInBuf_[static_cast<size_t>(i)] = inputHpf_.processSample(data[i]);
        }

        // Concurrent Stage 1b & Stage 2: Parallel Worker vs Audio Thread (race-free atomic claim)
        if (isMultithreaded_ && workerThread_.joinable()) {
            workerNumSamples_.store(numSamples, std::memory_order_relaxed);
            const uint32_t seq = jobSeq_.fetch_add(1, std::memory_order_relaxed) + 1;
            const uint32_t doneTag = (seq << 2) | 0u;
            jobCtrl_.store((seq << 2) | 1u, std::memory_order_release);

            // Thread 1 (Audio Thread) processes AVX2 Pitch-Synchronous Shifter
            pitchShifter_.processBlock(stretchInBuf_.data(), stretchOutBuf_.data(), numSamples);

            int spins = 0;
            uint32_t ctrl = 0;
            while ((ctrl = jobCtrl_.load(std::memory_order_acquire)) != doneTag) {
                if ((ctrl & 3u) == 1u && ++spins > 80000) {
                    uint32_t expected = (seq << 2) | 1u;
                    if (jobCtrl_.compare_exchange_strong(expected, doneTag, std::memory_order_acq_rel)) {
                        processWorkerStage1b(numSamples);
                        break;
                    }
                }
                #if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
                #else
                std::this_thread::yield();
                #endif
            }
        } else {
            // Single-threaded fallback
            processWorkerStage1b(numSamples);
            pitchShifter_.processBlock(stretchInBuf_.data(), stretchOutBuf_.data(), numSamples);
        }

        // Blend: 88% Pitch-Shifted string body + 12% Clean Analog Phase-Locked Sub-Octave Fundamental
        int i = 0;
#if defined(__AVX2__)
        const __m256 v88 = _mm256_set1_ps(0.88f);
        const __m256 v12 = _mm256_set1_ps(0.12f);
        for (; i + 8 <= numSamples; i += 8) {
            __m256 s = _mm256_loadu_ps(&stretchOutBuf_[static_cast<size_t>(i)]);
            __m256 o = _mm256_loadu_ps(&subOctBuf_[static_cast<size_t>(i)]);
            __m256 blended = _mm256_fmadd_ps(v12, o, _mm256_mul_ps(v88, s));
            _mm256_storeu_ps(&blendedBuf_[static_cast<size_t>(i)], blended);
        }
#endif
        for (; i < numSamples; ++i) {
            blendedBuf_[static_cast<size_t>(i)] =
                0.88f * stretchOutBuf_[static_cast<size_t>(i)] + 0.12f * subOctBuf_[static_cast<size_t>(i)];
        }

        // Lock-free atomic fetch of active compile-time RTNeural ModelT
        const int modelIdx = activeModelIdx_.load(std::memory_order_acquire);
        auto& activeModel = models_[modelIdx];

        const float clankScale = pickAttackGain * kClankReinjectScale;

        for (int j = 0; j < numSamples; ++j) {
            const float xBase = blendedBuf_[static_cast<size_t>(j)];

            // Stage 3: Neural Timbre Transfer (soft-clipped residual for clean analog warmth)
            float nnInput[1] = { xBase };
            const float rawNn = activeModel.forward(nnInput);
            const float nnResidual = 0.35f * std::tanh(rawNn);
            const float yNeural = xBase + subHarmonicsGain * nnResidual;

            // Stage 4: Transient Reinjection (band-passed 600-1900 Hz clank)
            const float yWithTransients = yNeural + clankScale * clankBuf_[static_cast<size_t>(j)];

            // Stage 5: DI Output Conditioning (Transparent waveshaper + FET 4:1 + 32 Hz HPF + 2.1 kHz LPF)
            const float yCompressed = fetCompressor_.processSample(yWithTransients, diDrive);
            const float ySubTight   = subRumbleHpf_.processSample(yCompressed);
            const float yFiltered   = outputLpf_.processSample(ySubTight);

            data[j] = yFiltered * outputLevelLinear;
        }
    }

    static void initializeFallbackIdentityWeights(NeuralModel& model) {
        auto& gru = model.template get<0>();
        auto& dense = model.template get<1>();

        std::vector<std::vector<float>> wVals(1, std::vector<float>(3 * 48, 0.0f));
        std::vector<std::vector<float>> uVals(48, std::vector<float>(3 * 48, 0.0f));
        std::vector<std::vector<float>> bVals(2, std::vector<float>(3 * 48, 0.0f));

        wVals[0][2 * 48 + 0] = 0.35f;
        wVals[0][2 * 48 + 1] = -0.20f;
        uVals[0][2 * 48 + 0] = 0.85f;
        uVals[1][2 * 48 + 1] = 0.70f;

        gru.setWVals(wVals);
        gru.setUVals(uVals);
        gru.setBVals(bVals);

        std::vector<std::vector<float>> denseWeights(1, std::vector<float>(48, 0.0f));
        denseWeights[0][0] = 0.25f;
        denseWeights[0][1] = 0.15f;
        std::vector<float> denseBias(1, 0.0f);

        dense.setWeights(denseWeights);
        dense.setBias(denseBias.data());
        model.reset();
    }

    double sampleRate_ = 48000.0;
    int maxBlockSize_ = 2048;

    BiquadFilter inputHpf_;
    TransientExtractor transientExtractor_;
    SubOctaveTracker subOctaveTracker_;
    PitchSyncOctaveShifter pitchShifter_;

    std::vector<float> stretchInBuf_;
    std::vector<float> stretchOutBuf_;
    std::vector<float> clankBuf_;
    std::vector<float> subOctBuf_;
    std::vector<float> blendedBuf_;

    NeuralModel models_[2];
    std::atomic<int> activeModelIdx_{0};

    FETCompressor fetCompressor_;
    BiquadFilter subRumbleHpf_;
    BiquadFilter outputLpf_;

    // Multi-threading state
    bool isMultithreaded_ = false;
    std::thread workerThread_;
    std::atomic<bool> workerRunning_{false};
    std::atomic<uint32_t> jobSeq_{0};
    std::atomic<uint32_t> jobCtrl_{0};
    std::atomic<int> workerNumSamples_{0};
};

} // namespace dsp
