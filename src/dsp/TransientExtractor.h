#pragma once

#include "BiquadFilter.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace dsp {

/**
 * Dual-RMS (Fast/Slow) Envelope Detector + Band-Pass (1st-order HPF @ 600 Hz + 2nd-order LPF @ 2600 Hz) Pick Clank Extractor.
 * Isolates the first 10-15 ms of raw guitar pick attack and band-limits it to 600 - 2600 Hz to match real electric bass
 * pick attack while eliminating harsh >3 kHz guitar pick scratch. Aligns it with the pitch-shifter latency using a
 * pre-allocated circular delay buffer.
 */
class TransientExtractor {
public:
    TransientExtractor() = default;

    void prepare(double sampleRate, int maxBlockSize, int maxDelaySamples = 8192) {
        sampleRate_ = (sampleRate > 0.0) ? sampleRate : 48000.0;

        // Fast RMS window (~1.2 ms attack, ~12 ms release) to capture immediate pick onset
        fastAttackCoeff_  = computeOnePoleCoeff(1.2f, sampleRate_);
        fastReleaseCoeff_ = computeOnePoleCoeff(12.0f, sampleRate_);

        // Slow RMS window (~15 ms attack, ~75 ms release) so differential isolates first 10-15 ms
        slowAttackCoeff_  = computeOnePoleCoeff(15.0f, sampleRate_);
        slowReleaseCoeff_ = computeOnePoleCoeff(75.0f, sampleRate_);

        // Mask smoothing (~2 ms) to avoid zipper clicks on transient gating
        maskSmoothCoeff_  = computeOnePoleCoeff(2.0f, sampleRate_);

        // Band-pass filter (1st-order HPF @ 600 Hz + 2nd-order Butterworth LPF @ 1900 Hz) for bass pick clank
        clankHpf_.setFirstOrderHighPass(sampleRate_, 600.0f);
        clankLpf_.setLowPass(sampleRate_, 1900.0f);

        // Pre-allocate circular delay line to time-align pick clank with Signalsmith Stretch latency
        const int pow2Size = nextPowerOfTwo(std::max(maxDelaySamples + maxBlockSize + 64, 16384));
        delayLine_.assign(static_cast<size_t>(pow2Size), 0.0f);
        delayMask_ = pow2Size - 1;
        writePos_ = 0;

        reset();
    }

    void reset() noexcept {
        fastMeanSq_ = 0.0f;
        slowMeanSq_ = 0.0f;
        smoothedMask_ = 0.0f;
        clankHpf_.reset();
        clankLpf_.reset();
        std::fill(delayLine_.begin(), delayLine_.end(), 0.0f);
        writePos_ = 0;
    }

    void setLatencyCompensation(int delaySamples) noexcept {
        delaySamples_ = std::clamp(delaySamples, 0, delayMask_);
    }

    /**
     * Processes one sample of conditioned dry guitar input, returning the
     * latency-aligned, transient-masked 600-2600 Hz band-passed pick clank.
     */
    inline float processSample(float drySample) noexcept {
        const float xSq = drySample * drySample;

        // Update Fast RMS squared envelope
        const float fastCoeff = (xSq > fastMeanSq_) ? fastAttackCoeff_ : fastReleaseCoeff_;
        fastMeanSq_ = fastCoeff * fastMeanSq_ + (1.0f - fastCoeff) * xSq;

        // Update Slow RMS squared envelope
        const float slowCoeff = (xSq > slowMeanSq_) ? slowAttackCoeff_ : slowReleaseCoeff_;
        slowMeanSq_ = slowCoeff * slowMeanSq_ + (1.0f - slowCoeff) * xSq;

        const float fastRms = std::sqrt(fastMeanSq_ + 1.0e-12f);
        const float slowRms = std::sqrt(slowMeanSq_ + 1.0e-12f);

        // Normalized differential isolates pick attacks while muting idle guitar hiss (< ~ -44 dBFS)
        const float diff = std::max(0.0f, fastRms - slowRms);
        const float noiseGate = std::clamp((fastRms - 0.006f) / 0.012f, 0.0f, 1.0f);
        const float rawMask = noiseGate * std::clamp((diff / (fastRms + 1.0e-5f)) * 2.2f, 0.0f, 1.0f);

        smoothedMask_ = maskSmoothCoeff_ * smoothedMask_ + (1.0f - maskSmoothCoeff_) * rawMask;

        // Band-pass dry guitar to 600 Hz - 2600 Hz and scale by transient mask
        const float bandLimitedClank = clankLpf_.processSample(clankHpf_.processSample(drySample));
        const float pickClank = bandLimitedClank * smoothedMask_;

        // Push into pre-allocated circular delay line for latency compensation
        if (!delayLine_.empty()) {
            delayLine_[static_cast<size_t>(writePos_)] = pickClank;
            const int readPos = (writePos_ - delaySamples_) & delayMask_;
            writePos_ = (writePos_ + 1) & delayMask_;
            return delayLine_[static_cast<size_t>(readPos)];
        }
        return pickClank;
    }

private:
    static float computeOnePoleCoeff(float timeMs, double sampleRate) noexcept {
        if (timeMs <= 0.0f || sampleRate <= 0.0) return 0.0f;
        const float tauSamples = (timeMs * 0.001f) * static_cast<float>(sampleRate);
        return std::exp(-1.0f / std::max(tauSamples, 1.0f));
    }

    static int nextPowerOfTwo(int v) noexcept {
        int p = 1;
        while (p < v) p <<= 1;
        return p;
    }

    double sampleRate_ = 48000.0;
    float fastAttackCoeff_ = 0.0f;
    float fastReleaseCoeff_ = 0.0f;
    float slowAttackCoeff_ = 0.0f;
    float slowReleaseCoeff_ = 0.0f;
    float maskSmoothCoeff_ = 0.0f;

    float fastMeanSq_ = 0.0f;
    float slowMeanSq_ = 0.0f;
    float smoothedMask_ = 0.0f;

    BiquadFilter clankHpf_;
    BiquadFilter clankLpf_;

    std::vector<float> delayLine_;
    int delayMask_ = 0;
    int writePos_ = 0;
    int delaySamples_ = 0;
};

} // namespace dsp
