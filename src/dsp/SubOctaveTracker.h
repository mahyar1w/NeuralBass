#pragma once

#include "BiquadFilter.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace dsp {

/**
 * Clean Analog-Style Sub-Octave Fundamental Tracker.
 * Generates a 100% phase-stable, non-warbly octave-down sub-bass fundamental
 * synchronized to the dry guitar input, latency-aligned to match Signalsmith Stretch.
 *
 * Signal Flow:
 * 1. 2nd-order Butterworth LPF @ 260 Hz isolates the dry guitar fundamental.
 * 2. Envelope follower (~2 ms attack, ~35 ms release) tracks playing dynamics.
 * 3. Hysteresis Schmitt trigger & toggle flip-flop divides frequency by 2 (-1 octave).
 * 4. Cascaded 2nd-order band-pass (38 Hz HPF + 145 Hz LPF) filters square harmonics
 *    into a deep, smooth, warm sub-sine wave.
 * 5. Pre-allocated circular delay line time-aligns sub-octave with the pitch-shifted signal.
 */
class SubOctaveTracker {
public:
    SubOctaveTracker() = default;

    void prepare(double sampleRate, int maxBlockSize, int maxDelaySamples = 8192) {
        sampleRate_ = (sampleRate > 0.0) ? sampleRate : 48000.0;

        // 4th-order Butterworth LPF at 125 Hz isolates E2/A2/D3 fundamental while strongly attenuating
        // loud 2nd harmonics (e.g., 165 Hz > 82.4 Hz on neck/bridge pickups) that would double the flip-flop.
        inLpf1_.setLowPass(sampleRate_, 125.0f, 0.5411961f);
        inLpf2_.setLowPass(sampleRate_, 125.0f, 1.3065630f);

        // 4th-order sub-bass band-pass: 38 Hz HPF strips DC/rumble, 110 Hz LPF turns square into clean sub-sine
        outHpf_.setHighPass(sampleRate_, 38.0f, 0.70710678f);
        outLpf1_.setLowPass(sampleRate_, 110.0f, 0.5411961f);
        outLpf2_.setLowPass(sampleRate_, 110.0f, 1.3065630f);

        // Envelope follower coefficients (~2 ms attack, ~35 ms release)
        envAttackCoeff_  = computeOnePoleCoeff(2.0f, sampleRate_);
        envReleaseCoeff_ = computeOnePoleCoeff(35.0f, sampleRate_);

        // Circular delay line for sample-accurate time alignment with Signalsmith Stretch
        const int pow2Size = nextPowerOfTwo(std::max(maxDelaySamples + maxBlockSize + 64, 16384));
        delayLine_.assign(static_cast<size_t>(pow2Size), 0.0f);
        delayMask_ = pow2Size - 1;
        writePos_ = 0;

        reset();
    }

    void reset() noexcept {
        inLpf1_.reset();
        inLpf2_.reset();
        outHpf_.reset();
        outLpf1_.reset();
        outLpf2_.reset();
        flipFlop_ = 1.0f;
        prevSign_ = 0;
        envFollower_ = 0.0f;
        std::fill(delayLine_.begin(), delayLine_.end(), 0.0f);
        writePos_ = 0;
    }

    void setLatencyCompensation(int delaySamples) noexcept {
        delaySamples_ = std::clamp(delaySamples, 0, delayMask_);
    }

    inline float processSample(float inSample) noexcept {
        // Step 1: Pre-filter fundamental with steep 4th-order LPF
        const float xFilt = inLpf2_.processSample(inLpf1_.processSample(inSample));

        // Step 2: Track envelope
        const float absX = std::abs(xFilt);
        const float coeff = (absX > envFollower_) ? envAttackCoeff_ : envReleaseCoeff_;
        envFollower_ = coeff * envFollower_ + (1.0f - coeff) * absX;

        // Step 3: Schmitt Trigger / Zero-Crossing with noise-immune hysteresis
        const float threshold = std::max(0.012f, envFollower_ * 0.18f);
        int currentSign = prevSign_;
        if (xFilt > threshold) {
            currentSign = 1;
        } else if (xFilt < -threshold) {
            currentSign = -1;
        }

        // Toggle on positive-going zero crossing (divide frequency by 2 = exactly -1 octave)
        if (currentSign == 1 && prevSign_ != 1) {
            flipFlop_ = -flipFlop_;
        }
        prevSign_ = currentSign;

        // Smooth noise gate: shuts off cleanly on guitar hum/room floor below ~ -42 dBFS
        const float gate = std::clamp((envFollower_ - 0.008f) / 0.016f, 0.0f, 1.0f);
        const float rawSub = flipFlop_ * envFollower_ * gate;

        // Step 4: 4th-order low-pass eliminates square-wave odd-harmonic buzz
        const float hpSub = outHpf_.processSample(rawSub);
        const float filteredSub = outLpf2_.processSample(outLpf1_.processSample(hpSub));

        // Step 5: Latency compensation buffer to align with pitch-shifter output
        if (!delayLine_.empty()) {
            delayLine_[static_cast<size_t>(writePos_)] = filteredSub;
            const int readPos = (writePos_ - delaySamples_) & delayMask_;
            writePos_ = (writePos_ + 1) & delayMask_;
            return delayLine_[static_cast<size_t>(readPos)];
        }
        return filteredSub;
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
    float envAttackCoeff_ = 0.0f;
    float envReleaseCoeff_ = 0.0f;
    float envFollower_ = 0.0f;

    BiquadFilter inLpf1_;
    BiquadFilter inLpf2_;
    BiquadFilter outHpf_;
    BiquadFilter outLpf1_;
    BiquadFilter outLpf2_;

    float flipFlop_ = 1.0f;
    int prevSign_ = 0;

    std::vector<float> delayLine_;
    int delayMask_ = 0;
    int writePos_ = 0;
    int delaySamples_ = 0;
};

} // namespace dsp
