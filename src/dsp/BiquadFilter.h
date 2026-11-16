#pragma once

#include <cmath>
#include <algorithm>

namespace dsp {

/**
 * Zero-allocation 1st- and 2nd-order IIR Filter (Direct Form II Transposed).
 * Real-time audio thread safe: no dynamic memory allocation, locks, or I/O.
 */
class BiquadFilter {
public:
    BiquadFilter() = default;

    void reset() noexcept {
        z1_ = 0.0f;
        z2_ = 0.0f;
    }

    /**
     * 2nd-order Butterworth High-Pass Filter (default Q = 1 / sqrt(2)).
     * Used at 75 Hz for input conditioning to remove guitar handling noise.
     */
    void setHighPass(double sampleRate, float cutoffHz, float q = 0.70710678f) noexcept {
        if (sampleRate <= 0.0) return;
        const float nyquist = static_cast<float>(sampleRate * 0.49);
        const float fc = std::clamp(cutoffHz, 5.0f, nyquist);
        const float w0 = 2.0f * kPi * (fc / static_cast<float>(sampleRate));
        const float cosW0 = std::cos(w0);
        const float sinW0 = std::sin(w0);
        const float alpha = sinW0 / (2.0f * std::max(q, 0.05f));

        const float b0 =  (1.0f + cosW0) * 0.5f;
        const float b1 = -(1.0f + cosW0);
        const float b2 =  (1.0f + cosW0) * 0.5f;
        const float a0 =   1.0f + alpha;
        const float a1 =  -2.0f * cosW0;
        const float a2 =   1.0f - alpha;

        setNormalizedCoefficients(b0, b1, b2, a0, a1, a2);
    }

    /**
     * 2nd-order Butterworth Low-Pass Filter (default Q = 1 / sqrt(2)).
     * Used at 3.8 kHz for DI output conditioning.
     */
    void setLowPass(double sampleRate, float cutoffHz, float q = 0.70710678f) noexcept {
        if (sampleRate <= 0.0) return;
        const float nyquist = static_cast<float>(sampleRate * 0.49);
        const float fc = std::clamp(cutoffHz, 20.0f, nyquist);
        const float w0 = 2.0f * kPi * (fc / static_cast<float>(sampleRate));
        const float cosW0 = std::cos(w0);
        const float sinW0 = std::sin(w0);
        const float alpha = sinW0 / (2.0f * std::max(q, 0.05f));

        const float b0 = (1.0f - cosW0) * 0.5f;
        const float b1 =  1.0f - cosW0;
        const float b2 = (1.0f - cosW0) * 0.5f;
        const float a0 =  1.0f + alpha;
        const float a1 = -2.0f * cosW0;
        const float a2 =  1.0f - alpha;

        setNormalizedCoefficients(b0, b1, b2, a0, a1, a2);
    }

    /**
     * 2nd-order Peaking EQ Filter.
     */
    void setPeakEQ(double sampleRate, float centerHz, float q, float gainDb) noexcept {
        if (sampleRate <= 0.0) return;
        const float nyquist = static_cast<float>(sampleRate * 0.49);
        const float fc = std::clamp(centerHz, 20.0f, nyquist);
        const float A = std::pow(10.0f, gainDb / 40.0f);
        const float w0 = 2.0f * kPi * (fc / static_cast<float>(sampleRate));
        const float cosW0 = std::cos(w0);
        const float sinW0 = std::sin(w0);
        const float alpha = sinW0 / (2.0f * std::max(q, 0.05f));

        const float b0 =  1.0f + alpha * A;
        const float b1 = -2.0f * cosW0;
        const float b2 =  1.0f - alpha * A;
        const float a0 =  1.0f + alpha / A;
        const float a1 = -2.0f * cosW0;
        const float a2 =  1.0f - alpha / A;

        setNormalizedCoefficients(b0, b1, b2, a0, a1, a2);
    }

    /**
     * 1st-order High-Pass Filter.
     * Used at 600 Hz to isolate dry guitar pick clank for transient reinjection.
     */
    void setFirstOrderHighPass(double sampleRate, float cutoffHz) noexcept {
        if (sampleRate <= 0.0) return;
        const float nyquist = static_cast<float>(sampleRate * 0.49);
        const float fc = std::clamp(cutoffHz, 5.0f, nyquist);
        const float k = std::tan(kPi * fc / static_cast<float>(sampleRate));
        const float norm = 1.0f / (1.0f + k);

        b0_ =  norm;
        b1_ = -norm;
        b2_ =  0.0f;
        a1_ = (k - 1.0f) * norm;
        a2_ =  0.0f;
    }

    inline float processSample(float x) noexcept {
        const float y = b0_ * x + z1_;
        z1_ = b1_ * x - a1_ * y + z2_;
        z2_ = b2_ * x - a2_ * y;

        // Prevent subnormal float stalls on x86 CPUs
        if (std::abs(z1_) < 1.0e-20f) z1_ = 0.0f;
        if (std::abs(z2_) < 1.0e-20f) z2_ = 0.0f;
        return y;
    }

    inline void processBlock(float* samples, int numSamples) noexcept {
        for (int i = 0; i < numSamples; ++i) {
            samples[i] = processSample(samples[i]);
        }
    }

private:
    static constexpr float kPi = 3.14159265358979323846f;

    void setNormalizedCoefficients(float b0, float b1, float b2,
                                   float a0, float a1, float a2) noexcept {
        const float invA0 = 1.0f / a0;
        b0_ = b0 * invA0;
        b1_ = b1 * invA0;
        b2_ = b2 * invA0;
        a1_ = a1 * invA0;
        a2_ = a2 * invA0;
    }

    float b0_ = 1.0f;
    float b1_ = 0.0f;
    float b2_ = 0.0f;
    float a1_ = 0.0f;
    float a2_ = 0.0f;

    float z1_ = 0.0f;
    float z2_ = 0.0f;
};

} // namespace dsp
