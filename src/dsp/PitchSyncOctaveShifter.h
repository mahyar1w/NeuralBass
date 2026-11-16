#pragma once

#include "BiquadFilter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace dsp {

/**
 * Low-Latency Pitch-Synchronous Time-Domain Octave-Down Shifter (-12 Semitones, 0.5x Speed).
 *
 * Designed specifically for guitar-to-bass transposition:
 * - Zero STFT bin smearing or metallic phase-vocoder artifacts on bottom strings (E2 -> 41.2 Hz).
 * - Mirrored power-of-2 ring buffer (`2 * kBufSize`) ensures contiguous memory for AVX2 FMA
 *   normalized cross-correlation + low-pass fundamental correlation weighting.
 * - Subharmonic guard prevents octave-up lock when pickup harmonics (2*f0) exceed f0.
 * - Mean group latency ~11.8 ms @ 48 kHz (even faster attack response than 18 ms STFT).
 */
class PitchSyncOctaveShifter {
public:
    static constexpr int kBufSize = 4096;
    static constexpr int kMask = kBufSize - 1;

    PitchSyncOctaveShifter() = default;

    void prepare(double sampleRate) {
        sr_ = (sampleRate > 0.0) ? sampleRate : 48000.0;
        ring_.assign(static_cast<size_t>(kBufSize * 2), 0.0f);
        lpRing_.assign(static_cast<size_t>(kBufSize * 2), 0.0f);

        // Dual-stage low-pass weighting isolates guitar fundamentals (~62-220 Hz) for pitch tracking
        lpFilt1_.setLowPass(sr_, 165.0f, 0.5411961f);
        lpFilt2_.setLowPass(sr_, 165.0f, 1.3065630f);

        minPeriod_ = std::max(48, static_cast<int>(std::round(sr_ / 660.0)));   // ~73 samp @ 48k (E5/G4 range)
        maxPeriod_ = std::min(1400, static_cast<int>(std::round(sr_ / 62.0)));  // ~774 samp @ 48k (Drop-B1..E2)
        fadeLen_   = std::max(128, static_cast<int>(std::round(sr_ * 0.0065))); // ~6.5 ms crossfade
        corrWin_   = std::max(192, static_cast<int>(std::round(sr_ * 0.0105))); // ~10.5 ms correlation window

        minSafeLag_  = static_cast<double>(fadeLen_) * 0.5 + 8.0;
        triggerLag_  = static_cast<double>(maxPeriod_) + minSafeLag_ + 8.0;
        // Center compensation at steady-state E2..D3 operating midpoint rather than extreme low bound
        const int targetCenter = static_cast<int>(std::round(triggerLag_ - std::round(sr_ * 0.0047)));
        meanLatency_ = std::clamp(targetCenter,
                                  static_cast<int>(std::ceil(minSafeLag_ + fadeLen_ * 0.5)),
                                  static_cast<int>(std::floor(triggerLag_ - minPeriod_)));

        scores_.assign(static_cast<size_t>(maxPeriod_ + 8), -2.0f);
        reset();
    }

    void reset() noexcept {
        std::fill(ring_.begin(), ring_.end(), 0.0f);
        std::fill(lpRing_.begin(), lpRing_.end(), 0.0f);
        lpFilt1_.reset();
        lpFilt2_.reset();
        writePos_ = 0;
        lagA_ = static_cast<double>(meanLatency_);
        lagB_ = lagA_;
        xfadePos_ = -1;
        lastCorr_ = 1.0f;
        lastPeriodSamples_ = static_cast<float>(sr_ / 110.0);
        lastPeriodValid_ = false;
        silentReadStreak_ = 0;
        silentNomStreak_ = 0;
        readMs_ = 0.0f;
        nomMs_ = 0.0f;
    }

    int getLatencySamples() const noexcept {
        return meanLatency_;
    }

    void processBlock(const float* input, float* output, int numSamples) noexcept {
        for (int i = 0; i < numSamples; ++i) {
            output[i] = processSample(input[i]);
        }
    }

    inline float processSample(float x) noexcept {
        const size_t wi = static_cast<size_t>(writePos_ & kMask);
        ring_[wi] = x;
        ring_[wi + static_cast<size_t>(kBufSize)] = x;

        const float lx = lpFilt2_.processSample(lpFilt1_.processSample(x));
        lpRing_[wi] = lx;
        lpRing_[wi + static_cast<size_t>(kBufSize)] = lx;

        const float nomReadVal = ring_[static_cast<size_t>((writePos_ - meanLatency_) & kMask)];
        nomMs_ += 0.001f * (nomReadVal * nomReadVal - nomMs_);
        silentNomStreak_ = (nomMs_ < 4.0e-6f) ? (silentNomStreak_ + 1) : 0; // RMS < -54 dBFS

        // During sustained idle silence, gently keep lag centered around meanLatency_ so note attacks align accurately
        if (xfadePos_ < 0 && silentReadStreak_ >= 256 && silentNomStreak_ >= 256 &&
            lagA_ >= static_cast<double>(meanLatency_) + 0.25 * static_cast<double>(fadeLen_)) {
            const double targetLag = std::max(minSafeLag_,
                                              static_cast<double>(meanLatency_) - 0.25 * static_cast<double>(fadeLen_));
            if (lagA_ - targetLag >= static_cast<double>(minPeriod_)) {
                lagB_ = targetLag;
                xfadePos_ = 0;
                lastCorr_ = 1.0f;
            }
        }

        if (xfadePos_ < 0 && lagA_ >= triggerLag_) {
            const int maxJump = std::min(maxPeriod_, static_cast<int>(std::floor(lagA_ - minSafeLag_)));
            if (maxJump >= minPeriod_) {
                const int bestP = findBestPeriod(maxJump);
                lagB_ = lagA_ - static_cast<double>(bestP);
                xfadePos_ = 0;
            }
        }

        float y = 0.0f;
        if (xfadePos_ >= 0) {
            const float sA = readHermite(static_cast<double>(writePos_) - lagA_);
            const float sB = readHermite(static_cast<double>(writePos_) - lagB_);
            const float t = static_cast<float>(xfadePos_) / static_cast<float>(fadeLen_);
            const float phase = 3.14159265358979323846f * t;

            // Blend seamlessly between coherent linear Hann and uncorrelated equal-power fade
            const float wBHann = 0.5f - 0.5f * std::cos(phase);
            const float wAHann = 1.0f - wBHann;
            const float wBEp   = std::sin(0.5f * phase);
            const float wAEp   = std::cos(0.5f * phase);

            const float alpha = std::clamp((lastCorr_ - 0.35f) / 0.45f, 0.0f, 1.0f);
            const float wA = alpha * wAHann + (1.0f - alpha) * wAEp;
            const float wB = alpha * wBHann + (1.0f - alpha) * wBEp;
            y = wA * sA + wB * sB;

            ++xfadePos_;
            if (xfadePos_ >= fadeLen_) {
                lagA_ = lagB_;
                xfadePos_ = -1;
            }
        } else {
            y = readHermite(static_cast<double>(writePos_) - lagA_);
        }

        readMs_ += 0.001f * (y * y - readMs_);
        silentReadStreak_ = (readMs_ < 4.0e-6f) ? (silentReadStreak_ + 1) : 0;

        writePos_ = (writePos_ + 1) & kMask;
        lagA_ += 0.5;
        if (xfadePos_ >= 0) {
            lagB_ += 0.5;
        }
        return y;
    }

private:
    inline float readHermite(double pos) const noexcept {
        const double pFloor = std::floor(pos);
        const float f = static_cast<float>(pos - pFloor);
        const int i0 = static_cast<int>(pFloor);

        const float xm1 = ring_[static_cast<size_t>((i0 - 1) & kMask)];
        const float x0  = ring_[static_cast<size_t>(i0 & kMask)];
        const float x1  = ring_[static_cast<size_t>((i0 + 1) & kMask)];
        const float x2  = ring_[static_cast<size_t>((i0 + 2) & kMask)];

        const float c0 = x0;
        const float c1 = 0.5f * (x1 - xm1);
        const float c2 = xm1 - 2.5f * x0 + 2.0f * x1 - 0.5f * x2;
        const float c3 = 0.5f * (x2 - xm1) + 1.5f * (x0 - x1);
        return ((c3 * f + c2) * f + c1) * f + c0;
    }

#if defined(__AVX2__)
    static inline float hsum256(__m256 v) noexcept {
        const __m128 hi = _mm256_extractf128_ps(v, 1);
        const __m128 lo = _mm256_castps256_ps128(v);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }
#endif

    static inline void computeDotAndNorm(const float* a,
                                         const float* b,
                                         int n,
                                         float& outDot,
                                         float& outNormB) noexcept {
        float dot = 0.0f;
        float normB = 0.0f;
        int k = 0;

#if defined(__AVX2__)
        __m256 vDot = _mm256_setzero_ps();
        __m256 vNorm = _mm256_setzero_ps();
        for (; k + 8 <= n; k += 8) {
            const __m256 va = _mm256_loadu_ps(a + k);
            const __m256 vb = _mm256_loadu_ps(b + k);
            vDot = _mm256_fmadd_ps(va, vb, vDot);
            vNorm = _mm256_fmadd_ps(vb, vb, vNorm);
        }
        dot = hsum256(vDot);
        normB = hsum256(vNorm);
#endif
        for (; k < n; ++k) {
            dot += a[k] * b[k];
            normB += b[k] * b[k];
        }
        outDot = dot;
        outNormB = normB;
    }

    static inline float computeSelfNorm(const float* a, int n) noexcept {
        float normA = 0.0f;
        int k = 0;
#if defined(__AVX2__)
        __m256 vNorm = _mm256_setzero_ps();
        for (; k + 8 <= n; k += 8) {
            const __m256 va = _mm256_loadu_ps(a + k);
            vNorm = _mm256_fmadd_ps(va, va, vNorm);
        }
        normA = hsum256(vNorm);
#endif
        for (; k < n; ++k) {
            normA += a[k] * a[k];
        }
        return normA;
    }

    int findBestPeriod(int maxJump) noexcept {
        const int rA = writePos_ - static_cast<int>(std::round(lagA_));
        const int win = corrWin_;
        const size_t baseA = static_cast<size_t>((rA - win) & kMask);

        const float* rawA = &ring_[baseA];
        const float* lpA  = &lpRing_[baseA];

        const float normA = computeSelfNorm(rawA, win);
        const float quietNorm = static_cast<float>(win) * 4.0e-6f; // RMS < -54 dBFS
        if (normA < quietNorm) {
            // Input at read head is silence/noise floor: re-center smoothly at nominal group latency
            lastPeriodValid_ = false;
            lastCorr_ = 1.0f;
            return std::clamp(
                static_cast<int>(std::round(lagA_ - static_cast<double>(meanLatency_))),
                minPeriod_, maxJump);
        }

        const float lpNormA = computeSelfNorm(lpA, win);
        const float lpRatio = lpNormA / (normA + 1.0e-9f);
        // Stronger fundamental low-pass weighting on low notes where pickup harmonics can exceed f0
        const float lpWeight = (lpRatio > 0.06f) ? 0.45f : 0.15f;
        const float rawWeight = 1.0f - lpWeight;

        std::fill(scores_.begin(), scores_.end(), -2.0f);
        float bestScore = -2.0f;
        int bestP = minPeriod_;

        // Coarse pass (step = 2 samples) across mirrored contiguous ring buffers
        for (int P = minPeriod_; P <= maxJump; P += 2) {
            const size_t baseB = static_cast<size_t>((rA + P - win) & kMask);
            float dot = 0.0f, normB = 0.0f;
            computeDotAndNorm(rawA, &ring_[baseB], win, dot, normB);
            const float nccRaw = dot / (std::sqrt(normA * normB) + 1.0e-9f);

            float nccLp = nccRaw;
            if (lpNormA > 1.0e-9f) {
                float lpDot = 0.0f, lpNormB = 0.0f;
                computeDotAndNorm(lpA, &lpRing_[baseB], win, lpDot, lpNormB);
                if (lpNormB > 1.0e-9f) {
                    nccLp = lpDot / (std::sqrt(lpNormA * lpNormB) + 1.0e-9f);
                }
            }

            const float sc = rawWeight * nccRaw + lpWeight * nccLp;
            scores_[static_cast<size_t>(P)] = sc;
            if (sc > bestScore) {
                bestScore = sc;
                bestP = P;
            }
        }

        // Subharmonic / octave guard: prevent locking onto 2*f0 or 3*f0 when upper harmonics dominate
        const int multMax = (lpRatio > 0.06f) ? 3 : 2;
        for (int mult = multMax; mult >= 2; --mult) {
            const int pTarget = mult * bestP;
            if (pTarget <= maxJump - 6) {
                float bestM = -2.0f;
                int pMBest = pTarget;
                const int lo = std::max(minPeriod_, pTarget - 8);
                const int hi = std::min(maxJump, pTarget + 8);
                for (int p = lo; p <= hi; p += 2) {
                    if (scores_[static_cast<size_t>(p)] > bestM) {
                        bestM = scores_[static_cast<size_t>(p)];
                        pMBest = p;
                    }
                }
                if (bestM >= bestScore - 0.07f && bestM > 0.62f) {
                    bestP = pMBest;
                    bestScore = bestM;
                    break;
                }
            }
        }

        // Fine 1-sample refinement around the winning period peak
        int refined = bestP;
        float fineBest = -2.0f;
        const int fLo = std::max(minPeriod_, bestP - 3);
        const int fHi = std::min(maxJump, bestP + 3);
        for (int P = fLo; P <= fHi; ++P) {
            const size_t baseB = static_cast<size_t>((rA + P - win) & kMask);
            float dot = 0.0f, normB = 0.0f;
            computeDotAndNorm(rawA, &ring_[baseB], win, dot, normB);
            const float nccRaw = dot / (std::sqrt(normA * normB) + 1.0e-9f);

            float nccLp = nccRaw;
            if (lpNormA > 1.0e-9f) {
                float lpDot = 0.0f, lpNormB = 0.0f;
                computeDotAndNorm(lpA, &lpRing_[baseB], win, lpDot, lpNormB);
                if (lpNormB > 1.0e-9f) {
                    nccLp = lpDot / (std::sqrt(lpNormA * lpNormB) + 1.0e-9f);
                }
            }

            const float sc = rawWeight * nccRaw + lpWeight * nccLp;
            if (sc > fineBest) {
                fineBest = sc;
                refined = P;
            }
        }

        lastCorr_ = std::clamp(fineBest, 0.0f, 1.0f);
        lastPeriodSamples_ = static_cast<float>(refined);
        lastPeriodValid_ = (fineBest > 0.55f);
        return refined;
    }

    double sr_ = 48000.0;
    std::vector<float> ring_;
    std::vector<float> lpRing_;
    std::vector<float> scores_;

    BiquadFilter lpFilt1_;
    BiquadFilter lpFilt2_;

    int minPeriod_ = 73;
    int maxPeriod_ = 774;
    int fadeLen_ = 336;
    int corrWin_ = 528;
    int meanLatency_ = 567;
    double minSafeLag_ = 176.0;
    double triggerLag_ = 958.0;

    int writePos_ = 0;
    double lagA_ = 567.0;
    double lagB_ = 567.0;
    int xfadePos_ = -1;

    float lastCorr_ = 1.0f;
    float lastPeriodSamples_ = 436.0f;
    bool lastPeriodValid_ = false;
    int silentReadStreak_ = 0;
    int silentNomStreak_ = 0;
    float readMs_ = 0.0f;
    float nomMs_ = 0.0f;
};

} // namespace dsp
