#include "../dsp/BiquadFilter.h"
#include "../dsp/SubOctaveTracker.h"
#include "../dsp/PitchSyncOctaveShifter.h"
#include "../dsp/HybridBassEngine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct WavData {
    uint32_t sampleRate = 48000;
    std::vector<float> samples; // Mono normalized [-1.0, 1.0]
};

bool readWavFile(const fs::path& filePath, WavData& outWav) {
    std::ifstream in(filePath, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    char riffHeader[12];
    if (!in.read(riffHeader, 12)) return false;
    if (std::memcmp(riffHeader, "RIFF", 4) != 0 || std::memcmp(riffHeader + 8, "WAVE", 4) != 0) {
        return false;
    }

    uint16_t audioFormat = 0;
    uint16_t numChannels = 0;
    uint32_t sampleRate = 0;
    uint16_t bitsPerSample = 0;
    std::vector<uint8_t> rawData;

    while (in.good() && !in.eof()) {
        char chunkId[4];
        uint32_t chunkSize = 0;
        if (!in.read(chunkId, 4)) break;
        if (!in.read(reinterpret_cast<char*>(&chunkSize), 4)) break;

        if (std::memcmp(chunkId, "fmt ", 4) == 0) {
            std::vector<uint8_t> fmtBuf(chunkSize);
            if (!in.read(reinterpret_cast<char*>(fmtBuf.data()), chunkSize)) return false;
            if (chunkSize >= 16) {
                std::memcpy(&audioFormat, fmtBuf.data() + 0, 2);
                std::memcpy(&numChannels, fmtBuf.data() + 2, 2);
                std::memcpy(&sampleRate, fmtBuf.data() + 4, 4);
                std::memcpy(&bitsPerSample, fmtBuf.data() + 14, 2);
                // Handle WAVE_FORMAT_EXTENSIBLE (0xFFFE)
                if (audioFormat == 0xFFFE && chunkSize >= 40) {
                    std::memcpy(&audioFormat, fmtBuf.data() + 24, 2);
                }
            }
        } else if (std::memcmp(chunkId, "data", 4) == 0) {
            rawData.resize(chunkSize);
            if (!in.read(reinterpret_cast<char*>(rawData.data()), chunkSize)) return false;
        } else {
            in.seekg(chunkSize, std::ios::cur);
        }

        // Handle odd-byte RIFF padding
        if (chunkSize & 1) {
            in.seekg(1, std::ios::cur);
        }
    }

    if (numChannels == 0 || sampleRate == 0 || rawData.empty()) {
        return false;
    }

    outWav.sampleRate = sampleRate;
    const size_t bytesPerSample = bitsPerSample / 8;
    const size_t frameSize = bytesPerSample * numChannels;
    if (frameSize == 0) return false;

    const size_t numFrames = rawData.size() / frameSize;
    outWav.samples.assign(numFrames, 0.0f);

    for (size_t f = 0; f < numFrames; ++f) {
        float monoSum = 0.0f;
        for (uint16_t ch = 0; ch < numChannels; ++ch) {
            const uint8_t* ptr = rawData.data() + f * frameSize + ch * bytesPerSample;
            float sample = 0.0f;
            if (audioFormat == 3 && bitsPerSample == 32) {
                std::memcpy(&sample, ptr, sizeof(float));
            } else if (audioFormat == 1 && bitsPerSample == 16) {
                int16_t val = 0;
                std::memcpy(&val, ptr, sizeof(int16_t));
                sample = static_cast<float>(val) / 32768.0f;
            } else if (audioFormat == 1 && bitsPerSample == 24) {
                int32_t val = (static_cast<int32_t>(ptr[0])) |
                              (static_cast<int32_t>(ptr[1]) << 8) |
                              (static_cast<int32_t>(static_cast<int8_t>(ptr[2])) << 16);
                sample = static_cast<float>(val) / 8388608.0f;
            } else if (audioFormat == 1 && bitsPerSample == 32) {
                int32_t val = 0;
                std::memcpy(&val, ptr, sizeof(int32_t));
                sample = static_cast<float>(static_cast<double>(val) / 2147483648.0);
            }
            monoSum += sample;
        }
        outWav.samples[f] = monoSum / static_cast<float>(numChannels);
    }

    return true;
}

bool writeWavFileFloat32(const fs::path& filePath, const WavData& wav) {
    if (filePath.has_parent_path()) {
        fs::create_directories(filePath.parent_path());
    }

    std::ofstream out(filePath, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    const uint16_t audioFormat = 3; // IEEE Float
    const uint16_t numChannels = 1;
    const uint16_t bitsPerSample = 32;
    const uint32_t byteRate = wav.sampleRate * numChannels * (bitsPerSample / 8);
    const uint16_t blockAlign = numChannels * (bitsPerSample / 8);
    const uint32_t dataChunkSize = static_cast<uint32_t>(wav.samples.size() * sizeof(float));
    const uint32_t fmtChunkSize = 16;
    const uint32_t riffChunkSize = 4 + (8 + fmtChunkSize) + (8 + dataChunkSize);

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&riffChunkSize), 4);
    out.write("WAVE", 4);

    out.write("fmt ", 4);
    out.write(reinterpret_cast<const char*>(&fmtChunkSize), 4);
    out.write(reinterpret_cast<const char*>(&audioFormat), 2);
    out.write(reinterpret_cast<const char*>(&numChannels), 2);
    out.write(reinterpret_cast<const char*>(&wav.sampleRate), 4);
    out.write(reinterpret_cast<const char*>(&byteRate), 4);
    out.write(reinterpret_cast<const char*>(&blockAlign), 2);
    out.write(reinterpret_cast<const char*>(&bitsPerSample), 2);

    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&dataChunkSize), 4);
    out.write(reinterpret_cast<const char*>(wav.samples.data()), dataChunkSize);

    return out.good();
}

/**
 * Applies Stage 1 (60 Hz 2nd-order Butterworth HPF) + Stage 2 (PitchSyncOctaveShifter:
 * -12.0st time-domain pitch shift, ~11.8 ms latency) + Stage 2b (SubOctaveTracker: 12% clean sub-octave)
 * with exact latency compensation so the shifted output aligns sample-for-sample with target_bass WAVs
 * and matches xBase inside HybridBassEngine 100% identically.
 */
void shiftGuitarBuffer(const std::vector<float>& input,
                       std::vector<float>& output,
                       uint32_t sampleRate) {
    dsp::BiquadFilter hpf;
    hpf.setHighPass(static_cast<double>(sampleRate), 60.0f, 0.70710678f);

    dsp::PitchSyncOctaveShifter pitchShifter;
    pitchShifter.prepare(static_cast<double>(sampleRate));
    const int totalLatency = pitchShifter.getLatencySamples();

    dsp::SubOctaveTracker subOctaveTracker;
    constexpr size_t kBlockSize = 512;
    subOctaveTracker.prepare(static_cast<double>(sampleRate), static_cast<int>(kBlockSize), totalLatency + 1024);
    subOctaveTracker.setLatencyCompensation(totalLatency);

    const size_t totalInSamples = input.size() + static_cast<size_t>(totalLatency);
    std::vector<float> conditioned(totalInSamples, 0.0f);
    std::vector<float> subOctAll(totalInSamples, 0.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        const float filtered = hpf.processSample(input[i]);
        conditioned[i] = filtered;
        subOctAll[i] = subOctaveTracker.processSample(filtered);
    }
    for (size_t i = input.size(); i < totalInSamples; ++i) {
        subOctAll[i] = subOctaveTracker.processSample(0.0f);
    }

    std::vector<float> stretchedAll(totalInSamples, 0.0f);
    pitchShifter.processBlock(conditioned.data(), stretchedAll.data(), static_cast<int>(totalInSamples));

    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const float s = stretchedAll[i + static_cast<size_t>(totalLatency)];
        const float o = subOctAll[i + static_cast<size_t>(totalLatency)];
        output[i] = 0.88f * s + 0.12f * o;
    }
}

bool processSingleFile(const fs::path& inPath, const fs::path& outPath) {
    WavData inWav;
    if (!readWavFile(inPath, inWav)) {
        std::cerr << "[OfflineShifterCLI] Failed to read WAV: " << inPath << "\n";
        return false;
    }

    WavData outWav;
    outWav.sampleRate = inWav.sampleRate;
    shiftGuitarBuffer(inWav.samples, outWav.samples, inWav.sampleRate);

    if (!writeWavFileFloat32(outPath, outWav)) {
        std::cerr << "[OfflineShifterCLI] Failed to write WAV: " << outPath << "\n";
        return false;
    }

    std::cout << "[OfflineShifterCLI] Shifted (-12st pitch, 0st formant + 12% sub-octave, low-latency): "
              << inPath.filename().string() << " -> " << outPath.string()
              << " (" << outWav.samples.size() << " samples @ " << outWav.sampleRate << " Hz)\n";
    return true;
}

bool renderFullEngineFile(const fs::path& inPath,
                          const fs::path& outPath,
                          const fs::path& modelJsonPath) {
    WavData inWav;
    if (!readWavFile(inPath, inWav)) {
        std::cerr << "[OfflineShifterCLI] Failed to read WAV: " << inPath << "\n";
        return false;
    }

    dsp::HybridBassEngine engine;
    constexpr int kBlockSize = 512;
    engine.prepare(static_cast<double>(inWav.sampleRate), kBlockSize);

    if (!modelJsonPath.empty() && fs::exists(modelJsonPath)) {
        if (engine.loadModelFromJsonFile(modelJsonPath.string())) {
            std::cout << "[OfflineShifterCLI] Loaded RTNeural weights: " << modelJsonPath.string() << "\n";
        }
    }

    const int latency = engine.getLatencySamples();
    std::vector<float> workBuf(inWav.samples.size() + static_cast<size_t>(latency), 0.0f);
    std::copy(inWav.samples.begin(), inWav.samples.end(), workBuf.begin());

    engine.processBlock(workBuf.data(),
                        static_cast<int>(workBuf.size()),
                        1.0f, // subHarmonicsGain
                        1.0f, // pickAttackGain
                        1.0f, // diDrive
                        1.0f); // outputLevelLinear

    WavData outWav;
    outWav.sampleRate = inWav.sampleRate;
    outWav.samples.assign(workBuf.begin() + latency, workBuf.end());

    if (!writeWavFileFloat32(outPath, outWav)) {
        std::cerr << "[OfflineShifterCLI] Failed to write WAV: " << outPath << "\n";
        return false;
    }

    std::cout << "[OfflineShifterCLI] Rendered full HybridBassEngine output: " << outPath.string() << "\n";
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc >= 4 && std::string(argv[1]) == "--full-engine") {
        const fs::path inPath = argv[2];
        const fs::path outPath = argv[3];
        const fs::path modelPath = (argc >= 5) ? fs::path(argv[4]) : fs::path("models/default_di_bass.json");
        return renderFullEngineFile(inPath, outPath, modelPath) ? 0 : 1;
    }

    fs::path inputPath  = "ml_training/dataset/raw_guitar";
    fs::path outputPath = "ml_training/dataset/shifted_guitar";

    if (argc >= 3) {
        inputPath  = argv[1];
        outputPath = argv[2];
    }

    if (fs::is_regular_file(inputPath)) {
        return processSingleFile(inputPath, outputPath) ? 0 : 1;
    }

    if (!fs::exists(inputPath) || !fs::is_directory(inputPath)) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " [input_dir_or_wav] [output_dir_or_wav]\n"
                  << "  " << argv[0] << " --full-engine <input.wav> <output.wav> [model.json]\n";
        return 1;
    }

    fs::create_directories(outputPath);

    std::vector<fs::path> wavFiles;
    for (const auto& entry : fs::directory_iterator(inputPath)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".wav") {
                wavFiles.push_back(entry.path());
            }
        }
    }
    std::sort(wavFiles.begin(), wavFiles.end());

    if (wavFiles.empty()) {
        std::cout << "[OfflineShifterCLI] No WAV files found in " << inputPath.string() << "\n";
        return 0;
    }

    int successCount = 0;
    for (const auto& wavFile : wavFiles) {
        const fs::path destFile = outputPath / wavFile.filename();
        if (processSingleFile(wavFile, destFile)) {
            ++successCount;
        }
    }

    std::cout << "[OfflineShifterCLI] Completed pre-shifting " << successCount << "/"
              << wavFiles.size() << " files into " << outputPath.string() << "\n";
    return (successCount == static_cast<int>(wavFiles.size())) ? 0 : 1;
}
