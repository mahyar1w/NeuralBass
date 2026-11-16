#pragma once

#include "../dsp/HybridBassEngine.h"
#include <juce_audio_processors/juce_audio_processors.h>
#include <atomic>

class GuitarToDIBassAudioProcessor : public juce::AudioProcessor {
public:
    GuitarToDIBassAudioProcessor();
    ~GuitarToDIBassAudioProcessor() override = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    using juce::AudioProcessor::processBlock;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& getAPVTS() noexcept { return apvts_; }

    /**
     * Loads a JSON RTNeural preset on the message thread and atomically swaps
     * the compile-time ModelT buffer used on the real-time audio thread.
     */
    bool loadNeuralModelFromFile(const juce::File& jsonFile);
    juce::String getLoadedModelName() const { return loadedModelName_; }

    struct BassPreset {
        int id;
        juce::String name;
        juce::String filename;
    };

    static const std::vector<BassPreset>& getBuiltInPresets();
    bool loadPresetByIndex(int index);
    int getCurrentPresetIndex() const noexcept { return currentPresetIndex_; }

    double getLatencyMs() const noexcept {
        const double sr = getSampleRate();
        if (sr <= 0.0) return 9.0;
        return (static_cast<double>(engine_.getLatencySamples()) / sr) * 1000.0;
    }

    bool isEngineMultithreaded() const noexcept {
        return engine_.isMultithreaded();
    }

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

private:
    void tryLoadDefaultModel();

    juce::AudioProcessorValueTreeState apvts_;
    dsp::HybridBassEngine engine_;

    // Lock-free atomic parameter pointers (safe to read inside processBlock)
    std::atomic<float>* subHarmonicsParam_ = nullptr;
    std::atomic<float>* pickAttackParam_   = nullptr;
    std::atomic<float>* diDriveParam_      = nullptr;
    std::atomic<float>* outputLevelParam_  = nullptr;

    juce::String loadedModelName_ { "Built-in Default DI Bass" };
    juce::String loadedModelPath_;
    int currentPresetIndex_ = 1;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GuitarToDIBassAudioProcessor)
};
