#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cmath>

GuitarToDIBassAudioProcessor::GuitarToDIBassAudioProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts_(*this, nullptr, "PARAMETERS", createParameterLayout()) {
    subHarmonicsParam_ = apvts_.getRawParameterValue("sub_harmonics");
    pickAttackParam_   = apvts_.getRawParameterValue("pick_attack");
    diDriveParam_      = apvts_.getRawParameterValue("di_drive");
    outputLevelParam_  = apvts_.getRawParameterValue("output_level");

    tryLoadDefaultModel();
}

juce::AudioProcessorValueTreeState::ParameterLayout
GuitarToDIBassAudioProcessor::createParameterLayout() {
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"sub_harmonics", 1},
        "Sub / Harmonics",
        juce::NormalisableRange<float>(0.0f, 2.0f, 0.01f),
        1.0f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"pick_attack", 1},
        "Pick Attack",
        juce::NormalisableRange<float>(0.0f, 2.0f, 0.01f),
        1.0f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"di_drive", 1},
        "DI Drive",
        juce::NormalisableRange<float>(0.5f, 4.0f, 0.01f),
        1.0f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"output_level", 1},
        "Output Level",
        juce::NormalisableRange<float>(-24.0f, 12.0f, 0.1f),
        0.0f));

    return { params.begin(), params.end() };
}

const std::vector<GuitarToDIBassAudioProcessor::BassPreset>&
GuitarToDIBassAudioProcessor::getBuiltInPresets() {
    static const std::vector<BassPreset> presets = {
        { 1,  "1. All-Around Studio DI Bass (Master)",    "default_di_bass.json" },
        { 2,  "2. Short Scale Vintage Bass (Bridge DI)",   "short_scale_bass.json" },
        { 3,  "3. Fame Baphomet 4 NTB — Fingerstyle",      "fame_baphomet_fs.json" },
        { 4,  "4. Fame Baphomet 4 NTB — Picked",           "fame_baphomet_pk.json" },
        { 5,  "5. Warwick Corvette 4 — Fingerstyle",       "warwick_fs.json" },
        { 6,  "6. Warwick Corvette 4 — Picked",            "warwick_pk.json" },
        { 7,  "7. Yamaha BB604 Jazz — Fingerstyle",        "yamaha_bb604_fs.json" },
        { 8,  "8. Yamaha BB604 Jazz — Picked",             "yamaha_bb604_pk.json" },
        { 9,  "9. Palm-Muted DI Bass (MU)",                "palm_muted_bass.json" },
        { 10, "10. Slap & Pop DI Bass (SP/ST)",            "slap_bass.json" },
        { 11, "11. IDMT Generic Fingerstyle Bass",         "idmt_fingerstyle_bass.json" },
        { 12, "12. IDMT Generic Picked Bass",              "idmt_picked_bass.json" }
    };
    return presets;
}

bool GuitarToDIBassAudioProcessor::loadPresetByIndex(int index) {
    const auto& presets = getBuiltInPresets();
    if (index < 1 || index > static_cast<int>(presets.size())) {
        return false;
    }

    const auto& preset = presets[static_cast<size_t>(index - 1)];
    const juce::String filename = preset.filename;

    std::vector<juce::File> candidateDirs = {
        juce::File::getCurrentWorkingDirectory().getChildFile("models"),
        juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory().getChildFile("models"),
        juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory().getParentDirectory().getChildFile("models")
    };
#ifdef DEFAULT_MODEL_PATH
    candidateDirs.push_back(juce::File(juce::String(DEFAULT_MODEL_PATH)).getParentDirectory());
#endif

    for (const auto& dir : candidateDirs) {
        auto file = dir.getChildFile(filename);
        if (file.existsAsFile() && loadNeuralModelFromFile(file)) {
            currentPresetIndex_ = index;
            loadedModelName_ = preset.name;
            return true;
        }
    }

    // Graceful fallback to default_di_bass.json if specific preset is still training
    for (const auto& dir : candidateDirs) {
        auto file = dir.getChildFile("default_di_bass.json");
        if (file.existsAsFile() && loadNeuralModelFromFile(file)) {
            currentPresetIndex_ = index;
            loadedModelName_ = preset.name + " (Default Fallback)";
            return true;
        }
    }

    return false;
}

void GuitarToDIBassAudioProcessor::tryLoadDefaultModel() {
    if (loadPresetByIndex(1)) {
        return;
    }

    std::vector<juce::File> candidateFiles;

#ifdef DEFAULT_MODEL_PATH
    candidateFiles.emplace_back(juce::String(DEFAULT_MODEL_PATH));
#endif
    candidateFiles.push_back(
        juce::File::getCurrentWorkingDirectory().getChildFile("models/default_di_bass.json"));

    const auto execFile = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
    candidateFiles.push_back(execFile.getParentDirectory().getChildFile("models/default_di_bass.json"));
    candidateFiles.push_back(
        execFile.getParentDirectory().getParentDirectory().getChildFile("models/default_di_bass.json"));

    for (const auto& candidate : candidateFiles) {
        if (candidate.existsAsFile() && loadNeuralModelFromFile(candidate)) {
            return;
        }
    }
}

bool GuitarToDIBassAudioProcessor::loadNeuralModelFromFile(const juce::File& jsonFile) {
    if (!jsonFile.existsAsFile()) {
        return false;
    }

    if (engine_.loadModelFromJsonFile(jsonFile.getFullPathName().toStdString())) {
        loadedModelName_ = jsonFile.getFileName();
        loadedModelPath_ = jsonFile.getFullPathName();
        return true;
    }
    return false;
}

const juce::String GuitarToDIBassAudioProcessor::getName() const {
    return JucePlugin_Name;
}

bool GuitarToDIBassAudioProcessor::acceptsMidi() const { return false; }
bool GuitarToDIBassAudioProcessor::producesMidi() const { return false; }
bool GuitarToDIBassAudioProcessor::isMidiEffect() const { return false; }
double GuitarToDIBassAudioProcessor::getTailLengthSeconds() const { return 0.1; }

int GuitarToDIBassAudioProcessor::getNumPrograms() { return 1; }
int GuitarToDIBassAudioProcessor::getCurrentProgram() { return 0; }
void GuitarToDIBassAudioProcessor::setCurrentProgram(int index) { juce::ignoreUnused(index); }
const juce::String GuitarToDIBassAudioProcessor::getProgramName(int index) {
    juce::ignoreUnused(index);
    return {};
}
void GuitarToDIBassAudioProcessor::changeProgramName(int index, const juce::String& newName) {
    juce::ignoreUnused(index, newName);
}

void GuitarToDIBassAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    engine_.prepare(sampleRate, samplesPerBlock);
    setLatencySamples(engine_.getLatencySamples());
}

void GuitarToDIBassAudioProcessor::releaseResources() {
    engine_.reset();
}

bool GuitarToDIBassAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto outSet = layouts.getMainOutputChannelSet();
    if (outSet != juce::AudioChannelSet::mono() && outSet != juce::AudioChannelSet::stereo()) {
        return false;
    }

    const auto inSet = layouts.getMainInputChannelSet();
    return (inSet == juce::AudioChannelSet::mono() || inSet == juce::AudioChannelSet::stereo());
}

void GuitarToDIBassAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                                juce::MidiBuffer& midiMessages) {
    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;

    const int totalNumInputChannels  = getTotalNumInputChannels();
    const int totalNumOutputChannels = getTotalNumOutputChannels();
    const int numSamples             = buffer.getNumSamples();

    for (int ch = totalNumInputChannels; ch < totalNumOutputChannels; ++ch) {
        buffer.clear(ch, 0, numSamples);
    }

    if (totalNumInputChannels == 0 || numSamples == 0) {
        return;
    }

    // Lock-free atomic parameter reads
    const float subHarmonics = subHarmonicsParam_->load(std::memory_order_relaxed);
    const float pickAttack   = pickAttackParam_->load(std::memory_order_relaxed);
    const float diDrive      = diDriveParam_->load(std::memory_order_relaxed);
    const float outputDb     = outputLevelParam_->load(std::memory_order_relaxed);
    const float outputLinear = std::pow(10.0f, outputDb * 0.05f);

    float* ch0 = buffer.getWritePointer(0);

    // Run the mono DI bass DSP + RTNeural chain on channel 0
    engine_.processBlock(ch0, numSamples, subHarmonics, pickAttack, diDrive, outputLinear);

    // Duplicate mono DI bass output across all remaining output channels
    for (int ch = 1; ch < totalNumOutputChannels; ++ch) {
        buffer.copyFrom(ch, 0, ch0, numSamples);
    }
}

bool GuitarToDIBassAudioProcessor::hasEditor() const { return true; }

juce::AudioProcessorEditor* GuitarToDIBassAudioProcessor::createEditor() {
    return new GuitarToDIBassAudioProcessorEditor(*this);
}

void GuitarToDIBassAudioProcessor::getStateInformation(juce::MemoryBlock& destData) {
    auto state = apvts_.copyState();
    state.setProperty("loadedModelPath", loadedModelPath_, nullptr);
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void GuitarToDIBassAudioProcessor::setStateInformation(const void* data, int sizeInBytes) {
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState != nullptr && xmlState->hasTagName(apvts_.state.getType())) {
        auto tree = juce::ValueTree::fromXml(*xmlState);
        if (tree.hasProperty("loadedModelPath")) {
            const juce::String savedPath = tree.getProperty("loadedModelPath").toString();
            if (savedPath.isNotEmpty()) {
                loadNeuralModelFromFile(juce::File(savedPath));
            }
        }
        apvts_.replaceState(tree);
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new GuitarToDIBassAudioProcessor();
}
