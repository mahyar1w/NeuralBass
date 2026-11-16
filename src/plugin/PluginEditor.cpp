#include "PluginEditor.h"

GuitarToDIBassAudioProcessorEditor::GuitarToDIBassAudioProcessorEditor(
    GuitarToDIBassAudioProcessor& p)
    : AudioProcessorEditor(&p), processorRef_(p) {
    setLookAndFeel(&lookAndFeel_);

    configureRotaryKnob(subHarmonicsSlider_, subHarmonicsLabel_, "SUB / HARMONICS");
    configureRotaryKnob(pickAttackSlider_,   pickAttackLabel_,   "PICK ATTACK");
    configureRotaryKnob(diDriveSlider_,      diDriveLabel_,      "DI DRIVE");
    configureRotaryKnob(outputLevelSlider_,  outputLevelLabel_,  "OUTPUT LEVEL");
    outputLevelSlider_.setTextValueSuffix(" dB");

    auto& apvts = processorRef_.getAPVTS();
    subHarmonicsAttachment_ = std::make_unique<SliderAttachment>(apvts, "sub_harmonics", subHarmonicsSlider_);
    pickAttackAttachment_   = std::make_unique<SliderAttachment>(apvts, "pick_attack",   pickAttackSlider_);
    diDriveAttachment_      = std::make_unique<SliderAttachment>(apvts, "di_drive",      diDriveSlider_);
    outputLevelAttachment_  = std::make_unique<SliderAttachment>(apvts, "output_level",  outputLevelSlider_);

    addAndMakeVisible(loadJsonButton_);
    loadJsonButton_.onClick = [this]() { openJsonModelChooser(); };

    // Built-in 10-model bass sound selector
    for (const auto& preset : processorRef_.getBuiltInPresets()) {
        presetSelector_.addItem(preset.name, preset.id);
    }
    presetSelector_.setSelectedId(processorRef_.getCurrentPresetIndex(), juce::dontSendNotification);
    presetSelector_.setColour(juce::ComboBox::backgroundColourId, juce::Colour(0xff1e293b));
    presetSelector_.setColour(juce::ComboBox::textColourId, juce::Colour(0xfff8fafc));
    presetSelector_.setColour(juce::ComboBox::outlineColourId, juce::Colour(0xff334155));
    presetSelector_.setColour(juce::ComboBox::arrowColourId, juce::Colour(0xff38bdf8));
    presetSelector_.onChange = [this]() {
        const int selectedId = presetSelector_.getSelectedId();
        if (selectedId > 0) {
            processorRef_.loadPresetByIndex(selectedId);
            modelStatusLabel_.setText("Model: " + processorRef_.getLoadedModelName(),
                                      juce::dontSendNotification);
            repaint();
        }
    };
    addAndMakeVisible(presetSelector_);

    modelStatusLabel_.setText("Model: " + processorRef_.getLoadedModelName(),
                              juce::dontSendNotification);
    modelStatusLabel_.setColour(juce::Label::textColourId, juce::Colour(0xff94a3b8));
    modelStatusLabel_.setFont(juce::FontOptions(12.0f));
    modelStatusLabel_.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(modelStatusLabel_);

    setSize(620, 310);
}

GuitarToDIBassAudioProcessorEditor::~GuitarToDIBassAudioProcessorEditor() {
    setLookAndFeel(nullptr);
}

void GuitarToDIBassAudioProcessorEditor::configureRotaryKnob(
    juce::Slider& slider, juce::Label& label, const juce::String& text) {
    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 72, 20);
    addAndMakeVisible(slider);

    label.setText(text, juce::dontSendNotification);
    label.setJustificationType(juce::Justification::centred);
    label.setColour(juce::Label::textColourId, juce::Colour(0xffcbd5e1));
    label.setFont(juce::FontOptions(12.0f, juce::Font::bold));
    addAndMakeVisible(label);
}

void GuitarToDIBassAudioProcessorEditor::openJsonModelChooser() {
    fileChooser_ = std::make_unique<juce::FileChooser>(
        "Select RTNeural GRU Bass Weights (.json)",
        juce::File::getCurrentWorkingDirectory().getChildFile("models"),
        "*.json");

    constexpr int chooserFlags = juce::FileBrowserComponent::openMode |
                                 juce::FileBrowserComponent::canSelectFiles;

    fileChooser_->launchAsync(chooserFlags, [this](const juce::FileChooser& chooser) {
        const auto selected = chooser.getResult();
        if (selected.existsAsFile()) {
            if (processorRef_.loadNeuralModelFromFile(selected)) {
                modelStatusLabel_.setText("Model: " + processorRef_.getLoadedModelName(),
                                          juce::dontSendNotification);
            } else {
                modelStatusLabel_.setText("Error: Invalid RTNeural JSON schema",
                                          juce::dontSendNotification);
            }
        }
    });
}

void GuitarToDIBassAudioProcessorEditor::paint(juce::Graphics& g) {
    // Dark studio DI chassis background
    g.fillAll(juce::Colour(0xff090d16));

    auto bounds = getLocalBounds().reduced(12);
    g.setColour(juce::Colour(0xff111827));
    g.fillRoundedRectangle(bounds.toFloat(), 10.0f);

    g.setColour(juce::Colour(0xff1f2937));
    g.drawRoundedRectangle(bounds.toFloat(), 10.0f, 1.5f);

    // Header title
    g.setColour(juce::Colour(0xfff8fafc));
    g.setFont(juce::FontOptions(19.0f, juce::Font::bold));
    g.drawText("GUITAR TO DI BASS", 28, 20, 240, 22, juce::Justification::centredLeft);

    // Dynamic latency & engine architecture subtitle
    const double latMs = processorRef_.getLatencyMs();
    const juce::String engineMode = processorRef_.isEngineMultithreaded() ? "Multi-Core" : "Single-Core";
    const juce::String subText = "AVX2 SIMD | ~" + juce::String(latMs, 1) + " ms (" + engineMode + ")";

    g.setColour(juce::Colour(0xff38bdf8));
    g.setFont(juce::FontOptions(11.0f));
    g.drawText(subText, 28, 42, 240, 16, juce::Justification::centredLeft);
}

void GuitarToDIBassAudioProcessorEditor::resized() {
    auto area = getLocalBounds().reduced(20);

    // Top header bar: Title on left, Preset Selector + Custom JSON on right
    auto headerArea = area.removeFromTop(48);
    auto rightHeader = headerArea.removeFromRight(370);

    loadJsonButton_.setBounds(rightHeader.removeFromRight(135).withSizeKeepingCentre(135, 28));
    rightHeader.removeFromRight(8);
    presetSelector_.setBounds(rightHeader.withSizeKeepingCentre(rightHeader.getWidth(), 28));

    // Bottom footer status line
    modelStatusLabel_.setBounds(area.removeFromBottom(22));

    area.removeFromTop(6);

    // 4 Rotary Knobs in a row
    const int numKnobs = 4;
    const int knobWidth = area.getWidth() / numKnobs;

    auto placeKnob = [&](juce::Slider& slider, juce::Label& label, juce::Rectangle<int> slot) {
        label.setBounds(slot.removeFromTop(20));
        slider.setBounds(slot.reduced(4));
    };

    placeKnob(subHarmonicsSlider_, subHarmonicsLabel_, area.removeFromLeft(knobWidth));
    placeKnob(pickAttackSlider_,   pickAttackLabel_,   area.removeFromLeft(knobWidth));
    placeKnob(diDriveSlider_,      diDriveLabel_,      area.removeFromLeft(knobWidth));
    placeKnob(outputLevelSlider_,  outputLevelLabel_,  area);
}
