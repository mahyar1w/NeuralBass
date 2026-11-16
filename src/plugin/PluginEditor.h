#pragma once

#include "PluginProcessor.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <memory>

class DarkDIKnobLookAndFeel : public juce::LookAndFeel_V4 {
public:
    DarkDIKnobLookAndFeel() {
        setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xffe2e8f0));
        setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        setColour(juce::TextButton::buttonColourId, juce::Colour(0xff1e293b));
        setColour(juce::TextButton::textColourOffId, juce::Colour(0xff38bdf8));
    }

    void drawRotarySlider(juce::Graphics& g,
                          int x, int y, int width, int height,
                          float sliderPosProportional,
                          float rotaryStartAngle,
                          float rotaryEndAngle,
                          juce::Slider& slider) override {
        juce::ignoreUnused(slider);
        const float radius = static_cast<float>(juce::jmin(width / 2, height / 2)) - 8.0f;
        const float centreX = static_cast<float>(x) + static_cast<float>(width) * 0.5f;
        const float centreY = static_cast<float>(y) + static_cast<float>(height) * 0.5f;
        const float rx = centreX - radius;
        const float ry = centreY - radius;
        const float rw = radius * 2.0f;
        const float angle = rotaryStartAngle + sliderPosProportional * (rotaryEndAngle - rotaryStartAngle);

        // Knob body
        g.setColour(juce::Colour(0xff0f172a));
        g.fillEllipse(rx, ry, rw, rw);

        // Background arc
        juce::Path bgArc;
        bgArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                            rotaryStartAngle, rotaryEndAngle, true);
        g.setColour(juce::Colour(0xff334155));
        g.strokePath(bgArc, juce::PathStrokeType(4.0f, juce::PathStrokeType::curved,
                                                 juce::PathStrokeType::rounded));

        // Active value arc
        juce::Path valueArc;
        valueArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                               rotaryStartAngle, angle, true);
        g.setColour(juce::Colour(0xff38bdf8));
        g.strokePath(valueArc, juce::PathStrokeType(4.0f, juce::PathStrokeType::curved,
                                                    juce::PathStrokeType::rounded));

        // Pointer tick
        juce::Path pointer;
        const float pointerLength = radius * 0.65f;
        const float pointerThickness = 3.0f;
        pointer.addRoundedRectangle(-pointerThickness * 0.5f, -radius + 4.0f,
                                    pointerThickness, pointerLength, 1.5f);
        pointer.applyTransform(juce::AffineTransform::rotation(angle).translated(centreX, centreY));
        g.setColour(juce::Colour(0xfff8fafc));
        g.fillPath(pointer);
    }
};

class GuitarToDIBassAudioProcessorEditor : public juce::AudioProcessorEditor {
public:
    explicit GuitarToDIBassAudioProcessorEditor(GuitarToDIBassAudioProcessor&);
    ~GuitarToDIBassAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;

    void configureRotaryKnob(juce::Slider& slider, juce::Label& label, const juce::String& text);
    void openJsonModelChooser();

    GuitarToDIBassAudioProcessor& processorRef_;
    DarkDIKnobLookAndFeel lookAndFeel_;

    juce::Slider subHarmonicsSlider_;
    juce::Slider pickAttackSlider_;
    juce::Slider diDriveSlider_;
    juce::Slider outputLevelSlider_;

    juce::Label subHarmonicsLabel_;
    juce::Label pickAttackLabel_;
    juce::Label diDriveLabel_;
    juce::Label outputLevelLabel_;

    std::unique_ptr<SliderAttachment> subHarmonicsAttachment_;
    std::unique_ptr<SliderAttachment> pickAttackAttachment_;
    std::unique_ptr<SliderAttachment> diDriveAttachment_;
    std::unique_ptr<SliderAttachment> outputLevelAttachment_;

    juce::ComboBox presetSelector_;
    juce::TextButton loadJsonButton_ { "Load Custom JSON..." };
    juce::Label modelStatusLabel_;
    std::unique_ptr<juce::FileChooser> fileChooser_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GuitarToDIBassAudioProcessorEditor)
};
