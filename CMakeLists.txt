cmake_minimum_required(VERSION 3.22)
project(GuitarToDIBass VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# ─── 1. RTNeural (Compile-Time Real-Time Neural Inference Engine) ─────────────
set(RTNEURAL_BUILD_TESTS OFF CACHE BOOL "Disable RTNeural tests" FORCE)
set(RTNEURAL_STL OFF CACHE BOOL "Disable scalar STL backend" FORCE)
set(RTNEURAL_XSIMD ON CACHE BOOL "Use RTNeural xsimd AVX2 SIMD backend" FORCE)
set(RTNEURAL_USE_AVX ON CACHE BOOL "Enable AVX2 32-byte SIMD alignment" FORCE)

FetchContent_Declare(
    RTNeural
    GIT_REPOSITORY https://github.com/jatinchowdhury18/RTNeural.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(RTNeural)

# ─── 2. JUCE Framework (Official 9.0.5 Release Archive, No Git) ──────────────
FetchContent_Declare(
    JUCE
    URL https://github.com/juce-framework/JUCE/releases/download/9.0.5/juce-9.0.5-linux.zip
    SOURCE_SUBDIR JUCE
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(JUCE)

# ─── Compiler Optimization Helper (Rule 4) ────────────────────────────────────
function(apply_realtime_dsp_optimizations target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /O2 /fp:fast /arch:AVX2)
    else()
        target_compile_options(${target_name} PRIVATE -O3 -ffast-math -mavx2 -mfma)
    endif()
endfunction()

set(DEFAULT_MODEL_JSON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/models/default_di_bass.json")

# ─── Target 1: Standalone Offline Dataset Shifter CLI ────────────────────────
add_executable(OfflineShifterCLI
    src/cli/OfflineShifterCLI.cpp
)

target_include_directories(OfflineShifterCLI PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(OfflineShifterCLI SYSTEM PRIVATE
    ${rtneural_SOURCE_DIR}
    ${rtneural_SOURCE_DIR}/modules/xsimd/include
    ${rtneural_SOURCE_DIR}/modules/json
)

target_link_libraries(OfflineShifterCLI PRIVATE
    RTNeural
)

target_compile_definitions(OfflineShifterCLI PRIVATE
    DEFAULT_MODEL_PATH="${DEFAULT_MODEL_JSON_PATH}"
)

apply_realtime_dsp_optimizations(OfflineShifterCLI)

# ─── Target 2: JUCE VST3 / Standalone Plugin ─────────────────────────────────
juce_add_plugin(GuitarToDIBass
    COMPANY_NAME                "HybridAudioDSP"
    PLUGIN_MANUFACTURER_CODE    HDSP
    PLUGIN_CODE                 GtBs
    FORMATS                     VST3 Standalone
    PRODUCT_NAME                "GuitarToDIBass"
    IS_SYNTH                    FALSE
    NEEDS_MIDI_INPUT            FALSE
    NEEDS_MIDI_OUTPUT           FALSE
    IS_MIDI_EFFECT              FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD     FALSE
    VST3_CATEGORIES             "Fx" "Pitch Shift" "Distortion"
)

target_sources(GuitarToDIBass PRIVATE
    src/plugin/PluginProcessor.cpp
    src/plugin/PluginEditor.cpp
)

target_include_directories(GuitarToDIBass PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(GuitarToDIBass SYSTEM PUBLIC
    ${rtneural_SOURCE_DIR}
    ${rtneural_SOURCE_DIR}/modules/xsimd/include
    ${rtneural_SOURCE_DIR}/modules/json
)

target_compile_definitions(GuitarToDIBass PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    DEFAULT_MODEL_PATH="${DEFAULT_MODEL_JSON_PATH}"
)

target_link_libraries(GuitarToDIBass
    PRIVATE
        RTNeural
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_gui_extra
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

apply_realtime_dsp_optimizations(GuitarToDIBass)
if(TARGET GuitarToDIBass_VST3)
    apply_realtime_dsp_optimizations(GuitarToDIBass_VST3)
endif()
if(TARGET GuitarToDIBass_Standalone)
    apply_realtime_dsp_optimizations(GuitarToDIBass_Standalone)
endif()
