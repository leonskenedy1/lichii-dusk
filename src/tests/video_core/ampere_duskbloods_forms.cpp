// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};

std::vector<u64> LowerMultisample(u32 sample_register) {
    Instruction multisample{{0x04007b66, 0x32000006, 0x049e4f02, 0x001fc200}};
    multisample.words[1] = (multisample.words[1] & ~0xffU) | sample_register;
    const auto lowered = sample_register >= 246
                             ? LowerToMaxwell(std::array{
                                   Instruction{{0x04088c08, 0x0000000a, 0x0d800000, 0x000fe400}},
                                   multisample, Exit})
                             : LowerToMaxwell(std::array{multisample, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    return lowered.instructions;
}
} // Anonymous namespace

TEST_CASE("Ampere TLD.MS stages the sample before writing coordinate scratch registers",
          "[video_core][ampere]") {
    for (const u32 source : {6U, 250U, 251U, 253U}) {
        const auto lowered = LowerMultisample(source);
        const auto sample = std::find_if(lowered.begin(), lowered.end(), [](u64 word) {
            return (word >> 48) == 0x5c98 && (word & 0xff) == 253;
        });
        const auto x = std::find_if(lowered.begin(), lowered.end(), [](u64 word) {
            return (word >> 48) == 0x5c98 && (word & 0xff) == 250;
        });
        const auto y = std::find_if(lowered.begin(), lowered.end(), [](u64 word) {
            return (word >> 48) == 0x5c98 && (word & 0xff) == 251;
        });
        const auto fetch = std::find_if(lowered.begin(), lowered.end(), [](u64 word) {
            return (word >> 56) == 0xdc;
        });
        REQUIRE(sample != lowered.end());
        REQUIRE(x != lowered.end());
        REQUIRE(y != lowered.end());
        REQUIRE(fetch != lowered.end());
        CHECK(sample < x);
        CHECK(sample < y);
        CHECK(((*sample >> 20) & 0xff) == source);
        CHECK(((*fetch >> 20) & 0xff) == 253);
        CHECK(((*fetch >> 50) & 1) == 1);
    }
}

TEST_CASE("Ampere TLD.MS sample operands participate in register-pressure shadowing",
          "[video_core][ampere]") {
    Instruction multisample{{0x04007b66, 0x32000006, 0x049e4f02, 0x001fc200}};
    multisample.words[1] = (multisample.words[1] & ~0xffU) | 250U;
    const auto pressure = Shader::Ampere::PrepareRegisterPressure(multisample, 216);
    REQUIRE(pressure.valid);
    CHECK(pressure.instruction.Bits(32, 8) < 216);
    CHECK(pressure.prefix.size() > 0);
}

TEST_CASE("Ampere bindless TLD4.B preserves AOFFI offsets",
          "[video_core][ampere]") {
    const auto lowered = LowerToMaxwell(
        std::array{Instruction{{0x02087f63, 0x30000409, 0x081e15ff, 0x000f4200}}, Exit});
    INFO(lowered.error);
    REQUIRE(lowered);
    const auto gather = std::find_if(lowered.instructions.begin(), lowered.instructions.end(),
                                     [](u64 word) { return (word >> 56) == 0xde; });
    REQUIRE(gather != lowered.instructions.end());
    CHECK(((*gather >> 36) & 3) == 1);
    const auto offset_copy = std::find_if(lowered.instructions.begin(), gather, [](u64 word) {
        return (word >> 48) == 0x5c98 && (word & 0xff) == 247 && ((word >> 20) & 0xff) == 9;
    });
    CHECK(offset_copy != gather);
}
