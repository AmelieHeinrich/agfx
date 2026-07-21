/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

// Shared scaffolding for the sampler-state tests (address mode, filter).
//
// Each of those seeds one 64x64 source texture with texture_ops.hlsl:main_write_cs, then samples it
// into a 64x64 destination through sampling.hlsl:main_sample_2d_cs. The only things that vary are
// the agfxSamplerCreateInfo and the uv scale/offset handed to the sampling shader — everything else
// (device, views, pipelines, barriers, readback) is identical across every variant and all three API
// flavors, which is ~200 lines that would otherwise be copied per file. Same reasoning as
// raster_common.h.
//
// Routing all three flavors through one entry point also makes "C, C++ and ez produce the same
// image for the same sampler" a property of the code rather than something kept in sync by hand.

#include "../test_gpu.h"

namespace agfxtest
{
    constexpr uint32_t kSamplerWidth = 64;
    constexpr uint32_t kSamplerHeight = 64;
    constexpr agfxTextureFormat kSamplerFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;

    /// @brief One sampler test's inputs: the sampler under test plus the coordinate transform that
    /// exercises it. Defaults describe an identity sample — the destination reproduces the source.
    struct SampleState
    {
        agfxSamplerCreateInfo sampler{};
        float uvScale[2] = {1.0f, 1.0f};
        float uvOffset[2] = {0.0f, 0.0f};
    };

    /// @brief A baseline agfxSamplerCreateInfo: linear, clamped, no mips, no comparison. Callers
    /// override only the field under test so an unrelated default cannot drift between tests.
    agfxSamplerCreateInfo DefaultSamplerInfo();

    /// @brief Seeds the source, samples it per `state`, and writes the result into `outImage`.
    /// Returns false if device creation, shader compilation or readback failed; the caller should
    /// AGFX_EXPECT on it before comparing against a golden.
    bool RenderSample(TestApi api, const SampleState& state, Image& outImage);
} // namespace agfxtest
