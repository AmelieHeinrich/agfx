/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

// Shared scaffolding for the MRT test: one render pass writing two color attachments, then a compute
// pass summing them into a third texture.
//
// Only one test uses this today, but it follows the same split as the other families so the three
// API flavors are written once and are guaranteed to drive identical work — which is the property
// the suite is really asserting when it runs C, C++ and ez against a single golden.

#include "../test_gpu.h"

#include <string>

namespace agfxtest
{
    constexpr uint32_t kMrtWidth = 128;
    constexpr uint32_t kMrtHeight = 128;
    constexpr agfxTextureFormat kMrtFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;

    /// @brief The summed result the compute pass must produce: red ramps with x from attachment 0,
    /// green ramps with y from attachment 1. Stated analytically so the golden is not the oracle —
    /// under --update-goldens a broken backend's output would otherwise become the expectation.
    std::vector<uint8_t> ExpectedMrtSum();

    /// @brief Renders both attachments, sums them, and writes the sum into `outImage`.
    /// @param outError Set to a description when the run failed; empty on success.
    bool RenderMrt(TestApi api, Image& outImage, std::string& outError);
} // namespace agfxtest
