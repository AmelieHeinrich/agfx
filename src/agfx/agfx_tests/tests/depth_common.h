/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

// Shared scaffolding for the depth-state tests: depth clamp, the eight depth comparison functions,
// and depth write disabled.
//
// Same reasoning as raster_common.h, plus one thing those tests never needed: a depth attachment.
// These are the first tests in the suite to bind one, so the depth texture, its render target, the
// DEPTH_WRITE barrier and the pass's depth attachment all live here rather than being reinvented
// three times per test family.
//
// The other reason this is centralized is that a depth test is rarely one draw. Establishing a known
// depth buffer to test *against* takes a prior draw that writes it, and proving a depth write landed
// takes a later draw that reads it -- so DepthState is a *list* of draws sharing one render pass and
// one depth buffer, not a single pipeline state. Each draw carries its own depth state and push
// constants, which is what lets one shader and one host path serve all three families.

#include "../test_gpu.h"

#include <vector>

namespace agfxtest
{
    constexpr uint32_t kDepthWidth = 128;
    constexpr uint32_t kDepthHeight = 128;
    constexpr agfxTextureFormat kDepthColorFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr agfxTextureFormat kDepthFormat = AGFX_TEXTURE_FORMAT_DEPTH32F;

    /// @brief Columns in the scene. Matches AGFX_TEST_DEPTH_COLUMNS in data/shaders/tests/depth.hlsl.
    constexpr uint32_t kDepthColumns = 3;

    /// @brief Mirrors DepthPushConstants in data/shaders/tests/depth.hlsl. Field order and padding
    /// must match exactly -- this is memcpy'd into the push-constant block.
    struct DepthConstants
    {
        /// @brief Per-column clip-space z; only the first kDepthColumns entries are read. A float4
        /// rather than a float[3] because an HLSL cbuffer pads array elements to 16 bytes.
        float depths[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float tint[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    };
    static_assert(sizeof(DepthConstants) == 32, "DepthConstants must match the HLSL layout");

    /// @brief One draw within the shared render pass, with the depth state it is drawn under.
    struct DepthDraw
    {
        /// @brief Draw the full-screen quad (main_vs_fullscreen, depth from constants.depths[0])
        /// instead of the three columns. Used both to lay down a depth floor and to probe one.
        bool fullscreen = false;

        bool depthTestEnable = true;
        bool depthWriteEnable = false;
        bool depthClampEnable = false;
        agfxComparisonFunction depthCompareOp = AGFX_COMPARISON_FUNCTION_ALWAYS;

        DepthConstants constants{};
    };

    /// @brief What one depth test varies: the cleared depth, and the draws made against it.
    struct DepthState
    {
        /// @brief The depth attachment's clear value. Meaningful only because agfxRenderPassAttachment
        /// carries clearDepth -- both backends used to hardcode 1.0f, which made every comparison in
        /// the GREATER family untestable, since no fragment can be farther than the far plane.
        float clearDepth = 1.0f;

        /// @brief Drawn in order into one render pass, sharing one depth buffer. Order is
        /// significant: that is the entire mechanism behind the depth-write tests.
        std::vector<DepthDraw> draws;
    };

    /// @brief Renders one DepthState through the given API flavor into `outImage`.
    /// Returns false if device creation, shader compilation, pipeline creation or readback failed.
    bool RenderDepth(TestApi api, const DepthState& state, Image& outImage);
} // namespace agfxtest
