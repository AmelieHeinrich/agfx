/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

// Shared scaffolding for the alpha-blending tests: one per agfxBlendOperation, one per
// agfxBlendFactor, and the canonical src-alpha over one-minus-src-alpha case.
//
// Structurally this is depth_common.h's sibling -- a list of draws sharing one render pass, each
// with its own pipeline state and push constants -- for the same reason: a blend result is only
// observable against a destination that an earlier draw put there. It is deliberately *not* the same
// file. These tests bind no depth attachment at all, and folding both families into one BlendDepthState
// would mean every depth test carried blend fields it never sets and vice versa.
//
// The scene is fixed (see data/shaders/tests/blend.hlsl); what a test varies is only the blend state
// on the second draw. The source color and alpha live here rather than in each test so that every
// golden in the family is comparable against every other.

#include "../test_gpu.h"

#include <vector>

namespace agfxtest
{
    constexpr uint32_t kBlendWidth = 128;
    constexpr uint32_t kBlendHeight = 128;
    constexpr agfxTextureFormat kBlendFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;

    /// @brief Columns in the scene. Matches AGFX_TEST_BLEND_COLUMNS in data/shaders/tests/blend.hlsl.
    constexpr uint32_t kBlendColumns = 3;

    /// @brief Mirrors BlendPushConstants in data/shaders/tests/blend.hlsl. Field order and padding
    /// must match exactly -- this is memcpy'd into the push-constant block.
    struct BlendConstants
    {
        float color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float columnScale[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float columnAlpha[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    };
    static_assert(sizeof(BlendConstants) == 48, "BlendConstants must match the HLSL layout");

    /// @brief One draw within the shared render pass, with the blend state it is drawn under.
    struct BlendDraw
    {
        /// @brief Draw the full-screen source quad instead of the three destination columns.
        bool fullscreen = false;

        bool blendEnable = false;
        agfxBlendFactor srcBlend = AGFX_BLEND_FACTOR_ONE;
        agfxBlendFactor dstBlend = AGFX_BLEND_FACTOR_ZERO;
        agfxBlendOperation blendOp = AGFX_BLEND_OPERATION_ADD;

        BlendConstants constants{};
    };

    struct BlendState
    {
        /// @brief Drawn in order into one render pass. The first draw establishes the destination
        /// the later one blends against.
        std::vector<BlendDraw> draws;
    };

    /// @brief The destination draw every blend test starts from: three dim columns of differing
    /// color *and* differing alpha, over a transparent-black clear.
    ///
    /// Both kinds of variation are load-bearing. Differing colors separate the factors that read the
    /// destination color; differing alphas separate DST_ALPHA and ONE_MINUS_DST_ALPHA, which against
    /// a uniformly opaque destination would be indistinguishable from ONE and ZERO. The columns are
    /// dim so that additive results stay inside the UNORM range: a clamped channel is a channel on
    /// which two different factors can produce the same golden.
    BlendDraw DestinationDraw();

    /// @brief The source draw, with the blend state under test.
    ///
    /// The source alpha is 0.25 rather than 0.5 on purpose: at 0.5, SRC_ALPHA and
    /// ONE_MINUS_SRC_ALPHA are the same multiplier and would produce byte-identical goldens.
    BlendDraw SourceDraw(agfxBlendFactor srcBlend, agfxBlendFactor dstBlend, agfxBlendOperation blendOp);

    /// @brief Convenience: the two-draw state for one blend configuration.
    BlendState State(agfxBlendFactor srcBlend, agfxBlendFactor dstBlend, agfxBlendOperation blendOp);

    /// @brief Renders one BlendState through the given API flavor into `outImage`.
    bool RenderBlend(TestApi api, const BlendState& state, Image& outImage);
} // namespace agfxtest
