/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw alpha blending (one for each blend test op/factor)" -- the operation half.
//
// One test per agfxBlendOperation, with both factors pinned to ONE so the arithmetic is the only
// variable:
//
//   ADD              -> src + dst
//   SUBTRACT         -> src - dst
//   REVERSE_SUBTRACT -> dst - src
//   MIN              -> min(src, dst)
//   MAX              -> max(src, dst)
//
// SUBTRACT and REVERSE_SUBTRACT are the pair worth having: they differ only in operand order, and a
// backend that mapped one onto the other would produce a perfectly plausible image for each in
// isolation. Against a destination that is brighter than the source in one channel and dimmer in
// another, they clamp in opposite places and cannot be confused.
//
// MIN and MAX ignore the blend factors entirely on both backends, which is why the factors are
// pinned to ONE rather than to something the operation might scale -- a factor that mattered for
// three of the five operations and not the other two would make these goldens harder to reason
// about, not more thorough.
//
// See test_draw_blend_factor.cpp for the factor half.

#include "blend_common.h"

namespace
{
    using namespace agfxtest;

    void RunBlendOpTest(TestContext& ctx, TestApi api, agfxBlendOperation blendOp, const char* golden)
    {
        Image image;
        const BlendState state = State(AGFX_BLEND_FACTOR_ONE, AGFX_BLEND_FACTOR_ONE, blendOp);
        AGFX_EXPECT_MSG(RenderBlend(api, state, image), "blend render failed");
        ExpectImageMatchesGolden(ctx, golden, image);
    }
} // namespace

#define AGFX_BLEND_OP_CASE(TestName, op, golden)                                                   \
    AGFX_TEST_TEXTURE(TestName, C, kBlendWidth, kBlendHeight)                                      \
    {                                                                                              \
        RunBlendOpTest(ctx, TestApi::C, op, golden);                                               \
    }                                                                                              \
    AGFX_TEST_TEXTURE(TestName, Cpp, kBlendWidth, kBlendHeight)                                    \
    {                                                                                              \
        RunBlendOpTest(ctx, TestApi::Cpp, op, golden);                                             \
    }                                                                                              \
    AGFX_TEST_TEXTURE(TestName, Ez, kBlendWidth, kBlendHeight)                                     \
    {                                                                                              \
        RunBlendOpTest(ctx, TestApi::Ez, op, golden);                                              \
    }

AGFX_BLEND_OP_CASE(DrawBlendOpAdd, AGFX_BLEND_OPERATION_ADD, "draw_blend_op_add.png")
AGFX_BLEND_OP_CASE(DrawBlendOpSubtract, AGFX_BLEND_OPERATION_SUBTRACT, "draw_blend_op_subtract.png")
AGFX_BLEND_OP_CASE(DrawBlendOpReverseSubtract, AGFX_BLEND_OPERATION_REVERSE_SUBTRACT,
                   "draw_blend_op_reverse_subtract.png")
AGFX_BLEND_OP_CASE(DrawBlendOpMin, AGFX_BLEND_OPERATION_MIN, "draw_blend_op_min.png")
AGFX_BLEND_OP_CASE(DrawBlendOpMax, AGFX_BLEND_OPERATION_MAX, "draw_blend_op_max.png")
