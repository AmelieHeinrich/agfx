/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw depth test (one for each depth test func)".
//
// One test per agfxComparisonFunction. Each renders three column-shaped probes at depths that
// straddle a known depth floor -- nearer than it, exactly equal to it, and farther than it -- and
// the set of columns that survive names the comparison function uniquely:
//
//   NEVER         -> nothing
//   LESS          -> red                 (0.25 only)
//   EQUAL         -> green               (0.50 only)
//   LESS_EQUAL    -> red + green
//   GREATER       -> blue                (0.75 only)
//   NOT_EQUAL     -> red + blue
//   GREATER_EQUAL -> green + blue
//   ALWAYS        -> red + green + blue
//
// All eight goldens are distinct, and each differs from its neighbours by a whole column rather than
// by an edge, so an off-by-one in the comparison mapping (LESS where LESS_EQUAL was meant, or an
// inverted sense) fails loudly. A backend that ignored the depth test entirely would pass ALWAYS and
// fail the other seven, rather than passing everything.
//
// Two details make this work, both in depth_common:
//
//  - The depth floor is laid down by a prior full-screen draw at z = 0.5 with writes enabled, tinted
//    black so it is invisible against the clear color. Clearing to 0.5 directly would do as well now
//    that clearDepth is honored, but writing it proves the floor is really in the depth buffer.
//  - The probe draw has depth *writes* disabled. Without that the columns would be tested against
//    each other's results instead of against the floor, and the outcome would depend on draw order
//    within a single draw call -- which is not what any of this is trying to measure.

#include "depth_common.h"

namespace
{
    using namespace agfxtest;

    /// @brief The depth floor every probe is compared against, and the probe depths around it.
    constexpr float kFloorDepth = 0.5f;
    constexpr float kNearer = 0.25f;
    constexpr float kFarther = 0.75f;

    DepthState State(agfxComparisonFunction compareOp)
    {
        DepthState state;
        // Cleared to the far plane, then overwritten with the floor. The probes never test against
        // this value, only against what the floor draw wrote.
        state.clearDepth = 1.0f;

        DepthDraw floorDraw;
        floorDraw.fullscreen = true;
        floorDraw.depthTestEnable = true;
        floorDraw.depthCompareOp = AGFX_COMPARISON_FUNCTION_ALWAYS; // unconditionally lay it down
        floorDraw.depthWriteEnable = true;
        floorDraw.constants.depths[0] = kFloorDepth;
        // Black: the floor is a depth-buffer operation, and must not show up in the color golden.
        floorDraw.constants.tint[0] = 0.0f;
        floorDraw.constants.tint[1] = 0.0f;
        floorDraw.constants.tint[2] = 0.0f;

        DepthDraw probes;
        probes.depthTestEnable = true;
        probes.depthCompareOp = compareOp;
        probes.depthWriteEnable = false; // keeps the three probes independent of each other
        probes.constants.depths[0] = kNearer;
        probes.constants.depths[1] = kFloorDepth; // exactly equal, for EQUAL / NOT_EQUAL
        probes.constants.depths[2] = kFarther;

        state.draws = {floorDraw, probes};
        return state;
    }

    void RunDepthTest(TestContext& ctx, TestApi api, agfxComparisonFunction compareOp,
                      const char* golden)
    {
        Image image;
        AGFX_EXPECT_MSG(RenderDepth(api, State(compareOp), image), "depth render failed");
        ExpectImageMatchesGolden(ctx, golden, image);
    }

    /// @brief Probes the *cleared* depth value directly, with no floor draw in front of it.
    ///
    /// Everything above clears to 1.0 and establishes its floor by drawing, so none of it would
    /// notice if agfxRenderPassAttachment::clearDepth were ignored -- which is exactly what both
    /// backends used to do, hardcoding 1.0f. This case clears to 0.5 and compares LESS, so only the
    /// nearest column survives; a backend that ignores clearDepth clears to 1.0 instead and all
    /// three columns pass, which is a different image rather than a marginally different one.
    DepthState ClearValueState()
    {
        DepthState state;
        state.clearDepth = kFloorDepth;

        DepthDraw probes;
        probes.depthTestEnable = true;
        probes.depthCompareOp = AGFX_COMPARISON_FUNCTION_LESS;
        probes.depthWriteEnable = false;
        probes.constants.depths[0] = kNearer;
        probes.constants.depths[1] = kFloorDepth;
        probes.constants.depths[2] = kFarther;

        state.draws = {probes};
        return state;
    }
} // namespace

#define AGFX_DEPTH_TEST_CASE(TestName, compareOp, golden)                                          \
    AGFX_TEST_TEXTURE(TestName, C, kDepthWidth, kDepthHeight)                                      \
    {                                                                                              \
        RunDepthTest(ctx, TestApi::C, compareOp, golden);                                          \
    }                                                                                              \
    AGFX_TEST_TEXTURE(TestName, Cpp, kDepthWidth, kDepthHeight)                                    \
    {                                                                                              \
        RunDepthTest(ctx, TestApi::Cpp, compareOp, golden);                                        \
    }                                                                                              \
    AGFX_TEST_TEXTURE(TestName, Ez, kDepthWidth, kDepthHeight)                                     \
    {                                                                                              \
        RunDepthTest(ctx, TestApi::Ez, compareOp, golden);                                         \
    }

AGFX_DEPTH_TEST_CASE(DrawDepthTestNever, AGFX_COMPARISON_FUNCTION_NEVER, "draw_depth_test_never.png")
AGFX_DEPTH_TEST_CASE(DrawDepthTestLess, AGFX_COMPARISON_FUNCTION_LESS, "draw_depth_test_less.png")
AGFX_DEPTH_TEST_CASE(DrawDepthTestEqual, AGFX_COMPARISON_FUNCTION_EQUAL, "draw_depth_test_equal.png")
AGFX_DEPTH_TEST_CASE(DrawDepthTestLessEqual, AGFX_COMPARISON_FUNCTION_LESS_EQUAL,
                     "draw_depth_test_less_equal.png")
AGFX_DEPTH_TEST_CASE(DrawDepthTestGreater, AGFX_COMPARISON_FUNCTION_GREATER,
                     "draw_depth_test_greater.png")
AGFX_DEPTH_TEST_CASE(DrawDepthTestNotEqual, AGFX_COMPARISON_FUNCTION_NOT_EQUAL,
                     "draw_depth_test_not_equal.png")
AGFX_DEPTH_TEST_CASE(DrawDepthTestGreaterEqual, AGFX_COMPARISON_FUNCTION_GREATER_EQUAL,
                     "draw_depth_test_greater_equal.png")
AGFX_DEPTH_TEST_CASE(DrawDepthTestAlways, AGFX_COMPARISON_FUNCTION_ALWAYS,
                     "draw_depth_test_always.png")

AGFX_TEST_TEXTURE(DrawDepthClearValue, C, kDepthWidth, kDepthHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderDepth(TestApi::C, ClearValueState(), image), "depth render failed");
    ExpectImageMatchesGolden(ctx, "draw_depth_clear_value.png", image);
}

AGFX_TEST_TEXTURE(DrawDepthClearValue, Cpp, kDepthWidth, kDepthHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderDepth(TestApi::Cpp, ClearValueState(), image), "depth render failed");
    ExpectImageMatchesGolden(ctx, "draw_depth_clear_value.png", image);
}

AGFX_TEST_TEXTURE(DrawDepthClearValue, Ez, kDepthWidth, kDepthHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderDepth(TestApi::Ez, ClearValueState(), image), "depth render failed");
    ExpectImageMatchesGolden(ctx, "draw_depth_clear_value.png", image);
}
