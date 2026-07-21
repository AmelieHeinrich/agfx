/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "depth write disabled".
//
// A depth write cannot be observed directly -- the tests read back color, not depth -- so it is
// observed through a later draw that depends on it. Two draws sharing one depth buffer:
//
//   1. Three columns at z = 0.5, depth test ALWAYS, depth writes under test (on or off).
//   2. A full-screen gray quad at z = 0.75, depth test LESS, writes off.
//
// If draw 1's writes landed, the depth buffer reads 0.5 inside the columns and 1.0 outside, so
// draw 2 fails LESS over the columns and passes everywhere else: colored columns on gray.
// If they did not, the buffer is still 1.0 everywhere and draw 2 passes everywhere: uniform gray,
// the columns painted over.
//
//   depthWriteEnable = true  -> red/green/blue columns on a gray field
//   depthWriteEnable = false -> gray, nothing else
//
// Both cases are tested rather than just the disabled one the TODO names. The disabled golden is
// "the columns are gone", which on its own is equally consistent with the first draw never having
// happened at all -- a shader compile that quietly produced nothing, or a pass that never ran, would
// produce exactly that image. The enabled twin uses the identical setup and differs in one pipeline
// flag, so it rules that out: the columns demonstrably render, and the flag is what removes them.
//
// The occluder is gray rather than another primary so it cannot be confused with a column in either
// golden, and it covers the full screen so a partial depth write shows up as a ragged edge rather
// than as a plausible-looking image.

#include "depth_common.h"

namespace
{
    using namespace agfxtest;

    constexpr float kColumnDepth = 0.5f;
    constexpr float kOccluderDepth = 0.75f;

    DepthState State(bool depthWriteEnable)
    {
        DepthState state;
        state.clearDepth = 1.0f;

        DepthDraw columns;
        columns.depthTestEnable = true;
        columns.depthCompareOp = AGFX_COMPARISON_FUNCTION_ALWAYS; // the test is about writes, not tests
        columns.depthWriteEnable = depthWriteEnable;
        columns.constants.depths[0] = kColumnDepth;
        columns.constants.depths[1] = kColumnDepth;
        columns.constants.depths[2] = kColumnDepth;

        DepthDraw occluder;
        occluder.fullscreen = true;
        occluder.depthTestEnable = true;
        occluder.depthCompareOp = AGFX_COMPARISON_FUNCTION_LESS; // reads what the columns wrote
        occluder.depthWriteEnable = false;
        occluder.constants.depths[0] = kOccluderDepth;
        occluder.constants.tint[0] = 0.5f;
        occluder.constants.tint[1] = 0.5f;
        occluder.constants.tint[2] = 0.5f;

        state.draws = {columns, occluder};
        return state;
    }

    void RunDepthWriteTest(TestContext& ctx, TestApi api, bool depthWriteEnable, const char* golden)
    {
        Image image;
        AGFX_EXPECT_MSG(RenderDepth(api, State(depthWriteEnable), image), "depth render failed");
        ExpectImageMatchesGolden(ctx, golden, image);
    }
} // namespace

AGFX_TEST_TEXTURE(DepthWriteDisabled, C, kDepthWidth, kDepthHeight)
{
    RunDepthWriteTest(ctx, TestApi::C, false, "depth_write_disabled.png");
}

AGFX_TEST_TEXTURE(DepthWriteDisabled, Cpp, kDepthWidth, kDepthHeight)
{
    RunDepthWriteTest(ctx, TestApi::Cpp, false, "depth_write_disabled.png");
}

AGFX_TEST_TEXTURE(DepthWriteDisabled, Ez, kDepthWidth, kDepthHeight)
{
    RunDepthWriteTest(ctx, TestApi::Ez, false, "depth_write_disabled.png");
}

AGFX_TEST_TEXTURE(DepthWriteEnabled, C, kDepthWidth, kDepthHeight)
{
    RunDepthWriteTest(ctx, TestApi::C, true, "depth_write_enabled.png");
}

AGFX_TEST_TEXTURE(DepthWriteEnabled, Cpp, kDepthWidth, kDepthHeight)
{
    RunDepthWriteTest(ctx, TestApi::Cpp, true, "depth_write_enabled.png");
}

AGFX_TEST_TEXTURE(DepthWriteEnabled, Ez, kDepthWidth, kDepthHeight)
{
    RunDepthWriteTest(ctx, TestApi::Ez, true, "depth_write_enabled.png");
}
