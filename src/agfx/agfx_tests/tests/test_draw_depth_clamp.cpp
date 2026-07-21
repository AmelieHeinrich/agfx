/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw depth clamp".
//
// A pair of tests -- clamp off and clamp on -- over identical geometry: three columns at depths that
// straddle the view volume, one before the near plane (z = -0.5), one inside it (z = 0.5), and one
// beyond the far plane (z = 1.5).
//
//   depthClampEnable = false -> only the middle column survives; the other two are clipped away.
//   depthClampEnable = true  -> all three survive, the out-of-range pair clamped to the depth range.
//
// The two goldens are the assertion, and neither can be reached by accident: a backend that ignores
// depthClampEnable renders the same image for both tests and so fails exactly one of them, which
// pins down not just that the flag does something but which way round it goes. Testing clamp on its
// own against a single golden could not distinguish "clamping works" from "nothing was ever clipped".
//
// This is the inverse of D3D11's DepthClipEnable, which is why the disabled case is the one that
// *drops* geometry -- see the note on agfxRenderPipelineCreateInfo::depthClampEnable.
//
// The depth test is left on with ALWAYS so the comparison can never remove a column: whatever
// survives here survived clipping, not a depth comparison. The out-of-range columns would otherwise
// be indistinguishable from ones that simply failed the test against the cleared depth.

#include "depth_common.h"

namespace
{
    using namespace agfxtest;

    DepthState State(bool depthClampEnable)
    {
        DepthState state;
        state.clearDepth = 1.0f;

        DepthDraw draw;
        draw.depthClampEnable = depthClampEnable;
        draw.depthTestEnable = true;
        draw.depthCompareOp = AGFX_COMPARISON_FUNCTION_ALWAYS; // clipping is the only thing filtering
        draw.depthWriteEnable = false;
        draw.constants.depths[0] = -0.5f; // before the near plane
        draw.constants.depths[1] = 0.5f;  // inside the view volume
        draw.constants.depths[2] = 1.5f;  // beyond the far plane

        state.draws = {draw};
        return state;
    }

    void RunDepthClampTest(TestContext& ctx, TestApi api, bool depthClampEnable, const char* golden)
    {
        Image image;
        AGFX_EXPECT_MSG(RenderDepth(api, State(depthClampEnable), image), "depth render failed");
        ExpectImageMatchesGolden(ctx, golden, image);
    }
} // namespace

AGFX_TEST_TEXTURE(DrawDepthClampDisabled, C, kDepthWidth, kDepthHeight)
{
    RunDepthClampTest(ctx, TestApi::C, false, "draw_depth_clamp_disabled.png");
}

AGFX_TEST_TEXTURE(DrawDepthClampDisabled, Cpp, kDepthWidth, kDepthHeight)
{
    RunDepthClampTest(ctx, TestApi::Cpp, false, "draw_depth_clamp_disabled.png");
}

AGFX_TEST_TEXTURE(DrawDepthClampDisabled, Ez, kDepthWidth, kDepthHeight)
{
    RunDepthClampTest(ctx, TestApi::Ez, false, "draw_depth_clamp_disabled.png");
}

AGFX_TEST_TEXTURE(DrawDepthClampEnabled, C, kDepthWidth, kDepthHeight)
{
    RunDepthClampTest(ctx, TestApi::C, true, "draw_depth_clamp_enabled.png");
}

AGFX_TEST_TEXTURE(DrawDepthClampEnabled, Cpp, kDepthWidth, kDepthHeight)
{
    RunDepthClampTest(ctx, TestApi::Cpp, true, "draw_depth_clamp_enabled.png");
}

AGFX_TEST_TEXTURE(DrawDepthClampEnabled, Ez, kDepthWidth, kDepthHeight)
{
    RunDepthClampTest(ctx, TestApi::Ez, true, "draw_depth_clamp_enabled.png");
}
