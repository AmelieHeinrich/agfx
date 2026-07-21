/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "render pass actions" — AGFX_LOAD_OPERATION_LOAD with AGFX_STORE_OPERATION_STORE.
//
// The attachment is seeded with a gradient, then a pass with loadOp LOAD draws a centered triangle
// over it. The seeded gradient must still be there around the triangle: LOAD is the one op whose
// entire contract is that the previous contents survive into the pass.
//
// The corner check is what gives this teeth. A backend that treated LOAD as CLEAR would produce a
// perfectly plausible image — triangle on a flat background — and a freshly captured golden would
// happily bless it. Comparing the corner against the seed's *own* per-pixel values is what makes
// "the old contents came back, unshifted" the thing being asserted.

#include "pass_actions_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "render_pass_action_load.png";

    PassActionState State()
    {
        PassActionState state;
        state.loadOp = AGFX_LOAD_OPERATION_LOAD;
        return state;
    }
} // namespace

AGFX_TEST_TEXTURE(RenderPassActionLoad, C, kPassActionWidth, kPassActionHeight)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RenderPassAction(TestApi::C, State(), image, error), error.c_str());
    AGFX_EXPECT_MSG(RegionMatchesSeed(image, 0, 0, kPassActionCornerSize, kPassActionCornerSize),
                    "loadOp LOAD did not preserve the attachment's previous contents");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(RenderPassActionLoad, Cpp, kPassActionWidth, kPassActionHeight)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RenderPassAction(TestApi::Cpp, State(), image, error), error.c_str());
    AGFX_EXPECT_MSG(RegionMatchesSeed(image, 0, 0, kPassActionCornerSize, kPassActionCornerSize),
                    "loadOp LOAD did not preserve the attachment's previous contents");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(RenderPassActionLoad, Ez, kPassActionWidth, kPassActionHeight)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RenderPassAction(TestApi::Ez, State(), image, error), error.c_str());
    AGFX_EXPECT_MSG(RegionMatchesSeed(image, 0, 0, kPassActionCornerSize, kPassActionCornerSize),
                    "loadOp LOAD did not preserve the attachment's previous contents");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
