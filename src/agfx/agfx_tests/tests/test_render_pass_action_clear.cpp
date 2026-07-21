/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "render pass actions" — AGFX_LOAD_OPERATION_CLEAR with AGFX_STORE_OPERATION_STORE.
//
// The mirror image of the LOAD test, run against the same seeded attachment and the same draw. Here
// the seed must be *gone*: the corner has to be the clear color exactly, not the gradient. Running
// both against an identically seeded target is the point — between them, the only way to pass both
// is to actually honor the load op rather than to always clear or always preserve.

#include "pass_actions_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "render_pass_action_clear.png";

    PassActionState State()
    {
        PassActionState state;
        state.loadOp = AGFX_LOAD_OPERATION_CLEAR;
        return state;
    }
} // namespace

AGFX_TEST_TEXTURE(RenderPassActionClear, C, kPassActionWidth, kPassActionHeight)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RenderPassAction(TestApi::C, State(), image, error), error.c_str());
    AGFX_EXPECT_MSG(RegionEquals(image, 0, 0, kPassActionCornerSize, kPassActionCornerSize,
                                 kPassActionClearRgba8),
                    "loadOp CLEAR did not replace the attachment's previous contents with the clear color");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(RenderPassActionClear, Cpp, kPassActionWidth, kPassActionHeight)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RenderPassAction(TestApi::Cpp, State(), image, error), error.c_str());
    AGFX_EXPECT_MSG(RegionEquals(image, 0, 0, kPassActionCornerSize, kPassActionCornerSize,
                                 kPassActionClearRgba8),
                    "loadOp CLEAR did not replace the attachment's previous contents with the clear color");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(RenderPassActionClear, Ez, kPassActionWidth, kPassActionHeight)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RenderPassAction(TestApi::Ez, State(), image, error), error.c_str());
    AGFX_EXPECT_MSG(RegionEquals(image, 0, 0, kPassActionCornerSize, kPassActionCornerSize,
                                 kPassActionClearRgba8),
                    "loadOp CLEAR did not replace the attachment's previous contents with the clear color");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
