/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "render pass actions" — AGFX_LOAD_OPERATION_DONT_CARE with AGFX_STORE_OPERATION_STORE.
//
// DONT_CARE says the attachment's contents at the start of the pass are undefined, which bounds what
// this test can honestly assert: nothing at all about any pixel the draw does not write. So the draw
// here is the full-coverage triangle, and the assertion is that *every* pixel is the draw's flat
// color — i.e. that a pass which declares its contents undefined and then writes all of them stores
// exactly what was written, with no undefined pixels surviving into the result.
//
// This is the weakest of the three by nature, and deliberately so: the alternative — seeding the
// attachment and asserting the seed is gone — would be asserting a behavior the API explicitly does
// not promise, and would fail on any backend that legitimately chose to preserve the contents.
//
// Store DONT_CARE has no test in this family at all: it leaves the attachment undefined *after* the
// pass, so there is nothing to read back, and D3D12 does not support it regardless.

#include "pass_actions_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "render_pass_action_dont_care.png";

    PassActionState State()
    {
        PassActionState state;
        state.loadOp = AGFX_LOAD_OPERATION_DONT_CARE;
        state.fullscreen = true; // every pixel must be written for any of them to be assertable
        return state;
    }
} // namespace

AGFX_TEST_TEXTURE(RenderPassActionDontCare, C, kPassActionWidth, kPassActionHeight)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RenderPassAction(TestApi::C, State(), image, error), error.c_str());
    AGFX_EXPECT_MSG(RegionEquals(image, 0, 0, kPassActionWidth, kPassActionHeight, kPassActionDrawRgba8),
                    "a fully-covered DONT_CARE pass did not store the drawn color everywhere");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(RenderPassActionDontCare, Cpp, kPassActionWidth, kPassActionHeight)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RenderPassAction(TestApi::Cpp, State(), image, error), error.c_str());
    AGFX_EXPECT_MSG(RegionEquals(image, 0, 0, kPassActionWidth, kPassActionHeight, kPassActionDrawRgba8),
                    "a fully-covered DONT_CARE pass did not store the drawn color everywhere");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(RenderPassActionDontCare, Ez, kPassActionWidth, kPassActionHeight)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RenderPassAction(TestApi::Ez, State(), image, error), error.c_str());
    AGFX_EXPECT_MSG(RegionEquals(image, 0, 0, kPassActionWidth, kPassActionHeight, kPassActionDrawRgba8),
                    "a fully-covered DONT_CARE pass did not store the drawn color everywhere");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
