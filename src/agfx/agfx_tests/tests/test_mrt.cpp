/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "MRT (render to 2 different color attachments then add their contents up in the output
// texture using compute)".
//
// One render pass writes a red x-ramp into SV_Target0 and a green y-ramp into SV_Target1, then a
// compute pass sums the two into a third texture. The sum is checked byte-exact against
// ExpectedMrtSum() rather than resting on the golden, so a backend that wrote the same output to
// both attachments, bound one twice, or swapped them fails on the analytic check.
//
// Deliberately different content per attachment: with identical content, every one of those failure
// modes would still produce a correct-looking sum.

#include "mrt_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "mrt.png";
} // namespace

#define AGFX_MRT_CASE(api)                                                                         \
    AGFX_TEST_TEXTURE(MRT, api, kMrtWidth, kMrtHeight)                                             \
    {                                                                                              \
        Image image;                                                                               \
        std::string error;                                                                         \
        AGFX_EXPECT_MSG(RenderMrt(TestApi::api, image, error), error.c_str());                     \
        AGFX_EXPECT_MSG(ImageEqualsRgba8(image, ExpectedMrtSum()),                                 \
                        "the summed image is not attachment 0's red ramp plus attachment 1's green"); \
        ExpectImageMatchesGolden(ctx, kGolden, image);                                              \
    }

AGFX_MRT_CASE(C)
AGFX_MRT_CASE(Cpp)
AGFX_MRT_CASE(Ez)
