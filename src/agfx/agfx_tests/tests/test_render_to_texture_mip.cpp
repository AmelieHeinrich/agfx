/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "render to texture mip".
//
// Draws a centered triangle into mip 1 of a 128x128 two-mip texture, over that mip's seeded contents
// (loadOp LOAD), and checks mip 0 is untouched. Beyond the subresource addressing the face/slice
// tests cover, this one pins the render *area*: the pass and viewport are 64x64, not 128x128, so a
// backend that took the pass dimensions from the base mip would either overrun the smaller
// subresource or clip the triangle. The triangle is in NDC, so it scales with the mip rather than
// occupying a corner of it — a clipped draw is obvious in the golden.
//
// 128 base with mip 1 at 64 keeps both subresources' row pitches 256-byte aligned, which the
// buffer-to-texture copies behind the seed and readback require on D3D12.

#include "subresource_common.h"

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kBaseSize = 128;
    constexpr uint32_t kTargetMip = 1;
    constexpr uint32_t kMipSize = kBaseSize >> kTargetMip;

    constexpr const char* kGolden = "render_to_texture_mip.png";

    SubresourceCase Case()
    {
        SubresourceCase testCase;
        testCase.type = AGFX_TEXTURE_TYPE_2D;
        testCase.baseSize = kBaseSize;
        testCase.layerCount = 1;
        testCase.mipLevels = 2;
        testCase.targetMip = kTargetMip;
        testCase.loadOp = AGFX_LOAD_OPERATION_LOAD;
        testCase.drawTriangle = true;
        testCase.name = "test render to mip";
        return testCase;
    }
} // namespace

#define AGFX_RENDER_TO_MIP_CASE(api)                                                               \
    AGFX_TEST_TEXTURE(RenderToTextureMip, api, kMipSize, kMipSize)                                 \
    {                                                                                              \
        Image image;                                                                               \
        std::string error;                                                                         \
        AGFX_EXPECT_MSG(RunSubresourcePass(TestApi::api, Case(), image, error), error.c_str());    \
        ExpectImageMatchesGolden(ctx, kGolden, image);                                              \
    }

AGFX_RENDER_TO_MIP_CASE(C)
AGFX_RENDER_TO_MIP_CASE(Cpp)
AGFX_RENDER_TO_MIP_CASE(Ez)
