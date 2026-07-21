/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "render pass clear mip".
//
// Clears mip 1 of a 128x128 two-mip texture and checks mip 0 still holds its seed. Beyond the
// subresource addressing the face/slice tests cover, this one pins the render *area*: the pass and
// its viewport are 64x64, not 128x128, so a backend that took the pass dimensions from the base mip
// would either overrun the smaller subresource or clip and leave part of it unwritten.
//
// 128 base with mip 1 at 64 keeps both subresources' row pitches 256-byte aligned, which the
// buffer-to-texture copies behind the seed and readback require on D3D12.

#include "subresource_common.h"

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kBaseSize = 128;
    constexpr uint32_t kClearMip = 1;
    constexpr uint32_t kMipSize = kBaseSize >> kClearMip;

    constexpr const char* kGolden = "render_pass_clear_mip.png";

    SubresourceCase Case()
    {
        SubresourceCase testCase;
        testCase.type = AGFX_TEXTURE_TYPE_2D;
        testCase.baseSize = kBaseSize;
        testCase.layerCount = 1;
        testCase.mipLevels = 2;
        testCase.targetMip = kClearMip;
        testCase.name = "test clear mip";
        return testCase;
    }
} // namespace

AGFX_TEST_TEXTURE(RenderPassClearMip, C, kMipSize, kMipSize)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RunSubresourcePass(TestApi::C, Case(), image, error), error.c_str());
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(RenderPassClearMip, Cpp, kMipSize, kMipSize)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RunSubresourcePass(TestApi::Cpp, Case(), image, error), error.c_str());
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(RenderPassClearMip, Ez, kMipSize, kMipSize)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RunSubresourcePass(TestApi::Ez, Case(), image, error), error.c_str());
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
