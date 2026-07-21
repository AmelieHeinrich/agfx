/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "render pass clear slice".
//
// The array-texture sibling of the clear face test: layer 2 of a four-layer 2D array is cleared
// through a render pass, and the other three layers are checked byte-exact against their seeds. The
// cube case and this one go down different creation paths on both backends (a cube's six faces are
// array layers with extra rules attached), so passing one says nothing about the other.

#include "subresource_common.h"

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kSize = 64; // 64 * 4 bytes = 256: D3D12's row-pitch alignment.
    constexpr const char* kGolden = "render_pass_clear_slice.png";

    SubresourceCase Case()
    {
        SubresourceCase testCase;
        testCase.type = AGFX_TEXTURE_TYPE_2D_ARRAY;
        testCase.baseSize = kSize;
        testCase.layerCount = 4;
        testCase.targetLayer = 2;
        testCase.name = "test clear slice";
        return testCase;
    }
} // namespace

AGFX_TEST_TEXTURE(RenderPassClearSlice, C, kSize, kSize)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RunSubresourcePass(TestApi::C, Case(), image, error), error.c_str());
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(RenderPassClearSlice, Cpp, kSize, kSize)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RunSubresourcePass(TestApi::Cpp, Case(), image, error), error.c_str());
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(RenderPassClearSlice, Ez, kSize, kSize)
{
    Image image;
    std::string error;
    AGFX_EXPECT_MSG(RunSubresourcePass(TestApi::Ez, Case(), image, error), error.c_str());
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
