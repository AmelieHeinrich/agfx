/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "render to texture slice".
//
// Draws a centered triangle into layer 2 of a four-layer 2D array, over that layer's seeded contents
// (loadOp LOAD), and checks the other three layers are untouched. The array-texture sibling of the
// render-to-face test; the two go down different creation paths on both backends, since a cube's
// faces are array layers with extra rules attached.

#include "subresource_common.h"

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kSize = 64; // 64 * 4 bytes = 256: D3D12's row-pitch alignment.
    constexpr const char* kGolden = "render_to_texture_slice.png";

    SubresourceCase Case()
    {
        SubresourceCase testCase;
        testCase.type = AGFX_TEXTURE_TYPE_2D_ARRAY;
        testCase.baseSize = kSize;
        testCase.layerCount = 4;
        testCase.targetLayer = 2;
        testCase.loadOp = AGFX_LOAD_OPERATION_LOAD;
        testCase.drawTriangle = true;
        testCase.name = "test render to slice";
        return testCase;
    }
} // namespace

#define AGFX_RENDER_TO_SLICE_CASE(api)                                                             \
    AGFX_TEST_TEXTURE(RenderToTextureSlice, api, kSize, kSize)                                     \
    {                                                                                              \
        Image image;                                                                               \
        std::string error;                                                                         \
        AGFX_EXPECT_MSG(RunSubresourcePass(TestApi::api, Case(), image, error), error.c_str());    \
        ExpectImageMatchesGolden(ctx, kGolden, image);                                              \
    }

AGFX_RENDER_TO_SLICE_CASE(C)
AGFX_RENDER_TO_SLICE_CASE(Cpp)
AGFX_RENDER_TO_SLICE_CASE(Ez)
