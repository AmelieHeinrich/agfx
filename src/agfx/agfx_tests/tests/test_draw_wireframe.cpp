/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw wireframe".
//
// Draws the two raster.hlsl triangles with AGFX_FILL_MODE_WIREFRAME instead of SOLID. The golden is
// two triangle outlines on black rather than two filled triangles, so a backend that silently
// ignores fillMode (Metal applies it as MTLTriangleFillModeLines on the encoder, D3D12 bakes it
// into the PSO rasterizer desc — two very different code paths to get wrong) fails loudly with a
// mostly-filled image rather than subtly.

#include "raster_common.h"

namespace
{
    using namespace agfxtest;

    /// @brief Six triangle edges cover ~512 pixels here. A floor rather than an exact count, for the
    /// same reason as the line test: mean FLIP error alone would not notice an empty image, but edge
    /// rasterization is allowed to differ by a pixel between backends.
    constexpr uint32_t kMinWireframePixels = 300;

    constexpr const char* kGolden = "draw_wireframe.png";

    RasterState State()
    {
        RasterState state;
        state.fillMode = AGFX_FILL_MODE_WIREFRAME;
        return state;
    }
} // namespace

AGFX_TEST_TEXTURE(DrawWireframe, C, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::C, State(), image), "wireframe render failed");
    AGFX_EXPECT(CountLitPixels(image) > kMinWireframePixels);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawWireframe, Cpp, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Cpp, State(), image), "wireframe render failed");
    AGFX_EXPECT(CountLitPixels(image) > kMinWireframePixels);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawWireframe, Ez, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Ez, State(), image), "wireframe render failed");
    AGFX_EXPECT(CountLitPixels(image) > kMinWireframePixels);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
