/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw points".
//
// Draws the six raster.hlsl vertices as AGFX_TOPOLOGY_POINTS: six isolated pixels at the triangle
// corners, each in that corner's color. This is the topology most likely to differ between
// backends — D3D12 rasterizes points at a fixed one-pixel size while Metal takes the size from the
// vertex shader's [[point_size]] output (which HLSL has no way to express, so it defaults) — which
// is exactly why it is worth pinning to a golden rather than assuming.

#include "raster_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "draw_points.png";

    /// @brief One pixel per vertex. Asserted exactly, because the golden's mean-error comparison
    /// cannot distinguish six lit pixels from none (see CountLitPixels). This also pins the point
    /// *size* to one pixel, which is the part most likely to drift between backends.
    constexpr uint32_t kExpectedPoints = 6;

    RasterState State()
    {
        RasterState state;
        state.topology = AGFX_TOPOLOGY_POINTS;
        return state;
    }
} // namespace

AGFX_TEST_TEXTURE(DrawPoints, C, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::C, State(), image), "point render failed");
    AGFX_EXPECT_EQ(CountLitPixels(image), kExpectedPoints);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawPoints, Cpp, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Cpp, State(), image), "point render failed");
    AGFX_EXPECT_EQ(CountLitPixels(image), kExpectedPoints);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawPoints, Ez, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Ez, State(), image), "point render failed");
    AGFX_EXPECT_EQ(CountLitPixels(image), kExpectedPoints);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
