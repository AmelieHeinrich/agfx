/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw lines".
//
// Reinterprets the same six raster.hlsl vertices as AGFX_TOPOLOGY_LINES: three disjoint line
// segments per triangle's worth of vertices, not a closed outline. That distinction is the point —
// the result is deliberately *not* the same image as the wireframe test, so a backend that maps
// LINES onto a line-strip, or falls back to triangles, produces a visibly different golden instead
// of an image that happens to look plausible.

#include "raster_common.h"

namespace
{
    using namespace agfxtest;

    /// @brief Three one-pixel-wide segments cover ~255 pixels here. Asserted as a floor because the
    /// golden's mean-error comparison cannot tell 1.5% coverage from an empty image (see
    /// CountLitPixels); an exact count would instead be hostage to each backend's line rasterization
    /// rules, which legitimately differ at the endpoints.
    constexpr uint32_t kMinLinePixels = 150;

    constexpr const char* kGolden = "draw_lines.png";

    RasterState State()
    {
        RasterState state;
        state.topology = AGFX_TOPOLOGY_LINES;
        return state;
    }
} // namespace

AGFX_TEST_TEXTURE(DrawLines, C, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::C, State(), image), "line render failed");
    AGFX_EXPECT(CountLitPixels(image) > kMinLinePixels);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawLines, Cpp, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Cpp, State(), image), "line render failed");
    AGFX_EXPECT(CountLitPixels(image) > kMinLinePixels);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawLines, Ez, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Ez, State(), image), "line render failed");
    AGFX_EXPECT(CountLitPixels(image) > kMinLinePixels);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
