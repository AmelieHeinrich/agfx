/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw ccw" and "draw cw".
//
// The dual of the cull mode test: cullMode is pinned to BACK and agfxFrontFace is what varies. Since
// raster.hlsl's two triangles are wound oppositely, flipping frontFace swaps which one is considered
// front-facing and therefore which half of the image survives:
//
//   COUNTER_CLOCKWISE -> the left triangle is front-facing and survives
//   CLOCKWISE         -> the right triangle is front-facing and survives
//
// This is worth testing separately from cull mode because the two are independently easy to get
// backwards, and a backend that inverts *both* would pass a cull-mode-only test while rendering
// every real scene inside out. Here the two goldens are exact mirrors, so one inverted setting
// cannot be masked by the other.

#include "raster_common.h"

namespace
{
    using namespace agfxtest;

    RasterState State(agfxFrontFace frontFace)
    {
        RasterState state;
        state.cullMode = AGFX_CULL_MODE_BACK;
        state.frontFace = frontFace;
        return state;
    }
} // namespace

AGFX_TEST_TEXTURE(DrawCCW, C, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::C, State(AGFX_FRONT_FACE_COUNTER_CLOCKWISE), image), "render failed");
    ExpectImageMatchesGolden(ctx, "draw_ccw.png", image);
}

AGFX_TEST_TEXTURE(DrawCCW, Cpp, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Cpp, State(AGFX_FRONT_FACE_COUNTER_CLOCKWISE), image), "render failed");
    ExpectImageMatchesGolden(ctx, "draw_ccw.png", image);
}

AGFX_TEST_TEXTURE(DrawCCW, Ez, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Ez, State(AGFX_FRONT_FACE_COUNTER_CLOCKWISE), image), "render failed");
    ExpectImageMatchesGolden(ctx, "draw_ccw.png", image);
}

AGFX_TEST_TEXTURE(DrawCW, C, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::C, State(AGFX_FRONT_FACE_CLOCKWISE), image), "render failed");
    ExpectImageMatchesGolden(ctx, "draw_cw.png", image);
}

AGFX_TEST_TEXTURE(DrawCW, Cpp, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Cpp, State(AGFX_FRONT_FACE_CLOCKWISE), image), "render failed");
    ExpectImageMatchesGolden(ctx, "draw_cw.png", image);
}

AGFX_TEST_TEXTURE(DrawCW, Ez, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Ez, State(AGFX_FRONT_FACE_CLOCKWISE), image), "render failed");
    ExpectImageMatchesGolden(ctx, "draw_cw.png", image);
}
