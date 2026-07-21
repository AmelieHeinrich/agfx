/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw with push constants".
//
// Draws both raster.hlsl triangles in a flat color supplied entirely by push constants, with the
// shader's per-vertex gradient switched off. A specific non-trivial color (not white, not a
// primary) is used deliberately: a backend that fails to upload the block leaves it zeroed and
// renders black, one that misaligns it picks up the neighbouring float and renders a
// recognizably-wrong channel, and one that only pushes to the vertex stage renders nothing at all.
// Each of those is a distinct failure in the golden rather than the same "it's dark" outcome.

#include "raster_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "draw_push_constants.png";

    RasterState State()
    {
        RasterState state;
        state.constants.useVertexColor = 0;
        state.constants.color[0] = 0.25f;
        state.constants.color[1] = 0.75f;
        state.constants.color[2] = 0.50f;
        state.constants.color[3] = 1.0f;
        return state;
    }
} // namespace

AGFX_TEST_TEXTURE(DrawPushConstants, C, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::C, State(), image), "push constant render failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawPushConstants, Cpp, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Cpp, State(), image), "push constant render failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawPushConstants, Ez, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Ez, State(), image), "push constant render failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
