/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "pipeline cull mode".
//
// One test per agfxCullMode, all with frontFace pinned to COUNTER_CLOCKWISE. raster.hlsl draws two
// triangles of opposite winding on opposite halves of the target, so each cull mode has a distinct,
// unambiguous golden:
//
//   NONE  -> both triangles survive
//   BACK  -> only the left (counter-clockwise, front-facing) triangle survives
//   FRONT -> only the right (clockwise, back-facing) triangle survives
//
// Because the two survivors are on opposite halves and carry different colors, an inverted culling
// sense fails as "the wrong half is lit" rather than as a subtle edge difference — and a backend
// that drops culling entirely fails all three at once instead of passing NONE by accident.

#include "raster_common.h"

namespace
{
    using namespace agfxtest;

    RasterState State(agfxCullMode cullMode)
    {
        RasterState state;
        state.cullMode = cullMode;
        state.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
        return state;
    }
} // namespace

AGFX_TEST_TEXTURE(PipelineCullModeNone, C, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::C, State(AGFX_CULL_MODE_NONE), image), "render failed");
    ExpectImageMatchesGolden(ctx, "pipeline_cull_mode_none.png", image);
}

AGFX_TEST_TEXTURE(PipelineCullModeNone, Cpp, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Cpp, State(AGFX_CULL_MODE_NONE), image), "render failed");
    ExpectImageMatchesGolden(ctx, "pipeline_cull_mode_none.png", image);
}

AGFX_TEST_TEXTURE(PipelineCullModeNone, Ez, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Ez, State(AGFX_CULL_MODE_NONE), image), "render failed");
    ExpectImageMatchesGolden(ctx, "pipeline_cull_mode_none.png", image);
}

AGFX_TEST_TEXTURE(PipelineCullModeBack, C, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::C, State(AGFX_CULL_MODE_BACK), image), "render failed");
    ExpectImageMatchesGolden(ctx, "pipeline_cull_mode_back.png", image);
}

AGFX_TEST_TEXTURE(PipelineCullModeBack, Cpp, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Cpp, State(AGFX_CULL_MODE_BACK), image), "render failed");
    ExpectImageMatchesGolden(ctx, "pipeline_cull_mode_back.png", image);
}

AGFX_TEST_TEXTURE(PipelineCullModeBack, Ez, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Ez, State(AGFX_CULL_MODE_BACK), image), "render failed");
    ExpectImageMatchesGolden(ctx, "pipeline_cull_mode_back.png", image);
}

AGFX_TEST_TEXTURE(PipelineCullModeFront, C, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::C, State(AGFX_CULL_MODE_FRONT), image), "render failed");
    ExpectImageMatchesGolden(ctx, "pipeline_cull_mode_front.png", image);
}

AGFX_TEST_TEXTURE(PipelineCullModeFront, Cpp, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Cpp, State(AGFX_CULL_MODE_FRONT), image), "render failed");
    ExpectImageMatchesGolden(ctx, "pipeline_cull_mode_front.png", image);
}

AGFX_TEST_TEXTURE(PipelineCullModeFront, Ez, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Ez, State(AGFX_CULL_MODE_FRONT), image), "render failed");
    ExpectImageMatchesGolden(ctx, "pipeline_cull_mode_front.png", image);
}
