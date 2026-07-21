/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw fragment discard".
//
// Draws both raster.hlsl triangles but sets the shader's discardX threshold to 0.5, so every
// fragment with NDC x > 0.5 is killed by clip().
//
// The threshold is chosen to fall *inside* the right triangle (which spans x 0.1 to 0.9) rather
// than between the two: the golden shows the left triangle untouched and the right one with a
// vertical slice taken out of its right side. That distinction matters — a threshold that merely
// removed the whole right primitive would produce an image identical to the cull-mode-back golden,
// and the test would pass just as happily on a backend that discarded nothing but culled wrongly.
// Cutting through the middle of a primitive can only be produced by per-fragment kills.
//
// The threshold rides in as a push constant rather than being hardcoded in the shader so this test
// shares raster.hlsl with the rest of the raster suite; that also means it doubles as a check that
// push constants reach the fragment stage at all.

#include "raster_common.h"

namespace
{
    using namespace agfxtest;

    constexpr const char* kGolden = "draw_fragment_discard.png";

    RasterState State()
    {
        RasterState state;
        state.constants.discardX = 0.5f;
        return state;
    }
} // namespace

AGFX_TEST_TEXTURE(DrawFragmentDiscard, C, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::C, State(), image), "discard render failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawFragmentDiscard, Cpp, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Cpp, State(), image), "discard render failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(DrawFragmentDiscard, Ez, kRasterWidth, kRasterHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderRaster(TestApi::Ez, State(), image), "discard render failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
