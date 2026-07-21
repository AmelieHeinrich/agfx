/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "sampler filter".
//
// One test per agfxSamplerFilter, both sampling the same seeded 64x64 source through a 4x
// magnification (scale 0.25 around the source's center). Magnification is the whole point: at 1:1
// every destination texel lands on a source texel center, where NEAREST and LINEAR agree exactly and
// the two goldens would be identical. Blown up 4x, each source texel covers a 4x4 block, so NEAREST
// produces hard blocky edges and LINEAR a smooth gradient across them — a difference of tens of
// percent per pixel at the block boundaries, far above the FLIP threshold.
//
// The window is 16 source texels per axis rather than the 8 a higher magnification would give,
// because texture_ops.hlsl's pattern carries its checker on an 8-texel cell: an 8-texel window is
// exactly one cell, leaving a single interior boundary per axis and a nearly flat u/v gradient over
// it. 16 texels spans two full cells, so the goldens differ across many boundaries instead of one.
//
// The pair matters more than either test alone: a backend that hardcodes one filter passes that
// one's golden and fails the other, which localizes the bug to the filter field rather than to
// sampling in general.

#include "sampling_common.h"

namespace
{
    using namespace agfxtest;

    // The source's center quarter, blown up to fill the destination. Offset keeps the window
    // centered: (1 - 0.25) / 2.
    constexpr float kUvScale = 0.25f;
    constexpr float kUvOffset = 0.375f;

    SampleState State(agfxSamplerFilter filter)
    {
        SampleState state;
        state.sampler = DefaultSamplerInfo();
        state.sampler.filter = filter;
        state.uvScale[0] = kUvScale;
        state.uvScale[1] = kUvScale;
        state.uvOffset[0] = kUvOffset;
        state.uvOffset[1] = kUvOffset;
        return state;
    }
} // namespace

AGFX_TEST_TEXTURE(SamplerFilterNearest, C, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::C, State(AGFX_SAMPLER_FILTER_NEAREST), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_filter_nearest.png", image);
}

AGFX_TEST_TEXTURE(SamplerFilterNearest, Cpp, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::Cpp, State(AGFX_SAMPLER_FILTER_NEAREST), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_filter_nearest.png", image);
}

AGFX_TEST_TEXTURE(SamplerFilterNearest, Ez, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::Ez, State(AGFX_SAMPLER_FILTER_NEAREST), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_filter_nearest.png", image);
}

AGFX_TEST_TEXTURE(SamplerFilterLinear, C, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::C, State(AGFX_SAMPLER_FILTER_LINEAR), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_filter_linear.png", image);
}

AGFX_TEST_TEXTURE(SamplerFilterLinear, Cpp, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::Cpp, State(AGFX_SAMPLER_FILTER_LINEAR), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_filter_linear.png", image);
}

AGFX_TEST_TEXTURE(SamplerFilterLinear, Ez, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::Ez, State(AGFX_SAMPLER_FILTER_LINEAR), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_filter_linear.png", image);
}
