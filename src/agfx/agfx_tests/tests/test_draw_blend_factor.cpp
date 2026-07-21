/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "draw alpha blending (one for each blend test op/factor)" -- the factor half.
//
// One test per agfxBlendFactor, each applied as the *source* factor with the destination factor
// pinned to ONE and the operation to ADD. Every result is therefore `src * factor + dst`, so the
// factor is the only unknown in the equation and each golden reads directly as what that factor
// evaluated to.
//
// Pinning dst to ONE rather than ZERO is deliberate: with ZERO the destination is discarded and the
// four regions of the scene collapse to one, taking DST_COLOR and the DST_ALPHA pair down with them.
//
// The scene (blend_common.h) supplies four destinations at once -- three dim columns of differing
// color and differing alpha, plus a transparent-black background -- which is what keeps the ten
// goldens apart. Several of these factors are equal to each other against *some* destination:
// DST_COLOR matches ZERO over the black background, DST_ALPHA matches ONE over the opaque column,
// and SRC_COLOR matches ONE_MINUS_SRC_COLOR in any channel where the source happens to be 0.5.
// None of them match across all four regions at once.
//
// See test_draw_blend_op.cpp for the operation half, which pins the factors and varies the
// arithmetic instead.

#include "blend_common.h"

namespace
{
    using namespace agfxtest;

    void RunBlendFactorTest(TestContext& ctx, TestApi api, agfxBlendFactor srcBlend, const char* golden)
    {
        Image image;
        const BlendState state = State(srcBlend, AGFX_BLEND_FACTOR_ONE, AGFX_BLEND_OPERATION_ADD);
        AGFX_EXPECT_MSG(RenderBlend(api, state, image), "blend render failed");
        ExpectImageMatchesGolden(ctx, golden, image);
    }
} // namespace

#define AGFX_BLEND_FACTOR_CASE(TestName, factor, golden)                                           \
    AGFX_TEST_TEXTURE(TestName, C, kBlendWidth, kBlendHeight)                                      \
    {                                                                                              \
        RunBlendFactorTest(ctx, TestApi::C, factor, golden);                                       \
    }                                                                                              \
    AGFX_TEST_TEXTURE(TestName, Cpp, kBlendWidth, kBlendHeight)                                    \
    {                                                                                              \
        RunBlendFactorTest(ctx, TestApi::Cpp, factor, golden);                                     \
    }                                                                                              \
    AGFX_TEST_TEXTURE(TestName, Ez, kBlendWidth, kBlendHeight)                                     \
    {                                                                                              \
        RunBlendFactorTest(ctx, TestApi::Ez, factor, golden);                                      \
    }

AGFX_BLEND_FACTOR_CASE(DrawBlendFactorZero, AGFX_BLEND_FACTOR_ZERO, "draw_blend_factor_zero.png")
AGFX_BLEND_FACTOR_CASE(DrawBlendFactorOne, AGFX_BLEND_FACTOR_ONE, "draw_blend_factor_one.png")
AGFX_BLEND_FACTOR_CASE(DrawBlendFactorSrcColor, AGFX_BLEND_FACTOR_SRC_COLOR,
                       "draw_blend_factor_src_color.png")
AGFX_BLEND_FACTOR_CASE(DrawBlendFactorOneMinusSrcColor, AGFX_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                       "draw_blend_factor_one_minus_src_color.png")
AGFX_BLEND_FACTOR_CASE(DrawBlendFactorDstColor, AGFX_BLEND_FACTOR_DST_COLOR,
                       "draw_blend_factor_dst_color.png")
AGFX_BLEND_FACTOR_CASE(DrawBlendFactorOneMinusDstColor, AGFX_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
                       "draw_blend_factor_one_minus_dst_color.png")
AGFX_BLEND_FACTOR_CASE(DrawBlendFactorSrcAlpha, AGFX_BLEND_FACTOR_SRC_ALPHA,
                       "draw_blend_factor_src_alpha.png")
AGFX_BLEND_FACTOR_CASE(DrawBlendFactorOneMinusSrcAlpha, AGFX_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                       "draw_blend_factor_one_minus_src_alpha.png")
AGFX_BLEND_FACTOR_CASE(DrawBlendFactorDstAlpha, AGFX_BLEND_FACTOR_DST_ALPHA,
                       "draw_blend_factor_dst_alpha.png")
AGFX_BLEND_FACTOR_CASE(DrawBlendFactorOneMinusDstAlpha, AGFX_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                       "draw_blend_factor_one_minus_dst_alpha.png")

// The canonical case the TODO entry is named after, and the only test here with *asymmetric*
// factors. Everything above pins dstBlend to ONE, which means a backend that swapped the source and
// destination factor slots would still produce the right image for most of them. Here the two
// factors are different numbers (0.25 and 0.75), so a swap changes the result -- this is the test
// that pins down the wiring rather than the enum mapping.
AGFX_TEST_TEXTURE(DrawAlphaBlend, C, kBlendWidth, kBlendHeight)
{
    Image image;
    const BlendState state = State(AGFX_BLEND_FACTOR_SRC_ALPHA, AGFX_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                   AGFX_BLEND_OPERATION_ADD);
    AGFX_EXPECT_MSG(RenderBlend(TestApi::C, state, image), "blend render failed");
    ExpectImageMatchesGolden(ctx, "draw_alpha_blend.png", image);
}

AGFX_TEST_TEXTURE(DrawAlphaBlend, Cpp, kBlendWidth, kBlendHeight)
{
    Image image;
    const BlendState state = State(AGFX_BLEND_FACTOR_SRC_ALPHA, AGFX_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                   AGFX_BLEND_OPERATION_ADD);
    AGFX_EXPECT_MSG(RenderBlend(TestApi::Cpp, state, image), "blend render failed");
    ExpectImageMatchesGolden(ctx, "draw_alpha_blend.png", image);
}

AGFX_TEST_TEXTURE(DrawAlphaBlend, Ez, kBlendWidth, kBlendHeight)
{
    Image image;
    const BlendState state = State(AGFX_BLEND_FACTOR_SRC_ALPHA, AGFX_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                   AGFX_BLEND_OPERATION_ADD);
    AGFX_EXPECT_MSG(RenderBlend(TestApi::Ez, state, image), "blend render failed");
    ExpectImageMatchesGolden(ctx, "draw_alpha_blend.png", image);
}
