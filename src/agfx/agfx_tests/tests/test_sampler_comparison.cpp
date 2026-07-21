/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "sampler comparison (one for each comparison function)".
//
// One test per agfxComparisonFunction, sampling a uniform depth source through a comparison sampler
// and reading which of three reference bands the comparison lets through. See
// sampler_comparison_common.h for the band-to-function table.
//
// ALWAYS is deliberately absent, for the same class of reason CLAMP_TO_BORDER is absent from the
// address mode tests: it is not expressible. AGFX uses AGFX_COMPARISON_FUNCTION_ALWAYS as the
// sentinel for "this is not a comparison sampler" -- DefaultSamplerInfo() in sampling_common.cpp
// sets it on every ordinary color sampler, and the D3D12 backend keys the comparison filter off
// `comparisonFunction != AGFX_COMPARISON_FUNCTION_ALWAYS` (agfx_d3d12.cpp:2182). So there is no way
// to ask for a comparison sampler whose function is ALWAYS, and a test for one would be asserting
// backend-specific behavior rather than AGFX's. That leaves seven.
//
// Each test checks the banding analytically via ImageMatchesBands *before* comparing the golden: the
// expected answer is known up front, so a golden captured against a broken backend cannot bless it.

#include "sampler_comparison_common.h"

#define AGFX_SAMPLER_COMPARISON_CASE(TestName, compareOp, golden)                                  \
    static void RunSamplerComparison##TestName(agfxtest::TestContext& ctx, agfxtest::TestApi api)  \
    {                                                                                              \
        using namespace agfxtest;                                                                  \
        Image image;                                                                               \
        std::string error;                                                                         \
        AGFX_EXPECT_MSG(RenderSamplerComparison(api, compareOp, image, error), error.c_str());     \
        AGFX_EXPECT_MSG(ImageMatchesBands(image, compareOp),                                       \
                        "the lit bands do not match the comparison function");                     \
        ExpectImageMatchesGolden(ctx, golden, image);                                              \
    }                                                                                              \
    AGFX_TEST_TEXTURE(TestName, C, agfxtest::kCmpWidth, agfxtest::kCmpHeight)                      \
    {                                                                                              \
        RunSamplerComparison##TestName(ctx, agfxtest::TestApi::C);                                 \
    }                                                                                              \
    AGFX_TEST_TEXTURE(TestName, Cpp, agfxtest::kCmpWidth, agfxtest::kCmpHeight)                    \
    {                                                                                              \
        RunSamplerComparison##TestName(ctx, agfxtest::TestApi::Cpp);                               \
    }                                                                                              \
    AGFX_TEST_TEXTURE(TestName, Ez, agfxtest::kCmpWidth, agfxtest::kCmpHeight)                     \
    {                                                                                              \
        RunSamplerComparison##TestName(ctx, agfxtest::TestApi::Ez);                                \
    }

AGFX_SAMPLER_COMPARISON_CASE(SamplerComparisonNever, AGFX_COMPARISON_FUNCTION_NEVER,
                             "sampler_comparison_never.png")
AGFX_SAMPLER_COMPARISON_CASE(SamplerComparisonLess, AGFX_COMPARISON_FUNCTION_LESS,
                             "sampler_comparison_less.png")
AGFX_SAMPLER_COMPARISON_CASE(SamplerComparisonEqual, AGFX_COMPARISON_FUNCTION_EQUAL,
                             "sampler_comparison_equal.png")
AGFX_SAMPLER_COMPARISON_CASE(SamplerComparisonLessEqual, AGFX_COMPARISON_FUNCTION_LESS_EQUAL,
                             "sampler_comparison_less_equal.png")
AGFX_SAMPLER_COMPARISON_CASE(SamplerComparisonGreater, AGFX_COMPARISON_FUNCTION_GREATER,
                             "sampler_comparison_greater.png")
AGFX_SAMPLER_COMPARISON_CASE(SamplerComparisonNotEqual, AGFX_COMPARISON_FUNCTION_NOT_EQUAL,
                             "sampler_comparison_not_equal.png")
AGFX_SAMPLER_COMPARISON_CASE(SamplerComparisonGreaterEqual, AGFX_COMPARISON_FUNCTION_GREATER_EQUAL,
                             "sampler_comparison_greater_equal.png")
