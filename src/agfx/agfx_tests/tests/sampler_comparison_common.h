/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

// Shared scaffolding for the sampler comparison tests: one per agfxComparisonFunction, driving
// data/shaders/tests/sampling_comparison.hlsl.
//
// Separate from sampling_common.h rather than folded into it, because almost nothing is shared. A
// comparison sampler needs a *depth* source, which means a depth texture, a render pass to establish
// its contents, and an R32F texture view to read it back through — none of which the color-sampling
// path has any use for. What is shared is the idea, not the code.
//
// The source depth is uniform, and the destination is split into three vertical bands whose
// reference values straddle it: nearer, exactly equal, farther. Which bands survive names the
// comparison function uniquely, on the same principle as the depth *test* tests:
//
//   NEVER         -> nothing
//   LESS          -> nearer
//   EQUAL         -> equal
//   LESS_EQUAL    -> nearer + equal
//   GREATER       -> farther
//   NOT_EQUAL     -> nearer + farther
//   GREATER_EQUAL -> equal + farther
//   ALWAYS        -> all three
//
// Because the expected bands are known analytically, these tests do not lean on their goldens as the
// oracle: ExpectedBands() states the answer up front, and a golden captured against a broken backend
// cannot bless it.

#include "../test_gpu.h"

#include <string>

namespace agfxtest
{
    constexpr uint32_t kCmpWidth = 64; // 64 * 4 bytes = 256: D3D12's row-pitch alignment.
    constexpr uint32_t kCmpHeight = 64;
    constexpr uint32_t kCmpBandCount = 3;

    /// @brief The depth stored in the source, and the per-band reference values compared against it.
    /// All three are exactly representable in float32, so EQUAL and NOT_EQUAL are exact rather than
    /// resting on a tolerance.
    constexpr float kCmpStoredDepth = 0.5f;
    constexpr float kCmpNearer = 0.25f;
    constexpr float kCmpFarther = 0.75f;

    /// @brief Which of the three bands the given comparison function should light up.
    /// Index 0 is the nearer band, 1 the equal band, 2 the farther band.
    void ExpectedBands(agfxComparisonFunction compareOp, bool outBands[kCmpBandCount]);

    /// @brief True if `image` is exactly the black/white banding `compareOp` calls for: every pixel
    /// of a lit band white, every pixel of an unlit band black.
    bool ImageMatchesBands(const Image& image, agfxComparisonFunction compareOp);

    /// @brief Establishes the depth source, samples it through a comparison sampler built with
    /// `compareOp`, and writes the banded result into `outImage`.
    /// @param outError Set to a description when the run failed; empty on success.
    bool RenderSamplerComparison(TestApi api, agfxComparisonFunction compareOp, Image& outImage,
                                 std::string& outError);
} // namespace agfxtest
