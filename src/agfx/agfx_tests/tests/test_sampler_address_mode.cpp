/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "sampler address mode".
//
// One test per agfxSamplerAddressMode, all sampling the same seeded 64x64 source through a uv
// transform that deliberately leaves [0,1]: scale 3, offset -1, so the destination sweeps u and v
// across [-1, 2]. That is one full period on either side of the source, which is the minimum that
// separates every mode from every other:
//
//   REPEAT          -> three identical tiles per axis
//   MIRRORED_REPEAT -> the outer two tiles are flipped, so the seams are mirror-symmetric
//   CLAMP_TO_EDGE   -> the outer thirds are edge texels smeared outward
//
// A backend that ignores the address mode entirely and always clamps passes CLAMP_TO_EDGE and fails
// the other two; one that confuses repeat with mirrored repeat fails on the outer tiles' handedness
// rather than on a subtle edge difference.
//
// The address mode is set on U and V together but the transform is symmetric, so a backend that
// swapped the two axes would not be caught here — that is texture_ops.hlsl's asymmetric pattern's
// job, and it is already pinned by the load tests.
//
// CLAMP_TO_BORDER is deliberately absent: agfxSamplerCreateInfo exposes no border color, so the
// border is whatever the backend defaults to (D3D12 and Metal do not agree), and any golden for it
// would be backend-specific rather than a statement about AGFX. Testing it needs a borderColor field
// on the create info first.

#include "sampling_common.h"

namespace
{
    using namespace agfxtest;

    // One full period of the source on either side of it, so every mode has room to show its
    // characteristic behavior twice rather than only at one edge.
    constexpr float kUvScale = 3.0f;
    constexpr float kUvOffset = -1.0f;

    SampleState State(agfxSamplerAddressMode mode)
    {
        SampleState state;
        state.sampler = DefaultSamplerInfo();
        state.sampler.addressModeU = mode;
        state.sampler.addressModeV = mode;
        state.uvScale[0] = kUvScale;
        state.uvScale[1] = kUvScale;
        state.uvOffset[0] = kUvOffset;
        state.uvOffset[1] = kUvOffset;
        return state;
    }
} // namespace

AGFX_TEST_TEXTURE(SamplerAddressModeRepeat, C, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::C, State(AGFX_SAMPLER_ADDRESS_MODE_REPEAT), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_address_mode_repeat.png", image);
}

AGFX_TEST_TEXTURE(SamplerAddressModeRepeat, Cpp, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::Cpp, State(AGFX_SAMPLER_ADDRESS_MODE_REPEAT), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_address_mode_repeat.png", image);
}

AGFX_TEST_TEXTURE(SamplerAddressModeRepeat, Ez, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::Ez, State(AGFX_SAMPLER_ADDRESS_MODE_REPEAT), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_address_mode_repeat.png", image);
}

AGFX_TEST_TEXTURE(SamplerAddressModeMirroredRepeat, C, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::C, State(AGFX_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_address_mode_mirrored_repeat.png", image);
}

AGFX_TEST_TEXTURE(SamplerAddressModeMirroredRepeat, Cpp, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::Cpp, State(AGFX_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_address_mode_mirrored_repeat.png", image);
}

AGFX_TEST_TEXTURE(SamplerAddressModeMirroredRepeat, Ez, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::Ez, State(AGFX_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_address_mode_mirrored_repeat.png", image);
}

AGFX_TEST_TEXTURE(SamplerAddressModeClampToEdge, C, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::C, State(AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_address_mode_clamp_to_edge.png", image);
}

AGFX_TEST_TEXTURE(SamplerAddressModeClampToEdge, Cpp, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::Cpp, State(AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_address_mode_clamp_to_edge.png", image);
}

AGFX_TEST_TEXTURE(SamplerAddressModeClampToEdge, Ez, kSamplerWidth, kSamplerHeight)
{
    Image image;
    AGFX_EXPECT_MSG(RenderSample(TestApi::Ez, State(AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE), image), "render failed");
    ExpectImageMatchesGolden(ctx, "sampler_address_mode_clamp_to_edge.png", image);
}
