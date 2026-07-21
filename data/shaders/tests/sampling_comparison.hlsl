//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: comparison (shadow) sampling, the one sampler feature the rest of sampling.hlsl does
// not reach. Where those entry points go through AGFXSampler and return a color, this one goes
// through AGFXComparisonSampler and returns the *result of a comparison* -- a 0 or 1 per tap decided
// by the sampler's comparisonFunction, not by the texel's value.
//
// The source is a depth texture viewed as R32F, which is the same shape the demo's cascaded shadow
// maps use (AGFXTexture2DArray<float> + AGFXComparisonSampler + SampleCmpLevelZero in
// data/shaders/demo/pbr_common.hlsl). Testing it as a plain 2D keeps the test about the comparison
// function rather than about array addressing, which sample_2d_array already covers.
//
// The destination is split into vertical bands, one per reference value. With a uniform depth in the
// source, each band reduces to a single comparison of a known reference against a known stored
// depth -- so the set of bands that light up names the comparison function uniquely, the same way
// the columns do in the depth *test* tests.
//
// SampleCmpLevelZero rather than SampleCmp: there are no derivatives in a compute shader, so the
// implicit-LOD form would be undefined here.

#include "data/shaders/agfx.h"

struct SamplingComparisonPushConstants
{
    ResourceHandle source;      // AGFXTexture2D<float>: an R32F view over the depth texture.
    ResourceHandle samplerId;   // AGFXComparisonSampler, out of SamplerDescriptorHeap.
    ResourceHandle destination; // AGFXRWTexture2D<float4>, writeable.
    uint width;                 // Destination width.
    uint height;                // Destination height.
    uint bandCount;             // Vertical bands to split the destination into.
    uint padding0;
    uint padding1;
    float4 references;          // Per-band reference value; only the first bandCount are read.
};

AGFX_PUSH_CONSTANTS(SamplingComparisonPushConstants, g_Constants);

[numthreads(8, 8, 1)]
void main_sample_cmp_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXTexture2D<float> src = AGFXTexture2D<float>::Create(g_Constants.source);
    AGFXComparisonSampler smp = AGFXComparisonSampler::Create(g_Constants.samplerId);
    AGFXRWTexture2D<float4> dst = AGFXRWTexture2D<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    // min() guards the last column: with width not divisible by bandCount, the final texel would
    // otherwise index one past the last band.
    const uint band = min((id.x * g_Constants.bandCount) / g_Constants.width, g_Constants.bandCount - 1);
    const float reference = g_Constants.references[band];

    // The texel center, as everywhere else in the suite -- though with a uniform source depth the
    // exact coordinate cannot change the result, which is deliberate: it leaves the comparison
    // function as the only thing this shader's output depends on.
    const float2 uv = (float2(id.xy) + 0.5f) / float2(g_Constants.width, g_Constants.height);

    // The comparison is `reference OP storedDepth`, with OP coming from the sampler. The result is
    // 0 or 1 per tap; with a point-filtered sampler there is exactly one tap, so no PCF averaging
    // stands between the comparison and the value written here.
    const float result = src.SampleCmpLevelZero(smp, uv, reference);

    dst.Store(int2(id.xy), float4(result, result, result, 1.0f));
}

// Ordinary (non-comparison) sampling of the same depth-texture-viewed-as-R32F source. This is the
// other half of what a D32 -> R32F view has to support: comparison sampling above proves the view
// can feed a SamplerComparisonState, this proves it reads back the stored depth *value* rather than
// a comparison result. The destination is RGBA32F so the depth survives readback exactly -- values
// like 0.25 and 0.75 are exact in float32 but not in 8-bit unorm, and quantizing them would turn an
// exactness check into a tolerance.
[numthreads(8, 8, 1)]
void main_sample_depth_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXTexture2D<float> src = AGFXTexture2D<float>::Create(g_Constants.source);
    AGFXSampler smp = AGFXSampler::Create(g_Constants.samplerId);
    AGFXRWTexture2D<float4> dst = AGFXRWTexture2D<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    const float2 uv = (float2(id.xy) + 0.5f) / float2(g_Constants.width, g_Constants.height);
    const float depth = src.SampleLevel(smp, uv, 0.0f);

    dst.Store(int2(id.xy), float4(depth, depth, depth, 1.0f));
}
