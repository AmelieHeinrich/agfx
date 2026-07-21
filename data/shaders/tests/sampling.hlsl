//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: the sampled counterparts of texture_ops.hlsl's Load() entry points. Where the load
// tests pin down texel addressing, these pin down the sampler itself — filtering, address mode, and
// the SamplerDescriptorHeap lookup that turns a bindless handle back into a SamplerState.
//
// Every entry point walks the destination texel grid, maps it to a normalized coordinate through a
// caller-supplied scale/offset, and samples the source there. The scale/offset is what lets one
// shader serve several tests: a scale below 1 magnifies (so LINEAR vs NEAREST separate visibly),
// and a scale above 1 with a negative offset pushes the coordinate outside [0,1] (so the address
// mode decides what comes back).
//
// SampleLevel is used rather than Sample throughout: there are no derivatives in a compute shader,
// so an implicit-LOD Sample would be undefined here.

#include "data/shaders/agfx.h"

struct SamplingPushConstants
{
    ResourceHandle source;      // AGFXTexture2D/2DArray/3D, read-only.
    ResourceHandle samplerId;   // AGFXSampler, looked up out of SamplerDescriptorHeap.
    ResourceHandle destination; // AGFXRWTexture2D, writeable. Always 2D: the layered entry points
                                // flatten their slices into vertical bands so one golden holds them.
    uint width;                 // Destination width.
    uint height;                // Destination height, per slice for the layered entry points.
    uint sliceCount;            // Source slices to walk. 1 for the plain 2D case.
    float2 uvScale;             // Destination [0,1] -> source coordinate, scale then offset.
    float2 uvOffset;
    float2 padding0;
};

AGFX_PUSH_CONSTANTS(SamplingPushConstants, g_Constants);

// The destination texel's center in normalized space, pushed through the caller's transform. Using
// the texel *center* (+0.5) rather than its corner keeps the mapping symmetric, so a backend with a
// half-texel offset shows up as a shifted image rather than as an edge-only difference.
float2 SourceUV(uint2 texel)
{
    const float2 uv = (float2(texel) + 0.5f) / float2(g_Constants.width, g_Constants.height);
    return uv * g_Constants.uvScale + g_Constants.uvOffset;
}

[numthreads(8, 8, 1)]
void main_sample_2d_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXTexture2D<float4> src = AGFXTexture2D<float4>::Create(g_Constants.source);
    AGFXSampler smp = AGFXSampler::Create(g_Constants.samplerId);
    AGFXRWTexture2D<float4> dst = AGFXRWTexture2D<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    dst.Store(int2(id.xy), src.SampleLevel(smp, SourceUV(id.xy), 0.0f));
}

// The layered entry points write a (width, height * sliceCount) destination: slice s lands in the
// band starting at y = s * height. A shader or backend that ignores the slice coordinate produces
// identical bands, which reads as an obviously wrong golden rather than a subtle one.

[numthreads(8, 8, 1)]
void main_sample_2d_array_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXTexture2DArray<float4> src = AGFXTexture2DArray<float4>::Create(g_Constants.source);
    AGFXSampler smp = AGFXSampler::Create(g_Constants.samplerId);
    AGFXRWTexture2D<float4> dst = AGFXRWTexture2D<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height || id.z >= g_Constants.sliceCount) {
        return;
    }

    // The array index is the third coordinate and is *not* filtered between slices, so it stays an
    // exact integer rather than a normalized value.
    const float3 coord = float3(SourceUV(id.xy), float(id.z));
    const float4 texel = src.SampleLevel(smp, coord, 0.0f);

    dst.Store(int2(id.x, id.y + id.z * g_Constants.height), texel);
}

[numthreads(8, 8, 1)]
void main_sample_3d_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXTexture3D<float4> src = AGFXTexture3D<float4>::Create(g_Constants.source);
    AGFXSampler smp = AGFXSampler::Create(g_Constants.samplerId);
    AGFXRWTexture2D<float4> dst = AGFXRWTexture2D<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height || id.z >= g_Constants.sliceCount) {
        return;
    }

    // Unlike the array case, w is normalized and *is* filtered. Aim at each source slice's center
    // so the result is that slice rather than a blend of two neighbours -- the depth filtering
    // itself is left to the filter test, which magnifies along z deliberately.
    const float w = (float(id.z) + 0.5f) / float(g_Constants.sliceCount);
    const float3 coord = float3(SourceUV(id.xy), w);
    const float4 texel = src.SampleLevel(smp, coord, 0.0f);

    dst.Store(int2(id.x, id.y + id.z * g_Constants.height), texel);
}
