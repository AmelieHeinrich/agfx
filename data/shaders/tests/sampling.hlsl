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

// The cube entry point takes the same (width, height * sliceCount) destination as the layered ones,
// with sliceCount pinned to 6: band f holds the result of sampling cube face f.
//
// A cube texture is not addressed by a coordinate but by a *direction*, so the test has to invert
// the hardware's face-selection rule: FaceDirection() maps (face, uv) to the direction that should
// land back on exactly that face at exactly that uv. If AGFX and the backend agree on face order and
// orientation, band f reproduces the seeded layer f. If a backend swaps two faces, or flips one's
// s/t axes, the corresponding band shows the wrong layer or a mirrored one -- which is visible
// against the seed's per-slice brightness gradient rather than being a subtle interior difference.
//
// The signs follow the standard D3D/Metal cube convention; they are the whole content of this
// helper, so getting them wrong here would mean testing the test rather than the backend.
float3 FaceDirection(uint face, float2 uv)
{
    const float s = 2.0f * uv.x - 1.0f;
    const float t = 2.0f * uv.y - 1.0f;

    switch (face) {
    case 0: return float3( 1.0f,   -t,   -s); // +X
    case 1: return float3(-1.0f,   -t,    s); // -X
    case 2: return float3(    s, 1.0f,    t); // +Y
    case 3: return float3(    s, -1.0f,  -t); // -Y
    case 4: return float3(    s,   -t, 1.0f); // +Z
    default: return float3(  -s,   -t, -1.0f); // -Z
    }
}

[numthreads(8, 8, 1)]
void main_sample_cube_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXTextureCube<float4> src = AGFXTextureCube<float4>::Create(g_Constants.source);
    AGFXSampler smp = AGFXSampler::Create(g_Constants.samplerId);
    AGFXRWTexture2D<float4> dst = AGFXRWTexture2D<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height || id.z >= g_Constants.sliceCount) {
        return;
    }

    // The direction is deliberately built from the *untransformed* uv: the uv scale/offset the other
    // entry points use to push coordinates out of range has no meaning for a cube, where leaving a
    // face wraps onto a neighbour rather than triggering the address mode.
    const float2 uv = (float2(id.xy) + 0.5f) / float2(g_Constants.width, g_Constants.height);
    const float4 texel = src.SampleLevel(smp, FaceDirection(id.z, uv), 0.0f);

    dst.Store(int2(id.x, id.y + id.z * g_Constants.height), texel);
}
