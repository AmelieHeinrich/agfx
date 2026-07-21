//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shaders for inline raytracing (RayQuery) against an AGFX acceleration structure.
//
// Every entry point traces one ray per texel through the same fixed camera and writes a colour that
// encodes what the query returned, so the result is a golden image rather than a buffer of numbers.
// The camera is orthographic on purpose: a perspective ray fan makes a one-primitive BLAS occupy a
// handful of texels, whereas parallel rays project the geometry at a stable size, so a BLAS that
// builds into the wrong place shifts the silhouette instead of merely dimming a few pixels.
//
// The whole suite shares this file because most RT tests differ only in how the AS under test was
// *produced* (built, updated, compacted, copied, rebuilt from a reused scratch buffer) and not in
// what tracing it should yield. Pointing those tests at one entry point and one golden makes the
// acceleration structure the variable and the image the invariant.

#include "data/shaders/agfx.h"

struct RaytracingPushConstants
{
    ResourceHandle tlas;   // AGFXRaytracingAccelerationStructure to trace against.
    ResourceHandle output; // AGFXRWTexture2D<float4>, writeable.
    uint width;
    uint height;
    uint padding0;
    uint padding1;
    uint padding2;
    uint padding3;
};

AGFX_PUSH_CONSTANTS(RaytracingPushConstants, g_Constants);

// The scene the host side builds always sits inside x,y in [-1,1] at around z = 0, so an
// orthographic frustum of exactly that extent puts the geometry edge-to-edge in the image. Rays
// start well in front of it and run along +z.
RayDesc PrimaryRay(uint2 texel, uint width, uint height)
{
    // +0.5 samples texel centres; without it the ray grid is offset by half a texel and geometry
    // edges land differently on the two backends.
    const float u = (float(texel.x) + 0.5f) / float(width);
    const float v = (float(texel.y) + 0.5f) / float(height);

    RayDesc ray;
    // v is flipped so +y in world space is up in the image, matching how the goldens read.
    ray.Origin = float3(u * 2.0f - 1.0f, 1.0f - v * 2.0f, -2.0f);
    ray.Direction = float3(0.0f, 0.0f, 1.0f);
    ray.TMin = 0.0f;
    ray.TMax = 10.0f;
    return ray;
}

// Distance is normalised across the slab the geometry lives in rather than across TMax, so a hit
// that moves by a small amount in z still produces a visible change in the golden.
float EncodeDistance(float t)
{
    return saturate((t - 1.0f) / 2.0f);
}

// Hit/miss and depth. The workhorse: this is what every "did the AS build correctly" test traces.
[numthreads(8, 8, 1)]
void main_trace_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    AGFXRaytracingAccelerationStructure tlas =
        AGFXRaytracingAccelerationStructure::Create(g_Constants.tlas);
    AGFXRWTexture2D<float4> output = AGFXRWTexture2D<float4>::Create(g_Constants.output);

    RayDesc ray = PrimaryRay(id.xy, g_Constants.width, g_Constants.height);

    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(tlas.Resource(), RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    while (q.Proceed()) {}

    float4 colour = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        colour.rgb = float3(1.0f, EncodeDistance(q.CommittedRayT()), 0.0f);
    }

    output.Store(int2(id.xy), colour);
}

// Barycentrics, primitive index and instance ID in one image. Each goes to its own channel so a
// backend that reports one of them correctly and another not at all still fails visibly.
[numthreads(8, 8, 1)]
void main_trace_attributes_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    AGFXRaytracingAccelerationStructure tlas =
        AGFXRaytracingAccelerationStructure::Create(g_Constants.tlas);
    AGFXRWTexture2D<float4> output = AGFXRWTexture2D<float4>::Create(g_Constants.output);

    RayDesc ray = PrimaryRay(id.xy, g_Constants.width, g_Constants.height);

    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(tlas.Resource(), RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    while (q.Proceed()) {}

    float4 colour = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        const float2 bary = q.CommittedTriangleBarycentrics();
        // The indices are scaled by a small constant rather than normalised by a count the shader
        // does not know: scenes here stay well under 8 primitives and 8 instances, and a fixed
        // scale keeps the same index the same colour across tests with different scene sizes.
        colour = float4(bary.x, bary.y, float(q.CommittedPrimitiveIndex()) / 8.0f, 1.0f);
        colour.b += float(q.CommittedInstanceID()) / 8.0f;
    }

    output.Store(int2(id.xy), colour);
}

// Procedural (AABB) geometry. RayQuery does not intersect AABBs for you — the candidate has to be
// committed explicitly — so this reports a hit only if the traversal surfaced the procedural
// candidate at all, which is exactly what the AABB BLAS tests are checking.
[numthreads(8, 8, 1)]
void main_trace_aabb_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    AGFXRaytracingAccelerationStructure tlas =
        AGFXRaytracingAccelerationStructure::Create(g_Constants.tlas);
    AGFXRWTexture2D<float4> output = AGFXRWTexture2D<float4>::Create(g_Constants.output);

    RayDesc ray = PrimaryRay(id.xy, g_Constants.width, g_Constants.height);

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas.Resource(), RAY_FLAG_NONE, 0xFF, ray);

    // Commit the candidate at a fixed T. The box is a slab in z, so a constant hit distance is
    // enough to distinguish "traversal reached the AABB" from "it did not".
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            q.CommitProceduralPrimitiveHit(2.0f);
        }
    }

    float4 colour = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
        colour.rgb = float3(0.0f, 1.0f, float(q.CommittedPrimitiveIndex()) / 8.0f);
    }

    output.Store(int2(id.xy), colour);
}

// Opacity. Without FORCE_OPAQUE the traversal reports non-opaque triangles as candidates the shader
// must accept or reject; this one rejects them, so geometry flagged non-opaque disappears from the
// image while opaque geometry stays. That difference is the entire point of the opacity tests.
[numthreads(8, 8, 1)]
void main_trace_opacity_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    AGFXRaytracingAccelerationStructure tlas =
        AGFXRaytracingAccelerationStructure::Create(g_Constants.tlas);
    AGFXRWTexture2D<float4> output = AGFXRWTexture2D<float4>::Create(g_Constants.output);

    RayDesc ray = PrimaryRay(id.xy, g_Constants.width, g_Constants.height);

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas.Resource(), RAY_FLAG_NONE, 0xFF, ray);

    uint rejected = 0;
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            rejected++; // Deliberately not committed: treat non-opaque as fully transparent.
        }
    }

    // Blue records that a non-opaque candidate was seen and skipped, so a build that silently drops
    // the opaque flag (reporting everything as opaque) differs from one that honours it.
    float4 colour = float4(0.0f, 0.0f, rejected > 0 ? 1.0f : 0.0f, 1.0f);
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        colour.r = 1.0f;
        colour.g = EncodeDistance(q.CommittedRayT());
    }

    output.Store(int2(id.xy), colour);
}
