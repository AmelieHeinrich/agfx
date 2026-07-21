//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: the compute-side texture entry points. main_write_cs writes a procedural pattern
// into an AGFXRWTexture2D; main_load_cs reads an AGFXTexture2D with Load() (no sampler, so texel
// addressing is tested without filtering in the way) and writes a transform of it into a second
// RW texture; main_hdr_cs writes deliberately out-of-[0,1] values so a target that is secretly LDR
// clamps and fails the comparison.

#include "data/shaders/agfx.h"

struct TexturePushConstants
{
    ResourceHandle source;      // AGFXTexture2D, read-only. Unused by main_write_cs/main_hdr_cs.
    ResourceHandle destination; // AGFXRWTexture2D, writeable.
    uint width;
    uint height;
};

AGFX_PUSH_CONSTANTS(TexturePushConstants, g_Constants);

// A pattern with strong horizontal, vertical and diagonal structure, so a transposed, flipped or
// half-written destination is obvious rather than plausible.
float4 Pattern(uint2 texel, uint width, uint height)
{
    const float u = float(texel.x) / float(max(width - 1u, 1u));
    const float v = float(texel.y) / float(max(height - 1u, 1u));
    const float checker = ((texel.x / 8u) + (texel.y / 8u)) % 2u ? 1.0f : 0.0f;
    return float4(u, v, checker, 1.0f);
}

[numthreads(8, 8, 1)]
void main_write_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXRWTexture2D<float4> dst = AGFXRWTexture2D<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    dst.Store(int2(id.xy), Pattern(id.xy, g_Constants.width, g_Constants.height));
}

[numthreads(8, 8, 1)]
void main_load_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXTexture2D<float4> src = AGFXTexture2D<float4>::Create(g_Constants.source);
    AGFXRWTexture2D<float4> dst = AGFXRWTexture2D<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    // Sample the mirrored texel: a Load() that ignores the coordinate, or a view bound to the wrong
    // texture, yields something that isn't this mirror.
    const int2 mirrored = int2(g_Constants.width - 1u - id.x, id.y);
    const float4 source = src.Load(mirrored);

    // Swizzle on top of the mirror so a channel-order mixup is caught too.
    dst.Store(int2(id.xy), float4(source.g, source.b, source.r, source.a));
}

[numthreads(8, 8, 1)]
void main_hdr_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXRWTexture2D<float4> dst = AGFXRWTexture2D<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    const float4 base = Pattern(id.xy, g_Constants.width, g_Constants.height);

    // Every channel lands in [1, 9]: comfortably past the 1.0 an LDR target would clamp to, which
    // is what this test is asserting, while staying inside a range the RGBE golden can round-trip.
    // (Radiance HDR shares one 8-bit exponent across RGB and cannot store negatives, so a texel
    // mixing 0.0 with 95.0 -- or holding a negative -- degrades badly on the way to disk and the
    // comparison fails against our own output.)
    dst.Store(int2(id.xy), float4(base.r * 8.0f + 1.0f, base.g * 8.0f + 1.0f, base.b * 8.0f + 1.0f, 1.0f));
}
