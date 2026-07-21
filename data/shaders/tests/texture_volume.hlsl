//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: the layered counterparts of texture_ops.hlsl's 2D entry points, for 2D array and 3D
// textures. Kept in its own file rather than added to texture_ops.hlsl because the push constant
// block needs a fifth field (the slice count) and the existing 2D tests mirror that struct
// byte-for-byte on the host side.
//
// Every write tints its output per slice, so a shader that ignores the z coordinate — writing every
// slice with the same content, or collapsing them all onto slice 0 — produces a visibly uniform
// stack instead of a graded one. The load entry points then read the *reversed* slice, so slice
// addressing has to be right in both directions to reproduce the golden.

#include "data/shaders/agfx.h"

struct VolumePushConstants
{
    ResourceHandle source;      // AGFXTexture2DArray/AGFXTexture3D, read-only. Unused by the writes.
    ResourceHandle destination; // AGFXRWTexture2DArray/AGFXRWTexture3D, writeable.
    uint width;
    uint height;
    uint sliceCount;            // Array layers for the 2D array case, depth for the 3D case.
    uint padding0;
    uint padding1;
    uint padding2;
};

AGFX_PUSH_CONSTANTS(VolumePushConstants, g_Constants);

// The same horizontal/vertical/diagonal structure texture_ops.hlsl uses, scaled down per slice so
// the stack has a gradient along z on top of the in-slice pattern.
float4 SlicePattern(uint3 texel, uint width, uint height, uint sliceCount)
{
    const float u = float(texel.x) / float(max(width - 1u, 1u));
    const float v = float(texel.y) / float(max(height - 1u, 1u));
    const float checker = ((texel.x / 8u) + (texel.y / 8u)) % 2u ? 1.0f : 0.0f;

    // Slice 0 is fully dark, the last slice fully bright: an off-by-one in slice addressing shifts
    // the whole gradient rather than perturbing it subtly.
    const float slice = float(texel.z + 1u) / float(sliceCount);
    return float4(u * slice, v * slice, checker * slice, 1.0f);
}

[numthreads(8, 8, 1)]
void main_write_array_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXRWTexture2DArray<float4> dst = AGFXRWTexture2DArray<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height || id.z >= g_Constants.sliceCount) {
        return;
    }

    dst.Store(int3(id), SlicePattern(id, g_Constants.width, g_Constants.height, g_Constants.sliceCount));
}

[numthreads(8, 8, 1)]
void main_load_array_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXTexture2DArray<float4> src = AGFXTexture2DArray<float4>::Create(g_Constants.source);
    AGFXRWTexture2DArray<float4> dst = AGFXRWTexture2DArray<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height || id.z >= g_Constants.sliceCount) {
        return;
    }

    // Mirror in x and reverse in z, so both the in-slice coordinate and the slice index have to be
    // handled correctly for the result to match.
    const int3 source = int3(g_Constants.width - 1u - id.x, id.y, g_Constants.sliceCount - 1u - id.z);
    const float4 texel = src.Load(source);

    // Swizzle on top so a channel-order mixup is caught too.
    dst.Store(int3(id), float4(texel.g, texel.b, texel.r, texel.a));
}

[numthreads(8, 8, 1)]
void main_write_3d_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXRWTexture3D<float4> dst = AGFXRWTexture3D<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height || id.z >= g_Constants.sliceCount) {
        return;
    }

    dst.Store(id, SlicePattern(id, g_Constants.width, g_Constants.height, g_Constants.sliceCount));
}

[numthreads(8, 8, 1)]
void main_load_3d_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXTexture3D<float4> src = AGFXTexture3D<float4>::Create(g_Constants.source);
    AGFXRWTexture3D<float4> dst = AGFXRWTexture3D<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height || id.z >= g_Constants.sliceCount) {
        return;
    }

    // AGFXTexture3D::Load takes (x, y, z, mip), unlike the array flavor's (x, y, slice).
    const int3 source = int3(g_Constants.width - 1u - id.x, id.y, g_Constants.sliceCount - 1u - id.z);
    const float4 texel = src.Load(int4(source, 0));

    dst.Store(id, float4(texel.g, texel.b, texel.r, texel.a));
}
