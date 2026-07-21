//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: a ping-pong dependency chain between a buffer UAV and a texture UAV, driven by
// alternating dispatches with a UAV barrier between every one.
//
// Each round trip: this dispatch reads the *other* resource's previous value and writes its own
// as (value + 1). Every write is a read the very next dispatch depends on, across a resource type
// switch each time -- so the chain only produces the analytically correct final value if
// agfxComputePassBufferUAVBarrier and agfxComputePassTextureUAVBarrier both actually order GPU-side
// reads after the writes they follow, rather than only reordering the command stream.
//
// This is deliberately not the same shape as multi_dispatch.hlsl (which chains a single buffer to
// itself): folding a texture into the chain means one barrier call being a no-op is not masked by
// the other, since each step exercises a different one.

#include "data/shaders/agfx.h"

struct UAVBarrierPushConstants
{
    ResourceHandle rwBuffer;  // AGFXRWStructuredBuffer<uint>, one element.
    ResourceHandle rwTexture; // AGFXRWTexture2D<float4>, value broadcast to every texel.
    uint width;
    uint height;
};

AGFX_PUSH_CONSTANTS(UAVBarrierPushConstants, g_Constants);

// texture = buffer[0] + 1, broadcast to every texel.
[numthreads(8, 8, 1)]
void main_texture_from_buffer_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    AGFXRWStructuredBuffer<uint> buf = AGFXRWStructuredBuffer<uint>::Create(g_Constants.rwBuffer);
    AGFXRWTexture2D<float4> tex = AGFXRWTexture2D<float4>::Create(g_Constants.rwTexture);

    const float value = float(buf.Load(0)) + 1.0f;
    tex.Store(int2(id.xy), float4(value, value, value, 1.0f));
}

// buffer[0] = texture[0,0] + 1.
[numthreads(1, 1, 1)]
void main_buffer_from_texture_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXRWStructuredBuffer<uint> buf = AGFXRWStructuredBuffer<uint>::Create(g_Constants.rwBuffer);
    AGFXRWTexture2D<float4> tex = AGFXRWTexture2D<float4>::Create(g_Constants.rwTexture);

    const uint value = (uint)(tex.Load(int2(0, 0)).r) + 1u;
    buf.Store(0, value);
}
