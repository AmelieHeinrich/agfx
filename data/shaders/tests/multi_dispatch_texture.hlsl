//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: the texture counterpart of multi_dispatch.hlsl. Same idea — one accumulate pass
// dispatched several times in a row with a UAV barrier between each, every pass reading what the
// previous one wrote — but over a storage texture rather than a buffer, which is the path a UAV
// *texture* barrier has to order. A backend that lets the dispatches overlap lands somewhere below
// the golden.
//
// It lives in its own file rather than as a second entry point in multi_dispatch.hlsl because
// AGFX_PUSH_CONSTANTS binds register(b0), so one file gets one push-constant block.

#include "data/shaders/agfx.h"

struct MultiDispatchTexturePushConstants
{
    ResourceHandle rwTexture; // AGFXRWTexture2D, writeable.
    uint width;
    uint height;
    uint passIndex;           // Which iteration this dispatch is; folded into the value.
};

AGFX_PUSH_CONSTANTS(MultiDispatchTexturePushConstants, g_Constants);

[numthreads(8, 8, 1)]
void main_texture_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXRWTexture2D<float4> dst = AGFXRWTexture2D<float4>::Create(g_Constants.rwTexture);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    // Read-modify-write: the dependency on the previous pass is what makes the barrier observable.
    // Each pass folds in its own index plus a position term, then halves the accumulator — the
    // halving keeps four passes inside [0,1] on an UNORM target, so the result stays representable
    // instead of saturating to white while every pass's contribution is still visible.
    const float4 previous = dst.Load(int2(id.xy));
    const float2 uv = (float2(id.xy) + 0.5f) / float2(g_Constants.width, g_Constants.height);
    const float weight = float(g_Constants.passIndex + 1u) * 0.25f;

    dst.Store(int2(id.xy), float4((previous.r + uv.x * weight) * 0.5f,
                                  (previous.g + uv.y * weight) * 0.5f,
                                  (previous.b + weight) * 0.5f,
                                  1.0f));
}
