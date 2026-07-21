//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: groupshared memory plus GroupMemoryBarrierWithGroupSync. Each group cooperatively
// loads its slice, reverses it through groupshared storage, then runs a log-step reduction over the
// same array. Both the reversal and the reduction depend on the barrier actually synchronizing the
// group — without it, threads read stale or half-written groupshared values and the golden diverges.

#include "data/shaders/agfx.h"

#define GROUP_SIZE 64

struct SharedMemoryPushConstants
{
    ResourceHandle readBuffer; // AGFX_BUFFER_VIEW_TYPE_RAW, read-only: the input words.
    ResourceHandle rwBuffer;   // AGFX_BUFFER_VIEW_TYPE_RAW, writeable: reversed slice + per-group sum.
    uint elementCount;
    uint padding;
};

AGFX_PUSH_CONSTANTS(SharedMemoryPushConstants, g_Constants);

groupshared uint g_Scratch[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID, uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    AGFXByteAddressBuffer src = AGFXByteAddressBuffer::Create(g_Constants.readBuffer);
    AGFXRWByteAddressBuffer dst = AGFXRWByteAddressBuffer::Create(g_Constants.rwBuffer);

    const uint index = id.x;
    g_Scratch[groupIndex] = (index < g_Constants.elementCount) ? src.Load(index * 4u) : 0u;

    GroupMemoryBarrierWithGroupSync();

    // Reversal: every thread reads a word another thread wrote, so a missing barrier is fatal.
    const uint reversed = g_Scratch[GROUP_SIZE - 1u - groupIndex];
    if (index < g_Constants.elementCount) {
        dst.Store(index * 4u, reversed);
    }

    // Log-step reduction over the same groupshared array, barriered on every step.
    for (uint stride = GROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        GroupMemoryBarrierWithGroupSync();
        if (groupIndex < stride) {
            g_Scratch[groupIndex] += g_Scratch[groupIndex + stride];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // One sum per group, parked past the reversed slice.
    if (groupIndex == 0u) {
        dst.Store((g_Constants.elementCount + groupId.x) * 4u, g_Scratch[0]);
    }
}
