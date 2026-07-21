//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: a single accumulate pass, dispatched several times in a row with a UAV barrier
// between each. Every pass reads what the previous one wrote, so the result is only correct if the
// barrier really orders the dispatches — a backend that lets them overlap produces a value below
// the expected one.

#include "data/shaders/agfx.h"

struct MultiDispatchPushConstants
{
    ResourceHandle rwBuffer; // AGFX_BUFFER_VIEW_TYPE_RAW, writeable
    uint elementCount;
    uint passIndex;          // Which iteration this dispatch is; folded into the value.
    uint padding;
};

AGFX_PUSH_CONSTANTS(MultiDispatchPushConstants, g_Constants);

[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXRWByteAddressBuffer dst = AGFXRWByteAddressBuffer::Create(g_Constants.rwBuffer);

    const uint index = id.x;
    if (index >= g_Constants.elementCount) {
        return;
    }

    // Read-modify-write: the dependency on the previous pass is what makes the barrier observable.
    const uint previous = dst.Load(index * 4u);
    dst.Store(index * 4u, previous * 2u + g_Constants.passIndex + index);
}
