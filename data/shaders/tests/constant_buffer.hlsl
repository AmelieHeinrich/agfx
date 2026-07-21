//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: exercises AGFX_BUFFER_VIEW_TYPE_CONSTANT. The parameters driving the written pattern
// live in a constant buffer view rather than in push constants, so a backend that mis-binds the
// constant view (wrong offset, wrong descriptor type) produces zeros or garbage parameters and the
// golden diverges everywhere rather than at a single word.

#include "data/shaders/agfx.h"

struct Parameters
{
    uint elementCount;
    uint seed;
    uint multiplier;
    uint bias;
    float4 vector; // Present so the view has to honor float4 alignment past the leading uints.
};

struct ConstantPushConstants
{
    ResourceHandle parameters; // AGFX_BUFFER_VIEW_TYPE_CONSTANT, read-only
    ResourceHandle rwBuffer;   // AGFX_BUFFER_VIEW_TYPE_RAW, writeable
    uint elementCount;
    uint padding;
};

AGFX_PUSH_CONSTANTS(ConstantPushConstants, g_Constants);

[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    ConstantBuffer<Parameters> params = ResourceDescriptorHeap[g_Constants.parameters];
    AGFXRWByteAddressBuffer dst = AGFXRWByteAddressBuffer::Create(g_Constants.rwBuffer);

    const uint index = id.x;
    if (index >= g_Constants.elementCount) {
        return;
    }

    // Every field of the constant buffer participates, so a partially-bound view still fails.
    const uint scalar = params.seed + index * params.multiplier + params.bias;
    const uint packed = uint(params.vector.x * 100.0f) | (uint(params.vector.w * 100.0f) << 16);

    dst.Store2(index * 8u, uint2(scalar, packed ^ params.elementCount));
}
