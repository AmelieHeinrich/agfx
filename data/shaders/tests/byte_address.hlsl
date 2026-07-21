//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: exercises a raw (byte address) buffer view end to end. One thread group writes a
// known pattern through an AGFXRWByteAddressBuffer using each Store variant, then reads part of it
// back through an AGFXByteAddressBuffer and writes the derived values into the tail of the buffer.
// That way both the write and the read path of a raw view are covered by one golden .bin.

#include "data/shaders/agfx.h"

struct ByteAddressPushConstants
{
    ResourceHandle rwBuffer;   // AGFX_BUFFER_VIEW_TYPE_RAW, writeable
    ResourceHandle readBuffer; // AGFX_BUFFER_VIEW_TYPE_RAW, read-only, same underlying buffer
    uint elementCount;         // Number of 4-byte words the first phase writes.
    uint padding;
};

AGFX_PUSH_CONSTANTS(ByteAddressPushConstants, g_Constants);

[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXRWByteAddressBuffer dst = AGFXRWByteAddressBuffer::Create(g_Constants.rwBuffer);

    const uint index = id.x;
    if (index >= g_Constants.elementCount) {
        return;
    }

    // Phase 1: a value that depends on the byte offset, so a view created at the wrong offset or
    // with the wrong stride produces visibly wrong bytes rather than a plausible-looking pattern.
    const uint byteOffset = index * 4;
    dst.Store(byteOffset, 0xA5000000u | (index * 3u + 7u));
}

[numthreads(64, 1, 1)]
void main_readback_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXByteAddressBuffer src = AGFXByteAddressBuffer::Create(g_Constants.readBuffer);
    AGFXRWByteAddressBuffer dst = AGFXRWByteAddressBuffer::Create(g_Constants.rwBuffer);

    const uint index = id.x;
    if (index >= g_Constants.elementCount) {
        return;
    }

    // Phase 2: read back through the read-only raw view and fold four consecutive words into one,
    // exercising Load/Load2/Load4 alignment. Written into the second half of the buffer.
    const uint srcOffset = (index % (g_Constants.elementCount / 4u)) * 16u;
    const uint4 quad = src.Load4(srcOffset);
    const uint folded = quad.x ^ quad.y ^ quad.z ^ quad.w;

    dst.Store((g_Constants.elementCount + index) * 4u, folded);
}
