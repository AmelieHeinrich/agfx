//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: exercises structured buffer views. Phase 1 writes a per-element pattern through an
// AGFXRWStructuredBuffer<Element>; phase 2 reads those elements back through a read-only
// AGFXStructuredBuffer<Element> view of the same buffer and writes derived values into the second
// half. A view built with the wrong stride lands the pattern on the wrong element boundaries, so
// the golden diverges rather than merely shifting.

#include "data/shaders/agfx.h"

struct Element
{
    uint index;
    uint tag;
    float value;
    uint checksum;
};

struct StructuredPushConstants
{
    ResourceHandle rwBuffer;   // AGFX_BUFFER_VIEW_TYPE_STRUCTURED, writeable
    ResourceHandle readBuffer; // AGFX_BUFFER_VIEW_TYPE_STRUCTURED, read-only, same buffer
    uint elementCount;         // Elements written by phase 1; the buffer holds twice that.
    uint padding;
};

AGFX_PUSH_CONSTANTS(StructuredPushConstants, g_Constants);

[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXRWStructuredBuffer<Element> dst = AGFXRWStructuredBuffer<Element>::Create(g_Constants.rwBuffer);

    const uint index = id.x;
    if (index >= g_Constants.elementCount) {
        return;
    }

    Element element;
    element.index = index;
    element.tag = 0x5AA50000u | index;
    element.value = float(index) * 0.5f;
    element.checksum = element.index ^ element.tag;
    dst.Store(index, element);
}

[numthreads(64, 1, 1)]
void main_readback_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXStructuredBuffer<Element> src = AGFXStructuredBuffer<Element>::Create(g_Constants.readBuffer);
    AGFXRWStructuredBuffer<Element> dst = AGFXRWStructuredBuffer<Element>::Create(g_Constants.rwBuffer);

    const uint index = id.x;
    if (index >= g_Constants.elementCount) {
        return;
    }

    // Read the mirrored element so a view that silently indexes from the wrong base produces a
    // self-consistent-but-wrong second half.
    const Element source = src.Load(g_Constants.elementCount - 1u - index);

    Element element;
    element.index = source.index + g_Constants.elementCount;
    element.tag = ~source.tag;
    element.value = source.value * 2.0f;
    element.checksum = source.checksum;
    dst.Store(g_Constants.elementCount + index, element);
}
