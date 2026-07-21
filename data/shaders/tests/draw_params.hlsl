//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: an indexed, instanced draw whose vertex fetch and color both depend on parameters
// agfxRenderPassDrawIndexed takes but that no other test exercises directly -- firstIndex,
// vertexOffset, and firstInstance (instanceCount comes along for free since two instances are what
// makes firstInstance observable at all).
//
// SV_VertexID here is *post*-index-buffer and post-vertexOffset, matching the D3D12/Metal contract:
// the hardware reads indexBuffer[firstIndex + i], adds vertexOffset, and that sum is what the vertex
// shader sees. The vertex buffer's first several entries are a degenerate point (0,0); any of
// firstIndex or vertexOffset being dropped routes the fetch back to that point for all three
// vertices of the triangle, which collapses it to zero area rather than producing a subtly wrong
// shape -- a dropped offset is invisible rather than merely inaccurate.
//
// SV_InstanceID is 0-based per draw call regardless of firstInstance -- that is the D3D12 spec's
// contract (StartInstanceLocation offsets per-instance-rate vertex fetches, not the system value
// itself), and AGFX's DXIL->Metal path preserves it. It is used twice: once to look up this
// instance's color from a bindless buffer, and once to shift the triangle sideways so the two
// instances do not overlap. See the host-side test for what that makes observable.

#include "data/shaders/agfx.h"

struct Vertex
{
    float2 position;
    float2 padding;
};

struct DrawParamsPushConstants
{
    ResourceHandle vertices;       // AGFX_BUFFER_VIEW_TYPE_STRUCTURED, stride sizeof(Vertex).
    ResourceHandle instanceColors; // AGFX_BUFFER_VIEW_TYPE_STRUCTURED, stride sizeof(float4).
    float instanceSpacing;         // NDC x added per SV_InstanceID, so instances do not overlap.
    uint padding0;
};

AGFX_PUSH_CONSTANTS(DrawParamsPushConstants, g_Constants);

struct vs_out
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

vs_out main_vs(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    AGFXStructuredBuffer<Vertex> vertices = AGFXStructuredBuffer<Vertex>::Create(g_Constants.vertices);
    AGFXStructuredBuffer<float4> instanceColors = AGFXStructuredBuffer<float4>::Create(g_Constants.instanceColors);

    const Vertex vertex = vertices.Load(vertexID);
    const float2 position = vertex.position + float2(float(instanceID) * g_Constants.instanceSpacing, 0.0f);

    vs_out output;
    output.position = float4(position, 0.0f, 1.0f);
    output.color = instanceColors.Load(instanceID);
    return output;
}

float4 main_ps(vs_out input) : SV_Target0
{
    return input.color;
}
