//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: draws an indexed quad by pulling vertices out of a bindless structured buffer with
// SV_VertexID — after the index buffer has already remapped it. The four corners carry four
// different colors and sit at four different positions, so a wrong index type (u16 read as u32 or
// vice versa), a wrong firstIndex, or a dropped vertexOffset picks up the wrong corner and shows
// up as a differently-colored or differently-shaped quad rather than as a subtle shift.

#include "data/shaders/agfx.h"

struct Vertex
{
    float2 position;
    float2 padding;
    float4 color;
};

struct IndexedPushConstants
{
    ResourceHandle vertices; // AGFX_BUFFER_VIEW_TYPE_STRUCTURED, read-only, stride sizeof(Vertex)
    uint padding0;
    uint padding1;
    uint padding2;
};

AGFX_PUSH_CONSTANTS(IndexedPushConstants, g_Constants);

struct vs_out
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

vs_out main_vs(uint vertexID : SV_VertexID)
{
    AGFXStructuredBuffer<Vertex> vertices = AGFXStructuredBuffer<Vertex>::Create(g_Constants.vertices);

    // vertexID is post-index-buffer: the index buffer decides which of the four corners this
    // invocation actually fetches, which is the whole point of the test.
    const Vertex vertex = vertices.Load(vertexID);

    vs_out output;
    output.position = float4(vertex.position, 0.0f, 1.0f);
    output.color = vertex.color;
    return output;
}

float4 main_ps(vs_out input) : SV_Target0
{
    return input.color;
}
