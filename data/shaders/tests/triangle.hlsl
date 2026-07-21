//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: draws one hardcoded RGB gradient triangle from SV_VertexID, with no vertex buffer
// and no bindless resources at all. Deliberately the smallest thing that can produce a stable,
// non-trivial golden image.

#include "data/shaders/agfx.h"

struct vs_out
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

vs_out main_vs(uint vertexID : SV_VertexID)
{
    // A single triangle covering most of the viewport, one primary color per corner.
    const float2 positions[3] = {
        float2( 0.0f,  0.8f),
        float2( 0.8f, -0.8f),
        float2(-0.8f, -0.8f)
    };
    const float3 colors[3] = {
        float3(1.0f, 0.0f, 0.0f),
        float3(0.0f, 1.0f, 0.0f),
        float3(0.0f, 0.0f, 1.0f)
    };

    vs_out output;
    output.position = float4(positions[vertexID], 0.0f, 1.0f);
    output.color = colors[vertexID];
    return output;
}

float4 main_ps(vs_out input) : SV_Target0
{
    return float4(input.color, 1.0f);
}
