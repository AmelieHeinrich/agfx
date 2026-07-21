//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: the shared vertex/fragment pair behind the rasterizer-state tests (wireframe,
// topology, cull mode / winding, fragment discard, push-constant color). Every one of those varies
// only pipeline state, so they all drive this one shader and differ purely in the agfxRenderPipeline
// they build — which is exactly what makes a difference in the golden attributable to the state and
// not to the shader.
//
// Geometry comes from SV_VertexID against a hardcoded table rather than a vertex buffer, so these
// tests stay independent of buffer upload and index-type handling (those are covered by
// indexed.hlsl). The table holds two triangles wound in *opposite* directions and placed on
// opposite halves of the target, so a single draw of 6 vertices renders one CW and one CCW
// triangle: with culling on, exactly one of them survives, and which one it is tells you whether
// frontFace/cullMode were honored. A backend that ignores winding draws both; one that inverts it
// draws the other. Neither can be mistaken for a pass.

#include "data/shaders/agfx.h"

struct RasterPushConstants
{
    float4 color;    // Fragment color. The push-constant test varies this; others leave it at white.
    float discardX;  // Fragments with position.x above this NDC threshold are clipped. >= 1 keeps all.
    uint useVertexColor; // Non-zero: shade from the per-vertex color instead of `color`.
    uint padding0;
    uint padding1;
};

AGFX_PUSH_CONSTANTS(RasterPushConstants, g_Constants);

struct vs_out
{
    float4 position : SV_Position;
    float3 color : COLOR0;
    float2 ndc : TEXCOORD0; // Passed through so the discard test can threshold on it.
};

vs_out main_vs(uint vertexID : SV_VertexID)
{
    // Vertices 0-2: left-half triangle, counter-clockwise in NDC.
    // Vertices 3-5: right-half triangle, clockwise in NDC (same shape, reversed order).
    const float2 positions[6] = {
        float2(-0.9f, -0.8f), float2(-0.1f, -0.8f), float2(-0.5f,  0.8f),
        float2( 0.1f, -0.8f), float2( 0.5f,  0.8f), float2( 0.9f, -0.8f)
    };
    // Distinct per-corner colors: a reordered or dropped vertex shows up as a different gradient,
    // not just a differently-shaped blob.
    const float3 colors[6] = {
        float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f),
        float3(1.0f, 1.0f, 0.0f), float3(0.0f, 1.0f, 1.0f), float3(1.0f, 0.0f, 1.0f)
    };

    vs_out output;
    output.position = float4(positions[vertexID], 0.0f, 1.0f);
    output.color = colors[vertexID];
    output.ndc = positions[vertexID];
    return output;
}

float4 main_ps(vs_out input) : SV_Target0
{
    // clip() rather than an early `return`: the discard test is asserting that fragments are
    // actually killed, and a returned transparent color would still write depth/color and pass
    // for the wrong reason.
    if (input.ndc.x > g_Constants.discardX) {
        clip(-1.0f);
    }

    const float3 rgb = g_Constants.useVertexColor ? input.color : g_Constants.color.rgb;
    return float4(rgb, 1.0f);
}
