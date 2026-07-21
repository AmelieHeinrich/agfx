//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: the vertex/fragment pair behind the alpha-blending tests -- one per agfxBlendOperation
// and one per agfxBlendFactor, plus the canonical src-alpha/one-minus-src-alpha case.
//
// Blending needs two things the other draw tests do not: a *destination* that is already in the
// attachment, and a source drawn over it. So every blend test is two draws --
//
//   1. main_vs: three opaque columns, each a different color *and a different alpha*, over a clear
//      color that is itself transparent black.
//   2. main_vs_fullscreen: one translucent quad covering everything, drawn with the blend state
//      under test.
//
// -- which gives four distinct destinations (three columns plus the background) in a single image.
// That matters because most blend factors only differ from each other against *some* destinations:
// DST_COLOR and ONE are indistinguishable over black, and DST_ALPHA is indistinguishable from ONE
// wherever the destination is opaque. Varying both destination color and destination alpha across
// the image is what makes all ten factor goldens differ from one another.
//
// The source alpha is deliberately not 0.5. At 0.5, SRC_ALPHA and ONE_MINUS_SRC_ALPHA are the same
// number and their goldens come out identical -- which would quietly test nothing.

#include "data/shaders/agfx.h"

#define AGFX_TEST_BLEND_COLUMNS 3

struct BlendPushConstants
{
    // The full-screen source quad's color, alpha included. Unused by the column draw.
    float4 color;
    // Per-column destination brightness (xyz) -- kept dim so that additive results stay inside the
    // UNORM range instead of clamping, which would collapse distinct factors onto the same golden.
    float4 columnScale;
    // Per-column destination alpha (xyz). Varied so the DST_ALPHA family has something to read.
    float4 columnAlpha;
};

AGFX_PUSH_CONSTANTS(BlendPushConstants, g_Constants);

// Matches DepthColumnCorner in depth.hlsl and IndirectColumnCorner in indirect.hlsl, deliberately:
// the blend goldens then share a column layout with the depth and indirect ones.
float2 BlendColumnCorner(uint column, uint corner)
{
    const float x0 = -0.85f + 0.6f * (float)column;
    const float x1 = x0 + 0.5f;
    const float2 corners[4] = {
        float2(x0, -0.5f), // 0: bottom left
        float2(x1, -0.5f), // 1: bottom right
        float2(x1,  0.5f), // 2: top right
        float2(x0,  0.5f)  // 3: top left
    };
    return corners[min(corner, 3u)];
}

float3 BlendColumnColor(uint column)
{
    const float3 colors[AGFX_TEST_BLEND_COLUMNS] = {
        float3(1.0f, 0.0f, 0.0f), // column 0: red
        float3(0.0f, 1.0f, 0.0f), // column 1: green
        float3(0.0f, 0.0f, 1.0f)  // column 2: blue
    };
    return colors[min(column, (uint)(AGFX_TEST_BLEND_COLUMNS - 1))];
}

struct vs_out
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

static const uint kQuadCorners[6] = {0, 1, 2, 0, 2, 3};

// The destination draw: all three columns in one draw of 18 vertices, blending disabled.
vs_out main_vs(uint vertexID : SV_VertexID)
{
    const uint column = min(vertexID / 6, (uint)(AGFX_TEST_BLEND_COLUMNS - 1));
    const uint corner = kQuadCorners[vertexID % 6];

    vs_out output;
    output.position = float4(BlendColumnCorner(column, corner), 0.0f, 1.0f);
    output.color = float4(BlendColumnColor(column) * g_Constants.columnScale[column],
                          g_Constants.columnAlpha[column]);
    return output;
}

// The source draw: one quad over the whole target, carrying a uniform color and alpha.
vs_out main_vs_fullscreen(uint vertexID : SV_VertexID)
{
    const float2 corners[4] = {
        float2(-1.0f, -1.0f), float2(1.0f, -1.0f), float2(1.0f, 1.0f), float2(-1.0f, 1.0f)
    };

    vs_out output;
    output.position = float4(corners[kQuadCorners[vertexID % 6]], 0.0f, 1.0f);
    output.color = g_Constants.color;
    return output;
}

// Alpha is passed through rather than forced to 1: it is both the source alpha the SRC_ALPHA family
// reads and the destination alpha the DST_ALPHA family will read on the next draw.
float4 main_ps(vs_out input) : SV_Target0
{
    return input.color;
}
