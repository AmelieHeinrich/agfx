//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: the vertex/fragment pair behind the depth-state tests -- depth clamp, the eight depth
// comparison functions, and depth write disabled.
//
// The scene is three columns drawn by a single 18-vertex draw, each at its own depth taken from a
// push constant. That one knob is what lets the same shader serve all three test families:
//
//   - depth test:  depths straddle the cleared value (nearer / equal / farther), so which columns
//                  survive identifies the comparison function.
//   - depth clamp: depths straddle the *view volume* (before near / inside / beyond far), so which
//                  columns survive identifies whether out-of-range fragments were clipped or clamped.
//   - depth write: the columns are drawn twice at different depths; whether the second pass sees the
//                  first pass's depth identifies whether writes landed.
//
// Geometry comes from SV_VertexID against a table rather than a vertex buffer, as in raster.hlsl, so
// these tests stay independent of buffer upload. The columns are non-touching with clear background
// between them, so a column that survives when it should not is never hidden behind one that should.

#include "data/shaders/agfx.h"

#define AGFX_TEST_DEPTH_COLUMNS 3

struct DepthPushConstants
{
    // Per-column clip-space z. Only .xyz are used; .w pads to the float4 the block is laid out in.
    // Held as a float4 rather than a float[3] because an HLSL float array in a cbuffer pads each
    // element to 16 bytes, which would not match the host struct.
    float4 depths;
    // Multiplied into every column's color. The depth-write tests use it to tell the two passes
    // apart; everything else leaves it white.
    float4 tint;
};

AGFX_PUSH_CONSTANTS(DepthPushConstants, g_Constants);

// Matches IndirectColumnCorner in indirect.hlsl, deliberately: the depth goldens and the indirect
// goldens then share a column layout and can be compared by eye.
float2 DepthColumnCorner(uint column, uint corner)
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

// One primary per column, so any two differ in every channel and a surviving column is identifiable
// on its own rather than only by position.
float3 DepthColumnColor(uint column)
{
    const float3 colors[AGFX_TEST_DEPTH_COLUMNS] = {
        float3(1.0f, 0.0f, 0.0f), // column 0: red
        float3(0.0f, 1.0f, 0.0f), // column 1: green
        float3(0.0f, 0.0f, 1.0f)  // column 2: blue
    };
    return colors[min(column, (uint)(AGFX_TEST_DEPTH_COLUMNS - 1))];
}

struct vs_out
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

// Two triangles per quad, sharing the 0-2 diagonal.
static const uint kQuadCorners[6] = {0, 1, 2, 0, 2, 3};

// All three columns in one draw of 18 vertices. One draw rather than three matters for the depth
// *test* family: the columns must not depend on each other's results, and with depth writes off a
// single draw makes each column an independent probe against the cleared depth value.
vs_out main_vs(uint vertexID : SV_VertexID)
{
    const uint column = min(vertexID / 6, (uint)(AGFX_TEST_DEPTH_COLUMNS - 1));
    const uint corner = kQuadCorners[vertexID % 6];

    vs_out output;
    output.position = float4(DepthColumnCorner(column, corner), g_Constants.depths[column], 1.0f);
    output.color = DepthColumnColor(column) * g_Constants.tint.rgb;
    return output;
}

// A full-screen quad at a uniform depth (depths.x), used to lay down a known depth floor before the
// probe draw. This is what makes the GREATER-family comparisons testable: it puts the depth buffer
// at a value the probes can be on both sides of.
vs_out main_vs_fullscreen(uint vertexID : SV_VertexID)
{
    const float2 corners[4] = {
        float2(-1.0f, -1.0f), float2(1.0f, -1.0f), float2(1.0f, 1.0f), float2(-1.0f, 1.0f)
    };

    vs_out output;
    output.position = float4(corners[kQuadCorners[vertexID % 6]], g_Constants.depths.x, 1.0f);
    output.color = g_Constants.tint.rgb;
    return output;
}

float4 main_ps(vs_out input) : SV_Target0
{
    return float4(input.color, 1.0f);
}
