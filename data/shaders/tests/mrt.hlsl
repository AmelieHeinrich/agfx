//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: multiple render targets, plus the compute pass that combines them.
//
// The fragment shader writes *different* content to SV_Target0 and SV_Target1 — a red ramp along x
// into the first, a green ramp along y into the second. Writing the same value to both would be a
// far weaker test: a backend that bound one attachment twice, or that broadcast one output to every
// attachment, would produce an identical result. With two different ramps, the sum is (x, y, 0), and
// swapping the attachments gives (y, x, 0), which differs everywhere off the diagonal.
//
// The ramps are built from integer pixel coordinates divided by 255 so every value lands exactly on
// an 8-bit code, which keeps the final comparison exact rather than a tolerance. The targets are 128
// wide, so the codes stay in 0..127 and the sum of the two channels cannot overflow.

#include "data/shaders/agfx.h"

struct MRTPushConstants
{
    ResourceHandle source0;     // AGFXTexture2D<float4>, the first attachment (compute pass only).
    ResourceHandle source1;     // AGFXTexture2D<float4>, the second attachment (compute pass only).
    ResourceHandle destination; // AGFXRWTexture2D<float4>, writeable (compute pass only).
    uint width;
    uint height;
    uint padding0;
    uint padding1;
    uint padding2;
};

AGFX_PUSH_CONSTANTS(MRTPushConstants, g_Constants);

struct mrt_out
{
    float4 color0 : SV_Target0;
    float4 color1 : SV_Target1;
};

float4 main_vs(uint vertexID : SV_VertexID) : SV_Position
{
    // The standard oversized fullscreen triangle: it covers all of NDC with one primitive and no
    // interior edge, so every pixel of both attachments is written.
    const float2 positions[3] = {
        float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)
    };
    return float4(positions[vertexID], 0.0f, 1.0f);
}

mrt_out main_ps(float4 position : SV_Position)
{
    // position.xy is the pixel center (x + 0.5), so the floor recovers the integer pixel index.
    const uint2 pixel = uint2(position.xy);

    mrt_out output;
    output.color0 = float4(float(pixel.x) / 255.0f, 0.0f, 0.0f, 1.0f);
    output.color1 = float4(0.0f, float(pixel.y) / 255.0f, 0.0f, 1.0f);
    return output;
}

// Sums the two attachments into the destination. Load() rather than a sampled read: this is a
// texel-for-texel combine, and involving a sampler would drag filtering and address modes into a
// test that is about the attachments.
[numthreads(8, 8, 1)]
void main_add_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXTexture2D<float4> source0 = AGFXTexture2D<float4>::Create(g_Constants.source0);
    AGFXTexture2D<float4> source1 = AGFXTexture2D<float4>::Create(g_Constants.source1);
    AGFXRWTexture2D<float4> dst = AGFXRWTexture2D<float4>::Create(g_Constants.destination);

    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    const float4 a = source0.Load(int2(id.xy));
    const float4 b = source1.Load(int2(id.xy));

    // Alpha is 1 in both attachments; summing it would clamp to 1 anyway, but taking it from one
    // source keeps the expected result trivially statable.
    dst.Store(int2(id.xy), float4(a.rgb + b.rgb, 1.0f));
}
