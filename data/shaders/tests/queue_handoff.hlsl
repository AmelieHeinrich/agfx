//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Create Time: 2026-07-21 00:00:00
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: the render-queue half of the cross-queue handoff tests ("copy queue to render
// queue", "compute queue to render queue"). Another queue produces a texture, signals a fence, and
// the graphics queue waits on it and draws with this pair.
//
// The draw is a fullscreen triangle sampling that texture straight through, so whatever the
// producing queue wrote is what lands in the golden: if the graphics queue starts before the
// producer's writes are visible, the sample comes back as undefined or clear-colored content rather
// than the seeded pattern. Sampling (rather than blitting) is deliberate — it forces the texture
// through a shader read on the consuming queue, which is the transition the handoff has to make
// legal.

#include "data/shaders/agfx.h"

struct QueueHandoffPushConstants
{
    ResourceHandle source;    // AGFXTexture2D, read-only: what the producing queue wrote.
    ResourceHandle samplerId; // AGFXSampler.
    uint padding0;
    uint padding1;
};

AGFX_PUSH_CONSTANTS(QueueHandoffPushConstants, g_Constants);

struct vs_out
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// Oversized single triangle covering the whole viewport: no vertex buffer, and no interior edge
// that could show up as a seam in the golden the way a two-triangle quad can.
vs_out main_vs(uint vertexID : SV_VertexID)
{
    const float2 uvs[3] = { float2(0.0f, 0.0f), float2(2.0f, 0.0f), float2(0.0f, 2.0f) };

    vs_out output;
    output.uv = uvs[vertexID];
    output.position = float4(uvs[vertexID] * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return output;
}

float4 main_ps(vs_out input) : SV_Target0
{
    AGFXTexture2D<float4> src = AGFXTexture2D<float4>::Create(g_Constants.source);
    AGFXSampler smp = AGFXSampler::Create(g_Constants.samplerId);

    return src.SampleLevel(smp, input.uv, 0.0f);
}
