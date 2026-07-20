/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-20 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// A small "for fun" compute post-process: hue-cycles the scene color over time, darkens
// toward the edges (vignette), and adds faint scanlines -- just enough to show a
// render-pass-produced texture being consumed and rewritten by a compute pass.

#include "data/shaders/agfx.h"

struct ColorFilterPushConstants {
    ResourceHandle sourceTex;
    ResourceHandle destTex;
    ResourceHandle pointSampler;
    float time;
    float2 screenSize;
};

AGFX_PUSH_CONSTANTS(ColorFilterPushConstants, g_Constants);

float3 RotateHue(float3 color, float angle) {
    float3x3 k = float3x3(0.299f, 0.587f, 0.114f,
                           0.299f, 0.587f, 0.114f,
                           0.299f, 0.587f, 0.114f);
    float3x3 s = float3x3(0.701f, -0.587f, -0.114f,
                           -0.299f, 0.413f, -0.114f,
                           -0.300f, -0.588f, 0.886f);
    float3x3 u = float3x3(0.168f, -0.331f, 0.500f,
                           0.328f, 0.035f, -0.500f,
                           -0.497f, 0.296f, 0.201f);
    float3x3 m = k + s * cos(angle) + u * sin(angle);
    return mul(m, color);
}

[numthreads(8, 8, 1)]
void main_cs(uint3 id : SV_DispatchThreadID) {
    if (id.x >= (uint)g_Constants.screenSize.x || id.y >= (uint)g_Constants.screenSize.y)
        return;

    AGFXTexture2D<float4> sourceTex = AGFXTexture2D<float4>::Create(g_Constants.sourceTex);
    AGFXRWTexture2D<float4> destTex = AGFXRWTexture2D<float4>::Create(g_Constants.destTex);
    AGFXSampler pointSamp = AGFXSampler::Create(g_Constants.pointSampler);

    float2 uv = (float2(id.xy) + 0.5f) / g_Constants.screenSize;

    float3 color = sourceTex.SampleLevel(pointSamp, uv, 0.0f).rgb;
    color = RotateHue(color, g_Constants.time * 0.6f);

    float2 centered = uv * 2.0f - 1.0f;
    float vignette = 1.0f - dot(centered, centered) * 0.35f;
    color *= saturate(vignette);

    float scanline = 0.94f + 0.06f * sin(uv.y * g_Constants.screenSize.y * 3.14159f);
    color *= scanline;

    destTex.Store(int2(id.xy), float4(saturate(color), 1.0f));
}
