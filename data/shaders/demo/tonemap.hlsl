/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 23:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "data/shaders/agfx.h"

struct TonemapPushConstants {
    ResourceHandle hdrTex;
    ResourceHandle textureSampler;
    uint isHDR;
};

AGFX_PUSH_CONSTANTS(TonemapPushConstants, g_Constants);

struct vs_out {
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
};

vs_out main_vs(uint vertexID : SV_VertexID) {
    float2 positions[3] = {
        float2(-1.0f, -1.0f),
        float2(-1.0f,  3.0f),
        float2( 3.0f, -1.0f)
    };

    vs_out output;
    output.position = float4(positions[vertexID], 0.0f, 1.0f);
    output.uv = positions[vertexID] * float2(0.5f, -0.5f) + 0.5f;
    return output;
}

float3 ACESFilm(float3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float4 main_ps(vs_out input) : SV_TARGET {
    AGFXTexture2D<float4> hdrTex = AGFXTexture2D<float4>::Create(g_Constants.hdrTex);
    AGFXSampler samp = AGFXSampler::Create(g_Constants.textureSampler);

    float3 color = hdrTex.Sample(samp, input.uv).rgb;

    if (g_Constants.isHDR != 0) {
        color = color / (1.0f + max(max(color.r, color.g), color.b) * 0.1f);
        return float4(color, 1.0f);
    }

    color = ACESFilm(color);
    color = pow(color, 1.0f / 2.2f);
    return float4(color, 1.0f);
}
