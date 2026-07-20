/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-20 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "data/shaders/agfx.h"

struct BlitPushConstants {
    ResourceHandle sourceTex;
    ResourceHandle textureSampler;
};

AGFX_PUSH_CONSTANTS(BlitPushConstants, g_Constants);

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

float4 main_ps(vs_out input) : SV_TARGET {
    AGFXTexture2D<float4> sourceTex = AGFXTexture2D<float4>::Create(g_Constants.sourceTex);
    AGFXSampler samp = AGFXSampler::Create(g_Constants.textureSampler);
    return float4(sourceTex.Sample(samp, input.uv).rgb, 1.0f);
}
