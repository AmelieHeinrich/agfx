/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 22:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "data/shaders/agfx.h"

struct ImGuiVertex {
    float2 pos;
    float2 uv;
    uint col;
};

struct ImGuiPushConstants {
    float2 scale;
    float2 translate;
    uint vertexOffset;
    ResourceHandle vertexBuffer;
    ResourceHandle texture;
    ResourceHandle textureSampler;
    ResourceHandle testCbv;
};

AGFX_PUSH_CONSTANTS(ImGuiPushConstants, g_Constants);

struct vs_out {
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
};

vs_out main_vs(uint vertexID : SV_VertexID) {
    AGFXStructuredBuffer<ImGuiVertex> vertices = AGFXStructuredBuffer<ImGuiVertex>::Create(g_Constants.vertexBuffer);
    ImGuiVertex v = vertices.Load(vertexID + g_Constants.vertexOffset);

    vs_out output;
    output.position = float4(v.pos * g_Constants.scale + g_Constants.translate, 0.0f, 1.0f);
    output.uv = v.uv;

    uint packed = v.col;
    output.color = float4(
        (packed & 0xFF) / 255.0f,
        ((packed >> 8) & 0xFF) / 255.0f,
        ((packed >> 16) & 0xFF) / 255.0f,
        ((packed >> 24) & 0xFF) / 255.0f
    );

    return output;
}

float4 main_ps(vs_out input) : SV_TARGET {
    AGFXTexture2D<float4> tex = AGFXTexture2D<float4>::Create(g_Constants.texture);
    AGFXSampler samp = AGFXSampler::Create(g_Constants.textureSampler);
    return input.color * tex.Sample(samp, input.uv);
}
