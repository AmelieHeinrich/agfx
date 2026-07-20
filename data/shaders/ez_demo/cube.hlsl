/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-20 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "data/shaders/agfx.h"

struct CubeVertex {
    float3 pos;
    float3 color;
};

struct CubePushConstants {
    float4x4 mvp;
    ResourceHandle vertexBuffer;
};

AGFX_PUSH_CONSTANTS(CubePushConstants, g_Constants);

struct vs_out {
    float4 position : SV_POSITION;
    float3 color : COLOR0;
};

vs_out main_vs(uint vertexID : SV_VertexID) {
    AGFXStructuredBuffer<CubeVertex> vertices = AGFXStructuredBuffer<CubeVertex>::Create(g_Constants.vertexBuffer);
    CubeVertex v = vertices.Load(vertexID);

    vs_out output;
    output.position = mul(g_Constants.mvp, float4(v.pos, 1.0f));
    output.color = v.color;
    return output;
}

float4 main_ps(vs_out input) : SV_TARGET {
    return float4(input.color, 1.0f);
}
