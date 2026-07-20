/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "data/shaders/agfx.h"

struct SceneVertex {
    float3 pos;
    float3 normal;
    float4 tangent;
    float2 uv;
};

struct CascadeConstants {
    float4x4 viewProj[4];
    float4 splitFar;
    float4 texelWorldSize;
    uint cascadeCount;
    float shadowMapResolution;
    float depthBiasConstant;
    float lightSizeUV;
    float pcssMaxPenumbraUV;
    uint visualizeCascades;
    float2 _pad0;
};

struct ShadowPushConstants {
    float4x4 worldMatrix;
    ResourceHandle vertexBuffer;
    ResourceHandle cascadeCB;
    uint cascadeIndex;
    uint vertexOffset;
};

AGFX_PUSH_CONSTANTS(ShadowPushConstants, g_Constants);

float4 main_vs(uint vertexID : SV_VertexID) : SV_POSITION {
    AGFXStructuredBuffer<SceneVertex> vertices = AGFXStructuredBuffer<SceneVertex>::Create(g_Constants.vertexBuffer);
    AGFXStructuredBuffer<CascadeConstants> cascadeCB = AGFXStructuredBuffer<CascadeConstants>::Create(g_Constants.cascadeCB);
    CascadeConstants cc = cascadeCB.Load(0);

    SceneVertex v = vertices.Load(vertexID + g_Constants.vertexOffset);
    float4 worldPos = mul(g_Constants.worldMatrix, float4(v.pos, 1.0f));
    float4 clipPos = mul(cc.viewProj[g_Constants.cascadeIndex], worldPos);
    clipPos.z += cc.depthBiasConstant * clipPos.w;

    return clipPos;
}
