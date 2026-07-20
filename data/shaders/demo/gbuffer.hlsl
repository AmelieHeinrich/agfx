/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 23:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "data/shaders/agfx.h"

struct SceneVertex {
    float3 pos;
    float3 normal;
    float4 tangent;
    float2 uv;
};

struct GBufferSceneConstants {
    float4x4 viewProj;
};

struct GBufferPushConstants {
    float4x4 worldMatrix;
    ResourceHandle vertexBuffer;
    ResourceHandle albedoTex;
    ResourceHandle normalTex;
    ResourceHandle metallicRoughnessTex;
    ResourceHandle textureSampler;
    ResourceHandle sceneCB;
    uint vertexOffset;
    float metallicFactor;
    float roughnessFactor;
};

AGFX_PUSH_CONSTANTS(GBufferPushConstants, g_Constants);

struct vs_out {
    float4 position : SV_POSITION;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float4 tangent : TEXCOORD2;
    float2 uv : TEXCOORD3;
};

vs_out main_vs(uint vertexID : SV_VertexID) {
    AGFXStructuredBuffer<SceneVertex> vertices = AGFXStructuredBuffer<SceneVertex>::Create(g_Constants.vertexBuffer);
    AGFXStructuredBuffer<GBufferSceneConstants> sceneCB = AGFXStructuredBuffer<GBufferSceneConstants>::Create(g_Constants.sceneCB);
    GBufferSceneConstants scene = sceneCB.Load(0);

    SceneVertex v = vertices.Load(vertexID + g_Constants.vertexOffset);

    float4 worldPos = mul(g_Constants.worldMatrix, float4(v.pos, 1.0f));

    vs_out output;
    output.position = mul(scene.viewProj, worldPos);
    output.worldPos = worldPos.xyz;
    output.normal = normalize(mul((float3x3)g_Constants.worldMatrix, v.normal));
    output.tangent = float4(normalize(mul((float3x3)g_Constants.worldMatrix, v.tangent.xyz)), v.tangent.w);
    output.uv = v.uv;
    return output;
}

struct ps_out {
    float4 albedo : SV_TARGET0;
    float4 normal : SV_TARGET1;
    float4 mra : SV_TARGET2;
};

ps_out main_ps(vs_out input) {
    AGFXTexture2D<float4> albedoTex = AGFXTexture2D<float4>::Create(g_Constants.albedoTex);
    AGFXTexture2D<float4> normalTex = AGFXTexture2D<float4>::Create(g_Constants.normalTex);
    AGFXTexture2D<float4> metallicRoughnessTex = AGFXTexture2D<float4>::Create(g_Constants.metallicRoughnessTex);
    AGFXSampler samp = AGFXSampler::Create(g_Constants.textureSampler);

    float4 albedoSample = albedoTex.Sample(samp, input.uv);
    if (albedoSample.a < 0.5f) discard;
    float3 albedo = pow(albedoSample.rgb, 2.2f);

    float3 n = normalize(input.normal);
    float3 t = normalize(input.tangent.xyz - n * dot(n, input.tangent.xyz));
    float3 b = cross(n, t) * input.tangent.w;
    float3x3 tbn = float3x3(t, b, n);

    float3 mapNormal = normalTex.Sample(samp, input.uv).xyz * 2.0f - 1.0f;
    float3 worldNormal = normalize(mul(mapNormal, tbn));

    // glTF convention: G channel = roughness, B channel = metallic.
    float3 mrSample = metallicRoughnessTex.Sample(samp, input.uv).rgb;
    float metallic = saturate(mrSample.b * g_Constants.metallicFactor);
    float roughness = saturate(mrSample.g * g_Constants.roughnessFactor);

    ps_out output;
    output.albedo = float4(albedo, 1.0f);
    output.normal = float4(worldNormal * 0.5f + 0.5f, 1.0f);
    output.mra = float4(metallic, roughness, 1.0f, 1.0f);
    return output;
}
