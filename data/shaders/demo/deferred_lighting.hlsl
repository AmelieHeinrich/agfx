/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 23:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "data/shaders/agfx.h"
#include "data/shaders/demo/pbr_common.hlsl"

struct LightingSceneConstants {
    float4x4 invViewProj;
    float3 cameraPos;
    float _pad0;
    float3 lightDir;
    float _pad1;
    float3 lightColor;
    float lightIntensity;
};

struct LightingPushConstants {
    ResourceHandle albedoTex;
    ResourceHandle normalTex;
    ResourceHandle mraTex;
    ResourceHandle depthTex;
    ResourceHandle textureSampler;
    ResourceHandle lightCB;
    ResourceHandle shadowMap;
    ResourceHandle cascadeCB;
    ResourceHandle shadowComparisonSampler;
    ResourceHandle pointSampler;
    ResourceHandle aoTex;
    ResourceHandle reflTex;
    uint reflectionsEnabled;
};

AGFX_PUSH_CONSTANTS(LightingPushConstants, g_Constants);

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
    AGFXTexture2D<float4> albedoTex = AGFXTexture2D<float4>::Create(g_Constants.albedoTex);
    AGFXTexture2D<float4> normalTex = AGFXTexture2D<float4>::Create(g_Constants.normalTex);
    AGFXTexture2D<float> depthTex = AGFXTexture2D<float>::Create(g_Constants.depthTex);
    AGFXSampler samp = AGFXSampler::Create(g_Constants.textureSampler);
    AGFXStructuredBuffer<LightingSceneConstants> lightCB = AGFXStructuredBuffer<LightingSceneConstants>::Create(g_Constants.lightCB);
    LightingSceneConstants scene = lightCB.Load(0);

    float depth = depthTex.Sample(samp, input.uv);
    if (depth >= 1.0f) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    float3 albedo = albedoTex.Sample(samp, input.uv).rgb;
    float3 normal = normalize(normalTex.Sample(samp, input.uv).xyz * 2.0f - 1.0f);

    AGFXTexture2D<float4> mraTex = AGFXTexture2D<float4>::Create(g_Constants.mraTex);
    float2 mr = mraTex.Sample(samp, input.uv).rg;
    float metallic = mr.r;
    float roughness = max(mr.g, 0.045f);

    AGFXTexture2D<float4> aoTex = AGFXTexture2D<float4>::Create(g_Constants.aoTex);
    float ao = aoTex.Sample(samp, input.uv).r;

    float2 ndc = float2(input.uv.x * 2.0f - 1.0f, (1.0f - input.uv.y) * 2.0f - 1.0f);
    float4 clipPos = float4(ndc, depth, 1.0f);
    float4 worldPos4 = mul(scene.invViewProj, clipPos);
    float3 worldPos = worldPos4.xyz / worldPos4.w;

    float3 lightDir = normalize(-scene.lightDir);
    float3 viewDir = normalize(scene.cameraPos - worldPos);

    AGFXTexture2DArray<float> shadowMap = AGFXTexture2DArray<float>::Create(g_Constants.shadowMap);
    AGFXComparisonSampler shadowCmpSamp = AGFXComparisonSampler::Create(g_Constants.shadowComparisonSampler);
    AGFXSampler shadowPointSamp = AGFXSampler::Create(g_Constants.pointSampler);
    AGFXStructuredBuffer<CascadeConstants> cascadeCB = AGFXStructuredBuffer<CascadeConstants>::Create(g_Constants.cascadeCB);
    CascadeConstants cc = cascadeCB.Load(0);

    float viewDepth = length(scene.cameraPos - worldPos);
    uint cascadeIndex = cc.cascadeCount - 1;
    float blendFactor = 0.0f;
    const float kBlendBandRatio = 0.1f;
    [unroll]
    for (uint i = 0; i < MAX_CASCADES; ++i) {
        if (i >= cc.cascadeCount) break;
        float splitFar = cc.splitFar[i];
        if (viewDepth < splitFar) {
            cascadeIndex = i;
            float bandStart = splitFar * (1.0f - kBlendBandRatio);
            if (viewDepth > bandStart && i + 1 < cc.cascadeCount) {
                blendFactor = saturate((viewDepth - bandStart) / (splitFar - bandStart));
            }
            break;
        }
    }

    float3 normalOffset = normal * cc.texelWorldSize[cascadeIndex] * 1.5f;
    float shadow = SampleCascadeShadow(shadowMap, shadowCmpSamp, shadowPointSamp, cc, cascadeIndex, worldPos + normalOffset);
    if (blendFactor > 0.0f) {
        uint nextCascade = min(cascadeIndex + 1, cc.cascadeCount - 1);
        float3 normalOffsetNext = normal * cc.texelWorldSize[nextCascade] * 1.5f;
        float shadowNext = SampleCascadeShadow(shadowMap, shadowCmpSamp, shadowPointSamp, cc, nextCascade, worldPos + normalOffsetNext);
        shadow = lerp(shadow, shadowNext, blendFactor);
    }

    float3 radiance = scene.lightColor * scene.lightIntensity;
    float3 directLight = PBR_DirectLight(albedo, metallic, roughness, normal, viewDir, lightDir, radiance) * shadow;

    float3 ambientF = F_Schlick(max(dot(normal, viewDir), 0.0f), lerp(float3(0.04f, 0.04f, 0.04f), albedo, metallic));
    float3 ambientKd = (1.0f - ambientF) * (1.0f - metallic);
    float3 ambient = ambientKd * albedo * 0.03f * ao;

    // Traced mirror reflections (reflections.hlsl) replace the flat ambient term
    // wherever a reflection ray was actually shot and hit something (alpha = 1).
    if (g_Constants.reflectionsEnabled != 0) {
        AGFXTexture2D<float4> reflTex = AGFXTexture2D<float4>::Create(g_Constants.reflTex);
        float4 reflColor = reflTex.Sample(samp, input.uv);
        ambient = lerp(ambient, reflColor.rgb, reflColor.a * max(ambientF.r, max(ambientF.g, ambientF.b)));
    }

    float3 color = ambient + directLight;

    if (cc.visualizeCascades != 0) {
        color = lerp(color, kCascadeDebugColors[cascadeIndex], 0.5f);
    }

    return float4(color, 1.0f);
}
