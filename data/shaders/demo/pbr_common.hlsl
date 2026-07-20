/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-20 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "data/shaders/agfx.h"

#define PI 3.14159265359f
#define MAX_CASCADES 4

// GGX/Trowbridge-Reitz normal distribution function.
float D_GGX(float NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float d = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
    return a2 / max(PI * d * d, 1e-6f);
}

// Smith geometry term with Schlick-GGX approximation (direct lighting k).
float G_SchlickGGX(float NdotV, float k)
{
    return NdotV / max(NdotV * (1.0f - k) + k, 1e-6f);
}

float G_Smith(float NdotV, float NdotL, float roughness)
{
    float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
    return G_SchlickGGX(NdotV, k) * G_SchlickGGX(NdotL, k);
}

float3 F_Schlick(float VdotH, float3 F0)
{
    return F0 + (1.0f - F0) * pow(saturate(1.0f - VdotH), 5.0f);
}

// Cook-Torrance specular + Lambertian diffuse for a single directional light.
float3 PBR_DirectLight(float3 albedo, float metallic, float roughness, float3 N, float3 V, float3 L, float3 radiance)
{
    float3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 1e-4f);
    float NdotL = max(dot(N, L), 0.0f);
    float NdotH = max(dot(N, H), 0.0f);
    float VdotH = max(dot(V, H), 0.0f);

    float3 F0 = lerp(float3(0.04f, 0.04f, 0.04f), albedo, metallic);

    float D = D_GGX(NdotH, roughness);
    float G = G_Smith(NdotV, NdotL, roughness);
    float3 F = F_Schlick(VdotH, F0);

    float3 specular = (D * G * F) / max(4.0f * NdotV * NdotL, 1e-6f);

    float3 kD = (1.0f - F) * (1.0f - metallic);
    float3 diffuse = kD * albedo / PI;

    return (diffuse + specular) * radiance * NdotL;
}

struct CascadeConstants {
    float4x4 viewProj[MAX_CASCADES];
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

static const float2 kPoissonDisk[8] = {
    float2(-0.326,-0.406), float2(-0.840,-0.074), float2(-0.696, 0.457), float2(-0.203, 0.621),
    float2( 0.962,-0.195), float2( 0.473,-0.480), float2( 0.519, 0.767), float2( 0.185,-0.893)
};

static const float3 kCascadeDebugColors[MAX_CASCADES] = {
    float3(1.0f, 0.3f, 0.3f), float3(0.3f, 1.0f, 0.3f), float3(0.3f, 0.3f, 1.0f), float3(1.0f, 1.0f, 0.3f)
};

float PCSS_BlockerSearch(AGFXTexture2DArray<float> shadowMap, AGFXSampler pointSamp, float3 uvSlice, float receiverDepth, float searchRadiusUV)
{
    float blockerSum = 0.0f;
    int count = 0;
    [unroll]
    for (int i = 0; i < 8; ++i) {
        float2 offset = kPoissonDisk[i] * searchRadiusUV;
        float sampleDepth = shadowMap.SampleLevel(pointSamp, float3(uvSlice.xy + offset, uvSlice.z), 0.0f);
        if (sampleDepth < receiverDepth) {
            blockerSum += sampleDepth;
            count++;
        }
    }
    return count > 0 ? blockerSum / (float)count : -1.0f;
}

float PCSS_Shadow(AGFXTexture2DArray<float> shadowMap, AGFXComparisonSampler cmpSamp, AGFXSampler pointSamp,
                   float3 uvSlice, float receiverDepth, float lightSizeUV, float maxPenumbraUV)
{
    float avgBlockerDepth = PCSS_BlockerSearch(shadowMap, pointSamp, uvSlice, receiverDepth, lightSizeUV);
    if (avgBlockerDepth < 0.0f) {
        return 1.0f;
    }

    float penumbraRatio = (receiverDepth - avgBlockerDepth) / max(avgBlockerDepth, 1e-4f);
    float penumbraUV = clamp(penumbraRatio * lightSizeUV, 0.0f, maxPenumbraUV);

    float sum = 0.0f;
    [unroll]
    for (int i = 0; i < 8; ++i) {
        float2 offset = kPoissonDisk[i] * penumbraUV;
        sum += shadowMap.SampleCmpLevelZero(cmpSamp, float3(uvSlice.xy + offset, uvSlice.z), receiverDepth);
    }
    return sum / 8.0f;
}

// Reprojects worldPos into cascade `cascadeIndex`'s clip space and evaluates PCSS soft shadow.
// Returns 1.0 (fully lit) if the point falls outside the cascade's shadow-map bounds.
float SampleCascadeShadow(AGFXTexture2DArray<float> shadowMap, AGFXComparisonSampler cmpSamp, AGFXSampler pointSamp,
                           CascadeConstants cc, uint cascadeIndex, float3 worldPos)
{
    float4 lsClip = mul(cc.viewProj[cascadeIndex], float4(worldPos, 1.0f));
    float3 lsNDC = lsClip.xyz / lsClip.w;
    float2 shadowUV = lsNDC.xy * float2(0.5f, -0.5f) + 0.5f;
    float receiverDepth = lsNDC.z;

    if (any(shadowUV < 0.0f) || any(shadowUV > 1.0f) || receiverDepth < 0.0f || receiverDepth > 1.0f) {
        return 1.0f;
    }

    return PCSS_Shadow(shadowMap, cmpSamp, pointSamp, float3(shadowUV, (float)cascadeIndex), receiverDepth, cc.lightSizeUV, cc.pcssMaxPenumbraUV);
}

// Selects the cascade for `worldPos` and evaluates its (optionally cross-cascade
// blended) soft shadow term, offsetting the sample point along `normal` to reduce
// shadow acne. Shared between the main deferred-lighting pass and the raytraced
// reflections pass.
float EvaluateShadow(AGFXTexture2DArray<float> shadowMap, AGFXComparisonSampler cmpSamp, AGFXSampler pointSamp,
                      CascadeConstants cc, float3 worldPos, float3 normal, float viewDepth)
{
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
    float shadow = SampleCascadeShadow(shadowMap, cmpSamp, pointSamp, cc, cascadeIndex, worldPos + normalOffset);
    if (blendFactor > 0.0f) {
        uint nextCascade = min(cascadeIndex + 1, cc.cascadeCount - 1);
        float3 normalOffsetNext = normal * cc.texelWorldSize[nextCascade] * 1.5f;
        float shadowNext = SampleCascadeShadow(shadowMap, cmpSamp, pointSamp, cc, nextCascade, worldPos + normalOffsetNext);
        shadow = lerp(shadow, shadowNext, blendFactor);
    }
    return shadow;
}
