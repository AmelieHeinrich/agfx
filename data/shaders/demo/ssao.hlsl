/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "data/shaders/agfx.h"

struct SSAOSceneConstants {
    float4x4 viewProj;
    float4x4 invViewProj;
    float2 screenSize;
    float radius;
    float bias;
    float power;
    uint enabled;
    float2 _pad0;
};

struct SSAOPushConstants {
    ResourceHandle normalTex;
    ResourceHandle depthTex;
    ResourceHandle aoTex;
    ResourceHandle sceneCB;
    ResourceHandle pointSampler;
};

AGFX_PUSH_CONSTANTS(SSAOPushConstants, g_Constants);

#define KERNEL_SIZE 8

// Hemisphere kernel (z >= 0), samples biased toward the origin so more
// occlusion detail is gathered close to the surface (standard SSAO kernel
// distribution, same spirit as the hardcoded kPoissonDisk in deferred_lighting.hlsl).
static const float3 kKernel[KERNEL_SIZE] = {
    float3( 0.1407f,  0.0699f,  0.1027f), float3(-0.0938f,  0.1513f,  0.2312f),
    float3( 0.2312f, -0.1867f,  0.3202f), float3(-0.3187f, -0.1215f,  0.1782f),
    float3( 0.0623f,  0.3421f,  0.3874f), float3(-0.2456f,  0.2789f,  0.4501f),
    float3( 0.4123f,  0.0912f,  0.2654f), float3(-0.0876f, -0.4234f,  0.3987f),
    //float3( 0.3654f, -0.3012f,  0.5231f), float3(-0.5102f,  0.1345f,  0.4123f),
    //float3( 0.1789f,  0.5432f,  0.6012f), float3(-0.4321f, -0.4567f,  0.5789f),
    //float3( 0.6234f,  0.2103f,  0.5012f), float3(-0.1987f,  0.6321f,  0.6789f),
    //float3( 0.3021f, -0.6012f,  0.7345f), float3(-0.6543f, -0.2789f,  0.7012f),
};

float3 HashToVec3(uint2 pixel)
{
    float h1 = frac(sin(dot(float2(pixel), float2(12.9898f, 78.233f))) * 43758.5453f);
    float h2 = frac(sin(dot(float2(pixel), float2(39.3467f, 11.135f))) * 24634.6345f);
    return normalize(float3(h1 * 2.0f - 1.0f, h2 * 2.0f - 1.0f, 0.0f) + float3(0.0001f, 0.0001f, 0.0f));
}

[numthreads(8, 8, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    AGFXStructuredBuffer<SSAOSceneConstants> sceneCB = AGFXStructuredBuffer<SSAOSceneConstants>::Create(g_Constants.sceneCB);
    SSAOSceneConstants scene = sceneCB.Load(0);

    if (id.x >= (uint)scene.screenSize.x || id.y >= (uint)scene.screenSize.y)
        return;

    AGFXRWTexture2D<float4> aoOut = AGFXRWTexture2D<float4>::Create(g_Constants.aoTex);

    if (scene.enabled == 0) {
        aoOut.Store(int2(id.xy), float4(1.0f, 1.0f, 1.0f, 1.0f));
        return;
    }

    AGFXTexture2D<float4> normalTex = AGFXTexture2D<float4>::Create(g_Constants.normalTex);
    AGFXTexture2D<float> depthTex = AGFXTexture2D<float>::Create(g_Constants.depthTex);
    AGFXSampler pointSamp = AGFXSampler::Create(g_Constants.pointSampler);

    float2 uv = (float2(id.xy) + 0.5f) / scene.screenSize;

    float depth = depthTex.SampleLevel(pointSamp, uv, 0.0f);
    if (depth >= 1.0f) {
        aoOut.Store(int2(id.xy), float4(1.0f, 1.0f, 1.0f, 1.0f));
        return;
    }

    float2 ndc = float2(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f);
    float4 clipPos = float4(ndc, depth, 1.0f);
    float4 worldPos4 = mul(scene.invViewProj, clipPos);
    float3 worldPos = worldPos4.xyz / worldPos4.w;

    float3 worldNormal = normalize(normalTex.SampleLevel(pointSamp, uv, 0.0f).xyz * 2.0f - 1.0f);

    float3 randomVec = HashToVec3(id.xy);
    float3 tangent = normalize(randomVec - worldNormal * dot(randomVec, worldNormal));
    float3 bitangent = cross(worldNormal, tangent);
    float3x3 tbn = float3x3(tangent, bitangent, worldNormal);

    float occlusion = 0.0f;
    [unroll]
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        float3 samplePos = worldPos + mul(kKernel[i], tbn) * scene.radius;

        float4 sampleClip = mul(scene.viewProj, float4(samplePos, 1.0f));
        float3 sampleNDC = sampleClip.xyz / sampleClip.w;
        float2 sampleUV = sampleNDC.xy * float2(0.5f, -0.5f) + 0.5f;

        if (any(sampleUV < 0.0f) || any(sampleUV > 1.0f))
            continue;

        float sceneDepth = depthTex.SampleLevel(pointSamp, sampleUV, 0.0f);
        if (sceneDepth >= 1.0f)
            continue;

        float2 sceneNDC = float2(sampleUV.x * 2.0f - 1.0f, (1.0f - sampleUV.y) * 2.0f - 1.0f);
        float4 sceneClipPos = float4(sceneNDC, sceneDepth, 1.0f);
        float4 sceneWorldPos4 = mul(scene.invViewProj, sceneClipPos);
        float3 sceneWorldPos = sceneWorldPos4.xyz / sceneWorldPos4.w;

        // Compare view-space depth (clip.w, linear in view space for a standard
        // perspective projection) rather than distance from worldPos - robust at
        // grazing angles and independent of the kernel sample's direction.
        float sampleViewZ = sampleClip.w;
        float sceneViewZ = mul(scene.viewProj, float4(sceneWorldPos, 1.0f)).w;

        float rangeCheck = smoothstep(0.0f, 1.0f, scene.radius / max(abs(sampleViewZ - sceneViewZ), 1e-4f));
        occlusion += (sceneViewZ <= sampleViewZ - scene.bias) ? rangeCheck : 0.0f;
    }

    float ao = 1.0f - saturate((occlusion / (float)KERNEL_SIZE) * scene.power);
    aoOut.Store(int2(id.xy), float4(ao, ao, ao, 1.0f));
}
