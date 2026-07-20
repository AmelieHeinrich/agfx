/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-20 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "data/shaders/agfx.h"
#include "data/shaders/demo/pbr_common.hlsl"

struct SceneVertex {
    float3 pos;
    float3 normal;
    float4 tangent;
    float2 uv;
};

// Mirrors GPUSceneInstance in gltf_scene.h.
struct GPUSceneInstance {
    float4x4 worldMatrix;
    uint vertexOffset;
    uint indexOffset;
    uint albedoTex;
    uint normalTex;
    uint metallicRoughnessTex;
    float metallicFactor;
    float roughnessFactor;
    float _pad0;
};

struct ReflectionsSceneConstants {
    float4x4 invViewProj;
    float3 cameraPos;
    float _pad0;
    float3 lightDir;
    float _pad1;
    float3 lightColor;
    float lightIntensity;
};

struct ReflectionsPushConstants {
    ResourceHandle tlas;
    ResourceHandle gpuScene;
    ResourceHandle indexBuffer;
    ResourceHandle vertexBuffer;
    ResourceHandle gbufferSampler;  // clamp/point - matches lighting pass's GBuffer reads
    ResourceHandle materialSampler; // repeat/linear - matches gbuffer.hlsl's scene texture reads
    ResourceHandle albedoTex;
    ResourceHandle normalTex;
    ResourceHandle mraTex;
    ResourceHandle depthTex;
    ResourceHandle reflOut;
    ResourceHandle sceneCB;
    ResourceHandle shadowMap;
    ResourceHandle cascadeCB;
    ResourceHandle shadowComparisonSampler;
    ResourceHandle pointSampler;
    float metallicThreshold;
    float roughnessThreshold;
    uint width;
    uint height;
};

AGFX_PUSH_CONSTANTS(ReflectionsPushConstants, g_Constants);

[numthreads(8, 8, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Constants.width || id.y >= g_Constants.height) return;

    AGFXRWTexture2D<float4> reflOut = AGFXRWTexture2D<float4>::Create(g_Constants.reflOut);

    float2 uv = (float2(id.xy) + 0.5f) / float2(g_Constants.width, g_Constants.height);

    AGFXSampler samp = AGFXSampler::Create(g_Constants.gbufferSampler);
    AGFXSampler materialSamp = AGFXSampler::Create(g_Constants.materialSampler);
    AGFXTexture2D<float> depthTex = AGFXTexture2D<float>::Create(g_Constants.depthTex);
    float depth = depthTex.SampleLevel(samp, uv, 0.0f);
    if (depth >= 1.0f) {
        reflOut.Store(int2(id.xy), float4(0.0f, 0.0f, 0.0f, 0.0f));
        return;
    }

    AGFXTexture2D<float4> mraTex = AGFXTexture2D<float4>::Create(g_Constants.mraTex);
    float2 mr = mraTex.SampleLevel(samp, uv, 0.0f).rg;
    float metallic = mr.r;
    float roughness = max(mr.g, 0.045f);

    if (metallic < g_Constants.metallicThreshold || roughness > g_Constants.roughnessThreshold) {
        reflOut.Store(int2(id.xy), float4(0.0f, 0.0f, 0.0f, 0.0f));
        return;
    }

    AGFXStructuredBuffer<ReflectionsSceneConstants> sceneCB = AGFXStructuredBuffer<ReflectionsSceneConstants>::Create(g_Constants.sceneCB);
    ReflectionsSceneConstants scene = sceneCB.Load(0);

    float2 ndc = float2(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f);
    float4 clipPos = float4(ndc, depth, 1.0f);
    float4 worldPos4 = mul(scene.invViewProj, clipPos);
    float3 worldPos = worldPos4.xyz / worldPos4.w;

    AGFXTexture2D<float4> normalTexG = AGFXTexture2D<float4>::Create(g_Constants.normalTex);
    float3 normal = normalize(normalTexG.SampleLevel(samp, uv, 0.0f).xyz * 2.0f - 1.0f);

    float3 viewDir = normalize(scene.cameraPos - worldPos);
    float3 reflectDir = normalize(reflect(-viewDir, normal));

    AGFXRaytracingAccelerationStructure tlas = AGFXRaytracingAccelerationStructure::Create(g_Constants.tlas);

    RayDesc ray;
    ray.Origin = worldPos + normal * 0.01f;
    ray.Direction = reflectDir;
    ray.TMin = 0.001f;
    ray.TMax = 1000.0f;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(tlas.Resource(), RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    while (q.Proceed()) {}

    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        reflOut.Store(int2(id.xy), float4(0.0f, 0.0f, 0.0f, 0.0f));
        return;
    }

    uint instanceIndex = q.CommittedInstanceID();
    uint primIndex = q.CommittedPrimitiveIndex();
    float2 bary = q.CommittedTriangleBarycentrics();
    float3 baryWeights = float3(1.0f - bary.x - bary.y, bary.x, bary.y);

    AGFXStructuredBuffer<GPUSceneInstance> gpuScene = AGFXStructuredBuffer<GPUSceneInstance>::Create(g_Constants.gpuScene);
    GPUSceneInstance inst = gpuScene.Load(instanceIndex);

    AGFXStructuredBuffer<uint> indices = AGFXStructuredBuffer<uint>::Create(g_Constants.indexBuffer);
    uint i0 = indices.Load(inst.indexOffset + primIndex * 3 + 0);
    uint i1 = indices.Load(inst.indexOffset + primIndex * 3 + 1);
    uint i2 = indices.Load(inst.indexOffset + primIndex * 3 + 2);

    AGFXStructuredBuffer<SceneVertex> vertices = AGFXStructuredBuffer<SceneVertex>::Create(g_Constants.vertexBuffer);
    SceneVertex v0 = vertices.Load(inst.vertexOffset + i0);
    SceneVertex v1 = vertices.Load(inst.vertexOffset + i1);
    SceneVertex v2 = vertices.Load(inst.vertexOffset + i2);

    float3 hitNormalLocal = v0.normal * baryWeights.x + v1.normal * baryWeights.y + v2.normal * baryWeights.z;
    float2 hitUV = v0.uv * baryWeights.x + v1.uv * baryWeights.y + v2.uv * baryWeights.z;

    float3 hitPos = mul(inst.worldMatrix, float4(v0.pos * baryWeights.x + v1.pos * baryWeights.y + v2.pos * baryWeights.z, 1.0f)).xyz;
    float3 hitNormal = normalize(mul((float3x3)inst.worldMatrix, hitNormalLocal));

    AGFXTexture2D<float4> hitAlbedoTex = AGFXTexture2D<float4>::Create(inst.albedoTex);
    AGFXTexture2D<float4> hitMRTex = AGFXTexture2D<float4>::Create(inst.metallicRoughnessTex);

    float3 hitAlbedo = pow(hitAlbedoTex.SampleLevel(materialSamp, hitUV, 0.0f).rgb, 2.2f);
    float3 hitMR = hitMRTex.SampleLevel(materialSamp, hitUV, 0.0f).rgb;
    // glTF convention: G channel = roughness, B channel = metallic (matches gbuffer.hlsl).
    float hitMetallic = saturate(hitMR.b * inst.metallicFactor);
    float hitRoughness = saturate(hitMR.g * inst.roughnessFactor);

    AGFXTexture2DArray<float> shadowMap = AGFXTexture2DArray<float>::Create(g_Constants.shadowMap);
    AGFXComparisonSampler shadowCmpSamp = AGFXComparisonSampler::Create(g_Constants.shadowComparisonSampler);
    AGFXSampler shadowPointSamp = AGFXSampler::Create(g_Constants.pointSampler);
    AGFXStructuredBuffer<CascadeConstants> cascadeCB = AGFXStructuredBuffer<CascadeConstants>::Create(g_Constants.cascadeCB);
    CascadeConstants cc = cascadeCB.Load(0);

    float3 hitLightDir = normalize(-scene.lightDir);
    float3 hitViewDir = normalize(scene.cameraPos - hitPos);
    float hitViewDepth = length(scene.cameraPos - hitPos);

    // No secondary visibility ray - reuse the existing directional-light shadow map.
    float shadow = EvaluateShadow(shadowMap, shadowCmpSamp, shadowPointSamp, cc, hitPos, hitNormal, hitViewDepth);

    float3 radiance = scene.lightColor * scene.lightIntensity;
    float3 shaded = PBR_DirectLight(hitAlbedo, hitMetallic, hitRoughness, hitNormal, hitViewDir, hitLightDir, radiance) * shadow;
    shaded += hitAlbedo * 0.03f; // flat ambient term, matching the primary lighting pass's fallback

    reflOut.Store(int2(id.xy), float4(shaded, 1.0f));
}
