/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "data/shaders/agfx.h"

struct GPUSceneInstance {
    float4x4 worldMatrix;
    uint vertexOffset;
    uint indexOffset;
    uint albedoTex;
    uint normalTex;
    uint metallicRoughnessTex;
    float metallicFactor;
    float roughnessFactor;
    uint indexCount;
    float3 boundsMin;
    float _pad0;
    float3 boundsMax;
    float _pad1;
};

struct CullingPushConstants {
    float4 frustumPlanes[6];
    uint gpuScene;
    uint primitiveCount;
    uint64_t bundleHandle;
};

AGFX_PUSH_CONSTANTS(CullingPushConstants, g_Constants);

bool AABBOutsideFrustum(float3 boundsMin, float3 boundsMax)
{
    for (int i = 0; i < 6; i++) {
        float3 p = float3(
            g_Constants.frustumPlanes[i].x >= 0.0f ? boundsMax.x : boundsMin.x,
            g_Constants.frustumPlanes[i].y >= 0.0f ? boundsMax.y : boundsMin.y,
            g_Constants.frustumPlanes[i].z >= 0.0f ? boundsMax.z : boundsMin.z);
        if (dot(g_Constants.frustumPlanes[i].xyz, p) + g_Constants.frustumPlanes[i].w < 0.0f)
            return true;
    }
    return false;
}

[numthreads(64, 1, 1)]
void main_cs(uint3 dtid : SV_DispatchThreadID)
{
    uint index = dtid.x;
    if (index >= g_Constants.primitiveCount) return;

    AGFXStructuredBuffer<GPUSceneInstance> gpuScene = AGFXStructuredBuffer<GPUSceneInstance>::Create(g_Constants.gpuScene);
    GPUSceneInstance inst = gpuScene.Load(index);

    if (AABBOutsideFrustum(inst.boundsMin, inst.boundsMax)) return;

    AGFXIndirectDrawIndexedBundle bundle = AGFXIndirectDrawIndexedBundle::Create(g_Constants.bundleHandle);
    bundle.DrawIndexed(0, 0, index, inst.indexCount, 1, inst.indexOffset, 0, 0);
}
