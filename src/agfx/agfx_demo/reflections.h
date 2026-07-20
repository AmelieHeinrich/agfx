/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-20 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <agfx/agfx.h>

// Raytraced mirror reflections, inline RayQuery against the scene TLAS. Gated by a
// metallic/roughness threshold on the GBuffer material; pixels that don't qualify
// (or whose ray misses) are written with alpha 0 so the lighting pass can fall back
// to its flat ambient term. Hit shading reuses the existing directional-light
// cascaded shadow map - no secondary visibility ray is traced.
// The caller owns barrier state: albedo/normal/mra/depth SRVs and the shadow map
// must be in AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE and reflOut in
// AGFX_RESOURCE_STATE_UNORDERED_ACCESS on entry.
class AgfxReflections {
public:
    AgfxReflections(agfxDevice* device);
    ~AgfxReflections();

    AgfxReflections(const AgfxReflections&) = delete;
    AgfxReflections& operator=(const AgfxReflections&) = delete;

    void Generate(agfxDevice* device, agfxCommandBuffer* cmdBuffer,
                  agfxAccelerationStructure* tlas, agfxBufferView* gpuSceneView,
                  agfxBufferView* indexBufferView, agfxBufferView* vertexBufferView,
                  agfxTextureView* albedoSRV, agfxTextureView* normalSRV, agfxTextureView* mraSRV, agfxTextureView* depthSRV,
                  agfxTextureView* reflUAV, agfxTexture* reflTexture,
                  agfxBufferView* sceneCBView, agfxTextureView* shadowMapView, agfxBufferView* cascadeCBView,
                  agfxSampler* gbufferSampler, agfxSampler* materialSampler, agfxSampler* shadowComparisonSampler, agfxSampler* pointSampler,
                  float metallicThreshold, float roughnessThreshold,
                  uint32_t width, uint32_t height);

private:
    agfxDevice* m_device = nullptr;
    agfxShaderModule* computeShader = nullptr;
    agfxComputePipeline* pipeline = nullptr;
};
