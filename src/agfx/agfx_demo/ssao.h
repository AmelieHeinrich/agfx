/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <agfx/agfx.h>

// Screen-space ambient occlusion, world-space hemisphere kernel sampling.
// Reads the GBuffer normal (world-space, SRV) and depth (SRV), writes a single
// RGBA8_UNORM AO term (in .r) to a UAV texture. The caller owns barrier state:
// normalSRV/depthSRV must be in AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
// and aoUAV in AGFX_RESOURCE_STATE_UNORDERED_ACCESS on entry.
class AgfxSSAO {
public:
    AgfxSSAO(agfxDevice* device);
    ~AgfxSSAO();

    AgfxSSAO(const AgfxSSAO&) = delete;
    AgfxSSAO& operator=(const AgfxSSAO&) = delete;

    void Generate(agfxDevice* device, agfxCommandBuffer* cmdBuffer,
                  agfxTextureView* normalSRV, agfxTextureView* depthSRV, agfxTextureView* aoUAV, agfxTexture* aoTexture,
                  agfxBufferView* sceneCBView, agfxSampler* pointSampler,
                  uint32_t width, uint32_t height);

private:
    agfxDevice* m_device = nullptr;
    agfxShaderModule* computeShader = nullptr;
    agfxComputePipeline* pipeline = nullptr;
};
