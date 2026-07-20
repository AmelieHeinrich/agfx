/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 23:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <agfx/agfx.h>

#include <vector>

// Single-pass box-filter mip chain generator. Reads mip N (as a bindless SRV)
// and writes mip N+1 (as a bindless UAV) with one compute dispatch per mip
// level. The caller owns barrier state for the texture before/after calling:
// mip 0 must be in AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE and all
// other mips in AGFX_RESOURCE_STATE_UNORDERED_ACCESS on entry; every mip is
// left in AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE on return.
class AgfxMipGen {
public:
    AgfxMipGen(agfxDevice* device);
    ~AgfxMipGen();

    AgfxMipGen(const AgfxMipGen&) = delete;
    AgfxMipGen& operator=(const AgfxMipGen&) = delete;

    // The bindless SRV/UAV views used by the dispatches are appended to outViews rather than
    // destroyed here: the command buffer this records into is not guaranteed to have been
    // submitted (let alone completed) by the time this returns, and freeing a descriptor slot
    // makes it immediately available for reuse by a later allocation. Destroying eagerly races
    // still-unexecuted GPU work for that slot. The caller must destroy every view in outViews
    // only after the submission has been confirmed complete (e.g. via a fence wait).
    void Generate(agfxDevice* device, agfxCommandBuffer* cmdBuffer, agfxTexture* texture, uint32_t width, uint32_t height, uint32_t mipLevels, std::vector<agfxTextureView*>& outViews);

private:
    agfxDevice* m_device = nullptr;
    agfxShaderModule* computeShader = nullptr;
    agfxComputePipeline* pipeline = nullptr;
};
