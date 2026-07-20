/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <agfx/agfx.h>

#include <vector>

// A minimal staging-buffer uploader: data is copied into a CPU-visible staging
// buffer, then transferred to the (GPU-only) destination resource via a GPU
// copy. This is the pattern required by non-UMA backends (D3D12's default
// heap resources aren't CPU-mappable) -- unlike agfxBufferMap/agfxTextureReplaceRegion,
// which only work because Metal4/UMA lets any resource be CPU-visible.
class AgfxUploader {
public:
    AgfxUploader(agfxDevice* device, agfxCommandQueue* queue);
    ~AgfxUploader();

    AgfxUploader(const AgfxUploader&) = delete;
    AgfxUploader& operator=(const AgfxUploader&) = delete;

    void UploadBuffer(agfxDevice* device, agfxBuffer* dstBuffer, uint64_t dstOffset, const void* data, uint64_t size);
    void UploadTexture(agfxDevice* device, agfxTexture* dstTexture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, const void* data, uint32_t dataSize, uint32_t bytesPerRow, uint32_t bytesPerImage);

    // Submits all pending copies, waits for the GPU to complete them, and frees the staging buffers.
    void Flush(agfxDevice* device);

private:
    void EnsurePass();

    agfxDevice* m_device = nullptr;
    agfxCommandQueue* queue = nullptr;
    agfxCommandBuffer* cmdBuffer = nullptr;
    agfxFence* fence = nullptr;
    uint64_t fenceValue = 0;

    agfxComputePass* activePass = nullptr;
    std::vector<agfxBuffer*> pendingStagingBuffers;
};
