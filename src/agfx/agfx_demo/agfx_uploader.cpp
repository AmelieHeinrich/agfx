/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "agfx_uploader.h"

#include <cstring>

void AgfxUploader::EnsurePass()
{
    if (!activePass) {
        activePass = agfxComputePassBegin(cmdBuffer, "Upload");
    }
}

AgfxUploader::AgfxUploader(agfxDevice* device, agfxCommandQueue* queue)
{
    m_device = device;
    this->queue = queue;
    cmdBuffer = agfxCommandBufferCreate(device, queue);
    fence = agfxFenceCreate(device);

    agfxCommandBufferReset(cmdBuffer);
    agfxCommandBufferBegin(cmdBuffer);
}

AgfxUploader::~AgfxUploader()
{
    agfxFenceDestroy(m_device, fence);
    agfxCommandBufferDestroy(m_device, cmdBuffer);
}

void AgfxUploader::UploadBuffer(agfxDevice* device, agfxBuffer* dstBuffer, uint64_t dstOffset, const void* data, uint64_t size)
{
    if (size == 0) return;

    agfxBufferCreateInfo stagingInfo = {};
    stagingInfo.size = size;
    stagingInfo.stride = size;
    stagingInfo.usage = AGFX_BUFFER_USAGE_SHADER_READ;
    stagingInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU;
    agfxBuffer* staging = agfxBufferCreate(device, &stagingInfo);
    agfxBufferSetName(staging, "Upload Staging Buffer");

    void* dst = agfxBufferMap(staging);
    memcpy(dst, data, size);
    agfxBufferUnmap(staging);

    EnsurePass();
    agfxComputePassCopyBufferToBuffer(activePass, staging, dstBuffer, 0, dstOffset, size);

    pendingStagingBuffers.push_back(staging);
}

void AgfxUploader::UploadTexture(agfxDevice* device, agfxTexture* dstTexture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, const void* data, uint32_t dataSize, uint32_t bytesPerRow, uint32_t bytesPerImage)
{
    if (dataSize == 0) return;

    agfxBufferCreateInfo stagingInfo = {};
    stagingInfo.size = dataSize;
    stagingInfo.stride = dataSize;
    stagingInfo.usage = AGFX_BUFFER_USAGE_SHADER_READ;
    stagingInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU;
    agfxBuffer* staging = agfxBufferCreate(device, &stagingInfo);
    agfxBufferSetName(staging, "Upload Staging Buffer");

    void* dst = agfxBufferMap(staging);
    memcpy(dst, data, dataSize);
    agfxBufferUnmap(staging);

    EnsurePass();
    agfxComputePassCopyBufferToTexture(activePass, staging, dstTexture, region, mipLevel, layer, bytesPerRow, bytesPerImage);

    pendingStagingBuffers.push_back(staging);
}

void AgfxUploader::Flush(agfxDevice* device)
{
    if (pendingStagingBuffers.empty()) return;

    agfxDeviceMakeResourcesResident(device);

    if (activePass) {
        agfxComputePassEnd(activePass);
        activePass = nullptr;
    }

    agfxCommandBufferEnd(cmdBuffer);
    agfxCommandQueueSubmit(queue, &cmdBuffer, 1);

    fenceValue++;
    agfxCommandQueueSignal(queue, fence, fenceValue);
    agfxFenceWait(fence, fenceValue, UINT64_MAX);

    for (agfxBuffer* staging : pendingStagingBuffers) {
        agfxBufferDestroy(device, staging);
    }
    pendingStagingBuffers.clear();

    agfxCommandBufferReset(cmdBuffer);
    agfxCommandBufferBegin(cmdBuffer);
}
