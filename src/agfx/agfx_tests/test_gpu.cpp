/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "test_gpu.h"

#include <cstdlib>
#include <cstring>

namespace agfxtest
{
    namespace
    {
        void* TestAllocate(uint64_t size) { return malloc((size_t)size); }
        void TestFree(void* ptr) { free(ptr); }

        uint32_t BytesPerPixel(agfxTextureFormat format)
        {
            switch (format) {
                case AGFX_TEXTURE_FORMAT_RGBA8_UNORM: return 4;
                case AGFX_TEXTURE_FORMAT_RGBA32F:     return 16;
                default:                              return 0;
            }
        }
    } // namespace

    agfxDeviceCreateInfo DefaultDeviceCreateInfo()
    {
        agfxDeviceCreateInfo info{};
        info.allocate = TestAllocate;
        info.free = TestFree;
        info.tempAllocate = TestAllocate;
        info.tempFree = TestFree;
        info.enableValidation = 1;
        return info;
    }

    std::string DeviceName(agfxDevice* device)
    {
        if (!device) {
            return "unknown";
        }
        agfxDeviceInfo info{};
        agfxDeviceGetInfo(device, &info);
        return info.name;
    }

    // --- GpuFixture -----------------------------------------------------------------------

    GpuFixture::GpuFixture()
    {
        const agfxDeviceCreateInfo deviceInfo = DefaultDeviceCreateInfo();
        mDevice = agfxDeviceCreate(&deviceInfo);
        if (!mDevice) {
            return;
        }

        agfxCommandQueueCreateInfo queueInfo{};
        queueInfo.type = AGFX_COMMAND_QUEUE_TYPE_GRAPHICS;
        mQueue = agfxCommandQueueCreate(mDevice, &queueInfo);
        mCommandBuffer = agfxCommandBufferCreate(mDevice, mQueue);
        mFence = agfxFenceCreate(mDevice);
    }

    GpuFixture::~GpuFixture()
    {
        if (!mDevice) {
            return;
        }
        if (mFence) agfxFenceDestroy(mDevice, mFence);
        if (mCommandBuffer) agfxCommandBufferDestroy(mDevice, mCommandBuffer);
        if (mQueue) agfxCommandQueueDestroy(mDevice, mQueue);
        agfxDeviceDestroy(mDevice);
    }

    void GpuFixture::SubmitAndWait()
    {
        agfxCommandBuffer* buffers[] = {mCommandBuffer};
        agfxCommandQueueSubmit(mQueue, buffers, 1);

        ++mFenceValue;
        agfxCommandQueueSignal(mQueue, mFence, mFenceValue);
        agfxFenceWait(mFence, mFenceValue, UINT64_MAX);
    }

    // --- Shaders --------------------------------------------------------------------------

    CompiledShader CompileTestShader(const std::string& fileName, agfxShaderStage stage,
                                     const char* entryPoint)
    {
        CompiledShader result;

        std::string source;
        if (!ReadTextFile("data/shaders/tests/" + fileName, source) || source.empty()) {
            return result;
        }

        agfxShaderCompilerOptions options{};
        options.stage = stage;
        strncpy(options.entryPoint, entryPoint, sizeof(options.entryPoint) - 1);
        options.sourceCode = source.data();
        options.sourceCodeSize = (uint32_t)source.size();

        agfxShaderCompilerResult compiled{};
        agfxCompileShader(&options, &compiled);
        if (!compiled.compiledCode || compiled.compiledSize == 0) {
            return result;
        }

        result.code.assign(compiled.compiledCode, compiled.compiledCode + compiled.compiledSize);
        result.groupSizeX = compiled.tgSizeX;
        result.groupSizeY = compiled.tgSizeY;
        result.groupSizeZ = compiled.tgSizeZ;
        free(compiled.compiledCode);
        return result;
    }

    agfxShaderModule* CreateShaderModule(agfxDevice* device, const CompiledShader& shader,
                                         const char* entryPoint, agfxShaderModuleType type)
    {
        if (!shader.Valid()) {
            return nullptr;
        }
        agfxShaderModuleCreateInfo info{};
        info.code = const_cast<uint8_t*>(shader.code.data());
        info.codeSize = shader.code.size();
        info.entryPoint = entryPoint;
        info.type = type;
        return agfxShaderModuleCreate(device, &info);
    }

    // --- Readback -------------------------------------------------------------------------

    namespace
    {
        /// @brief Records `record` onto a throwaway command buffer on `queue` and blocks until it
        /// completes. Used by the readback helpers so they don't need a GpuFixture.
        template<typename Fn>
        void RecordAndWait(agfxDevice* device, agfxCommandQueue* queue, Fn&& record)
        {
            agfxCommandBuffer* cmd = agfxCommandBufferCreate(device, queue);
            agfxFence* fence = agfxFenceCreate(device);

            agfxCommandBufferBegin(cmd);
            record(cmd);
            agfxCommandBufferEnd(cmd);

            agfxCommandBuffer* buffers[] = {cmd};
            agfxCommandQueueSubmit(queue, buffers, 1);
            agfxCommandQueueSignal(queue, fence, 1);
            agfxFenceWait(fence, 1, UINT64_MAX);

            agfxFenceDestroy(device, fence);
            agfxCommandBufferDestroy(device, cmd);
        }
    } // namespace

    bool ReadbackBuffer(agfxDevice* device, agfxCommandQueue* queue, agfxBuffer* buffer, uint64_t size,
                        agfxResourceState currentState, std::vector<uint8_t>& outBytes)
    {
        if (!device || !queue || !buffer || size == 0) {
            return false;
        }

        agfxBufferCreateInfo stagingInfo{};
        stagingInfo.size = size;
        stagingInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_READBACK;
        agfxBuffer* staging = agfxBufferCreate(device, &stagingInfo);
        if (!staging) {
            return false;
        }
        agfxDeviceMakeResourcesResident(device);

        RecordAndWait(device, queue, [&](agfxCommandBuffer* cmd) {
            agfxCommandBufferBufferBarrier(cmd, buffer, currentState, AGFX_RESOURCE_STATE_COPY_SOURCE, 0);
            agfxComputePass* pass = agfxComputePassBegin(cmd, "readback buffer");
            agfxComputePassCopyBufferToBuffer(pass, buffer, staging, 0, 0, size);
            agfxComputePassEnd(pass);
            agfxCommandBufferBufferBarrier(cmd, buffer, AGFX_RESOURCE_STATE_COPY_SOURCE, currentState, 0);
        });

        bool ok = false;
        if (void* mapped = agfxBufferMap(staging)) {
            outBytes.resize((size_t)size);
            memcpy(outBytes.data(), mapped, (size_t)size);
            agfxBufferUnmap(staging);
            ok = true;
        }

        agfxBufferDestroy(device, staging);
        return ok;
    }

    bool ReadbackTexture2D(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                           uint32_t width, uint32_t height, agfxTextureFormat format,
                           agfxResourceState currentState, Image& outImage)
    {
        const uint32_t bpp = BytesPerPixel(format);
        if (!device || !queue || !texture || width == 0 || height == 0 || bpp == 0) {
            return false;
        }

        const uint32_t bytesPerRow = width * bpp;
        const uint64_t byteSize = uint64_t(bytesPerRow) * height;

        agfxBufferCreateInfo stagingInfo{};
        stagingInfo.size = byteSize;
        stagingInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_READBACK;
        agfxBuffer* staging = agfxBufferCreate(device, &stagingInfo);
        if (!staging) {
            return false;
        }
        agfxDeviceMakeResourcesResident(device);

        agfxTextureRegion region{};
        region.width = width;
        region.height = height;
        region.depth = 1;

        RecordAndWait(device, queue, [&](agfxCommandBuffer* cmd) {
            agfxCommandBufferTextureBarrier(cmd, texture, currentState, AGFX_RESOURCE_STATE_COPY_SOURCE, 0, 0, 0);
            agfxComputePass* pass = agfxComputePassBegin(cmd, "readback texture");
            agfxComputePassCopyTextureToBuffer(pass, texture, staging, 0, &region, 0, 0, bytesPerRow,
                                               (uint32_t)byteSize);
            agfxComputePassEnd(pass);
            agfxCommandBufferTextureBarrier(cmd, texture, AGFX_RESOURCE_STATE_COPY_SOURCE, currentState, 0, 0, 0);
        });

        void* mapped = agfxBufferMap(staging);
        if (!mapped) {
            agfxBufferDestroy(device, staging);
            return false;
        }

        outImage.width = width;
        outImage.height = height;
        outImage.pixels.resize(size_t(width) * height * 4);

        if (format == AGFX_TEXTURE_FORMAT_RGBA8_UNORM) {
            const uint8_t* src = (const uint8_t*)mapped;
            for (size_t i = 0; i < outImage.pixels.size(); ++i) {
                outImage.pixels[i] = src[i] / 255.0f;
            }
        } else { // AGFX_TEXTURE_FORMAT_RGBA32F
            memcpy(outImage.pixels.data(), mapped, (size_t)byteSize);
        }

        agfxBufferUnmap(staging);
        agfxBufferDestroy(device, staging);
        return true;
    }
} // namespace agfxtest
