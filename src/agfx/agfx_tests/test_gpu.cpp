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
        info.enableValidation = 0;
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
        result.meshSizeX = compiled.meshSizeX;
        result.meshSizeY = compiled.meshSizeY;
        result.meshSizeZ = compiled.meshSizeZ;
        result.taskSizeX = compiled.taskSizeX;
        result.taskSizeY = compiled.taskSizeY;
        result.taskSizeZ = compiled.taskSizeZ;
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

    bool UploadBuffer(agfxDevice* device, agfxCommandQueue* queue, agfxBuffer* buffer, const void* data,
                      uint64_t size, agfxResourceState currentState)
    {
        if (!device || !queue || !buffer || !data || size == 0) {
            return false;
        }

        agfxBufferCreateInfo stagingInfo{};
        stagingInfo.size = size;
        stagingInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_UPLOAD;
        agfxBuffer* staging = agfxBufferCreate(device, &stagingInfo);
        if (!staging) {
            return false;
        }
        agfxDeviceMakeResourcesResident(device);

        void* mapped = agfxBufferMap(staging);
        if (!mapped) {
            agfxBufferDestroy(device, staging);
            return false;
        }
        memcpy(mapped, data, (size_t)size);
        agfxBufferUnmap(staging);

        RecordAndWait(device, queue, [&](agfxCommandBuffer* cmd) {
            agfxCommandBufferBufferBarrier(cmd, buffer, currentState, AGFX_RESOURCE_STATE_COPY_DEST, 0);
            agfxComputePass* pass = agfxComputePassBegin(cmd, "upload buffer");
            agfxComputePassCopyBufferToBuffer(pass, staging, buffer, 0, 0, size);
            agfxComputePassEnd(pass);
            agfxCommandBufferBufferBarrier(cmd, buffer, AGFX_RESOURCE_STATE_COPY_DEST, currentState, 0);
        });

        agfxBufferDestroy(device, staging);
        return true;
    }

    bool UploadTexture2D(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                         uint32_t width, uint32_t height, agfxTextureFormat format, const void* pixels,
                         agfxResourceState currentState)
    {
        return UploadTextureSubresource(device, queue, texture, width, height, format, pixels,
                                        currentState, 0, 0);
    }

    bool UploadTextureSubresource(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                                  uint32_t width, uint32_t height, agfxTextureFormat format,
                                  const void* pixels, agfxResourceState currentState,
                                  uint32_t mipLevel, uint32_t layer)
    {
        const uint32_t bpp = BytesPerPixel(format);
        if (!device || !queue || !texture || !pixels || width == 0 || height == 0 || bpp == 0) {
            return false;
        }

        const uint32_t bytesPerRow = width * bpp;
        const uint64_t byteSize = uint64_t(bytesPerRow) * height;

        agfxBufferCreateInfo stagingInfo{};
        stagingInfo.size = byteSize;
        stagingInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_UPLOAD;
        agfxBuffer* staging = agfxBufferCreate(device, &stagingInfo);
        if (!staging) {
            return false;
        }
        agfxDeviceMakeResourcesResident(device);

        void* mapped = agfxBufferMap(staging);
        if (!mapped) {
            agfxBufferDestroy(device, staging);
            return false;
        }
        memcpy(mapped, pixels, (size_t)byteSize);
        agfxBufferUnmap(staging);

        agfxTextureRegion region{};
        region.width = width;
        region.height = height;
        region.depth = 1;

        RecordAndWait(device, queue, [&](agfxCommandBuffer* cmd) {
            agfxCommandBufferTextureBarrier(cmd, texture, currentState, AGFX_RESOURCE_STATE_COPY_DEST,
                                            mipLevel, layer, 0);
            agfxComputePass* pass = agfxComputePassBegin(cmd, "upload texture");
            agfxComputePassCopyBufferToTexture(pass, staging, texture, &region, mipLevel, layer,
                                               bytesPerRow, (uint32_t)byteSize);
            agfxComputePassEnd(pass);
            agfxCommandBufferTextureBarrier(cmd, texture, AGFX_RESOURCE_STATE_COPY_DEST, currentState,
                                            mipLevel, layer, 0);
        });

        agfxBufferDestroy(device, staging);
        return true;
    }

    bool ReadbackTexture2D(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                           uint32_t width, uint32_t height, agfxTextureFormat format,
                           agfxResourceState currentState, Image& outImage)
    {
        return ReadbackTextureSubresource(device, queue, texture, width, height, format, currentState,
                                          0, 0, outImage);
    }

    /// @brief The shared body behind ReadbackTextureSubresource and ReadbackTexture3DSlice. Array
    /// layers and 3D depth slices reach the same texel through different parameters — `layer` for
    /// the former, the region's z origin for the latter — so both are plumbed through here and the
    /// two public wrappers pick the one that applies.
    static bool ReadbackTextureImpl(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                                    uint32_t width, uint32_t height, agfxTextureFormat format,
                                    agfxResourceState currentState, uint32_t mipLevel, uint32_t layer,
                                    uint32_t z, Image& outImage)
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
        region.z = z; // 0 for everything but a 3D texture's depth slices.
        region.width = width;
        region.height = height;
        region.depth = 1;

        RecordAndWait(device, queue, [&](agfxCommandBuffer* cmd) {
            agfxCommandBufferTextureBarrier(cmd, texture, currentState, AGFX_RESOURCE_STATE_COPY_SOURCE,
                                            mipLevel, layer, 0);
            agfxComputePass* pass = agfxComputePassBegin(cmd, "readback texture");
            agfxComputePassCopyTextureToBuffer(pass, texture, staging, 0, &region, mipLevel, layer,
                                               bytesPerRow, (uint32_t)byteSize);
            agfxComputePassEnd(pass);
            agfxCommandBufferTextureBarrier(cmd, texture, AGFX_RESOURCE_STATE_COPY_SOURCE, currentState,
                                            mipLevel, layer, 0);
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

    bool ReadbackTextureSubresource(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                                    uint32_t width, uint32_t height, agfxTextureFormat format,
                                    agfxResourceState currentState, uint32_t mipLevel, uint32_t layer,
                                    Image& outImage)
    {
        return ReadbackTextureImpl(device, queue, texture, width, height, format, currentState,
                                   mipLevel, layer, /*z*/ 0, outImage);
    }

    /// @brief Reads `sliceCount` slices one at a time and concatenates them vertically. `readSlice`
    /// is what distinguishes the array case from the 3D case.
    template<typename ReadSliceFn>
    static bool ReadbackStack(uint32_t width, uint32_t height, uint32_t sliceCount, Image& outImage,
                              ReadSliceFn&& readSlice)
    {
        if (width == 0 || height == 0 || sliceCount == 0) {
            return false;
        }

        outImage.width = width;
        outImage.height = height * sliceCount;
        outImage.pixels.assign(size_t(outImage.width) * outImage.height * 4, 0.0f);

        const size_t floatsPerSlice = size_t(width) * height * 4;
        for (uint32_t slice = 0; slice < sliceCount; ++slice) {
            Image sliceImage;
            if (!readSlice(slice, sliceImage) || !sliceImage.Valid()) {
                return false;
            }
            if (sliceImage.width != width || sliceImage.height != height) {
                return false;
            }
            memcpy(outImage.pixels.data() + floatsPerSlice * slice, sliceImage.pixels.data(),
                   floatsPerSlice * sizeof(float));
        }
        return true;
    }

    bool ReadbackTexture2DArrayStack(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                                     uint32_t width, uint32_t height, uint32_t layerCount,
                                     agfxTextureFormat format, agfxResourceState currentState,
                                     Image& outImage)
    {
        return ReadbackStack(width, height, layerCount, outImage, [&](uint32_t layer, Image& slice) {
            return ReadbackTextureSubresource(device, queue, texture, width, height, format,
                                              currentState, /*mipLevel*/ 0, layer, slice);
        });
    }

    bool ReadbackTexture3DStack(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                                uint32_t width, uint32_t height, uint32_t depth,
                                agfxTextureFormat format, agfxResourceState currentState,
                                Image& outImage)
    {
        return ReadbackStack(width, height, depth, outImage, [&](uint32_t z, Image& slice) {
            return ReadbackTexture3DSlice(device, queue, texture, width, height, format, currentState,
                                          /*mipLevel*/ 0, z, slice);
        });
    }

    bool ReadbackTexture3DSlice(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                                uint32_t width, uint32_t height, agfxTextureFormat format,
                                agfxResourceState currentState, uint32_t mipLevel, uint32_t z,
                                Image& outImage)
    {
        return ReadbackTextureImpl(device, queue, texture, width, height, format, currentState,
                                   mipLevel, /*layer*/ 0, z, outImage);
    }
} // namespace agfxtest
