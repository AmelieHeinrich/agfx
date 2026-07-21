/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include "test_compare.h"
#include "test_framework.h"

#include <agfx/agfx.h>
#include <agfx/agfx.hpp>
#include <agfx_shader/agfx_shader_compiler.h>

#include <string>
#include <vector>

namespace agfxtest
{
    /// @brief A headless device plus the one queue/command buffer/fence a test needs, in raw C
    /// handles. Cleans up in reverse creation order on destruction.
    ///
    /// Both the C and C++ tests submit work the same way: RecordAndSubmit() begins the command
    /// buffer, hands it to the caller's lambda, then ends, submits, and blocks until the GPU is
    /// idle — tests are synchronous by design, there is no frame pacing to model.
    class GpuFixture
    {
    public:
        GpuFixture();
        ~GpuFixture();

        GpuFixture(const GpuFixture&) = delete;
        GpuFixture& operator=(const GpuFixture&) = delete;

        bool Valid() const { return mDevice != nullptr; }

        agfxDevice* Device() const { return mDevice; }
        agfxCommandQueue* Queue() const { return mQueue; }
        agfxCommandBuffer* CommandBuffer() const { return mCommandBuffer; }

        /// @brief Records via `record(cmd)` and blocks until the submission completes.
        template<typename Fn>
        void RecordAndSubmit(Fn&& record)
        {
            agfxCommandBufferReset(mCommandBuffer);
            agfxCommandBufferBegin(mCommandBuffer);
            record(mCommandBuffer);
            agfxCommandBufferEnd(mCommandBuffer);
            SubmitAndWait();
        }

        void SubmitAndWait();

    private:
        agfxDevice* mDevice = nullptr;
        agfxCommandQueue* mQueue = nullptr;
        agfxCommandBuffer* mCommandBuffer = nullptr;
        agfxFence* mFence = nullptr;
        uint64_t mFenceValue = 0;
    };

    /// @brief The device create info every test uses: plain malloc/free and validation on, so a
    /// validation error surfaces as a test failure rather than silent misbehavior.
    agfxDeviceCreateInfo DefaultDeviceCreateInfo();

    /// @brief Reads the device's reported name; used for the report header.
    std::string DeviceName(agfxDevice* device);

    // --- Shaders --------------------------------------------------------------------------

    /// @brief A compiled shader blob plus the thread group size the compiler reflected out of it.
    struct CompiledShader
    {
        std::vector<uint8_t> code;
        uint32_t groupSizeX = 0;
        uint32_t groupSizeY = 0;
        uint32_t groupSizeZ = 0;

        bool Valid() const { return !code.empty(); }
    };

    /// @brief Compiles `data/shaders/tests/<fileName>` at the given stage/entry point. Returns an
    /// invalid CompiledShader on failure; the caller should AGFX_EXPECT on Valid().
    CompiledShader CompileTestShader(const std::string& fileName, agfxShaderStage stage,
                                     const char* entryPoint);

    /// @brief Creates a shader module from a CompiledShader. Modules may be destroyed as soon as
    /// the pipelines built from them exist.
    agfxShaderModule* CreateShaderModule(agfxDevice* device, const CompiledShader& shader,
                                         const char* entryPoint, agfxShaderModuleType type);

    // --- Readback -------------------------------------------------------------------------

    // Both readback helpers record onto their own throwaway command buffer and block until it
    // completes, so they work equally well against a GpuFixture's queue or an ez::Context's — the
    // caller only has to hand over the device and queue that own the resource.

    /// @brief Copies `size` bytes out of a GPU buffer into host memory, via a readback staging
    /// buffer. `currentState` is the buffer's state on entry; it is restored before returning.
    bool ReadbackBuffer(agfxDevice* device, agfxCommandQueue* queue, agfxBuffer* buffer, uint64_t size,
                        agfxResourceState currentState, std::vector<uint8_t>& outBytes);

    /// @brief Copies `size` bytes of host memory into a GPU buffer, via an upload staging buffer.
    /// `currentState` is the buffer's state on entry; it is restored before returning. The mirror of
    /// ReadbackBuffer, for tests that need a GPU-only buffer seeded with a known pattern.
    bool UploadBuffer(agfxDevice* device, agfxCommandQueue* queue, agfxBuffer* buffer, const void* data,
                      uint64_t size, agfxResourceState currentState);

    /// @brief Copies host pixels into a 2D texture's mip 0, via an upload staging buffer and a
    /// buffer-to-texture copy. `currentState` is the texture's state on entry; it is restored before
    /// returning. Only RGBA8_UNORM and RGBA32F are supported, as with ReadbackTexture2D.
    /// @note `width * bytesPerPixel` should stay 256-byte aligned; D3D12 requires that row pitch for
    ///       buffer-to-texture copies, so unaligned widths would pass on Metal and fail on Windows.
    bool UploadTexture2D(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                         uint32_t width, uint32_t height, agfxTextureFormat format, const void* pixels,
                         agfxResourceState currentState);

    /// @brief As UploadTexture2D, but targets an arbitrary mip level and array layer. `width` and
    /// `height` are that subresource's dimensions, not the base mip's. Only the addressed
    /// subresource is transitioned, so the rest of the texture keeps whatever state it was in.
    bool UploadTextureSubresource(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                                  uint32_t width, uint32_t height, agfxTextureFormat format,
                                  const void* pixels, agfxResourceState currentState,
                                  uint32_t mipLevel, uint32_t layer);

    /// @brief As ReadbackTexture2D, but reads an arbitrary mip level and array layer. `width` and
    /// `height` are that subresource's dimensions. The mirror of UploadTextureSubresource, and what
    /// the mip/slice copy tests use both to fetch the copy's destination and to confirm the
    /// neighbouring subresources were left alone.
    bool ReadbackTextureSubresource(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                                    uint32_t width, uint32_t height, agfxTextureFormat format,
                                    agfxResourceState currentState, uint32_t mipLevel, uint32_t layer,
                                    Image& outImage);

    /// @brief Copies a 2D texture's mip 0 into a host Image, via a readback staging buffer.
    /// Only RGBA8_UNORM and RGBA32F are supported — the two formats the test suite renders to.
    /// `currentState` is the texture's state on entry; it is restored before returning.
    bool ReadbackTexture2D(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* texture,
                           uint32_t width, uint32_t height, agfxTextureFormat format,
                           agfxResourceState currentState, Image& outImage);
} // namespace agfxtest
