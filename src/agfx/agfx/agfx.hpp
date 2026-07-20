/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-20 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// agfx.hpp: Header-only, move-only RAII wrapper over the agfx C API. Every agfx*
// object is destroyed automatically when its wrapper goes out of scope. Requires C++17.

#pragma once

#include <agfx/agfx.h>

namespace agfx
{
    // Generic move-only owner for any agfx object destroyed via Destroy(device, handle).
    template<typename T, void(*Destroy)(agfxDevice*, T*)>
    class Handle
    {
    public:
        Handle() = default;
        Handle(agfxDevice* device, T* handle) : mDevice(device), mHandle(handle) {}
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;

        Handle(Handle&& other) noexcept
            : mDevice(other.mDevice), mHandle(other.mHandle)
        {
            other.mDevice = nullptr;
            other.mHandle = nullptr;
        }

        Handle& operator=(Handle&& other) noexcept
        {
            if (this != &other) {
                Reset();
                mDevice = other.mDevice;
                mHandle = other.mHandle;
                other.mDevice = nullptr;
                other.mHandle = nullptr;
            }
            return *this;
        }

        ~Handle() { Reset(); }

        /// @brief Destroys the owned object now, if any. Leaves the wrapper empty.
        void Reset()
        {
            if (mHandle) {
                Destroy(mDevice, mHandle);
            }
            mHandle = nullptr;
            mDevice = nullptr;
        }

        T* Get() const { return mHandle; }
        agfxDevice* GetDevice() const { return mDevice; }
        operator T*() const { return mHandle; }
        explicit operator bool() const { return mHandle != nullptr; }

        /// @brief Relinquishes ownership without destroying the object, e.g. when handing it to
        /// an API (such as ImGui_ImplAGFX_Init) that takes ownership and destroys it itself.
        /// @return The previously owned handle. The wrapper is left empty.
        T* Release()
        {
            T* handle = mHandle;
            mHandle = nullptr;
            mDevice = nullptr;
            return handle;
        }

    protected:
        agfxDevice* mDevice = nullptr;
        T* mHandle = nullptr;
    };

    class Fence : public Handle<agfxFence, agfxFenceDestroy>
    {
    public:
        using Handle::Handle;

        void Wait(uint64_t value, uint64_t timeoutMs = UINT64_MAX) { agfxFenceWait(mHandle, value, timeoutMs); }
        void Signal(uint64_t value) { agfxFenceSignal(mHandle, value); }
        uint64_t GetCompletedValue() { return agfxFenceGetCompletedValue(mHandle); }
    };

    class QueryPool : public Handle<agfxQueryPool, agfxQueryPoolDestroy>
    {
    public:
        using Handle::Handle;

        void Readback(uint32_t firstIndex, uint32_t count, uint64_t* outTimestampsNanoseconds)
        {
            agfxQueryPoolReadback(mDevice, mHandle, firstIndex, count, outTimestampsNanoseconds);
        }
    };

    class Texture : public Handle<agfxTexture, agfxTextureDestroy>
    {
    public:
        using Handle::Handle;

        agfxTextureCreateInfo GetInfo() const
        {
            agfxTextureCreateInfo info{};
            agfxTextureGetInfo(mHandle, &info);
            return info;
        }

        void ReplaceRegion(const agfxTextureRegion& region, uint32_t mipLevel, uint32_t layer, const void* data, uint32_t dataSize, uint32_t bytesPerRow, uint32_t bytesPerImage)
        {
            agfxTextureReplaceRegion(mDevice, mHandle, &region, mipLevel, layer, data, dataSize, bytesPerRow, bytesPerImage);
        }

        void SetName(const char* name) { agfxTextureSetName(mHandle, name); }
    };

    class Buffer : public Handle<agfxBuffer, agfxBufferDestroy>
    {
    public:
        using Handle::Handle;

        void* Map() { return agfxBufferMap(mHandle); }
        void Unmap() { agfxBufferUnmap(mHandle); }
        void SetName(const char* name) { agfxBufferSetName(mHandle, name); }

        agfxBufferCreateInfo GetInfo() const
        {
            agfxBufferCreateInfo info{};
            agfxBufferGetInfo(mHandle, &info);
            return info;
        }
    };

    /// @brief RAII guard around a mapped agfxBuffer; unmaps automatically when it goes out of scope.
    class MappedBuffer
    {
    public:
        explicit MappedBuffer(Buffer& buffer) : mBuffer(&buffer), mPtr(buffer.Map()) {}
        MappedBuffer(const MappedBuffer&) = delete;
        MappedBuffer& operator=(const MappedBuffer&) = delete;
        MappedBuffer(MappedBuffer&& other) noexcept : mBuffer(other.mBuffer), mPtr(other.mPtr) { other.mBuffer = nullptr; other.mPtr = nullptr; }

        ~MappedBuffer()
        {
            if (mBuffer) {
                mBuffer->Unmap();
            }
        }

        void* Get() const { return mPtr; }
        template<typename T> T* As() const { return static_cast<T*>(mPtr); }

    private:
        Buffer* mBuffer = nullptr;
        void* mPtr = nullptr;
    };

    class TextureView : public Handle<agfxTextureView, agfxTextureViewDestroy>
    {
    public:
        using Handle::Handle;

        uint64_t GetHandle() const { return agfxTextureViewGetHandle(mHandle); }
    };

    class Sampler : public Handle<agfxSampler, agfxSamplerDestroy>
    {
    public:
        using Handle::Handle;

        uint64_t GetHandle() const { return agfxSamplerGetHandle(mHandle); }
    };

    class BufferView : public Handle<agfxBufferView, agfxBufferViewDestroy>
    {
    public:
        using Handle::Handle;

        uint64_t GetHandle() const { return agfxBufferViewGetHandle(mHandle); }
    };

    class RenderTarget : public Handle<agfxRenderTarget, agfxRenderTargetDestroy>
    {
    public:
        using Handle::Handle;
    };

    class ShaderModule : public Handle<agfxShaderModule, agfxShaderModuleDestroy>
    {
    public:
        using Handle::Handle;
    };

    class RenderPipeline : public Handle<agfxRenderPipeline, agfxRenderPipelineDestroy>
    {
    public:
        using Handle::Handle;
    };

    class ComputePipeline : public Handle<agfxComputePipeline, agfxComputePipelineDestroy>
    {
    public:
        using Handle::Handle;
    };

    /// @brief RAII scope guard around an agfxComputePass. Ends the pass automatically when it goes out of scope.
    class ComputePass
    {
    public:
        ComputePass() = default;
        explicit ComputePass(agfxComputePass* pass) : mPass(pass) {}
        ComputePass(const ComputePass&) = delete;
        ComputePass& operator=(const ComputePass&) = delete;
        ComputePass(ComputePass&& other) noexcept : mPass(other.mPass) { other.mPass = nullptr; }
        ComputePass& operator=(ComputePass&& other) noexcept
        {
            if (this != &other) { End(); mPass = other.mPass; other.mPass = nullptr; }
            return *this;
        }

        ~ComputePass() { End(); }

        void End()
        {
            if (mPass) {
                agfxComputePassEnd(mPass);
                mPass = nullptr;
            }
        }

        void TextureUAVBarrier(Texture& texture) { agfxComputePassTextureUAVBarrier(mPass, texture); }
        void BufferUAVBarrier(Buffer& buffer) { agfxComputePassBufferUAVBarrier(mPass, buffer); }

        void CopyTextureToBuffer(Texture& texture, Buffer& buffer, uint64_t bufferOffset, const agfxTextureRegion& region, uint32_t mipLevel, uint32_t layer, uint32_t bytesPerRow, uint32_t bytesPerImage)
        {
            agfxComputePassCopyTextureToBuffer(mPass, texture, buffer, bufferOffset, &region, mipLevel, layer, bytesPerRow, bytesPerImage);
        }

        void CopyBufferToTexture(Buffer& buffer, Texture& texture, const agfxTextureRegion& region, uint32_t mipLevel, uint32_t layer, uint32_t bytesPerRow, uint32_t bytesPerImage)
        {
            agfxComputePassCopyBufferToTexture(mPass, buffer, texture, &region, mipLevel, layer, bytesPerRow, bytesPerImage);
        }

        void CopyBufferToBuffer(Buffer& src, Buffer& dst, uint64_t srcOffset, uint64_t dstOffset, uint64_t size)
        {
            agfxComputePassCopyBufferToBuffer(mPass, src, dst, srcOffset, dstOffset, size);
        }

        void CopyTextureToTexture(Texture& src, Texture& dst, const agfxTextureRegion& region, uint32_t mipLevel, uint32_t layer)
        {
            agfxComputePassCopyTextureToTexture(mPass, src, dst, &region, mipLevel, layer);
        }

        void SetPipeline(ComputePipeline& pipeline) { agfxComputePassSetPipeline(mPass, pipeline); }
        void PushConstants(const void* data, uint32_t size) { agfxComputePassPushConstants(mPass, data, size); }
        template<typename T> void PushConstants(const T& data) { PushConstants(&data, sizeof(T)); }
        void Dispatch(uint32_t x, uint32_t y, uint32_t z) { agfxComputePassDispatch(mPass, x, y, z); }

        operator agfxComputePass*() const { return mPass; }

    private:
        agfxComputePass* mPass = nullptr;
    };

    /// @brief RAII scope guard around an agfxRenderPass. Ends the pass automatically when it goes out of scope.
    class RenderPass
    {
    public:
        RenderPass() = default;
        explicit RenderPass(agfxRenderPass* pass) : mPass(pass) {}
        RenderPass(const RenderPass&) = delete;
        RenderPass& operator=(const RenderPass&) = delete;
        RenderPass(RenderPass&& other) noexcept : mPass(other.mPass) { other.mPass = nullptr; }
        RenderPass& operator=(RenderPass&& other) noexcept
        {
            if (this != &other) { End(); mPass = other.mPass; other.mPass = nullptr; }
            return *this;
        }

        ~RenderPass() { End(); }

        void End()
        {
            if (mPass) {
                agfxRenderPassEnd(mPass);
                mPass = nullptr;
            }
        }

        void SetViewport(float x, float y, float width, float height, float minDepth = 0.0f, float maxDepth = 1.0f)
        {
            agfxRenderPassSetViewport(mPass, x, y, width, height, minDepth, maxDepth);
        }

        void SetScissor(uint32_t x, uint32_t y, uint32_t width, uint32_t height) { agfxRenderPassSetScissor(mPass, x, y, width, height); }
        void SetPipeline(RenderPipeline& pipeline) { agfxRenderPassSetPipeline(mPass, pipeline); }
        void PushConstants(const void* data, uint32_t size) { agfxRenderPassPushConstants(mPass, data, size); }
        template<typename T> void PushConstants(const T& data) { PushConstants(&data, sizeof(T)); }

        void Draw(uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t firstVertex = 0, uint32_t firstInstance = 0)
        {
            agfxRenderPassDraw(mPass, vertexCount, instanceCount, firstVertex, firstInstance);
        }

        void DrawIndexed(Buffer& indexBuffer, uint32_t indexCount, uint32_t instanceCount = 1, uint32_t firstIndex = 0, uint32_t vertexOffset = 0, uint32_t firstInstance = 0)
        {
            agfxRenderPassDrawIndexed(mPass, indexBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        }

        void DrawMesh(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1)
        {
            agfxRenderPassDrawMesh(mPass, groupCountX, groupCountY, groupCountZ);
        }

        operator agfxRenderPass*() const { return mPass; }

    private:
        agfxRenderPass* mPass = nullptr;
    };

    class CommandBuffer : public Handle<agfxCommandBuffer, agfxCommandBufferDestroy>
    {
    public:
        using Handle::Handle;

        void Reset() { agfxCommandBufferReset(mHandle); }
        void Begin() { agfxCommandBufferBegin(mHandle); }
        void End() { agfxCommandBufferEnd(mHandle); }

        void TextureBarrier(Texture& texture, agfxResourceState oldState, agfxResourceState newState, uint32_t mip = AGFX_SUBRESOURCE_ALL_MIPS, uint32_t layer = AGFX_SUBRESOURCE_ALL_LAYERS, bool agglomerate = true)
        {
            agfxCommandBufferTextureBarrier(mHandle, texture, oldState, newState, mip, layer, agglomerate);
        }

        void BufferBarrier(Buffer& buffer, agfxResourceState oldState, agfxResourceState newState, bool agglomerate = true)
        {
            agfxCommandBufferBufferBarrier(mHandle, buffer, oldState, newState, agglomerate);
        }

        void WriteTimestamp(QueryPool& pool, uint32_t index) { agfxCommandBufferWriteTimestamp(mHandle, pool, index); }

        void ResolveQueryPool(QueryPool& pool, uint32_t firstIndex, uint32_t count)
        {
            agfxCommandBufferResolveQueryPool(mHandle, pool, firstIndex, count);
        }

        /// @brief Begins a compute pass, returned as a scope guard that ends it automatically.
        [[nodiscard]] ComputePass BeginComputePass(const char* name)
        {
            return ComputePass(agfxComputePassBegin(mHandle, name));
        }

        /// @brief Begins a render pass, returned as a scope guard that ends it automatically.
        [[nodiscard]] RenderPass BeginRenderPass(const agfxRenderPassCreateInfo& info)
        {
            return RenderPass(agfxRenderPassBegin(mHandle, &info));
        }
    };

    class CommandQueue : public Handle<agfxCommandQueue, agfxCommandQueueDestroy>
    {
    public:
        using Handle::Handle;

        void Signal(Fence& fence, uint64_t value) { agfxCommandQueueSignal(mHandle, fence, value); }
        void Wait(Fence& fence, uint64_t value) { agfxCommandQueueWait(mHandle, fence, value); }

        /// @brief Submits up to 64 command buffers at once.
        void Submit(CommandBuffer* const* commandBuffers, uint32_t count)
        {
            agfxCommandBuffer* raw[64];
            count = count > 64 ? 64 : count;
            for (uint32_t i = 0; i < count; ++i) {
                raw[i] = commandBuffers[i]->Get();
            }
            agfxCommandQueueSubmit(mHandle, raw, count);
        }

        void Submit(CommandBuffer& commandBuffer)
        {
            agfxCommandBuffer* raw = commandBuffer;
            agfxCommandQueueSubmit(mHandle, &raw, 1);
        }
    };

    class SwapChain : public Handle<agfxSwapChain, agfxSwapChainDestroy>
    {
    public:
        using Handle::Handle;

        void Resize(uint32_t width, uint32_t height) { agfxSwapChainResize(mDevice, mHandle, width, height); }
        agfxTextureFormat GetFormat() const { return agfxSwapChainGetFormat(mHandle); }

        /// @brief Acquires the next back buffer. The returned Texture is non-owning; the swap chain owns it.
        agfxTexture* AcquireNextTexture() { return agfxSwapChainAcquireNextTexture(mHandle); }

        void Present() { agfxSwapChainPresent(mHandle); }
    };

    /// @brief Owns and destroys an agfxDevice. The entry point for creating every other agfx object.
    class Device
    {
    public:
        Device() = default;

        explicit Device(const agfxDeviceCreateInfo& info) : mDevice(agfxDeviceCreate(&info)) {}

        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;

        Device(Device&& other) noexcept : mDevice(other.mDevice) { other.mDevice = nullptr; }
        Device& operator=(Device&& other) noexcept
        {
            if (this != &other) {
                Reset();
                mDevice = other.mDevice;
                other.mDevice = nullptr;
            }
            return *this;
        }

        ~Device() { Reset(); }

        void Reset()
        {
            if (mDevice) {
                agfxDeviceDestroy(mDevice);
                mDevice = nullptr;
            }
        }

        agfxDevice* Get() const { return mDevice; }
        operator agfxDevice*() const { return mDevice; }
        explicit operator bool() const { return mDevice != nullptr; }

        agfxDeviceInfo GetInfo() const
        {
            agfxDeviceInfo info{};
            agfxDeviceGetInfo(mDevice, &info);
            return info;
        }

        void MakeResourcesResident() { agfxDeviceMakeResourcesResident(mDevice); }

        Fence CreateFence() { return Fence(mDevice, agfxFenceCreate(mDevice)); }

        QueryPool CreateQueryPool(CommandQueue& queue, const agfxQueryPoolCreateInfo& info)
        {
            return QueryPool(mDevice, agfxQueryPoolCreate(mDevice, queue, &info));
        }

        CommandQueue CreateCommandQueue(agfxCommandQueueType type)
        {
            agfxCommandQueueCreateInfo info{type};
            return CommandQueue(mDevice, agfxCommandQueueCreate(mDevice, &info));
        }

        CommandBuffer CreateCommandBuffer(CommandQueue& queue)
        {
            return CommandBuffer(mDevice, agfxCommandBufferCreate(mDevice, queue));
        }

        Texture CreateTexture(const agfxTextureCreateInfo& info)
        {
            return Texture(mDevice, agfxTextureCreate(mDevice, &info));
        }

        Buffer CreateBuffer(const agfxBufferCreateInfo& info)
        {
            return Buffer(mDevice, agfxBufferCreate(mDevice, &info));
        }

        TextureView CreateTextureView(const agfxTextureViewCreateInfo& info)
        {
            return TextureView(mDevice, agfxTextureViewCreate(mDevice, &info));
        }

        Sampler CreateSampler(const agfxSamplerCreateInfo& info)
        {
            return Sampler(mDevice, agfxSamplerCreate(mDevice, &info));
        }

        BufferView CreateBufferView(const agfxBufferViewCreateInfo& info)
        {
            return BufferView(mDevice, agfxBufferViewCreate(mDevice, &info));
        }

        RenderTarget CreateRenderTarget(const agfxRenderTargetCreateInfo& info)
        {
            return RenderTarget(mDevice, agfxRenderTargetCreate(mDevice, &info));
        }

        SwapChain CreateSwapChain(const agfxSwapChainCreateInfo& info)
        {
            return SwapChain(mDevice, agfxSwapChainCreate(mDevice, &info));
        }

        ShaderModule CreateShaderModule(const agfxShaderModuleCreateInfo& info)
        {
            return ShaderModule(mDevice, agfxShaderModuleCreate(mDevice, &info));
        }

        RenderPipeline CreateRenderPipeline(const agfxRenderPipelineCreateInfo& info)
        {
            return RenderPipeline(mDevice, agfxRenderPipelineCreate(mDevice, &info));
        }

        ComputePipeline CreateComputePipeline(const agfxComputePipelineCreateInfo& info)
        {
            return ComputePipeline(mDevice, agfxComputePipelineCreate(mDevice, &info));
        }

    private:
        agfxDevice* mDevice = nullptr;
    };
}
