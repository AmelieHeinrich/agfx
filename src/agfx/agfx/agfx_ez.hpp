/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-20 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// agfx_ez.hpp: a D3D11/OpenGL-style "immediate context" layer built on top of agfx.hpp.
// Owns its own frame loop (agfx::ez::Context), caches render pipelines and resource views,
// offers one-call texture/buffer creation with synchronous upload, a ring-buffered dynamic
// constant allocator, and a bindless push-constant builder (agfx::ez::ShaderBindings).
//
// IMPORTANT — barrier tracking is best-effort, not automatic hazard tracking. AGFX is fully
// bindless: a shader can touch a resource through a handle at any point, and there is no
// host-visible hook to observe that. Context::TransitionTexture/TransitionBuffer only track
// the last state *this ez layer* moved a resource into via its own calls (SetRenderTargets,
// resource creation, explicit Transition* calls). It does NOT: see inside shader bindless
// reads, track resources created directly through raw agfx.hpp/agfx.h (mixing raw and ez
// creation for the same resource silently defeats tracking for it), operate at subresource
// (mip/layer) granularity, or infer intra-pass UAV read/write hazards (use the re-exposed
// TextureUAVBarrier/BufferUAVBarrier on agfx::ComputePass for that). This is convenience for
// the common "render to it, then sample it next pass" pattern, nothing more.

#pragma once

#include <agfx/agfx.hpp>

#include <array>
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <unordered_map>
#include <vector>

namespace agfx::ez
{
    inline constexpr float kDefaultClearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    namespace detail
    {
        inline uint64_t FnvHash(uint64_t hash, const void* data, size_t size)
        {
            const uint8_t* bytes = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                hash ^= bytes[i];
                hash *= 1099511628211ull;
            }
            return hash;
        }

        template<typename T>
        uint64_t FnvHashValue(uint64_t hash, const T& value)
        {
            return FnvHash(hash, &value, sizeof(T));
        }

        inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
        {
            return (value + alignment - 1) & ~(alignment - 1);
        }
    }

    /// @brief Tracks the last agfxResourceState an ez-wrapped resource was transitioned into.
    /// @note Only reflects transitions made through the ez API. See the header's top comment for scope.
    class ResourceStateTracker
    {
    public:
        agfxResourceState Current() const { return mState; }
        void Set(agfxResourceState state) { mState = state; }

    private:
        agfxResourceState mState = AGFX_RESOURCE_STATE_COMMON;
    };

    /// @brief A 2D texture with lazily created and cached SRV/RTV/UAV views.
    class Texture2D
    {
    public:
        Texture2D() = default;
        Texture2D(agfx::Texture&& texture, const agfxTextureCreateInfo& info)
            : mTexture(std::move(texture)), mInfo(info)
        {
        }

        Texture2D(const Texture2D&) = delete;
        Texture2D& operator=(const Texture2D&) = delete;
        Texture2D(Texture2D&&) = default;
        Texture2D& operator=(Texture2D&&) = default;

        agfx::Texture& Raw() { return mTexture; }
        uint32_t Width() const { return mInfo.width; }
        uint32_t Height() const { return mInfo.height; }
        agfxTextureFormat Format() const { return mInfo.format; }
        agfxTextureUsage Usage() const { return mInfo.usage; }

        agfxResourceState State() const { return mTracker.Current(); }
        void SetState(agfxResourceState state) { mTracker.Set(state); }

        /// @brief The sampled/shader-resource view. Created on first call.
        agfx::TextureView& SRV()
        {
            if (!mSRV) {
                agfxTextureViewCreateInfo info{};
                info.texture = mTexture;
                info.format = mInfo.format;
                info.type = mInfo.type;
                info.baseMipLevel = 0;
                info.mipLevelCount = mInfo.mipLevels;
                info.baseArrayLayer = 0;
                info.arrayLayerCount = mInfo.depthOrArrayLayers;
                info.writeable = 0;
                mSRV.emplace(mTexture.GetDevice(), agfxTextureViewCreate(mTexture.GetDevice(), &info));
            }
            return *mSRV;
        }

        /// @brief The read/write (UAV) view. Requires AGFX_TEXTURE_USAGE_STORAGE at creation. Created on first call.
        agfx::TextureView& UAV()
        {
            assert((mInfo.usage & AGFX_TEXTURE_USAGE_STORAGE) && "Texture2D::UAV() requires AGFX_TEXTURE_USAGE_STORAGE");
            if (!mUAV) {
                agfxTextureViewCreateInfo info{};
                info.texture = mTexture;
                info.format = mInfo.format;
                info.type = mInfo.type;
                info.baseMipLevel = 0;
                info.mipLevelCount = mInfo.mipLevels;
                info.baseArrayLayer = 0;
                info.arrayLayerCount = mInfo.depthOrArrayLayers;
                info.writeable = 1;
                mUAV.emplace(mTexture.GetDevice(), agfxTextureViewCreate(mTexture.GetDevice(), &info));
            }
            return *mUAV;
        }

        /// @brief The render-target (RTV/DSV) view. Requires a COLOR_ATTACHMENT or DEPTH_STENCIL_ATTACHMENT usage bit. Created on first call.
        agfx::RenderTarget& RTV()
        {
            assert((mInfo.usage & (AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT)) &&
                   "Texture2D::RTV() requires a color or depth-stencil attachment usage bit");
            if (!mRTV) {
                agfxRenderTargetCreateInfo info{};
                info.texture = mTexture;
                info.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
                info.mipLevel = 0;
                info.arrayLayer = 0;
                info.isDepth = (mInfo.usage & AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT) ? 1 : 0;
                mRTV.emplace(mTexture.GetDevice(), agfxRenderTargetCreate(mTexture.GetDevice(), &info));
            }
            return *mRTV;
        }

    private:
        agfx::Texture mTexture;
        agfxTextureCreateInfo mInfo{};
        ResourceStateTracker mTracker;
        std::optional<agfx::TextureView> mSRV;
        std::optional<agfx::TextureView> mUAV;
        std::optional<agfx::RenderTarget> mRTV;
    };

    /// @brief A buffer with lazily created and cached views (one per agfxBufferViewType).
    class Buffer
    {
    public:
        Buffer() = default;
        Buffer(agfx::Buffer&& buffer, const agfxBufferCreateInfo& info)
            : mBuffer(std::move(buffer)), mInfo(info)
        {
        }

        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&&) = default;
        Buffer& operator=(Buffer&&) = default;

        agfx::Buffer& Raw() { return mBuffer; }
        uint64_t Size() const { return mInfo.size; }
        uint64_t Stride() const { return mInfo.stride; }

        agfxResourceState State() const { return mTracker.Current(); }
        void SetState(agfxResourceState state) { mTracker.Set(state); }

        /// @brief The view of the given type. Created on first call for that type.
        agfx::BufferView& View(agfxBufferViewType type, bool writeable = false)
        {
            std::optional<agfx::BufferView>& slot = mViews[static_cast<size_t>(type)];
            if (!slot) {
                agfxBufferViewCreateInfo info{};
                info.buffer = mBuffer;
                info.type = type;
                info.offset = 0;
                info.writeable = writeable ? 1 : 0;
                slot.emplace(mBuffer.GetDevice(), agfxBufferViewCreate(mBuffer.GetDevice(), &info));
            }
            return *slot;
        }

    private:
        agfx::Buffer mBuffer;
        agfxBufferCreateInfo mInfo{};
        ResourceStateTracker mTracker;
        std::array<std::optional<agfx::BufferView>, 3> mViews;
    };

    namespace detail
    {
        /// @brief Header-only staging-buffer uploader (same pattern as agfx_demo's AgfxUploader),
        /// reimplemented here so agfx_ez.hpp has no dependency on agfx_demo. Not user-facing.
        class Uploader
        {
        public:
            Uploader(agfx::Device& device, agfx::CommandQueue& queue)
                : mDevice(&device), mQueue(&queue), mCommandBuffer(device.CreateCommandBuffer(queue)), mFence(device.CreateFence())
            {
            }

            // Note: uploaded resources are intentionally left in AGFX_RESOURCE_STATE_COMMON --
            // matching agfx_demo's own AgfxUploader/GltfScene usage, which never barriers a
            // freshly-uploaded buffer or bindless-read texture out of COMMON before reading it.
            // Emitting a COMMON -> read-only-state barrier here is both unnecessary (COMMON's
            // producer stages are 0, same as every "read" state's producer stages, so no future
            // barrier computed from the tracked state differs whether it says COMMON or the
            // "real" post-upload state) and actively broken on the Metal4 backend: any state
            // whose producer stages are 0 paired with a target whose consumer stages are
            // non-zero yields barrierAfterQueueStages:0, which Metal's validation layer rejects
            // outright (see agfxMetal4BarrierTracker::addBarrier/encode in agfx_metal4.mm).
            void UploadTexture(agfx::Texture& dst, const agfxTextureRegion& region, uint32_t mipLevel, uint32_t layer,
                                const void* data, uint32_t dataSize, uint32_t bytesPerRow, uint32_t bytesPerImage)
            {
                if (dataSize == 0) {
                    return;
                }
                EnsureRecording();
                agfx::Buffer staging = CreateStaging(dataSize, data);
                EnsurePass();
                mComputePass->CopyBufferToTexture(staging, dst, region, mipLevel, layer, bytesPerRow, bytesPerImage);
                mStagingBuffers.push_back(std::move(staging));
            }

            void UploadBuffer(agfx::Buffer& dst, uint64_t dstOffset, const void* data, uint64_t size)
            {
                if (size == 0) {
                    return;
                }
                EnsureRecording();
                agfx::Buffer staging = CreateStaging(size, data);
                EnsurePass();
                mComputePass->CopyBufferToBuffer(staging, dst, 0, dstOffset, size);
                mStagingBuffers.push_back(std::move(staging));
            }

            /// @brief Submits all pending copies, waits for completion, and frees the staging buffers.
            void Flush()
            {
                if (!mRecording) {
                    return;
                }

                mComputePass.reset();

                // Residency must be established before submission -- the staging buffers and
                // copy destinations this command buffer references have to already be resident
                // when the GPU executes it, not after.
                mDevice->MakeResourcesResident();

                mCommandBuffer.End();
                mQueue->Submit(mCommandBuffer);

                ++mFenceValue;
                mQueue->Signal(mFence, mFenceValue);
                mFence.Wait(mFenceValue, UINT64_MAX);

                mStagingBuffers.clear();
                mRecording = false;
            }

        private:
            agfx::Buffer CreateStaging(uint64_t size, const void* data)
            {
                agfxBufferCreateInfo stagingInfo{};
                stagingInfo.size = size;
                stagingInfo.stride = size;
                stagingInfo.usage = AGFX_BUFFER_USAGE_SHADER_READ;
                stagingInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU;
                agfx::Buffer staging = mDevice->CreateBuffer(stagingInfo);
                void* mapped = staging.Map();
                memcpy(mapped, data, size);
                staging.Unmap();
                return staging;
            }

            void EnsureRecording()
            {
                if (!mRecording) {
                    mCommandBuffer.Reset();
                    mCommandBuffer.Begin();
                    mRecording = true;
                }
            }

            void EnsurePass()
            {
                if (!mComputePass) {
                    mComputePass.emplace(mCommandBuffer.BeginComputePass("ez upload"));
                }
            }

            agfx::Device* mDevice;
            agfx::CommandQueue* mQueue;
            agfx::CommandBuffer mCommandBuffer;
            agfx::Fence mFence;
            uint64_t mFenceValue = 0;
            bool mRecording = false;
            std::optional<agfx::ComputePass> mComputePass;
            std::vector<agfx::Buffer> mStagingBuffers;
        };
    }

    /// @brief Ring-buffered, persistently mapped constant buffer allocator, rotated per frame-in-flight slot.
    /// @note Not usually constructed directly; owned by Context and driven through Context::AllocateConstants.
    class DynamicRingBuffer
    {
    public:
        DynamicRingBuffer() = default;

        DynamicRingBuffer(agfx::Device& device, uint64_t budgetPerFrame, uint32_t framesInFlight)
            : mBudgetPerFrame(budgetPerFrame), mFramesInFlight(framesInFlight)
        {
            agfxBufferCreateInfo info{};
            info.size = budgetPerFrame * framesInFlight;
            info.stride = 0;
            info.usage = agfxBufferUsage(AGFX_BUFFER_USAGE_CONSTANT | AGFX_BUFFER_USAGE_SHADER_READ);
            info.memoryType = AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU;
            mBuffer = device.CreateBuffer(info);
            mMapped = static_cast<uint8_t*>(mBuffer.Map());
            mViewPools.resize(framesInFlight);
        }

        DynamicRingBuffer(const DynamicRingBuffer&) = delete;
        DynamicRingBuffer& operator=(const DynamicRingBuffer&) = delete;
        DynamicRingBuffer(DynamicRingBuffer&&) = default;
        DynamicRingBuffer& operator=(DynamicRingBuffer&&) = default;

        void BeginFrame(uint32_t frameSlot)
        {
            mFrameSlot = frameSlot;
            mCursor = 0;
            mViewPools[frameSlot].clear(); // safe: this slot's fence has already been waited on by the caller
        }

        agfx::BufferView& Allocate(const void* data, uint64_t size)
        {
            assert(mCursor + size <= mBudgetPerFrame && "DynamicRingBuffer: per-frame constant budget exceeded");
            uint64_t slotBase = uint64_t(mFrameSlot) * mBudgetPerFrame;
            uint64_t offset = slotBase + mCursor;
            memcpy(mMapped + offset, data, size);
            mCursor += detail::AlignUp(size, 256);

            agfxBufferViewCreateInfo viewInfo{};
            viewInfo.buffer = mBuffer;
            viewInfo.type = AGFX_BUFFER_VIEW_TYPE_CONSTANT;
            viewInfo.offset = offset;
            viewInfo.writeable = 0;
            mViewPools[mFrameSlot].emplace_back(mBuffer.GetDevice(), agfxBufferViewCreate(mBuffer.GetDevice(), &viewInfo));
            return mViewPools[mFrameSlot].back();
        }

    private:
        agfx::Buffer mBuffer;
        uint8_t* mMapped = nullptr;
        uint64_t mBudgetPerFrame = 0;
        uint32_t mFramesInFlight = 0;
        uint32_t mFrameSlot = 0;
        uint64_t mCursor = 0;
        std::vector<std::vector<agfx::BufferView>> mViewPools;
    };

    /// @brief A 128-byte bindless push-constant builder, matching AGFX's fixed push-constant budget.
    /// @note Bindless handles from TextureView/BufferView/Sampler are widened to uint64_t only for a
    ///       uniform C ABI; only the lower 32 bits are ever meaningful, matching HLSL's `typedef uint ResourceHandle`.
    class ShaderBindings
    {
    public:
        static constexpr uint32_t kCapacity = 128;

        uint32_t Write(const void* data, uint32_t size)
        {
            assert(mCursor + size <= kCapacity && "ShaderBindings: push-constant buffer overflow (128 byte limit)");
            uint32_t offset = mCursor;
            uint32_t writable = (offset < kCapacity) ? std::min(size, kCapacity - offset) : 0;
            if (writable > 0) {
                memcpy(mBytes + offset, data, writable);
            }
            mCursor += size;
            return offset;
        }

        template<typename T>
        uint32_t Write(const T& value)
        {
            return Write(&value, static_cast<uint32_t>(sizeof(T)));
        }

        uint32_t BindTexture(agfx::TextureView& view)
        {
            uint32_t handle = static_cast<uint32_t>(view.GetHandle());
            return Write(handle);
        }

        uint32_t BindBuffer(agfx::BufferView& view)
        {
            uint32_t handle = static_cast<uint32_t>(view.GetHandle());
            return Write(handle);
        }

        uint32_t BindSampler(agfx::Sampler& sampler)
        {
            uint32_t handle = static_cast<uint32_t>(sampler.GetHandle());
            return Write(handle);
        }

        const void* Data() const { return mBytes; }
        uint32_t Size() const { return mCursor < kCapacity ? mCursor : kCapacity; }

    private:
        uint8_t mBytes[kCapacity] = {};
        uint32_t mCursor = 0;
    };

    /// @brief A simplified render pipeline description with D3D11-typical defaults.
    /// @note Uniform blend state across all bound color attachments; no per-attachment arrays,
    ///       no indirect draw support, no mesh/task shaders — advanced users drop to raw agfx.hpp.
    struct PipelineDesc
    {
        const char* name = "ez pipeline";
        agfx::ShaderModule* vertexShader = nullptr;
        agfx::ShaderModule* fragmentShader = nullptr;

        agfxFillMode fillMode = AGFX_FILL_MODE_SOLID;
        agfxCullMode cullMode = AGFX_CULL_MODE_BACK;
        agfxFrontFace frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
        agfxTopology topology = AGFX_TOPOLOGY_TRIANGLES;

        bool depthTestEnable = true;
        bool depthWriteEnable = true;
        agfxComparisonFunction depthCompareOp = AGFX_COMPARISON_FUNCTION_LESS;

        bool blendEnable = false;
        agfxBlendFactor srcBlend = AGFX_BLEND_FACTOR_SRC_ALPHA;
        agfxBlendFactor dstBlend = AGFX_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        agfxBlendOperation blendOp = AGFX_BLEND_OPERATION_ADD;
    };

    namespace detail
    {
        /// @brief Caches agfx::RenderPipeline objects keyed by a hash of everything that defines the PSO.
        class PipelineCache
        {
        public:
            agfx::RenderPipeline& GetOrCreate(agfx::Device& device, const PipelineDesc& desc,
                                               const agfxTextureFormat* colorFormats, uint32_t colorCount,
                                               agfxTextureFormat depthFormat, bool hasDepth)
            {
                uint64_t hash = 1469598103934665603ull;
                hash = FnvHashValue(hash, desc.fillMode);
                hash = FnvHashValue(hash, desc.cullMode);
                hash = FnvHashValue(hash, desc.frontFace);
                hash = FnvHashValue(hash, desc.topology);
                hash = FnvHashValue(hash, desc.depthTestEnable);
                hash = FnvHashValue(hash, desc.depthWriteEnable);
                hash = FnvHashValue(hash, desc.depthCompareOp);
                hash = FnvHashValue(hash, desc.blendEnable);
                hash = FnvHashValue(hash, desc.srcBlend);
                hash = FnvHashValue(hash, desc.dstBlend);
                hash = FnvHashValue(hash, desc.blendOp);
                hash = FnvHashValue(hash, desc.vertexShader);
                hash = FnvHashValue(hash, desc.fragmentShader);
                hash = FnvHash(hash, colorFormats, sizeof(agfxTextureFormat) * colorCount);
                hash = FnvHashValue(hash, colorCount);
                hash = FnvHashValue(hash, hasDepth ? depthFormat : AGFX_TEXTURE_FORMAT_UNKNOWN);

                auto it = mPipelines.find(hash);
                if (it != mPipelines.end()) {
                    return it->second;
                }

                agfxRenderPipelineCreateInfo info{};
                info.name = desc.name;
                info.supportsIndirect = 0;
                info.fillMode = desc.fillMode;
                info.cullMode = desc.cullMode;
                info.frontFace = desc.frontFace;
                info.topology = desc.topology;
                info.depthTestEnable = desc.depthTestEnable ? 1 : 0;
                info.depthWriteEnable = desc.depthWriteEnable ? 1 : 0;
                info.depthClampEnable = 0;
                info.depthCompareOp = desc.depthCompareOp;
                info.depthFormat = hasDepth ? depthFormat : AGFX_TEXTURE_FORMAT_UNKNOWN;
                info.colorAttachmentCount = colorCount;
                for (uint32_t i = 0; i < colorCount; ++i) {
                    info.colorFormats[i] = colorFormats[i];
                    info.blendEnable[i] = desc.blendEnable ? 1 : 0;
                    info.srcColorBlendFactor[i] = desc.srcBlend;
                    info.dstColorBlendFactor[i] = desc.dstBlend;
                    info.colorBlendOp[i] = desc.blendOp;
                    info.srcAlphaBlendFactor[i] = desc.srcBlend;
                    info.dstAlphaBlendFactor[i] = desc.dstBlend;
                    info.alphaBlendOp[i] = desc.blendOp;
                }
                info.vertexShader = *desc.vertexShader;
                info.fragmentShader = *desc.fragmentShader;

                auto [inserted, _] = mPipelines.emplace(hash, device.CreateRenderPipeline(info));
                return inserted->second;
            }

        private:
            std::unordered_map<uint64_t, agfx::RenderPipeline> mPipelines;
        };
    }

    class Context;

    /// @brief RAII guard for one frame, returned by Context::BeginFrame(). Ends the frame automatically
    /// on destruction. Context::BeginFrame()/EndFrame() may also be called directly (this guard just
    /// calls Context::EndFrame() from its destructor, so both styles drive the same state machine).
    class Frame
    {
    public:
        Frame() = default;
        explicit Frame(Context* context) : mContext(context) {}
        Frame(const Frame&) = delete;
        Frame& operator=(const Frame&) = delete;
        Frame(Frame&& other) noexcept : mContext(other.mContext) { other.mContext = nullptr; }
        Frame& operator=(Frame&& other) noexcept
        {
            if (this != &other) {
                End();
                mContext = other.mContext;
                other.mContext = nullptr;
            }
            return *this;
        }

        ~Frame() { End(); }

        void End();

    private:
        Context* mContext = nullptr;
    };

    struct ContextCreateInfo
    {
        agfxDeviceCreateInfo deviceInfo{};
        agfx::Device* existingDevice = nullptr; // if set, Context does not own/destroy this device
        void* windowHandle = nullptr;           // HWND on Windows, CAMetalLayer* on macOS
        uint32_t width = 0;
        uint32_t height = 0;
        bool vsync = true;
        bool hdr = false;
        uint32_t swapChainImageCount = 2;
        uint32_t framesInFlight = 3;
        uint64_t dynamicConstantsBudgetPerFrame = 4 * 1024 * 1024;
    };

    /// @brief The entry point of the ez layer. Owns the device (optionally), graphics queue, swap
    /// chain, and per-frame command buffers/fences, and exposes an immediate-mode rendering API.
    class Context
    {
    public:
        explicit Context(const ContextCreateInfo& info)
            : mFramesInFlight(info.framesInFlight), mWidth(info.width), mHeight(info.height)
        {
            if (info.existingDevice) {
                mDevice = info.existingDevice;
                mOwnsDevice = false;
            } else {
                mOwnedDevice = agfx::Device(info.deviceInfo);
                mDevice = &mOwnedDevice;
                mOwnsDevice = true;
            }

            mQueue = mDevice->CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
            mWindowHandle = info.windowHandle;

            agfxSwapChainCreateInfo swapChainInfo{};
            swapChainInfo.queue = mQueue;
            swapChainInfo.imageCount = info.swapChainImageCount;
            swapChainInfo.width = info.width;
            swapChainInfo.height = info.height;
            swapChainInfo.isHDR = info.hdr ? 1 : 0;
            swapChainInfo.vsync = info.vsync ? 1 : 0;
            swapChainInfo.handle = info.windowHandle;
            mSwapChain = mDevice->CreateSwapChain(swapChainInfo);

            mCommandBuffers.reserve(mFramesInFlight);
            mSlotFenceValues.assign(mFramesInFlight, 0);
            for (uint32_t i = 0; i < mFramesInFlight; ++i) {
                mCommandBuffers.push_back(mDevice->CreateCommandBuffer(mQueue));
            }
            mFrameFence = mDevice->CreateFence();

            mUploader.emplace(*mDevice, mQueue);
            mDynamicConstants = DynamicRingBuffer(*mDevice, info.dynamicConstantsBudgetPerFrame, mFramesInFlight);
        }

        ~Context()
        {
            if (mDevice) {
                DrainGPU();
            }
        }

        Context(const Context&) = delete;
        Context& operator=(const Context&) = delete;
        Context(Context&&) = delete;
        Context& operator=(Context&&) = delete;

        // --- Frame loop ---

        [[nodiscard]] Frame BeginFrame()
        {
            assert(!mFrameActive && "Context::BeginFrame() called while a frame is already active");
            mFrameActive = true;

            mFrameSlot = static_cast<uint32_t>(mFrameIndex % mFramesInFlight);
            mFrameFence.Wait(mSlotFenceValues[mFrameSlot], UINT64_MAX);

            agfx::CommandBuffer& cmd = mCommandBuffers[mFrameSlot];
            cmd.Reset();
            cmd.Begin();

            mDynamicConstants.BeginFrame(mFrameSlot);

            mBackBufferTexture = mSwapChain.AcquireNextTexture();
            agfxCommandBufferTextureBarrier(cmd.Get(), mBackBufferTexture, AGFX_RESOURCE_STATE_PRESENT, AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 0);

            return Frame(this);
        }

        void EndFrame()
        {
            assert(mFrameActive && "Context::EndFrame() called without an active frame");

            mActiveRenderPass.reset();
            mBackBufferRenderTarget.reset();

            agfx::CommandBuffer& cmd = mCommandBuffers[mFrameSlot];
            agfxCommandBufferTextureBarrier(cmd.Get(), mBackBufferTexture, AGFX_RESOURCE_STATE_RENDER_TARGET, AGFX_RESOURCE_STATE_PRESENT, 0, 0, 0);
            cmd.End();

            mQueue.Submit(cmd);
            mSwapChain.Present();

            ++mFenceValue;
            mQueue.Signal(mFrameFence, mFenceValue);
            mSlotFenceValues[mFrameSlot] = mFenceValue;

            ++mFrameIndex;
            mFrameActive = false;
        }

        // --- Resize / HDR toggle ---

        void Resize(uint32_t width, uint32_t height)
        {
            DrainGPU();
            mSwapChain.Resize(width, height);
            mWidth = width;
            mHeight = height;
        }

        void SetHDR(bool enabled, bool vsync)
        {
            DrainGPU();
            agfxSwapChainCreateInfo info{};
            info.queue = mQueue;
            info.imageCount = 2;
            info.width = mWidth;
            info.height = mHeight;
            info.isHDR = enabled ? 1 : 0;
            info.vsync = vsync ? 1 : 0;
            info.handle = mWindowHandle;
            mSwapChain = mDevice->CreateSwapChain(info);
        }

        // --- Immediate-mode rendering (call between BeginFrame/EndFrame) ---

        void SetRenderTargets(std::initializer_list<Texture2D*> colorTargets, Texture2D* depthTarget = nullptr,
                               agfxLoadOp loadOp = AGFX_LOAD_OPERATION_CLEAR, const float* clearColor = kDefaultClearColor,
                               agfxLoadOp depthLoadOp = AGFX_LOAD_OPERATION_CLEAR, float clearDepth = 1.0f)
        {
            assert(mFrameActive);
            mActiveRenderPass.reset();

            agfxRenderPassCreateInfo info{};
            uint32_t idx = 0;
            uint32_t w = 0, h = 0;
            for (Texture2D* target : colorTargets) {
                TransitionTexture(*target, AGFX_RESOURCE_STATE_RENDER_TARGET);
                agfxRenderPassAttachment& att = info.colorAttachments[idx];
                att.renderTarget = target->RTV();
                att.loadOp = loadOp;
                att.storeOp = AGFX_STORE_OPERATION_STORE;
                if (clearColor) {
                    memcpy(att.clearColor, clearColor, sizeof(float) * 4);
                }
                mBoundColorFormats[idx] = target->Format();
                w = target->Width();
                h = target->Height();
                ++idx;
            }
            info.colorAttachmentCount = idx;
            mBoundColorCount = idx;

            if (depthTarget) {
                TransitionTexture(*depthTarget, AGFX_RESOURCE_STATE_DEPTH_WRITE);
                info.hasDepthAttachment = 1;
                info.depthAttachment.renderTarget = depthTarget->RTV();
                info.depthAttachment.loadOp = depthLoadOp;
                info.depthAttachment.storeOp = AGFX_STORE_OPERATION_STORE;
                info.depthAttachment.clearColor[0] = clearDepth;
                mBoundDepthFormat = depthTarget->Format();
                mHasDepth = true;
                w = depthTarget->Width();
                h = depthTarget->Height();
            } else {
                mHasDepth = false;
            }

            info.width = w;
            info.height = h;
            info.name = "ez render pass";

            mActiveRenderPass.emplace(mCommandBuffers[mFrameSlot].BeginRenderPass(info));
        }

        void SetBackBufferRenderTarget(agfxLoadOp loadOp = AGFX_LOAD_OPERATION_CLEAR, const float* clearColor = kDefaultClearColor)
        {
            assert(mFrameActive);
            mActiveRenderPass.reset();
            mBackBufferRenderTarget.reset();

            agfxRenderTargetCreateInfo rtInfo{};
            rtInfo.texture = mBackBufferTexture;
            rtInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            mBackBufferRenderTarget.emplace(mDevice->CreateRenderTarget(rtInfo));

            agfxRenderPassCreateInfo info{};
            info.colorAttachmentCount = 1;
            info.colorAttachments[0].renderTarget = *mBackBufferRenderTarget;
            info.colorAttachments[0].loadOp = loadOp;
            info.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
            if (clearColor) {
                memcpy(info.colorAttachments[0].clearColor, clearColor, sizeof(float) * 4);
            }
            info.width = mWidth;
            info.height = mHeight;
            info.name = "ez back buffer pass";

            mBoundColorCount = 1;
            mBoundColorFormats[0] = mSwapChain.GetFormat();
            mHasDepth = false;

            mActiveRenderPass.emplace(mCommandBuffers[mFrameSlot].BeginRenderPass(info));
        }

        /// @brief Ends the active render pass without starting a new one. Required before recording a
        /// compute pass on the same command buffer (e.g. via GetCurrentCommandBuffer().BeginComputePass()).
        void EndActivePass() { mActiveRenderPass.reset(); }

        /// @brief The currently active render pass, e.g. to hand to ImGui_ImplAGFX_RenderDrawData.
        agfx::RenderPass& GetActiveRenderPass()
        {
            assert(mActiveRenderPass);
            return *mActiveRenderPass;
        }

        uint32_t CurrentFrameSlot() const { return mFrameSlot; }

        void SetViewportScissor(float x, float y, float w, float h)
        {
            assert(mActiveRenderPass);
            mActiveRenderPass->SetViewport(x, y, w, h);
            mActiveRenderPass->SetScissor(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(w), static_cast<uint32_t>(h));
        }

        void SetPipeline(const PipelineDesc& desc)
        {
            assert(mActiveRenderPass && "Context::SetPipeline() requires SetRenderTargets()/SetBackBufferRenderTarget() to be called first");
            agfx::RenderPipeline& pipeline = mPipelineCache.GetOrCreate(*mDevice, desc, mBoundColorFormats.data(), mBoundColorCount, mBoundDepthFormat, mHasDepth);
            mActiveRenderPass->SetPipeline(pipeline);
        }

        void PushShaderBindings(const ShaderBindings& bindings)
        {
            assert(mActiveRenderPass);
            mActiveRenderPass->PushConstants(bindings.Data(), bindings.Size());
        }

        void Draw(uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t firstVertex = 0, uint32_t firstInstance = 0)
        {
            assert(mActiveRenderPass);
            mActiveRenderPass->Draw(vertexCount, instanceCount, firstVertex, firstInstance);
        }

        void DrawIndexed(Buffer& indexBuffer, uint32_t indexCount, uint32_t instanceCount = 1,
                          uint32_t firstIndex = 0, uint32_t vertexOffset = 0, uint32_t firstInstance = 0)
        {
            assert(mActiveRenderPass);
            mActiveRenderPass->DrawIndexed(indexBuffer.Raw(), indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        }

        // --- One-call resource creation (immediate, synchronous upload) ---

        Texture2D CreateTexture2D(uint32_t width, uint32_t height, agfxTextureFormat format, agfxTextureUsage usage,
                                   const void* pixels = nullptr, uint32_t bytesPerRow = 0)
        {
            agfxTextureCreateInfo info{};
            info.type = AGFX_TEXTURE_TYPE_2D;
            info.format = format;
            info.usage = static_cast<agfxTextureUsage>(usage | AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = width;
            info.height = height;
            info.depthOrArrayLayers = 1;
            info.mipLevels = 1;

            Texture2D texture(mDevice->CreateTexture(info), info);

            if (pixels) {
                uint32_t rowBytes = bytesPerRow ? bytesPerRow : width * BytesPerPixel(format);
                agfxTextureRegion region{0, 0, 0, width, height, 1};
                mUploader->UploadTexture(texture.Raw(), region, 0, 0, pixels, rowBytes * height, rowBytes, rowBytes * height);
                mUploader->Flush();
                // Left in AGFX_RESOURCE_STATE_COMMON (the tracker's default) -- see the note on
                // detail::Uploader for why no post-upload barrier is emitted or needed here.
            }

            // Every resource -- including render-target/storage-only textures with no initial
            // data, which never otherwise go through the uploader's Flush() -- must be made
            // GPU-resident before use, or the GPU faults trying to access it.
            mDevice->MakeResourcesResident();

            return texture;
        }

        Buffer CreateVertexBuffer(const void* data, uint64_t size, uint64_t stride)
        {
            return CreateInitializedBuffer(data, size, stride, AGFX_BUFFER_USAGE_SHADER_READ);
        }

        Buffer CreateIndexBuffer(const void* data, uint64_t size)
        {
            return CreateInitializedBuffer(data, size, 0, AGFX_BUFFER_USAGE_INDEX);
        }

        Buffer CreateStructuredBuffer(const void* data, uint64_t size, uint64_t stride, bool shaderWritable = false)
        {
            agfxBufferUsage usage = static_cast<agfxBufferUsage>(AGFX_BUFFER_USAGE_SHADER_READ | (shaderWritable ? AGFX_BUFFER_USAGE_SHADER_WRITE : 0));
            return CreateInitializedBuffer(data, size, stride, usage);
        }

        // --- Per-frame dynamic constants ---

        agfx::BufferView& AllocateConstants(const void* data, size_t size) { return mDynamicConstants.Allocate(data, size); }

        template<typename T>
        agfx::BufferView& AllocateConstants(const T& data)
        {
            return AllocateConstants(&data, sizeof(T));
        }

        // --- Explicit barrier escape hatch ---

        void TransitionTexture(Texture2D& tex, agfxResourceState newState)
        {
            if (tex.State() == newState) {
                return;
            }
            mCommandBuffers[mFrameSlot].TextureBarrier(tex.Raw(), tex.State(), newState, AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
            tex.SetState(newState);
        }

        void TransitionBuffer(Buffer& buf, agfxResourceState newState)
        {
            if (buf.State() == newState) {
                return;
            }
            mCommandBuffers[mFrameSlot].BufferBarrier(buf.Raw(), buf.State(), newState, true);
            buf.SetState(newState);
        }

        // --- Raw access for advanced/mixed use ---

        agfx::Device& GetDevice() { return *mDevice; }
        agfx::CommandQueue& GetGraphicsQueue() { return mQueue; }
        agfx::CommandBuffer& GetCurrentCommandBuffer() { return mCommandBuffers[mFrameSlot]; }
        agfxTextureFormat GetSwapChainFormat() const { return mSwapChain.GetFormat(); }

    private:
        static uint32_t BytesPerPixel(agfxTextureFormat format)
        {
            switch (format) {
            case AGFX_TEXTURE_FORMAT_R8_UNORM: return 1;
            case AGFX_TEXTURE_FORMAT_RG8_UNORM: return 2;
            case AGFX_TEXTURE_FORMAT_RGBA8_UNORM:
            case AGFX_TEXTURE_FORMAT_BGRA8_UNORM:
            case AGFX_TEXTURE_FORMAT_RGBA8_UNORM_SRGB:
            case AGFX_TEXTURE_FORMAT_BGRA8_UNORM_SRGB:
            case AGFX_TEXTURE_FORMAT_R32F:
                return 4;
            case AGFX_TEXTURE_FORMAT_R16_UNORM:
            case AGFX_TEXTURE_FORMAT_R16F:
                return 2;
            case AGFX_TEXTURE_FORMAT_RG16_UNORM:
            case AGFX_TEXTURE_FORMAT_RG16F:
            case AGFX_TEXTURE_FORMAT_RG32F:
                return 4;
            case AGFX_TEXTURE_FORMAT_RGBA16_UNORM:
            case AGFX_TEXTURE_FORMAT_RGBA16F:
                return 8;
            case AGFX_TEXTURE_FORMAT_RGBA32F:
                return 16;
            case AGFX_TEXTURE_FORMAT_DEPTH32F:
                return 4;
            default:
                assert(false && "BytesPerPixel: block-compressed formats require an explicit bytesPerRow");
                return 4;
            }
        }

        Buffer CreateInitializedBuffer(const void* data, uint64_t size, uint64_t stride, agfxBufferUsage usage)
        {
            agfxBufferCreateInfo info{};
            info.size = size;
            info.stride = stride;
            info.usage = usage;
            info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;

            Buffer buffer(mDevice->CreateBuffer(info), info);

            if (data) {
                mUploader->UploadBuffer(buffer.Raw(), 0, data, size);
                mUploader->Flush();
                // Left in AGFX_RESOURCE_STATE_COMMON (the tracker's default) -- see the note on
                // detail::Uploader for why no post-upload barrier is emitted or needed here.
            }

            mDevice->MakeResourcesResident();

            return buffer;
        }

        void DrainGPU()
        {
            ++mFenceValue;
            mQueue.Signal(mFrameFence, mFenceValue);
            mFrameFence.Wait(mFenceValue, UINT64_MAX);
        }

        agfx::Device mOwnedDevice;
        agfx::Device* mDevice = nullptr;
        bool mOwnsDevice = true;

        agfx::CommandQueue mQueue;
        agfx::SwapChain mSwapChain;
        void* mWindowHandle = nullptr;
        uint32_t mWidth = 0, mHeight = 0;

        uint32_t mFramesInFlight = 3;
        std::vector<agfx::CommandBuffer> mCommandBuffers;
        agfx::Fence mFrameFence;
        std::vector<uint64_t> mSlotFenceValues;
        uint64_t mFenceValue = 0;
        uint64_t mFrameIndex = 0;
        uint32_t mFrameSlot = 0;
        bool mFrameActive = false;

        agfxTexture* mBackBufferTexture = nullptr;
        std::optional<agfx::RenderTarget> mBackBufferRenderTarget;
        std::optional<agfx::RenderPass> mActiveRenderPass;

        std::array<agfxTextureFormat, 8> mBoundColorFormats{};
        uint32_t mBoundColorCount = 0;
        agfxTextureFormat mBoundDepthFormat = AGFX_TEXTURE_FORMAT_UNKNOWN;
        bool mHasDepth = false;

        std::optional<detail::Uploader> mUploader;
        detail::PipelineCache mPipelineCache;
        DynamicRingBuffer mDynamicConstants;
    };

    inline void Frame::End()
    {
        if (mContext) {
            mContext->EndFrame();
            mContext = nullptr;
        }
    }
}
