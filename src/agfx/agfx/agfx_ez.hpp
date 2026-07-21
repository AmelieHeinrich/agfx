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
// creation for the same resource silently defeats tracking for it), or infer intra-pass UAV
// read/write hazards (use the re-exposed TextureUAVBarrier/BufferUAVBarrier on agfx::ComputePass
// for that). It DOES track at subresource (mip/layer) granularity -- TransitionTexture takes an
// optional mip and layer, so mip-chain patterns are expressible. This is convenience for the
// common "render to it, then sample it next pass" pattern, nothing more.

#pragma once

#include <agfx/agfx.hpp>

#include <array>
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <memory>
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

    /// @brief Selects "every mip" / "every layer" in the subresource-taking ez entry points.
    inline constexpr uint32_t kAllMips = AGFX_SUBRESOURCE_ALL_MIPS;
    inline constexpr uint32_t kAllLayers = AGFX_SUBRESOURCE_ALL_LAYERS;

    /// @brief Tracks the last agfxResourceState an ez-wrapped resource was transitioned into, at
    /// (mip, layer) subresource granularity.
    /// @note Only reflects transitions made through the ez API. See the header's top comment for scope.
    ///
    /// The representation is uniform-or-split: while every subresource agrees (the overwhelmingly
    /// common case, and the only one buffers ever hit) a single agfxResourceState is stored and no
    /// allocation happens. The first per-subresource transition splits it into one state per
    /// subresource; it re-collapses as soon as they agree again, so whole-resource transitions stay
    /// a single barrier.
    class ResourceStateTracker
    {
    public:
        /// @brief Sets the subresource grid. Not needed for buffers, which only use the uniform path.
        void Init(uint32_t mipLevels, uint32_t arrayLayers)
        {
            mMipLevels = mipLevels ? mipLevels : 1;
            mArrayLayers = arrayLayers ? arrayLayers : 1;
        }

        uint32_t MipLevels() const { return mMipLevels; }
        uint32_t ArrayLayers() const { return mArrayLayers; }

        /// @brief True while every subresource shares one state, so Current() fully describes it.
        bool Uniform() const { return mSplit.empty(); }

        /// @brief The shared state. Only meaningful when Uniform(); asserts otherwise.
        agfxResourceState Current() const
        {
            assert(Uniform() && "ResourceStateTracker::Current() on a split tracker -- use At(mip, layer)");
            return mState;
        }

        agfxResourceState At(uint32_t mip, uint32_t layer) const
        {
            if (Uniform()) {
                return mState;
            }
            return mSplit[Index(mip, layer)];
        }

        /// @brief Collapses every subresource back onto one state.
        void Set(agfxResourceState state)
        {
            mSplit.clear();
            mState = state;
        }

        /// @brief Sets one subresource, splitting the representation if it disagrees with the rest.
        void SetAt(uint32_t mip, uint32_t layer, agfxResourceState state)
        {
            if (Uniform()) {
                if (mState == state) {
                    return;
                }
                mSplit.assign((size_t)mMipLevels * mArrayLayers, mState);
            }
            mSplit[Index(mip, layer)] = state;
            Collapse();
        }

    private:
        size_t Index(uint32_t mip, uint32_t layer) const
        {
            assert(mip < mMipLevels && layer < mArrayLayers);
            return (size_t)mip * mArrayLayers + layer;
        }

        /// @brief Drops back to the uniform representation once every subresource agrees again.
        void Collapse()
        {
            for (agfxResourceState state : mSplit) {
                if (state != mSplit[0]) {
                    return;
                }
            }
            mState = mSplit[0];
            mSplit.clear();
        }

        agfxResourceState mState = AGFX_RESOURCE_STATE_COMMON;
        uint32_t mMipLevels = 1;
        uint32_t mArrayLayers = 1;
        std::vector<agfxResourceState> mSplit; // Empty while uniform.
    };

    namespace detail
    {
        /// @brief Storage and lazily created, cached views shared by every ez texture type.
        ///
        /// The public Texture2D/Texture2DArray/Texture3D/TextureCube wrappers differ only in which
        /// agfxTextureType they are created with and which dimension accessors make sense on them;
        /// everything below -- view creation, view caching, state tracking -- is identical, so it
        /// lives here once. Views are cached on their full parameter set rather than one slot each,
        /// so per-mip UAVs and per-mip render targets coexist with the whole-resource views.
        class TextureBase
        {
        public:
            TextureBase() = default;
            TextureBase(agfx::Texture&& texture, const agfxTextureCreateInfo& info)
                : mTexture(std::move(texture)), mInfo(info)
            {
                mTracker.Init(info.mipLevels, TrackedLayers(info));
            }

            TextureBase(const TextureBase&) = delete;
            TextureBase& operator=(const TextureBase&) = delete;
            TextureBase(TextureBase&&) = default;
            TextureBase& operator=(TextureBase&&) = default;

            agfx::Texture& Raw() { return mTexture; }
            const agfxTextureCreateInfo& Info() const { return mInfo; }
            uint32_t Width() const { return mInfo.width; }
            uint32_t Height() const { return mInfo.height; }
            uint32_t MipLevels() const { return mInfo.mipLevels; }
            agfxTextureFormat Format() const { return mInfo.format; }
            agfxTextureUsage Usage() const { return mInfo.usage; }

            ResourceStateTracker& Tracker() { return mTracker; }
            const ResourceStateTracker& Tracker() const { return mTracker; }

            /// @brief The whole-resource state. Only valid while no per-subresource transition has
            /// split it; use StateAt(mip, layer) in mip-chain-style code that transitions one at a time.
            agfxResourceState State() const { return mTracker.Current(); }
            agfxResourceState StateAt(uint32_t mip, uint32_t layer) const { return mTracker.At(mip, layer); }
            void SetState(agfxResourceState state) { mTracker.Set(state); }

            /// @brief A sampled/shader-resource view. Defaults to the whole resource. Cached per subrange.
            agfx::TextureView& SRV(uint32_t baseMip = 0, uint32_t mipCount = kAllMips,
                                   uint32_t baseLayer = 0, uint32_t layerCount = kAllLayers)
            {
                return View(baseMip, mipCount, baseLayer, layerCount, /*writeable*/ false);
            }

            /// @brief A read/write (UAV) view. Requires AGFX_TEXTURE_USAGE_STORAGE at creation.
            /// Pass (mip, 1) to get the single-mip UAV that mip-chain generation needs.
            agfx::TextureView& UAV(uint32_t baseMip = 0, uint32_t mipCount = kAllMips,
                                   uint32_t baseLayer = 0, uint32_t layerCount = kAllLayers)
            {
                assert((mInfo.usage & AGFX_TEXTURE_USAGE_STORAGE) && "UAV() requires AGFX_TEXTURE_USAGE_STORAGE");
                return View(baseMip, mipCount, baseLayer, layerCount, /*writeable*/ true);
            }

            /// @brief A render-target (RTV/DSV) view for one mip of one layer. Requires a
            /// COLOR_ATTACHMENT or DEPTH_STENCIL_ATTACHMENT usage bit. Cached per (mip, layer).
            agfx::RenderTarget& RTV(uint32_t mipLevel = 0, uint32_t arrayLayer = 0)
            {
                assert((mInfo.usage & (AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT)) &&
                       "RTV() requires a color or depth-stencil attachment usage bit");
                for (auto& entry : mRenderTargets) {
                    if (entry.first.mipLevel == mipLevel && entry.first.arrayLayer == arrayLayer) {
                        return *entry.second;
                    }
                }

                agfxRenderTargetCreateInfo info{};
                info.texture = mTexture;
                info.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
                info.mipLevel = mipLevel;
                info.arrayLayer = arrayLayer;
                info.isDepth = (mInfo.usage & AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT) ? 1 : 0;

                mRenderTargets.emplace_back(RenderTargetKey{mipLevel, arrayLayer},
                                            std::make_unique<agfx::RenderTarget>(
                                                mTexture.GetDevice(), agfxRenderTargetCreate(mTexture.GetDevice(), &info)));
                return *mRenderTargets.back().second;
            }

        protected:
            /// @brief How many subresource layers to track. A 3D texture's depthOrArrayLayers is
            /// depth, not layers -- its slices are addressed by copy-region z and it has exactly one
            /// subresource layer, so it must not be tracked as depthOrArrayLayers of them.
            static uint32_t TrackedLayers(const agfxTextureCreateInfo& info)
            {
                return info.type == AGFX_TEXTURE_TYPE_3D ? 1u : info.depthOrArrayLayers;
            }

        private:
            struct ViewKey
            {
                uint32_t baseMip;
                uint32_t mipCount;
                uint32_t baseLayer;
                uint32_t layerCount;
                bool writeable;

                bool operator==(const ViewKey& other) const
                {
                    return baseMip == other.baseMip && mipCount == other.mipCount &&
                           baseLayer == other.baseLayer && layerCount == other.layerCount &&
                           writeable == other.writeable;
                }
            };

            struct RenderTargetKey
            {
                uint32_t mipLevel;
                uint32_t arrayLayer;
            };

            agfx::TextureView& View(uint32_t baseMip, uint32_t mipCount, uint32_t baseLayer, uint32_t layerCount,
                                    bool writeable)
            {
                if (mipCount == kAllMips) {
                    mipCount = mInfo.mipLevels - baseMip;
                }
                if (layerCount == kAllLayers) {
                    layerCount = TrackedLayers(mInfo) - baseLayer;
                }

                const ViewKey key{baseMip, mipCount, baseLayer, layerCount, writeable};
                for (auto& entry : mViews) {
                    if (entry.first == key) {
                        return *entry.second;
                    }
                }

                agfxTextureViewCreateInfo info{};
                info.texture = mTexture;
                info.format = mInfo.format;
                info.type = mInfo.type;
                info.baseMipLevel = baseMip;
                info.mipLevelCount = mipCount;
                info.baseArrayLayer = baseLayer;
                info.arrayLayerCount = layerCount;
                info.writeable = writeable ? 1 : 0;

                mViews.emplace_back(key, std::make_unique<agfx::TextureView>(
                                             mTexture.GetDevice(), agfxTextureViewCreate(mTexture.GetDevice(), &info)));
                return *mViews.back().second;
            }

            agfx::Texture mTexture;
            agfxTextureCreateInfo mInfo{};
            ResourceStateTracker mTracker;
            // Linear search: a texture accumulates a handful of views at most, and these are looked
            // up far less often than the per-draw pipeline cache is. Held by pointer so that adding
            // a view never reallocates a reference an earlier SRV()/UAV()/RTV() call handed out.
            std::vector<std::pair<ViewKey, std::unique_ptr<agfx::TextureView>>> mViews;
            std::vector<std::pair<RenderTargetKey, std::unique_ptr<agfx::RenderTarget>>> mRenderTargets;
        };
    } // namespace detail

    /// @brief A 2D texture, optionally mipped, with lazily created and cached SRV/RTV/UAV views.
    class Texture2D : public detail::TextureBase
    {
    public:
        using TextureBase::TextureBase;
    };

    /// @brief An array of 2D textures. Layers are addressed by the layer parameter on views,
    /// render targets, uploads and copies.
    class Texture2DArray : public detail::TextureBase
    {
    public:
        using TextureBase::TextureBase;

        uint32_t ArrayLayers() const { return Info().depthOrArrayLayers; }
    };

    /// @brief A volume texture. Its depth slices are addressed by an agfxTextureRegion's z/depth,
    /// *not* by the layer parameter -- a 3D texture has exactly one subresource layer per mip.
    class Texture3D : public detail::TextureBase
    {
    public:
        using TextureBase::TextureBase;

        uint32_t Depth() const { return Info().depthOrArrayLayers; }
    };

    /// @brief A cube map: six square layers, ordered +X, -X, +Y, -Y, +Z, -Z.
    class TextureCube : public detail::TextureBase
    {
    public:
        using TextureBase::TextureBase;

        uint32_t ArrayLayers() const { return Info().depthOrArrayLayers; }
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

    /// @brief GPU-driven indirect draw/dispatch commands buffer + count buffer. Fill it from a
    /// compute pass via the AGFXIndirectDraw*Bundle HLSL helpers, then replay it with
    /// Context::ExecuteIndirectBundle (draw/drawIndexed/drawMesh) or, for dispatch bundles, the raw
    /// agfx::ComputePass::ExecuteIndirectBundle on a compute pass opened via GetCurrentCommandBuffer().
    class IndirectBundle
    {
    public:
        IndirectBundle() = default;
        explicit IndirectBundle(agfx::IndirectBundle&& bundle) : mBundle(std::move(bundle)) {}

        IndirectBundle(const IndirectBundle&) = delete;
        IndirectBundle& operator=(const IndirectBundle&) = delete;
        IndirectBundle(IndirectBundle&&) = default;
        IndirectBundle& operator=(IndirectBundle&&) = default;

        agfx::IndirectBundle& Raw() { return mBundle; }

        /// @brief Bindless handle to pass into the AGFXIndirectDraw*Bundle::Create HLSL helpers.
        uint64_t GetHandle() const { return const_cast<agfx::IndirectBundle&>(mBundle).GetHandle(); }

        agfxResourceState State() const { return mTracker.Current(); }
        void SetState(agfxResourceState state) { mTracker.Set(state); }

    private:
        agfx::IndirectBundle mBundle;
        ResourceStateTracker mTracker; // shared for both the commands and count buffers, which always transition together
    };

    /// @brief A built acceleration structure (bottom- or top-level). Created and built in one call
    /// via Context::CreateBottomLevel / CreateTopLevel, which hide scratch allocation, residency,
    /// and the AS->AS build barriers. Bind() gives the bindless handle to write into a ShaderBindings.
    class AccelerationStructure
    {
    public:
        AccelerationStructure() = default;
        explicit AccelerationStructure(agfx::AccelerationStructure&& as) : mAS(std::move(as)) {}

        AccelerationStructure(const AccelerationStructure&) = delete;
        AccelerationStructure& operator=(const AccelerationStructure&) = delete;
        AccelerationStructure(AccelerationStructure&&) = default;
        AccelerationStructure& operator=(AccelerationStructure&&) = default;

        agfx::AccelerationStructure& Raw() { return mAS; }

        /// @brief True once a valid structure has been built (false on non-RT devices).
        bool Valid() const { return static_cast<bool>(const_cast<agfx::AccelerationStructure&>(mAS)); }

        /// @brief Bindless handle to trace against; write into a ShaderBindings slot. 0 if invalid.
        uint32_t Bind() { return mAS ? static_cast<uint32_t>(mAS.GetHandle()) : 0u; }

    private:
        agfx::AccelerationStructure mAS;
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
                // Unlike the post-copy barrier this class deliberately skips (see the class comment),
                // this one isn't optional: CopyBufferToTexture requires the destination to actually be
                // in COPY_DEST, so it's transitioned in and back out explicitly around the copy.
                mCommandBuffer.TextureBarrier(dst, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_DEST,
                                              mipLevel, layer, false);
                EnsurePass();
                mComputePass->CopyBufferToTexture(staging, dst, region, mipLevel, layer, bytesPerRow, bytesPerImage);
                mComputePass.reset();
                mCommandBuffer.TextureBarrier(dst, AGFX_RESOURCE_STATE_COPY_DEST, AGFX_RESOURCE_STATE_COMMON,
                                              mipLevel, layer, false);
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
    ///       no indirect draw support — advanced users drop to raw agfx.hpp.
    /// @note Either set vertexShader (classic pipeline, draw with Context::Draw/DrawIndexed) or
    ///       meshShader (and optionally taskShader; draw with Context::DrawMesh) — not both.
    struct PipelineDesc
    {
        const char* name = "ez pipeline";
        agfx::ShaderModule* vertexShader = nullptr;
        agfx::ShaderModule* fragmentShader = nullptr;
        agfx::ShaderModule* taskShader = nullptr;
        agfx::ShaderModule* meshShader = nullptr;
        // Thread-group sizes for meshShader/taskShader, required on the Metal backend for dispatch.
        // Not reflected by ez -- copy from agfxShaderCompilerResult::meshSizeX/Y/Z (taskSizeX/Y/Z)
        // returned by agfxCompileShader when compiling the mesh/task shader; must match the
        // shader's declared [numthreads(...)] or dispatch counts disagree between backends.
        uint32_t meshGroupSizeX = 1, meshGroupSizeY = 1, meshGroupSizeZ = 1;
        uint32_t taskGroupSizeX = 1, taskGroupSizeY = 1, taskGroupSizeZ = 1;

        agfxFillMode fillMode = AGFX_FILL_MODE_SOLID;
        agfxCullMode cullMode = AGFX_CULL_MODE_BACK;
        agfxFrontFace frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
        agfxTopology topology = AGFX_TOPOLOGY_TRIANGLES;

        bool depthTestEnable = true;
        bool depthWriteEnable = true;
        /// @brief Clamps fragments to the viewport's depth range instead of clipping them against
        /// the near/far planes. The D3D11 rasterizer's DepthClipEnable, inverted.
        bool depthClampEnable = false;
        agfxComparisonFunction depthCompareOp = AGFX_COMPARISON_FUNCTION_LESS;

        /// @brief Allows this pipeline to be referenced from an indirect bundle
        /// (Context::ExecuteIndirectBundle). Off by default because it constrains the PSO on the
        /// Metal backend and most pipelines never replay indirectly.
        /// @note Required for indirect replay on Metal, where a PSO without it cannot legally be
        ///       encoded into an MTLIndirectCommandBuffer -- the failure is a GPU address fault
        ///       rather than a validation message. D3D12 has no equivalent requirement, so leaving
        ///       this off renders correctly on Windows and faults on macOS.
        bool supportsIndirect = false;

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
                hash = FnvHashValue(hash, desc.depthClampEnable);
                hash = FnvHashValue(hash, desc.depthCompareOp);
                hash = FnvHashValue(hash, desc.supportsIndirect);
                hash = FnvHashValue(hash, desc.blendEnable);
                hash = FnvHashValue(hash, desc.srcBlend);
                hash = FnvHashValue(hash, desc.dstBlend);
                hash = FnvHashValue(hash, desc.blendOp);
                hash = FnvHashValue(hash, desc.vertexShader);
                hash = FnvHashValue(hash, desc.fragmentShader);
                hash = FnvHashValue(hash, desc.taskShader);
                hash = FnvHashValue(hash, desc.meshShader);
                hash = FnvHashValue(hash, desc.meshGroupSizeX);
                hash = FnvHashValue(hash, desc.meshGroupSizeY);
                hash = FnvHashValue(hash, desc.meshGroupSizeZ);
                hash = FnvHashValue(hash, desc.taskGroupSizeX);
                hash = FnvHashValue(hash, desc.taskGroupSizeY);
                hash = FnvHashValue(hash, desc.taskGroupSizeZ);
                hash = FnvHash(hash, colorFormats, sizeof(agfxTextureFormat) * colorCount);
                hash = FnvHashValue(hash, colorCount);
                hash = FnvHashValue(hash, hasDepth ? depthFormat : AGFX_TEXTURE_FORMAT_UNKNOWN);

                auto it = mPipelines.find(hash);
                if (it != mPipelines.end()) {
                    return it->second;
                }

                agfxRenderPipelineCreateInfo info{};
                info.name = desc.name;
                info.supportsIndirect = desc.supportsIndirect ? 1 : 0;
                info.fillMode = desc.fillMode;
                info.cullMode = desc.cullMode;
                info.frontFace = desc.frontFace;
                info.topology = desc.topology;
                info.depthTestEnable = desc.depthTestEnable ? 1 : 0;
                info.depthWriteEnable = desc.depthWriteEnable ? 1 : 0;
                info.depthClampEnable = desc.depthClampEnable ? 1 : 0;
                info.depthCompareOp = desc.depthCompareOp;
                info.depthFormat = hasDepth ? depthFormat : AGFX_TEXTURE_FORMAT_UNKNOWN;
                // The *_COLOR blend factors manipulate RGB and are rejected outright by D3D12 when
                // used as an alpha factor, so mirror the color factor onto alpha via its
                // alpha-channel analog (e.g. SRC_COLOR -> SRC_ALPHA) rather than passing it through
                // unchanged. ZERO/ONE and the already-alpha factors are unaffected.
                auto alphaEquivalent = [](agfxBlendFactor factor) {
                    switch (factor) {
                        case AGFX_BLEND_FACTOR_SRC_COLOR:           return AGFX_BLEND_FACTOR_SRC_ALPHA;
                        case AGFX_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return AGFX_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                        case AGFX_BLEND_FACTOR_DST_COLOR:           return AGFX_BLEND_FACTOR_DST_ALPHA;
                        case AGFX_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return AGFX_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
                        default:                                    return factor;
                    }
                };

                info.colorAttachmentCount = colorCount;
                for (uint32_t i = 0; i < colorCount; ++i) {
                    info.colorFormats[i] = colorFormats[i];
                    info.blendEnable[i] = desc.blendEnable ? 1 : 0;
                    info.srcColorBlendFactor[i] = desc.srcBlend;
                    info.dstColorBlendFactor[i] = desc.dstBlend;
                    info.colorBlendOp[i] = desc.blendOp;
                    info.srcAlphaBlendFactor[i] = alphaEquivalent(desc.srcBlend);
                    info.dstAlphaBlendFactor[i] = alphaEquivalent(desc.dstBlend);
                    info.alphaBlendOp[i] = desc.blendOp;
                }
                assert(((desc.vertexShader && !desc.meshShader) || (!desc.vertexShader && desc.meshShader)) &&
                       "PipelineDesc: set exactly one of vertexShader (classic) or meshShader (mesh shading)");
                if (desc.meshShader) {
                    info.meshShader = *desc.meshShader;
                    info.meshGroupSizeX = desc.meshGroupSizeX;
                    info.meshGroupSizeY = desc.meshGroupSizeY;
                    info.meshGroupSizeZ = desc.meshGroupSizeZ;
                    if (desc.taskShader) {
                        info.taskShader = *desc.taskShader;
                        info.taskGroupSizeX = desc.taskGroupSizeX;
                        info.taskGroupSizeY = desc.taskGroupSizeY;
                        info.taskGroupSizeZ = desc.taskGroupSizeZ;
                    }
                } else {
                    info.vertexShader = *desc.vertexShader;
                }
                info.fragmentShader = *desc.fragmentShader;

                auto [inserted, _] = mPipelines.emplace(hash, device.CreateRenderPipeline(info));
                return inserted->second;
            }

        private:
            std::unordered_map<uint64_t, agfx::RenderPipeline> mPipelines;
        };
    }

    class Context;

    /// @brief One attachment for Context::SetRenderTargets: a texture plus which mip and array
    /// layer of it to render into.
    ///
    /// Implicitly constructible from a plain texture pointer, so the common whole-texture case
    /// stays `SetRenderTargets({&color}, &depth)`. Naming a mip is what makes rendering down a mip
    /// chain expressible; naming a layer is how you address one slice of a Texture2DArray or one
    /// face of a TextureCube (faces are ordered +X, -X, +Y, -Y, +Z, -Z). A 3D texture's depth
    /// slices are *not* layers and cannot be bound this way.
    struct RenderTargetBinding
    {
        detail::TextureBase* texture = nullptr;
        uint32_t mipLevel = 0;
        uint32_t arrayLayer = 0;

        RenderTargetBinding(detail::TextureBase* tex) : texture(tex) {}
        RenderTargetBinding(detail::TextureBase& tex, uint32_t mip, uint32_t layer = 0)
            : texture(&tex), mipLevel(mip), arrayLayer(layer)
        {
        }
    };

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
        void* windowHandle = nullptr;           // HWND on Windows, CAMetalLayer* on macOS. If null, the
                                                // Context is headless: no swap chain is created, and the
                                                // back buffer / resize / HDR entry points are unavailable.
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

            mHeadless = (info.windowHandle == nullptr);

            if (!mHeadless) {
                agfxSwapChainCreateInfo swapChainInfo{};
                swapChainInfo.queue = mQueue;
                swapChainInfo.imageCount = info.swapChainImageCount;
                swapChainInfo.width = info.width;
                swapChainInfo.height = info.height;
                swapChainInfo.isHDR = info.hdr ? 1 : 0;
                swapChainInfo.vsync = info.vsync ? 1 : 0;
                swapChainInfo.handle = info.windowHandle;
                mSwapChain = mDevice->CreateSwapChain(swapChainInfo);
            }

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

            if (!mHeadless) {
                mBackBufferTexture = mSwapChain.AcquireNextTexture();
                agfxCommandBufferTextureBarrier(cmd.Get(), mBackBufferTexture, AGFX_RESOURCE_STATE_PRESENT, AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 0);
            }

            return Frame(this);
        }

        void EndFrame()
        {
            assert(mFrameActive && "Context::EndFrame() called without an active frame");

            mActiveRenderPass.reset();
            mBackBufferRenderTarget.reset();

            agfx::CommandBuffer& cmd = mCommandBuffers[mFrameSlot];
            if (!mHeadless) {
                agfxCommandBufferTextureBarrier(cmd.Get(), mBackBufferTexture, AGFX_RESOURCE_STATE_RENDER_TARGET, AGFX_RESOURCE_STATE_PRESENT, 0, 0, 0);
            }
            cmd.End();

            mQueue.Submit(cmd);
            if (!mHeadless) {
                mSwapChain.Present();
            }

            ++mFenceValue;
            mQueue.Signal(mFrameFence, mFenceValue);
            mSlotFenceValues[mFrameSlot] = mFenceValue;

            ++mFrameIndex;
            mFrameActive = false;
        }

        // --- Resize / HDR toggle ---

        void Resize(uint32_t width, uint32_t height)
        {
            assert(!mHeadless && "Context::Resize() requires a swap chain (non-headless Context)");
            DrainGPU();
            mSwapChain.Resize(width, height);
            mWidth = width;
            mHeight = height;
        }

        void SetHDR(bool enabled, bool vsync)
        {
            assert(!mHeadless && "Context::SetHDR() requires a swap chain (non-headless Context)");
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

        /// @brief Opens a render pass over the given color attachments and optional depth attachment.
        ///
        /// Each binding may name a mip level and array layer (see RenderTargetBinding); the pass
        /// dimensions are taken from the bound subresource, not the base mip, so rendering into mip
        /// N gets an N-sized pass rather than a full-size one clipped down to it. Every attachment
        /// is transitioned for you -- only the bound subresource, so the other mips/layers keep
        /// whatever state they were in.
        void SetRenderTargets(std::initializer_list<RenderTargetBinding> colorTargets, RenderTargetBinding depthTarget = nullptr,
                               agfxLoadOp loadOp = AGFX_LOAD_OPERATION_CLEAR, const float* clearColor = kDefaultClearColor,
                               agfxLoadOp depthLoadOp = AGFX_LOAD_OPERATION_CLEAR, float clearDepth = 1.0f)
        {
            assert(mFrameActive);
            mActiveRenderPass.reset();

            agfxRenderPassCreateInfo info{};
            uint32_t idx = 0;
            uint32_t w = 0, h = 0;
            for (const RenderTargetBinding& binding : colorTargets) {
                detail::TextureBase* target = binding.texture;
                TransitionSubresourceForAttachment(*target, AGFX_RESOURCE_STATE_RENDER_TARGET, binding);
                agfxRenderPassAttachment& att = info.colorAttachments[idx];
                att.renderTarget = target->RTV(binding.mipLevel, binding.arrayLayer);
                att.loadOp = loadOp;
                att.storeOp = AGFX_STORE_OPERATION_STORE;
                if (clearColor) {
                    memcpy(att.clearColor, clearColor, sizeof(float) * 4);
                }
                mBoundColorFormats[idx] = target->Format();
                w = MipExtent(target->Width(), binding.mipLevel);
                h = MipExtent(target->Height(), binding.mipLevel);
                ++idx;
            }
            info.colorAttachmentCount = idx;
            mBoundColorCount = idx;

            if (depthTarget.texture) {
                detail::TextureBase* target = depthTarget.texture;
                TransitionSubresourceForAttachment(*target, AGFX_RESOURCE_STATE_DEPTH_WRITE, depthTarget);
                info.hasDepthAttachment = 1;
                info.depthAttachment.renderTarget = target->RTV(depthTarget.mipLevel, depthTarget.arrayLayer);
                info.depthAttachment.loadOp = depthLoadOp;
                info.depthAttachment.storeOp = AGFX_STORE_OPERATION_STORE;
                info.depthAttachment.clearDepth = clearDepth;
                mBoundDepthFormat = target->Format();
                mHasDepth = true;
                w = MipExtent(target->Width(), depthTarget.mipLevel);
                h = MipExtent(target->Height(), depthTarget.mipLevel);
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
            assert(!mHeadless && "Context::SetBackBufferRenderTarget() requires a swap chain (non-headless Context)");
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

        /// @brief Sets the viewport (the transform applied to clip space) alone, leaving the scissor
        /// as-is. Use this with SetScissor when the two need to disagree -- squeezing the image into
        /// a sub-rect is a viewport job, cropping it is a scissor job, and they are not the same.
        void SetViewport(float x, float y, float w, float h)
        {
            assert(mActiveRenderPass);
            mActiveRenderPass->SetViewport(x, y, w, h);
        }

        /// @brief Sets the scissor rect (the crop applied to rasterized pixels) alone.
        void SetScissor(uint32_t x, uint32_t y, uint32_t w, uint32_t h)
        {
            assert(mActiveRenderPass);
            mActiveRenderPass->SetScissor(x, y, w, h);
        }

        /// @brief Sets viewport and scissor to the same rect -- the common case. When they need to
        /// differ, call SetViewport and SetScissor separately.
        void SetViewportScissor(float x, float y, float w, float h)
        {
            SetViewport(x, y, w, h);
            SetScissor(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(w), static_cast<uint32_t>(h));
        }

        void SetPipeline(const PipelineDesc& desc)
        {
            assert(mActiveRenderPass && "Context::SetPipeline() requires SetRenderTargets()/SetBackBufferRenderTarget() to be called first");
            agfx::RenderPipeline& pipeline = mPipelineCache.GetOrCreate(*mDevice, desc, mBoundColorFormats.data(), mBoundColorCount, mBoundDepthFormat, mHasDepth);
            mActiveRenderPass->SetPipeline(pipeline);
        }

        /// @brief Resolves a PipelineDesc to the cached agfxRenderPipeline it would produce against
        /// the given attachment formats, without needing an active render pass.
        ///
        /// SetPipeline is enough for ordinary drawing, but indirect replay needs the raw pipeline
        /// pointer for agfxIndirectBundleExecuteInfo::renderPipeline -- and needs it at *prepare*
        /// time, inside a compute pass, before any render pass has been begun. Resolving from the
        /// same cache and the same desc guarantees prepare and execute name the identical PSO.
        ///
        /// `colorTargets` and `depthTarget` must be the same attachments the subsequent
        /// SetRenderTargets call receives: the PSO is keyed on their formats, so passing different
        /// ones silently returns a different pipeline.
        agfx::RenderPipeline& GetOrCreatePipeline(const PipelineDesc& desc,
                                                  std::initializer_list<RenderTargetBinding> colorTargets,
                                                  RenderTargetBinding depthTarget = nullptr)
        {
            std::array<agfxTextureFormat, 8> formats{}; // Matches mBoundColorFormats.
            uint32_t count = 0;
            for (const RenderTargetBinding& binding : colorTargets) {
                formats[count++] = binding.texture->Format();
            }
            const bool hasDepth = depthTarget.texture != nullptr;
            return mPipelineCache.GetOrCreate(*mDevice, desc, formats.data(), count,
                                              hasDepth ? depthTarget.texture->Format()
                                                       : AGFX_TEXTURE_FORMAT_UNKNOWN,
                                              hasDepth);
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

        /// @brief Dispatches a mesh-shading pipeline (bound via SetPipeline with PipelineDesc::meshShader
        /// set). groupCount* are thread-group counts, not vertex/index counts -- same as raw agfxRenderPassDrawMesh.
        void DrawMesh(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1)
        {
            assert(mActiveRenderPass);
            mActiveRenderPass->DrawMesh(groupCountX, groupCountY, groupCountZ);
        }

        /// @brief Replays an AGFX_INDIRECT_BUNDLE_TYPE_DRAW/DRAW_INDEXED/DRAW_MESH bundle. For
        /// DISPATCH bundles, use the raw agfx::ComputePass::ExecuteIndirectBundle instead (via
        /// GetCurrentCommandBuffer().BeginComputePass()) -- ez has no compute-pass sugar, by design.
        void ExecuteIndirectBundle(IndirectBundle& bundle, const agfxIndirectBundleExecuteInfo& info)
        {
            assert(mActiveRenderPass);
            mActiveRenderPass->ExecuteIndirectBundle(bundle.Raw(), info);
        }

        // --- One-call resource creation (immediate, synchronous upload) ---

        /// @brief Creates a 2D texture, optionally with a mip chain and optionally seeding mip 0.
        /// mipLevels > 1 allocates the chain but only mip 0 is populated from pixels; fill the rest
        /// with UploadTexture, a copy, or a downsample pass.
        Texture2D CreateTexture2D(uint32_t width, uint32_t height, agfxTextureFormat format, agfxTextureUsage usage,
                                   const void* pixels = nullptr, uint32_t bytesPerRow = 0, uint32_t mipLevels = 1)
        {
            Texture2D texture = CreateTextureTyped<Texture2D>(AGFX_TEXTURE_TYPE_2D, width, height, 1, format, usage, mipLevels);

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

        /// @brief Creates an array of arrayLayers 2D textures. Seed layers with UploadTexture.
        Texture2DArray CreateTexture2DArray(uint32_t width, uint32_t height, uint32_t arrayLayers,
                                             agfxTextureFormat format, agfxTextureUsage usage, uint32_t mipLevels = 1)
        {
            Texture2DArray texture = CreateTextureTyped<Texture2DArray>(AGFX_TEXTURE_TYPE_2D_ARRAY, width, height,
                                                                        arrayLayers, format, usage, mipLevels);
            mDevice->MakeResourcesResident();
            return texture;
        }

        /// @brief Creates a volume texture. Seed slices with UploadTexture, addressing them through
        /// the region's z/depth -- a 3D texture has no array layers.
        Texture3D CreateTexture3D(uint32_t width, uint32_t height, uint32_t depth, agfxTextureFormat format,
                                   agfxTextureUsage usage, uint32_t mipLevels = 1)
        {
            Texture3D texture = CreateTextureTyped<Texture3D>(AGFX_TEXTURE_TYPE_3D, width, height, depth, format,
                                                              usage, mipLevels);
            mDevice->MakeResourcesResident();
            return texture;
        }

        /// @brief Creates a cube map of six size x size faces, ordered +X, -X, +Y, -Y, +Z, -Z.
        TextureCube CreateTextureCube(uint32_t size, agfxTextureFormat format, agfxTextureUsage usage,
                                       uint32_t mipLevels = 1)
        {
            TextureCube texture = CreateTextureTyped<TextureCube>(AGFX_TEXTURE_TYPE_CUBE, size, size, 6, format,
                                                                  usage, mipLevels);
            mDevice->MakeResourcesResident();
            return texture;
        }

        /// @brief Synchronously uploads CPU data into one (mip, layer) subresource, blocking until the
        /// copy completes. This is the general form of the pixels argument to CreateTexture2D: use it
        /// to seed mips below 0, array layers, and 3D slices (which are addressed by region.z/depth
        /// rather than by layer). bytesPerImage defaults to bytesPerRow * region.height.
        ///
        /// The destination is left in AGFX_RESOURCE_STATE_COMMON, matching CreateTexture2D -- see the
        /// note on detail::Uploader for why that is both correct and deliberate.
        void UploadTexture(detail::TextureBase& dst, const agfxTextureRegion& region, uint32_t mipLevel,
                           uint32_t layer, const void* data, uint32_t dataSize, uint32_t bytesPerRow,
                           uint32_t bytesPerImage = 0)
        {
            mUploader->UploadTexture(dst.Raw(), region, mipLevel, layer, data, dataSize, bytesPerRow,
                                     bytesPerImage ? bytesPerImage : bytesPerRow * region.height);
            mUploader->Flush();
            mDevice->MakeResourcesResident();
        }

        // --- Copies (recorded into the current frame, between BeginFrame/EndFrame) ---
        //
        // Each opens and closes its own scoped compute pass on this frame's command buffer. Callers
        // are responsible for the COPY_SOURCE/COPY_DEST transitions via TransitionTexture /
        // TransitionBuffer -- these do not transition on your behalf, matching the rest of ez's
        // best-effort, explicit barrier model.

        void CopyTextureToTexture(detail::TextureBase& src, detail::TextureBase& dst, const agfxTextureRegion& region,
                                  uint32_t mipLevel = 0, uint32_t layer = 0)
        {
            assert(mFrameActive);
            agfx::ComputePass pass = mCommandBuffers[mFrameSlot].BeginComputePass("ez copy texture to texture");
            pass.CopyTextureToTexture(src.Raw(), dst.Raw(), region, mipLevel, layer);
        }

        void CopyBufferToTexture(Buffer& src, detail::TextureBase& dst, const agfxTextureRegion& region,
                                 uint32_t mipLevel, uint32_t layer, uint32_t bytesPerRow, uint32_t bytesPerImage = 0)
        {
            assert(mFrameActive);
            agfx::ComputePass pass = mCommandBuffers[mFrameSlot].BeginComputePass("ez copy buffer to texture");
            pass.CopyBufferToTexture(src.Raw(), dst.Raw(), region, mipLevel, layer, bytesPerRow,
                                     bytesPerImage ? bytesPerImage : bytesPerRow * region.height);
        }

        void CopyTextureToBuffer(detail::TextureBase& src, Buffer& dst, uint64_t bufferOffset,
                                 const agfxTextureRegion& region, uint32_t mipLevel, uint32_t layer,
                                 uint32_t bytesPerRow, uint32_t bytesPerImage = 0)
        {
            assert(mFrameActive);
            agfx::ComputePass pass = mCommandBuffers[mFrameSlot].BeginComputePass("ez copy texture to buffer");
            pass.CopyTextureToBuffer(src.Raw(), dst.Raw(), bufferOffset, region, mipLevel, layer, bytesPerRow,
                                     bytesPerImage ? bytesPerImage : bytesPerRow * region.height);
        }

        Buffer CreateVertexBuffer(const void* data, uint64_t size, uint64_t stride)
        {
            return CreateInitializedBuffer(data, size, stride, AGFX_BUFFER_USAGE_SHADER_READ);
        }

        /// @brief Creates an index buffer. stride selects the index width: 4 for u32 (the default),
        /// 2 for u16. The backends derive the index type from the buffer's stride, so this must match
        /// the data actually uploaded -- a u16 buffer left at stride 4 is read as u32 and draws garbage.
        Buffer CreateIndexBuffer(const void* data, uint64_t size, uint64_t stride = 4)
        {
            assert((stride == 2 || stride == 4) && "CreateIndexBuffer: index stride must be 2 (u16) or 4 (u32)");
            return CreateInitializedBuffer(data, size, stride, AGFX_BUFFER_USAGE_INDEX);
        }

        Buffer CreateStructuredBuffer(const void* data, uint64_t size, uint64_t stride, bool shaderWritable = false)
        {
            agfxBufferUsage usage = static_cast<agfxBufferUsage>(AGFX_BUFFER_USAGE_SHADER_READ | (shaderWritable ? AGFX_BUFFER_USAGE_SHADER_WRITE : 0));
            return CreateInitializedBuffer(data, size, stride, usage);
        }

        /// @brief Creates an agfxIndirectBundle sized for maxCommandCount commands and maxCountCount
        /// independent count slots (see agfx.h's notes on multiple count slots per bundle).
        IndirectBundle CreateIndirectBundle(agfxIndirectBundleType type, uint32_t maxCommandCount, uint32_t maxCountCount = 1)
        {
            agfxIndirectBundleCreateInfo info{};
            info.type = type;
            info.maxCommandCount = maxCommandCount;
            info.maxCountCount = maxCountCount;
            return IndirectBundle(mDevice->CreateIndirectBundle(info));
        }

        // --- Ray tracing (acceleration structures) ---

        /// @brief True if this device can build/trace acceleration structures. Guard all RT work with it.
        bool SupportsRayTracing() const { return mDevice->GetInfo().supportsRayTracing != 0; }

        /// @brief Creates and synchronously builds a bottom-level acceleration structure from a triangle
        /// mesh in the given ez::Buffers. vertexOffset/indexOffset are element indices into their buffers;
        /// the first vertex attribute must be float3 position. Returns an invalid AS on non-RT devices.
        AccelerationStructure CreateBottomLevel(Buffer& vertexBuffer, uint32_t vertexCount, uint32_t vertexOffset,
                                                Buffer& indexBuffer, uint32_t indexCount, uint32_t indexOffset,
                                                bool opaque = true, const char* name = "ez BLAS")
        {
            if (!SupportsRayTracing()) {
                return AccelerationStructure();
            }

            agfxAccelerationStructureGeometry geometry{};
            geometry.type = AGFX_ACCELERATION_STRUCTURE_GEOMETRY_TYPE_TRIANGLES;
            geometry.opaque = opaque;
            geometry.triangles.vertexBuffer = vertexBuffer.Raw();
            geometry.triangles.vertexOffset = static_cast<uint32_t>(vertexOffset * vertexBuffer.Stride());
            geometry.triangles.vertexCount = vertexCount;
            geometry.triangles.indexBuffer = indexBuffer.Raw();
            geometry.triangles.indexCount = indexCount;
            geometry.triangles.indexOffset = static_cast<uint32_t>(indexOffset * indexBuffer.Stride());

            agfxAccelerationStructureCreateInfo info{};
            info.type = AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            info.bottomLevel.geometries = &geometry;
            info.bottomLevel.geometryCount = 1;
            info.name = name;

            agfx::AccelerationStructure blas = mDevice->CreateAccelerationStructure(info);
            BuildOne(blas);
            return AccelerationStructure(std::move(blas));
        }

        /// @brief Creates and synchronously builds a top-level acceleration structure over the given
        /// instances. Each instance's transform is a row-major 3x4 matrix. The referenced BLAS must have
        /// been built (e.g. via CreateBottomLevel) before this call. Returns invalid on non-RT devices.
        AccelerationStructure CreateTopLevel(const agfxAccelerationStructureInstance* instances, uint32_t instanceCount,
                                             const char* name = "ez TLAS")
        {
            if (!SupportsRayTracing()) {
                return AccelerationStructure();
            }

            agfxAccelerationStructureCreateInfo info{};
            info.type = AGFX_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
            info.topLevel.maxInstanceCount = instanceCount;
            info.name = name;

            agfx::AccelerationStructure tlas = mDevice->CreateAccelerationStructure(info);
            tlas.AddInstances(instances, instanceCount);
            BuildOne(tlas);
            return AccelerationStructure(std::move(tlas));
        }

        // --- Per-frame dynamic constants ---

        agfx::BufferView& AllocateConstants(const void* data, size_t size) { return mDynamicConstants.Allocate(data, size); }

        template<typename T>
        agfx::BufferView& AllocateConstants(const T& data)
        {
            return AllocateConstants(&data, sizeof(T));
        }

        // --- Explicit barrier escape hatch ---

        /// @brief Transitions a texture, or one (mip, layer) subresource of it, into newState.
        ///
        /// Defaults to the whole resource, which is a single barrier. Passing an explicit mip and/or
        /// layer is what makes mip-chain generation expressible: mip N can sit in RENDER_TARGET or
        /// UNORDERED_ACCESS while mip N-1 is read as a shader resource. Transitioning the whole
        /// resource after that is still correct -- the subresources are re-grouped by their current
        /// state and one barrier is emitted per distinct old state, since a single barrier can only
        /// name one.
        void TransitionTexture(detail::TextureBase& tex, agfxResourceState newState,
                               uint32_t mipLevel = kAllMips, uint32_t arrayLayer = kAllLayers)
        {
            ResourceStateTracker& tracker = tex.Tracker();
            agfx::CommandBuffer& cmd = mCommandBuffers[mFrameSlot];

            if (mipLevel != kAllMips || arrayLayer != kAllLayers) {
                TransitionSubresources(cmd, tex, newState, mipLevel, arrayLayer);
                return;
            }

            if (tracker.Uniform()) {
                if (tracker.Current() == newState) {
                    return;
                }
                cmd.TextureBarrier(tex.Raw(), tracker.Current(), newState, AGFX_SUBRESOURCE_ALL_MIPS,
                                   AGFX_SUBRESOURCE_ALL_LAYERS, true);
                tracker.Set(newState);
                return;
            }

            TransitionSubresources(cmd, tex, newState, kAllMips, kAllLayers);
            tracker.Set(newState);
        }

        void TransitionBuffer(Buffer& buf, agfxResourceState newState)
        {
            if (buf.State() == newState) {
                return;
            }
            mCommandBuffers[mFrameSlot].BufferBarrier(buf.Raw(), buf.State(), newState, true);
            buf.SetState(newState);
        }

        /// @brief Transitions both the commands and count buffers of an indirect bundle together --
        /// they always move in lockstep (UAV while being culled/prepared, INDIRECT_ARGUMENT while
        /// being replayed). Uses the raw C API directly since the bundle owns these buffers itself
        /// (see agfx::IndirectBundle::CommandsBuffer/CountBuffer).
        void TransitionIndirectBundle(IndirectBundle& bundle, agfxResourceState newState)
        {
            if (bundle.State() == newState) {
                return;
            }
            agfxCommandBuffer* cmd = mCommandBuffers[mFrameSlot].Get();
            agfxCommandBufferBufferBarrier(cmd, bundle.Raw().CommandsBuffer(), bundle.State(), newState, true);
            agfxCommandBufferBufferBarrier(cmd, bundle.Raw().CountBuffer(), bundle.State(), newState, true);
            bundle.SetState(newState);
        }

        // --- Raw access for advanced/mixed use ---

        agfx::Device& GetDevice() { return *mDevice; }
        agfx::CommandQueue& GetGraphicsQueue() { return mQueue; }
        agfx::CommandBuffer& GetCurrentCommandBuffer() { return mCommandBuffers[mFrameSlot]; }
        agfxTextureFormat GetSwapChainFormat() const
        {
            assert(!mHeadless && "Context::GetSwapChainFormat() requires a swap chain (non-headless Context)");
            return mSwapChain.GetFormat();
        }

        /// @brief True when the Context was created without a windowHandle: no swap chain exists, and
        /// the back buffer / resize / HDR entry points must not be called.
        bool IsHeadless() const { return mHeadless; }

        // Block until all GPU work submitted so far has completed. Callers must drain before
        // destroying/replacing any resource that an in-flight command buffer may still reference
        // (e.g. before tearing down a renderer built on top of this Context).
        void DrainGPU()
        {
            ++mFenceValue;
            mQueue.Signal(mFrameFence, mFenceValue);
            mFrameFence.Wait(mFenceValue, UINT64_MAX);
        }

    private:
        /// @brief A mip level's extent, floored at 1 the way every backend sizes its mip chain.
        static uint32_t MipExtent(uint32_t base, uint32_t mipLevel)
        {
            const uint32_t extent = base >> mipLevel;
            return extent ? extent : 1u;
        }

        /// @brief Transitions exactly what SetRenderTargets is about to bind. A whole-texture
        /// binding (the default mip 0 / layer 0 on a single-mip, single-layer texture) transitions
        /// as one barrier; anything naming a mip or layer on a larger texture transitions only that
        /// subresource, so binding one face or one mip does not disturb the others.
        void TransitionSubresourceForAttachment(detail::TextureBase& tex, agfxResourceState state,
                                                const RenderTargetBinding& binding)
        {
            const bool wholeTexture = binding.mipLevel == 0 && binding.arrayLayer == 0 &&
                                      tex.Tracker().MipLevels() == 1 && tex.Tracker().ArrayLayers() == 1;
            if (wholeTexture) {
                TransitionTexture(tex, state);
            } else {
                TransitionTexture(tex, state, binding.mipLevel, binding.arrayLayer);
            }
        }

        /// @brief Emits the barriers for TransitionTexture's subresource paths and updates the tracker.
        /// Walks the addressed (mip, layer) subresources, skipping any already in newState, and emits
        /// one barrier per subresource -- each may be coming from a different old state, and a barrier
        /// names exactly one.
        void TransitionSubresources(agfx::CommandBuffer& cmd, detail::TextureBase& tex,
                                    agfxResourceState newState, uint32_t mipLevel, uint32_t arrayLayer)
        {
            ResourceStateTracker& tracker = tex.Tracker();
            const uint32_t mipBegin = (mipLevel == kAllMips) ? 0u : mipLevel;
            const uint32_t mipEnd = (mipLevel == kAllMips) ? tracker.MipLevels() : mipLevel + 1u;
            const uint32_t layerBegin = (arrayLayer == kAllLayers) ? 0u : arrayLayer;
            const uint32_t layerEnd = (arrayLayer == kAllLayers) ? tracker.ArrayLayers() : arrayLayer + 1u;

            for (uint32_t mip = mipBegin; mip < mipEnd; ++mip) {
                for (uint32_t layer = layerBegin; layer < layerEnd; ++layer) {
                    const agfxResourceState oldState = tracker.At(mip, layer);
                    if (oldState == newState) {
                        continue;
                    }
                    cmd.TextureBarrier(tex.Raw(), oldState, newState, mip, layer, true);
                    tracker.SetAt(mip, layer, newState);
                }
            }
        }

        /// @brief Fills in an agfxTextureCreateInfo and wraps the result in the matching ez type.
        /// SAMPLED is force-ORed into every texture's usage, as CreateTexture2D has always done.
        template<typename T>
        T CreateTextureTyped(agfxTextureType type, uint32_t width, uint32_t height, uint32_t depthOrArrayLayers,
                             agfxTextureFormat format, agfxTextureUsage usage, uint32_t mipLevels)
        {
            agfxTextureCreateInfo info{};
            info.type = type;
            info.format = format;
            info.usage = static_cast<agfxTextureUsage>(usage | AGFX_TEXTURE_USAGE_SAMPLED);
            info.width = width;
            info.height = height;
            info.depthOrArrayLayers = depthOrArrayLayers;
            info.mipLevels = mipLevels ? mipLevels : 1;
            return T(mDevice->CreateTexture(info), info);
        }

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

        /// @brief Synchronously builds one acceleration structure on the graphics queue: allocates a
        /// scratch buffer, makes everything resident, records the build with an AS->AS barrier (so the
        /// result is safe to consume), submits, and blocks until it completes.
        void BuildOne(agfx::AccelerationStructure& as)
        {
            agfxAccelerationStructureSizes sizes = as.GetSizes();

            agfxBufferCreateInfo scratchInfo{};
            scratchInfo.size = sizes.scratchBufferSize;
            scratchInfo.stride = 4;
            scratchInfo.usage = AGFX_BUFFER_USAGE_SHADER_WRITE;
            scratchInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
            agfx::Buffer scratch = mDevice->CreateBuffer(scratchInfo);

            // The AS storage, its instance buffer, and the scratch buffer were all added to the
            // residency set after any earlier commit -- re-commit before the GPU build references them.
            mDevice->MakeResourcesResident();

            agfx::CommandBuffer cmd = mDevice->CreateCommandBuffer(mQueue);
            cmd.Reset();
            cmd.Begin();
            {
                agfx::ComputePass pass = cmd.BeginComputePass("ez AS build");
                pass.BuildAccelerationStructure(as, scratch, 0);
            }
            // AS->AS barrier: producer/consumer both MTLStageAccelerationStructure, so consumers of this
            // structure (further builds, or the trace) see a completed build. A COMMON source is dropped.
            cmd.AccelerationStructureBarrier(as, AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                                             AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, true);
            cmd.End();

            mQueue.Submit(cmd);
            ++mFenceValue;
            mQueue.Signal(mFrameFence, mFenceValue);
            mFrameFence.Wait(mFenceValue, UINT64_MAX);
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

        agfx::Device mOwnedDevice;
        agfx::Device* mDevice = nullptr;
        bool mOwnsDevice = true;

        agfx::CommandQueue mQueue;
        agfx::SwapChain mSwapChain;
        bool mHeadless = false;
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
