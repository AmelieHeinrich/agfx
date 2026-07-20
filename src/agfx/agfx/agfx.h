/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 00:11:12
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <stdint.h>

/// @brief Allocation function signature for agfxDeviceCreate.
/// @param size The number of bytes to allocate.
typedef void* (*agfxAllocate)(uint64_t size);

/// @brief Free function signature for agfxDeviceCreate.
/// @param ptr The pointer to free, or nullptr.
typedef void (*agfxFree)(void* ptr);

/// @brief A boolean type for agfx. Use 0 for false, non-zero for true.
typedef int8_t agfxBool;

/// @brief The main agfx device object. This is the entry point for all agfx operations.
typedef struct agfxDevice agfxDevice;

/// @brief A texture object. Represents a GPU texture resource.
typedef struct agfxTexture agfxTexture;

/// @brief A fence object. Used for GPU-CPU synchronization.
typedef struct agfxFence agfxFence;

/// @brief A query pool object. Used for GPU timestamp queries.
typedef struct agfxQueryPool agfxQueryPool;

/// @brief A command queue object. Used to submit command buffers to the GPU.
typedef struct agfxCommandQueue agfxCommandQueue;

/// @brief A command buffer object. Used to record GPU commands.
typedef struct agfxCommandBuffer agfxCommandBuffer;

/// @brief A pass object to encode compute, copy, or raytracing commands. Created from a command buffer.
typedef struct agfxComputePass agfxComputePass;

/// @brief A buffer object. Represents a GPU buffer resource.
typedef struct agfxBuffer agfxBuffer;

/// @brief A texture view object. Represents a view into a texture resource, used for sampling or writing.
typedef struct agfxTextureView agfxTextureView;

/// @brief A sampler object. Represents a sampler state for sampling textures.
typedef struct agfxSampler agfxSampler;

/// @brief A buffer view object. Represents a view into a buffer resource, used for reading or writing.
typedef struct agfxBufferView agfxBufferView;

/// @brief A render target object. Represents a render target for rendering operations, on color or depth targets.
typedef struct agfxRenderTarget agfxRenderTarget;

/// @brief A render pass object. Represents a render pass for rendering operations, created from a command buffer.
typedef struct agfxRenderPass agfxRenderPass;

/// @brief A swap chain object. Represents a swap chain for presenting rendered images to the screen.
typedef struct agfxSwapChain agfxSwapChain;

/// @brief A shader module object. Represents a compiled shader module for use in a pipeline.
typedef struct agfxShaderModule agfxShaderModule;

/// @brief A render pipeline object. Represents a graphics pipeline for rendering operations.
typedef struct agfxRenderPipeline agfxRenderPipeline;

/// @brief A compute pipeline object. Represents a compute pipeline for compute operations.
typedef struct agfxComputePipeline agfxComputePipeline;

/// @brief A structure containing information for creating an agfxDevice.
typedef struct agfxDeviceCreateInfo {
    /// @brief The allocation function to use for device memory allocations.
    agfxAllocate allocate;
    /// @brief The free function to use for device memory deallocations.
    agfxFree free;
    /// @brief The allocation function to use for temporary allocations, such as for encoders.
    agfxAllocate tempAllocate;
    /// @brief The free function to use for temporary deallocations, such as for encoders.
    agfxFree tempFree;
    /// @brief Whether to enable validation layers for debugging. Set to 1 to enable, 0 to disable.
    agfxBool enableValidation;
} agfxDeviceCreateInfo;

/// @brief Describes the capabilities of an agfxDevice, retrieved via agfxDeviceGetInfo.
typedef struct agfxDeviceInfo {
    /// @brief The name of the device, e.g., "NVIDIA GeForce RTX 4090".
    char name[256];

    /// @brief Whether the device supports ray tracing. Set to 1 if supported, 0 otherwise.
    agfxBool supportsRayTracing;

    /// @brief Whether the device supports mesh shaders. Set to 1 if supported, 0 otherwise.
    agfxBool supportsMeshShaders;

    /// @brief Whether the device supports multi-draw indirect. Set to 1 if supported, 0 otherwise.
    agfxBool supportsMultiDrawIndirect;
} agfxDeviceInfo;

/// @brief Creates a new agfxDevice with the specified creation info.
/// @param createInfo A pointer to an agfxDeviceCreateInfo structure containing the creation parameters
/// @return A pointer to the newly created agfxDevice, or nullptr on failure.
agfxDevice* agfxDeviceCreate(const agfxDeviceCreateInfo* createInfo);

/// @brief Destroys the specified agfxDevice and releases all associated resources.
/// @param device A pointer to the agfxDevice to destroy.
void agfxDeviceDestroy(agfxDevice* device);

/// @brief Retrieves information about the specified agfxDevice.
/// @param device A pointer to the agfxDevice to query.
/// @param info A pointer to an agfxDeviceInfo structure that will be filled with the device information.
void agfxDeviceGetInfo(agfxDevice* device, agfxDeviceInfo* info);

/// @brief Makes all resources associated with the specified agfxDevice resident in GPU memory.
/// @param device A pointer to the agfxDevice whose resources should be made resident.
/// @note Equivalent of MTLResidencySet.commit. Call to make resources GPU resident
void agfxDeviceMakeResourcesResident(agfxDevice* device);

/// @brief Creates a new agfxFence for GPU-CPU synchronization.
/// @param device A pointer to the agfxDevice to create the fence on.
/// @return A pointer to the newly created agfxFence, or nullptr on failure.
agfxFence* agfxFenceCreate(agfxDevice* device);

/// @brief Destroys the specified agfxFence and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the fence.
/// @param fence A pointer to the agfxFence to destroy.
void agfxFenceDestroy(agfxDevice* device, agfxFence* fence);

/// @brief Waits for the specified agfxFence to reach the given value, with an optional timeout.
/// @param fence A pointer to the agfxFence to wait on.
/// @param value The fence value to wait for.
/// @param timeout The maximum time to wait in miliseconds. Use UINT64_MAX for an infinite wait.
/// @note This is a *CPU-GPU* wait. Full blocking.
void agfxFenceWait(agfxFence* fence, uint64_t value, uint64_t timeout);

/// @brief Signals the specified agfxFence with the given value from the GPU.
/// @param fence A pointer to the agfxFence to signal.
/// @param value The fence value to signal.
void agfxFenceSignal(agfxFence* fence, uint64_t value);

/// @brief Retrieves the last completed value of the specified agfxFence.
/// @param fence A pointer to the agfxFence to query.
/// @return The last completed fence value.
uint64_t agfxFenceGetCompletedValue(agfxFence* fence);

/// @brief Creates a new agfxQueryPool for GPU timestamp queries.
typedef struct agfxQueryPoolCreateInfo {
    /// @brief The number of queries in the pool.
    uint32_t count;
} agfxQueryPoolCreateInfo;

/// @brief Creates a new agfxQueryPool for GPU timestamp queries.
/// @param device A pointer to the agfxDevice to create the query pool on.
/// @param queue A pointer to the agfxCommandQueue that will be used with the query pool.
/// @param createInfo A pointer to an agfxQueryPoolCreateInfo structure containing the creation parameters.
/// @return A pointer to the newly created agfxQueryPool, or nullptr on failure.
agfxQueryPool* agfxQueryPoolCreate(agfxDevice* device, agfxCommandQueue* queue, const agfxQueryPoolCreateInfo* createInfo);

/// @brief Destroys the specified agfxQueryPool and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the query pool.
/// @param pool A pointer to the agfxQueryPool to destroy.
void agfxQueryPoolDestroy(agfxDevice* device, agfxQueryPool* pool);

/// @brief Writes a timestamp to the specified agfxQueryPool at the given index from the command buffer.
/// @param commandBuffer A pointer to the agfxCommandBuffer to record the timestamp command in.
/// @param pool A pointer to the agfxQueryPool to write the timestamp to.
/// @param index The index in the query pool to write the timestamp to.
void agfxCommandBufferWriteTimestamp(agfxCommandBuffer* commandBuffer, agfxQueryPool* pool, uint32_t index);

/// @brief Resolves the timestamps in the specified agfxQueryPool to a buffer for readback.
/// @param commandBuffer A pointer to the agfxCommandBuffer to record the resolve command in.
/// @param pool A pointer to the agfxQueryPool to resolve.
/// @param firstIndex The first index in the query pool to resolve.
/// @param count The number of queries to resolve.
void agfxCommandBufferResolveQueryPool(agfxCommandBuffer* commandBuffer, agfxQueryPool* pool, uint32_t firstIndex, uint32_t count);

/// @brief Reads back the resolved timestamps from the specified agfxQueryPool into CPU memory.
/// @param device A pointer to the agfxDevice that owns the query pool.
/// @param pool A pointer to the agfxQueryPool to read back from.
/// @param firstIndex The first index in the query pool to read back.
/// @param count The number of queries to read back.
/// @param outTimestampsNanoseconds A pointer to an array of uint64_t to receive the timestamps in nanoseconds. The array must have at least 'count' elements.
/// @note You are responsible to making sure the query pool has been resolved and the GPU has been drained of work before doing any readback. 
void agfxQueryPoolReadback(agfxDevice* device, agfxQueryPool* pool, uint32_t firstIndex, uint32_t count, uint64_t* outTimestampsNanoseconds);

/// @brief A type of command queue. Used to specify the type of command queue to create.
typedef enum agfxCommandQueueType {
    /// @brief A graphics command queue. Used for rendering operations.
    AGFX_COMMAND_QUEUE_TYPE_GRAPHICS,
    /// @brief A compute command queue. Used for compute operations. Cannot create render passes on this queue.
    AGFX_COMMAND_QUEUE_TYPE_COMPUTE,
    /// @brief A transfer command queue. Used for copy operations. Cannot create render passes on this queue or perform compute operations.
    AGFX_COMMAND_QUEUE_TYPE_TRANSFER,
} agfxCommandQueueType;

/// @brief A structure containing information for creating an agfxCommandQueue.
typedef struct agfxCommandQueueCreateInfo {
    /// @brief The type of command queue to create. Must be one of the values in agfxCommandQueueType.
    agfxCommandQueueType type;
} agfxCommandQueueCreateInfo;

/// @brief Creates a new agfxCommandQueue with the specified creation info.
/// @param device A pointer to the agfxDevice to create the queue on.
/// @param createInfo A pointer to an agfxCommandQueueCreateInfo structure containing the creation parameters.
/// @return A pointer to the newly created agfxCommandQueue, or nullptr on failure.
agfxCommandQueue* agfxCommandQueueCreate(agfxDevice* device, const agfxCommandQueueCreateInfo* createInfo);

/// @brief Destroys the specified agfxCommandQueue and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the queue.
/// @param queue A pointer to the agfxCommandQueue to destroy.
void agfxCommandQueueDestroy(agfxDevice* device, agfxCommandQueue* queue);

/// @brief Enqueues a GPU-side signal of the specified agfxFence to the given value once the queue reaches this point.
/// @param queue A pointer to the agfxCommandQueue to signal from.
/// @param fence A pointer to the agfxFence to signal.
/// @param value The fence value to signal.
void agfxCommandQueueSignal(agfxCommandQueue* queue, agfxFence* fence, uint64_t value);

/// @brief Enqueues a GPU-side wait on the specified agfxFence until it reaches the given value before the queue continues.
/// @param queue A pointer to the agfxCommandQueue to wait on.
/// @param fence A pointer to the agfxFence to wait for.
/// @param value The fence value to wait for.
void agfxCommandQueueWait(agfxCommandQueue* queue, agfxFence* fence, uint64_t value);

/// @brief Submits the specified command buffers to the queue for execution on the GPU.
/// @param queue A pointer to the agfxCommandQueue to submit to.
/// @param commandBuffers A pointer to an array of agfxCommandBuffer pointers to submit, in order.
/// @param commandBufferCount The number of command buffers in the array.
/// @note Each command buffer must have been ended with agfxCommandBufferEnd before being submitted.
void agfxCommandQueueSubmit(agfxCommandQueue* queue, agfxCommandBuffer** commandBuffers, uint32_t commandBufferCount);

// Command buffer
/// @brief Passed as the mip/layer of a barrier to target all mip levels of a texture instead of a single one.
#define AGFX_SUBRESOURCE_ALL_MIPS -1
/// @brief Passed as the mip/layer of a barrier to target all array layers of a texture instead of a single one.
#define AGFX_SUBRESOURCE_ALL_LAYERS -1

/// @brief The state a texture or buffer resource is in, used to describe barrier transitions.
typedef enum agfxResourceState {
    /// @brief The default/initial state a resource is created in.
    AGFX_RESOURCE_STATE_COMMON,
    /// @brief Readable as a vertex buffer or constant/uniform buffer.
    AGFX_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
    /// @brief Readable as an index buffer.
    AGFX_RESOURCE_STATE_INDEX_BUFFER,
    /// @brief Writable as a color render target attachment.
    AGFX_RESOURCE_STATE_RENDER_TARGET,
    /// @brief Readable/writable as an unordered-access-view (UAV/storage) resource from a shader.
    AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
    /// @brief Writable as a depth-stencil attachment.
    AGFX_RESOURCE_STATE_DEPTH_WRITE,
    /// @brief Readable as a depth-stencil attachment without writing.
    AGFX_RESOURCE_STATE_DEPTH_READ,
    /// @brief Readable as a shader resource from non-pixel stages (vertex, compute).
    AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    /// @brief Readable as a shader resource from the pixel/fragment stage.
    AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    /// @brief Readable as indirect draw/dispatch arguments.
    AGFX_RESOURCE_STATE_INDIRECT_ARGUMENT,
    /// @brief Writable as the destination of a copy operation.
    AGFX_RESOURCE_STATE_COPY_DEST,
    /// @brief Readable as the source of a copy operation.
    AGFX_RESOURCE_STATE_COPY_SOURCE,
    /// @brief Usable as a raytracing acceleration structure.
    AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
    /// @brief Readable from any shader stage and any read-only usage; a broad "any read" state.
    AGFX_RESOURCE_STATE_GENERIC_READ,
    /// @brief Readable as a shader resource from any shader stage (vertex, pixel, and compute).
    AGFX_RESOURCE_STATE_ALL_SHADER_RESOURCE,
    /// @brief The state a swap chain texture must be in before agfxSwapChainPresent is called.
    AGFX_RESOURCE_STATE_PRESENT
} agfxResourceState;

/// @brief Creates a new agfxCommandBuffer for recording GPU commands on the specified queue.
/// @param device A pointer to the agfxDevice to create the command buffer on.
/// @param queue A pointer to the agfxCommandQueue this command buffer will be submitted to.
/// @return A pointer to the newly created agfxCommandBuffer, or nullptr on failure.
agfxCommandBuffer* agfxCommandBufferCreate(agfxDevice* device, agfxCommandQueue* queue);

/// @brief Destroys the specified agfxCommandBuffer and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the command buffer.
/// @param commandBuffer A pointer to the agfxCommandBuffer to destroy.
void agfxCommandBufferDestroy(agfxDevice* device, agfxCommandBuffer* commandBuffer);

/// @brief Resets the specified agfxCommandBuffer, discarding any previously recorded commands, and begins recording again.
/// @param commandBuffer A pointer to the agfxCommandBuffer to reset.
void agfxCommandBufferReset(agfxCommandBuffer* commandBuffer);

/// @brief Begins recording of the specified agfxCommandBuffer.
/// @param commandBuffer A pointer to the agfxCommandBuffer to begin recording on.
/// @note Must be called (or agfxCommandBufferReset) before recording any commands and before agfxCommandBufferEnd.
void agfxCommandBufferBegin(agfxCommandBuffer* commandBuffer);

/// @brief Ends recording of the specified agfxCommandBuffer, making it ready for submission.
/// @param commandBuffer A pointer to the agfxCommandBuffer to end recording on.
/// @note Must be called before the command buffer is passed to agfxCommandQueueSubmit.
void agfxCommandBufferEnd(agfxCommandBuffer* commandBuffer);

/// @brief Records a barrier transitioning the given texture subresource(s) from one resource state to another.
/// @param commandBuffer A pointer to the agfxCommandBuffer to record the barrier in.
/// @param texture A pointer to the agfxTexture to transition.
/// @param oldState The resource state the texture (subresource) is currently in.
/// @param newState The resource state to transition the texture (subresource) to.
/// @param mip The mip level to transition, or AGFX_SUBRESOURCE_ALL_MIPS for all mips.
/// @param layer The array layer to transition, or AGFX_SUBRESOURCE_ALL_LAYERS for all layers.
/// @param agglomerate On the Metal backend, barriers have no per-resource hazard tracking: passing agfxBool(true)
///        merges this transition's producer/consumer stages into a single pending barrier that is flushed
///        automatically at the start of the next compute or render pass; passing agfxBool(false) is a no-op
///        on Metal. The D3D12 backend ignores this flag and always emits the transition immediately regardless
///        of its value. Pass true for ordinary resource transitions so Metal actually tracks the hazard. Pass
///        false only for the AGFX_RESOURCE_STATE_PRESENT <-> AGFX_RESOURCE_STATE_RENDER_TARGET transitions
///        around swap chain acquire/present (see agfx_demo_main.cpp), since presentable drawables need no
///        explicit barrier on Metal but still need the D3D12 transition, which happens unconditionally.
void agfxCommandBufferTextureBarrier(agfxCommandBuffer* commandBuffer, agfxTexture* texture,  agfxResourceState oldState, agfxResourceState newState, uint32_t mip, uint32_t layer, agfxBool agglomerate);

/// @brief Records a barrier transitioning the given buffer from one resource state to another.
/// @param commandBuffer A pointer to the agfxCommandBuffer to record the barrier in.
/// @param buffer A pointer to the agfxBuffer to transition.
/// @param oldState The resource state the buffer is currently in.
/// @param newState The resource state to transition the buffer to.
/// @param agglomerate See the note on agfxCommandBufferTextureBarrier: on Metal, pass true so the barrier is
///        queued and flushed at the next pass boundary; D3D12 ignores this flag and always transitions immediately.
void agfxCommandBufferBufferBarrier(agfxCommandBuffer* commandBuffer, agfxBuffer* buffer, agfxResourceState oldState, agfxResourceState newState, agfxBool agglomerate);

// Texture
/// @brief The pixel format of a texture.
/// @note BCn formats require desktop/BC-texture-compression support (D3D12; Metal via supportsBCTextureCompression).
///       ASTC formats are supported by the Metal backend only on Apple-GPU-family devices, not on Intel Macs.
typedef enum agfxTextureFormat {
    /// @brief Unspecified format; used with agfxRenderTargetCreateInfo to inherit the wrapped texture's own format.
    AGFX_TEXTURE_FORMAT_UNKNOWN,

    /// @brief Single-channel 8-bit unsigned normalized format.
    AGFX_TEXTURE_FORMAT_R8_UNORM,
    /// @brief Two-channel 8-bit unsigned normalized format.
    AGFX_TEXTURE_FORMAT_RG8_UNORM,
    AGFX_TEXTURE_FORMAT_RGBA8_UNORM,
    AGFX_TEXTURE_FORMAT_BGRA8_UNORM,
    /// @brief 8-bit unsigned normalized RGBA with an sRGB gamma curve applied on read/write. Use for final color-managed render targets.
    AGFX_TEXTURE_FORMAT_RGBA8_UNORM_SRGB,
    /// @brief 8-bit unsigned normalized BGRA with an sRGB gamma curve applied on read/write. Use for final color-managed render targets.
    AGFX_TEXTURE_FORMAT_BGRA8_UNORM_SRGB,

    /// @brief Single-channel 16-bit unsigned normalized format.
    AGFX_TEXTURE_FORMAT_R16_UNORM,
    /// @brief Two-channel 16-bit unsigned normalized format.
    AGFX_TEXTURE_FORMAT_RG16_UNORM,
    /// @brief Four-channel 16-bit unsigned normalized format.
    AGFX_TEXTURE_FORMAT_RGBA16_UNORM,

    /// @brief Single-channel 16-bit floating point format.
    AGFX_TEXTURE_FORMAT_R16F,
    /// @brief Two-channel 16-bit floating point format.
    AGFX_TEXTURE_FORMAT_RG16F,
    AGFX_TEXTURE_FORMAT_RGBA16F,

    AGFX_TEXTURE_FORMAT_R32F,
    /// @brief Two-channel 32-bit floating point format.
    AGFX_TEXTURE_FORMAT_RG32F,
    AGFX_TEXTURE_FORMAT_RGBA32F,

    /// @brief 32-bit floating point depth format, for use as a depth attachment.
    AGFX_TEXTURE_FORMAT_DEPTH32F,

    /// @brief BC1 (DXT1) block-compressed RGB(A), 4 bits per texel.
    AGFX_TEXTURE_FORMAT_BC1_UNORM,
    /// @brief BC1 (DXT1) block-compressed RGB(A) with an sRGB gamma curve, 4 bits per texel.
    AGFX_TEXTURE_FORMAT_BC1_UNORM_SRGB,
    /// @brief BC3 (DXT5) block-compressed RGBA with a separate alpha block, 8 bits per texel.
    AGFX_TEXTURE_FORMAT_BC3_UNORM,
    /// @brief BC3 (DXT5) block-compressed RGBA with an sRGB gamma curve, 8 bits per texel.
    AGFX_TEXTURE_FORMAT_BC3_UNORM_SRGB,
    /// @brief BC4 block-compressed single-channel format, 4 bits per texel. Common for grayscale/mask maps.
    AGFX_TEXTURE_FORMAT_BC4_UNORM,
    /// @brief BC5 block-compressed two-channel format, 8 bits per texel. Common for tangent-space normal maps.
    AGFX_TEXTURE_FORMAT_BC5_UNORM,
    /// @brief BC6H block-compressed unsigned floating point HDR RGB format, 8 bits per texel.
    AGFX_TEXTURE_FORMAT_BC6H_UFLOAT,
    /// @brief BC7 block-compressed high-quality RGBA format, 8 bits per texel.
    AGFX_TEXTURE_FORMAT_BC7_UNORM,
    /// @brief BC7 block-compressed high-quality RGBA format with an sRGB gamma curve, 8 bits per texel.
    AGFX_TEXTURE_FORMAT_BC7_UNORM_SRGB,

    /// @brief ASTC block-compressed RGBA format with a 4x4 texel block size (highest quality/bitrate of the ASTC block sizes listed here).
    AGFX_TEXTURE_FORMAT_ASTC_4X4_UNORM,
    /// @brief ASTC block-compressed RGBA format with a 4x4 texel block size and an sRGB gamma curve.
    AGFX_TEXTURE_FORMAT_ASTC_4X4_UNORM_SRGB,
    /// @brief ASTC block-compressed RGBA format with an 8x8 texel block size (lower bitrate than 4x4).
    AGFX_TEXTURE_FORMAT_ASTC_8X8_UNORM,
    /// @brief ASTC block-compressed RGBA format with an 8x8 texel block size and an sRGB gamma curve.
    AGFX_TEXTURE_FORMAT_ASTC_8X8_UNORM_SRGB,
} agfxTextureFormat;

/// @brief The dimensionality of a texture.
typedef enum agfxTextureType {
    AGFX_TEXTURE_TYPE_1D,
    AGFX_TEXTURE_TYPE_2D,
    AGFX_TEXTURE_TYPE_2D_ARRAY,
    AGFX_TEXTURE_TYPE_3D,
    AGFX_TEXTURE_TYPE_CUBE,
} agfxTextureType;

/// @brief Bitflags describing how a texture will be used, combined with bitwise OR. Must match the operations performed on it.
typedef enum agfxTextureUsage {
    /// @brief The texture can be sampled/read as a shader resource (requires an agfxTextureView).
    AGFX_TEXTURE_USAGE_SAMPLED = 1 << 0,
    /// @brief The texture can be bound as a read/write unordered-access (storage) resource (requires a writeable agfxTextureView).
    AGFX_TEXTURE_USAGE_STORAGE = 1 << 1,
    /// @brief The texture can be used as a color render target (requires an agfxRenderTarget).
    AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT = 1 << 2,
    /// @brief The texture can be used as a depth-stencil render target (requires an agfxRenderTarget with isDepth set).
    AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT = 1 << 3,
} agfxTextureUsage;

/// @brief A rectangular (or box, for 3D textures) region of a texture, used for copies and uploads.
typedef struct agfxTextureRegion {
    /// @brief The region's origin X coordinate, in texels.
    uint32_t x;
    /// @brief The region's origin Y coordinate, in texels.
    uint32_t y;
    /// @brief The region's origin Z coordinate (3D textures only), in texels.
    uint32_t z;
    /// @brief The region's width, in texels.
    uint32_t width;
    /// @brief The region's height, in texels.
    uint32_t height;
    /// @brief The region's depth (3D textures only), in texels.
    uint32_t depth;
} agfxTextureRegion;

/// @brief A structure containing information for creating an agfxTexture.
typedef struct agfxTextureCreateInfo {
    /// @brief The texture's dimensionality.
    agfxTextureType type;
    /// @brief The texture's pixel format.
    agfxTextureFormat format;
    /// @brief Bitflags (agfxTextureUsage) describing how the texture will be used.
    agfxTextureUsage usage;
    /// @brief The texture's width, in texels.
    uint32_t width;
    /// @brief The texture's height, in texels.
    uint32_t height;
    /// @brief The texture's depth (for AGFX_TEXTURE_TYPE_3D) or array layer count (for array/cube types); 1 otherwise.
    uint32_t depthOrArrayLayers;
    /// @brief The number of mip levels to allocate.
    uint32_t mipLevels;
} agfxTextureCreateInfo;

/// @brief Creates a new agfxTexture with the specified creation info.
/// @param device A pointer to the agfxDevice to create the texture on.
/// @param createInfo A pointer to an agfxTextureCreateInfo structure containing the creation parameters.
/// @return A pointer to the newly created agfxTexture, or nullptr on failure. The texture starts in AGFX_RESOURCE_STATE_COMMON.
agfxTexture* agfxTextureCreate(agfxDevice* device, const agfxTextureCreateInfo* createInfo);

/// @brief Destroys the specified agfxTexture and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the texture.
/// @param texture A pointer to the agfxTexture to destroy.
void agfxTextureDestroy(agfxDevice* device, agfxTexture* texture);

/// @brief Retrieves the creation info used to create the specified agfxTexture.
/// @param texture A pointer to the agfxTexture to query.
/// @param info A pointer to an agfxTextureCreateInfo structure that will be filled with the texture's creation parameters.
void agfxTextureGetInfo(agfxTexture* texture, agfxTextureCreateInfo* info);

/// @brief Directly uploads CPU data into a region of a texture.
/// @param device A pointer to the agfxDevice that owns the texture.
/// @param texture A pointer to the agfxTexture to upload into.
/// @param region A pointer to an agfxTextureRegion describing the destination region.
/// @param mipLevel The mip level to upload into.
/// @param layer The array layer to upload into.
/// @param data A pointer to the source CPU data.
/// @param dataSize The size in bytes of the source data.
/// @param bytesPerRow The number of bytes per row of the source data.
/// @param bytesPerImage The number of bytes per image (slice) of the source data.
/// @note Unsupported on the D3D12 backend, which does not expose CPU-writable UMA textures; use an
///       agfxUploader-style staging buffer plus agfxComputePassCopyBufferToTexture there instead.
void agfxTextureReplaceRegion(agfxDevice* device, agfxTexture* texture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, const void* data, uint32_t dataSize, uint32_t bytesPerRow, uint32_t bytesPerImage);

/// @brief Sets a debug name for the specified agfxTexture, visible in graphics debuggers.
/// @param texture A pointer to the agfxTexture to name.
/// @param name The null-terminated name string to assign.
void agfxTextureSetName(agfxTexture* texture, const char* name);

// Compute pass
/// @brief Begins a new agfxComputePass on the specified command buffer, used for compute dispatches and copy/blit operations.
/// @param commandBuffer A pointer to the agfxCommandBuffer to begin the pass on. Cannot be a pass created on a queue used only for graphics-incompatible work.
/// @param name A debug name for the pass, visible in graphics debuggers.
/// @return A pointer to the newly created agfxComputePass. Must be ended with agfxComputePassEnd.
/// @note Also flushes any pending "agglomerated" barriers recorded via agfxCommandBufferTextureBarrier/agfxCommandBufferBufferBarrier on the Metal backend.
agfxComputePass* agfxComputePassBegin(agfxCommandBuffer* commandBuffer, const char* name);

/// @brief Inserts an unordered-access-view (read/write hazard) barrier for the specified texture within the pass.
/// @param computePass A pointer to the agfxComputePass to record the barrier in.
/// @param texture A pointer to the agfxTexture to barrier.
/// @note Use between two dispatches that both read/write the same UAV texture to ensure correct ordering.
void agfxComputePassTextureUAVBarrier(agfxComputePass* computePass, agfxTexture* texture);

/// @brief Inserts an unordered-access-view (read/write hazard) barrier for the specified buffer within the pass.
/// @param computePass A pointer to the agfxComputePass to record the barrier in.
/// @param buffer A pointer to the agfxBuffer to barrier.
/// @note Use between two dispatches that both read/write the same UAV buffer to ensure correct ordering.
void agfxComputePassBufferUAVBarrier(agfxComputePass* computePass, agfxBuffer* buffer);

/// @brief Records a copy from a texture region into a buffer.
/// @param computePass A pointer to the agfxComputePass to record the copy in.
/// @param texture A pointer to the source agfxTexture.
/// @param buffer A pointer to the destination agfxBuffer.
/// @param bufferOffset The byte offset into the destination buffer to copy to.
/// @param region A pointer to an agfxTextureRegion describing the source region.
/// @param mipLevel The source mip level.
/// @param layer The source array layer.
/// @param bytesPerRow The number of bytes per row to write into the destination buffer.
/// @param bytesPerImage The number of bytes per image (slice) to write into the destination buffer.
void agfxComputePassCopyTextureToBuffer(agfxComputePass* computePass, agfxTexture* texture, agfxBuffer* buffer, uint64_t bufferOffset, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, uint32_t bytesPerRow, uint32_t bytesPerImage);

/// @brief Records a copy from a buffer into a texture region.
/// @param computePass A pointer to the agfxComputePass to record the copy in.
/// @param buffer A pointer to the source agfxBuffer.
/// @param texture A pointer to the destination agfxTexture.
/// @param region A pointer to an agfxTextureRegion describing the destination region.
/// @param mipLevel The destination mip level.
/// @param layer The destination array layer.
/// @param bytesPerRow The number of bytes per row in the source buffer.
/// @param bytesPerImage The number of bytes per image (slice) in the source buffer.
/// @note This is the standard way to upload CPU texture data on the D3D12 backend: write into a CPU-mappable
///       staging agfxBuffer (see agfxBufferMap), then copy from the staging buffer to the destination texture here.
void agfxComputePassCopyBufferToTexture(agfxComputePass* computePass, agfxBuffer* buffer, agfxTexture* texture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, uint32_t bytesPerRow, uint32_t bytesPerImage);

/// @brief Records a copy between two buffer regions.
/// @param computePass A pointer to the agfxComputePass to record the copy in.
/// @param srcBuffer A pointer to the source agfxBuffer.
/// @param dstBuffer A pointer to the destination agfxBuffer.
/// @param srcOffset The byte offset into the source buffer to copy from.
/// @param dstOffset The byte offset into the destination buffer to copy to.
/// @param size The number of bytes to copy.
void agfxComputePassCopyBufferToBuffer(agfxComputePass* computePass, agfxBuffer* srcBuffer, agfxBuffer* dstBuffer, uint64_t srcOffset, uint64_t dstOffset, uint64_t size);

/// @brief Records a copy from one texture region to another.
/// @param computePass A pointer to the agfxComputePass to record the copy in.
/// @param srcTexture A pointer to the source agfxTexture.
/// @param dstTexture A pointer to the destination agfxTexture.
/// @param region A pointer to an agfxTextureRegion describing the region to copy.
/// @param mipLevel The mip level to copy, used for both source and destination.
/// @param layer The array layer to copy, used for both source and destination.
void agfxComputePassCopyTextureToTexture(agfxComputePass* computePass, agfxTexture* srcTexture, agfxTexture* dstTexture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer);

/// @brief Binds the specified agfxComputePipeline for subsequent dispatches in the pass.
/// @param computePass A pointer to the agfxComputePass to bind the pipeline on.
/// @param pipeline A pointer to the agfxComputePipeline to bind.
void agfxComputePassSetPipeline(agfxComputePass* computePass, agfxComputePipeline* pipeline);

/// @brief Uploads inline push-constant data visible to the currently bound compute pipeline.
/// @param computePass A pointer to the agfxComputePass to push constants on.
/// @param data A pointer to the constant data to upload.
/// @param size The size in bytes of the constant data.
void agfxComputePassPushConstants(agfxComputePass* computePass, const void* data, uint32_t size);

/// @brief Dispatches the currently bound compute pipeline with the given thread group counts.
/// @param computePass A pointer to the agfxComputePass to dispatch on.
/// @param groupCountX The number of thread groups in the X dimension.
/// @param groupCountY The number of thread groups in the Y dimension.
/// @param groupCountZ The number of thread groups in the Z dimension.
void agfxComputePassDispatch(agfxComputePass* computePass, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);

/// @brief Ends the specified agfxComputePass, finalizing its recorded commands.
/// @param computePass A pointer to the agfxComputePass to end.
void agfxComputePassEnd(agfxComputePass* computePass);

// Buffer
/// @brief Bitflags describing how a buffer will be used, combined with bitwise OR. Must match the operations performed on it.
typedef enum agfxBufferUsage {
    /// @brief The buffer can be bound as an index buffer for agfxRenderPassDrawIndexed.
    AGFX_BUFFER_USAGE_INDEX = 1 << 0,
    /// @brief The buffer can be viewed as a constant/uniform buffer (AGFX_BUFFER_VIEW_TYPE_CONSTANT).
    AGFX_BUFFER_USAGE_CONSTANT = 1 << 1,
    /// @brief The buffer can be read from a shader (e.g. via a structured or raw agfxBufferView).
    AGFX_BUFFER_USAGE_SHADER_READ = 1 << 2,
    /// @brief The buffer can be written to from a shader (requires a writeable agfxBufferView).
    AGFX_BUFFER_USAGE_SHADER_WRITE = 1 << 3
} agfxBufferUsage;

/// @brief Where a buffer's memory resides and which side of the PCIe/UMA boundary can access it directly.
typedef enum agfxBufferMemoryType {
    AGFX_BUFFER_MEMORY_TYPE_NONE,
    /// @brief GPU-local memory, not CPU-mappable. Fastest for GPU access; upload via a staging buffer and a copy.
    AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY,
    /// @brief CPU-writable, GPU-readable memory. Mappable with agfxBufferMap for per-frame CPU writes.
    AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU,
    /// @brief GPU-writable, CPU-readable memory. Mappable with agfxBufferMap for reading back GPU-written data.
    AGFX_BUFFER_MEMORY_TYPE_GPU_TO_CPU,
    /// @brief Alias of AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY; the default choice for GPU-only resident buffers.
    AGFX_BUFFER_MEMORY_TYPE_DEFAULT = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY,
    /// @brief Alias of AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU, for staging/upload buffers.
    AGFX_BUFFER_MEMORY_TYPE_UPLOAD = AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU,
    /// @brief Alias of AGFX_BUFFER_MEMORY_TYPE_GPU_TO_CPU, for readback buffers.
    AGFX_BUFFER_MEMORY_TYPE_READBACK = AGFX_BUFFER_MEMORY_TYPE_GPU_TO_CPU
} agfxBufferMemoryType;

/// @brief A structure containing information for creating an agfxBuffer.
typedef struct agfxBufferCreateInfo {
    /// @brief The size of the buffer, in bytes.
    uint64_t size;
    /// @brief The stride of one element, in bytes; used for structured buffer views (AGFX_BUFFER_VIEW_TYPE_STRUCTURED).
    uint64_t stride;
    /// @brief Bitflags (agfxBufferUsage) describing how the buffer will be used.
    agfxBufferUsage usage;
    /// @brief Which memory type/heap the buffer is allocated from.
    agfxBufferMemoryType memoryType;
} agfxBufferCreateInfo;

/// @brief Creates a new agfxBuffer with the specified creation info.
/// @param device A pointer to the agfxDevice to create the buffer on.
/// @param createInfo A pointer to an agfxBufferCreateInfo structure containing the creation parameters.
/// @return A pointer to the newly created agfxBuffer, or nullptr on failure.
agfxBuffer* agfxBufferCreate(agfxDevice* device, const agfxBufferCreateInfo* createInfo);

/// @brief Destroys the specified agfxBuffer and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the buffer.
/// @param buffer A pointer to the agfxBuffer to destroy.
void agfxBufferDestroy(agfxDevice* device, agfxBuffer* buffer);

/// @brief Maps the specified agfxBuffer into CPU-addressable memory.
/// @param buffer A pointer to the agfxBuffer to map.
/// @return A CPU pointer to the buffer's contents.
/// @note Only valid for buffers created with AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU (upload) or
///       AGFX_BUFFER_MEMORY_TYPE_GPU_TO_CPU (readback). Must be paired with agfxBufferUnmap once writes/reads
///       are done. Typical usage is per-frame: map, write updated constant/vertex/index data, unmap, then
///       reference the buffer from a command buffer for that frame.
void* agfxBufferMap(agfxBuffer* buffer);

/// @brief Unmaps a previously mapped agfxBuffer.
/// @param buffer A pointer to the agfxBuffer to unmap.
void agfxBufferUnmap(agfxBuffer* buffer);

/// @brief Sets a debug name for the specified agfxBuffer, visible in graphics debuggers.
/// @param buffer A pointer to the agfxBuffer to name.
/// @param name The null-terminated name string to assign.
void agfxBufferSetName(agfxBuffer* buffer, const char* name);

/// @brief Retrieves the creation info used to create the specified agfxBuffer.
/// @param buffer A pointer to the agfxBuffer to query.
/// @param info A pointer to an agfxBufferCreateInfo structure that will be filled with the buffer's creation parameters.
void agfxBufferGetInfo(agfxBuffer* buffer, agfxBufferCreateInfo* info);

// Texture view
/// @brief A structure containing information for creating an agfxTextureView.
typedef struct agfxTextureViewCreateInfo {
    /// @brief The texture this view is created from.
    agfxTexture* texture;
    /// @brief The pixel format the view reinterprets the texture as.
    agfxTextureFormat format;
    /// @brief The dimensionality the view reinterprets the texture as (e.g. viewing one layer of a 2D array as AGFX_TEXTURE_TYPE_2D).
    agfxTextureType type;
    /// @brief The first mip level covered by the view.
    uint32_t baseMipLevel;
    /// @brief The number of mip levels covered by the view.
    uint32_t mipLevelCount;
    /// @brief The first array layer covered by the view.
    uint32_t baseArrayLayer;
    /// @brief The number of array layers covered by the view.
    uint32_t arrayLayerCount;
    /// @brief Whether this is a writeable UAV/storage-image view (1) or a read-only sampled view (0).
    agfxBool writeable;
} agfxTextureViewCreateInfo;

/// @brief Creates a new agfxTextureView with the specified creation info.
/// @param device A pointer to the agfxDevice to create the view on.
/// @param createInfo A pointer to an agfxTextureViewCreateInfo structure containing the creation parameters.
/// @return A pointer to the newly created agfxTextureView, or nullptr on failure.
/// @note Set writeable to non-zero for a UAV/storage-image view (used with a compute pipeline for read/write
///       access), or zero for a sampled/shader-resource view. baseMipLevel/mipLevelCount and
///       baseArrayLayer/arrayLayerCount select the subresource range the view covers.
agfxTextureView* agfxTextureViewCreate(agfxDevice* device, const agfxTextureViewCreateInfo* createInfo);

/// @brief Destroys the specified agfxTextureView and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the view.
/// @param textureView A pointer to the agfxTextureView to destroy.
void agfxTextureViewDestroy(agfxDevice* device, agfxTextureView* textureView);

/// @brief Retrieves the bindless descriptor handle for the specified agfxTextureView.
/// @param textureView A pointer to the agfxTextureView to query.
/// @return The bindless handle, to be embedded in a shader-visible resource index (e.g. via push constants).
uint64_t agfxTextureViewGetHandle(agfxTextureView* textureView);

// Sampler
/// @brief The filtering mode applied when sampling a texture. Used for magnification, minification, and mip filtering.
typedef enum agfxSamplerFilter {
    AGFX_SAMPLER_FILTER_NEAREST,
    AGFX_SAMPLER_FILTER_LINEAR
} agfxSamplerFilter;

/// @brief How texture coordinates outside the [0, 1] range are handled.
typedef enum agfxSamplerAddressMode {
    /// @brief The texture tiles/repeats.
    AGFX_SAMPLER_ADDRESS_MODE_REPEAT,
    /// @brief The texture tiles, mirroring on every other repeat.
    AGFX_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    /// @brief Coordinates are clamped to the edge texels.
    AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    /// @brief Coordinates outside the range sample a fixed border color.
    AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
} agfxSamplerAddressMode;

/// @brief A comparison function, used for both sampler comparison (shadow map) sampling and depth testing.
typedef enum agfxComparisonFunction {
    AGFX_COMPARISON_FUNCTION_NEVER,
    AGFX_COMPARISON_FUNCTION_LESS,
    AGFX_COMPARISON_FUNCTION_EQUAL,
    AGFX_COMPARISON_FUNCTION_LESS_EQUAL,
    AGFX_COMPARISON_FUNCTION_GREATER,
    AGFX_COMPARISON_FUNCTION_NOT_EQUAL,
    AGFX_COMPARISON_FUNCTION_GREATER_EQUAL,
    AGFX_COMPARISON_FUNCTION_ALWAYS
} agfxComparisonFunction;

/// @brief A structure containing information for creating an agfxSampler.
typedef struct agfxSamplerCreateInfo {
    /// @brief The filtering mode used for magnification, minification, and mip filtering.
    agfxSamplerFilter filter;
    /// @brief The address mode applied to the U (horizontal) texture coordinate.
    agfxSamplerAddressMode addressModeU;
    /// @brief The address mode applied to the V (vertical) texture coordinate.
    agfxSamplerAddressMode addressModeV;
    /// @brief The address mode applied to the W (depth) texture coordinate, for 3D textures.
    agfxSamplerAddressMode addressModeW;
    /// @brief A bias added to the computed mip level before sampling.
    float mipLodBias;
    /// @brief The maximum anisotropic filtering level to apply.
    float maxAnisotropy;
    /// @brief The comparison function used for comparison (shadow) sampling; ignored for ordinary color sampling.
    agfxComparisonFunction comparisonFunction;
    /// @brief The minimum mip level clamp.
    float minLod;
    /// @brief The maximum mip level clamp.
    float maxLod;
    /// @brief A bias added to the sampled LOD, applied before the min/max LOD clamp.
    float lodBias;
} agfxSamplerCreateInfo;

/// @brief Creates a new agfxSampler with the specified creation info.
/// @param device A pointer to the agfxDevice to create the sampler on.
/// @param createInfo A pointer to an agfxSamplerCreateInfo structure containing the creation parameters.
/// @return A pointer to the newly created agfxSampler, or nullptr on failure.
agfxSampler* agfxSamplerCreate(agfxDevice* device, const agfxSamplerCreateInfo* createInfo);

/// @brief Destroys the specified agfxSampler and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the sampler.
/// @param sampler A pointer to the agfxSampler to destroy.
void agfxSamplerDestroy(agfxDevice* device, agfxSampler* sampler);

/// @brief Retrieves the bindless descriptor handle for the specified agfxSampler.
/// @param sampler A pointer to the agfxSampler to query.
/// @return The bindless handle, to be embedded in a shader-visible resource index (e.g. via push constants).
uint64_t agfxSamplerGetHandle(agfxSampler* sampler);

// Buffer view
/// @brief The kind of view a buffer is interpreted as when bound to a shader.
typedef enum agfxBufferViewType {
    /// @brief A byte-address (raw) buffer, indexed by byte offset in the shader.
    AGFX_BUFFER_VIEW_TYPE_RAW,
    /// @brief A structured buffer, indexed by element using the buffer's stride.
    AGFX_BUFFER_VIEW_TYPE_STRUCTURED,
    /// @brief A constant/uniform buffer.
    AGFX_BUFFER_VIEW_TYPE_CONSTANT
} agfxBufferViewType;

/// @brief A structure containing information for creating an agfxBufferView.
typedef struct agfxBufferViewCreateInfo {
    /// @brief The buffer this view is created from.
    agfxBuffer* buffer;
    /// @brief How the buffer is interpreted when bound to a shader.
    agfxBufferViewType type;
    /// @brief The byte offset into the buffer where the view begins.
    uint64_t offset;
    /// @brief Whether this is a writeable UAV view (1) or a read-only shader-resource view (0).
    agfxBool writeable;
} agfxBufferViewCreateInfo;

/// @brief Creates a new agfxBufferView with the specified creation info.
/// @param device A pointer to the agfxDevice to create the view on.
/// @param createInfo A pointer to an agfxBufferViewCreateInfo structure containing the creation parameters.
/// @return A pointer to the newly created agfxBufferView, or nullptr on failure.
/// @note Use AGFX_BUFFER_VIEW_TYPE_CONSTANT for constant/uniform buffers, AGFX_BUFFER_VIEW_TYPE_STRUCTURED for
///       structured shader-resource/UAV buffers (uses the stride from the buffer's creation info), or
///       AGFX_BUFFER_VIEW_TYPE_RAW for byte-address access. Set writeable to non-zero for UAV access.
agfxBufferView* agfxBufferViewCreate(agfxDevice* device, const agfxBufferViewCreateInfo* createInfo);

/// @brief Destroys the specified agfxBufferView and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the view.
/// @param bufferView A pointer to the agfxBufferView to destroy.
void agfxBufferViewDestroy(agfxDevice* device, agfxBufferView* bufferView);

/// @brief Retrieves the bindless descriptor handle for the specified agfxBufferView.
/// @param bufferView A pointer to the agfxBufferView to query.
/// @return The bindless handle, to be embedded in a shader-visible resource index (e.g. via push constants).
uint64_t agfxBufferViewGetHandle(agfxBufferView* bufferView);

// Render target
/// @brief A structure containing information for creating an agfxRenderTarget.
typedef struct agfxRenderTargetCreateInfo {
    /// @brief The texture this render target renders into.
    agfxTexture* texture;
    /// @brief The pixel format to render with, or AGFX_TEXTURE_FORMAT_UNKNOWN to inherit the texture's own format.
    agfxTextureFormat format;
    /// @brief The mip level to render into.
    uint32_t mipLevel;
    /// @brief The array layer to render into.
    uint32_t arrayLayer;
    /// @brief Whether this render target is attached as a depth-stencil target (1) or a color target (0).
    agfxBool isDepth;
} agfxRenderTargetCreateInfo;

/// @brief Creates a new agfxRenderTarget with the specified creation info.
/// @param device A pointer to the agfxDevice to create the render target on.
/// @param createInfo A pointer to an agfxRenderTargetCreateInfo structure containing the creation parameters.
/// @return A pointer to the newly created agfxRenderTarget, or nullptr on failure.
/// @note For AGFX_TEXTURE_FORMAT_UNKNOWN, the render target inherits the texture's own format; this is used
///       for the swap chain's back buffer, whose format is only known via agfxSwapChainGetFormat. Set isDepth
///       to non-zero when wrapping a depth texture so it is attached as a depth rather than color target.
///       Render targets are lightweight and are typically created per-frame and destroyed after the pass ends.
agfxRenderTarget* agfxRenderTargetCreate(agfxDevice* device, const agfxRenderTargetCreateInfo* createInfo);

/// @brief Destroys the specified agfxRenderTarget and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the render target.
/// @param renderTarget A pointer to the agfxRenderTarget to destroy.
/// @note Only destroys the render target wrapper; the underlying agfxTexture is unaffected and must be destroyed separately.
void agfxRenderTargetDestroy(agfxDevice* device, agfxRenderTarget* renderTarget);

// Render pass
/// @brief Specifies how a render pass attachment's existing contents are treated at the start of the pass.
typedef enum agfxLoadOperation {
    /// @brief Preserve the attachment's existing contents.
    AGFX_LOAD_OPERATION_LOAD,
    /// @brief Clear the attachment to the attachment's clearColor at the start of the pass.
    AGFX_LOAD_OPERATION_CLEAR,
    /// @brief The attachment's initial contents are undefined; use when every pixel will be written.
    AGFX_LOAD_OPERATION_DONT_CARE
} agfxLoadOp;

/// @brief Specifies whether a render pass attachment's contents are written back to memory at the end of the pass.
typedef enum agfxStoreOperation {
    /// @brief Write the attachment's contents back to memory.
    AGFX_STORE_OPERATION_STORE,
    /// @brief Discard the attachment's contents; use for transient attachments that are not read afterwards.
    AGFX_STORE_OPERATION_DONT_CARE
} agfxStoreOp;

/// @brief Describes a single color or depth attachment used by an agfxRenderPass.
typedef struct agfxRenderPassAttachment {
    /// @brief The render target to attach.
    agfxRenderTarget* renderTarget;
    /// @brief How the attachment's existing contents are treated at the start of the pass.
    agfxLoadOp loadOp;
    /// @brief Whether the attachment's contents are written back to memory at the end of the pass.
    agfxStoreOp storeOp;
    /// @brief The RGBA clear color used when loadOp is AGFX_LOAD_OPERATION_CLEAR. Unused for depth attachments.
    float clearColor[4];
} agfxRenderPassAttachment;

/// @brief A structure containing information for beginning an agfxRenderPass.
typedef struct agfxRenderPassCreateInfo {
    /// @brief The color attachments to render into.
    agfxRenderPassAttachment colorAttachments[8];
    /// @brief The number of active color attachments (and thus valid entries in colorAttachments).
    uint32_t colorAttachmentCount;
    /// @brief The depth attachment to render into. Only used if hasDepthAttachment is non-zero.
    agfxRenderPassAttachment depthAttachment;
    /// @brief Whether depthAttachment is valid and should be bound. Set to 1 to enable, 0 to disable.
    agfxBool hasDepthAttachment;
    /// @brief The width of the render area, in pixels; must match the attachments' dimensions.
    uint32_t width;
    /// @brief The height of the render area, in pixels; must match the attachments' dimensions.
    uint32_t height;
    /// @brief A debug name for the pass, visible in graphics debuggers.
    const char* name;
} agfxRenderPassCreateInfo;

/// @brief Begins a new agfxRenderPass on the specified command buffer, targeting the given color/depth attachments.
/// @param cmdBuffer A pointer to the agfxCommandBuffer to begin the pass on.
/// @param createInfo A pointer to an agfxRenderPassCreateInfo structure describing the attachments and dimensions.
/// @return A pointer to the newly created agfxRenderPass. Must be ended with agfxRenderPassEnd.
/// @note Also flushes any pending "agglomerated" barriers recorded via agfxCommandBufferTextureBarrier/agfxCommandBufferBufferBarrier on the Metal backend.
///       Every texture used as a color or depth attachment must already be in AGFX_RESOURCE_STATE_RENDER_TARGET
///       or AGFX_RESOURCE_STATE_DEPTH_WRITE respectively before the pass begins.
agfxRenderPass* agfxRenderPassBegin(agfxCommandBuffer* cmdBuffer, const agfxRenderPassCreateInfo* createInfo);

/// @brief Sets the viewport transform used for subsequent draws in the pass.
/// @param renderPass A pointer to the agfxRenderPass to set the viewport on.
/// @param x The viewport's origin X coordinate, in pixels.
/// @param y The viewport's origin Y coordinate, in pixels.
/// @param width The viewport's width, in pixels.
/// @param height The viewport's height, in pixels.
/// @param minDepth The minimum depth value of the viewport, typically 0.
/// @param maxDepth The maximum depth value of the viewport, typically 1.
void agfxRenderPassSetViewport(agfxRenderPass* renderPass, float x, float y, float width, float height, float minDepth, float maxDepth);

/// @brief Sets the scissor rectangle used for subsequent draws in the pass.
/// @param renderPass A pointer to the agfxRenderPass to set the scissor on.
/// @param x The scissor rectangle's origin X coordinate, in pixels.
/// @param y The scissor rectangle's origin Y coordinate, in pixels.
/// @param width The scissor rectangle's width, in pixels.
/// @param height The scissor rectangle's height, in pixels.
void agfxRenderPassSetScissor(agfxRenderPass* renderPass, uint32_t x, uint32_t y, uint32_t width, uint32_t height);

/// @brief Binds the specified agfxRenderPipeline for subsequent draws in the pass.
/// @param renderPass A pointer to the agfxRenderPass to bind the pipeline on.
/// @param pipeline A pointer to the agfxRenderPipeline to bind.
void agfxRenderPassSetPipeline(agfxRenderPass* renderPass, agfxRenderPipeline* pipeline);

/// @brief Uploads inline push-constant data visible to the currently bound render pipeline's shader stages.
/// @param renderPass A pointer to the agfxRenderPass to push constants on.
/// @param data A pointer to the constant data to upload.
/// @param size The size in bytes of the constant data.
void agfxRenderPassPushConstants(agfxRenderPass* renderPass, const void* data, uint32_t size);

/// @brief Issues a non-indexed draw call using the currently bound render pipeline.
/// @param renderPass A pointer to the agfxRenderPass to draw on.
/// @param vertexCount The number of vertices to draw per instance.
/// @param instanceCount The number of instances to draw.
/// @param firstVertex The index of the first vertex to draw.
/// @param firstInstance The index of the first instance to draw.
void agfxRenderPassDraw(agfxRenderPass* renderPass, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance);

/// @brief Issues an indexed draw call using the currently bound render pipeline.
/// @param renderPass A pointer to the agfxRenderPass to draw on.
/// @param indexBuffer A pointer to the agfxBuffer containing indices, created with AGFX_BUFFER_USAGE_INDEX.
/// @param indexCount The number of indices to draw per instance.
/// @param instanceCount The number of instances to draw.
/// @param firstIndex The index of the first index to read from the index buffer.
/// @param vertexOffset A value added to each index before indexing into the vertex buffer(s).
/// @param firstInstance The index of the first instance to draw.
void agfxRenderPassDrawIndexed(agfxRenderPass* renderPass, agfxBuffer* indexBuffer, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, uint32_t vertexOffset, uint32_t firstInstance);

/// @brief Issues a mesh-shader draw call using the currently bound render pipeline's task/mesh shaders.
/// @param renderPass A pointer to the agfxRenderPass to draw on.
/// @param groupCountX The number of task/mesh groups in the X dimension.
/// @param groupCountY The number of task/mesh groups in the Y dimension.
/// @param groupCountZ The number of task/mesh groups in the Z dimension.
/// @note Requires a pipeline created with meshShader (and optionally taskShader) set in its agfxRenderPipelineCreateInfo.
void agfxRenderPassDrawMesh(agfxRenderPass* renderPass, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);

/// @brief Ends the specified agfxRenderPass, finalizing its recorded commands.
/// @param renderPass A pointer to the agfxRenderPass to end.
void agfxRenderPassEnd(agfxRenderPass* renderPass);

// Swap chain
/// @brief A structure containing information for creating an agfxSwapChain.
typedef struct agfxSwapChainCreateInfo {
    /// @brief The agfxCommandQueue that presents to this swap chain; must be a graphics queue.
    agfxCommandQueue* queue;
    /// @brief The number of back buffer images in the swap chain (e.g. 2 for double buffering).
    uint32_t imageCount;
    /// @brief The width of the swap chain images, in pixels.
    uint32_t width;
    /// @brief The height of the swap chain images, in pixels.
    uint32_t height;

    /// @brief Whether to create an HDR swap chain. Set to 1 to enable, 0 to disable.
    agfxBool isHDR;
    /// @brief Whether to synchronize presentation with the display's refresh rate. Set to 1 to enable, 0 to disable.
    agfxBool vsync;
    /// @brief The native window/layer handle to present to: an HWND on Windows, or a CAMetalLayer* on macOS.
    void* handle; // HWND or CAMetalLayer*
} agfxSwapChainCreateInfo;

/// @brief Creates a new agfxSwapChain with the specified creation info.
/// @param device A pointer to the agfxDevice to create the swap chain on.
/// @param createInfo A pointer to an agfxSwapChainCreateInfo structure containing the creation parameters.
/// @return A pointer to the newly created agfxSwapChain, or nullptr on failure.
agfxSwapChain* agfxSwapChainCreate(agfxDevice* device, const agfxSwapChainCreateInfo* createInfo);

/// @brief Destroys the specified agfxSwapChain and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the swap chain.
/// @param swapChain A pointer to the agfxSwapChain to destroy.
void agfxSwapChainDestroy(agfxDevice* device, agfxSwapChain* swapChain);

/// @brief Resizes the specified agfxSwapChain to the given dimensions, e.g. in response to a window resize.
/// @param device A pointer to the agfxDevice that owns the swap chain.
/// @param swapChain A pointer to the agfxSwapChain to resize.
/// @param width The new width, in pixels.
/// @param height The new height, in pixels.
/// @note Ensure the GPU has finished all work referencing the swap chain's textures before resizing; the
///       common pattern (see agfx_demo_main.cpp) is to destroy and recreate the swap chain on resize instead.
void agfxSwapChainResize(agfxDevice* device, agfxSwapChain* swapChain, uint32_t width, uint32_t height);

/// @brief Retrieves the pixel format of the specified agfxSwapChain's back buffer textures.
/// @param swapChain A pointer to the agfxSwapChain to query.
/// @return The agfxTextureFormat of the swap chain's back buffer textures.
agfxTextureFormat agfxSwapChainGetFormat(agfxSwapChain* swapChain);

/// @brief Acquires the next available back buffer texture from the specified agfxSwapChain to render into.
/// @param swapChain A pointer to the agfxSwapChain to acquire from.
/// @return A pointer to the acquired agfxTexture, starting in AGFX_RESOURCE_STATE_PRESENT. Must be
///         barriered to AGFX_RESOURCE_STATE_RENDER_TARGET before use and back to AGFX_RESOURCE_STATE_PRESENT
///         before agfxSwapChainPresent. Wrap it in an agfxRenderTarget (with format AGFX_TEXTURE_FORMAT_UNKNOWN) to render to it.
agfxTexture* agfxSwapChainAcquireNextTexture(agfxSwapChain* swapChain);

/// @brief Presents the most recently acquired back buffer texture to the screen.
/// @param swapChain A pointer to the agfxSwapChain to present.
/// @note Call after the command buffer that renders to and barriers the acquired texture back to
///       AGFX_RESOURCE_STATE_PRESENT has been submitted via agfxCommandQueueSubmit.
void agfxSwapChainPresent(agfxSwapChain* swapChain);

// Shader module
/// @brief The shader stage a compiled agfxShaderModule targets.
typedef enum agfxShaderModuleType {
    /// @brief A vertex shader, used with agfxRenderPipelineCreateInfo::vertexShader.
    AGFX_SHADER_MODULE_TYPE_VERTEX,
    /// @brief A fragment (pixel) shader, used with agfxRenderPipelineCreateInfo::fragmentShader.
    AGFX_SHADER_MODULE_TYPE_FRAGMENT,
    /// @brief A compute shader, used with agfxComputePipelineCreateInfo::computeShader.
    AGFX_SHADER_MODULE_TYPE_COMPUTE,
    /// @brief A task (amplification) shader, used with agfxRenderPipelineCreateInfo::taskShader.
    AGFX_SHADER_MODULE_TYPE_TASK,
    /// @brief A mesh shader, used with agfxRenderPipelineCreateInfo::meshShader.
    AGFX_SHADER_MODULE_TYPE_MESH
} agfxShaderModuleType;

/// @brief A structure containing information for creating an agfxShaderModule.
typedef struct agfxShaderModuleCreateInfo {
    /// @brief A pointer to the compiled shader bytecode (DXIL, compiled via the Metal shader converter into a metallib, etc. depending on backend).
    uint8_t* code;
    /// @brief The size in bytes of the compiled shader bytecode.
    uint64_t codeSize;
    /// @brief The name of the entry point function within the compiled shader.
    const char* entryPoint;
    /// @brief The shader stage this module targets. Must match the pipeline slot it will be attached to.
    agfxShaderModuleType type;
} agfxShaderModuleCreateInfo;

/// @brief Creates a new agfxShaderModule from compiled shader bytecode.
/// @param device A pointer to the agfxDevice to create the shader module on.
/// @param createInfo A pointer to an agfxShaderModuleCreateInfo structure containing the creation parameters.
/// @return A pointer to the newly created agfxShaderModule, or nullptr on failure.
agfxShaderModule* agfxShaderModuleCreate(agfxDevice* device, const agfxShaderModuleCreateInfo* createInfo);

/// @brief Destroys the specified agfxShaderModule and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the shader module.
/// @param shaderModule A pointer to the agfxShaderModule to destroy.
/// @note Safe to destroy once the pipeline(s) built from it have been created; the module is not referenced afterwards.
void agfxShaderModuleDestroy(agfxDevice* device, agfxShaderModule* shaderModule);

// Render pipeline

/// @brief The primitive topology assembled from vertex/index data by a render pipeline.
typedef enum agfxTopology {
    AGFX_TOPOLOGY_TRIANGLES,
    AGFX_TOPOLOGY_LINES,
    AGFX_TOPOLOGY_POINTS
} agfxTopology;

/// @brief Specifies which triangle winding face(s) are culled by a render pipeline.
typedef enum agfxCullMode {
    /// @brief No triangles are culled.
    AGFX_CULL_MODE_NONE,
    /// @brief Front-facing triangles (per agfxFrontFace) are culled.
    AGFX_CULL_MODE_FRONT,
    /// @brief Back-facing triangles (per agfxFrontFace) are culled.
    AGFX_CULL_MODE_BACK
} agfxCullMode;

/// @brief Specifies which vertex winding order is considered front-facing.
typedef enum agfxFrontFace {
    AGFX_FRONT_FACE_CLOCKWISE,
    AGFX_FRONT_FACE_COUNTER_CLOCKWISE
} agfxFrontFace;

/// @brief Specifies how triangles are rasterized: filled or as wireframe outlines.
typedef enum agfxFillMode {
    AGFX_FILL_MODE_SOLID,
    AGFX_FILL_MODE_WIREFRAME
} agfxFillMode;

/// @brief A blend factor applied to the source or destination color/alpha during blending.
typedef enum agfxBlendFactor {
    AGFX_BLEND_FACTOR_ZERO,
    AGFX_BLEND_FACTOR_ONE,
    AGFX_BLEND_FACTOR_SRC_COLOR,
    AGFX_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
    AGFX_BLEND_FACTOR_DST_COLOR,
    AGFX_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
    AGFX_BLEND_FACTOR_SRC_ALPHA,
    AGFX_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    AGFX_BLEND_FACTOR_DST_ALPHA,
    AGFX_BLEND_FACTOR_ONE_MINUS_DST_ALPHA
} agfxBlendFactor;

/// @brief The arithmetic operation combining the source and destination values during blending.
typedef enum agfxBlendOperation {
    AGFX_BLEND_OPERATION_ADD,
    AGFX_BLEND_OPERATION_SUBTRACT,
    AGFX_BLEND_OPERATION_REVERSE_SUBTRACT,
    AGFX_BLEND_OPERATION_MIN,
    AGFX_BLEND_OPERATION_MAX
} agfxBlendOperation;

/// @brief A structure containing information for creating an agfxRenderPipeline.
/// @note Either both vertexShader/fragmentShader (classic pipeline) or meshShader (and optionally taskShader)
///       (mesh-shading pipeline) should be provided, matching the draw call used with the pipeline
///       (agfxRenderPassDraw/agfxRenderPassDrawIndexed vs. agfxRenderPassDrawMesh). depthFormat and
///       colorFormats[]/colorAttachmentCount must match the render pass attachments the pipeline will be used with.
typedef struct agfxRenderPipelineCreateInfo {
    /// @brief A debug name for the pipeline, visible in graphics debuggers.
    const char* name;
    /// @brief Whether the pipeline supports indirect (GPU-driven) draw arguments. Set to 1 to enable, 0 to disable.
    agfxBool supportsIndirect;

    /// @brief The rasterizer fill mode.
    agfxFillMode fillMode;
    /// @brief The rasterizer cull mode.
    agfxCullMode cullMode;
    /// @brief The winding order considered front-facing.
    agfxFrontFace frontFace;
    /// @brief The primitive topology used by non-mesh-shading draws.
    agfxTopology topology;

    /// @brief Whether depth testing is enabled. Set to 1 to enable, 0 to disable.
    agfxBool depthTestEnable;
    /// @brief Whether depth writes are enabled. Set to 1 to enable, 0 to disable.
    agfxBool depthWriteEnable;
    /// @brief Whether depth clamping is enabled instead of clipping. Set to 1 to enable, 0 to disable.
    agfxBool depthClampEnable;
    /// @brief The comparison function used for depth testing.
    agfxComparisonFunction depthCompareOp;
    /// @brief The format of the depth attachment this pipeline will render to. Ignored if there is no depth attachment.
    agfxTextureFormat depthFormat;

    /// @brief Per-color-attachment: whether blending is enabled. Set to 1 to enable, 0 to disable.
    agfxBool blendEnable[8];
    /// @brief Per-color-attachment: the source color blend factor.
    agfxBlendFactor srcColorBlendFactor[8];
    /// @brief Per-color-attachment: the destination color blend factor.
    agfxBlendFactor dstColorBlendFactor[8];
    /// @brief Per-color-attachment: the color blend operation.
    agfxBlendOperation colorBlendOp[8];
    /// @brief Per-color-attachment: the source alpha blend factor.
    agfxBlendFactor srcAlphaBlendFactor[8];
    /// @brief Per-color-attachment: the destination alpha blend factor.
    agfxBlendFactor dstAlphaBlendFactor[8];
    /// @brief Per-color-attachment: the alpha blend operation.
    agfxBlendOperation alphaBlendOp[8];
    /// @brief The pixel format of each color attachment this pipeline will render to.
    agfxTextureFormat colorFormats[8];
    /// @brief The number of active color attachments (and thus valid entries in the per-attachment arrays above).
    uint32_t colorAttachmentCount;

    /// @brief The vertex shader stage. Used together with fragmentShader for a classic (non-mesh) pipeline.
    agfxShaderModule* vertexShader;
    /// @brief The fragment (pixel) shader stage.
    agfxShaderModule* fragmentShader;
    /// @brief The optional task (amplification) shader stage, used together with meshShader.
    agfxShaderModule* taskShader;
    /// @brief The mesh shader stage. When set, draw with agfxRenderPassDrawMesh instead of agfxRenderPassDraw/DrawIndexed.
    agfxShaderModule* meshShader;

    /// @brief The task shader's thread group size in the X dimension. Must match the shader's declared group size.
    uint32_t taskGroupSizeX;
    /// @brief The task shader's thread group size in the Y dimension. Must match the shader's declared group size.
    uint32_t taskGroupSizeY;
    /// @brief The task shader's thread group size in the Z dimension. Must match the shader's declared group size.
    uint32_t taskGroupSizeZ;

    /// @brief The mesh shader's thread group size in the X dimension. Must match the shader's declared group size.
    uint32_t meshGroupSizeX;
    /// @brief The mesh shader's thread group size in the Y dimension. Must match the shader's declared group size.
    uint32_t meshGroupSizeY;
    /// @brief The mesh shader's thread group size in the Z dimension. Must match the shader's declared group size.
    uint32_t meshGroupSizeZ;
} agfxRenderPipelineCreateInfo;

/// @brief Creates a new agfxRenderPipeline with the specified creation info.
/// @param device A pointer to the agfxDevice to create the pipeline on.
/// @param createInfo A pointer to an agfxRenderPipelineCreateInfo structure containing the creation parameters.
/// @return A pointer to the newly created agfxRenderPipeline, or nullptr on failure.
agfxRenderPipeline* agfxRenderPipelineCreate(agfxDevice* device, const agfxRenderPipelineCreateInfo* createInfo);

/// @brief Destroys the specified agfxRenderPipeline and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the pipeline.
/// @param pipeline A pointer to the agfxRenderPipeline to destroy.
void agfxRenderPipelineDestroy(agfxDevice* device, agfxRenderPipeline* pipeline);

// Compute pipeline

/// @brief A structure containing information for creating an agfxComputePipeline.
typedef struct agfxComputePipelineCreateInfo {
    /// @brief A debug name for the pipeline, visible in graphics debuggers.
    const char* name;
    /// @brief The compute shader stage.
    agfxShaderModule* computeShader;
    /// @brief The shader's thread group size in the X dimension. Must match the shader's declared group size.
    uint32_t groupSizeX;
    /// @brief The shader's thread group size in the Y dimension. Must match the shader's declared group size.
    uint32_t groupSizeY;
    /// @brief The shader's thread group size in the Z dimension. Must match the shader's declared group size.
    uint32_t groupSizeZ;
} agfxComputePipelineCreateInfo;

/// @brief Creates a new agfxComputePipeline with the specified creation info.
/// @param device A pointer to the agfxDevice to create the pipeline on.
/// @param createInfo A pointer to an agfxComputePipelineCreateInfo structure containing the creation parameters.
/// @return A pointer to the newly created agfxComputePipeline, or nullptr on failure.
agfxComputePipeline* agfxComputePipelineCreate(agfxDevice* device, const agfxComputePipelineCreateInfo* createInfo);

/// @brief Destroys the specified agfxComputePipeline and releases all associated resources.
/// @param device A pointer to the agfxDevice that owns the pipeline.
/// @param pipeline A pointer to the agfxComputePipeline to destroy.
void agfxComputePipelineDestroy(agfxDevice* device, agfxComputePipeline* pipeline);
