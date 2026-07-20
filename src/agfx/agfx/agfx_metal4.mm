/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 00:13:06
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "agfx.h"
#include <Metal/Metal.h>
#include <QuartzCore/QuartzCore.h>
#include <vector>
#include <Foundation/Foundation.h>

#define IR_PRIVATE_IMPLEMENTATION
#include "msc_runtime/metal_irconverter_runtime.h"

#include <dispatch/dispatch.h>
#include <mach/mach_time.h>

// Types
struct agfxRenderPipeline {
    agfxRenderPipelineCreateInfo createInfo;
    id<MTLRenderPipelineState> pipelineState;
    id<MTLDepthStencilState> depthStencilState;
};

struct agfxComputePipeline {
    agfxComputePipelineCreateInfo createInfo;
    id<MTLComputePipelineState> pipelineState;
};

struct agfxTLAB {
    uint8_t bytes[128];
    uint32_t drawID;
};

struct agfxBuffer {
    agfxBufferCreateInfo createInfo;
    id<MTLBuffer> buffer;
};

struct agfxSwapChain {
    agfxSwapChainCreateInfo createInfo;
    CAMetalLayer* metalLayer;
    agfxDevice* device;
    id<CAMetalDrawable> currentDrawable;
    agfxTexture* wrapperTexture; // reused each frame, ->texture repointed at the current drawable's texture
};

struct agfxMetalBindlessAllocation {
    uint64_t handle;
    MTLResourceID resourceID;
};

struct agfxAccelerationStructure {
    agfxAccelerationStructureCreateInfo createInfo;
    id<MTLAccelerationStructure> accelerationStructure;
    MTLAccelerationStructureSizes sizes;

    // For top level
    id<MTLBuffer> instanceBuffer;
    MTLIndirectAccelerationStructureInstanceDescriptor* mappedInstanceBuffer;
    agfxMetalBindlessAllocation asBindlessHandle;

    id<MTLBuffer> resourceIDBuffer; // Needed by MSC
    void* mappedResourceIDBuffer;
    MTL4InstanceAccelerationStructureDescriptor* mtlTopLevelDescriptor;

    uint32_t currentInstanceCount;

    // For bottom level
    MTL4PrimitiveAccelerationStructureDescriptor* mtlBottomLevelDescriptor;
};

struct agfxSlotAllocator {
    agfxSlotAllocator(uint64_t maxSlots) {
        this->maxSlots = maxSlots;
        bitmapSize = (maxSlots + 63) / 64;

        bitmap.resize(bitmapSize, 0);
        freeSlots.reserve(maxSlots);
        for (uint64_t i = 0; i < maxSlots; ++i) {
            freeSlots.push_back(i);
        }
    }

    ~agfxSlotAllocator() {
        bitmap.clear();
        freeSlots.clear();
    }

    uint64_t allocate() {
        if (freeSlots.empty()) return UINT64_MAX;
        int32_t slot = freeSlots.back();
        freeSlots.pop_back();
        setBit(slot);
        return slot;
    }

    void free(uint64_t slot) {
        if (slot >= maxSlots || slot < 0) return;
        if (testBit(slot)) {
            clearBit(slot);
            freeSlots.push_back(slot);
        }
    }

    void setBit(int index) {
        bitmap[index / 64] |= (1ULL << (index % 64));
    }

    void clearBit(int index) {
        bitmap[index / 64] &= ~(1ULL << (index % 64));
    }

    bool testBit(int index) {
        return (bitmap[index / 64] & (1ULL << (index % 64))) != 0;
    }

    uint64_t maxSlots;
    uint64_t bitmapSize;
    std::vector<uint64_t> bitmap;
    std::vector<uint32_t> freeSlots;
};

struct agfxMetalAllocation {
    uint64_t offset;
    uint64_t gpuAddress;
    void* cpuPointer;
};

struct agfxMetalLinearAllocator {
    agfxMetalLinearAllocator(id<MTLDevice> device, id<MTLResidencySet> residencySet, uint64_t size) {
        buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        [residencySet addAllocation:buffer];
        offset = 0;
    }

    ~agfxMetalLinearAllocator() {
        buffer = nil;
    }

    agfxMetalAllocation allocate(uint64_t size) {
        static const uint64_t kAlignment = 16;
        offset = (offset + kAlignment - 1) & ~(kAlignment - 1);

        agfxMetalAllocation allocation;
        allocation.offset = offset;
        allocation.gpuAddress = buffer.gpuAddress + offset;
        allocation.cpuPointer = (uint8_t*)buffer.contents + offset;

        offset += size;
        return allocation;
    }

    void reset() {
        offset = 0;
    }

    id<MTLBuffer> buffer;
    uint64_t offset;
};

struct agfxMetalBindlessManager {
    const uint32_t maxResources = 512'000;
    const uint32_t maxSamplers = 2048;
    const uint32_t maxTextureViews = 256'000;

    agfxMetalBindlessManager(id<MTLDevice> device, id<MTLResidencySet> residencySet)
        : resourceAllocator(maxResources), samplerAllocator(maxSamplers), textureViewAllocator(maxTextureViews) {
        this->device = device;
        this->residencySet = residencySet;

        resourceHeapBuffer = [device newBufferWithLength:maxResources * sizeof(IRDescriptorTableEntry) options:MTLResourceStorageModeShared];
        [residencySet addAllocation:resourceHeapBuffer];

        samplerHeapBuffer = [device newBufferWithLength:maxSamplers * sizeof(IRDescriptorTableEntry) options:MTLResourceStorageModeShared];
        [residencySet addAllocation:samplerHeapBuffer];

        MTLResourceViewPoolDescriptor* textureViewPoolDesc = [MTLResourceViewPoolDescriptor new];
        textureViewPoolDesc.label = @"TextureViewPool";
        textureViewPoolDesc.resourceViewCount = maxTextureViews;

        textureViewPool = [device newTextureViewPoolWithDescriptor:textureViewPoolDesc error:nil];
    }

    ~agfxMetalBindlessManager() {
        resourceHeapBuffer = nil;
        samplerHeapBuffer = nil;
        textureViewPool = nil;
    }

    agfxMetalBindlessAllocation writeTextureView(MTLTextureViewDescriptor* descriptor, id<MTLTexture> texture) {
        uint64_t slotIndex = textureViewAllocator.allocate();
        if (slotIndex == UINT64_MAX) {
            return {};
        }

        agfxMetalBindlessAllocation allocation;
        allocation.handle = slotIndex;
        allocation.resourceID = [textureViewPool setTextureView:texture descriptor:descriptor atIndex:slotIndex];

        IRDescriptorTableEntry* entry = (IRDescriptorTableEntry*)resourceHeapBuffer.contents;
        entry[slotIndex].textureViewID = allocation.resourceID._impl;
        entry[slotIndex].metadata = 0;
        return allocation;
    }

    void freeTextureView(uint64_t handle) {
        textureViewAllocator.free(handle);
    }

    agfxMetalBindlessAllocation writeSampler(id<MTLSamplerState> sampler, float lodBias) {
        uint64_t slotIndex = samplerAllocator.allocate();
        if (slotIndex == UINT64_MAX) {
            return {};
        }

        agfxMetalBindlessAllocation allocation;
        allocation.handle = slotIndex;
        allocation.resourceID = sampler.gpuResourceID;

        IRDescriptorTableEntry* entry = (IRDescriptorTableEntry*)samplerHeapBuffer.contents;
        IRDescriptorTableSetSampler(&entry[slotIndex], sampler, lodBias);
        return allocation;
    }

    void freeSampler(uint64_t handle) {
        samplerAllocator.free(handle);
    }

    agfxMetalBindlessAllocation writeBuffer(id<MTLBuffer> buffer, uint64_t offset) {
        uint64_t slotIndex = resourceAllocator.allocate();
        if (slotIndex == UINT64_MAX) {
            return {};
        }

        agfxMetalBindlessAllocation allocation;
        allocation.handle = slotIndex;
        allocation.resourceID = {};

        IRDescriptorTableEntry* entry = (IRDescriptorTableEntry*)resourceHeapBuffer.contents;
        IRDescriptorTableSetBuffer(&entry[slotIndex], buffer.gpuAddress + offset, 0);
        return allocation;
    }

    void freeBuffer(uint64_t handle) {
        resourceAllocator.free(handle);
    }

    agfxMetalBindlessAllocation writeAccelerationStructure(agfxAccelerationStructure* accelerationStructure) {
        uint64_t slotIndex = resourceAllocator.allocate();
        if (slotIndex == UINT64_MAX) {
            return {};
        }

        agfxMetalBindlessAllocation allocation;
        allocation.handle = slotIndex;
        allocation.resourceID = {};

        IRDescriptorTableEntry* entry = (IRDescriptorTableEntry*)resourceHeapBuffer.contents;
        entry[slotIndex].gpuVA = accelerationStructure->resourceIDBuffer.gpuAddress;
        return allocation;
    }

    void freeAccelerationStructure(uint64_t handle) {
        resourceAllocator.free(handle);
    }

    id<MTLDevice> device;
    id<MTLResidencySet> residencySet;

    id<MTLBuffer> resourceHeapBuffer;
    agfxSlotAllocator resourceAllocator;

    id<MTLBuffer> samplerHeapBuffer;
    agfxSlotAllocator samplerAllocator;

    id<MTLTextureViewPool> textureViewPool;
    agfxSlotAllocator textureViewAllocator;
};

// Enum conversion
MTLTextureType agfxTextureTypeToMTL(agfxTextureType type);
MTLTextureUsage agfxTextureUsageToMTL(agfxTextureUsage usage);
MTLPixelFormat agfxPixelFormatToMTL(agfxTextureFormat format);
MTLRegion agfxTextureRegionToMTL(const agfxTextureRegion* region);
MTLSamplerMinMagFilter agfxSamplerFilterToMTL(agfxSamplerFilter filter);
MTLSamplerMipFilter agfxSamplerMipFilterToMTL(agfxSamplerFilter filter);
MTLSamplerAddressMode agfxSamplerAddressModeToMTL(agfxSamplerAddressMode mode);
MTLCompareFunction agfxCompareFunctionToMTL(agfxComparisonFunction func);
MTLLoadAction agfxLoadActionToMTL(agfxLoadOp action);
MTLStoreAction agfxStoreActionToMTL(agfxStoreOp action);
MTLCullMode agfxCullModeToMTL(agfxCullMode mode);
MTLWinding agfxWindingToMTL(agfxFrontFace winding);
MTLTriangleFillMode agfxFillModeToMTL(agfxFillMode mode);
MTLBlendFactor agfxBlendFactorToMTL(agfxBlendFactor factor);
MTLBlendOperation agfxBlendOpToMTL(agfxBlendOperation op);
MTLPrimitiveType agfxPrimitiveTypeToMTL(agfxTopology topology);
MTLPrimitiveTopologyClass agfxPrimitiveTopologyClassToMTL(agfxTopology topology);

// Metal4 barriers are global (no per-resource hazard tracking), so this
// agglomerates producer/consumer stages across all pending transitions and
// flushes them as a single consumer barrier on the next encoder.
static void agfxResourceStateToMTLStages(agfxResourceState state, MTLStages* outProducer, MTLStages* outConsumer)
{
    switch (state)
    {
        case AGFX_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER: *outProducer = 0; *outConsumer = MTLStageVertex; break;
        case AGFX_RESOURCE_STATE_INDEX_BUFFER: *outProducer = 0; *outConsumer = MTLStageVertex; break;
        case AGFX_RESOURCE_STATE_RENDER_TARGET: *outProducer = MTLStageFragment; *outConsumer = 0; break;
        case AGFX_RESOURCE_STATE_UNORDERED_ACCESS: *outProducer = MTLStageDispatch; *outConsumer = MTLStageDispatch; break;
        case AGFX_RESOURCE_STATE_DEPTH_WRITE: *outProducer = MTLStageFragment; *outConsumer = 0; break;
        case AGFX_RESOURCE_STATE_DEPTH_READ: *outProducer = 0; *outConsumer = MTLStageFragment; break;
        case AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE: *outProducer = 0; *outConsumer = MTLStageVertex | MTLStageDispatch; break;
        case AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE: *outProducer = 0; *outConsumer = MTLStageFragment; break;
        case AGFX_RESOURCE_STATE_INDIRECT_ARGUMENT: *outProducer = 0; *outConsumer = MTLStageVertex | MTLStageDispatch; break;
        case AGFX_RESOURCE_STATE_COPY_DEST: *outProducer = MTLStageBlit; *outConsumer = 0; break;
        case AGFX_RESOURCE_STATE_COPY_SOURCE: *outProducer = 0; *outConsumer = MTLStageBlit; break;
        case AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE: *outProducer = MTLStageAccelerationStructure; *outConsumer = MTLStageAccelerationStructure; break;
        case AGFX_RESOURCE_STATE_GENERIC_READ:
        case AGFX_RESOURCE_STATE_ALL_SHADER_RESOURCE:
            *outProducer = 0;
            *outConsumer = MTLStageVertex | MTLStageFragment | MTLStageDispatch | MTLStageBlit;
            break;
        case AGFX_RESOURCE_STATE_COMMON:
        case AGFX_RESOURCE_STATE_PRESENT:
        default:
            *outProducer = 0;
            *outConsumer = 0;
            break;
    }
}

struct agfxMetal4BarrierTracker {
    void addBarrier(agfxResourceState oldState, agfxResourceState newState) {
        MTLStages producerUnused, afterStages;
        MTLStages beforeStages, consumerUnused;
        agfxResourceStateToMTLStages(oldState, &afterStages, &consumerUnused);
        agfxResourceStateToMTLStages(newState, &producerUnused, &beforeStages);

        // Both sides must be non-empty: MTL4DebugCommandEncoder rejects
        // barrierAfterQueueStages:beforeStages: if either mask is 0, so an empty producer
        // side (e.g. transitioning out of a read-only or COMMON state, which writes nothing)
        // means there is nothing to synchronize against regardless of the consumer side.
        if (afterStages == 0 || beforeStages == 0) return;

        this->afterStages |= afterStages;
        this->beforeStages |= beforeStages;
        visibility = MTL4VisibilityOptionDevice;
        pending = true;
    }

    void encode(id<MTL4CommandEncoder> encoder) {
        if (!pending) return;

        [encoder barrierAfterQueueStages:afterStages beforeStages:beforeStages visibilityOptions:visibility];

        afterStages = 0;
        beforeStages = 0;
        visibility = MTL4VisibilityOptionNone;
        pending = false;
    }

    MTLStages afterStages = 0;
    MTLStages beforeStages = 0;
    MTL4VisibilityOptions visibility = MTL4VisibilityOptionNone;
    bool pending = false;
};

// Device
struct agfxDevice {
    agfxDeviceCreateInfo createInfo;
    id<MTLDevice> device;
    id<MTLResidencySet> residencySet;
    
    agfxMetalBindlessManager* bindlessManager = nullptr;
};

agfxDevice* agfxDeviceCreate(const agfxDeviceCreateInfo* createInfo) {
    agfxDevice* device = (agfxDevice*)createInfo->allocate(sizeof(agfxDevice));
    memcpy(&device->createInfo, createInfo, sizeof(agfxDeviceCreateInfo));
    device->device = MTLCreateSystemDefaultDevice();

    MTLResidencySetDescriptor* residencySetDesc = [MTLResidencySetDescriptor new];
    residencySetDesc.label = @"ResidencySet";
    device->residencySet = [device->device newResidencySetWithDescriptor:residencySetDesc error:nil];

    void* memory = createInfo->allocate(sizeof(agfxMetalBindlessManager));
    device->bindlessManager = new(memory) agfxMetalBindlessManager(device->device, device->residencySet);
    return device;
}

void agfxDeviceDestroy(agfxDevice* device) {
    device->bindlessManager->~agfxMetalBindlessManager();
    device->createInfo.free(device->bindlessManager);
    device->residencySet = nil;
    device->device = nil;
    device->createInfo.free(device);
}

void agfxDeviceGetInfo(agfxDevice* device, agfxDeviceInfo* info) {
    memcpy(info->name, device->device.name.UTF8String, 256);
    info->supportsRayTracing = device->device.supportsRaytracing;
    info->supportsMeshShaders = device->device.supportsRaytracing; // Hardware RT and MS is both M3+ so if one is supported, the other is too
    info->supportsMultiDrawIndirect = YES;
}

void agfxDeviceMakeResourcesResident(agfxDevice* device) {
    [device->residencySet commit];
}

// Fence
struct agfxFence {
    id<MTLSharedEvent> fence;
};

agfxFence* agfxFenceCreate(agfxDevice* device) {
    agfxFence* fence = (agfxFence*)device->createInfo.allocate(sizeof(agfxFence));
    fence->fence = [device->device newSharedEvent];
    return fence;
}

void agfxFenceDestroy(agfxDevice* device, agfxFence* fence) {
    fence->fence = nil;
    device->createInfo.free(fence);
}

void agfxFenceWait(agfxFence* fence, uint64_t value, uint64_t timeout) {
    [fence->fence waitUntilSignaledValue:value timeoutMS:timeout];
}

void agfxFenceSignal(agfxFence* fence, uint64_t value) {
    fence->fence.signaledValue = value;
}

uint64_t agfxFenceGetCompletedValue(agfxFence* fence) {
    return fence->fence.signaledValue;
}

// Query pool
struct agfxQueryPool {
    id<MTL4CounterHeap> heap;
    uint32_t count;
};

agfxQueryPool* agfxQueryPoolCreate(agfxDevice* device, agfxCommandQueue* queue, const agfxQueryPoolCreateInfo* createInfo) {
    agfxQueryPool* pool = (agfxQueryPool*)device->createInfo.allocate(sizeof(agfxQueryPool));
    pool->count = createInfo->count;

    MTL4CounterHeapDescriptor* desc = [MTL4CounterHeapDescriptor new];
    desc.type = MTL4CounterHeapTypeTimestamp;
    desc.count = createInfo->count;
    pool->heap = [device->device newCounterHeapWithDescriptor:desc error:nil];
    return pool;
}

void agfxQueryPoolDestroy(agfxDevice* device, agfxQueryPool* pool) {
    pool->heap = nil;
    device->createInfo.free(pool);
}

void agfxCommandBufferResolveQueryPool(agfxCommandBuffer* commandBuffer, agfxQueryPool* pool, uint32_t firstIndex, uint32_t count) {
    // MTL4CounterHeap needs no GPU-recorded resolve step; agfxQueryPoolReadback resolves directly from the heap.
}

void agfxQueryPoolReadback(agfxDevice* device, agfxQueryPool* pool, uint32_t firstIndex, uint32_t count, uint64_t* outTimestampsNanoseconds) {
    NSData* data = [pool->heap resolveCounterRange:NSMakeRange(firstIndex, count)];
    const MTL4TimestampHeapEntry* entries = (const MTL4TimestampHeapEntry*)data.bytes;

    static mach_timebase_info_data_t timebase;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{ mach_timebase_info(&timebase); });

    for (uint32_t i = 0; i < count; ++i) {
        outTimestampsNanoseconds[i] = (uint64_t)((entries[i].timestamp * (double)timebase.numer) / (double)timebase.denom);
    }
    [pool->heap invalidateCounterRange:NSMakeRange(firstIndex, count)];
}

// Command queue
struct agfxCommandQueue {
    agfxCommandQueueCreateInfo createInfo;
    id<MTL4CommandQueue> commandQueue;
};

agfxCommandQueue* agfxCommandQueueCreate(agfxDevice* device, const agfxCommandQueueCreateInfo* createInfo) {
    agfxCommandQueue* queue = (agfxCommandQueue*)device->createInfo.allocate(sizeof(agfxCommandQueue));
    memcpy(&queue->createInfo, createInfo, sizeof(agfxCommandQueueCreateInfo));
    queue->commandQueue = [device->device newMTL4CommandQueue];
    [queue->commandQueue addResidencySet:device->residencySet];
    return queue;
}

void agfxCommandQueueDestroy(agfxDevice* device, agfxCommandQueue* queue) {
    queue->commandQueue = nil;
    device->createInfo.free(queue);
}

void agfxCommandQueueSignal(agfxCommandQueue* queue, agfxFence* fence, uint64_t value) {
    [queue->commandQueue signalEvent:fence->fence value:value];
}

void agfxCommandQueueWait(agfxCommandQueue* queue, agfxFence* fence, uint64_t value) {
    [queue->commandQueue waitForEvent:fence->fence value:value];
}

// Command buffer
struct agfxCommandBuffer {
    agfxDevice* device;
    id<MTL4CommandBuffer> commandBuffer;
    id<MTL4CommandAllocator> commandAllocator;
    id<MTL4ArgumentTable> renderArgumentTable;
    id<MTL4ArgumentTable> computeArgumentTable;
    agfxMetal4BarrierTracker barrierTracker;

    agfxMetalLinearAllocator* topLevelArgBufferAllocator;
    agfxMetalLinearAllocator* drawArgumentAllocator;
    agfxMetalLinearAllocator* drawUniformAllocator;
};

agfxCommandBuffer* agfxCommandBufferCreate(agfxDevice* device, agfxCommandQueue* queue) {
    agfxCommandBuffer* commandBuffer = (agfxCommandBuffer*)device->createInfo.allocate(sizeof(agfxCommandBuffer));
    commandBuffer->device = device;
    commandBuffer->commandAllocator = [device->device newCommandAllocator];
    commandBuffer->commandBuffer = [device->device newCommandBuffer];

    MTL4ArgumentTableDescriptor* renderArgTableDesc = [MTL4ArgumentTableDescriptor new];
    renderArgTableDesc.label = @"RenderArgumentTable";
    renderArgTableDesc.maxBufferBindCount = 31;
    renderArgTableDesc.maxTextureBindCount = 31;
    
    commandBuffer->renderArgumentTable = [device->device newArgumentTableWithDescriptor:renderArgTableDesc error:nil];
    [commandBuffer->renderArgumentTable setAddress:device->bindlessManager->resourceHeapBuffer.gpuAddress atIndex:kIRDescriptorHeapBindPoint];
    [commandBuffer->renderArgumentTable setAddress:device->bindlessManager->samplerHeapBuffer.gpuAddress atIndex:kIRSamplerHeapBindPoint];

    MTL4ArgumentTableDescriptor* computeArgTableDesc = [MTL4ArgumentTableDescriptor new];
    computeArgTableDesc.label = @"ComputeArgumentTable";
    computeArgTableDesc.maxBufferBindCount = 31;
    computeArgTableDesc.maxTextureBindCount = 31;

    commandBuffer->computeArgumentTable = [device->device newArgumentTableWithDescriptor:computeArgTableDesc error:nil];
    [commandBuffer->computeArgumentTable setAddress:device->bindlessManager->resourceHeapBuffer.gpuAddress atIndex:kIRDescriptorHeapBindPoint];
    [commandBuffer->computeArgumentTable setAddress:device->bindlessManager->samplerHeapBuffer.gpuAddress atIndex:kIRSamplerHeapBindPoint];

    void* topLevelArgBufferMemory = device->createInfo.allocate(sizeof(agfxMetalLinearAllocator));
    void* drawArgumentMemory = device->createInfo.allocate(sizeof(agfxMetalLinearAllocator));
    void* drawUniformMemory = device->createInfo.allocate(sizeof(agfxMetalLinearAllocator));

    commandBuffer->topLevelArgBufferAllocator = new(topLevelArgBufferMemory) agfxMetalLinearAllocator(device->device, device->residencySet, sizeof(agfxTLAB) * 1024);
    commandBuffer->drawArgumentAllocator = new(drawArgumentMemory) agfxMetalLinearAllocator(device->device, device->residencySet, (sizeof(uint) * 5) * 4096);
    commandBuffer->drawUniformAllocator = new(drawUniformMemory) agfxMetalLinearAllocator(device->device, device->residencySet, sizeof(agfxTLAB) * 8192);

    return commandBuffer;
}

void agfxCommandBufferDestroy(agfxDevice* device, agfxCommandBuffer* commandBuffer) {
    commandBuffer->topLevelArgBufferAllocator->~agfxMetalLinearAllocator();
    device->createInfo.free(commandBuffer->topLevelArgBufferAllocator);
    commandBuffer->drawArgumentAllocator->~agfxMetalLinearAllocator();
    device->createInfo.free(commandBuffer->drawArgumentAllocator);
    commandBuffer->drawUniformAllocator->~agfxMetalLinearAllocator();
    device->createInfo.free(commandBuffer->drawUniformAllocator);
    commandBuffer->renderArgumentTable = nil;
    commandBuffer->computeArgumentTable = nil;
    commandBuffer->commandBuffer = nil;
    commandBuffer->commandAllocator = nil;
    device->createInfo.free(commandBuffer);
}

void agfxCommandBufferReset(agfxCommandBuffer* commandBuffer) {
    [commandBuffer->commandAllocator reset];
    commandBuffer->topLevelArgBufferAllocator->reset();
    commandBuffer->drawArgumentAllocator->reset();
    commandBuffer->drawUniformAllocator->reset();
}

void agfxCommandBufferBegin(agfxCommandBuffer* commandBuffer) {
    [commandBuffer->commandBuffer beginCommandBufferWithAllocator:commandBuffer->commandAllocator];
}

void agfxCommandBufferEnd(agfxCommandBuffer* commandBuffer) {
    [commandBuffer->commandBuffer endCommandBuffer];
}

void agfxCommandBufferWriteTimestamp(agfxCommandBuffer* commandBuffer, agfxQueryPool* pool, uint32_t index) {
    [commandBuffer->commandBuffer writeTimestampIntoHeap:pool->heap atIndex:index];
}

void agfxCommandQueueSubmit(agfxCommandQueue* queue, agfxCommandBuffer** commandBuffers, uint32_t commandBufferCount) {
    id<MTL4CommandBuffer> mtlCommandBuffers[16];
    for (uint32_t i = 0; i < commandBufferCount; ++i) {
        mtlCommandBuffers[i] = commandBuffers[i]->commandBuffer;
    }
    [queue->commandQueue commit:mtlCommandBuffers count:commandBufferCount];
}

void agfxCommandBufferTextureBarrier(agfxCommandBuffer* commandBuffer, agfxTexture* texture,  agfxResourceState oldState, agfxResourceState newState, uint32_t mip, uint32_t layer, agfxBool agglomerate) {
    if (!agglomerate) return;
    commandBuffer->barrierTracker.addBarrier(oldState, newState);
}

void agfxCommandBufferBufferBarrier(agfxCommandBuffer* commandBuffer, agfxBuffer* buffer, agfxResourceState oldState, agfxResourceState newState, agfxBool agglomerate) {
    if (!agglomerate) return;
    commandBuffer->barrierTracker.addBarrier(oldState, newState);
}

void agfxCommandBufferAccelerationStructureBarrier(agfxCommandBuffer* commandBuffer, agfxAccelerationStructure* accelerationStructure, agfxResourceState oldState, agfxResourceState newState, agfxBool agglomerate) {
    if (!agglomerate) return;
    commandBuffer->barrierTracker.addBarrier(oldState, newState);
}

// Texture
struct agfxTexture {
    agfxTextureCreateInfo createInfo;
    id<MTLTexture> texture;
};

agfxTexture* agfxTextureCreate(agfxDevice* device, const agfxTextureCreateInfo* createInfo) {
    agfxTexture* texture = (agfxTexture*)device->createInfo.allocate(sizeof(agfxTexture));
    memcpy(&texture->createInfo, createInfo, sizeof(agfxTextureCreateInfo));

    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
    descriptor.textureType = agfxTextureTypeToMTL(createInfo->type);
    descriptor.pixelFormat = agfxPixelFormatToMTL(createInfo->format);
    descriptor.usage = agfxTextureUsageToMTL(createInfo->usage);
    descriptor.width = createInfo->width;
    descriptor.height = createInfo->height;
    descriptor.depth = descriptor.textureType == MTLTextureType3D ? createInfo->depthOrArrayLayers : 1;
    descriptor.arrayLength = descriptor.textureType == MTLTextureType2DArray ? createInfo->depthOrArrayLayers : 1;
    descriptor.mipmapLevelCount = createInfo->mipLevels;
    descriptor.resourceOptions = MTLResourceStorageModeShared;

    texture->texture = [device->device newTextureWithDescriptor:descriptor];
    [device->residencySet addAllocation:texture->texture];

    return texture;
}

void agfxTextureDestroy(agfxDevice* device, agfxTexture* texture) {
    [device->residencySet removeAllocation:texture->texture];
    texture->texture = nil;
    device->createInfo.free(texture);
}

void agfxTextureGetInfo(agfxTexture* texture, agfxTextureCreateInfo* info) {
    memcpy(info, &texture->createInfo, sizeof(agfxTextureCreateInfo));
}

void agfxTextureReplaceRegion(agfxDevice* device, agfxTexture* texture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, const void* data, uint32_t dataSize, uint32_t bytesPerRow, uint32_t bytesPerImage) {
    MTLRegion mtlRegion = agfxTextureRegionToMTL(region);
    [texture->texture replaceRegion:mtlRegion mipmapLevel:mipLevel slice:layer withBytes:data bytesPerRow:bytesPerRow bytesPerImage:bytesPerImage];
}

void agfxTextureSetName(agfxTexture* texture, const char* name) {
    texture->texture.label = [NSString stringWithUTF8String:name];
}

// Compute pass
struct agfxComputePass {
    id<MTL4ComputeCommandEncoder> encoder;
    agfxDevice* device;

    agfxComputePipeline* currentPipeline = nullptr;
    agfxCommandBuffer* commandBuffer = nullptr;
};

agfxComputePass* agfxComputePassBegin(agfxCommandBuffer* commandBuffer, const char* name) {
    agfxComputePass* computePass = (agfxComputePass*)commandBuffer->device->createInfo.tempAllocate(sizeof(agfxComputePass));
    computePass->encoder = [commandBuffer->commandBuffer computeCommandEncoder];
    computePass->encoder.label = [NSString stringWithUTF8String:name];
    computePass->device = commandBuffer->device;
    computePass->commandBuffer = commandBuffer;
    commandBuffer->barrierTracker.encode(computePass->encoder);

    [computePass->encoder setArgumentTable:commandBuffer->computeArgumentTable];

    return computePass;
}

void agfxComputePassEnd(agfxComputePass* computePass) {
    [computePass->encoder endEncoding];
    computePass->encoder = nil;
    computePass->device->createInfo.tempFree(computePass);
}

void agfxComputePassTextureUAVBarrier(agfxComputePass* computePass, agfxTexture* texture) {
    [computePass->encoder barrierAfterQueueStages:MTLStageDispatch beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
}

void agfxComputePassBufferUAVBarrier(agfxComputePass* computePass, agfxBuffer* buffer) {
    [computePass->encoder barrierAfterQueueStages:MTLStageDispatch beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
}

void agfxComputePassCopyTextureToBuffer(agfxComputePass* computePass, agfxTexture* texture, agfxBuffer* buffer, uint64_t bufferOffset, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, uint32_t bytesPerRow, uint32_t bytesPerImage) {
    [computePass->encoder copyFromTexture:texture->texture sourceSlice:layer sourceLevel:mipLevel sourceOrigin:agfxTextureRegionToMTL(region).origin sourceSize:agfxTextureRegionToMTL(region).size toBuffer:buffer->buffer destinationOffset:bufferOffset destinationBytesPerRow:bytesPerRow destinationBytesPerImage:bytesPerImage];
}

void agfxComputePassCopyBufferToTexture(agfxComputePass* computePass, agfxBuffer* buffer, agfxTexture* texture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, uint32_t bytesPerRow, uint32_t bytesPerImage) {
    [computePass->encoder copyFromBuffer:buffer->buffer sourceOffset:0 sourceBytesPerRow:bytesPerRow sourceBytesPerImage:bytesPerImage sourceSize:agfxTextureRegionToMTL(region).size toTexture:texture->texture destinationSlice:layer destinationLevel:mipLevel destinationOrigin:agfxTextureRegionToMTL(region).origin];
}

void agfxComputePassCopyBufferToBuffer(agfxComputePass* computePass, agfxBuffer* srcBuffer, agfxBuffer* dstBuffer, uint64_t srcOffset, uint64_t dstOffset, uint64_t size) {
    [computePass->encoder copyFromBuffer:srcBuffer->buffer sourceOffset:srcOffset toBuffer:dstBuffer->buffer destinationOffset:dstOffset size:size];
}

void agfxComputePassCopyTextureToTexture(agfxComputePass* computePass, agfxTexture* srcTexture, agfxTexture* dstTexture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer) {
    [computePass->encoder copyFromTexture:srcTexture->texture sourceSlice:layer sourceLevel:mipLevel sourceOrigin:agfxTextureRegionToMTL(region).origin sourceSize:agfxTextureRegionToMTL(region).size toTexture:dstTexture->texture destinationSlice:layer destinationLevel:mipLevel destinationOrigin:agfxTextureRegionToMTL(region).origin];
}

void agfxComputePassSetPipeline(agfxComputePass* computePass, agfxComputePipeline* pipeline) {
    computePass->currentPipeline = pipeline;
    [computePass->encoder setComputePipelineState:pipeline->pipelineState];
}

void agfxComputePassPushConstants(agfxComputePass* computePass, const void* data, uint32_t size) {
    agfxMetalAllocation allocation = computePass->commandBuffer->drawUniformAllocator->allocate(size);
    memcpy(allocation.cpuPointer, data, size);

    [computePass->commandBuffer->computeArgumentTable setAddress:allocation.gpuAddress atIndex:kIRArgumentBufferBindPoint];
}

void agfxComputePassDispatch(agfxComputePass* computePass, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    [computePass->encoder dispatchThreadgroups:MTLSizeMake(groupCountX, groupCountY, groupCountZ)
                        threadsPerThreadgroup:MTLSizeMake(computePass->currentPipeline->createInfo.groupSizeX, computePass->currentPipeline->createInfo.groupSizeY, computePass->currentPipeline->createInfo.groupSizeZ)];
}

void agfxComputePassBuildAccelerationStructure(agfxComputePass* computePass, agfxAccelerationStructure* accelerationStructure, agfxBuffer* scratchBuffer, uint64_t scratchBufferOffset) {
    if (accelerationStructure->createInfo.type == AGFX_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL) {
        [computePass->encoder buildAccelerationStructure:accelerationStructure->accelerationStructure
                                descriptor:accelerationStructure->mtlTopLevelDescriptor
                                scratchBuffer:MTL4BufferRangeMake(scratchBuffer->buffer.gpuAddress + scratchBufferOffset, scratchBuffer->createInfo.size)];
    } else {
        [computePass->encoder buildAccelerationStructure:accelerationStructure->accelerationStructure
                                descriptor:accelerationStructure->mtlBottomLevelDescriptor
                                scratchBuffer:MTL4BufferRangeMake(scratchBuffer->buffer.gpuAddress + scratchBufferOffset, scratchBuffer->createInfo.size)];
    }
}

void agfxComputePassUpdateAccelerationStructure(agfxComputePass* computePass, agfxAccelerationStructure* srcAccelerationStructure, agfxAccelerationStructure* dstAccelerationStructure, agfxBuffer* scratchBuffer, uint64_t scratchBufferOffset) {
    if (!dstAccelerationStructure->createInfo.allowUpdate) return;

    if (dstAccelerationStructure->createInfo.type == AGFX_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL) {
        [computePass->encoder refitAccelerationStructure:srcAccelerationStructure->accelerationStructure
                                descriptor:srcAccelerationStructure->mtlTopLevelDescriptor
                                destination:dstAccelerationStructure->accelerationStructure
                                scratchBuffer:MTL4BufferRangeMake(scratchBuffer->buffer.gpuAddress + scratchBufferOffset, scratchBuffer->createInfo.size)];
    } else {
        [computePass->encoder refitAccelerationStructure:srcAccelerationStructure->accelerationStructure
                                descriptor:srcAccelerationStructure->mtlBottomLevelDescriptor
                                destination:dstAccelerationStructure->accelerationStructure
                                scratchBuffer:MTL4BufferRangeMake(scratchBuffer->buffer.gpuAddress + scratchBufferOffset, scratchBuffer->createInfo.size)];
    }
}

void agfxComputePassCopyAccelerationStructure(agfxComputePass* computePass, agfxAccelerationStructure* srcAccelerationStructure, agfxAccelerationStructure* dstAccelerationStructure) {
    [computePass->encoder copyAccelerationStructure:srcAccelerationStructure->accelerationStructure
                                toAccelerationStructure:dstAccelerationStructure->accelerationStructure];
}

void agfxComputePassWriteCompactedSizeToBuffer(agfxComputePass* computePass, agfxAccelerationStructure* accelerationStructure, agfxBuffer* dstBuffer, uint64_t dstBufferOffset) {
    [computePass->encoder writeCompactedAccelerationStructureSize:accelerationStructure->accelerationStructure
                                toBuffer:MTL4BufferRangeMake(dstBuffer->buffer.gpuAddress + dstBufferOffset, sizeof(uint64_t))];
}

void agfxComputePassCompactAccelerationStructure(agfxComputePass* computePass, agfxAccelerationStructure* srcAccelerationStructure, agfxAccelerationStructure* dstAccelerationStructure) {
    [computePass->encoder copyAndCompactAccelerationStructure:srcAccelerationStructure->accelerationStructure
                                toAccelerationStructure:dstAccelerationStructure->accelerationStructure];
}

// Buffer

agfxBuffer* agfxBufferCreate(agfxDevice* device, const agfxBufferCreateInfo* createInfo) {
    agfxBuffer* buffer = (agfxBuffer*)device->createInfo.allocate(sizeof(agfxBuffer));
    memcpy(&buffer->createInfo, createInfo, sizeof(agfxBufferCreateInfo));
    buffer->buffer = [device->device newBufferWithLength:createInfo->size options:MTLResourceStorageModeShared];
    [device->residencySet addAllocation:buffer->buffer];
    return buffer;
}

void agfxBufferDestroy(agfxDevice* device, agfxBuffer* buffer) {
    [device->residencySet removeAllocation:buffer->buffer];
    buffer->buffer = nil;
    device->createInfo.free(buffer);
}

void* agfxBufferMap(agfxBuffer* buffer) {
    return buffer->buffer.contents;
}

void agfxBufferUnmap(agfxBuffer* buffer) {}

void agfxBufferSetName(agfxBuffer* buffer, const char* name) {
    buffer->buffer.label = [NSString stringWithUTF8String:name];
}

void agfxBufferGetInfo(agfxBuffer* buffer, agfxBufferCreateInfo* info) {
    memcpy(info, &buffer->createInfo, sizeof(agfxBufferCreateInfo));
}

// Texture view
struct agfxTextureView {
    agfxTextureViewCreateInfo createInfo;
    agfxMetalBindlessManager* bindlessManager;
    agfxMetalBindlessAllocation allocation;
};

agfxTextureView* agfxTextureViewCreate(agfxDevice* device, const agfxTextureViewCreateInfo* createInfo) {
    agfxTextureView* textureView = (agfxTextureView*)device->createInfo.allocate(sizeof(agfxTextureView));
    memcpy(&textureView->createInfo, createInfo, sizeof(agfxTextureViewCreateInfo));
    textureView->bindlessManager = device->bindlessManager;

    MTLTextureViewDescriptor* descriptor = [MTLTextureViewDescriptor new];
    descriptor.textureType = agfxTextureTypeToMTL(createInfo->type);
    descriptor.pixelFormat = agfxPixelFormatToMTL(createInfo->format);
    descriptor.levelRange = NSMakeRange(createInfo->baseMipLevel, createInfo->mipLevelCount);
    descriptor.sliceRange = NSMakeRange(createInfo->baseArrayLayer, createInfo->arrayLayerCount);

    textureView->allocation = device->bindlessManager->writeTextureView(descriptor, createInfo->texture->texture);
    return textureView;
}

void agfxTextureViewDestroy(agfxDevice* device, agfxTextureView* textureView) {
    device->bindlessManager->freeTextureView(textureView->allocation.handle);
    device->createInfo.free(textureView);
}

uint64_t agfxTextureViewGetHandle(agfxTextureView* textureView) {
    return textureView->allocation.handle;
}

// Sampler
struct agfxSampler {
    agfxSamplerCreateInfo createInfo;
    agfxMetalBindlessManager* bindlessManager;
    agfxMetalBindlessAllocation allocation;

    id<MTLSamplerState> samplerState;
};

agfxSampler* agfxSamplerCreate(agfxDevice* device, const agfxSamplerCreateInfo* createInfo) {
    agfxSampler* sampler = (agfxSampler*)device->createInfo.allocate(sizeof(agfxSampler));
    memcpy(&sampler->createInfo, createInfo, sizeof(agfxSamplerCreateInfo));
    sampler->bindlessManager = device->bindlessManager;

    MTLSamplerDescriptor* descriptor = [MTLSamplerDescriptor new];
    descriptor.minFilter = agfxSamplerFilterToMTL(createInfo->filter);
    descriptor.magFilter = agfxSamplerFilterToMTL(createInfo->filter);
    descriptor.mipFilter = agfxSamplerMipFilterToMTL(createInfo->filter);
    descriptor.sAddressMode = agfxSamplerAddressModeToMTL(createInfo->addressModeU);
    descriptor.tAddressMode = agfxSamplerAddressModeToMTL(createInfo->addressModeV);
    descriptor.rAddressMode = agfxSamplerAddressModeToMTL(createInfo->addressModeW);
    descriptor.lodMinClamp = createInfo->minLod;
    descriptor.lodMaxClamp = createInfo->maxLod;
    descriptor.compareFunction = agfxCompareFunctionToMTL(createInfo->comparisonFunction);
    descriptor.borderColor = MTLSamplerBorderColorTransparentBlack;
    descriptor.supportArgumentBuffers = YES;

    sampler->samplerState = [device->device newSamplerStateWithDescriptor:descriptor];
    sampler->allocation = device->bindlessManager->writeSampler(sampler->samplerState, createInfo->lodBias);

    return sampler;
}

void agfxSamplerDestroy(agfxDevice* device, agfxSampler* sampler) {
    device->bindlessManager->freeSampler(sampler->allocation.handle);
    device->createInfo.free(sampler);
}

uint64_t agfxSamplerGetHandle(agfxSampler* sampler) {
    return sampler->allocation.handle;
}

// Buffer view
struct agfxBufferView {
    agfxBufferViewCreateInfo createInfo;
    agfxMetalBindlessManager* bindlessManager;
    agfxMetalBindlessAllocation allocation;
};

agfxBufferView* agfxBufferViewCreate(agfxDevice* device, const agfxBufferViewCreateInfo* createInfo) {
    agfxBufferView* bufferView = (agfxBufferView*)device->createInfo.allocate(sizeof(agfxBufferView));
    memcpy(&bufferView->createInfo, createInfo, sizeof(agfxBufferViewCreateInfo));
    bufferView->bindlessManager = device->bindlessManager;
    bufferView->allocation = device->bindlessManager->writeBuffer(createInfo->buffer->buffer, createInfo->offset);
    return bufferView;
}

void agfxBufferViewDestroy(agfxDevice* device, agfxBufferView* bufferView) {
    bufferView->bindlessManager->freeBuffer(bufferView->allocation.handle);
    device->createInfo.free(bufferView);
}

uint64_t agfxBufferViewGetHandle(agfxBufferView* bufferView) {
    return bufferView->allocation.handle;
}

// Render target
struct agfxRenderTarget {
    agfxRenderTargetCreateInfo createInfo;
    id<MTLTexture> view;
    bool isOwned;
};

agfxRenderTarget* agfxRenderTargetCreate(agfxDevice* device, const agfxRenderTargetCreateInfo* createInfo) {
    agfxRenderTarget* renderTarget = (agfxRenderTarget*)device->createInfo.allocate(sizeof(agfxRenderTarget));
    memcpy(&renderTarget->createInfo, createInfo, sizeof(agfxRenderTargetCreateInfo));

    id<MTLTexture> base = createInfo->texture->texture;
    bool isDefaultView = (createInfo->mipLevel == 0
                       && createInfo->arrayLayer == 0
                       && createInfo->format == AGFX_TEXTURE_FORMAT_UNKNOWN);
    if (isDefaultView) {
        renderTarget->view = base;
        renderTarget->isOwned = false;
    } else {
        MTLPixelFormat pixelFormat = (createInfo->format == AGFX_TEXTURE_FORMAT_UNKNOWN) ? agfxPixelFormatToMTL(createInfo->texture->createInfo.format) : agfxPixelFormatToMTL(createInfo->format);
        renderTarget->view = [base newTextureViewWithPixelFormat:pixelFormat
                                    textureType:MTLTextureType2D
                                    levels:NSMakeRange(createInfo->mipLevel, 1)
                                    slices:NSMakeRange(createInfo->arrayLayer, 1)];
        renderTarget->isOwned = true;
    }
    return renderTarget;
}

void agfxRenderTargetDestroy(agfxDevice* device, agfxRenderTarget* renderTarget) {
    if (renderTarget->isOwned) {
        renderTarget->view = nil;
    }
    device->createInfo.free(renderTarget);
}

// Render pass
struct agfxRenderPass {
    id<MTL4RenderCommandEncoder> encoder;
    agfxDevice* device;

    agfxCommandBuffer* cmdBuffer;
    agfxRenderPipeline* currentPipeline;
};

agfxRenderPass* agfxRenderPassBegin(agfxCommandBuffer* cmdBuffer, const agfxRenderPassCreateInfo* createInfo) {
    agfxRenderPass* renderPass = (agfxRenderPass*)cmdBuffer->device->createInfo.tempAllocate(sizeof(agfxRenderPass));
    renderPass->device = cmdBuffer->device;
    renderPass->cmdBuffer = cmdBuffer;
    
    MTL4RenderPassDescriptor* descriptor = [MTL4RenderPassDescriptor new];
    descriptor.renderTargetWidth = createInfo->width;
    descriptor.renderTargetHeight = createInfo->height;
    for (uint32_t i = 0; i < createInfo->colorAttachmentCount; ++i) {
        descriptor.colorAttachments[i].clearColor = MTLClearColorMake(createInfo->colorAttachments[i].clearColor[0], createInfo->colorAttachments[i].clearColor[1],
                                                                     createInfo->colorAttachments[i].clearColor[2], createInfo->colorAttachments[i].clearColor[3]);
        descriptor.colorAttachments[i].loadAction = agfxLoadActionToMTL(createInfo->colorAttachments[i].loadOp);
        descriptor.colorAttachments[i].storeAction = agfxStoreActionToMTL(createInfo->colorAttachments[i].storeOp);
        descriptor.colorAttachments[i].texture = createInfo->colorAttachments[i].renderTarget->view;
    }
    if (createInfo->hasDepthAttachment) {
        descriptor.depthAttachment.clearDepth = 1.0f;
        descriptor.depthAttachment.loadAction = agfxLoadActionToMTL(createInfo->depthAttachment.loadOp);
        descriptor.depthAttachment.storeAction = agfxStoreActionToMTL(createInfo->depthAttachment.storeOp);
        descriptor.depthAttachment.texture = createInfo->depthAttachment.renderTarget->view;
    }

    renderPass->encoder = [cmdBuffer->commandBuffer renderCommandEncoderWithDescriptor:descriptor];
    renderPass->encoder.label = [NSString stringWithUTF8String:createInfo->name];

    [renderPass->encoder setArgumentTable:cmdBuffer->renderArgumentTable atStages:MTLStageVertex | MTLStageFragment | MTLStageObject | MTLStageMesh];

    cmdBuffer->barrierTracker.encode(renderPass->encoder);
    return renderPass;
}

void agfxRenderPassEnd(agfxRenderPass* renderPass) {
    [renderPass->encoder endEncoding];
    renderPass->encoder = nil;
    renderPass->device->createInfo.tempFree(renderPass);
}

void agfxRenderPassSetViewport(agfxRenderPass* renderPass, float x, float y, float width, float height, float minDepth, float maxDepth) {
    MTLViewport viewport = {x, y, width, height, minDepth, maxDepth};
    [renderPass->encoder setViewport:viewport];
}

void agfxRenderPassSetScissor(agfxRenderPass* renderPass, uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
    MTLScissorRect scissor = {x, y, width, height};
    [renderPass->encoder setScissorRect:scissor];
}

void agfxRenderPassSetPipeline(agfxRenderPass* renderPass, agfxRenderPipeline* pipeline) {
    renderPass->currentPipeline = pipeline;

    [renderPass->encoder setRenderPipelineState:pipeline->pipelineState];
    if (pipeline->depthStencilState) {
        [renderPass->encoder setDepthStencilState:pipeline->depthStencilState];
        [renderPass->encoder setDepthClipMode:pipeline->createInfo.depthClampEnable ? MTLDepthClipModeClamp : MTLDepthClipModeClip];
    }
    [renderPass->encoder setFrontFacingWinding:agfxWindingToMTL(pipeline->createInfo.frontFace)];
    [renderPass->encoder setCullMode:agfxCullModeToMTL(pipeline->createInfo.cullMode)];
    [renderPass->encoder setTriangleFillMode:agfxFillModeToMTL(pipeline->createInfo.fillMode)];
}

void agfxRenderPassPushConstants(agfxRenderPass* renderPass, const void* data, uint32_t size) {
    agfxMetalAllocation allocation = renderPass->cmdBuffer->drawUniformAllocator->allocate(sizeof(agfxTLAB));
    memcpy(allocation.cpuPointer, data, size);

    [renderPass->cmdBuffer->renderArgumentTable setAddress:allocation.gpuAddress atIndex:kIRArgumentBufferBindPoint];
}

void agfxRenderPassDraw(agfxRenderPass* renderPass, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
    IRRuntimeDrawArgument drawArg = { vertexCount, instanceCount, firstVertex, firstInstance };
    IRRuntimeDrawParams drawParams = { .draw = drawArg };

    agfxMetalAllocation allocation = renderPass->cmdBuffer->drawArgumentAllocator->allocate(sizeof(IRRuntimeDrawParams));
    memcpy(allocation.cpuPointer, &drawParams, sizeof(IRRuntimeDrawParams));

    agfxMetalAllocation uniformAlloc = renderPass->cmdBuffer->drawUniformAllocator->allocate(sizeof(uint16_t));
    memcpy(uniformAlloc.cpuPointer, &kIRNonIndexedDraw, sizeof(uint16_t));

    [renderPass->cmdBuffer->renderArgumentTable setAddress:allocation.gpuAddress atIndex:kIRArgumentBufferDrawArgumentsBindPoint];
    [renderPass->cmdBuffer->renderArgumentTable setAddress:uniformAlloc.gpuAddress atIndex:kIRArgumentBufferUniformsBindPoint];

    MTLPrimitiveType type = agfxPrimitiveTypeToMTL(renderPass->currentPipeline->createInfo.topology);
    [renderPass->encoder drawPrimitives:type vertexStart:firstVertex vertexCount:vertexCount instanceCount:instanceCount baseInstance:firstInstance];
}

void agfxRenderPassDrawIndexed(agfxRenderPass* renderPass, agfxBuffer* indexBuffer, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, uint32_t vertexOffset, uint32_t firstInstance) {
    MTLIndexType indexType = indexBuffer->createInfo.stride == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;

    IRRuntimeDrawIndexedArgument drawIndexedArg = {
        indexCount, instanceCount, firstIndex, (int)vertexOffset, firstInstance
    };
    IRRuntimeDrawParams drawParams = { .drawIndexed = drawIndexedArg };

    agfxMetalAllocation allocation = renderPass->cmdBuffer->drawArgumentAllocator->allocate(sizeof(IRRuntimeDrawParams));
    memcpy(allocation.cpuPointer, &drawParams, sizeof(IRRuntimeDrawParams));

    uint16_t irIndexType = (uint16_t)(indexType + 1);
    agfxMetalAllocation uniformAlloc = renderPass->cmdBuffer->drawUniformAllocator->allocate(sizeof(uint16_t));
    memcpy(uniformAlloc.cpuPointer, &irIndexType, sizeof(uint16_t));

    [renderPass->cmdBuffer->renderArgumentTable setAddress:allocation.gpuAddress atIndex:kIRArgumentBufferDrawArgumentsBindPoint];
    [renderPass->cmdBuffer->renderArgumentTable setAddress:uniformAlloc.gpuAddress atIndex:kIRArgumentBufferUniformsBindPoint];

    MTLPrimitiveType type = agfxPrimitiveTypeToMTL(renderPass->currentPipeline->createInfo.topology);
    [renderPass->encoder drawIndexedPrimitives:type
                                indexCount:indexCount
                                 indexType:indexType
                               indexBuffer:indexBuffer->buffer.gpuAddress + (firstIndex * indexBuffer->createInfo.stride)
                         indexBufferLength:indexCount * indexBuffer->createInfo.stride
                             instanceCount:instanceCount
                                baseVertex:vertexOffset
                              baseInstance:firstInstance];
}

void agfxRenderPassDrawMesh(agfxRenderPass* renderPass, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    MTLSize threadgroupsPerGrid = { groupCountX, groupCountY, groupCountZ };
    MTLSize meshThreadgroupSize = { renderPass->currentPipeline->createInfo.meshGroupSizeX, renderPass->currentPipeline->createInfo.meshGroupSizeY, renderPass->currentPipeline->createInfo.meshGroupSizeZ };
    MTLSize taskThreadgroupSize = { renderPass->currentPipeline->createInfo.taskGroupSizeX, renderPass->currentPipeline->createInfo.taskGroupSizeY, renderPass->currentPipeline->createInfo.taskGroupSizeZ };
    
    [renderPass->encoder drawMeshThreadgroups:threadgroupsPerGrid
            threadsPerObjectThreadgroup:taskThreadgroupSize
            threadsPerMeshThreadgroup:meshThreadgroupSize];
}

// Swapchain
agfxSwapChain* agfxSwapChainCreate(agfxDevice* device, const agfxSwapChainCreateInfo* createInfo) {
    agfxSwapChain* swapChain = (agfxSwapChain*)device->createInfo.allocate(sizeof(agfxSwapChain));
    memcpy(&swapChain->createInfo, createInfo, sizeof(agfxSwapChainCreateInfo));
    swapChain->device = device;
    swapChain->currentDrawable = nil;

    swapChain->metalLayer = (__bridge CAMetalLayer*)createInfo->handle;
    swapChain->metalLayer.device = device->device;
    swapChain->metalLayer.pixelFormat = createInfo->isHDR ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
    swapChain->metalLayer.framebufferOnly = YES;
    swapChain->metalLayer.opaque = YES;
    swapChain->metalLayer.colorspace = createInfo->isHDR ? CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB) : CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    swapChain->metalLayer.maximumDrawableCount = createInfo->imageCount;
    swapChain->metalLayer.drawableSize = CGSizeMake(createInfo->width, createInfo->height);
    swapChain->metalLayer.displaySyncEnabled = createInfo->vsync;
    [createInfo->queue->commandQueue addResidencySet:swapChain->metalLayer.residencySet];

    swapChain->wrapperTexture = (agfxTexture*)device->createInfo.allocate(sizeof(agfxTexture));
    swapChain->wrapperTexture->texture = nil;
    swapChain->wrapperTexture->createInfo.type = AGFX_TEXTURE_TYPE_2D;
    swapChain->wrapperTexture->createInfo.format = createInfo->isHDR ? AGFX_TEXTURE_FORMAT_RGBA16F : AGFX_TEXTURE_FORMAT_BGRA8_UNORM;
    swapChain->wrapperTexture->createInfo.usage = AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT;
    swapChain->wrapperTexture->createInfo.width = createInfo->width;
    swapChain->wrapperTexture->createInfo.height = createInfo->height;
    swapChain->wrapperTexture->createInfo.depthOrArrayLayers = 1;
    swapChain->wrapperTexture->createInfo.mipLevels = 1;

    return swapChain;
}

void agfxSwapChainDestroy(agfxDevice* device, agfxSwapChain* swapChain) {
    device->createInfo.free(swapChain->wrapperTexture);
    device->createInfo.free(swapChain);
}

void agfxSwapChainResize(agfxDevice* device, agfxSwapChain* swapChain, uint32_t width, uint32_t height) {
    swapChain->metalLayer.drawableSize = CGSizeMake(width, height);
    swapChain->wrapperTexture->createInfo.width = width;
    swapChain->wrapperTexture->createInfo.height = height;
}

agfxTextureFormat agfxSwapChainGetFormat(agfxSwapChain* swapChain) {
    return swapChain->createInfo.isHDR ? AGFX_TEXTURE_FORMAT_RGBA16F : AGFX_TEXTURE_FORMAT_BGRA8_UNORM;
}

agfxTexture* agfxSwapChainAcquireNextTexture(agfxSwapChain* swapChain) {
    swapChain->currentDrawable = [swapChain->metalLayer nextDrawable];

    swapChain->wrapperTexture->texture = swapChain->currentDrawable.texture;
    swapChain->wrapperTexture->createInfo.width = (uint32_t)swapChain->currentDrawable.texture.width;
    swapChain->wrapperTexture->createInfo.height = (uint32_t)swapChain->currentDrawable.texture.height;

    [swapChain->createInfo.queue->commandQueue waitForDrawable:swapChain->currentDrawable];

    return swapChain->wrapperTexture;
}

void agfxSwapChainPresent(agfxSwapChain* swapChain) {
    [swapChain->createInfo.queue->commandQueue signalDrawable:swapChain->currentDrawable];
    [swapChain->currentDrawable present];
    swapChain->currentDrawable = nil;
}

// Shader module
struct agfxShaderModule {
    agfxShaderModuleCreateInfo createInfo;
    id<MTLLibrary> library;
    id<MTLFunction> function;
};

agfxShaderModule* agfxShaderModuleCreate(agfxDevice* device, const agfxShaderModuleCreateInfo* createInfo) {
    agfxShaderModule* shaderModule = (agfxShaderModule*)device->createInfo.allocate(sizeof(agfxShaderModule));
    memcpy(&shaderModule->createInfo, createInfo, sizeof(agfxShaderModuleCreateInfo));

    dispatch_data_t data = dispatch_data_create(createInfo->code, createInfo->codeSize, dispatch_get_main_queue(), nullptr);
    id<MTLLibrary> library = [device->device newLibraryWithData:data error:nil];
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:createInfo->entryPoint]];
    if (!library || !function) {
        device->createInfo.free(shaderModule);
        return nullptr;
    }

    shaderModule->library = library;
    shaderModule->function = function;
    return shaderModule;
}

void agfxShaderModuleDestroy(agfxDevice* device, agfxShaderModule* shaderModule) {
    shaderModule->library = nil;
    shaderModule->function = nil;
    device->createInfo.free(shaderModule);
}

// Render pipeline
agfxRenderPipeline* agfxRenderPipelineCreate(agfxDevice* device, const agfxRenderPipelineCreateInfo* createInfo) {
    agfxRenderPipeline* pipeline = (agfxRenderPipeline*)device->createInfo.allocate(sizeof(agfxRenderPipeline));
    memcpy(&pipeline->createInfo, createInfo, sizeof(agfxRenderPipelineCreateInfo));

    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    MTLMeshRenderPipelineDescriptor* meshDescriptor = [MTLMeshRenderPipelineDescriptor new];
    MTLDepthStencilDescriptor* depthStencilDescriptor = [MTLDepthStencilDescriptor new];

    if (createInfo->meshShader) {
        // Fill meshDescriptor with mesh shader pipeline info
        meshDescriptor.label = [NSString stringWithUTF8String:createInfo->name];
        meshDescriptor.objectFunction = createInfo->taskShader ? createInfo->taskShader->function : nil;
        meshDescriptor.meshFunction = createInfo->meshShader->function;
        meshDescriptor.fragmentFunction = createInfo->fragmentShader ? createInfo->fragmentShader->function : nil;
        meshDescriptor.supportIndirectCommandBuffers = createInfo->supportsIndirect;
        meshDescriptor.rasterizationEnabled = YES;
        for (uint32_t i = 0; i < createInfo->colorAttachmentCount; ++i) {
            meshDescriptor.colorAttachments[i].pixelFormat = agfxPixelFormatToMTL(createInfo->colorFormats[i]);
            meshDescriptor.colorAttachments[i].blendingEnabled = createInfo->blendEnable[i];
            meshDescriptor.colorAttachments[i].sourceRGBBlendFactor = agfxBlendFactorToMTL(createInfo->srcColorBlendFactor[i]);
            meshDescriptor.colorAttachments[i].destinationRGBBlendFactor = agfxBlendFactorToMTL(createInfo->dstColorBlendFactor[i]);
            meshDescriptor.colorAttachments[i].rgbBlendOperation = agfxBlendOpToMTL(createInfo->colorBlendOp[i]);
            meshDescriptor.colorAttachments[i].sourceAlphaBlendFactor = agfxBlendFactorToMTL(createInfo->srcAlphaBlendFactor[i]);
            meshDescriptor.colorAttachments[i].destinationAlphaBlendFactor = agfxBlendFactorToMTL(createInfo->dstAlphaBlendFactor[i]);
            meshDescriptor.colorAttachments[i].alphaBlendOperation = agfxBlendOpToMTL(createInfo->alphaBlendOp[i]);
        }
        meshDescriptor.depthAttachmentPixelFormat = agfxPixelFormatToMTL(createInfo->depthFormat);

        pipeline->pipelineState = [device->device newRenderPipelineStateWithMeshDescriptor:meshDescriptor options:MTLPipelineOptionNone reflection:nil error:nil];
    } else {
        descriptor.label = [NSString stringWithUTF8String:createInfo->name];
        descriptor.vertexFunction = createInfo->vertexShader ? createInfo->vertexShader->function : nil;
        descriptor.fragmentFunction = createInfo->fragmentShader ? createInfo->fragmentShader->function : nil;
        descriptor.supportIndirectCommandBuffers = createInfo->supportsIndirect;
        descriptor.rasterizationEnabled = YES;
        for (uint32_t i = 0; i < createInfo->colorAttachmentCount; ++i) {
            descriptor.colorAttachments[i].pixelFormat = agfxPixelFormatToMTL(createInfo->colorFormats[i]);
            descriptor.colorAttachments[i].blendingEnabled = createInfo->blendEnable[i];
            descriptor.colorAttachments[i].sourceRGBBlendFactor = agfxBlendFactorToMTL(createInfo->srcColorBlendFactor[i]);
            descriptor.colorAttachments[i].destinationRGBBlendFactor = agfxBlendFactorToMTL(createInfo->dstColorBlendFactor[i]);
            descriptor.colorAttachments[i].rgbBlendOperation = agfxBlendOpToMTL(createInfo->colorBlendOp[i]);
            descriptor.colorAttachments[i].sourceAlphaBlendFactor = agfxBlendFactorToMTL(createInfo->srcAlphaBlendFactor[i]);
            descriptor.colorAttachments[i].destinationAlphaBlendFactor = agfxBlendFactorToMTL(createInfo->dstAlphaBlendFactor[i]);
            descriptor.colorAttachments[i].alphaBlendOperation = agfxBlendOpToMTL(createInfo->alphaBlendOp[i]);
        }
        descriptor.depthAttachmentPixelFormat = agfxPixelFormatToMTL(createInfo->depthFormat);
        descriptor.inputPrimitiveTopology = agfxPrimitiveTopologyClassToMTL(createInfo->topology);

        pipeline->pipelineState = [device->device newRenderPipelineStateWithDescriptor:descriptor error:nil];
    }

    if (createInfo->depthFormat != AGFX_TEXTURE_FORMAT_UNKNOWN) {
        depthStencilDescriptor.depthCompareFunction = agfxCompareFunctionToMTL(createInfo->depthCompareOp);
        depthStencilDescriptor.depthWriteEnabled = createInfo->depthWriteEnable;
        pipeline->depthStencilState = [device->device newDepthStencilStateWithDescriptor:depthStencilDescriptor];
    }

    return pipeline;
}

void agfxRenderPipelineDestroy(agfxDevice* device, agfxRenderPipeline* pipeline) {
    pipeline->pipelineState = nil;
    pipeline->depthStencilState = nil;
    device->createInfo.free(pipeline);
}

// Compute pipeline
agfxComputePipeline* agfxComputePipelineCreate(agfxDevice* device, const agfxComputePipelineCreateInfo* createInfo) {
    MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
    descriptor.label = [NSString stringWithUTF8String:createInfo->name];
    descriptor.computeFunction = createInfo->computeShader->function;

    agfxComputePipeline* pipeline = (agfxComputePipeline*)device->createInfo.allocate(sizeof(agfxComputePipeline));
    memcpy(&pipeline->createInfo, createInfo, sizeof(agfxComputePipelineCreateInfo));
    pipeline->pipelineState = [device->device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:nil];
    return pipeline;
}

void agfxComputePipelineDestroy(agfxDevice* device, agfxComputePipeline* pipeline) {
    pipeline->pipelineState = nil;
    device->createInfo.free(pipeline);
}

// Acceleration structure
agfxAccelerationStructure* agfxAccelerationStructureCreate(agfxDevice* device, const agfxAccelerationStructureCreateInfo* createInfo) {
    agfxAccelerationStructure* accelerationStructure = (agfxAccelerationStructure*)device->createInfo.allocate(sizeof(agfxAccelerationStructure));
    memcpy(&accelerationStructure->createInfo, createInfo, sizeof(agfxAccelerationStructureCreateInfo));
    
    if (createInfo->type == AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL) {
        uint32_t geometryCount = createInfo->bottomLevel.geometryCount;
        NSMutableArray<MTL4AccelerationStructureGeometryDescriptor*>* geometries = [NSMutableArray arrayWithCapacity:geometryCount];

        for (uint32_t i = 0; i < createInfo->bottomLevel.geometryCount; i++) {
            agfxAccelerationStructureGeometry* geometry = &createInfo->bottomLevel.geometries[i];
            if (geometry->type == AGFX_ACCELERATION_STRUCTURE_GEOMETRY_TYPE_TRIANGLES) {
                uint64_t vertexStride = geometry->triangles.vertexBuffer->createInfo.stride;
                uint64_t indexStride = geometry->triangles.indexBuffer->createInfo.stride;

                MTL4AccelerationStructureTriangleGeometryDescriptor* geometryDescriptor = [[MTL4AccelerationStructureTriangleGeometryDescriptor alloc] init];
                geometryDescriptor.vertexBuffer = MTL4BufferRangeMake(geometry->triangles.vertexBuffer->buffer.gpuAddress + geometry->triangles.vertexOffset,
                                                                      geometry->triangles.vertexCount * vertexStride);
                geometryDescriptor.vertexStride = vertexStride;
                geometryDescriptor.vertexFormat = MTLAttributeFormatFloat3;
                geometryDescriptor.indexBuffer = MTL4BufferRangeMake(geometry->triangles.indexBuffer->buffer.gpuAddress + geometry->triangles.indexOffset,
                                                                     geometry->triangles.indexCount * indexStride);
                geometryDescriptor.indexType = indexStride == 4 ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
                geometryDescriptor.triangleCount = geometry->triangles.indexCount / 3;
                geometryDescriptor.opaque = geometry->opaque ? YES : NO;

                [geometries addObject:geometryDescriptor];
            } else {
                MTL4AccelerationStructureBoundingBoxGeometryDescriptor* geometryDescriptor = [[MTL4AccelerationStructureBoundingBoxGeometryDescriptor alloc] init];
                geometryDescriptor.boundingBoxBuffer = MTL4BufferRangeMake(geometry->aabbs.aabbBuffer->buffer.gpuAddress + geometry->aabbs.aabbOffset,
                                                                           geometry->aabbs.aabbCount * geometry->aabbs.aabbStride);
                geometryDescriptor.boundingBoxCount = geometry->aabbs.aabbCount;
                geometryDescriptor.boundingBoxStride = geometry->aabbs.aabbStride;
                geometryDescriptor.opaque = geometry->opaque ? YES : NO;

                [geometries addObject:geometryDescriptor];
            }
        }

        MTL4PrimitiveAccelerationStructureDescriptor* asDesc = [[MTL4PrimitiveAccelerationStructureDescriptor alloc] init];
        asDesc.usage = MTLAccelerationStructureUsagePreferFastBuild;
        if (createInfo->allowUpdate) {
            asDesc.usage |= MTLAccelerationStructureUsageRefit;
        }
        asDesc.geometryDescriptors = geometries;

        accelerationStructure->sizes = [device->device accelerationStructureSizesWithDescriptor:asDesc];
        accelerationStructure->mtlBottomLevelDescriptor = asDesc;

        accelerationStructure->accelerationStructure = [device->device newAccelerationStructureWithSize:accelerationStructure->sizes.accelerationStructureSize];
        [device->residencySet addAllocation:accelerationStructure->accelerationStructure];
    } else {
        MTL4InstanceAccelerationStructureDescriptor* asDesc = [[MTL4InstanceAccelerationStructureDescriptor alloc] init];
        asDesc.instanceCount = createInfo->topLevel.maxInstanceCount;
        asDesc.usage = MTLAccelerationStructureUsagePreferFastIntersection;
        if (createInfo->allowUpdate) {
            asDesc.usage |= MTLAccelerationStructureUsageRefit;
        }

        accelerationStructure->instanceBuffer = [device->device newBufferWithLength:createInfo->topLevel.maxInstanceCount * sizeof(MTLIndirectAccelerationStructureInstanceDescriptor) options:MTLResourceStorageModeShared];
        accelerationStructure->mappedInstanceBuffer = (MTLIndirectAccelerationStructureInstanceDescriptor*)accelerationStructure->instanceBuffer.contents;
        [device->residencySet addAllocation:accelerationStructure->instanceBuffer];

        asDesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeIndirect;
        asDesc.instanceDescriptorStride = sizeof(MTLIndirectAccelerationStructureInstanceDescriptor);
        asDesc.instanceDescriptorBuffer = MTL4BufferRangeMake(accelerationStructure->instanceBuffer.gpuAddress, accelerationStructure->instanceBuffer.length);

        accelerationStructure->resourceIDBuffer = [device->device newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];
        accelerationStructure->mappedResourceIDBuffer = (uint64_t*)accelerationStructure->resourceIDBuffer.contents;
        [device->residencySet addAllocation:accelerationStructure->resourceIDBuffer];

        accelerationStructure->sizes = [device->device accelerationStructureSizesWithDescriptor:asDesc];
        accelerationStructure->accelerationStructure = [device->device newAccelerationStructureWithSize:accelerationStructure->sizes.accelerationStructureSize];
        [device->residencySet addAllocation:accelerationStructure->accelerationStructure];
        
        uint64_t resourceID = accelerationStructure->accelerationStructure.gpuResourceID._impl;
        memcpy(accelerationStructure->mappedResourceIDBuffer, &resourceID, sizeof(uint64_t));

        accelerationStructure->asBindlessHandle = device->bindlessManager->writeAccelerationStructure(accelerationStructure);
        accelerationStructure->mtlTopLevelDescriptor = asDesc;
    }

    accelerationStructure->accelerationStructure.label = [NSString stringWithUTF8String:createInfo->name];
    return accelerationStructure;
}

agfxAccelerationStructure* agfxAccelerationStructureCreateCompacted(agfxDevice* device, const agfxAccelerationStructureCreateInfo* createInfo, uint64_t compactedSize) {
    agfxAccelerationStructure* accelerationStructure = (agfxAccelerationStructure*)device->createInfo.allocate(sizeof(agfxAccelerationStructure));
    memcpy(&accelerationStructure->createInfo, createInfo, sizeof(agfxAccelerationStructureCreateInfo));
    accelerationStructure->sizes.accelerationStructureSize = compactedSize;
    accelerationStructure->accelerationStructure = [device->device newAccelerationStructureWithSize:compactedSize];
    [device->residencySet addAllocation:accelerationStructure->accelerationStructure];
    accelerationStructure->accelerationStructure.label = [NSString stringWithUTF8String:createInfo->name];
    return accelerationStructure;
}

void agfxAccelerationStructureDestroy(agfxDevice* device, agfxAccelerationStructure* accelerationStructure) {
    accelerationStructure->accelerationStructure = nil;
    if (accelerationStructure->instanceBuffer) accelerationStructure->instanceBuffer = nil;
    if (accelerationStructure->resourceIDBuffer) accelerationStructure->resourceIDBuffer = nil;
    device->bindlessManager->freeAccelerationStructure(accelerationStructure->asBindlessHandle.handle);
    device->createInfo.free(accelerationStructure);
}

void agfxAccelerationStructureGetSizes(agfxDevice* device, agfxAccelerationStructure* accelerationStructure, agfxAccelerationStructureSizes* sizes) {
    sizes->scratchBufferSize = accelerationStructure->sizes.buildScratchBufferSize;
    sizes->updateScratchBufferSize = accelerationStructure->sizes.refitScratchBufferSize;
}

uint64_t agfxAccelerationStructureGetHandle(agfxAccelerationStructure* accelerationStructure) {
    return accelerationStructure->asBindlessHandle.handle;
}

void agfxAccelerationStructureAddInstances(agfxAccelerationStructure* accelerationStructure, const agfxAccelerationStructureInstance* instances, uint32_t instanceCount) {
    for (uint32_t i = 0; i < instanceCount; ++i) {
        const agfxAccelerationStructureInstance* instance = &instances[i];

        MTLIndirectAccelerationStructureInstanceDescriptor* mtlInstance = &accelerationStructure->mappedInstanceBuffer[accelerationStructure->currentInstanceCount + i];
        mtlInstance->accelerationStructureID = instance->blas->accelerationStructure.gpuResourceID;
        // AGFX transform is row-major 3x4 (transform[row*4+col]); MTLPackedFloat4x3 is
        // column-major (columns[col][row]). A straight memcpy transposes the matrix and
        // flings the instance to a garbage world position, so unpack it explicitly.
        for (int col = 0; col < 4; ++col) {
            mtlInstance->transformationMatrix.columns[col][0] = instance->transform[0 * 4 + col];
            mtlInstance->transformationMatrix.columns[col][1] = instance->transform[1 * 4 + col];
            mtlInstance->transformationMatrix.columns[col][2] = instance->transform[2 * 4 + col];
        }
        mtlInstance->options = instance->opaque ? MTLAccelerationStructureInstanceOptionOpaque :  MTLAccelerationStructureInstanceOptionNonOpaque;
        mtlInstance->mask = 0xFF;
        mtlInstance->userID = instance->userID;
    }
    accelerationStructure->currentInstanceCount += instanceCount;
}

void agfxAccelerationStructureResetInstances(agfxAccelerationStructure* accelerationStructure) {
    accelerationStructure->currentInstanceCount = 0;
}

//

MTLTextureType agfxTextureTypeToMTL(agfxTextureType type) {
    switch (type)
    {
        case AGFX_TEXTURE_TYPE_1D: return MTLTextureType1D;
        case AGFX_TEXTURE_TYPE_2D: return MTLTextureType2D;
        case AGFX_TEXTURE_TYPE_2D_ARRAY: return MTLTextureType2DArray;
        case AGFX_TEXTURE_TYPE_3D: return MTLTextureType3D;
        case AGFX_TEXTURE_TYPE_CUBE: return MTLTextureTypeCube;
        default: return MTLTextureType2D; // Default to 2D if unknown
    }
}

MTLTextureUsage agfxTextureUsageToMTL(agfxTextureUsage usage) {
    MTLTextureUsage mtlUsage = MTLTextureUsagePixelFormatView;
    if (usage & AGFX_TEXTURE_USAGE_SAMPLED) mtlUsage |= MTLTextureUsageShaderRead;
    if (usage & AGFX_TEXTURE_USAGE_STORAGE) mtlUsage |= MTLTextureUsageShaderWrite;
    if (usage & AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT) mtlUsage |= MTLTextureUsageRenderTarget;
    if (usage & AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT) mtlUsage |= MTLTextureUsageRenderTarget;
    return mtlUsage;
}

MTLPixelFormat agfxPixelFormatToMTL(agfxTextureFormat format) {
    switch (format)
    {
        case AGFX_TEXTURE_FORMAT_UNKNOWN: return MTLPixelFormatInvalid;

        case AGFX_TEXTURE_FORMAT_R8_UNORM: return MTLPixelFormatR8Unorm;
        case AGFX_TEXTURE_FORMAT_RG8_UNORM: return MTLPixelFormatRG8Unorm;
        case AGFX_TEXTURE_FORMAT_RGBA8_UNORM: return MTLPixelFormatRGBA8Unorm;
        case AGFX_TEXTURE_FORMAT_BGRA8_UNORM: return MTLPixelFormatBGRA8Unorm;
        case AGFX_TEXTURE_FORMAT_RGBA8_UNORM_SRGB: return MTLPixelFormatRGBA8Unorm_sRGB;
        case AGFX_TEXTURE_FORMAT_BGRA8_UNORM_SRGB: return MTLPixelFormatBGRA8Unorm_sRGB;

        case AGFX_TEXTURE_FORMAT_R16_UNORM: return MTLPixelFormatR16Unorm;
        case AGFX_TEXTURE_FORMAT_RG16_UNORM: return MTLPixelFormatRG16Unorm;
        case AGFX_TEXTURE_FORMAT_RGBA16_UNORM: return MTLPixelFormatRGBA16Unorm;

        case AGFX_TEXTURE_FORMAT_R16F: return MTLPixelFormatR16Float;
        case AGFX_TEXTURE_FORMAT_RG16F: return MTLPixelFormatRG16Float;
        case AGFX_TEXTURE_FORMAT_RGBA16F: return MTLPixelFormatRGBA16Float;

        case AGFX_TEXTURE_FORMAT_R32F: return MTLPixelFormatR32Float;
        case AGFX_TEXTURE_FORMAT_RG32F: return MTLPixelFormatRG32Float;
        case AGFX_TEXTURE_FORMAT_RGBA32F: return MTLPixelFormatRGBA32Float;

        case AGFX_TEXTURE_FORMAT_DEPTH32F: return MTLPixelFormatDepth32Float;

        case AGFX_TEXTURE_FORMAT_BC1_UNORM: return MTLPixelFormatBC1_RGBA;
        case AGFX_TEXTURE_FORMAT_BC1_UNORM_SRGB: return MTLPixelFormatBC1_RGBA_sRGB;
        case AGFX_TEXTURE_FORMAT_BC3_UNORM: return MTLPixelFormatBC3_RGBA;
        case AGFX_TEXTURE_FORMAT_BC3_UNORM_SRGB: return MTLPixelFormatBC3_RGBA_sRGB;
        case AGFX_TEXTURE_FORMAT_BC4_UNORM: return MTLPixelFormatBC4_RUnorm;
        case AGFX_TEXTURE_FORMAT_BC5_UNORM: return MTLPixelFormatBC5_RGUnorm;
        case AGFX_TEXTURE_FORMAT_BC6H_UFLOAT: return MTLPixelFormatBC6H_RGBUfloat;
        case AGFX_TEXTURE_FORMAT_BC7_UNORM: return MTLPixelFormatBC7_RGBAUnorm;
        case AGFX_TEXTURE_FORMAT_BC7_UNORM_SRGB: return MTLPixelFormatBC7_RGBAUnorm_sRGB;

        case AGFX_TEXTURE_FORMAT_ASTC_4X4_UNORM: return MTLPixelFormatASTC_4x4_LDR;
        case AGFX_TEXTURE_FORMAT_ASTC_4X4_UNORM_SRGB: return MTLPixelFormatASTC_4x4_sRGB;
        case AGFX_TEXTURE_FORMAT_ASTC_8X8_UNORM: return MTLPixelFormatASTC_8x8_LDR;
        case AGFX_TEXTURE_FORMAT_ASTC_8X8_UNORM_SRGB: return MTLPixelFormatASTC_8x8_sRGB;

        default: return MTLPixelFormatInvalid; // Default to invalid if unknown
    }
}

MTLRegion agfxTextureRegionToMTL(const agfxTextureRegion* region) {
    MTLRegion mtlRegion;
    mtlRegion.origin.x = region->x;
    mtlRegion.origin.y = region->y;
    mtlRegion.origin.z = region->z;
    mtlRegion.size.width = region->width;
    mtlRegion.size.height = region->height;
    mtlRegion.size.depth = region->depth;
    return mtlRegion;
}

MTLSamplerMinMagFilter agfxSamplerFilterToMTL(agfxSamplerFilter filter) {
    switch (filter)
    {
        case AGFX_SAMPLER_FILTER_NEAREST: return MTLSamplerMinMagFilterNearest;
        case AGFX_SAMPLER_FILTER_LINEAR: return MTLSamplerMinMagFilterLinear;
        default: return MTLSamplerMinMagFilterNearest; // Default to nearest if unknown
    }
}

MTLSamplerMipFilter agfxSamplerMipFilterToMTL(agfxSamplerFilter filter) {
    switch (filter)
    {
        case AGFX_SAMPLER_FILTER_NEAREST: return MTLSamplerMipFilterNearest;
        case AGFX_SAMPLER_FILTER_LINEAR: return MTLSamplerMipFilterLinear;
        default: return MTLSamplerMipFilterNotMipmapped; // Default to not mipmapped if unknown
    }
}

MTLSamplerAddressMode agfxSamplerAddressModeToMTL(agfxSamplerAddressMode mode) {
    switch (mode)
    {
        case AGFX_SAMPLER_ADDRESS_MODE_REPEAT: return MTLSamplerAddressModeRepeat;
        case AGFX_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return MTLSamplerAddressModeMirrorRepeat;
        case AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: return MTLSamplerAddressModeClampToEdge;
        case AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: return MTLSamplerAddressModeClampToBorderColor;
        default: return MTLSamplerAddressModeClampToEdge; // Default to clamp to edge if unknown
    }
}

MTLCompareFunction agfxCompareFunctionToMTL(agfxComparisonFunction func) {
    switch (func)
    {
        case AGFX_COMPARISON_FUNCTION_NEVER: return MTLCompareFunctionNever;
        case AGFX_COMPARISON_FUNCTION_LESS: return MTLCompareFunctionLess;
        case AGFX_COMPARISON_FUNCTION_EQUAL: return MTLCompareFunctionEqual;
        case AGFX_COMPARISON_FUNCTION_LESS_EQUAL: return MTLCompareFunctionLessEqual;
        case AGFX_COMPARISON_FUNCTION_GREATER: return MTLCompareFunctionGreater;
        case AGFX_COMPARISON_FUNCTION_NOT_EQUAL: return MTLCompareFunctionNotEqual;
        case AGFX_COMPARISON_FUNCTION_GREATER_EQUAL: return MTLCompareFunctionGreaterEqual;
        case AGFX_COMPARISON_FUNCTION_ALWAYS: return MTLCompareFunctionAlways;
        default: return MTLCompareFunctionAlways; // Default to always if unknown
    }
}

MTLLoadAction agfxLoadActionToMTL(agfxLoadOp action) {
    switch (action) {
        case AGFX_LOAD_OPERATION_LOAD: return MTLLoadActionLoad;
        case AGFX_LOAD_OPERATION_CLEAR: return MTLLoadActionClear;
        case AGFX_LOAD_OPERATION_DONT_CARE: return MTLLoadActionDontCare;
        default: return MTLLoadActionDontCare; // Default to don't care if unknown
    }
}

MTLStoreAction agfxStoreActionToMTL(agfxStoreOp action) {
    switch (action) {
        case AGFX_STORE_OPERATION_STORE: return MTLStoreActionStore;
        case AGFX_STORE_OPERATION_DONT_CARE: return MTLStoreActionDontCare;
        default: return MTLStoreActionDontCare; // Default to don't care if unknown
    }
}

MTLCullMode agfxCullModeToMTL(agfxCullMode mode) {
    switch (mode) {
        case AGFX_CULL_MODE_NONE: return MTLCullModeNone;
        case AGFX_CULL_MODE_FRONT: return MTLCullModeFront;
        case AGFX_CULL_MODE_BACK: return MTLCullModeBack;
        default: return MTLCullModeNone; // Default to none if unknown
    }
}

MTLWinding agfxWindingToMTL(agfxFrontFace winding) {
    switch (winding) {
        case AGFX_FRONT_FACE_COUNTER_CLOCKWISE: return MTLWindingCounterClockwise;
        case AGFX_FRONT_FACE_CLOCKWISE: return MTLWindingClockwise;
        default: return MTLWindingCounterClockwise; // Default to counter-clockwise if unknown
    }
}

MTLTriangleFillMode agfxFillModeToMTL(agfxFillMode mode) {
    switch (mode) {
        case AGFX_FILL_MODE_SOLID: return MTLTriangleFillModeFill;
        case AGFX_FILL_MODE_WIREFRAME: return MTLTriangleFillModeLines;
        default: return MTLTriangleFillModeFill; // Default to fill if unknown
    }
}

MTLBlendFactor agfxBlendFactorToMTL(agfxBlendFactor factor) {
    switch (factor) {
        case AGFX_BLEND_FACTOR_ZERO: return MTLBlendFactorZero;
        case AGFX_BLEND_FACTOR_ONE: return MTLBlendFactorOne;
        case AGFX_BLEND_FACTOR_SRC_COLOR: return MTLBlendFactorSourceColor;
        case AGFX_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return MTLBlendFactorOneMinusSourceColor;
        case AGFX_BLEND_FACTOR_DST_COLOR: return MTLBlendFactorDestinationColor;
        case AGFX_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return MTLBlendFactorOneMinusDestinationColor;
        case AGFX_BLEND_FACTOR_SRC_ALPHA: return MTLBlendFactorSourceAlpha;
        case AGFX_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
        case AGFX_BLEND_FACTOR_DST_ALPHA: return MTLBlendFactorDestinationAlpha;
        case AGFX_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return MTLBlendFactorOneMinusDestinationAlpha;
        default: return MTLBlendFactorOne; // Default to one if unknown
    }
}

MTLBlendOperation agfxBlendOpToMTL(agfxBlendOperation op) {
    switch (op) {
        case AGFX_BLEND_OPERATION_ADD: return MTLBlendOperationAdd;
        case AGFX_BLEND_OPERATION_SUBTRACT: return MTLBlendOperationSubtract;
        case AGFX_BLEND_OPERATION_REVERSE_SUBTRACT: return MTLBlendOperationReverseSubtract;
        case AGFX_BLEND_OPERATION_MIN: return MTLBlendOperationMin;
        case AGFX_BLEND_OPERATION_MAX: return MTLBlendOperationMax;
        default: return MTLBlendOperationAdd; // Default to add if unknown
    }
}

MTLPrimitiveType agfxPrimitiveTypeToMTL(agfxTopology topology) {
    switch (topology) {
        case AGFX_TOPOLOGY_TRIANGLES: return MTLPrimitiveTypeTriangle;
        case AGFX_TOPOLOGY_LINES: return MTLPrimitiveTypeLine;
        case AGFX_TOPOLOGY_POINTS: return MTLPrimitiveTypePoint;
        default: return MTLPrimitiveTypeTriangle; // Default to triangle if unknown
    }
}

MTLPrimitiveTopologyClass agfxPrimitiveTopologyClassToMTL(agfxTopology topology) {
    switch (topology) {
        case AGFX_TOPOLOGY_TRIANGLES: return MTLPrimitiveTopologyClassTriangle;
        case AGFX_TOPOLOGY_LINES: return MTLPrimitiveTopologyClassLine;
        case AGFX_TOPOLOGY_POINTS: return MTLPrimitiveTopologyClassPoint;
        default: return MTLPrimitiveTopologyClassTriangle; // Default to triangle if unknown
    }
}
