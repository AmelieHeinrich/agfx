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
#include <cstdio>
#include <cstdarg>

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
        // Object/Mesh are included because AGFX_INDIRECT_BUNDLE_TYPE_DRAW_MESH's ICB commands read
        // the same commands/argument buffers from the object and mesh stages.
        case AGFX_RESOURCE_STATE_INDIRECT_ARGUMENT: *outProducer = 0; *outConsumer = MTLStageVertex | MTLStageDispatch | MTLStageObject | MTLStageMesh; break;
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
        MTLStages oldProducer, oldConsumer;
        MTLStages newProducer, newConsumer;
        agfxResourceStateToMTLStages(oldState, &oldProducer, &oldConsumer);
        agfxResourceStateToMTLStages(newState, &newProducer, &newConsumer);

        // Stages that must finish first: whoever wrote the resource in its old state - or, when the
        // old state writes nothing, whoever *read* it. That read-only case is a write-after-read
        // hazard (e.g. INDIRECT_ARGUMENT -> UNORDERED_ACCESS, where the next frame's culling pass
        // overwrites a commands buffer the current frame's draws are still reading), and dropping it
        // for want of a producer would leave the write unordered against those reads.
        MTLStages afterStages = oldProducer ? oldProducer : oldConsumer;
        // Stages that must wait: whoever reads the resource in its new state - or, when the new
        // state only writes (e.g. RENDER_TARGET), whoever writes it.
        MTLStages beforeStages = newConsumer ? newConsumer : newProducer;

        // MTL4DebugCommandEncoder rejects barrierAfterQueueStages:beforeStages: if either mask is 0,
        // which now only happens for states that neither read nor write (COMMON/PRESENT).
        if (afterStages == 0 || beforeStages == 0) return;

        this->afterStages |= afterStages;
        this->beforeStages |= beforeStages;
        visibility = MTL4VisibilityOptionDevice;
        pending = true;
    }

    // Flushed at a pass boundary: everything being ordered against lives in prior encoders, which is
    // exactly what afterQueueStages covers.
    void encode(id<MTL4CommandEncoder> encoder) {
        if (!pending) return;

        [encoder barrierAfterQueueStages:afterStages beforeStages:beforeStages visibilityOptions:visibility];

        afterStages = 0;
        beforeStages = 0;
        visibility = MTL4VisibilityOptionNone;
        pending = false;
    }

    // Flushed in the middle of an open encoder. afterQueueStages explicitly excludes the current
    // encoder, so a queue barrier alone would not order against work already encoded in this pass
    // (e.g. a copy earlier in the same compute encoder). Emit the encoder barrier as well, with
    // stages masked to those this encoder can actually encode.
    void encodeInline(id<MTL4CommandEncoder> encoder, MTLStages encoderStages) {
        if (!pending) return;

        MTLStages afterInEncoder = afterStages & encoderStages;
        MTLStages beforeInEncoder = beforeStages & encoderStages;
        if (afterInEncoder != 0 && beforeInEncoder != 0) {
            [encoder barrierAfterEncoderStages:afterInEncoder beforeEncoderStages:beforeInEncoder visibilityOptions:visibility];
        }

        encode(encoder);
    }

    MTLStages afterStages = 0;
    MTLStages beforeStages = 0;
    MTL4VisibilityOptions visibility = MTL4VisibilityOptionNone;
    bool pending = false;
};

// Indirect bundle (ICB conversion) internals
//
// Metal has no ExecuteIndirect. Indirect draws must be pre-encoded into an MTLIndirectCommandBuffer,
// one command per draw, which AGFX does from a compute kernel that reads the D3D12-shaped commands
// buffer the culling shader wrote. See notes/mdi.md.

static uint32_t agfxIndirectBundleTypeStride(agfxIndirectBundleType type) {
    switch (type) {
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW: return sizeof(agfxDrawCommand);
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW_INDEXED: return sizeof(agfxDrawIndexedCommand);
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW_MESH: return sizeof(agfxDrawMeshCommand);
        case AGFX_INDIRECT_BUNDLE_TYPE_DISPATCH: return sizeof(agfxDispatchCommand);
        default: return 0;
    }
}

// Argument table slots the conversion kernels read. 0 and 1 are the bindless heaps, already bound
// on every command buffer's compute argument table; the rest are set per PrepareIndirectBundle call.
enum {
    kAGFXICBParamsBindPoint       = 11,
    kAGFXICBPushConstantsBindPoint= 12,
    kAGFXICBCommandsBindPoint     = 13,
    kAGFXICBCountBindPoint        = 14,
    kAGFXICBTLABBindPoint         = 15,
    kAGFXICBDrawArgsBindPoint     = 16,
    kAGFXICBExecRangeBindPoint    = 17,
    kAGFXICBUniformsBindPoint     = 18,
    kAGFXICBIndexBufferBindPoint  = 19,
    kAGFXICBTargetBindPoint       = 20,
};

// One per-command slice of each of these, sized to the bundle's maxCommandCount.
struct agfxICBDrawArgsSlice {
    // Mirrors IRRuntimeDrawParams (msc_runtime/metal_irconverter_runtime.h): the largest member is
    // IRRuntimeDrawIndexedArgument at 5 x 4 bytes.
    uint32_t words[5];
};

struct agfxICBConvertParams {
    uint32_t commandOffset;
    uint32_t countIndex;
    uint32_t maxCommandCount;
    uint32_t primitiveType;     // MTLPrimitiveType
    uint32_t use16BitIndices;
    uint32_t indexStride;
    uint32_t objectThreadsPerGroup[3];
    uint32_t meshThreadsPerGroup[3];
    uint32_t threadsPerGroup[3];
};

// Compiled with newLibraryWithSource: at device creation - target("agfx") has no offline shader
// build step, so the few MSC runtime structs these kernels need are restated here rather than
// #include'd from msc_runtime/metal_irconverter_runtime.h. The bind-point constants below must stay
// in sync with kIRDescriptorHeapBindPoint/kIRSamplerHeapBindPoint/kIRArgumentBufferBindPoint/
// kIRArgumentBufferDrawArgumentsBindPoint/kIRArgumentBufferUniformsBindPoint in that header, and the
// command structs with agfxDrawCommand & friends in agfx.h.
static const char* kAGFXICBConvertSource = R"METAL(
#include <metal_stdlib>
#include <metal_command_buffer>
using namespace metal;

// Must match kIR*BindPoint in metal_irconverter_runtime.h.
constant uint kIRDescriptorHeap  = 0;
constant uint kIRSamplerHeap     = 1;
constant uint kIRArgumentBuffer  = 2;
constant uint kIRDrawArguments   = 4;
constant uint kIRUniforms        = 5;

struct ICBConvertParams {
    uint commandOffset;
    uint countIndex;
    uint maxCommandCount;
    uint primitiveType;
    uint use16BitIndices;
    uint indexStride;
    uint objectThreadsPerGroup[3];
    uint meshThreadsPerGroup[3];
    uint threadsPerGroup[3];
};

struct ICBExecutionRange {
    uint location;
    uint length;
};

// Mirrors agfxTLAB: 128 bytes of push constants followed by the per-draw drawID, which is where
// Metal Shader Converter expects the b1 root constant to live.
struct TLAB {
    uchar bytes[128];
    uint  drawID;
};

// Mirrors agfxDrawCommand / agfxDrawIndexedCommand / agfxDrawMeshCommand / agfxDispatchCommand
// (agfx.h). drawID is the leading field on all but dispatch.
struct DrawCommand        { uint drawID; uint vertexCount; uint instanceCount; uint firstVertex; uint firstInstance; };
struct DrawIndexedCommand { uint drawID; uint indexCount; uint instanceCount; uint firstIndex; int vertexOffset; uint firstInstance; };
struct DrawMeshCommand    { uint drawID; uint groupSizeX; uint groupSizeY; uint groupSizeZ; };
struct DispatchCommand    { uint groupCountX; uint groupCountY; uint groupCountZ; };

// An MTLIndirectCommandBuffer cannot be a direct kernel buffer argument; it is reached through an
// argument-buffer struct holding its resource ID, which the CPU side writes once at bundle creation.
struct ICBContainer { command_buffer icb; };

// Mirrors IRRuntimeDrawArgument / IRRuntimeDrawIndexedArgument.
struct RuntimeDrawArgument        { uint vertexCountPerInstance; uint instanceCount; uint startVertexLocation; uint startInstanceLocation; uint pad; };
struct RuntimeDrawIndexedArgument { uint indexCountPerInstance; uint instanceCount; uint startIndexLocation; int baseVertexLocation; uint startInstanceLocation; };

// Writes the execution range that actually bounds ICB replay. Commands past the live count are
// never executed however stale their contents, so no reset pass is needed when the draw count
// shrinks frame to frame.
static uint icb_live_count(device const uint* countBuffer,
                           device ICBExecutionRange* execRanges,
                           constant ICBConvertParams& params,
                           uint tid)
{
    uint live = min(countBuffer[params.countIndex], params.maxCommandCount);
    if (tid == 0) {
        execRanges[params.countIndex].location = params.commandOffset;
        execRanges[params.countIndex].length = live;
    }
    return live;
}

static void icb_fill_tlab(device TLAB& tlab, constant uchar* pushConstants, uint drawID)
{
    for (uint b = 0; b < 128; ++b) {
        tlab.bytes[b] = pushConstants[b];
    }
    tlab.drawID = drawID;
}

// ICB commands inherit nothing (inheritBuffers = NO), so every binding the converted shaders expect
// is rebound per command, on every stage that can read it.
static void icb_bind_render(thread render_command& cmd,
                            device const void* resourceHeap,
                            device const void* samplerHeap,
                            device const void* tlab,
                            device const void* drawArgs,
                            device const void* uniforms)
{
    cmd.set_vertex_buffer(resourceHeap, kIRDescriptorHeap);
    cmd.set_vertex_buffer(samplerHeap, kIRSamplerHeap);
    cmd.set_vertex_buffer(tlab, kIRArgumentBuffer);
    cmd.set_vertex_buffer(drawArgs, kIRDrawArguments);
    cmd.set_vertex_buffer(uniforms, kIRUniforms);

    cmd.set_fragment_buffer(resourceHeap, kIRDescriptorHeap);
    cmd.set_fragment_buffer(samplerHeap, kIRSamplerHeap);
    cmd.set_fragment_buffer(tlab, kIRArgumentBuffer);
    cmd.set_fragment_buffer(drawArgs, kIRDrawArguments);
    cmd.set_fragment_buffer(uniforms, kIRUniforms);
}

static void icb_bind_mesh(thread render_command& cmd,
                          device const void* resourceHeap,
                          device const void* samplerHeap,
                          device const void* tlab,
                          device const void* drawArgs,
                          device const void* uniforms)
{
    cmd.set_object_buffer(resourceHeap, kIRDescriptorHeap);
    cmd.set_object_buffer(samplerHeap, kIRSamplerHeap);
    cmd.set_object_buffer(tlab, kIRArgumentBuffer);
    cmd.set_object_buffer(drawArgs, kIRDrawArguments);
    cmd.set_object_buffer(uniforms, kIRUniforms);

    cmd.set_mesh_buffer(resourceHeap, kIRDescriptorHeap);
    cmd.set_mesh_buffer(samplerHeap, kIRSamplerHeap);
    cmd.set_mesh_buffer(tlab, kIRArgumentBuffer);
    cmd.set_mesh_buffer(drawArgs, kIRDrawArguments);
    cmd.set_mesh_buffer(uniforms, kIRUniforms);

    cmd.set_fragment_buffer(resourceHeap, kIRDescriptorHeap);
    cmd.set_fragment_buffer(samplerHeap, kIRSamplerHeap);
    cmd.set_fragment_buffer(tlab, kIRArgumentBuffer);
    cmd.set_fragment_buffer(drawArgs, kIRDrawArguments);
    cmd.set_fragment_buffer(uniforms, kIRUniforms);
}

static primitive_type icb_primitive_type(uint type)
{
    // Matches MTLPrimitiveType.
    switch (type) {
        case 0: return primitive_type::point;
        case 1: return primitive_type::line;
        case 2: return primitive_type::line_strip;
        case 4: return primitive_type::triangle_strip;
        default: return primitive_type::triangle;
    }
}

kernel void icb_convert_draw(device const DrawCommand* commands       [[buffer(13)]],
                             device const uint* countBuffer           [[buffer(14)]],
                             device TLAB* tlabSlices                  [[buffer(15)]],
                             device RuntimeDrawArgument* drawArgs     [[buffer(16)]],
                             device ICBExecutionRange* execRanges     [[buffer(17)]],
                             device const void* uniforms              [[buffer(18)]],
                             device const void* resourceHeap          [[buffer(0)]],
                             device const void* samplerHeap           [[buffer(1)]],
                             constant uchar* pushConstants            [[buffer(12)]],
                             constant ICBConvertParams& params        [[buffer(11)]],
                             device ICBContainer* icbContainer         [[buffer(20)]],
                             uint tid [[thread_position_in_grid]])
{
    uint live = icb_live_count(countBuffer, execRanges, params, tid);
    if (tid >= live) return;

    uint slot = params.commandOffset + tid;
    DrawCommand c = commands[slot];

    icb_fill_tlab(tlabSlices[slot], pushConstants, c.drawID);

    RuntimeDrawArgument arg;
    arg.vertexCountPerInstance = c.vertexCount;
    arg.instanceCount = c.instanceCount;
    arg.startVertexLocation = c.firstVertex;
    arg.startInstanceLocation = c.firstInstance;
    arg.pad = 0;
    drawArgs[slot] = arg;

    render_command cmd(icbContainer->icb, slot);
    icb_bind_render(cmd, resourceHeap, samplerHeap, &tlabSlices[slot], &drawArgs[slot], uniforms);
    cmd.draw_primitives(icb_primitive_type(params.primitiveType),
                        c.firstVertex, c.vertexCount, c.instanceCount, c.firstInstance);
}

kernel void icb_convert_draw_indexed(device const DrawIndexedCommand* commands  [[buffer(13)]],
                                     device const uint* countBuffer             [[buffer(14)]],
                                     device TLAB* tlabSlices                    [[buffer(15)]],
                                     device RuntimeDrawIndexedArgument* drawArgs[[buffer(16)]],
                                     device ICBExecutionRange* execRanges       [[buffer(17)]],
                                     device const void* uniforms                [[buffer(18)]],
                                     device const uchar* indexBuffer            [[buffer(19)]],
                                     device const void* resourceHeap            [[buffer(0)]],
                                     device const void* samplerHeap             [[buffer(1)]],
                                     constant uchar* pushConstants              [[buffer(12)]],
                                     constant ICBConvertParams& params          [[buffer(11)]],
                                     device ICBContainer* icbContainer           [[buffer(20)]],
                                     uint tid [[thread_position_in_grid]])
{
    uint live = icb_live_count(countBuffer, execRanges, params, tid);
    if (tid >= live) return;

    uint slot = params.commandOffset + tid;
    DrawIndexedCommand c = commands[slot];

    icb_fill_tlab(tlabSlices[slot], pushConstants, c.drawID);

    RuntimeDrawIndexedArgument arg;
    arg.indexCountPerInstance = c.indexCount;
    arg.instanceCount = c.instanceCount;
    arg.startIndexLocation = c.firstIndex;
    arg.baseVertexLocation = c.vertexOffset;
    arg.startInstanceLocation = c.firstInstance;
    drawArgs[slot] = arg;

    render_command cmd(icbContainer->icb, slot);
    icb_bind_render(cmd, resourceHeap, samplerHeap, &tlabSlices[slot], &drawArgs[slot], uniforms);

    // draw_indexed_primitives takes a typed pointer already advanced to the first index, mirroring
    // the gpuAddress + firstIndex * stride offset the direct DrawIndexed path applies.
    device const uchar* first = indexBuffer + (uint64_t)c.firstIndex * params.indexStride;
    if (params.use16BitIndices != 0) {
        cmd.draw_indexed_primitives(icb_primitive_type(params.primitiveType),
                                    c.indexCount, (device const ushort*)first,
                                    c.instanceCount, c.vertexOffset, c.firstInstance);
    } else {
        cmd.draw_indexed_primitives(icb_primitive_type(params.primitiveType),
                                    c.indexCount, (device const uint*)first,
                                    c.instanceCount, c.vertexOffset, c.firstInstance);
    }
}

kernel void icb_convert_draw_mesh(device const DrawMeshCommand* commands  [[buffer(13)]],
                                  device const uint* countBuffer          [[buffer(14)]],
                                  device TLAB* tlabSlices                 [[buffer(15)]],
                                  device RuntimeDrawArgument* drawArgs    [[buffer(16)]],
                                  device ICBExecutionRange* execRanges    [[buffer(17)]],
                                  device const void* uniforms             [[buffer(18)]],
                                  device const void* resourceHeap         [[buffer(0)]],
                                  device const void* samplerHeap          [[buffer(1)]],
                                  constant uchar* pushConstants           [[buffer(12)]],
                                  constant ICBConvertParams& params       [[buffer(11)]],
                                  device ICBContainer* icbContainer        [[buffer(20)]],
                                  uint tid [[thread_position_in_grid]])
{
    uint live = icb_live_count(countBuffer, execRanges, params, tid);
    if (tid >= live) return;

    uint slot = params.commandOffset + tid;
    DrawMeshCommand c = commands[slot];

    icb_fill_tlab(tlabSlices[slot], pushConstants, c.drawID);

    RuntimeDrawArgument arg;
    arg.vertexCountPerInstance = c.groupSizeX;
    arg.instanceCount = c.groupSizeY;
    arg.startVertexLocation = c.groupSizeZ;
    arg.startInstanceLocation = 0;
    arg.pad = 0;
    drawArgs[slot] = arg;

    render_command cmd(icbContainer->icb, slot);
    icb_bind_mesh(cmd, resourceHeap, samplerHeap, &tlabSlices[slot], &drawArgs[slot], uniforms);
    cmd.draw_mesh_threadgroups(uint3(c.groupSizeX, c.groupSizeY, c.groupSizeZ),
                               uint3(params.objectThreadsPerGroup[0], params.objectThreadsPerGroup[1], params.objectThreadsPerGroup[2]),
                               uint3(params.meshThreadsPerGroup[0], params.meshThreadsPerGroup[1], params.meshThreadsPerGroup[2]));
}

// agfxDispatchCommand carries no drawID (see notes/mdi.md open question #5), so there is no
// per-command varying part of the TLAB - every command gets the same push constants.
kernel void icb_convert_dispatch(device const DispatchCommand* commands [[buffer(13)]],
                                 device const uint* countBuffer         [[buffer(14)]],
                                 device TLAB* tlabSlices                [[buffer(15)]],
                                 device ICBExecutionRange* execRanges   [[buffer(17)]],
                                 device const void* resourceHeap        [[buffer(0)]],
                                 device const void* samplerHeap         [[buffer(1)]],
                                 constant uchar* pushConstants          [[buffer(12)]],
                                 constant ICBConvertParams& params      [[buffer(11)]],
                                 device ICBContainer* icbContainer       [[buffer(20)]],
                                 uint tid [[thread_position_in_grid]])
{
    uint live = icb_live_count(countBuffer, execRanges, params, tid);
    if (tid >= live) return;

    uint slot = params.commandOffset + tid;
    DispatchCommand c = commands[slot];

    icb_fill_tlab(tlabSlices[slot], pushConstants, 0);

    compute_command cmd(icbContainer->icb, slot);
    cmd.set_kernel_buffer(resourceHeap, kIRDescriptorHeap);
    cmd.set_kernel_buffer(samplerHeap, kIRSamplerHeap);
    cmd.set_kernel_buffer(&tlabSlices[slot], kIRArgumentBuffer);
    cmd.concurrent_dispatch_threadgroups(uint3(c.groupCountX, c.groupCountY, c.groupCountZ),
                                         uint3(params.threadsPerGroup[0], params.threadsPerGroup[1], params.threadsPerGroup[2]));
}
)METAL";

// Device

struct agfxDevice {
    agfxDeviceCreateInfo createInfo;
    id<MTLDevice> device;
    id<MTLResidencySet> residencySet;

    agfxMetalBindlessManager* bindlessManager = nullptr;

    id<MTLLibrary> internalLibrary;
    id<MTLComputePipelineState> icbConvertPipelines[4]; // indexed by agfxIndirectBundleType
};

static void agfxLog(agfxDevice* device, agfxLogSeverity severity, const char* fmt, ...) {
    if (!device->createInfo.logFunction) return;

    char message[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);

    device->createInfo.logFunction(severity, message);
}

agfxDevice* agfxDeviceCreate(const agfxDeviceCreateInfo* createInfo) {
    agfxDevice* device = (agfxDevice*)createInfo->allocate(sizeof(agfxDevice));
    memcpy(&device->createInfo, createInfo, sizeof(agfxDeviceCreateInfo));
    device->device = MTLCreateSystemDefaultDevice();
    if (!device->device) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxDeviceCreate: MTLCreateSystemDefaultDevice returned nil, no Metal-capable GPU found");
        createInfo->free(device);
        return nullptr;
    }

    MTLResidencySetDescriptor* residencySetDesc = [MTLResidencySetDescriptor new];
    residencySetDesc.label = @"ResidencySet";
    NSError* residencySetError = nil;
    device->residencySet = [device->device newResidencySetWithDescriptor:residencySetDesc error:&residencySetError];
    if (!device->residencySet) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxDeviceCreate: newResidencySetWithDescriptor failed: %s",
            residencySetError.localizedDescription.UTF8String);
    }

    void* memory = createInfo->allocate(sizeof(agfxMetalBindlessManager));
    device->bindlessManager = new(memory) agfxMetalBindlessManager(device->device, device->residencySet);

    // Engine-internal ICB conversion kernels, compiled from source once per device.
    NSError* internalLibraryError = nil;
    device->internalLibrary = [device->device newLibraryWithSource:[NSString stringWithUTF8String:kAGFXICBConvertSource]
                                                          options:nil
                                                            error:&internalLibraryError];
    if (!device->internalLibrary) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxDeviceCreate: failed to compile internal shader library: %s",
            internalLibraryError.localizedDescription.UTF8String);
    } else {
        static const char* kICBConvertEntryPoints[4] = {
            "icb_convert_draw",         // AGFX_INDIRECT_BUNDLE_TYPE_DRAW
            "icb_convert_draw_indexed", // AGFX_INDIRECT_BUNDLE_TYPE_DRAW_INDEXED
            "icb_convert_draw_mesh",    // AGFX_INDIRECT_BUNDLE_TYPE_DRAW_MESH
            "icb_convert_dispatch",     // AGFX_INDIRECT_BUNDLE_TYPE_DISPATCH
        };
        for (uint32_t i = 0; i < 4; ++i) {
            id<MTLFunction> function = [device->internalLibrary newFunctionWithName:[NSString stringWithUTF8String:kICBConvertEntryPoints[i]]];
            NSError* pipelineError = nil;
            device->icbConvertPipelines[i] = [device->device newComputePipelineStateWithFunction:function error:&pipelineError];
            if (!device->icbConvertPipelines[i]) {
                agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxDeviceCreate: failed to create %s pipeline: %s",
                    kICBConvertEntryPoints[i], pipelineError.localizedDescription.UTF8String);
            }
        }
    }

    return device;
}

void agfxDeviceDestroy(agfxDevice* device) {
    for (uint32_t i = 0; i < 4; ++i) {
        device->icbConvertPipelines[i] = nil;
    }
    device->internalLibrary = nil;
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
    NSError* heapError = nil;
    pool->heap = [device->device newCounterHeapWithDescriptor:desc error:&heapError];
    if (!pool->heap) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxQueryPoolCreate: newCounterHeapWithDescriptor failed: %s",
            heapError.localizedDescription.UTF8String);
        device->createInfo.free(pool);
        return nullptr;
    }
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
    if (!queue->commandQueue) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxCommandQueueCreate: newMTL4CommandQueue failed");
        device->createInfo.free(queue);
        return nullptr;
    }
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
    // The encoder currently open on this command buffer, if any. Barriers requested while a pass is
    // open must be encoded immediately: the tracker is otherwise only flushed at the next pass
    // begin, which would defer a mid-pass transition past the very work it is meant to order.
    id<MTL4CommandEncoder> currentEncoder;
    // The MTLStages the open encoder can encode barriers for; encoder barriers may only name these.
    MTLStages currentEncoderStages;

    agfxMetalLinearAllocator* topLevelArgBufferAllocator;
    agfxMetalLinearAllocator* drawArgumentAllocator;
    agfxMetalLinearAllocator* drawUniformAllocator;
};

agfxCommandBuffer* agfxCommandBufferCreate(agfxDevice* device, agfxCommandQueue* queue) {
    agfxCommandBuffer* commandBuffer = (agfxCommandBuffer*)device->createInfo.allocate(sizeof(agfxCommandBuffer));
    // The allocator hands back raw memory; the ARC-managed Metal object members must start nil.
    memset((void*)commandBuffer, 0, sizeof(agfxCommandBuffer));
    commandBuffer->device = device;
    commandBuffer->commandAllocator = [device->device newCommandAllocator];
    commandBuffer->commandBuffer = [device->device newCommandBuffer];
    if (!commandBuffer->commandAllocator || !commandBuffer->commandBuffer) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxCommandBufferCreate: newCommandAllocator or newCommandBuffer failed");
    }

    MTL4ArgumentTableDescriptor* renderArgTableDesc = [MTL4ArgumentTableDescriptor new];
    renderArgTableDesc.label = @"RenderArgumentTable";
    renderArgTableDesc.maxBufferBindCount = 31;
    renderArgTableDesc.maxTextureBindCount = 31;

    NSError* renderArgTableError = nil;
    commandBuffer->renderArgumentTable = [device->device newArgumentTableWithDescriptor:renderArgTableDesc error:&renderArgTableError];
    if (!commandBuffer->renderArgumentTable) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxCommandBufferCreate: newArgumentTableWithDescriptor (render) failed: %s",
            renderArgTableError.localizedDescription.UTF8String);
    }
    [commandBuffer->renderArgumentTable setAddress:device->bindlessManager->resourceHeapBuffer.gpuAddress atIndex:kIRDescriptorHeapBindPoint];
    [commandBuffer->renderArgumentTable setAddress:device->bindlessManager->samplerHeapBuffer.gpuAddress atIndex:kIRSamplerHeapBindPoint];

    MTL4ArgumentTableDescriptor* computeArgTableDesc = [MTL4ArgumentTableDescriptor new];
    computeArgTableDesc.label = @"ComputeArgumentTable";
    computeArgTableDesc.maxBufferBindCount = 31;
    computeArgTableDesc.maxTextureBindCount = 31;

    NSError* computeArgTableError = nil;
    commandBuffer->computeArgumentTable = [device->device newArgumentTableWithDescriptor:computeArgTableDesc error:&computeArgTableError];
    if (!commandBuffer->computeArgumentTable) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxCommandBufferCreate: newArgumentTableWithDescriptor (compute) failed: %s",
            computeArgTableError.localizedDescription.UTF8String);
    }
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
    if (commandBuffer->currentEncoder) {
        commandBuffer->barrierTracker.encodeInline(commandBuffer->currentEncoder, commandBuffer->currentEncoderStages);
    }
}

void agfxCommandBufferBufferBarrier(agfxCommandBuffer* commandBuffer, agfxBuffer* buffer, agfxResourceState oldState, agfxResourceState newState, agfxBool agglomerate) {
    if (!agglomerate) return;
    commandBuffer->barrierTracker.addBarrier(oldState, newState);
    if (commandBuffer->currentEncoder) {
        commandBuffer->barrierTracker.encodeInline(commandBuffer->currentEncoder, commandBuffer->currentEncoderStages);
    }
}

void agfxCommandBufferAccelerationStructureBarrier(agfxCommandBuffer* commandBuffer, agfxAccelerationStructure* accelerationStructure, agfxResourceState oldState, agfxResourceState newState, agfxBool agglomerate) {
    if (!agglomerate) return;
    commandBuffer->barrierTracker.addBarrier(oldState, newState);
    if (commandBuffer->currentEncoder) {
        commandBuffer->barrierTracker.encodeInline(commandBuffer->currentEncoder, commandBuffer->currentEncoderStages);
    }
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
    if (!texture->texture) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxTextureCreate: newTextureWithDescriptor failed");
        device->createInfo.free(texture);
        return nullptr;
    }
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
    commandBuffer->currentEncoder = computePass->encoder;
    commandBuffer->currentEncoderStages = MTLStageDispatch | MTLStageBlit | MTLStageAccelerationStructure;

    [computePass->encoder setArgumentTable:commandBuffer->computeArgumentTable];

    return computePass;
}

void agfxComputePassEnd(agfxComputePass* computePass) {
    [computePass->encoder endEncoding];
    computePass->commandBuffer->currentEncoder = nil;
    computePass->encoder = nil;
    computePass->device->createInfo.tempFree(computePass);
}

void agfxComputePassTextureUAVBarrier(agfxComputePass* computePass, agfxTexture* texture) {
    // Within one encoder, so this is an encoder barrier: afterQueueStages explicitly excludes the
    // current encoder and would not order this against the dispatch that just wrote the resource.
    [computePass->encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
}

void agfxComputePassBufferUAVBarrier(agfxComputePass* computePass, agfxBuffer* buffer) {
    // See agfxComputePassTextureUAVBarrier: intra-encoder ordering needs the encoder barrier.
    [computePass->encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
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
    if (!buffer->buffer) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxBufferCreate: newBufferWithLength failed");
        device->createInfo.free(buffer);
        return nullptr;
    }
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
    if (!sampler->samplerState) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxSamplerCreate: newSamplerStateWithDescriptor failed");
        device->createInfo.free(sampler);
        return nullptr;
    }
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
        descriptor.depthAttachment.clearDepth = createInfo->depthAttachment.clearDepth;
        descriptor.depthAttachment.loadAction = agfxLoadActionToMTL(createInfo->depthAttachment.loadOp);
        descriptor.depthAttachment.storeAction = agfxStoreActionToMTL(createInfo->depthAttachment.storeOp);
        descriptor.depthAttachment.texture = createInfo->depthAttachment.renderTarget->view;
    }

    renderPass->encoder = [cmdBuffer->commandBuffer renderCommandEncoderWithDescriptor:descriptor];
    renderPass->encoder.label = [NSString stringWithUTF8String:createInfo->name];

    [renderPass->encoder setArgumentTable:cmdBuffer->renderArgumentTable atStages:MTLStageVertex | MTLStageFragment | MTLStageObject | MTLStageMesh];

    cmdBuffer->barrierTracker.encode(renderPass->encoder);
    cmdBuffer->currentEncoder = renderPass->encoder;
    cmdBuffer->currentEncoderStages = MTLStageVertex | MTLStageFragment | MTLStageTile | MTLStageObject | MTLStageMesh;
    return renderPass;
}

void agfxRenderPassEnd(agfxRenderPass* renderPass) {
    [renderPass->encoder endEncoding];
    renderPass->cmdBuffer->currentEncoder = nil;
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
    if (!swapChain->currentDrawable) {
        agfxLog(swapChain->device, AGFX_LOG_SEVERITY_WARNING, "agfxSwapChainAcquireNextTexture: nextDrawable returned nil");
        return nullptr;
    }

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
    NSError* libraryError = nil;
    id<MTLLibrary> library = [device->device newLibraryWithData:data error:&libraryError];
    if (!library) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxShaderModuleCreate: newLibraryWithData failed: %s",
            libraryError.localizedDescription.UTF8String);
        device->createInfo.free(shaderModule);
        return nullptr;
    }
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:createInfo->entryPoint]];
    if (!function) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxShaderModuleCreate: newFunctionWithName failed for entry point \"%s\"",
            createInfo->entryPoint);
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

        NSError* pipelineError = nil;
        pipeline->pipelineState = [device->device newRenderPipelineStateWithMeshDescriptor:meshDescriptor options:MTLPipelineOptionNone reflection:nil error:&pipelineError];
        if (!pipeline->pipelineState) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxRenderPipelineCreate: newRenderPipelineStateWithMeshDescriptor failed: %s",
                pipelineError.localizedDescription.UTF8String);
            device->createInfo.free(pipeline);
            return nullptr;
        }
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

        NSError* pipelineError = nil;
        pipeline->pipelineState = [device->device newRenderPipelineStateWithDescriptor:descriptor error:&pipelineError];
        if (!pipeline->pipelineState) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxRenderPipelineCreate: newRenderPipelineStateWithDescriptor failed: %s",
                pipelineError.localizedDescription.UTF8String);
            device->createInfo.free(pipeline);
            return nullptr;
        }
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
    // agfxComputePipelineCreateInfo has no supportsIndirect flag (unlike the render one), and a
    // compute PSO used from an AGFX_INDIRECT_BUNDLE_TYPE_DISPATCH ICB must declare support, so it is
    // always on. Compute PSOs are cheap enough that the blanket cost isn't worth an API change.
    descriptor.supportIndirectCommandBuffers = YES;

    agfxComputePipeline* pipeline = (agfxComputePipeline*)device->createInfo.allocate(sizeof(agfxComputePipeline));
    memcpy(&pipeline->createInfo, createInfo, sizeof(agfxComputePipelineCreateInfo));
    NSError* pipelineError = nil;
    pipeline->pipelineState = [device->device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&pipelineError];
    if (!pipeline->pipelineState) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxComputePipelineCreate: newComputePipelineStateWithDescriptor failed: %s",
            pipelineError.localizedDescription.UTF8String);
        device->createInfo.free(pipeline);
        return nullptr;
    }
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
        if (!accelerationStructure->accelerationStructure) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxAccelerationStructureCreate: newAccelerationStructureWithSize failed (bottom level)");
        }
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
        if (!accelerationStructure->accelerationStructure) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxAccelerationStructureCreate: newAccelerationStructureWithSize failed (top level)");
        }
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
    if (!accelerationStructure->accelerationStructure) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxAccelerationStructureCreateCompacted: newAccelerationStructureWithSize failed");
    }
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

// Indirect bundle
struct agfxIndirectBundle {
    agfxIndirectBundleCreateInfo createInfo;
    agfxBuffer* commandsBuffer;
    agfxBufferView* commandsBufferView;
    agfxBuffer* countBuffer;
    agfxBufferView* countBufferView;
    uint32_t stride;

    // Metal-only: there is no ExecuteIndirect, so draws are pre-encoded into an ICB by the
    // conversion kernel, which also needs somewhere to put the per-command bindings those commands
    // reference (ICB commands inherit no buffer state).
    id<MTLIndirectCommandBuffer> icb;
    id<MTLBuffer> tlabSlices;       // agfxTLAB per command
    id<MTLBuffer> drawArgsSlices;   // agfxICBDrawArgsSlice per command
    id<MTLBuffer> execRanges;       // MTLIndirectCommandBufferExecutionRange per count slot
    id<MTLBuffer> uniforms;         // draw-type tag, shared by every command in the bundle
    id<MTLBuffer> icbContainer;     // argument buffer holding icb.gpuResourceID, the only way a kernel can reach an ICB
};

agfxIndirectBundle* agfxIndirectBundleCreate(agfxDevice* device, const agfxIndirectBundleCreateInfo* createInfo) {
    agfxIndirectBundle* bundle = (agfxIndirectBundle*)device->createInfo.allocate(sizeof(agfxIndirectBundle));
    // The allocator hands back raw memory; the ARC-managed Metal object members must start nil.
    memset((void*)bundle, 0, sizeof(agfxIndirectBundle));
    memcpy(&bundle->createInfo, createInfo, sizeof(agfxIndirectBundleCreateInfo));
    bundle->stride = agfxIndirectBundleTypeStride(createInfo->type);

    agfxBufferCreateInfo commandsBufferInfo = {};
    commandsBufferInfo.size = (uint64_t)createInfo->maxCommandCount * bundle->stride;
    commandsBufferInfo.stride = bundle->stride;
    commandsBufferInfo.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
    commandsBufferInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    bundle->commandsBuffer = agfxBufferCreate(device, &commandsBufferInfo);

    agfxBufferCreateInfo countBufferInfo = {};
    countBufferInfo.size = (uint64_t)createInfo->maxCountCount * sizeof(uint32_t);
    countBufferInfo.stride = sizeof(uint32_t);
    countBufferInfo.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
    countBufferInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    bundle->countBuffer = agfxBufferCreate(device, &countBufferInfo);

    agfxBufferViewCreateInfo commandsViewInfo = {};
    commandsViewInfo.buffer = bundle->commandsBuffer;
    commandsViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    commandsViewInfo.writeable = true;
    bundle->commandsBufferView = agfxBufferViewCreate(device, &commandsViewInfo);

    agfxBufferViewCreateInfo countViewInfo = {};
    countViewInfo.buffer = bundle->countBuffer;
    countViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    countViewInfo.writeable = true;
    bundle->countBufferView = agfxBufferViewCreate(device, &countViewInfo);

    MTLIndirectCommandBufferDescriptor* icbDesc = [MTLIndirectCommandBufferDescriptor new];
    switch (createInfo->type) {
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW:
            icbDesc.commandTypes = MTLIndirectCommandTypeDraw;
            break;
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW_INDEXED:
            icbDesc.commandTypes = MTLIndirectCommandTypeDrawIndexed;
            break;
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW_MESH:
            icbDesc.commandTypes = MTLIndirectCommandTypeDrawMeshThreadgroups;
            break;
        case AGFX_INDIRECT_BUNDLE_TYPE_DISPATCH:
            icbDesc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
            break;
    }
    // Commands are fully self-contained (the conversion kernel rebinds everything per command), but
    // the pipeline is uniform across an execute call and is set on the encoder instead.
    icbDesc.inheritBuffers = NO;
    icbDesc.inheritPipelineState = YES;
    icbDesc.maxVertexBufferBindCount = 8;
    icbDesc.maxFragmentBufferBindCount = 8;
    icbDesc.maxObjectBufferBindCount = 8;
    icbDesc.maxMeshBufferBindCount = 8;
    icbDesc.maxKernelBufferBindCount = 8;

    bundle->icb = [device->device newIndirectCommandBufferWithDescriptor:icbDesc
                                                         maxCommandCount:createInfo->maxCommandCount
                                                                 options:MTLResourceStorageModePrivate];
    if (!bundle->icb) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxIndirectBundleCreate: newIndirectCommandBufferWithDescriptor failed");
    }
    [device->residencySet addAllocation:bundle->icb];

    bundle->icbContainer = [device->device newBufferWithLength:sizeof(MTLResourceID) options:MTLResourceStorageModeShared];
    *(MTLResourceID*)bundle->icbContainer.contents = bundle->icb.gpuResourceID;
    [device->residencySet addAllocation:bundle->icbContainer];

    bundle->tlabSlices = [device->device newBufferWithLength:(uint64_t)createInfo->maxCommandCount * sizeof(agfxTLAB)
                                                    options:MTLResourceStorageModePrivate];
    [device->residencySet addAllocation:bundle->tlabSlices];

    bundle->drawArgsSlices = [device->device newBufferWithLength:(uint64_t)createInfo->maxCommandCount * sizeof(agfxICBDrawArgsSlice)
                                                        options:MTLResourceStorageModePrivate];
    [device->residencySet addAllocation:bundle->drawArgsSlices];

    bundle->execRanges = [device->device newBufferWithLength:(uint64_t)createInfo->maxCountCount * sizeof(MTLIndirectCommandBufferExecutionRange)
                                                     options:MTLResourceStorageModePrivate];
    [device->residencySet addAllocation:bundle->execRanges];

    // The draw-type tag Metal Shader Converter's vertex prologue reads to reconstruct SV_VertexID /
    // SV_InstanceID. Constant for the whole bundle, so it is filled once here rather than per
    // conversion; index type is patched in PrepareIndirectBundle once the index buffer is known.
    bundle->uniforms = [device->device newBufferWithLength:sizeof(uint16_t) options:MTLResourceStorageModeShared];
    *(uint16_t*)bundle->uniforms.contents = kIRNonIndexedDraw;
    [device->residencySet addAllocation:bundle->uniforms];

    return bundle;
}

void agfxIndirectBundleDestroy(agfxDevice* device, agfxIndirectBundle* bundle) {
    if (bundle->icbContainer) [device->residencySet removeAllocation:bundle->icbContainer];
    if (bundle->uniforms) [device->residencySet removeAllocation:bundle->uniforms];
    if (bundle->execRanges) [device->residencySet removeAllocation:bundle->execRanges];
    if (bundle->drawArgsSlices) [device->residencySet removeAllocation:bundle->drawArgsSlices];
    if (bundle->tlabSlices) [device->residencySet removeAllocation:bundle->tlabSlices];
    if (bundle->icb) [device->residencySet removeAllocation:bundle->icb];
    bundle->icbContainer = nil;
    bundle->uniforms = nil;
    bundle->execRanges = nil;
    bundle->drawArgsSlices = nil;
    bundle->tlabSlices = nil;
    bundle->icb = nil;

    if (bundle->commandsBufferView) agfxBufferViewDestroy(device, bundle->commandsBufferView);
    if (bundle->commandsBuffer) agfxBufferDestroy(device, bundle->commandsBuffer);
    if (bundle->countBufferView) agfxBufferViewDestroy(device, bundle->countBufferView);
    if (bundle->countBuffer) agfxBufferDestroy(device, bundle->countBuffer);
    device->createInfo.free(bundle);
}

uint64_t agfxIndirectBundleGetHandle(agfxIndirectBundle* bundle) {
    return ((uint64_t)agfxBufferViewGetHandle(bundle->countBufferView) << 32) | agfxBufferViewGetHandle(bundle->commandsBufferView);
}

agfxBuffer* agfxIndirectBundleGetCommandsBuffer(agfxIndirectBundle* bundle) {
    return bundle->commandsBuffer;
}

agfxBuffer* agfxIndirectBundleGetCountBuffer(agfxIndirectBundle* bundle) {
    return bundle->countBuffer;
}

void agfxComputePassPrepareIndirectBundle(agfxComputePass* computePass, agfxIndirectBundle* bundle, const agfxIndirectBundleExecuteInfo* executeInfo) {
    agfxDevice* device = computePass->device;
    agfxCommandBuffer* commandBuffer = computePass->commandBuffer;
    agfxIndirectBundleType type = bundle->createInfo.type;

    id<MTLComputePipelineState> convertPipeline = device->icbConvertPipelines[type];
    if (!convertPipeline || executeInfo->maxCommandCount == 0) return;

    agfxICBConvertParams params = {};
    params.commandOffset = executeInfo->commandOffset;
    params.countIndex = executeInfo->countIndex;
    params.maxCommandCount = executeInfo->maxCommandCount;

    if (type == AGFX_INDIRECT_BUNDLE_TYPE_DISPATCH) {
        const agfxComputePipelineCreateInfo& info = executeInfo->computePipeline->createInfo;
        params.threadsPerGroup[0] = info.groupSizeX;
        params.threadsPerGroup[1] = info.groupSizeY;
        params.threadsPerGroup[2] = info.groupSizeZ;
    } else {
        const agfxRenderPipelineCreateInfo& info = executeInfo->renderPipeline->createInfo;
        params.primitiveType = (uint32_t)agfxPrimitiveTypeToMTL(info.topology);
        params.objectThreadsPerGroup[0] = info.taskGroupSizeX;
        params.objectThreadsPerGroup[1] = info.taskGroupSizeY;
        params.objectThreadsPerGroup[2] = info.taskGroupSizeZ;
        params.meshThreadsPerGroup[0] = info.meshGroupSizeX;
        params.meshThreadsPerGroup[1] = info.meshGroupSizeY;
        params.meshThreadsPerGroup[2] = info.meshGroupSizeZ;
    }

    if (type == AGFX_INDIRECT_BUNDLE_TYPE_DRAW_INDEXED && executeInfo->indexBuffer) {
        MTLIndexType indexType = executeInfo->indexBuffer->createInfo.stride == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
        params.use16BitIndices = (indexType == MTLIndexTypeUInt16) ? 1 : 0;
        params.indexStride = executeInfo->indexBuffer->createInfo.stride;
        // Same +1 encoding the direct DrawIndexed path uses (kIRNonIndexedDraw is 0).
        *(uint16_t*)bundle->uniforms.contents = (uint16_t)(indexType + 1);
    }

    agfxMetalAllocation paramsAlloc = commandBuffer->drawArgumentAllocator->allocate(sizeof(agfxICBConvertParams));
    memcpy(paramsAlloc.cpuPointer, &params, sizeof(agfxICBConvertParams));

    agfxMetalAllocation pushConstantsAlloc = commandBuffer->drawUniformAllocator->allocate(sizeof(executeInfo->pushConstants));
    memcpy(pushConstantsAlloc.cpuPointer, executeInfo->pushConstants, sizeof(executeInfo->pushConstants));

    id<MTL4ArgumentTable> argumentTable = commandBuffer->computeArgumentTable;
    [argumentTable setAddress:paramsAlloc.gpuAddress atIndex:kAGFXICBParamsBindPoint];
    [argumentTable setAddress:pushConstantsAlloc.gpuAddress atIndex:kAGFXICBPushConstantsBindPoint];
    [argumentTable setAddress:bundle->commandsBuffer->buffer.gpuAddress atIndex:kAGFXICBCommandsBindPoint];
    [argumentTable setAddress:bundle->countBuffer->buffer.gpuAddress atIndex:kAGFXICBCountBindPoint];
    [argumentTable setAddress:bundle->tlabSlices.gpuAddress atIndex:kAGFXICBTLABBindPoint];
    [argumentTable setAddress:bundle->drawArgsSlices.gpuAddress atIndex:kAGFXICBDrawArgsBindPoint];
    [argumentTable setAddress:bundle->execRanges.gpuAddress atIndex:kAGFXICBExecRangeBindPoint];
    [argumentTable setAddress:bundle->uniforms.gpuAddress atIndex:kAGFXICBUniformsBindPoint];
    if (executeInfo->indexBuffer) {
        [argumentTable setAddress:executeInfo->indexBuffer->buffer.gpuAddress atIndex:kAGFXICBIndexBufferBindPoint];
    }
    [argumentTable setAddress:bundle->icbContainer.gpuAddress atIndex:kAGFXICBTargetBindPoint];

    static const uint32_t kThreadsPerGroup = 64;
    uint32_t groupCount = (executeInfo->maxCommandCount + kThreadsPerGroup - 1) / kThreadsPerGroup;

    [computePass->encoder setComputePipelineState:convertPipeline];
    [computePass->encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];

    // The caller's pipeline is no longer bound; force the next Dispatch to rebind rather than
    // silently dispatching the conversion kernel again.
    computePass->currentPipeline = nullptr;
}

void agfxRenderPassExecuteIndirectBundle(agfxRenderPass* renderPass, agfxIndirectBundle* bundle, const agfxIndirectBundleExecuteInfo* executeInfo) {
    agfxRenderPassSetPipeline(renderPass, executeInfo->renderPipeline);

    [renderPass->encoder executeCommandsInBuffer:bundle->icb
                                  indirectBuffer:bundle->execRanges.gpuAddress + (uint64_t)executeInfo->countIndex * sizeof(MTLIndirectCommandBufferExecutionRange)];
}

void agfxComputePassExecuteIndirectBundle(agfxComputePass* computePass, agfxIndirectBundle* bundle, const agfxIndirectBundleExecuteInfo* executeInfo) {
    agfxComputePassSetPipeline(computePass, executeInfo->computePipeline);

    [computePass->encoder executeCommandsInBuffer:bundle->icb
                                   indirectBuffer:bundle->execRanges.gpuAddress + (uint64_t)executeInfo->countIndex * sizeof(MTLIndirectCommandBufferExecutionRange)];
}
