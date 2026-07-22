/**
 * @Author: Amélie Heinrich
 * @Create Time: 2026-07-19 09:46:06
 * @Copyright: Copyright (c) 2026 Moonscorched Productions. All rights reserved.
 */

#include "agfx.h"

#include <cstring>
#include <cstdio>
#include <cstdarg>
#include <vector>
#include <string>
#include <new>

#include <dxgi1_6.h>
#include "d3dx12/d3d12.h"
#include "d3dx12/d3dx12.h"

// Enum stuff
DXGI_FORMAT agfxTextureFormatToDXGIFormat(agfxTextureFormat format);
D3D12_RESOURCE_DIMENSION agfxTextureTypeToD3D12ResourceDimension(agfxTextureType type);
D3D12_COMMAND_LIST_TYPE agfxCommandQueueTypeToD3D12CommandListType(agfxCommandQueueType type);
D3D12_RESOURCE_STATES agfxResourceStateToD3D12ResourceStates(agfxResourceState state);
D3D12_RESOURCE_FLAGS agfxTextureUsageToD3D12ResourceFlags(agfxTextureUsage usage);
D3D12_RESOURCE_FLAGS agfxBufferUsageToD3D12ResourceFlags(agfxBufferUsage usage);
D3D12_HEAP_TYPE agfxBufferMemoryTypeToD3D12HeapType(agfxBufferMemoryType memoryType);
D3D12_SHADER_RESOURCE_VIEW_DESC agfxTextureViewTypeToD3D12ShaderResourceViewDesc(agfxTextureViewCreateInfo* createInfo);
D3D12_UNORDERED_ACCESS_VIEW_DESC agfxTextureViewTypeToD3D12UnorderedAccessViewDesc(agfxTextureViewCreateInfo* createInfo);
D3D12_SHADER_RESOURCE_VIEW_DESC agfxBufferViewTypeToD3D12ShaderResourceViewDesc(agfxBufferViewCreateInfo* createInfo);
D3D12_UNORDERED_ACCESS_VIEW_DESC agfxBufferViewTypeToD3D12UnorderedAccessViewDesc(agfxBufferViewCreateInfo* createInfo);
D3D12_CONSTANT_BUFFER_VIEW_DESC agfxBufferViewTypeToD3D12ConstantBufferViewDesc(agfxBufferViewCreateInfo* createInfo);
D3D12_SAMPLER_DESC agfxSamplerCreateInfoToD3D12SamplerDesc(agfxSamplerCreateInfo* createInfo);
D3D12_COMPARISON_FUNC agfxComparisonFunctionToD3D12ComparisonFunc(agfxComparisonFunction func);
D3D12_FILTER agfxSamplerFilterToD3D12Filter(agfxSamplerFilter filter, bool isComparison);
D3D12_TEXTURE_ADDRESS_MODE agfxSamplerAddressModeToD3D12TextureAddressMode(agfxSamplerAddressMode mode);
D3D12_RENDER_TARGET_VIEW_DESC agfxTextureViewTypeToD3D12RenderTargetViewDesc(agfxRenderTargetCreateInfo* createInfo);
D3D12_DEPTH_STENCIL_VIEW_DESC agfxTextureViewTypeToD3D12DepthStencilViewDesc(agfxRenderTargetCreateInfo* createInfo);
D3D12_PRIMITIVE_TOPOLOGY_TYPE agfxTopologyToD3D12PrimitiveTopologyType(agfxTopology topology);
D3D12_CULL_MODE agfxCullModeToD3D12CullMode(agfxCullMode mode);
D3D12_FILL_MODE agfxFillModeToD3D12FillMode(agfxFillMode mode);
BOOL agfxFrontFaceToD3D12FrontCounterClockwise(agfxFrontFace face);
D3D12_BLEND agfxBlendFactorToD3D12Blend(agfxBlendFactor factor);
D3D12_BLEND_OP agfxBlendOperationToD3D12BlendOp(agfxBlendOperation op);
D3D12_PRIMITIVE_TOPOLOGY agfxTopologyToD3D12PrimitiveTopology(agfxTopology topology);

// MSVC/clang-cl return small COM structs (e.g. D3D12_CPU_DESCRIPTOR_HANDLE) directly from
// GetCPUDescriptorHandleForHeapStart(); mingw/GCC's COM vtable calling convention does not
// reliably match that ABI for this method, so route through the explicit out-param form there.
static inline D3D12_CPU_DESCRIPTOR_HANDLE agfxGetCPUDescriptorHandleForHeapStart(ID3D12DescriptorHeap* heap) {
#if defined(_MSC_VER) && !defined(__clang__)
    return heap->GetCPUDescriptorHandleForHeapStart();
#else
    D3D12_CPU_DESCRIPTOR_HANDLE handle;
    heap->GetCPUDescriptorHandleForHeapStart(&handle);
    return handle;
#endif
}

struct agfxCommandBuffer {
    ID3D12GraphicsCommandList10* d3d12CommandList;
    ID3D12CommandAllocator* d3d12CommandAllocator;
    bool isRecording;
    agfxDevice* device;
    agfxCommandQueueType queueType;
};

struct agfxTexture {
    ID3D12Resource* d3d12Resource;
    agfxTextureCreateInfo createInfo;
};

struct agfxBuffer {
    ID3D12Resource* d3d12Resource;
    agfxBufferCreateInfo createInfo;
};

struct agfxCommandQueue {
    ID3D12CommandQueue* d3d12CommandQueue;
    agfxCommandQueueType type;
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

struct agfxDescriptorAllocation {
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle;
    uint64_t index;
};

struct agfxDescriptorManager {
    const int resourceCount = 1'000'000;
    const int samplerCount = 2048;
    const int rtvCount = 2048;
    const int dsvCount = 2048;

    agfxDescriptorManager(ID3D12Device7* inDevice)
        : resourceSlotAllocator(resourceCount),
          samplerSlotAllocator(samplerCount),
          rtvSlotAllocator(rtvCount),
          dsvSlotAllocator(dsvCount),
          device(inDevice) {
        D3D12_DESCRIPTOR_HEAP_DESC resourceHeapDesc = {};
        resourceHeapDesc.NumDescriptors = resourceCount;
        resourceHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        resourceHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

        D3D12_DESCRIPTOR_HEAP_DESC samplerHeapDesc = {};
        samplerHeapDesc.NumDescriptors = samplerCount;
        samplerHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        samplerHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

        D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
        rtvHeapDesc.NumDescriptors = rtvCount;
        rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

        D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
        dsvHeapDesc.NumDescriptors = dsvCount;
        dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

        inDevice->CreateDescriptorHeap(&resourceHeapDesc, IID_PPV_ARGS(&resourceHeap));
        inDevice->CreateDescriptorHeap(&samplerHeapDesc, IID_PPV_ARGS(&samplerHeap));
        inDevice->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap));
        inDevice->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&dsvHeap));
    }

    ~agfxDescriptorManager() {
        if (resourceHeap) resourceHeap->Release();
        if (samplerHeap) samplerHeap->Release();
        if (rtvHeap) rtvHeap->Release();
        if (dsvHeap) dsvHeap->Release();
    }

    agfxDescriptorAllocation writeSRV(ID3D12Resource *resource, D3D12_SHADER_RESOURCE_VIEW_DESC *srvDesc) {
        uint64_t slot = resourceSlotAllocator.allocate();
        if (slot == UINT64_MAX) return { {0}, {0}, UINT32_MAX };

        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = agfxGetCPUDescriptorHandleForHeapStart(resourceHeap);
        cpuHandle.ptr += slot * device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        device->CreateShaderResourceView(resource, srvDesc, cpuHandle);
        return { cpuHandle, {0}, static_cast<UINT32>(slot) };
    }

    agfxDescriptorAllocation writeUAV(ID3D12Resource *resource, D3D12_UNORDERED_ACCESS_VIEW_DESC *uavDesc) {
        uint64_t slot = resourceSlotAllocator.allocate();
        if (slot == UINT64_MAX) return { {0}, {0}, UINT32_MAX };

        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = agfxGetCPUDescriptorHandleForHeapStart(resourceHeap);
        cpuHandle.ptr += slot * device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        device->CreateUnorderedAccessView(resource, nullptr, uavDesc, cpuHandle);
        return { cpuHandle, {0}, static_cast<UINT32>(slot) };
    }

    agfxDescriptorAllocation writeCBV(D3D12_CONSTANT_BUFFER_VIEW_DESC *cbvDesc) {
        uint64_t slot = resourceSlotAllocator.allocate();
        if (slot == UINT64_MAX) return { {0}, {0}, UINT32_MAX };

        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = agfxGetCPUDescriptorHandleForHeapStart(resourceHeap);
        cpuHandle.ptr += slot * device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        device->CreateConstantBufferView(cbvDesc, cpuHandle);
        return { cpuHandle, {0}, static_cast<UINT32>(slot) };
    }

    agfxDescriptorAllocation writeSampler(D3D12_SAMPLER_DESC *samplerDesc) {
        uint64_t slot = samplerSlotAllocator.allocate();
        if (slot == UINT64_MAX) return { {0}, {0}, UINT32_MAX };

        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = agfxGetCPUDescriptorHandleForHeapStart(samplerHeap);
        cpuHandle.ptr += slot * device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        device->CreateSampler(samplerDesc, cpuHandle);
        return { cpuHandle, {0}, static_cast<UINT32>(slot) };
    }

    agfxDescriptorAllocation writeRTV(ID3D12Resource *resource, D3D12_RENDER_TARGET_VIEW_DESC *rtvDesc) {
        uint64_t slot = rtvSlotAllocator.allocate();
        if (slot == UINT64_MAX) return { {0}, {0}, UINT32_MAX };

        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = agfxGetCPUDescriptorHandleForHeapStart(rtvHeap);
        cpuHandle.ptr += slot * device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        device->CreateRenderTargetView(resource, rtvDesc, cpuHandle);
        return { cpuHandle, {0}, static_cast<UINT32>(slot) };
    }

    agfxDescriptorAllocation writeDSV(ID3D12Resource *resource, D3D12_DEPTH_STENCIL_VIEW_DESC *dsvDesc) {
        uint64_t slot = dsvSlotAllocator.allocate();
        if (slot == UINT64_MAX) return { {0}, {0}, UINT32_MAX };

        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = agfxGetCPUDescriptorHandleForHeapStart(dsvHeap);
        cpuHandle.ptr += slot * device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        device->CreateDepthStencilView(resource, dsvDesc, cpuHandle);
        return { cpuHandle, {0}, static_cast<UINT32>(slot) };
    }

    void freeRTVSlot(uint32_t slot) {
        rtvSlotAllocator.free(slot);
    }

    void freeDSVSlot(uint32_t slot) {
        dsvSlotAllocator.free(slot);
    }

    void freeSamplerSlot(uint32_t slot) {
        samplerSlotAllocator.free(slot);
    }

    void agfxFreeResourceSlot(uint32_t slot) {
        resourceSlotAllocator.free(slot);
    }

    ID3D12Device7* device;
    ID3D12DescriptorHeap* resourceHeap;
    ID3D12DescriptorHeap* samplerHeap;
    ID3D12DescriptorHeap* rtvHeap;
    ID3D12DescriptorHeap* dsvHeap;
    agfxSlotAllocator resourceSlotAllocator;
    agfxSlotAllocator samplerSlotAllocator;
    agfxSlotAllocator rtvSlotAllocator;
    agfxSlotAllocator dsvSlotAllocator;
};

struct agfxAccelerationStructure {
    agfxAccelerationStructureCreateInfo createInfo;
    ID3D12Resource* d3d12Resource = nullptr;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
    agfxDescriptorAllocation descriptor = { {0}, {0}, UINT64_MAX }; // Bindless SRV slot, top level only

    // Bottom level
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometryDescs;

    // Top level
    ID3D12Resource* instanceBuffer = nullptr;
    D3D12_RAYTRACING_INSTANCE_DESC* mappedInstanceBuffer = nullptr;
    uint32_t currentInstanceCount = 0;
};

// device

struct agfxDevice {
    agfxDeviceCreateInfo createInfo;
    ID3D12Device7* d3d12Device;
    IDXGIFactory6* dxgiFactory;
    IDXGIAdapter4* dxgiAdapter;
    ID3D12Debug* d3d12Debug;
    ID3D12DebugDevice* d3d12DebugDevice;
    ID3D12InfoQueue* d3d12InfoQueue;

    ID3D12RootSignature* globalRootSignature;
    ID3D12CommandSignature* indirectSignatures[4]; // indexed by agfxIndirectBundleType

    agfxDescriptorManager* descriptorManager;
};

static uint32_t agfxIndirectBundleTypeStride(agfxIndirectBundleType type) {
    switch (type) {
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW: return sizeof(agfxDrawCommand);
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW_INDEXED: return sizeof(agfxDrawIndexedCommand);
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW_MESH: return sizeof(agfxDrawMeshCommand);
        case AGFX_INDIRECT_BUNDLE_TYPE_DISPATCH: return sizeof(agfxDispatchCommand);
        default: return 0;
    }
}

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
    memset(device, 0, sizeof(agfxDevice));
    memcpy(&device->createInfo, createInfo, sizeof(agfxDeviceCreateInfo));

    UINT dxgiFactoryFlags = 0;
    if (createInfo->enableValidation) {
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&device->d3d12Debug)))) {
            device->d3d12Debug->EnableDebugLayer();
        } else {
            agfxLog(device, AGFX_LOG_SEVERITY_WARNING, "agfxDeviceCreate: D3D12GetDebugInterface failed, debug layer will not be enabled");
        }
        dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
    }

    if (FAILED(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&device->dxgiFactory)))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxDeviceCreate: CreateDXGIFactory2 failed");
        createInfo->free(device);
        return NULL;
    }

    // Walk adapters in high-performance order and pick the first one that can actually create a D3D12 device.
    IDXGIAdapter1* adapter1 = nullptr;
    for (UINT i = 0; device->dxgiFactory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter1)) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc = {};
        adapter1->GetDesc1(&desc);

        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(D3D12CreateDevice(adapter1, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
            break;
        }

        adapter1->Release();
        adapter1 = nullptr;
    }

    if (!adapter1) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxDeviceCreate: no suitable D3D12 adapter found");
        device->dxgiFactory->Release();
        createInfo->free(device);
        return NULL;
    }

    adapter1->QueryInterface(IID_PPV_ARGS(&device->dxgiAdapter));
    adapter1->Release();

    if (FAILED(D3D12CreateDevice(device->dxgiAdapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device->d3d12Device)))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxDeviceCreate: D3D12CreateDevice failed");
        device->dxgiAdapter->Release();
        device->dxgiFactory->Release();
        createInfo->free(device);
        return NULL;
    }

    if (createInfo->enableValidation) {
        if (FAILED(device->d3d12Device->QueryInterface(IID_PPV_ARGS(&device->d3d12DebugDevice)))) {
            agfxLog(device, AGFX_LOG_SEVERITY_WARNING, "agfxDeviceCreate: failed to query ID3D12DebugDevice");
        }

        D3D12_MESSAGE_SEVERITY supressSeverities[] = {
            D3D12_MESSAGE_SEVERITY_INFO,
            D3D12_MESSAGE_SEVERITY_WARNING
        };

        D3D12_MESSAGE_ID supressIDs[] = {
            D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
            D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE,
            D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,
            D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE,
        };

        D3D12_INFO_QUEUE_FILTER filter = { 0 };
        filter.DenyList.NumSeverities = ARRAYSIZE(supressSeverities);
        filter.DenyList.pSeverityList = supressSeverities;
        filter.DenyList.NumIDs = ARRAYSIZE(supressIDs);
        filter.DenyList.pIDList = supressIDs;

        if (SUCCEEDED(device->d3d12Device->QueryInterface(IID_PPV_ARGS(&device->d3d12InfoQueue)))) {
            device->d3d12InfoQueue->PushStorageFilter(&filter);
            device->d3d12InfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
            device->d3d12InfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
        } else {
            agfxLog(device, AGFX_LOG_SEVERITY_WARNING, "agfxDeviceCreate: failed to query ID3D12InfoQueue");
        }
    }

    void* descriptorManagerMemory = createInfo->allocate(sizeof(agfxDescriptorManager));
    device->descriptorManager = new (descriptorManagerMemory) agfxDescriptorManager(device->d3d12Device);

    D3D12_ROOT_PARAMETER rootParameters[2] = {};
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rootParameters[0].Constants.Num32BitValues = 128 / 4; // 128 bytes
    rootParameters[0].Constants.RegisterSpace = 0;
    rootParameters[0].Constants.ShaderRegister = 0;
    rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rootParameters[1].Constants.Num32BitValues = 1;
    rootParameters[1].Constants.RegisterSpace = 0;
    rootParameters[1].Constants.ShaderRegister = 1;

    D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
    rootSignatureDesc.NumParameters = 2;
    rootSignatureDesc.pParameters = rootParameters;
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    
    ID3DBlob* signatureBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signatureBlob, &errorBlob))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxDeviceCreate: D3D12SerializeRootSignature failed: %s",
            errorBlob ? (const char*)errorBlob->GetBufferPointer() : "<no error message>");
        if (signatureBlob) signatureBlob->Release();
        if (errorBlob) errorBlob->Release();
        device->descriptorManager->~agfxDescriptorManager();
        device->createInfo.free(device->descriptorManager);
        device->d3d12Device->Release();
        device->dxgiAdapter->Release();
        device->dxgiFactory->Release();
        createInfo->free(device);
        return NULL;
    }
    if (FAILED(device->d3d12Device->CreateRootSignature(0, signatureBlob->GetBufferPointer(), signatureBlob->GetBufferSize(), IID_PPV_ARGS(&device->globalRootSignature)))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxDeviceCreate: CreateRootSignature failed");
        signatureBlob->Release();
        if (errorBlob) errorBlob->Release();
        device->descriptorManager->~agfxDescriptorManager();
        device->createInfo.free(device->descriptorManager);
        device->d3d12Device->Release();
        device->dxgiAdapter->Release();
        device->dxgiFactory->Release();
        createInfo->free(device);
        return NULL;
    }
    signatureBlob->Release();
    if (errorBlob) errorBlob->Release();

    // Get device info to determine if mesh shading and ray tracing are supported, so we can create the indirect command signatures accordingly.
    agfxDeviceInfo deviceInfo = {};
    agfxDeviceGetInfo(device, &deviceInfo);

    // Indirect command signatures: one per bundle type. Argument 0 (when present) is the drawID
    // CONSTANT patch into root param 1, argument 1 is the terminal draw/dispatch stage - so the
    // argument buffer's per-command memory layout is [drawID, drawArguments...] (drawID leading),
    // matching agfxDrawCommand/agfxDrawIndexedCommand/agfxDrawMeshCommand's field order.
    auto createIndirectSignature = [&](agfxIndirectBundleType type, D3D12_INDIRECT_ARGUMENT_TYPE terminalType) {
        if (!deviceInfo.supportsMeshShaders && type == AGFX_INDIRECT_BUNDLE_TYPE_DRAW_MESH) {
            agfxLog(device, AGFX_LOG_SEVERITY_WARNING, "agfxDeviceCreate: mesh shading not supported, skipping indirect signature for DRAW_MESH");
            return;
        }

        uint32_t stride = agfxIndirectBundleTypeStride(type);

        D3D12_INDIRECT_ARGUMENT_DESC argumentDescs[2] = {};
        UINT argumentCount = 0;

        if (type != AGFX_INDIRECT_BUNDLE_TYPE_DISPATCH) {
            argumentDescs[argumentCount].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
            argumentDescs[argumentCount].Constant.RootParameterIndex = 1;
            argumentDescs[argumentCount].Constant.DestOffsetIn32BitValues = 0;
            argumentDescs[argumentCount].Constant.Num32BitValuesToSet = 1;
            argumentCount++;
        }
        argumentDescs[argumentCount].Type = terminalType;
        argumentCount++;

        D3D12_COMMAND_SIGNATURE_DESC signatureDesc = {};
        signatureDesc.pArgumentDescs = argumentDescs;
        signatureDesc.NumArgumentDescs = argumentCount;
        signatureDesc.ByteStride = stride;
        if (FAILED(device->d3d12Device->CreateCommandSignature(&signatureDesc, type != AGFX_INDIRECT_BUNDLE_TYPE_DISPATCH ? device->globalRootSignature : nullptr, IID_PPV_ARGS(&device->indirectSignatures[type])))) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxDeviceCreate: CreateCommandSignature failed for indirect bundle type %d", (int)type);
        }
    };

    createIndirectSignature(AGFX_INDIRECT_BUNDLE_TYPE_DRAW, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW);
    createIndirectSignature(AGFX_INDIRECT_BUNDLE_TYPE_DRAW_INDEXED, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED);
    createIndirectSignature(AGFX_INDIRECT_BUNDLE_TYPE_DRAW_MESH, D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH);
    createIndirectSignature(AGFX_INDIRECT_BUNDLE_TYPE_DISPATCH, D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH);

    return device;
}

void agfxDeviceDestroy(agfxDevice* device) {
    if (device->descriptorManager) {
        device->descriptorManager->~agfxDescriptorManager();
        device->createInfo.free(device->descriptorManager);
    }
    for (uint32_t i = 0; i < 4; ++i) {
        if (device->indirectSignatures[i]) device->indirectSignatures[i]->Release();
    }
    if (device->globalRootSignature) device->globalRootSignature->Release();
    if (device->d3d12InfoQueue) device->d3d12InfoQueue->Release();
    if (device->d3d12DebugDevice) device->d3d12DebugDevice->Release();
    if (device->d3d12Debug) device->d3d12Debug->Release();
    if (device->dxgiAdapter) device->dxgiAdapter->Release();
    if (device->dxgiFactory) device->dxgiFactory->Release();
    if (device->d3d12Device) device->d3d12Device->Release();
    device->createInfo.free(device);
}

void agfxDeviceGetInfo(agfxDevice* device, agfxDeviceInfo* info) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
    if (SUCCEEDED(device->d3d12Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5)))) {
        info->supportsRayTracing = options5.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
    } else {
        info->supportsRayTracing = 0;
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7 = {};
    if (SUCCEEDED(device->d3d12Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options7, sizeof(options7)))) {
        info->supportsMeshShaders = options7.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED;
    } else {
        info->supportsMeshShaders = 0;
    }

    info->supportsMultiDrawIndirect = 1; // D3D12 supports multi-draw indirect by default

    DXGI_ADAPTER_DESC3 adapterDesc = {};
    if (device->dxgiAdapter) {
        device->dxgiAdapter->GetDesc3(&adapterDesc);
        wcstombs(info->name, adapterDesc.Description, sizeof(info->name));
    } else {
        strcpy(info->name, "Unknown Adapter");
    }
}

void agfxDeviceMakeResourcesResident(agfxDevice* device) {} // Nothing to do here

// Fence

struct agfxFence {
    ID3D12Fence* d3d12Fence;
    HANDLE fenceEvent;
    uint64_t fenceValue;
};

agfxFence* agfxFenceCreate(agfxDevice* device) {
    agfxFence* fence = (agfxFence*)device->createInfo.allocate(sizeof(agfxFence));
    if (FAILED(device->d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence->d3d12Fence)))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxFenceCreate: CreateFence failed");
        device->createInfo.free(fence);
        return NULL;
    }
    fence->fenceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    fence->fenceValue = 0;
    return fence;
}

void agfxFenceDestroy(agfxDevice* device, agfxFence* fence) {
    if (fence->d3d12Fence) fence->d3d12Fence->Release();
    if (fence->fenceEvent) CloseHandle(fence->fenceEvent);
    device->createInfo.free(fence);
}

void agfxFenceWait(agfxFence* fence, uint64_t value, uint64_t timeout) {
    if (fence->d3d12Fence->GetCompletedValue() < value) {
        fence->d3d12Fence->SetEventOnCompletion(value, fence->fenceEvent);
        WaitForSingleObject(fence->fenceEvent, (DWORD)timeout);
    }
}

void agfxFenceSignal(agfxFence* fence, uint64_t value) {
    fence->d3d12Fence->Signal(value);
}

uint64_t agfxFenceGetCompletedValue(agfxFence* fence) {
    return fence->d3d12Fence->GetCompletedValue();
}

// Query pool

typedef struct agfxQueryPool {
    ID3D12QueryHeap* d3d12QueryHeap;
    ID3D12Resource* d3d12ReadbackBuffer;
    uint32_t count;
    uint64_t timestampFrequency;
} agfxQueryPool;

agfxQueryPool* agfxQueryPoolCreate(agfxDevice* device, agfxCommandQueue* queue, const agfxQueryPoolCreateInfo* createInfo) {
    D3D12_QUERY_HEAP_DESC queryHeapDesc = {};
    queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    queryHeapDesc.Count = createInfo->count;

    agfxQueryPool* pool = (agfxQueryPool*)device->createInfo.allocate(sizeof(agfxQueryPool));
    if (FAILED(device->d3d12Device->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(&pool->d3d12QueryHeap)))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxQueryPoolCreate: CreateQueryHeap failed");
        device->createInfo.free(pool);
        return NULL;
    }

    D3D12_HEAP_PROPERTIES heapProperties = {};
    heapProperties.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC readbackBufferDesc = {};
    readbackBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    readbackBufferDesc.Width = createInfo->count * sizeof(uint64_t);
    readbackBufferDesc.Height = 1;
    readbackBufferDesc.DepthOrArraySize = 1;
    readbackBufferDesc.MipLevels = 1;
    readbackBufferDesc.SampleDesc.Count = 1;
    readbackBufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    readbackBufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    readbackBufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    if (FAILED(device->d3d12Device->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &readbackBufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&pool->d3d12ReadbackBuffer)))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxQueryPoolCreate: CreateCommittedResource for readback buffer failed");
        pool->d3d12QueryHeap->Release();
        device->createInfo.free(pool);
        return NULL;
    }

    if (FAILED(queue->d3d12CommandQueue->GetTimestampFrequency(&pool->timestampFrequency))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxQueryPoolCreate: GetTimestampFrequency failed");
        pool->d3d12ReadbackBuffer->Release();
        pool->d3d12QueryHeap->Release();
        device->createInfo.free(pool);
        return NULL;
    }
    pool->count = createInfo->count;

    return pool;
}

void agfxQueryPoolDestroy(agfxDevice* device, agfxQueryPool* pool) {
    if (pool->d3d12ReadbackBuffer) pool->d3d12ReadbackBuffer->Release();
    if (pool->d3d12QueryHeap) pool->d3d12QueryHeap->Release();
    device->createInfo.free(pool);
}

void agfxCommandBufferWriteTimestamp(agfxCommandBuffer* commandBuffer, agfxQueryPool* pool, uint32_t index) {
    commandBuffer->d3d12CommandList->EndQuery(pool->d3d12QueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, index);
}

void agfxCommandBufferResolveQueryPool(agfxCommandBuffer* commandBuffer, agfxQueryPool* pool, uint32_t firstIndex, uint32_t count) {
    commandBuffer->d3d12CommandList->ResolveQueryData(pool->d3d12QueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, firstIndex, count, pool->d3d12ReadbackBuffer, firstIndex * sizeof(uint64_t));
}

void agfxQueryPoolReadback(agfxDevice* device, agfxQueryPool* pool, uint32_t firstIndex, uint32_t count, uint64_t* outTimestampsNanoseconds) {
    D3D12_RANGE readRange = { firstIndex * sizeof(uint64_t), (firstIndex + count) * sizeof(uint64_t) };
    void* mappedData = nullptr;
    if (SUCCEEDED(pool->d3d12ReadbackBuffer->Map(0, &readRange, &mappedData))) {
        const uint64_t* ticks = (const uint64_t*)((const uint8_t*)mappedData + firstIndex * sizeof(uint64_t));
        for (uint32_t i = 0; i < count; ++i) {
            outTimestampsNanoseconds[i] = (uint64_t)((double)ticks[i] * 1e9 / (double)pool->timestampFrequency);
        }

        D3D12_RANGE writtenRange = { 0, 0 }; // We did not write to this resource on the CPU.
        pool->d3d12ReadbackBuffer->Unmap(0, &writtenRange);
    }
}

// Command queue
agfxCommandQueue* agfxCommandQueueCreate(agfxDevice* device, const agfxCommandQueueCreateInfo* createInfo) {
    agfxCommandQueue* queue = (agfxCommandQueue*)device->createInfo.allocate(sizeof(agfxCommandQueue));
    queue->type = createInfo->type;

    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = agfxCommandQueueTypeToD3D12CommandListType(createInfo->type);

    if (FAILED(device->d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue->d3d12CommandQueue)))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxCommandQueueCreate: CreateCommandQueue failed");
        device->createInfo.free(queue);
        return NULL;
    }
    return queue;
}

void agfxCommandQueueDestroy(agfxDevice* device, agfxCommandQueue* queue) {
    if (queue->d3d12CommandQueue) queue->d3d12CommandQueue->Release();
}

void agfxCommandQueueSignal(agfxCommandQueue* queue, agfxFence* fence, uint64_t value) {
    queue->d3d12CommandQueue->Signal(fence->d3d12Fence, value);
}

void agfxCommandQueueWait(agfxCommandQueue* queue, agfxFence* fence, uint64_t value) {
    queue->d3d12CommandQueue->Wait(fence->d3d12Fence, value);
}

void agfxCommandQueueSubmit(agfxCommandQueue* queue, agfxCommandBuffer** commandBuffers, uint32_t commandBufferCount) {
    ID3D12GraphicsCommandList* maxLists[16];
    for (uint32_t i = 0; i < commandBufferCount && i < 16; ++i) {
        maxLists[i] = commandBuffers[i]->d3d12CommandList;
    }
    queue->d3d12CommandQueue->ExecuteCommandLists(commandBufferCount, (ID3D12CommandList**)maxLists);
}

// Command buffer

agfxCommandBuffer* agfxCommandBufferCreate(agfxDevice* device, agfxCommandQueue* queue) {
    agfxCommandBuffer* commandBuffer = (agfxCommandBuffer*)device->createInfo.allocate(sizeof(agfxCommandBuffer));
    commandBuffer->isRecording = true;
    commandBuffer->device = device;
    commandBuffer->queueType = queue->type;

    if (FAILED(device->d3d12Device->CreateCommandAllocator(agfxCommandQueueTypeToD3D12CommandListType(queue->type), IID_PPV_ARGS(&commandBuffer->d3d12CommandAllocator)))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxCommandBufferCreate: CreateCommandAllocator failed");
        device->createInfo.free(commandBuffer);
        return NULL;
    }
    if (FAILED(device->d3d12Device->CreateCommandList(0, agfxCommandQueueTypeToD3D12CommandListType(queue->type), commandBuffer->d3d12CommandAllocator, nullptr, IID_PPV_ARGS(&commandBuffer->d3d12CommandList)))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxCommandBufferCreate: CreateCommandList failed");
        commandBuffer->d3d12CommandAllocator->Release();
        device->createInfo.free(commandBuffer);
        return NULL;
    }
    return commandBuffer;
}

void agfxCommandBufferDestroy(agfxDevice* device, agfxCommandBuffer* commandBuffer) {
    if (commandBuffer->d3d12CommandList) commandBuffer->d3d12CommandList->Release();
    if (commandBuffer->d3d12CommandAllocator) commandBuffer->d3d12CommandAllocator->Release();
    device->createInfo.free(commandBuffer);
}

void agfxCommandBufferReset(agfxCommandBuffer* commandBuffer) {
    if (commandBuffer->isRecording) {
        commandBuffer->d3d12CommandList->Close();
        commandBuffer->isRecording = false;
    }

    commandBuffer->d3d12CommandAllocator->Reset();
    commandBuffer->d3d12CommandList->Reset(commandBuffer->d3d12CommandAllocator, nullptr);
    commandBuffer->isRecording = true;
}

void agfxCommandBufferBegin(agfxCommandBuffer* commandBuffer) {
    if (!commandBuffer->isRecording) {
        commandBuffer->d3d12CommandList->Reset(commandBuffer->d3d12CommandAllocator, nullptr);
        commandBuffer->isRecording = true;
    }

    if (commandBuffer->queueType != AGFX_COMMAND_QUEUE_TYPE_TRANSFER) {
        ID3D12DescriptorHeap* heaps[] = { commandBuffer->device->descriptorManager->resourceHeap, commandBuffer->device->descriptorManager->samplerHeap };
        commandBuffer->d3d12CommandList->SetDescriptorHeaps(2, heaps);
    }
}

void agfxCommandBufferEnd(agfxCommandBuffer* commandBuffer) {
    commandBuffer->d3d12CommandList->Close();
    commandBuffer->isRecording = false;
}

void agfxCommandBufferTextureBarrier(agfxCommandBuffer* commandBuffer, agfxTexture* texture,  agfxResourceState oldState, agfxResourceState newState, uint32_t mip, uint32_t layer, agfxBool agglomerate) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = texture->d3d12Resource;
    barrier.Transition.StateBefore = agfxResourceStateToD3D12ResourceStates(oldState);
    barrier.Transition.StateAfter = agfxResourceStateToD3D12ResourceStates(newState);
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;

    bool allMips = (mip == (uint32_t)AGFX_SUBRESOURCE_ALL_MIPS);
    bool allLayers = (layer == (uint32_t)AGFX_SUBRESOURCE_ALL_LAYERS);
    if (allMips && allLayers) {
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        commandBuffer->d3d12CommandList->ResourceBarrier(1, &barrier);
    } else if (!allMips && !allLayers) {
        barrier.Transition.Subresource = D3D12CalcSubresource(mip, layer, 0, texture->createInfo.mipLevels, texture->createInfo.depthOrArrayLayers);
        commandBuffer->d3d12CommandList->ResourceBarrier(1, &barrier);
    } else {
        // Mixed single-axis sentinel (all mips at one layer, or all layers at one mip):
        // loop and issue one barrier per subresource on the "ALL" axis.
        uint32_t mipStart = allMips ? 0 : mip, mipEnd = allMips ? texture->createInfo.mipLevels : mip + 1;
        uint32_t layerStart = allLayers ? 0 : layer, layerEnd = allLayers ? texture->createInfo.depthOrArrayLayers : layer + 1;
        for (uint32_t l = layerStart; l < layerEnd; l++) {
            for (uint32_t m = mipStart; m < mipEnd; m++) {
                D3D12_RESOURCE_BARRIER sub = barrier;
                sub.Transition.Subresource = D3D12CalcSubresource(m, l, 0, texture->createInfo.mipLevels, texture->createInfo.depthOrArrayLayers);
                commandBuffer->d3d12CommandList->ResourceBarrier(1, &sub);
            }
        }
    }
}

void agfxCommandBufferBufferBarrier(agfxCommandBuffer* commandBuffer, agfxBuffer* buffer, agfxResourceState oldState, agfxResourceState newState, agfxBool agglomerate) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = buffer->d3d12Resource;
    barrier.Transition.StateBefore = agfxResourceStateToD3D12ResourceStates(oldState);
    barrier.Transition.StateAfter = agfxResourceStateToD3D12ResourceStates(newState);
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;

    commandBuffer->d3d12CommandList->ResourceBarrier(1, &barrier);
}

void agfxCommandBufferAccelerationStructureBarrier(agfxCommandBuffer* commandBuffer, agfxAccelerationStructure* accelerationStructure, agfxResourceState oldState, agfxResourceState newState, agfxBool agglomerate) {
    D3D12_RESOURCE_BARRIER barrier = {};

    // AS resources live permanently in RAYTRACING_ACCELERATION_STRUCTURE state; ordering builds/reads
    // against each other (e.g. BLAS build -> TLAS build) is a same-state wait, which D3D12 expresses
    // as a UAV barrier, not a state transition (D3D12 rejects a transition whose before/after states match).
    if (oldState == newState) {
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = accelerationStructure->d3d12Resource;
    } else {
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = accelerationStructure->d3d12Resource;
        barrier.Transition.StateBefore = agfxResourceStateToD3D12ResourceStates(oldState);
        barrier.Transition.StateAfter = agfxResourceStateToD3D12ResourceStates(newState);
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    }

    commandBuffer->d3d12CommandList->ResourceBarrier(1, &barrier);
}

// Texture

agfxTexture* agfxTextureCreate(agfxDevice* device, const agfxTextureCreateInfo* createInfo) {
    agfxTexture* texture = (agfxTexture*)device->createInfo.allocate(sizeof(agfxTexture));
    memcpy(&texture->createInfo, createInfo, sizeof(agfxTextureCreateInfo));

    D3D12_HEAP_PROPERTIES heapProperties = {};
    heapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = agfxTextureTypeToD3D12ResourceDimension(createInfo->type);
    resourceDesc.Width = createInfo->width;
    resourceDesc.Height = createInfo->height;
    resourceDesc.DepthOrArraySize = createInfo->depthOrArrayLayers;
    resourceDesc.MipLevels = createInfo->mipLevels;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resourceDesc.Format = agfxTextureFormatToDXGIFormat(createInfo->format);
    resourceDesc.Flags = agfxTextureUsageToD3D12ResourceFlags(createInfo->usage);
    
    if (FAILED(device->d3d12Device->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&texture->d3d12Resource)))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxTextureCreate: CreateCommittedResource failed");
        device->createInfo.free(texture);
        return NULL;
    }
    return texture;
}

void agfxTextureDestroy(agfxDevice* device, agfxTexture* texture) {
    if (texture->d3d12Resource) texture->d3d12Resource->Release();
    device->createInfo.free(texture);
}

void agfxTextureGetInfo(agfxTexture* texture, agfxTextureCreateInfo* info) {
    memcpy(info, &texture->createInfo, sizeof(agfxTextureCreateInfo));
}

void agfxTextureReplaceRegion(agfxDevice* device, agfxTexture* texture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, const void* data, uint32_t dataSize, uint32_t bytesPerRow, uint32_t bytesPerImage) {
    // Unsupported in D3D12, this is UMA only
}

void agfxTextureSetName(agfxTexture* texture, const char* name) {
    std::wstring wname(name, name + strlen(name));
    texture->d3d12Resource->SetName(wname.c_str());
}

// Buffer
agfxBuffer* agfxBufferCreate(agfxDevice* device, const agfxBufferCreateInfo* createInfo) {
    agfxBuffer* buffer = (agfxBuffer*)device->createInfo.allocate(sizeof(agfxBuffer));
    memcpy(&buffer->createInfo, createInfo, sizeof(agfxBufferCreateInfo));

    D3D12_HEAP_PROPERTIES heapProperties = {};
    heapProperties.Type = agfxBufferMemoryTypeToD3D12HeapType(createInfo->memoryType);

    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Width = createInfo->size;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.Flags = agfxBufferUsageToD3D12ResourceFlags(createInfo->usage);

    if (FAILED(device->d3d12Device->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&buffer->d3d12Resource)))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxBufferCreate: CreateCommittedResource failed");
        device->createInfo.free(buffer);
        return NULL;
    }
    return buffer;
}

void agfxBufferDestroy(agfxDevice* device, agfxBuffer* buffer) {
    if (buffer->d3d12Resource) buffer->d3d12Resource->Release();
    device->createInfo.free(buffer);
}

void* agfxBufferMap(agfxBuffer* buffer) {
    D3D12_RANGE readRange = { 0, 0 }; // We do not intend to read from this resource on the CPU.
    void* mappedData = nullptr;
    if (SUCCEEDED(buffer->d3d12Resource->Map(0, &readRange, &mappedData))) {
        return mappedData;
    }
    return nullptr;
}

void agfxBufferUnmap(agfxBuffer* buffer) {
    D3D12_RANGE writtenRange = { 0, buffer->createInfo.size }; // We wrote to the entire buffer.
    buffer->d3d12Resource->Unmap(0, &writtenRange);
}

void agfxBufferSetName(agfxBuffer* buffer, const char* name) {
    std::wstring wname(name, name + strlen(name));
    buffer->d3d12Resource->SetName(wname.c_str());
}

void agfxBufferGetInfo(agfxBuffer* buffer, agfxBufferCreateInfo* info) {
    memcpy(info, &buffer->createInfo, sizeof(agfxBufferCreateInfo));
}

// Acceleration structure

static D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS agfxAccelerationStructureBuildInputs(agfxAccelerationStructure* accelerationStructure) {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
    inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD
        | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    if (accelerationStructure->createInfo.allowUpdate) {
        inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    }

    if (accelerationStructure->createInfo.type == AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL) {
        inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        inputs.NumDescs = (UINT)accelerationStructure->geometryDescs.size();
        inputs.pGeometryDescs = accelerationStructure->geometryDescs.data();
    } else {
        inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        // Always build/size for the full instance capacity (matches the Metal4 backend, which
        // fixes MTL4InstanceAccelerationStructureDescriptor.instanceCount to maxInstanceCount at
        // create time) so the prebuild sizing call and every later build call agree on NumDescs,
        // which D3D12 requires for the allocated AS buffer to be large enough.
        inputs.NumDescs = accelerationStructure->createInfo.topLevel.maxInstanceCount;
        inputs.InstanceDescs = accelerationStructure->instanceBuffer ? accelerationStructure->instanceBuffer->GetGPUVirtualAddress() : 0;
    }

    return inputs;
}

agfxAccelerationStructure* agfxAccelerationStructureCreate(agfxDevice* device, const agfxAccelerationStructureCreateInfo* createInfo) {
    agfxAccelerationStructure* accelerationStructure = (agfxAccelerationStructure*)device->createInfo.allocate(sizeof(agfxAccelerationStructure));
    new (accelerationStructure) agfxAccelerationStructure();
    memcpy(&accelerationStructure->createInfo, createInfo, sizeof(agfxAccelerationStructureCreateInfo));

    if (createInfo->type == AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL) {
        accelerationStructure->geometryDescs.reserve(createInfo->bottomLevel.geometryCount);
        for (uint32_t i = 0; i < createInfo->bottomLevel.geometryCount; i++) {
            agfxAccelerationStructureGeometry* geometry = &createInfo->bottomLevel.geometries[i];

            D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc = {};
            geometryDesc.Flags = geometry->opaque ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;

            if (geometry->type == AGFX_ACCELERATION_STRUCTURE_GEOMETRY_TYPE_TRIANGLES) {
                geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                geometryDesc.Triangles.Transform3x4 = 0;
                geometryDesc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
                geometryDesc.Triangles.VertexCount = geometry->triangles.vertexCount;
                geometryDesc.Triangles.VertexBuffer.StartAddress = geometry->triangles.vertexBuffer->d3d12Resource->GetGPUVirtualAddress() + geometry->triangles.vertexOffset;
                geometryDesc.Triangles.VertexBuffer.StrideInBytes = geometry->triangles.vertexBuffer->createInfo.stride;
                geometryDesc.Triangles.IndexFormat = geometry->triangles.indexBuffer->createInfo.stride == 2 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
                geometryDesc.Triangles.IndexCount = geometry->triangles.indexCount;
                geometryDesc.Triangles.IndexBuffer = geometry->triangles.indexBuffer->d3d12Resource->GetGPUVirtualAddress() + geometry->triangles.indexOffset;
            } else {
                geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
                geometryDesc.AABBs.AABBCount = geometry->aabbs.aabbCount;
                geometryDesc.AABBs.AABBs.StartAddress = geometry->aabbs.aabbBuffer->d3d12Resource->GetGPUVirtualAddress() + geometry->aabbs.aabbOffset;
                geometryDesc.AABBs.AABBs.StrideInBytes = geometry->aabbs.aabbStride;
            }

            accelerationStructure->geometryDescs.push_back(geometryDesc);
        }
    } else {
        uint32_t maxInstanceCount = createInfo->topLevel.maxInstanceCount;

        D3D12_HEAP_PROPERTIES instanceHeapProperties = {};
        instanceHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;

        D3D12_RESOURCE_DESC instanceBufferDesc = {};
        instanceBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        instanceBufferDesc.Width = maxInstanceCount * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
        instanceBufferDesc.Height = 1;
        instanceBufferDesc.DepthOrArraySize = 1;
        instanceBufferDesc.MipLevels = 1;
        instanceBufferDesc.SampleDesc.Count = 1;
        instanceBufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        instanceBufferDesc.Format = DXGI_FORMAT_UNKNOWN;

        if (FAILED(device->d3d12Device->CreateCommittedResource(
            &instanceHeapProperties,
            D3D12_HEAP_FLAG_NONE,
            &instanceBufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&accelerationStructure->instanceBuffer)))) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxAccelerationStructureCreate: CreateCommittedResource for instance buffer failed");
            accelerationStructure->~agfxAccelerationStructure();
            device->createInfo.free(accelerationStructure);
            return NULL;
        }

        D3D12_RANGE readRange = { 0, 0 };
        accelerationStructure->instanceBuffer->Map(0, &readRange, (void**)&accelerationStructure->mappedInstanceBuffer);
        // Builds always cover the full maxInstanceCount range (see agfxAccelerationStructureBuildInputs);
        // zero-fill so slots beyond currentInstanceCount are inert (InstanceMask 0 -> always culled)
        // instead of tracing against whatever garbage the upload heap handed back.
        memset(accelerationStructure->mappedInstanceBuffer, 0, maxInstanceCount * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = agfxAccelerationStructureBuildInputs(accelerationStructure);
    device->d3d12Device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &accelerationStructure->prebuildInfo);

    D3D12_HEAP_PROPERTIES heapProperties = {};
    heapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Width = accelerationStructure->prebuildInfo.ResultDataMaxSizeInBytes;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    if (FAILED(device->d3d12Device->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        nullptr,
        IID_PPV_ARGS(&accelerationStructure->d3d12Resource)))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxAccelerationStructureCreate: CreateCommittedResource failed");
        if (accelerationStructure->instanceBuffer) accelerationStructure->instanceBuffer->Release();
        accelerationStructure->~agfxAccelerationStructure();
        device->createInfo.free(accelerationStructure);
        return NULL;
    }

    if (createInfo->type == AGFX_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.RaytracingAccelerationStructure.Location = accelerationStructure->d3d12Resource->GetGPUVirtualAddress();
        accelerationStructure->descriptor = device->descriptorManager->writeSRV(nullptr, &srvDesc);
    }

    if (createInfo->name) {
        std::wstring wname(createInfo->name, createInfo->name + strlen(createInfo->name));
        accelerationStructure->d3d12Resource->SetName(wname.c_str());
    }

    return accelerationStructure;
}

agfxAccelerationStructure* agfxAccelerationStructureCreateCompacted(agfxDevice* device, const agfxAccelerationStructureCreateInfo* createInfo, uint64_t compactedSize) {
    agfxAccelerationStructure* accelerationStructure = (agfxAccelerationStructure*)device->createInfo.allocate(sizeof(agfxAccelerationStructure));
    new (accelerationStructure) agfxAccelerationStructure();
    memcpy(&accelerationStructure->createInfo, createInfo, sizeof(agfxAccelerationStructureCreateInfo));
    accelerationStructure->prebuildInfo.ResultDataMaxSizeInBytes = compactedSize;

    D3D12_HEAP_PROPERTIES heapProperties = {};
    heapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Width = compactedSize;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    if (FAILED(device->d3d12Device->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        nullptr,
        IID_PPV_ARGS(&accelerationStructure->d3d12Resource)))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxAccelerationStructureCreateCompacted: CreateCommittedResource failed");
        accelerationStructure->~agfxAccelerationStructure();
        device->createInfo.free(accelerationStructure);
        return NULL;
    }

    if (createInfo->type == AGFX_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.RaytracingAccelerationStructure.Location = accelerationStructure->d3d12Resource->GetGPUVirtualAddress();
        accelerationStructure->descriptor = device->descriptorManager->writeSRV(nullptr, &srvDesc);
    }

    if (createInfo->name) {
        std::wstring wname(createInfo->name, createInfo->name + strlen(createInfo->name));
        accelerationStructure->d3d12Resource->SetName(wname.c_str());
    }

    return accelerationStructure;
}

void agfxAccelerationStructureDestroy(agfxDevice* device, agfxAccelerationStructure* accelerationStructure) {
    if (accelerationStructure->createInfo.type == AGFX_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL && accelerationStructure->descriptor.index != UINT64_MAX) {
        device->descriptorManager->agfxFreeResourceSlot((uint32_t)accelerationStructure->descriptor.index);
    }
    if (accelerationStructure->instanceBuffer) {
        accelerationStructure->instanceBuffer->Unmap(0, nullptr);
        accelerationStructure->instanceBuffer->Release();
    }
    if (accelerationStructure->d3d12Resource) accelerationStructure->d3d12Resource->Release();
    accelerationStructure->~agfxAccelerationStructure();
    device->createInfo.free(accelerationStructure);
}

void agfxAccelerationStructureGetSizes(agfxDevice* device, agfxAccelerationStructure* accelerationStructure, agfxAccelerationStructureSizes* sizes) {
    sizes->scratchBufferSize = accelerationStructure->prebuildInfo.ScratchDataSizeInBytes;
    sizes->updateScratchBufferSize = accelerationStructure->prebuildInfo.UpdateScratchDataSizeInBytes;
}

uint64_t agfxAccelerationStructureGetHandle(agfxAccelerationStructure* accelerationStructure) {
    return accelerationStructure->descriptor.index;
}

void agfxAccelerationStructureAddInstances(agfxAccelerationStructure* accelerationStructure, const agfxAccelerationStructureInstance* instances, uint32_t instanceCount) {
    for (uint32_t i = 0; i < instanceCount; ++i) {
        const agfxAccelerationStructureInstance* instance = &instances[i];
        D3D12_RAYTRACING_INSTANCE_DESC& d3d12Instance = accelerationStructure->mappedInstanceBuffer[accelerationStructure->currentInstanceCount + i];
        memcpy(d3d12Instance.Transform, instance->transform, sizeof(float) * 12);
        d3d12Instance.InstanceID = instance->userID;
        d3d12Instance.InstanceMask = 0xFF;
        d3d12Instance.InstanceContributionToHitGroupIndex = 0;
        d3d12Instance.Flags = instance->opaque ? D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE : D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE;
        d3d12Instance.AccelerationStructure = instance->blas->d3d12Resource->GetGPUVirtualAddress();
    }
    accelerationStructure->currentInstanceCount += instanceCount;
}

void agfxAccelerationStructureResetInstances(agfxAccelerationStructure* accelerationStructure) {
    accelerationStructure->currentInstanceCount = 0;
}

// Texture view
struct agfxTextureView {
    agfxTexture* texture;
    agfxDescriptorManager* descriptorManager;
    agfxTextureViewCreateInfo createInfo;
    agfxDescriptorAllocation descriptor;
};

agfxTextureView* agfxTextureViewCreate(agfxDevice* device, const agfxTextureViewCreateInfo* createInfo) {
    agfxTextureView* textureView = (agfxTextureView*)device->createInfo.allocate(sizeof(agfxTextureView));
    memset(textureView, 0, sizeof(agfxTextureView));
    memcpy(&textureView->createInfo, createInfo, sizeof(agfxTextureViewCreateInfo));
    textureView->texture = createInfo->texture;
    textureView->descriptorManager = device->descriptorManager;

    if (createInfo->writeable) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = agfxTextureViewTypeToD3D12UnorderedAccessViewDesc((agfxTextureViewCreateInfo*)createInfo);
        textureView->descriptor = device->descriptorManager->writeUAV(createInfo->texture->d3d12Resource, &uavDesc);
    } else {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = agfxTextureViewTypeToD3D12ShaderResourceViewDesc((agfxTextureViewCreateInfo*)createInfo);
        textureView->descriptor = device->descriptorManager->writeSRV(createInfo->texture->d3d12Resource, &srvDesc);
    }

    return textureView;
}

void agfxTextureViewDestroy(agfxDevice* device, agfxTextureView* textureView) {
    if (textureView->descriptor.index != UINT32_MAX) {
        device->descriptorManager->agfxFreeResourceSlot(textureView->descriptor.index);
    }
    device->createInfo.free(textureView);
}

uint64_t agfxTextureViewGetHandle(agfxTextureView* textureView) {
    return textureView->descriptor.index;
}

// Sampler
struct agfxSampler {
    agfxDescriptorManager* descriptorManager;
    agfxDescriptorAllocation descriptor;
    agfxSamplerCreateInfo createInfo;
};

agfxSampler* agfxSamplerCreate(agfxDevice* device, const agfxSamplerCreateInfo* createInfo) {
    agfxSampler* sampler = (agfxSampler*)device->createInfo.allocate(sizeof(agfxSampler));
    memset(sampler, 0, sizeof(agfxSampler));
    memcpy(&sampler->createInfo, createInfo, sizeof(agfxSamplerCreateInfo));
    sampler->descriptorManager = device->descriptorManager;

    D3D12_SAMPLER_DESC samplerDesc = agfxSamplerCreateInfoToD3D12SamplerDesc((agfxSamplerCreateInfo*)createInfo);
    sampler->descriptor = device->descriptorManager->writeSampler(&samplerDesc);

    return sampler;
}

void agfxSamplerDestroy(agfxDevice* device, agfxSampler* sampler) {
    if (sampler->descriptor.index != UINT32_MAX) {
        device->descriptorManager->freeSamplerSlot(sampler->descriptor.index);
    }
    device->createInfo.free(sampler);
}

uint64_t agfxSamplerGetHandle(agfxSampler* sampler) {
    return sampler->descriptor.index;
}

// Buffer view
struct agfxBufferView {
    agfxBuffer* buffer;
    agfxDescriptorManager* descriptorManager;
    agfxBufferViewCreateInfo createInfo;
    agfxDescriptorAllocation descriptor;
};

agfxBufferView* agfxBufferViewCreate(agfxDevice* device, const agfxBufferViewCreateInfo* createInfo) {
    agfxBufferView* bufferView = (agfxBufferView*)device->createInfo.allocate(sizeof(agfxBufferView));
    memset(bufferView, 0, sizeof(agfxBufferView));

    if (createInfo->type == AGFX_BUFFER_VIEW_TYPE_RAW || createInfo->type == AGFX_BUFFER_VIEW_TYPE_STRUCTURED) {
        if (!createInfo->writeable) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = agfxBufferViewTypeToD3D12ShaderResourceViewDesc((agfxBufferViewCreateInfo*)createInfo);
            bufferView->descriptor = device->descriptorManager->writeSRV(createInfo->buffer->d3d12Resource, &srvDesc);
        } else {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = agfxBufferViewTypeToD3D12UnorderedAccessViewDesc((agfxBufferViewCreateInfo*)createInfo);
            bufferView->descriptor = device->descriptorManager->writeUAV(createInfo->buffer->d3d12Resource, &uavDesc);
        }
    } else if (createInfo->type == AGFX_BUFFER_VIEW_TYPE_CONSTANT) {
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = agfxBufferViewTypeToD3D12ConstantBufferViewDesc((agfxBufferViewCreateInfo*)createInfo);
        bufferView->descriptor = device->descriptorManager->writeCBV(&cbvDesc);
    }
    return bufferView;
}

void agfxBufferViewDestroy(agfxDevice* device, agfxBufferView* bufferView) {
    device->descriptorManager->agfxFreeResourceSlot(bufferView->descriptor.index);
    device->createInfo.free(bufferView);
}

uint64_t agfxBufferViewGetHandle(agfxBufferView* bufferView) {
    return bufferView->descriptor.index;
}

// Render target
struct agfxRenderTarget {
    agfxTexture* texture;
    agfxDescriptorManager* descriptorManager;
    agfxRenderTargetCreateInfo createInfo;
    agfxDescriptorAllocation descriptor;
};

agfxRenderTarget* agfxRenderTargetCreate(agfxDevice* device, const agfxRenderTargetCreateInfo* createInfo) {
    agfxRenderTarget* renderTarget = (agfxRenderTarget*)device->createInfo.allocate(sizeof(agfxRenderTarget));
    memset(renderTarget, 0, sizeof(agfxRenderTarget));
    memcpy(&renderTarget->createInfo, createInfo, sizeof(agfxRenderTargetCreateInfo));
    renderTarget->texture = createInfo->texture;
    renderTarget->descriptorManager = device->descriptorManager;

    if (createInfo->isDepth) {
        D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = agfxTextureViewTypeToD3D12DepthStencilViewDesc((agfxRenderTargetCreateInfo*)createInfo);
        renderTarget->descriptor = device->descriptorManager->writeDSV(createInfo->texture->d3d12Resource, &dsvDesc);
    } else {
        D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = agfxTextureViewTypeToD3D12RenderTargetViewDesc((agfxRenderTargetCreateInfo*)createInfo);
        renderTarget->descriptor = device->descriptorManager->writeRTV(createInfo->texture->d3d12Resource, &rtvDesc);
    }

    return renderTarget;
}

void agfxRenderTargetDestroy(agfxDevice* device, agfxRenderTarget* renderTarget) {
    if (renderTarget->createInfo.isDepth) {
        device->descriptorManager->freeDSVSlot(renderTarget->descriptor.index);
    } else {
        device->descriptorManager->freeRTVSlot(renderTarget->descriptor.index);
    }
    device->createInfo.free(renderTarget);
}

// Swap chain
struct agfxSwapChain {
    agfxSwapChainCreateInfo createInfo;
    agfxDevice* device;
    agfxCommandQueue* queue;
    IDXGISwapChain4* dxgiSwapChain;
    agfxTexture** backBuffers;
    uint32_t imageCount;
    uint32_t frameIndex;
    agfxTextureFormat format;
    agfxBool tearingSupported;
    agfxBool hdrEnabled;
};

static bool agfxD3D12CheckTearingSupport(IDXGIFactory6* factory) {
    BOOL allowTearing = FALSE;
    IDXGIFactory5* factory5 = nullptr;
    if (SUCCEEDED(factory->QueryInterface(IID_PPV_ARGS(&factory5)))) {
        if (FAILED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing)))) {
            allowTearing = FALSE;
        }
        factory5->Release();
    }
    return allowTearing == TRUE;
}

// A display only reports the HDR10 (G2084/P2020) color space once the OS has actually
// negotiated HDR output for it, so this must be checked per-swapchain, not just per-adapter.
static bool agfxD3D12CheckHDRSupport(IDXGISwapChain4* swapChain) {
    IDXGIOutput* output = nullptr;
    if (FAILED(swapChain->GetContainingOutput(&output)) || !output) {
        return false;
    }

    bool hdrSupported = false;
    IDXGIOutput6* output6 = nullptr;
    if (SUCCEEDED(output->QueryInterface(IID_PPV_ARGS(&output6)))) {
        DXGI_OUTPUT_DESC1 desc1 = {};
        if (SUCCEEDED(output6->GetDesc1(&desc1))) {
            hdrSupported = desc1.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
        }
        output6->Release();
    }
    output->Release();
    return hdrSupported;
}

agfxSwapChain* agfxSwapChainCreate(agfxDevice* device, const agfxSwapChainCreateInfo* createInfo) {
    agfxSwapChain* swapChain = (agfxSwapChain*)device->createInfo.allocate(sizeof(agfxSwapChain));
    memset(swapChain, 0, sizeof(agfxSwapChain));
    memcpy(&swapChain->createInfo, createInfo, sizeof(agfxSwapChainCreateInfo));
    swapChain->device = device;
    swapChain->queue = createInfo->queue;
    swapChain->imageCount = createInfo->imageCount;
    swapChain->tearingSupported = agfxD3D12CheckTearingSupport(device->dxgiFactory);

    HWND hwnd = (HWND)createInfo->handle;

    DXGI_SWAP_CHAIN_DESC1 desc = {};
    desc.Width = createInfo->width;
    desc.Height = createInfo->height;
    desc.Format = createInfo->isHDR ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.Stereo = FALSE;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = createInfo->imageCount;
    desc.Scaling = DXGI_SCALING_STRETCH;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    desc.Flags = swapChain->tearingSupported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

    IDXGISwapChain1* swapChain1 = nullptr;
    if (FAILED(device->dxgiFactory->CreateSwapChainForHwnd(swapChain->queue->d3d12CommandQueue, hwnd, &desc, nullptr, nullptr, &swapChain1))) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxSwapChainCreate: CreateSwapChainForHwnd failed");
        device->createInfo.free(swapChain);
        return nullptr;
    }

    // We handle fullscreen ourselves (borderless windowed); don't let DXGI intercept alt+enter.
    device->dxgiFactory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);

    swapChain1->QueryInterface(IID_PPV_ARGS(&swapChain->dxgiSwapChain));
    swapChain1->Release();

    swapChain->format = createInfo->isHDR ? AGFX_TEXTURE_FORMAT_RGBA16F : AGFX_TEXTURE_FORMAT_BGRA8_UNORM;
    swapChain->hdrEnabled = false;

    if (createInfo->isHDR && agfxD3D12CheckHDRSupport(swapChain->dxgiSwapChain)) {
        // scRGB linear (extended sRGB primaries) matches the R16G16B16A16_FLOAT buffer format
        // and is what the OS composits as HDR10 to the display; mirrors the Metal EDR path.
        if (SUCCEEDED(swapChain->dxgiSwapChain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709))) {
            swapChain->hdrEnabled = true;
        }
    }

    swapChain->backBuffers = (agfxTexture**)device->createInfo.allocate(sizeof(agfxTexture*) * createInfo->imageCount);
    for (uint32_t i = 0; i < createInfo->imageCount; i++) {
        ID3D12Resource* resource = nullptr;
        swapChain->dxgiSwapChain->GetBuffer(i, IID_PPV_ARGS(&resource));

        agfxTexture* texture = (agfxTexture*)device->createInfo.allocate(sizeof(agfxTexture));
        memset(texture, 0, sizeof(agfxTexture));
        texture->d3d12Resource = resource;
        texture->createInfo.type = AGFX_TEXTURE_TYPE_2D;
        texture->createInfo.format = swapChain->format;
        texture->createInfo.usage = AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT;
        texture->createInfo.width = createInfo->width;
        texture->createInfo.height = createInfo->height;
        texture->createInfo.depthOrArrayLayers = 1;
        texture->createInfo.mipLevels = 1;

        swapChain->backBuffers[i] = texture;
    }

    swapChain->frameIndex = swapChain->dxgiSwapChain->GetCurrentBackBufferIndex();
    return swapChain;
}

void agfxSwapChainDestroy(agfxDevice* device, agfxSwapChain* swapChain) {
    for (uint32_t i = 0; i < swapChain->imageCount; i++) {
        if (swapChain->backBuffers[i]->d3d12Resource) {
            swapChain->backBuffers[i]->d3d12Resource->Release();
        }
        device->createInfo.free(swapChain->backBuffers[i]);
    }
    device->createInfo.free(swapChain->backBuffers);

    if (swapChain->dxgiSwapChain) swapChain->dxgiSwapChain->Release();
    device->createInfo.free(swapChain);
}

void agfxSwapChainResize(agfxDevice* device, agfxSwapChain* swapChain, uint32_t width, uint32_t height) {
    for (uint32_t i = 0; i < swapChain->imageCount; i++) {
        if (swapChain->backBuffers[i]->d3d12Resource) {
            swapChain->backBuffers[i]->d3d12Resource->Release();
            swapChain->backBuffers[i]->d3d12Resource = nullptr;
        }
    }

    DXGI_SWAP_CHAIN_DESC1 currentDesc = {};
    swapChain->dxgiSwapChain->GetDesc1(&currentDesc);
    swapChain->dxgiSwapChain->ResizeBuffers(swapChain->imageCount, width, height, currentDesc.Format, currentDesc.Flags);

    for (uint32_t i = 0; i < swapChain->imageCount; i++) {
        ID3D12Resource* resource = nullptr;
        swapChain->dxgiSwapChain->GetBuffer(i, IID_PPV_ARGS(&resource));
        swapChain->backBuffers[i]->d3d12Resource = resource;
        swapChain->backBuffers[i]->createInfo.width = width;
        swapChain->backBuffers[i]->createInfo.height = height;
    }

    swapChain->createInfo.width = width;
    swapChain->createInfo.height = height;
    swapChain->frameIndex = swapChain->dxgiSwapChain->GetCurrentBackBufferIndex();
}

agfxTextureFormat agfxSwapChainGetFormat(agfxSwapChain* swapChain) {
    return swapChain->format;
}

agfxTexture* agfxSwapChainAcquireNextTexture(agfxSwapChain* swapChain) {
    swapChain->frameIndex = swapChain->dxgiSwapChain->GetCurrentBackBufferIndex();
    return swapChain->backBuffers[swapChain->frameIndex];
}

void agfxSwapChainPresent(agfxSwapChain* swapChain) {
    UINT syncInterval = swapChain->createInfo.vsync ? 1 : 0;
    UINT presentFlags = (!swapChain->createInfo.vsync && swapChain->tearingSupported) ? DXGI_PRESENT_ALLOW_TEARING : 0;
    swapChain->dxgiSwapChain->Present(syncInterval, presentFlags);
}

// Shader module
struct agfxShaderModule {
    agfxShaderModuleCreateInfo createInfo;
};

agfxShaderModule* agfxShaderModuleCreate(agfxDevice* device, const agfxShaderModuleCreateInfo* createInfo) {
    agfxShaderModule* shaderModule = (agfxShaderModule*)device->createInfo.allocate(sizeof(agfxShaderModule));
    memcpy(&shaderModule->createInfo, createInfo, sizeof(agfxShaderModuleCreateInfo));

    shaderModule->createInfo.code = (uint8_t*)device->createInfo.allocate(createInfo->codeSize);
    memcpy(shaderModule->createInfo.code, createInfo->code, createInfo->codeSize);

    return shaderModule;
}

void agfxShaderModuleDestroy(agfxDevice* device, agfxShaderModule* shaderModule) {
    device->createInfo.free(shaderModule->createInfo.code);
    device->createInfo.free(shaderModule);
}

// Compute pipeline
struct agfxComputePipeline {
    agfxComputePipelineCreateInfo createInfo;
    ID3D12PipelineState* d3d12PipelineState;
};

agfxComputePipeline* agfxComputePipelineCreate(agfxDevice* device, const agfxComputePipelineCreateInfo* createInfo) {
    agfxComputePipeline* pipeline = (agfxComputePipeline*)device->createInfo.allocate(sizeof(agfxComputePipeline));

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = device->globalRootSignature;
    psoDesc.CS.pShaderBytecode = createInfo->computeShader->createInfo.code;
    psoDesc.CS.BytecodeLength = createInfo->computeShader->createInfo.codeSize;

	if (FAILED(device->d3d12Device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pipeline->d3d12PipelineState)))) {
		agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxComputePipelineCreate: CreateComputePipelineState failed");
		device->createInfo.free(pipeline);
		return NULL;
	}
    return pipeline;
}

void agfxComputePipelineDestroy(agfxDevice* device, agfxComputePipeline* pipeline) {
    device->createInfo.free(pipeline);
}

// Render pipeline
struct agfxRenderPipeline {
    agfxRenderPipelineCreateInfo createInfo;
    ID3D12PipelineState* d3d12PipelineState;
};

agfxRenderPipeline* agfxRenderPipelineCreate(agfxDevice* device, const agfxRenderPipelineCreateInfo* createInfo) {
    agfxRenderPipeline* pipeline = (agfxRenderPipeline*)device->createInfo.allocate(sizeof(agfxRenderPipeline));
    memcpy(&pipeline->createInfo, createInfo, sizeof(agfxRenderPipelineCreateInfo));

    if (createInfo->meshShader || createInfo->taskShader) {
        D3DX12_MESH_SHADER_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = device->globalRootSignature;
        if (createInfo->taskShader) {
            psoDesc.AS.pShaderBytecode = createInfo->taskShader->createInfo.code;
            psoDesc.AS.BytecodeLength = createInfo->taskShader->createInfo.codeSize;
        }
        psoDesc.MS.pShaderBytecode = createInfo->meshShader->createInfo.code;
        psoDesc.MS.BytecodeLength = createInfo->meshShader->createInfo.codeSize;
        if (createInfo->fragmentShader) {
            psoDesc.PS.pShaderBytecode = createInfo->fragmentShader->createInfo.code;
            psoDesc.PS.BytecodeLength = createInfo->fragmentShader->createInfo.codeSize;
        }
        psoDesc.RasterizerState.FillMode = agfxFillModeToD3D12FillMode(createInfo->fillMode);
        psoDesc.RasterizerState.CullMode = agfxCullModeToD3D12CullMode(createInfo->cullMode);
        psoDesc.RasterizerState.FrontCounterClockwise = agfxFrontFaceToD3D12FrontCounterClockwise(createInfo->frontFace);
        psoDesc.RasterizerState.DepthClipEnable = !createInfo->depthClampEnable;
        psoDesc.RasterizerState.DepthBias = 0;
        psoDesc.RasterizerState.DepthBiasClamp = 0.0f;
        psoDesc.RasterizerState.SlopeScaledDepthBias = 0.0f;
        psoDesc.RasterizerState.MultisampleEnable = FALSE;
        psoDesc.RasterizerState.AntialiasedLineEnable = FALSE;
        psoDesc.DepthStencilState.DepthEnable = createInfo->depthTestEnable;
        psoDesc.DepthStencilState.DepthWriteMask = createInfo->depthWriteEnable ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
        psoDesc.DepthStencilState.DepthFunc = agfxComparisonFunctionToD3D12ComparisonFunc(createInfo->depthCompareOp);
        psoDesc.DepthStencilState.StencilEnable = FALSE;
        psoDesc.DSVFormat = createInfo->depthFormat != AGFX_TEXTURE_FORMAT_UNKNOWN ? agfxTextureFormatToDXGIFormat(createInfo->depthFormat) : DXGI_FORMAT_UNKNOWN;
        psoDesc.BlendState.IndependentBlendEnable = TRUE;
        for (uint32_t i = 0; i < createInfo->colorAttachmentCount && i < 8; ++i) {
            D3D12_RENDER_TARGET_BLEND_DESC& rt = psoDesc.BlendState.RenderTarget[i];
            rt.BlendEnable = createInfo->blendEnable[i];
            rt.SrcBlend = agfxBlendFactorToD3D12Blend(createInfo->srcColorBlendFactor[i]);
            rt.DestBlend = agfxBlendFactorToD3D12Blend(createInfo->dstColorBlendFactor[i]);
            rt.BlendOp = agfxBlendOperationToD3D12BlendOp(createInfo->colorBlendOp[i]);
            rt.SrcBlendAlpha = agfxBlendFactorToD3D12Blend(createInfo->srcAlphaBlendFactor[i]);
            rt.DestBlendAlpha = agfxBlendFactorToD3D12Blend(createInfo->dstAlphaBlendFactor[i]);
            rt.BlendOpAlpha = agfxBlendOperationToD3D12BlendOp(createInfo->alphaBlendOp[i]);
            rt.LogicOpEnable = FALSE;
            rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
            psoDesc.RTVFormats[i] = agfxTextureFormatToDXGIFormat(createInfo->colorFormats[i]);
        }
        psoDesc.NumRenderTargets = createInfo->colorAttachmentCount;
        psoDesc.PrimitiveTopologyType = agfxTopologyToD3D12PrimitiveTopologyType(createInfo->topology);
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.SampleDesc.Count = 1;

        auto stream = CD3DX12_PIPELINE_MESH_STATE_STREAM(psoDesc);

        D3D12_PIPELINE_STATE_STREAM_DESC streamDesc;
        streamDesc.pPipelineStateSubobjectStream = &stream;
        streamDesc.SizeInBytes = sizeof(stream);

        if (FAILED(device->d3d12Device->CreatePipelineState(&streamDesc, IID_PPV_ARGS(&pipeline->d3d12PipelineState)))) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxRenderPipelineCreate: CreatePipelineState (mesh pipeline) failed");
            device->createInfo.free(pipeline);
            return NULL;
        }
    } else {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = device->globalRootSignature;
        psoDesc.VS.pShaderBytecode = createInfo->vertexShader->createInfo.code;
        psoDesc.VS.BytecodeLength = createInfo->vertexShader->createInfo.codeSize;
        if (createInfo->fragmentShader) {
            psoDesc.PS.pShaderBytecode = createInfo->fragmentShader->createInfo.code;
            psoDesc.PS.BytecodeLength = createInfo->fragmentShader->createInfo.codeSize;
        }
        psoDesc.RasterizerState.FillMode = agfxFillModeToD3D12FillMode(createInfo->fillMode);
        psoDesc.RasterizerState.CullMode = agfxCullModeToD3D12CullMode(createInfo->cullMode);
        psoDesc.RasterizerState.FrontCounterClockwise = agfxFrontFaceToD3D12FrontCounterClockwise(createInfo->frontFace);
        psoDesc.RasterizerState.DepthClipEnable = !createInfo->depthClampEnable;
        psoDesc.RasterizerState.DepthBias = 0;
        psoDesc.RasterizerState.DepthBiasClamp = 0.0f;
        psoDesc.RasterizerState.SlopeScaledDepthBias = 0.0f;
        psoDesc.RasterizerState.MultisampleEnable = FALSE;
        psoDesc.RasterizerState.AntialiasedLineEnable = FALSE;
        psoDesc.DepthStencilState.DepthEnable = createInfo->depthTestEnable;
        psoDesc.DepthStencilState.DepthWriteMask = createInfo->depthWriteEnable ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
        psoDesc.DepthStencilState.DepthFunc = agfxComparisonFunctionToD3D12ComparisonFunc(createInfo->depthCompareOp);
        psoDesc.DepthStencilState.StencilEnable = FALSE;
        psoDesc.DSVFormat = createInfo->depthFormat != AGFX_TEXTURE_FORMAT_UNKNOWN ? agfxTextureFormatToDXGIFormat(createInfo->depthFormat) : DXGI_FORMAT_UNKNOWN;
        psoDesc.BlendState.IndependentBlendEnable = TRUE;
        for (uint32_t i = 0; i < createInfo->colorAttachmentCount && i < 8; ++i) {
            D3D12_RENDER_TARGET_BLEND_DESC& rt = psoDesc.BlendState.RenderTarget[i];
            rt.BlendEnable = createInfo->blendEnable[i];
            rt.SrcBlend = agfxBlendFactorToD3D12Blend(createInfo->srcColorBlendFactor[i]);
            rt.DestBlend = agfxBlendFactorToD3D12Blend(createInfo->dstColorBlendFactor[i]);
            rt.BlendOp = agfxBlendOperationToD3D12BlendOp(createInfo->colorBlendOp[i]);
            rt.SrcBlendAlpha = agfxBlendFactorToD3D12Blend(createInfo->srcAlphaBlendFactor[i]);
            rt.DestBlendAlpha = agfxBlendFactorToD3D12Blend(createInfo->dstAlphaBlendFactor[i]);
            rt.BlendOpAlpha = agfxBlendOperationToD3D12BlendOp(createInfo->alphaBlendOp[i]);
            rt.LogicOpEnable = FALSE;
            rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
            psoDesc.RTVFormats[i] = agfxTextureFormatToDXGIFormat(createInfo->colorFormats[i]);
        }
        psoDesc.NumRenderTargets = createInfo->colorAttachmentCount;
        psoDesc.PrimitiveTopologyType = agfxTopologyToD3D12PrimitiveTopologyType(createInfo->topology);
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.SampleDesc.Count = 1;
        psoDesc.InputLayout = { nullptr, 0 };

        if (FAILED(device->d3d12Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pipeline->d3d12PipelineState)))) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxRenderPipelineCreate: CreateGraphicsPipelineState failed");
            device->createInfo.free(pipeline);
            return NULL;
        }
    }

    return pipeline;
}

void agfxRenderPipelineDestroy(agfxDevice* device, agfxRenderPipeline* pipeline) {
    if (pipeline->d3d12PipelineState) pipeline->d3d12PipelineState->Release();
    device->createInfo.free(pipeline);
}

// Render pass
struct agfxRenderPass {
    agfxCommandBuffer* commandBuffer;
    agfxRenderPassCreateInfo createInfo;
};

agfxRenderPass* agfxRenderPassBegin(agfxCommandBuffer* cmdBuffer, const agfxRenderPassCreateInfo* createInfo) {
    agfxRenderPass* renderPass = (agfxRenderPass*)cmdBuffer->device->createInfo.tempAllocate(sizeof(agfxRenderPass));
    memcpy(&renderPass->createInfo, createInfo, sizeof(agfxRenderPassCreateInfo));
    renderPass->commandBuffer = cmdBuffer;

    // Bind render targets and clear them if needed
    ID3D12GraphicsCommandList* commandList = cmdBuffer->d3d12CommandList;
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandles[8] = {};
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = {};
    for (uint32_t i = 0; i < createInfo->colorAttachmentCount && i < 8; ++i) {
        rtvHandles[i] = createInfo->colorAttachments[i].renderTarget->descriptor.cpuHandle;
        if (createInfo->colorAttachments[i].loadOp == AGFX_LOAD_OPERATION_CLEAR) {
            commandList->ClearRenderTargetView(rtvHandles[i], createInfo->colorAttachments[i].clearColor, 0, nullptr);
        }
    }
    if (createInfo->hasDepthAttachment) {
        dsvHandle = createInfo->depthAttachment.renderTarget->descriptor.cpuHandle;
        if (createInfo->depthAttachment.loadOp == AGFX_LOAD_OPERATION_CLEAR) {
            commandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, createInfo->depthAttachment.clearDepth, 0, 0, nullptr);
        }
    }

    commandList->OMSetRenderTargets(createInfo->colorAttachmentCount, rtvHandles, FALSE, createInfo->hasDepthAttachment ? &dsvHandle : nullptr);
    return renderPass;
}

void agfxRenderPassSetViewport(agfxRenderPass* renderPass, float x, float y, float width, float height, float minDepth, float maxDepth) {
    D3D12_VIEWPORT viewport = {};
    viewport.TopLeftX = x;
    viewport.TopLeftY = y;
    viewport.Width = width;
    viewport.Height = height;
    viewport.MinDepth = minDepth;
    viewport.MaxDepth = maxDepth;

    renderPass->commandBuffer->d3d12CommandList->RSSetViewports(1, &viewport);
}

void agfxRenderPassSetScissor(agfxRenderPass* renderPass, uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
    D3D12_RECT scissorRect = {};
    scissorRect.left = x;
    scissorRect.top = y;
    scissorRect.right = x + width;
    scissorRect.bottom = y + height;

    renderPass->commandBuffer->d3d12CommandList->RSSetScissorRects(1, &scissorRect);
}

void agfxRenderPassSetPipeline(agfxRenderPass* renderPass, agfxRenderPipeline* pipeline) {
	renderPass->commandBuffer->d3d12CommandList->SetGraphicsRootSignature(renderPass->commandBuffer->device->globalRootSignature);
    renderPass->commandBuffer->d3d12CommandList->SetPipelineState(pipeline->d3d12PipelineState);
    renderPass->commandBuffer->d3d12CommandList->IASetPrimitiveTopology(agfxTopologyToD3D12PrimitiveTopology(pipeline->createInfo.topology));
}

void agfxRenderPassPushConstants(agfxRenderPass* renderPass, const void* data, uint32_t size) {
    renderPass->commandBuffer->d3d12CommandList->SetGraphicsRoot32BitConstants(0, size / 4, data, 0);
}

void agfxRenderPassDraw(agfxRenderPass* renderPass, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
    renderPass->commandBuffer->d3d12CommandList->DrawInstanced(vertexCount, instanceCount, firstVertex, firstInstance);
}

void agfxRenderPassDrawIndexed(agfxRenderPass* renderPass, agfxBuffer* indexBuffer, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, uint32_t vertexOffset, uint32_t firstInstance) {
    D3D12_INDEX_BUFFER_VIEW ibv = {};
	ibv.BufferLocation = indexBuffer->d3d12Resource->GetGPUVirtualAddress();
	ibv.Format = indexBuffer->createInfo.stride == 2 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
	ibv.SizeInBytes = indexBuffer->createInfo.size;

	renderPass->commandBuffer->d3d12CommandList->IASetIndexBuffer(&ibv);
    renderPass->commandBuffer->d3d12CommandList->DrawIndexedInstanced(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

void agfxRenderPassDrawMesh(agfxRenderPass* renderPass, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    renderPass->commandBuffer->d3d12CommandList->DispatchMesh(groupCountX, groupCountY, groupCountZ);
}

void agfxRenderPassEnd(agfxRenderPass* renderPass) {
    renderPass->commandBuffer->device->createInfo.tempFree(renderPass);
}

// Compute pass
struct agfxComputePass {
    agfxCommandBuffer* commandBuffer;
    agfxDevice* device;
};

agfxComputePass* agfxComputePassBegin(agfxCommandBuffer* commandBuffer, const char* name) {
    agfxComputePass* computePass = (agfxComputePass*)commandBuffer->device->createInfo.tempAllocate(sizeof(agfxComputePass));
    computePass->commandBuffer = commandBuffer;
    computePass->device = commandBuffer->device;
    return computePass;
}

void agfxComputePassTextureUAVBarrier(agfxComputePass* computePass, agfxTexture* texture) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = texture->d3d12Resource;
    computePass->commandBuffer->d3d12CommandList->ResourceBarrier(1, &barrier);
}

void agfxComputePassBufferUAVBarrier(agfxComputePass* computePass, agfxBuffer* buffer) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = buffer->d3d12Resource;
    computePass->commandBuffer->d3d12CommandList->ResourceBarrier(1, &barrier);
}

void agfxComputePassBuildAccelerationStructure(agfxComputePass* computePass, agfxAccelerationStructure* accelerationStructure, agfxBuffer* scratchBuffer, uint64_t scratchBufferOffset) {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc = {};
    desc.DestAccelerationStructureData = accelerationStructure->d3d12Resource->GetGPUVirtualAddress();
    desc.ScratchAccelerationStructureData = scratchBuffer->d3d12Resource->GetGPUVirtualAddress() + scratchBufferOffset;
    desc.Inputs = agfxAccelerationStructureBuildInputs(accelerationStructure);

    computePass->commandBuffer->d3d12CommandList->BuildRaytracingAccelerationStructure(&desc, 0, nullptr);
}

void agfxComputePassUpdateAccelerationStructure(agfxComputePass* computePass, agfxAccelerationStructure* srcAccelerationStructure, agfxAccelerationStructure* dstAccelerationStructure, agfxBuffer* scratchBuffer, uint64_t scratchBufferOffset) {
    if (!dstAccelerationStructure->createInfo.allowUpdate) return;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc = {};
    desc.DestAccelerationStructureData = dstAccelerationStructure->d3d12Resource->GetGPUVirtualAddress();
    desc.SourceAccelerationStructureData = srcAccelerationStructure->d3d12Resource->GetGPUVirtualAddress();
    desc.ScratchAccelerationStructureData = scratchBuffer->d3d12Resource->GetGPUVirtualAddress() + scratchBufferOffset;
    desc.Inputs = agfxAccelerationStructureBuildInputs(srcAccelerationStructure);
    desc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;

    computePass->commandBuffer->d3d12CommandList->BuildRaytracingAccelerationStructure(&desc, 0, nullptr);
}

void agfxComputePassCopyAccelerationStructure(agfxComputePass* computePass, agfxAccelerationStructure* srcAccelerationStructure, agfxAccelerationStructure* dstAccelerationStructure) {
    computePass->commandBuffer->d3d12CommandList->CopyRaytracingAccelerationStructure(
        dstAccelerationStructure->d3d12Resource->GetGPUVirtualAddress(),
        srcAccelerationStructure->d3d12Resource->GetGPUVirtualAddress(),
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE);
}

void agfxComputePassWriteCompactedSizeToBuffer(agfxComputePass* computePass, agfxAccelerationStructure* accelerationStructure, agfxBuffer* dstBuffer, uint64_t dstBufferOffset) {
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postbuildInfoDesc = {};
    postbuildInfoDesc.DestBuffer = dstBuffer->d3d12Resource->GetGPUVirtualAddress() + dstBufferOffset;
    postbuildInfoDesc.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;

    D3D12_GPU_VIRTUAL_ADDRESS sourceAS = accelerationStructure->d3d12Resource->GetGPUVirtualAddress();
    computePass->commandBuffer->d3d12CommandList->EmitRaytracingAccelerationStructurePostbuildInfo(&postbuildInfoDesc, 1, &sourceAS);
}

void agfxComputePassCompactAccelerationStructure(agfxComputePass* computePass, agfxAccelerationStructure* srcAccelerationStructure, agfxAccelerationStructure* dstAccelerationStructure) {
    computePass->commandBuffer->d3d12CommandList->CopyRaytracingAccelerationStructure(
        dstAccelerationStructure->d3d12Resource->GetGPUVirtualAddress(),
        srcAccelerationStructure->d3d12Resource->GetGPUVirtualAddress(),
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
}

void agfxComputePassCopyTextureToBuffer(agfxComputePass* computePass, agfxTexture* texture, agfxBuffer* buffer, uint64_t bufferOffset, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, uint32_t bytesPerRow, uint32_t bytesPerImage) {
    UINT subresourceIndex = D3D12CalcSubresource(mipLevel, layer, 0, texture->createInfo.mipLevels, texture->createInfo.depthOrArrayLayers);

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    footprint.Offset = bufferOffset;
    footprint.Footprint.Format = agfxTextureFormatToDXGIFormat(texture->createInfo.format);
    footprint.Footprint.Width = region->width;
    footprint.Footprint.Height = region->height;
    footprint.Footprint.Depth = region->depth;
    footprint.Footprint.RowPitch = (UINT)bytesPerRow;

    D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
    srcLoc.pResource = texture->d3d12Resource;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex = subresourceIndex;

    D3D12_BOX srcBox = {};
    srcBox.left = region->x;
    srcBox.top = region->y;
    srcBox.front = region->z;
    srcBox.right = region->x + region->width;
    srcBox.bottom = region->y + region->height;
    srcBox.back = region->z + region->depth;

    D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
    dstLoc.pResource = buffer->d3d12Resource;
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLoc.PlacedFootprint = footprint;

    computePass->commandBuffer->d3d12CommandList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, &srcBox);
}

void agfxComputePassCopyBufferToTexture(agfxComputePass* computePass, agfxBuffer* buffer, agfxTexture* texture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, uint32_t bytesPerRow, uint32_t bytesPerImage) {
    UINT subresourceIndex = D3D12CalcSubresource(mipLevel, layer, 0, texture->createInfo.mipLevels, texture->createInfo.depthOrArrayLayers);

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    footprint.Offset = 0; // Assuming bufferOffset is 0 for simplicity; adjust as needed
    footprint.Footprint.Format = agfxTextureFormatToDXGIFormat(texture->createInfo.format);
    footprint.Footprint.Width = region->width;
    footprint.Footprint.Height = region->height;
    footprint.Footprint.Depth = region->depth;
    footprint.Footprint.RowPitch = (UINT)bytesPerRow;

    D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
    srcLoc.pResource = buffer->d3d12Resource;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint = footprint;

    D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
    dstLoc.pResource = texture->d3d12Resource;
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = subresourceIndex;

    D3D12_BOX srcBox = {};
    srcBox.left = 0;
    srcBox.top = 0;
    srcBox.front = 0;
    srcBox.right = region->width;
    srcBox.bottom = region->height;
    srcBox.back = region->depth;

    computePass->commandBuffer->d3d12CommandList->CopyTextureRegion(&dstLoc, region->x, region->y, region->z, &srcLoc, &srcBox);
}

void agfxComputePassCopyBufferToBuffer(agfxComputePass* computePass, agfxBuffer* srcBuffer, agfxBuffer* dstBuffer, uint64_t srcOffset, uint64_t dstOffset, uint64_t size) {
    computePass->commandBuffer->d3d12CommandList->CopyBufferRegion(dstBuffer->d3d12Resource, dstOffset, srcBuffer->d3d12Resource, srcOffset, size);
}

void agfxComputePassCopyTextureToTexture(agfxComputePass* computePass, agfxTexture* srcTexture, agfxTexture* dstTexture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer) {
    UINT subresourceIndex = D3D12CalcSubresource(mipLevel, layer, 0, srcTexture->createInfo.mipLevels, srcTexture->createInfo.depthOrArrayLayers);

    D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
    srcLoc.pResource = srcTexture->d3d12Resource;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex = subresourceIndex;

    D3D12_BOX srcBox = {};
    srcBox.left = region->x;
    srcBox.top = region->y;
    srcBox.front = region->z;
    srcBox.right = region->x + region->width;
    srcBox.bottom = region->y + region->height;
    srcBox.back = region->z + region->depth;

    D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
    dstLoc.pResource = dstTexture->d3d12Resource;
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = subresourceIndex;

    computePass->commandBuffer->d3d12CommandList->CopyTextureRegion(&dstLoc, region->x, region->y, region->z, &srcLoc, &srcBox);
}

void agfxComputePassSetPipeline(agfxComputePass* computePass, agfxComputePipeline* pipeline) {
    computePass->commandBuffer->d3d12CommandList->SetComputeRootSignature(computePass->device->globalRootSignature);
    computePass->commandBuffer->d3d12CommandList->SetPipelineState(pipeline->d3d12PipelineState);
}

void agfxComputePassPushConstants(agfxComputePass* computePass, const void* data, uint32_t size) {
    computePass->commandBuffer->d3d12CommandList->SetComputeRoot32BitConstants(0, size / 4, data, 0);
}

void agfxComputePassDispatch(agfxComputePass* computePass, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    computePass->commandBuffer->d3d12CommandList->Dispatch(groupCountX, groupCountY, groupCountZ);
}

void agfxComputePassEnd(agfxComputePass* computePass) {
    computePass->device->createInfo.tempFree(computePass);
}

// Indirect bundle
struct agfxIndirectBundle {
    agfxIndirectBundleCreateInfo createInfo;
    agfxBuffer* commandsBuffer;
    agfxBufferView* commandsBufferView;
    agfxBuffer* countBuffer;
    agfxBufferView* countBufferView;
    uint32_t stride;
};

agfxIndirectBundle* agfxIndirectBundleCreate(agfxDevice* device, const agfxIndirectBundleCreateInfo* createInfo) {
    agfxIndirectBundle* bundle = (agfxIndirectBundle*)device->createInfo.allocate(sizeof(agfxIndirectBundle));
    memset(bundle, 0, sizeof(agfxIndirectBundle));
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

    return bundle;
}

void agfxIndirectBundleDestroy(agfxDevice* device, agfxIndirectBundle* bundle) {
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
    // No-op on D3D12: ExecuteIndirect reads the commands/count buffers directly. Kept as a real
    // call so callers don't need to branch per backend (Metal's ICB-conversion pass does the real
    // work here).
    (void)computePass;
    (void)bundle;
    (void)executeInfo;
}

void agfxRenderPassExecuteIndirectBundle(agfxRenderPass* renderPass, agfxIndirectBundle* bundle, const agfxIndirectBundleExecuteInfo* executeInfo) {
    ID3D12GraphicsCommandList* commandList = renderPass->commandBuffer->d3d12CommandList;

    commandList->SetGraphicsRootSignature(renderPass->commandBuffer->device->globalRootSignature);
    commandList->SetPipelineState(executeInfo->renderPipeline->d3d12PipelineState);
    commandList->IASetPrimitiveTopology(agfxTopologyToD3D12PrimitiveTopology(executeInfo->renderPipeline->createInfo.topology));
    commandList->SetGraphicsRoot32BitConstants(0, sizeof(executeInfo->pushConstants) / 4, executeInfo->pushConstants, 0);

    if (executeInfo->indexBuffer) {
        D3D12_INDEX_BUFFER_VIEW ibv = {};
        ibv.BufferLocation = executeInfo->indexBuffer->d3d12Resource->GetGPUVirtualAddress();
        ibv.Format = executeInfo->indexBuffer->createInfo.stride == 2 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
        ibv.SizeInBytes = executeInfo->indexBuffer->createInfo.size;
        commandList->IASetIndexBuffer(&ibv);
    }

    commandList->ExecuteIndirect(
        renderPass->commandBuffer->device->indirectSignatures[bundle->createInfo.type],
        executeInfo->maxCommandCount,
        bundle->commandsBuffer->d3d12Resource,
        (UINT64)executeInfo->commandOffset * bundle->stride,
        bundle->countBuffer->d3d12Resource,
        (UINT64)executeInfo->countIndex * sizeof(uint32_t));
}

void agfxComputePassExecuteIndirectBundle(agfxComputePass* computePass, agfxIndirectBundle* bundle, const agfxIndirectBundleExecuteInfo* executeInfo) {
    ID3D12GraphicsCommandList* commandList = computePass->commandBuffer->d3d12CommandList;

    commandList->SetComputeRootSignature(computePass->device->globalRootSignature);
    commandList->SetPipelineState(executeInfo->computePipeline->d3d12PipelineState);
    commandList->SetComputeRoot32BitConstants(0, sizeof(executeInfo->pushConstants) / 4, executeInfo->pushConstants, 0);

    commandList->ExecuteIndirect(
        computePass->device->indirectSignatures[bundle->createInfo.type],
        executeInfo->maxCommandCount,
        bundle->commandsBuffer->d3d12Resource,
        (UINT64)executeInfo->commandOffset * bundle->stride,
        bundle->countBuffer->d3d12Resource,
        (UINT64)executeInfo->countIndex * sizeof(uint32_t));
}

// Stuff
D3D12_RENDER_TARGET_VIEW_DESC agfxTextureViewTypeToD3D12RenderTargetViewDesc(agfxRenderTargetCreateInfo* createInfo) {
    D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = {};
    rtvDesc.Format = createInfo->format ? agfxTextureFormatToDXGIFormat(createInfo->format) : agfxTextureFormatToDXGIFormat(createInfo->texture->createInfo.format);
    if (createInfo->texture->createInfo.depthOrArrayLayers > 1) {
        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
        rtvDesc.Texture2DArray.MipSlice = createInfo->mipLevel;
        rtvDesc.Texture2DArray.FirstArraySlice = createInfo->arrayLayer;
        rtvDesc.Texture2DArray.ArraySize = 1;
    } else {
        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        rtvDesc.Texture2D.MipSlice = createInfo->mipLevel;
    }
    return rtvDesc;
}

D3D12_DEPTH_STENCIL_VIEW_DESC agfxTextureViewTypeToD3D12DepthStencilViewDesc(agfxRenderTargetCreateInfo* createInfo) {
    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
    dsvDesc.Format = createInfo->format ? agfxTextureFormatToDXGIFormat(createInfo->format) : agfxTextureFormatToDXGIFormat(createInfo->texture->createInfo.format);
    if (createInfo->texture->createInfo.depthOrArrayLayers > 1) {
        dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        dsvDesc.Texture2DArray.MipSlice = createInfo->mipLevel;
        dsvDesc.Texture2DArray.FirstArraySlice = createInfo->arrayLayer;
        dsvDesc.Texture2DArray.ArraySize = 1;
    } else {
        dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        dsvDesc.Texture2D.MipSlice = createInfo->mipLevel;
    }
    return dsvDesc;
}

D3D12_SHADER_RESOURCE_VIEW_DESC agfxBufferViewTypeToD3D12ShaderResourceViewDesc(agfxBufferViewCreateInfo* createInfo) {
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    if (createInfo->type == AGFX_BUFFER_VIEW_TYPE_RAW) {
		srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        srvDesc.Buffer.FirstElement = createInfo->offset / 4;
        srvDesc.Buffer.NumElements = (createInfo->buffer->createInfo.size - createInfo->offset) / 4;
        srvDesc.Buffer.StructureByteStride = 0;
    } else if (createInfo->type == AGFX_BUFFER_VIEW_TYPE_STRUCTURED) {
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Buffer.FirstElement = createInfo->offset / createInfo->buffer->createInfo.stride;
        srvDesc.Buffer.NumElements = (createInfo->buffer->createInfo.size - createInfo->offset) / createInfo->buffer->createInfo.stride;
        srvDesc.Buffer.StructureByteStride = createInfo->buffer->createInfo.stride;
        srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    }
    return srvDesc;
}

D3D12_UNORDERED_ACCESS_VIEW_DESC agfxBufferViewTypeToD3D12UnorderedAccessViewDesc(agfxBufferViewCreateInfo* createInfo) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    if (createInfo->type == AGFX_BUFFER_VIEW_TYPE_RAW) {
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        uavDesc.Buffer.FirstElement = createInfo->offset / 4;
        uavDesc.Buffer.NumElements = (createInfo->buffer->createInfo.size - createInfo->offset) / 4;
        uavDesc.Buffer.StructureByteStride = 0;
        uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    } else if (createInfo->type == AGFX_BUFFER_VIEW_TYPE_STRUCTURED) {
        uavDesc.Buffer.FirstElement = createInfo->offset / createInfo->buffer->createInfo.stride;
        uavDesc.Buffer.NumElements = (createInfo->buffer->createInfo.size - createInfo->offset) / createInfo->buffer->createInfo.stride;
        uavDesc.Buffer.StructureByteStride = createInfo->buffer->createInfo.stride;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    }
    return uavDesc;
}

D3D12_CONSTANT_BUFFER_VIEW_DESC agfxBufferViewTypeToD3D12ConstantBufferViewDesc(agfxBufferViewCreateInfo* createInfo) {
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
    cbvDesc.BufferLocation = createInfo->buffer->d3d12Resource->GetGPUVirtualAddress() + createInfo->offset;
    cbvDesc.SizeInBytes = (UINT)(createInfo->buffer->createInfo.size - createInfo->offset);
    return cbvDesc;
}

D3D12_COMPARISON_FUNC agfxComparisonFunctionToD3D12ComparisonFunc(agfxComparisonFunction func) {
    switch (func) {
        case AGFX_COMPARISON_FUNCTION_NEVER:
            return D3D12_COMPARISON_FUNC_NEVER;
        case AGFX_COMPARISON_FUNCTION_LESS:
            return D3D12_COMPARISON_FUNC_LESS;
        case AGFX_COMPARISON_FUNCTION_EQUAL:
            return D3D12_COMPARISON_FUNC_EQUAL;
        case AGFX_COMPARISON_FUNCTION_LESS_EQUAL:
            return D3D12_COMPARISON_FUNC_LESS_EQUAL;
        case AGFX_COMPARISON_FUNCTION_GREATER:
            return D3D12_COMPARISON_FUNC_GREATER;
        case AGFX_COMPARISON_FUNCTION_NOT_EQUAL:
            return D3D12_COMPARISON_FUNC_NOT_EQUAL;
        case AGFX_COMPARISON_FUNCTION_GREATER_EQUAL:
            return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
        case AGFX_COMPARISON_FUNCTION_ALWAYS:
            return D3D12_COMPARISON_FUNC_ALWAYS;
        default:
            return D3D12_COMPARISON_FUNC_ALWAYS;
    }
}

D3D12_PRIMITIVE_TOPOLOGY_TYPE agfxTopologyToD3D12PrimitiveTopologyType(agfxTopology topology) {
    switch (topology) {
        case AGFX_TOPOLOGY_TRIANGLES:
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        case AGFX_TOPOLOGY_LINES:
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        case AGFX_TOPOLOGY_POINTS:
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
        default:
            return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    }
}

D3D12_PRIMITIVE_TOPOLOGY agfxTopologyToD3D12PrimitiveTopology(agfxTopology topology) {
    switch (topology) {
        case AGFX_TOPOLOGY_TRIANGLES:
            return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        case AGFX_TOPOLOGY_LINES:
            return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
        case AGFX_TOPOLOGY_POINTS:
            return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
        default:
            return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    }
}

D3D12_CULL_MODE agfxCullModeToD3D12CullMode(agfxCullMode mode) {
    switch (mode) {
        case AGFX_CULL_MODE_NONE:
            return D3D12_CULL_MODE_NONE;
        case AGFX_CULL_MODE_FRONT:
            return D3D12_CULL_MODE_FRONT;
        case AGFX_CULL_MODE_BACK:
            return D3D12_CULL_MODE_BACK;
        default:
            return D3D12_CULL_MODE_NONE;
    }
}

D3D12_FILL_MODE agfxFillModeToD3D12FillMode(agfxFillMode mode) {
    switch (mode) {
        case AGFX_FILL_MODE_SOLID:
            return D3D12_FILL_MODE_SOLID;
        case AGFX_FILL_MODE_WIREFRAME:
            return D3D12_FILL_MODE_WIREFRAME;
        default:
            return D3D12_FILL_MODE_SOLID;
    }
}

BOOL agfxFrontFaceToD3D12FrontCounterClockwise(agfxFrontFace face) {
    return face == AGFX_FRONT_FACE_COUNTER_CLOCKWISE ? TRUE : FALSE;
}

D3D12_BLEND agfxBlendFactorToD3D12Blend(agfxBlendFactor factor) {
    switch (factor) {
        case AGFX_BLEND_FACTOR_ZERO:
            return D3D12_BLEND_ZERO;
        case AGFX_BLEND_FACTOR_ONE:
            return D3D12_BLEND_ONE;
        case AGFX_BLEND_FACTOR_SRC_COLOR:
            return D3D12_BLEND_SRC_COLOR;
        case AGFX_BLEND_FACTOR_ONE_MINUS_SRC_COLOR:
            return D3D12_BLEND_INV_SRC_COLOR;
        case AGFX_BLEND_FACTOR_DST_COLOR:
            return D3D12_BLEND_DEST_COLOR;
        case AGFX_BLEND_FACTOR_ONE_MINUS_DST_COLOR:
            return D3D12_BLEND_INV_DEST_COLOR;
        case AGFX_BLEND_FACTOR_SRC_ALPHA:
            return D3D12_BLEND_SRC_ALPHA;
        case AGFX_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:
            return D3D12_BLEND_INV_SRC_ALPHA;
        case AGFX_BLEND_FACTOR_DST_ALPHA:
            return D3D12_BLEND_DEST_ALPHA;
        case AGFX_BLEND_FACTOR_ONE_MINUS_DST_ALPHA:
            return D3D12_BLEND_INV_DEST_ALPHA;
        default:
            return D3D12_BLEND_ONE;
    }
}

D3D12_BLEND_OP agfxBlendOperationToD3D12BlendOp(agfxBlendOperation op) {
    switch (op) {
        case AGFX_BLEND_OPERATION_ADD:
            return D3D12_BLEND_OP_ADD;
        case AGFX_BLEND_OPERATION_SUBTRACT:
            return D3D12_BLEND_OP_SUBTRACT;
        case AGFX_BLEND_OPERATION_REVERSE_SUBTRACT:
            return D3D12_BLEND_OP_REV_SUBTRACT;
        case AGFX_BLEND_OPERATION_MIN:
            return D3D12_BLEND_OP_MIN;
        case AGFX_BLEND_OPERATION_MAX:
            return D3D12_BLEND_OP_MAX;
        default:
            return D3D12_BLEND_OP_ADD;
    }
}

D3D12_FILTER agfxSamplerFilterToD3D12Filter(agfxSamplerFilter filter, bool isComparison) {
    switch (filter) {
        case AGFX_SAMPLER_FILTER_NEAREST:
            return isComparison ? D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT : D3D12_FILTER_MIN_MAG_MIP_POINT;
        case AGFX_SAMPLER_FILTER_LINEAR:
            return isComparison ? D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR : D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        default:
            return isComparison ? D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT : D3D12_FILTER_MIN_MAG_MIP_POINT;
    }
}

D3D12_TEXTURE_ADDRESS_MODE agfxSamplerAddressModeToD3D12TextureAddressMode(agfxSamplerAddressMode mode) {
    switch (mode) {
        case AGFX_SAMPLER_ADDRESS_MODE_REPEAT:
            return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        case AGFX_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
            return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
        case AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
            return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        case AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
            return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
        default:
            return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    }
}

D3D12_SAMPLER_DESC agfxSamplerCreateInfoToD3D12SamplerDesc(agfxSamplerCreateInfo* createInfo) {
    D3D12_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = agfxSamplerFilterToD3D12Filter(createInfo->filter, createInfo->comparisonFunction != AGFX_COMPARISON_FUNCTION_ALWAYS);
    samplerDesc.AddressU = agfxSamplerAddressModeToD3D12TextureAddressMode(createInfo->addressModeU);
    samplerDesc.AddressV = agfxSamplerAddressModeToD3D12TextureAddressMode(createInfo->addressModeV);
    samplerDesc.AddressW = agfxSamplerAddressModeToD3D12TextureAddressMode(createInfo->addressModeW);
    samplerDesc.MipLODBias = createInfo->mipLodBias;
    samplerDesc.MaxAnisotropy = createInfo->maxAnisotropy;
    samplerDesc.ComparisonFunc = agfxComparisonFunctionToD3D12ComparisonFunc(createInfo->comparisonFunction);
    samplerDesc.MinLOD = createInfo->minLod;
    samplerDesc.MaxLOD = createInfo->maxLod;
    return samplerDesc;
}

D3D12_SHADER_RESOURCE_VIEW_DESC agfxTextureViewTypeToD3D12ShaderResourceViewDesc(agfxTextureViewCreateInfo* createInfo) {
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = createInfo->format == AGFX_TEXTURE_FORMAT_UNKNOWN ? agfxTextureFormatToDXGIFormat(createInfo->texture->createInfo.format) : agfxTextureFormatToDXGIFormat(createInfo->format);
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    switch (createInfo->type) {
        case AGFX_TEXTURE_TYPE_1D:
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            srvDesc.Texture1D.MostDetailedMip = createInfo->baseMipLevel;
            srvDesc.Texture1D.MipLevels = createInfo->mipLevelCount == AGFX_SUBRESOURCE_ALL_MIPS ? createInfo->texture->createInfo.mipLevels - createInfo->baseMipLevel : createInfo->mipLevelCount;
            break;    
        case AGFX_TEXTURE_TYPE_2D:
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srvDesc.Texture2D.MostDetailedMip = createInfo->baseMipLevel;
            srvDesc.Texture2D.MipLevels = createInfo->mipLevelCount == AGFX_SUBRESOURCE_ALL_MIPS ? createInfo->texture->createInfo.mipLevels - createInfo->baseMipLevel : createInfo->mipLevelCount;
            srvDesc.Texture2D.PlaneSlice = 0; // PlaneSlice addresses planar formats (e.g. NV12), not array layers.
            break;
        case AGFX_TEXTURE_TYPE_2D_ARRAY:
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            srvDesc.Texture2DArray.MostDetailedMip = createInfo->baseMipLevel;
            srvDesc.Texture2DArray.MipLevels = createInfo->mipLevelCount == AGFX_SUBRESOURCE_ALL_MIPS ? createInfo->texture->createInfo.mipLevels - createInfo->baseMipLevel : createInfo->mipLevelCount;
            srvDesc.Texture2DArray.FirstArraySlice = createInfo->baseArrayLayer;
            srvDesc.Texture2DArray.ArraySize = createInfo->arrayLayerCount == AGFX_SUBRESOURCE_ALL_LAYERS ? createInfo->texture->createInfo.depthOrArrayLayers - createInfo->baseArrayLayer : createInfo->arrayLayerCount;
            break;
        case AGFX_TEXTURE_TYPE_3D:
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            srvDesc.Texture3D.MostDetailedMip = createInfo->baseMipLevel;
            srvDesc.Texture3D.MipLevels = createInfo->mipLevelCount == AGFX_SUBRESOURCE_ALL_MIPS ? createInfo->texture->createInfo.mipLevels - createInfo->baseMipLevel : createInfo->mipLevelCount;
            srvDesc.Texture3D.ResourceMinLODClamp = 0.0f;
            break;
        case AGFX_TEXTURE_TYPE_CUBE:
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            srvDesc.TextureCube.MostDetailedMip = createInfo->baseMipLevel;
            srvDesc.TextureCube.MipLevels = createInfo->mipLevelCount == AGFX_SUBRESOURCE_ALL_MIPS ? createInfo->texture->createInfo.mipLevels - createInfo->baseMipLevel : createInfo->mipLevelCount;
            break;
    }
    return srvDesc;
}

D3D12_UNORDERED_ACCESS_VIEW_DESC agfxTextureViewTypeToD3D12UnorderedAccessViewDesc(agfxTextureViewCreateInfo* createInfo) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = createInfo->format == AGFX_TEXTURE_FORMAT_UNKNOWN ? agfxTextureFormatToDXGIFormat(createInfo->texture->createInfo.format) : agfxTextureFormatToDXGIFormat(createInfo->format);
    switch (createInfo->type) {
        case AGFX_TEXTURE_TYPE_1D:
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
            uavDesc.Texture1D.MipSlice = createInfo->baseMipLevel;
            break;
        case AGFX_TEXTURE_TYPE_2D:
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            uavDesc.Texture2D.MipSlice = createInfo->baseMipLevel;
            uavDesc.Texture2D.PlaneSlice = 0; // PlaneSlice addresses planar formats (e.g. NV12), not array layers.
            break;
        case AGFX_TEXTURE_TYPE_2D_ARRAY:
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            uavDesc.Texture2DArray.MipSlice = createInfo->baseMipLevel;
            uavDesc.Texture2DArray.FirstArraySlice = createInfo->baseArrayLayer;
            uavDesc.Texture2DArray.ArraySize = createInfo->arrayLayerCount == AGFX_SUBRESOURCE_ALL_LAYERS ? createInfo->texture->createInfo.depthOrArrayLayers - createInfo->baseArrayLayer : createInfo->arrayLayerCount;
            break;
        case AGFX_TEXTURE_TYPE_3D:
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            uavDesc.Texture3D.MipSlice = createInfo->baseMipLevel;
            uavDesc.Texture3D.FirstWSlice = 0;
            uavDesc.Texture3D.WSize = createInfo->texture->createInfo.depthOrArrayLayers;
            break;
        case AGFX_TEXTURE_TYPE_CUBE:
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            uavDesc.Texture2DArray.MipSlice = createInfo->baseMipLevel;
            uavDesc.Texture2DArray.FirstArraySlice = createInfo->baseArrayLayer;
            uavDesc.Texture2DArray.ArraySize = createInfo->arrayLayerCount == AGFX_SUBRESOURCE_ALL_LAYERS ? createInfo->texture->createInfo.depthOrArrayLayers - createInfo->baseArrayLayer : createInfo->arrayLayerCount;
            break;
    }
    return uavDesc;
}

D3D12_RESOURCE_FLAGS agfxTextureUsageToD3D12ResourceFlags(agfxTextureUsage usage) {
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
    if (usage & AGFX_TEXTURE_USAGE_STORAGE)
        flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (usage & AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT)
        flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (usage & AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT)
        flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    return flags;
}

D3D12_RESOURCE_FLAGS agfxBufferUsageToD3D12ResourceFlags(agfxBufferUsage usage) {
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
    if (usage & AGFX_BUFFER_USAGE_SHADER_WRITE)
        flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return flags;
}

D3D12_HEAP_TYPE agfxBufferMemoryTypeToD3D12HeapType(agfxBufferMemoryType memoryType) {
    switch (memoryType) {
        case AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY:
            return D3D12_HEAP_TYPE_DEFAULT;
        case AGFX_BUFFER_MEMORY_TYPE_UPLOAD:
            return D3D12_HEAP_TYPE_UPLOAD;
        case AGFX_BUFFER_MEMORY_TYPE_READBACK:
            return D3D12_HEAP_TYPE_READBACK;
        default:
            return D3D12_HEAP_TYPE_DEFAULT;
    }
}

DXGI_FORMAT agfxTextureFormatToDXGIFormat(agfxTextureFormat format) {
    switch (format) {
        case AGFX_TEXTURE_FORMAT_R8_UNORM:
            return DXGI_FORMAT_R8_UNORM;
        case AGFX_TEXTURE_FORMAT_RG8_UNORM:
            return DXGI_FORMAT_R8G8_UNORM;
        case AGFX_TEXTURE_FORMAT_RGBA8_UNORM:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case AGFX_TEXTURE_FORMAT_BGRA8_UNORM:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        case AGFX_TEXTURE_FORMAT_RGBA8_UNORM_SRGB:
            return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        case AGFX_TEXTURE_FORMAT_BGRA8_UNORM_SRGB:
            return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        case AGFX_TEXTURE_FORMAT_R16_UNORM:
            return DXGI_FORMAT_R16_UNORM;
        case AGFX_TEXTURE_FORMAT_RG16_UNORM:
            return DXGI_FORMAT_R16G16_UNORM;
        case AGFX_TEXTURE_FORMAT_RGBA16_UNORM:
            return DXGI_FORMAT_R16G16B16A16_UNORM;
        case AGFX_TEXTURE_FORMAT_R16F:
            return DXGI_FORMAT_R16_FLOAT;
        case AGFX_TEXTURE_FORMAT_RG16F:
            return DXGI_FORMAT_R16G16_FLOAT;
        case AGFX_TEXTURE_FORMAT_RGBA16F:
            return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case AGFX_TEXTURE_FORMAT_R32F:
            return DXGI_FORMAT_R32_FLOAT;
        case AGFX_TEXTURE_FORMAT_RG32F:
            return DXGI_FORMAT_R32G32_FLOAT;
        case AGFX_TEXTURE_FORMAT_RGBA32F:
            return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case AGFX_TEXTURE_FORMAT_DEPTH32F:
            return DXGI_FORMAT_D32_FLOAT;
        case AGFX_TEXTURE_FORMAT_BC1_UNORM:
            return DXGI_FORMAT_BC1_UNORM;
        case AGFX_TEXTURE_FORMAT_BC1_UNORM_SRGB:
            return DXGI_FORMAT_BC1_UNORM_SRGB;
        case AGFX_TEXTURE_FORMAT_BC3_UNORM:
            return DXGI_FORMAT_BC3_UNORM;
        case AGFX_TEXTURE_FORMAT_BC3_UNORM_SRGB:
            return DXGI_FORMAT_BC3_UNORM_SRGB;
        case AGFX_TEXTURE_FORMAT_BC4_UNORM:
            return DXGI_FORMAT_BC4_UNORM;
        case AGFX_TEXTURE_FORMAT_BC5_UNORM:
            return DXGI_FORMAT_BC5_UNORM;
        case AGFX_TEXTURE_FORMAT_BC6H_UFLOAT:
            return DXGI_FORMAT_BC6H_UF16;
        case AGFX_TEXTURE_FORMAT_BC7_UNORM:
            return DXGI_FORMAT_BC7_UNORM;
        case AGFX_TEXTURE_FORMAT_BC7_UNORM_SRGB:
            return DXGI_FORMAT_BC7_UNORM_SRGB;
        default:
            // ASTC has no DXGI_FORMAT equivalent; D3D12 does not support ASTC textures.
            return DXGI_FORMAT_UNKNOWN;
    }
}

D3D12_RESOURCE_DIMENSION agfxTextureTypeToD3D12ResourceDimension(agfxTextureType type) {
    switch (type) {
        case AGFX_TEXTURE_TYPE_1D:
            return D3D12_RESOURCE_DIMENSION_TEXTURE1D;
        case AGFX_TEXTURE_TYPE_2D:
            return D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        case AGFX_TEXTURE_TYPE_2D_ARRAY:
            return D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        case AGFX_TEXTURE_TYPE_3D:
            return D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        case AGFX_TEXTURE_TYPE_CUBE:
            return D3D12_RESOURCE_DIMENSION_TEXTURE2D; // Cubes are treated as 2D arrays in D3D12
        default:
            return D3D12_RESOURCE_DIMENSION_UNKNOWN;
    }
}

D3D12_COMMAND_LIST_TYPE agfxCommandQueueTypeToD3D12CommandListType(agfxCommandQueueType type) {
    switch (type) {
        case AGFX_COMMAND_QUEUE_TYPE_GRAPHICS:
            return D3D12_COMMAND_LIST_TYPE_DIRECT;
        case AGFX_COMMAND_QUEUE_TYPE_COMPUTE:
            return D3D12_COMMAND_LIST_TYPE_COMPUTE;
        case AGFX_COMMAND_QUEUE_TYPE_TRANSFER:
            return D3D12_COMMAND_LIST_TYPE_COPY;
        default:
            return D3D12_COMMAND_LIST_TYPE_DIRECT; // Default to graphics if unknown
    }
}

D3D12_RESOURCE_STATES agfxResourceStateToD3D12ResourceStates(agfxResourceState state) {
    switch (state) {
        case AGFX_RESOURCE_STATE_COMMON:
            return D3D12_RESOURCE_STATE_COMMON;
        case AGFX_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER:
            return D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        case AGFX_RESOURCE_STATE_INDEX_BUFFER:
            return D3D12_RESOURCE_STATE_INDEX_BUFFER;
        case AGFX_RESOURCE_STATE_RENDER_TARGET:
            return D3D12_RESOURCE_STATE_RENDER_TARGET;
        case AGFX_RESOURCE_STATE_UNORDERED_ACCESS:
            return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        case AGFX_RESOURCE_STATE_DEPTH_WRITE:
            return D3D12_RESOURCE_STATE_DEPTH_WRITE;
        case AGFX_RESOURCE_STATE_DEPTH_READ:
            return D3D12_RESOURCE_STATE_DEPTH_READ;
        case AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:
            return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        case AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:
            return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        case AGFX_RESOURCE_STATE_INDIRECT_ARGUMENT:
            return D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        case AGFX_RESOURCE_STATE_COPY_DEST:
            return D3D12_RESOURCE_STATE_COPY_DEST;
        case AGFX_RESOURCE_STATE_COPY_SOURCE:
            return D3D12_RESOURCE_STATE_COPY_SOURCE;
        case AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE:
            return D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        case AGFX_RESOURCE_STATE_GENERIC_READ:
            return D3D12_RESOURCE_STATE_GENERIC_READ;
        case AGFX_RESOURCE_STATE_ALL_SHADER_RESOURCE:
            return D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
        case AGFX_RESOURCE_STATE_PRESENT:
            return D3D12_RESOURCE_STATE_PRESENT;
        default:
            return D3D12_RESOURCE_STATE_COMMON; // Default to common if unknown
    }
}
