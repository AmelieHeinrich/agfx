/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-08-01 10:21:06
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "agfx.h"

#define VK_NO_PROTOTYPES
#include "vk/volk.h"
#include "vk/volk.c"

#define AGFX_EXPOSE_VULKAN
#define AGFX_NATIVE_NO_INCLUDES
#include "agfx_native.h"

#include <new>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <vector>

// Bindless descriptor counts. Mirrors the fixed-size, single-descriptor-set layout used in a prior
// Vulkan renderer: one binding per resource class rather than one set per class, since a mutable
// descriptor type lets binding 0 hold every buffer/image type behind a single array.
static constexpr uint32_t kMaxBindlessResources = 400'000;
static constexpr uint32_t kMaxBindlessSamplers = 2'000;
static constexpr uint32_t kMaxBindlessAccelerationStructures = 8;

static const VkDescriptorType kBindlessResourceTypes[] = {
    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
    VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
};

// Copy of the bitmap/free-list slot allocator used by the D3D12 backend's descriptor manager
// (agfx_d3d12.cpp) -- same allocation shape, just handing out indices into Vulkan descriptor arrays
// instead of D3D12 descriptor heap slots.
struct agfxSlotAllocator {
    agfxSlotAllocator(uint64_t maxSlots) {
        this->maxSlots = maxSlots;
        bitmapSize = (maxSlots + 63) / 64;

        bitmap.resize(bitmapSize, 0);
        freeSlots.reserve(maxSlots);
        for (uint32_t i = 0; i < static_cast<uint32_t>(maxSlots); ++i) {
            freeSlots.push_back(i);
        }
    }

    uint64_t allocate() {
        if (freeSlots.empty()) return UINT64_MAX;
        int32_t slot = freeSlots.back();
        freeSlots.pop_back();
        setBit(slot);
        return slot;
    }

    void free(uint64_t slot) {
        if (slot >= maxSlots) return;
        if (testBit(static_cast<int>(slot))) {
            clearBit(static_cast<int>(slot));
            freeSlots.push_back(static_cast<uint32_t>(slot));
        }
    }

    void setBit(int index) { bitmap[index / 64] |= (1ULL << (index % 64)); }
    void clearBit(int index) { bitmap[index / 64] &= ~(1ULL << (index % 64)); }
    bool testBit(int index) { return (bitmap[index / 64] & (1ULL << (index % 64))) != 0; }

    uint64_t maxSlots;
    uint64_t bitmapSize;
    std::vector<uint64_t> bitmap;
    std::vector<uint32_t> freeSlots;
};

// Texture: struct defined up here (rather than down by its own create/destroy functions, where the
// rest of this file's sections put theirs) because agfxCommandBufferTextureBarrier/AliasingBarrier
// need the complete type earlier in this translation unit.
struct agfxTexture {
    agfxTextureCreateInfo createInfo;
    VkImage vkImage = VK_NULL_HANDLE;
    // Owned dedicated allocation; VK_NULL_HANDLE when placed in a heap, since the heap owns the memory.
    VkDeviceMemory vkMemory = VK_NULL_HANDLE;
    // Stashed at create time so agfxTextureSetName (no agfxDevice parameter in the public API) can
    // still reach vkSetDebugUtilsObjectNameEXT, which requires a VkDevice.
    agfxDevice* device = nullptr;
    // False for swap chain back buffer wrappers: their VkImages belong to the VkSwapchainKHR and
    // must not be vkDestroyImage'd when the wrapper dies.
    bool ownsImage = true;
};

// Heap: defined up here (like agfxTexture above) because agfxTextureCreate/agfxBufferCreate need the
// complete type to read heap->vkMemory/heap->createInfo when placing a resource.
struct agfxHeap {
    agfxHeapCreateInfo createInfo;
    VkDeviceMemory vkMemory = VK_NULL_HANDLE;
};

// Compute pipeline: defined up here too, because agfxComputePassSetPipeline (in the Compute pass
// section, well before the Compute pipeline section further down) needs the complete type to read
// pipeline->vkPipeline.
struct agfxComputePipeline {
    VkPipeline vkPipeline = VK_NULL_HANDLE;
    VkPipelineCache vkPipelineCache = VK_NULL_HANDLE;
};

// Render pipeline: defined up here for the same reason -- agfxRenderPassSetPipeline (Render pass
// section) needs the complete type to read pipeline->vkPipeline.
struct agfxRenderPipeline {
    VkPipeline vkPipeline = VK_NULL_HANDLE;
    VkPipelineCache vkPipelineCache = VK_NULL_HANDLE;
};

// Buffer: defined up here because the compute-pass copy functions and agfxRenderPassDrawIndexed
// (both well before the Buffer section) need the complete type to read buffer->vkBuffer/createInfo.
struct agfxBuffer {
    agfxBufferCreateInfo createInfo;
    VkBuffer vkBuffer = VK_NULL_HANDLE;
    // Owned dedicated allocation; VK_NULL_HANDLE when placed in a heap, since the heap owns the memory.
    VkDeviceMemory vkMemory = VK_NULL_HANDLE;
    // Memory actually bound -- buffer->vkMemory for a committed buffer, or the heap's memory for a
    // placed one -- plus the offset into it. What Map/Unmap actually operate on.
    VkDeviceMemory boundMemory = VK_NULL_HANDLE;
    VkDeviceSize boundOffset = 0;
    // Stashed at create time: agfxBufferMap/Unmap/SetName take no agfxDevice parameter in the public API.
    agfxDevice* device = nullptr;
};

// Device
struct agfxDevice {
    agfxDeviceCreateInfo createInfo{};

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    uint32_t graphicsFamily = UINT32_MAX;
    uint32_t computeFamily = UINT32_MAX;
    uint32_t transferFamily = UINT32_MAX;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkQueue transferQueue = VK_NULL_HANDLE;

    // One set, three bindings: 0 = mutable resources (CBV/SRV/UAV-equivalent), 1 = samplers,
    // 2 = acceleration structures (only present if supportsRayTracing).
    VkDescriptorSetLayout globalSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool globalPool = VK_NULL_HANDLE;
    VkDescriptorSet globalSet = VK_NULL_HANDLE;
    VkPipelineLayout globalPipelineLayout = VK_NULL_HANDLE;

    agfxSlotAllocator resourceSlotAllocator;
    agfxSlotAllocator samplerSlotAllocator;
    agfxSlotAllocator accelSlotAllocator;

    VkPhysicalDeviceMemoryProperties memoryProperties{};

    agfxBool supportsRayTracing = 0;
    agfxBool supportsMeshShaders = 0;
    agfxBool supportsMultiDrawIndirect = 0;

    agfxDevice()
        : resourceSlotAllocator(kMaxBindlessResources)
        , samplerSlotAllocator(kMaxBindlessSamplers)
        , accelSlotAllocator(kMaxBindlessAccelerationStructures)
    {}
};

template<typename T>
static T* AgfxAlloc(agfxDevice* device)
{
    T* object = (T*)device->createInfo.allocate(sizeof(T), device->createInfo.userData);
    new (object) T();
    return object;
}

template<typename T>
static void AgfxFree(agfxDevice* device, T* object)
{
    if (!object)
        return;
    object->~T();
    device->createInfo.free(object, device->createInfo.userData);
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

// Vulkan analogue of the D3D12 backend's agfxDescriptorManager::writeSRV/writeUAV/writeCBV: allocate
// a slot in the bindless mutable array (set 0, binding 0 -- what HLSL's ResourceDescriptorHeap
// indexes) and write one descriptor into it. The returned slot IS the shader-visible handle;
// UINT64_MAX means the array is exhausted. Legal while command buffers are in flight thanks to
// UPDATE_AFTER_BIND on every binding.
static uint64_t agfxVkWriteResourceDescriptor(agfxDevice* device, VkDescriptorType type, const VkDescriptorImageInfo* imageInfo, const VkDescriptorBufferInfo* bufferInfo)
{
    uint64_t slot = device->resourceSlotAllocator.allocate();
    if (slot == UINT64_MAX) return UINT64_MAX;

    VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = device->globalSet;
    write.dstBinding = 0;
    write.dstArrayElement = static_cast<uint32_t>(slot);
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pImageInfo = imageInfo;
    write.pBufferInfo = bufferInfo;
    vkUpdateDescriptorSets(device->device, 1, &write, 0, nullptr);
    return slot;
}

// Same as above for set 0, binding 1 (HLSL's SamplerDescriptorHeap).
static uint64_t agfxVkWriteSamplerDescriptor(agfxDevice* device, VkSampler sampler)
{
    uint64_t slot = device->samplerSlotAllocator.allocate();
    if (slot == UINT64_MAX) return UINT64_MAX;

    VkDescriptorImageInfo imageInfo = {};
    imageInfo.sampler = sampler;

    VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = device->globalSet;
    write.dstBinding = 1;
    write.dstArrayElement = static_cast<uint32_t>(slot);
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(device->device, 1, &write, 0, nullptr);
    return slot;
}

static bool agfxVkHasExtension(const std::vector<VkExtensionProperties>& extensions, const char* name)
{
    for (const VkExtensionProperties& extension : extensions) {
        if (strcmp(extension.extensionName, name) == 0) return true;
    }
    return false;
}

// strncpy(dst, src, dstSize - 1) trips -Wstringop-truncation when src and dst are same-sized fixed
// buffers (GCC can't prove src is shorter), which -Werror turns fatal -- copy the bounded length by
// hand instead. srcMaxSize guards against a driver that doesn't null-terminate within its own array.
static void agfxVkCopyBoundedString(char* dst, size_t dstSize, const char* src, size_t srcMaxSize)
{
    size_t len = strnlen(src, srcMaxSize);
    if (len >= dstSize) len = dstSize - 1;
    memcpy(dst, src, len);
    dst[len] = '\0';
}

// Finds one family per agfxCommandQueueType up front, since Vulkan (unlike D3D12) needs every queue
// requested at vkCreateDevice time -- agfxCommandQueueCreate can only ever hand out queues from
// families decided here. Falls back to sharing families when the hardware has no dedicated one.
static bool agfxVkSelectQueueFamilies(VkPhysicalDevice physicalDevice, uint32_t* outGraphics, uint32_t* outCompute, uint32_t* outTransfer)
{
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
    if (familyCount == 0) return false;

    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, families.data());

    uint32_t graphics = UINT32_MAX, compute = UINT32_MAX, transfer = UINT32_MAX;
    for (uint32_t i = 0; i < familyCount; ++i) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            graphics = i;
            break;
        }
    }
    if (graphics == UINT32_MAX) return false;
    if (families[graphics].timestampValidBits == 0) return false; // Timestamp queries are required.

    for (uint32_t i = 0; i < familyCount; ++i) {
        bool hasGraphics = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        bool hasCompute = (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        if (hasCompute && !hasGraphics) {
            compute = i;
            break;
        }
    }
    if (compute == UINT32_MAX) compute = graphics;

    for (uint32_t i = 0; i < familyCount; ++i) {
        bool hasGraphics = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        bool hasCompute = (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        if (!hasGraphics && !hasCompute) {
            transfer = i;
            break;
        }
    }
    if (transfer == UINT32_MAX) transfer = compute;

    *outGraphics = graphics;
    *outCompute = compute;
    *outTransfer = transfer;
    return true;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL agfxVkDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* userData)
{
    agfxDevice* device = (agfxDevice*)userData;
    agfxLogSeverity agfxSeverity = AGFX_LOG_SEVERITY_INFO;
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) agfxSeverity = AGFX_LOG_SEVERITY_WARNING;
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) agfxSeverity = AGFX_LOG_SEVERITY_ERROR;
    agfxLog(device, agfxSeverity, "%s", callbackData->pMessage);
    return VK_FALSE;
}

// Tears down whatever prefix of device creation has already succeeded, then frees the device and
// returns nullptr. Mirrors the fatal-unwind-on-failure style used by agfx_d3d12.cpp's agfxDeviceCreate.
static agfxDevice* agfxVkDeviceCreateFail(agfxDevice* device, const agfxDeviceCreateInfo* createInfo, const char* message)
{
    agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxDeviceCreate: %s", message);
    if (device->globalPipelineLayout) vkDestroyPipelineLayout(device->device, device->globalPipelineLayout, nullptr);
    if (device->globalPool) vkDestroyDescriptorPool(device->device, device->globalPool, nullptr);
    if (device->globalSetLayout) vkDestroyDescriptorSetLayout(device->device, device->globalSetLayout, nullptr);
    if (device->device) vkDestroyDevice(device->device, nullptr);
    if (device->debugMessenger) vkDestroyDebugUtilsMessengerEXT(device->instance, device->debugMessenger, nullptr);
    if (device->instance) vkDestroyInstance(device->instance, nullptr);
    device->~agfxDevice();
    createInfo->free(device, createInfo->userData);
    return nullptr;
}

agfxDevice* agfxDeviceCreate(const agfxDeviceCreateInfo* createInfo)
{
    agfxDevice* device = (agfxDevice*)createInfo->allocate(sizeof(agfxDevice), createInfo->userData);
    new (device) agfxDevice();
    memcpy(&device->createInfo, createInfo, sizeof(*createInfo));

    if (volkInitialize() != VK_SUCCESS) {
        return agfxVkDeviceCreateFail(device, createInfo, "failed to initialize volk (Vulkan loader not found)");
    }

    // Instance
    VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    applicationInfo.pApplicationName = "AGFX";
    applicationInfo.apiVersion = VK_API_VERSION_1_4;

    std::vector<const char*> instanceExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };
    switch (createInfo->displayServerProtocol) {
        case AGFX_DISPLAY_SERVER_PROTOCOL_X11: instanceExtensions.push_back("VK_KHR_xlib_surface"); break;
        case AGFX_DISPLAY_SERVER_PROTOCOL_XCB: instanceExtensions.push_back("VK_KHR_xcb_surface"); break;
        case AGFX_DISPLAY_SERVER_PROTOCOL_WAYLAND: instanceExtensions.push_back("VK_KHR_wayland_surface"); break;
    }

    uint32_t availableInstanceExtensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &availableInstanceExtensionCount, nullptr);
    std::vector<VkExtensionProperties> availableInstanceExtensions(availableInstanceExtensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &availableInstanceExtensionCount, availableInstanceExtensions.data());

    // Drop any requested extension the loader doesn't actually report -- keeps device creation
    // working headless/CI environments without X11/Wayland dev packages, since surface creation
    // itself is a later (swap chain) phase, not something this function needs to succeed.
    for (size_t i = 0; i < instanceExtensions.size();) {
        if (!agfxVkHasExtension(availableInstanceExtensions, instanceExtensions[i])) {
            agfxLog(device, AGFX_LOG_SEVERITY_WARNING, "agfxDeviceCreate: instance extension %s not available, surface creation for this protocol will not work", instanceExtensions[i]);
            instanceExtensions.erase(instanceExtensions.begin() + i);
        } else {
            ++i;
        }
    }

    bool debugUtilsAvailable = agfxVkHasExtension(availableInstanceExtensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (createInfo->enableValidation && debugUtilsAvailable) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    uint32_t availableLayerCount = 0;
    vkEnumerateInstanceLayerProperties(&availableLayerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(availableLayerCount);
    vkEnumerateInstanceLayerProperties(&availableLayerCount, availableLayers.data());
    bool validationLayerAvailable = false;
    for (const VkLayerProperties& layer : availableLayers) {
        if (strcmp(layer.layerName, validationLayer) == 0) { validationLayerAvailable = true; break; }
    }

    VkInstanceCreateInfo instanceCreateInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instanceCreateInfo.pApplicationInfo = &applicationInfo;
    instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
    instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
    if (createInfo->enableValidation && validationLayerAvailable) {
        instanceCreateInfo.enabledLayerCount = 1;
        instanceCreateInfo.ppEnabledLayerNames = &validationLayer;
    } else if (createInfo->enableValidation) {
        agfxLog(device, AGFX_LOG_SEVERITY_WARNING, "agfxDeviceCreate: VK_LAYER_KHRONOS_validation not available, validation will not be enabled");
    }

    if (vkCreateInstance(&instanceCreateInfo, nullptr, &device->instance) != VK_SUCCESS) {
        return agfxVkDeviceCreateFail(device, createInfo, "vkCreateInstance failed");
    }
    volkLoadInstance(device->instance);

    if (createInfo->enableValidation && debugUtilsAvailable) {
        VkDebugUtilsMessengerCreateInfoEXT messengerInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
        messengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        messengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        messengerInfo.pfnUserCallback = agfxVkDebugCallback;
        messengerInfo.pUserData = device;
        if (vkCreateDebugUtilsMessengerEXT(device->instance, &messengerInfo, nullptr, &device->debugMessenger) != VK_SUCCESS) {
            agfxLog(device, AGFX_LOG_SEVERITY_WARNING, "agfxDeviceCreate: vkCreateDebugUtilsMessengerEXT failed");
        }
    }

    // Physical device: rank discrete GPUs first, and only accept one that has everything required.
    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(device->instance, &physicalDeviceCount, nullptr);
    if (physicalDeviceCount == 0) {
        return agfxVkDeviceCreateFail(device, createInfo, "no Vulkan-capable physical devices found");
    }
    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(device->instance, &physicalDeviceCount, physicalDevices.data());

    uint32_t selectedGraphicsFamily = UINT32_MAX, selectedComputeFamily = UINT32_MAX, selectedTransferFamily = UINT32_MAX;
    std::vector<VkExtensionProperties> selectedDeviceExtensions;
    bool rayTracingSupported = false, meshShadersSupported = false;

    auto tryCandidate = [&](VkPhysicalDevice candidate) -> bool {
        VkPhysicalDeviceProperties properties = {};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (properties.apiVersion < VK_API_VERSION_1_4) return false;

        VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT mutableFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT };
        VkPhysicalDeviceVulkan13Features features13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
        features13.pNext = &mutableFeatures;
        VkPhysicalDeviceVulkan12Features features12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        features12.pNext = &features13;
        VkPhysicalDeviceVulkan11Features features11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
        features11.pNext = &features12;
        VkPhysicalDeviceFeatures2 features2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        features2.pNext = &features11;
        vkGetPhysicalDeviceFeatures2(candidate, &features2);

        if (!features11.shaderDrawParameters) return false;
        if (!features13.dynamicRendering) return false;
        if (!features13.synchronization2) return false; // Barriers are recorded as vkCmdPipelineBarrier2.
        if (!features12.descriptorIndexing || !features12.shaderSampledImageArrayNonUniformIndexing ||
            !features12.descriptorBindingPartiallyBound || !features12.descriptorBindingVariableDescriptorCount ||
            !features12.runtimeDescriptorArray || !features12.descriptorBindingUpdateUnusedWhilePending ||
            !features12.descriptorBindingSampledImageUpdateAfterBind) return false; // The global set's sampler binding is UPDATE_AFTER_BIND.
        if (!features12.bufferDeviceAddress) return false;
        if (!features12.timelineSemaphore) return false; // agfxFence is a timeline semaphore.
        if (!mutableFeatures.mutableDescriptorType) return false;

        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, extensions.data());

        if (!agfxVkHasExtension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) return false;
        if (!agfxVkHasExtension(extensions, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME)) return false;

        uint32_t graphicsFamily, computeFamily, transferFamily;
        if (!agfxVkSelectQueueFamilies(candidate, &graphicsFamily, &computeFamily, &transferFamily)) return false;

        selectedGraphicsFamily = graphicsFamily;
        selectedComputeFamily = computeFamily;
        selectedTransferFamily = transferFamily;
        selectedDeviceExtensions = extensions;
        rayTracingSupported = agfxVkHasExtension(extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
                               agfxVkHasExtension(extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
                               agfxVkHasExtension(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) &&
                               agfxVkHasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        if (rayTracingSupported) {
            // Binding 2 of the global set is UPDATE_AFTER_BIND, so extension presence alone isn't
            // enough -- the optional UAB feature must be there too or the binding flag is invalid.
            // rayTracingPipeline is required as well: the AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE
            // barrier mapping uses VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, which is only
            // valid with that feature enabled.
            VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
            VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
            accelFeatures.pNext = &rtPipelineFeatures;
            VkPhysicalDeviceFeatures2 accelQuery = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
            accelQuery.pNext = &accelFeatures;
            vkGetPhysicalDeviceFeatures2(candidate, &accelQuery);
            rayTracingSupported = accelFeatures.accelerationStructure &&
                                  accelFeatures.descriptorBindingAccelerationStructureUpdateAfterBind &&
                                  rtPipelineFeatures.rayTracingPipeline;
        }
        meshShadersSupported = agfxVkHasExtension(extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME);
        return true;
    };

    for (VkPhysicalDeviceType preferredType : { VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU, VK_PHYSICAL_DEVICE_TYPE_CPU }) {
        for (VkPhysicalDevice candidate : physicalDevices) {
            VkPhysicalDeviceProperties properties = {};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            if (properties.deviceType != preferredType) continue;
            if (tryCandidate(candidate)) {
                device->physicalDevice = candidate;
                break;
            }
        }
        if (device->physicalDevice) break;
    }

    if (!device->physicalDevice) {
        return agfxVkDeviceCreateFail(device, createInfo, "no physical device supports the required Vulkan 1.4 feature set (shader draw parameters, dynamic rendering, descriptor indexing, buffer device address, mutable descriptor type, timestamp queries)");
    }

    device->graphicsFamily = selectedGraphicsFamily;
    device->computeFamily = selectedComputeFamily;
    device->transferFamily = selectedTransferFamily;
    device->supportsRayTracing = rayTracingSupported ? 1 : 0;
    device->supportsMeshShaders = meshShadersSupported ? 1 : 0;
    vkGetPhysicalDeviceMemoryProperties(device->physicalDevice, &device->memoryProperties);

    VkPhysicalDeviceFeatures2 enabledFeatures2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    VkPhysicalDeviceVulkan12Features supportedFeatures12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    enabledFeatures2.pNext = &supportedFeatures12;
    vkGetPhysicalDeviceFeatures2(device->physicalDevice, &enabledFeatures2); // Re-query to enable exactly what the hardware reports.
    // AGFX MDI needs both halves: multiDrawIndirect for several commands per call, drawIndirectCount
    // for the GPU-written count buffer (vkCmdDraw*IndirectCount).
    device->supportsMultiDrawIndirect = (enabledFeatures2.features.multiDrawIndirect && supportedFeatures12.drawIndirectCount) ? 1 : 0;

    // Enabling the RT/mesh extensions alone isn't enough -- their feature structs must also be
    // chained into vkCreateDevice or every vkCreateAccelerationStructureKHR/mesh pipeline use is
    // invalid, however tolerant a given driver happens to be.
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR enabledRtPipelineFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
    enabledRtPipelineFeatures.rayTracingPipeline = VK_TRUE;
    VkPhysicalDeviceRayQueryFeaturesKHR enabledRayQueryFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
    enabledRayQueryFeatures.rayQuery = VK_TRUE;
    enabledRayQueryFeatures.pNext = &enabledRtPipelineFeatures;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR enabledAccelFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
    enabledAccelFeatures.accelerationStructure = VK_TRUE;
    enabledAccelFeatures.descriptorBindingAccelerationStructureUpdateAfterBind = VK_TRUE;
    enabledAccelFeatures.pNext = &enabledRayQueryFeatures;
    VkPhysicalDeviceMeshShaderFeaturesEXT enabledMeshFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT };
    enabledMeshFeatures.meshShader = VK_TRUE;
    enabledMeshFeatures.taskShader = VK_TRUE;

    VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT enabledMutableFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT };
    enabledMutableFeatures.mutableDescriptorType = VK_TRUE;
    if (device->supportsRayTracing && device->supportsMeshShaders) {
        enabledMutableFeatures.pNext = &enabledMeshFeatures;
        enabledMeshFeatures.pNext = &enabledAccelFeatures;
    } else if (device->supportsRayTracing) {
        enabledMutableFeatures.pNext = &enabledAccelFeatures;
    } else if (device->supportsMeshShaders) {
        enabledMutableFeatures.pNext = &enabledMeshFeatures;
    }
    VkPhysicalDeviceVulkan13Features enabledFeatures13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    enabledFeatures13.pNext = &enabledMutableFeatures;
    enabledFeatures13.dynamicRendering = VK_TRUE;
    enabledFeatures13.synchronization2 = VK_TRUE;
    enabledFeatures13.shaderDemoteToHelperInvocation = VK_TRUE;
    VkPhysicalDeviceVulkan12Features enabledFeatures12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    enabledFeatures12.pNext = &enabledFeatures13;
    enabledFeatures12.descriptorIndexing = VK_TRUE;
    enabledFeatures12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    enabledFeatures12.descriptorBindingPartiallyBound = VK_TRUE;
    enabledFeatures12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    enabledFeatures12.runtimeDescriptorArray = VK_TRUE;
    enabledFeatures12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    enabledFeatures12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    enabledFeatures12.bufferDeviceAddress = VK_TRUE;
    enabledFeatures12.timelineSemaphore = VK_TRUE;
    // The shader compiler emits scalar-layout SPIR-V (-fvk-use-scalar-layout), so structured
    // buffer strides and push constant members are packed tighter than std430 allows.
    enabledFeatures12.scalarBlockLayout = VK_TRUE;
    enabledFeatures12.drawIndirectCount = device->supportsMultiDrawIndirect ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceVulkan11Features enabledFeatures11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    enabledFeatures11.pNext = &enabledFeatures12;
    enabledFeatures11.shaderDrawParameters = VK_TRUE;
    enabledFeatures2.pNext = &enabledFeatures11;
    enabledFeatures2.features.multiDrawIndirect = device->supportsMultiDrawIndirect;

    std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME };
    if (device->supportsRayTracing) {
        deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    }
    if (device->supportsMeshShaders) {
        deviceExtensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    }

    float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    for (uint32_t family : { device->graphicsFamily, device->computeFamily, device->transferFamily }) {
        bool alreadyRequested = false;
        for (const VkDeviceQueueCreateInfo& existing : queueCreateInfos) {
            if (existing.queueFamilyIndex == family) { alreadyRequested = true; break; }
        }
        if (alreadyRequested) continue;

        VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueInfo);
    }

    VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceCreateInfo.pNext = &enabledFeatures2;
    deviceCreateInfo.queueCreateInfoCount = (uint32_t)queueCreateInfos.size();
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (vkCreateDevice(device->physicalDevice, &deviceCreateInfo, nullptr, &device->device) != VK_SUCCESS) {
        return agfxVkDeviceCreateFail(device, createInfo, "vkCreateDevice failed");
    }
    volkLoadDevice(device->device);

    vkGetDeviceQueue(device->device, device->graphicsFamily, 0, &device->graphicsQueue);
    vkGetDeviceQueue(device->device, device->computeFamily, 0, &device->computeQueue);
    vkGetDeviceQueue(device->device, device->transferFamily, 0, &device->transferQueue);

    // Global bindless descriptor set: one set, bindings 0 (mutable resources), 1 (sampler), and,
    // if supported, 2 (acceleration structure) -- not three separate sets.
    VkDescriptorSetLayoutBinding bindings[3] = {};
    VkDescriptorBindingFlags bindingFlags[3] = {};
    uint32_t bindingCount = 0;

    VkMutableDescriptorTypeListEXT mutableTypeList = {};
    mutableTypeList.descriptorTypeCount = (uint32_t)(sizeof(kBindlessResourceTypes) / sizeof(kBindlessResourceTypes[0]));
    mutableTypeList.pDescriptorTypes = kBindlessResourceTypes;
    VkMutableDescriptorTypeCreateInfoEXT mutableTypeInfo = { VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT };
    mutableTypeInfo.mutableDescriptorTypeListCount = 1;
    mutableTypeInfo.pMutableDescriptorTypeLists = &mutableTypeList;

    bindings[bindingCount].binding = 0;
    bindings[bindingCount].descriptorType = VK_DESCRIPTOR_TYPE_MUTABLE_EXT;
    bindings[bindingCount].descriptorCount = kMaxBindlessResources;
    bindings[bindingCount].stageFlags = VK_SHADER_STAGE_ALL;
    bindingFlags[bindingCount] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    bindingCount++;

    bindings[bindingCount].binding = 1;
    bindings[bindingCount].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[bindingCount].descriptorCount = kMaxBindlessSamplers;
    bindings[bindingCount].stageFlags = VK_SHADER_STAGE_ALL;
    bindingFlags[bindingCount] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    bindingCount++;

    if (device->supportsRayTracing) {
        bindings[bindingCount].binding = 2;
        bindings[bindingCount].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[bindingCount].descriptorCount = kMaxBindlessAccelerationStructures;
        bindings[bindingCount].stageFlags = VK_SHADER_STAGE_ALL;
        bindingFlags[bindingCount] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        bindingCount++;
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
    bindingFlagsInfo.pNext = &mutableTypeInfo;
    bindingFlagsInfo.bindingCount = bindingCount;
    bindingFlagsInfo.pBindingFlags = bindingFlags;

    VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.pNext = &bindingFlagsInfo;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = bindingCount;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device->device, &layoutInfo, nullptr, &device->globalSetLayout) != VK_SUCCESS) {
        return agfxVkDeviceCreateFail(device, createInfo, "vkCreateDescriptorSetLayout failed");
    }

    VkDescriptorPoolSize poolSizes[3] = {
        { VK_DESCRIPTOR_TYPE_MUTABLE_EXT, kMaxBindlessResources },
        { VK_DESCRIPTOR_TYPE_SAMPLER, kMaxBindlessSamplers },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, kMaxBindlessAccelerationStructures },
    };
    uint32_t poolSizeCount = device->supportsRayTracing ? 3 : 2;

    VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = poolSizeCount;
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(device->device, &poolInfo, nullptr, &device->globalPool) != VK_SUCCESS) {
        return agfxVkDeviceCreateFail(device, createInfo, "vkCreateDescriptorPool failed");
    }

    VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = device->globalPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &device->globalSetLayout;

    if (vkAllocateDescriptorSets(device->device, &allocInfo, &device->globalSet) != VK_SUCCESS) {
        return agfxVkDeviceCreateFail(device, createInfo, "vkAllocateDescriptorSets failed");
    }

    // A single 128-byte push constant range covers both stages: unlike D3D12, Vulkan's required
    // shaderDrawParameters feature gives shaders the SPIR-V DrawIndex builtin natively, so there is
    // no need for a second constant carrying a manually patched draw ID for indirect draws.
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 128;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &device->globalSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(device->device, &pipelineLayoutInfo, nullptr, &device->globalPipelineLayout) != VK_SUCCESS) {
        return agfxVkDeviceCreateFail(device, createInfo, "vkCreatePipelineLayout failed");
    }

    return device;
}

void agfxDeviceDestroy(agfxDevice* device)
{
    if (device->globalPipelineLayout) vkDestroyPipelineLayout(device->device, device->globalPipelineLayout, nullptr);
    if (device->globalSet) vkFreeDescriptorSets(device->device, device->globalPool, 1, &device->globalSet);
    if (device->globalPool) vkDestroyDescriptorPool(device->device, device->globalPool, nullptr);
    if (device->globalSetLayout) vkDestroyDescriptorSetLayout(device->device, device->globalSetLayout, nullptr);
    if (device->device) vkDestroyDevice(device->device, nullptr);
    if (device->debugMessenger) vkDestroyDebugUtilsMessengerEXT(device->instance, device->debugMessenger, nullptr);
    if (device->instance) vkDestroyInstance(device->instance, nullptr);

    agfxFree freeFn = device->createInfo.free;
    device->~agfxDevice();
    freeFn(device);
}

void agfxDeviceGetInfo(agfxDevice* device, agfxDeviceInfo* info)
{
    *info = {};

    VkPhysicalDeviceDriverProperties driverProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
    VkPhysicalDeviceProperties2 properties2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    properties2.pNext = &driverProperties;
    vkGetPhysicalDeviceProperties2(device->physicalDevice, &properties2);

    agfxVkCopyBoundedString(info->name, sizeof(info->name), properties2.properties.deviceName, sizeof(properties2.properties.deviceName));
    agfxVkCopyBoundedString(info->driverVersion, sizeof(info->driverVersion), driverProperties.driverInfo, sizeof(driverProperties.driverInfo));

    info->supportsRayTracing = device->supportsRayTracing;
    info->supportsMeshShaders = device->supportsMeshShaders;
    info->supportsMultiDrawIndirect = device->supportsMultiDrawIndirect;
}

void agfxDeviceMakeResourcesResident(agfxDevice* device)
{
}

void agfxDeviceWaitIdle(agfxDevice* device)
{
    // Also retires swapchain acquire/present semaphores, which a per-queue fence wait would not.
    vkDeviceWaitIdle(device->device);
}

VkInstance agfxNativeGetVkInstance(agfxDevice* device) { return device->instance; }
VkPhysicalDevice agfxNativeGetVkPhysicalDevice(agfxDevice* device) { return device->physicalDevice; }
VkDevice agfxNativeGetVkDevice(agfxDevice* device) { return device->device; }

// Fence: a timeline semaphore. agfxFenceWait/Signal/GetCompletedValue take no device parameter, so
// the device is stashed on the fence itself for the host-side wait/signal/query calls to use.
struct agfxFence {
    agfxDevice* device;
    VkSemaphore semaphore;
};

agfxFence* agfxFenceCreate(agfxDevice* device)
{
    agfxFence* fence = AgfxAlloc<agfxFence>(device);
    fence->device = device;

    VkSemaphoreTypeCreateInfo typeInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    typeInfo.initialValue = 0;

    VkSemaphoreCreateInfo createInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    createInfo.pNext = &typeInfo;

    if (vkCreateSemaphore(device->device, &createInfo, nullptr, &fence->semaphore) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxFenceCreate: vkCreateSemaphore failed");
        AgfxFree(device, fence);
        return nullptr;
    }

    return fence;
}

void agfxFenceDestroy(agfxDevice* device, agfxFence* fence)
{
    if (fence->semaphore) vkDestroySemaphore(device->device, fence->semaphore, nullptr);
    AgfxFree(device, fence);
}

VkSemaphore agfxNativeGetVkSemaphore(agfxFence* fence) { return fence->semaphore; }

void agfxFenceWait(agfxFence* fence, uint64_t value, uint64_t timeout)
{
    // timeout is documented in milliseconds (matching the D3D12/WaitForSingleObject backend);
    // vkWaitSemaphores wants nanoseconds. UINT64_MAX is the infinite-wait sentinel in both domains.
    uint64_t timeoutNanoseconds = (timeout == UINT64_MAX) ? UINT64_MAX : timeout * 1'000'000ULL;

    VkSemaphoreWaitInfo waitInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &fence->semaphore;
    waitInfo.pValues = &value;
    vkWaitSemaphores(fence->device->device, &waitInfo, timeoutNanoseconds);
}

void agfxFenceSignal(agfxFence* fence, uint64_t value)
{
    VkSemaphoreSignalInfo signalInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO };
    signalInfo.semaphore = fence->semaphore;
    signalInfo.value = value;
    vkSignalSemaphore(fence->device->device, &signalInfo);
}

uint64_t agfxFenceGetCompletedValue(agfxFence* fence)
{
    uint64_t value = 0;
    vkGetSemaphoreCounterValue(fence->device->device, fence->semaphore, &value);
    return value;
}

// Query pool
// Defined further down (Acceleration structure section) but needed here for the readback buffer.
static bool agfxVkCreateDedicatedBuffer(agfxDevice* device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryFlags, VkBuffer* outBuffer, VkDeviceMemory* outMemory);

struct agfxQueryPool {
    VkQueryPool vkQueryPool = VK_NULL_HANDLE;
    // Host-visible destination for vkCmdCopyQueryPoolResults -- the Vulkan analogue of the D3D12
    // backend's readback-heap buffer, persistently mapped.
    VkBuffer readbackBuffer = VK_NULL_HANDLE;
    VkDeviceMemory readbackMemory = VK_NULL_HANDLE;
    uint64_t* mappedReadback = nullptr;
    uint32_t count = 0;
    // Nanoseconds per timestamp tick (VkPhysicalDeviceLimits::timestampPeriod); device-wide, unlike
    // D3D12's per-queue GetTimestampFrequency.
    float timestampPeriod = 1.0f;
};

agfxQueryPool* agfxQueryPoolCreate(agfxDevice* device, agfxCommandQueue* queue, const agfxQueryPoolCreateInfo* createInfo)
{
    agfxQueryPool* pool = AgfxAlloc<agfxQueryPool>(device);
    pool->count = createInfo->count;

    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(device->physicalDevice, &properties);
    pool->timestampPeriod = properties.limits.timestampPeriod;

    VkQueryPoolCreateInfo queryPoolInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = createInfo->count;
    if (vkCreateQueryPool(device->device, &queryPoolInfo, nullptr, &pool->vkQueryPool) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxQueryPoolCreate: vkCreateQueryPool failed");
        AgfxFree(device, pool);
        return nullptr;
    }

    if (!agfxVkCreateDedicatedBuffer(device, (VkDeviceSize)createInfo->count * sizeof(uint64_t),
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                     &pool->readbackBuffer, &pool->readbackMemory) ||
        vkMapMemory(device->device, pool->readbackMemory, 0, VK_WHOLE_SIZE, 0, (void**)&pool->mappedReadback) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxQueryPoolCreate: readback buffer creation failed");
        agfxQueryPoolDestroy(device, pool);
        return nullptr;
    }

    return pool;
}

void agfxQueryPoolDestroy(agfxDevice* device, agfxQueryPool* pool)
{
    if (pool->mappedReadback) vkUnmapMemory(device->device, pool->readbackMemory);
    if (pool->readbackBuffer) vkDestroyBuffer(device->device, pool->readbackBuffer, nullptr);
    if (pool->readbackMemory) vkFreeMemory(device->device, pool->readbackMemory, nullptr);
    if (pool->vkQueryPool) vkDestroyQueryPool(device->device, pool->vkQueryPool, nullptr);
    AgfxFree(device, pool);
}

// agfxCommandBufferWriteTimestamp / agfxCommandBufferResolveQueryPool are defined further down,
// after agfxCommandBuffer is a complete type.

void agfxQueryPoolReadback(agfxDevice* device, agfxQueryPool* pool, uint32_t firstIndex, uint32_t count, uint64_t* outTimestampsNanoseconds)
{
    for (uint32_t i = 0; i < count; ++i) {
        outTimestampsNanoseconds[i] = (uint64_t)((double)pool->mappedReadback[firstIndex + i] * (double)pool->timestampPeriod);
    }
}

// Command queue
struct agfxCommandQueue {
    agfxDevice* device;
    agfxCommandQueueType type;
    VkQueue vkQueue;
    uint32_t familyIndex;
};

agfxCommandQueue* agfxCommandQueueCreate(agfxDevice* device, const agfxCommandQueueCreateInfo* createInfo)
{
    agfxCommandQueue* queue = AgfxAlloc<agfxCommandQueue>(device);
    queue->device = device;
    queue->type = createInfo->type;

    // All three families were already selected and requested from vkCreateDevice at device
    // creation time (Vulkan, unlike D3D12, needs every queue declared up front) -- this just
    // resolves the VkQueue AGFX already has a handle to via vkGetDeviceQueue.
    switch (createInfo->type) {
        case AGFX_COMMAND_QUEUE_TYPE_GRAPHICS:
            queue->vkQueue = device->graphicsQueue;
            queue->familyIndex = device->graphicsFamily;
            break;
        case AGFX_COMMAND_QUEUE_TYPE_COMPUTE:
            queue->vkQueue = device->computeQueue;
            queue->familyIndex = device->computeFamily;
            break;
        case AGFX_COMMAND_QUEUE_TYPE_TRANSFER:
            queue->vkQueue = device->transferQueue;
            queue->familyIndex = device->transferFamily;
            break;
    }

    return queue;
}

void agfxCommandQueueDestroy(agfxDevice* device, agfxCommandQueue* queue)
{
    AgfxFree(device, queue);
}

VkQueue agfxNativeGetVkQueue(agfxCommandQueue* queue) { return queue->vkQueue; }

void agfxCommandQueueSignal(agfxCommandQueue* queue, agfxFence* fence, uint64_t value)
{
    // An empty batch whose only job is the timeline signal -- queue submission order guarantees
    // this lands after every command buffer already submitted to this queue.
    VkTimelineSemaphoreSubmitInfo timelineInfo = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &value;

    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.pNext = &timelineInfo;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &fence->semaphore;

    vkQueueSubmit(queue->vkQueue, 1, &submitInfo, VK_NULL_HANDLE);
}

void agfxCommandQueueWait(agfxCommandQueue* queue, agfxFence* fence, uint64_t value)
{
    // An empty batch that blocks on the timeline value before completing -- everything submitted
    // to this queue afterwards waits behind it, since a queue executes its submissions in order.
    VkTimelineSemaphoreSubmitInfo timelineInfo = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
    timelineInfo.waitSemaphoreValueCount = 1;
    timelineInfo.pWaitSemaphoreValues = &value;

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.pNext = &timelineInfo;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &fence->semaphore;
    submitInfo.pWaitDstStageMask = &waitStage;

    vkQueueSubmit(queue->vkQueue, 1, &submitInfo, VK_NULL_HANDLE);
}

// Defined further down, after agfxCommandBuffer is a complete type: needs commandBuffers[i]->commandBuffer.
void agfxCommandQueueSubmit(agfxCommandQueue* queue, agfxCommandBuffer** commandBuffers, uint32_t commandBufferCount);

// Resource state -> Vulkan sync2 mapping. Mirrors agfxResourceStateToD3D12BarrierSync/Access/Layout
// in agfx_d3d12.cpp switch-for-switch: VkPipelineStageFlags2/VkAccessFlags2 split sync scope from
// access scope exactly like D3D12_BARRIER_SYNC/D3D12_BARRIER_ACCESS do, so barriers translate 1:1.
// One deliberate difference: D3D12 clamps AccessAfter to COMMON when AccessBefore is COMMON, purely
// to satisfy a D3D12 global-barrier validation rule -- Vulkan has no such restriction, so it's omitted.
static constexpr VkPipelineStageFlags2 kAllShadingStages2 =
    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
    VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
static constexpr VkPipelineStageFlags2 kNonPixelShadingStages2 =
    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
    VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;

static VkPipelineStageFlags2 agfxResourceStateToVkPipelineStage(agfxResourceState state)
{
    switch (state) {
        case AGFX_RESOURCE_STATE_COMMON:
            return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        case AGFX_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER:
            return kAllShadingStages2;
        case AGFX_RESOURCE_STATE_INDEX_BUFFER:
            return VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
        case AGFX_RESOURCE_STATE_RENDER_TARGET:
            return VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        case AGFX_RESOURCE_STATE_UNORDERED_ACCESS:
            return kAllShadingStages2;
        case AGFX_RESOURCE_STATE_DEPTH_WRITE:
        case AGFX_RESOURCE_STATE_DEPTH_READ:
            return VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        case AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:
            return kNonPixelShadingStages2;
        case AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:
            return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        case AGFX_RESOURCE_STATE_INDIRECT_ARGUMENT:
            return VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        case AGFX_RESOURCE_STATE_COPY_DEST:
        case AGFX_RESOURCE_STATE_COPY_SOURCE:
            return VK_PIPELINE_STAGE_2_COPY_BIT;
        case AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE:
            return VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR;
        case AGFX_RESOURCE_STATE_GENERIC_READ:
            return kAllShadingStages2 | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
        case AGFX_RESOURCE_STATE_ALL_SHADER_RESOURCE:
            return kAllShadingStages2;
        case AGFX_RESOURCE_STATE_PRESENT:
            return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        default:
            return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
}

static VkAccessFlags2 agfxResourceStateToVkAccessFlags(agfxResourceState state)
{
    switch (state) {
        case AGFX_RESOURCE_STATE_COMMON:
            return VK_ACCESS_2_NONE;
        case AGFX_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER:
            return VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT;
        case AGFX_RESOURCE_STATE_INDEX_BUFFER:
            return VK_ACCESS_2_INDEX_READ_BIT;
        case AGFX_RESOURCE_STATE_RENDER_TARGET:
            return VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        case AGFX_RESOURCE_STATE_UNORDERED_ACCESS:
            return VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        case AGFX_RESOURCE_STATE_DEPTH_WRITE:
            return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        case AGFX_RESOURCE_STATE_DEPTH_READ:
            return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        case AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:
        case AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:
        case AGFX_RESOURCE_STATE_ALL_SHADER_RESOURCE:
            return VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        case AGFX_RESOURCE_STATE_INDIRECT_ARGUMENT:
            return VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        case AGFX_RESOURCE_STATE_COPY_DEST:
            return VK_ACCESS_2_TRANSFER_WRITE_BIT;
        case AGFX_RESOURCE_STATE_COPY_SOURCE:
            return VK_ACCESS_2_TRANSFER_READ_BIT;
        case AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE:
            return VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        case AGFX_RESOURCE_STATE_GENERIC_READ:
            return VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT |
                   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                   VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
        case AGFX_RESOURCE_STATE_PRESENT:
            return VK_ACCESS_2_NONE;
        default:
            return VK_ACCESS_2_NONE;
    }
}

// Layout is meaningful only for textures -- buffers and acceleration structures are barriered via
// agfxCommandBufferMemoryBarrier (a global VkMemoryBarrier2), which has no layout concept at all.
static VkImageLayout agfxResourceStateToVkImageLayout(agfxResourceState state)
{
    switch (state) {
        case AGFX_RESOURCE_STATE_COMMON:
            return VK_IMAGE_LAYOUT_GENERAL;
        case AGFX_RESOURCE_STATE_RENDER_TARGET:
            return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        case AGFX_RESOURCE_STATE_UNORDERED_ACCESS:
            return VK_IMAGE_LAYOUT_GENERAL;
        case AGFX_RESOURCE_STATE_DEPTH_WRITE:
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        case AGFX_RESOURCE_STATE_DEPTH_READ:
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        case AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:
        case AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE:
        case AGFX_RESOURCE_STATE_ALL_SHADER_RESOURCE:
            return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        case AGFX_RESOURCE_STATE_COPY_DEST:
            return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        case AGFX_RESOURCE_STATE_COPY_SOURCE:
            return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        case AGFX_RESOURCE_STATE_GENERIC_READ:
            return VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
        case AGFX_RESOURCE_STATE_PRESENT:
            return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        default:
            return VK_IMAGE_LAYOUT_GENERAL;
    }
}

static VkImageAspectFlags agfxTextureFormatToVkImageAspect(agfxTextureFormat format)
{
    return (format == AGFX_TEXTURE_FORMAT_DEPTH32F) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
}

// Mirrors agfxTextureFormatToDXGIFormat in agfx_d3d12.cpp switch-for-switch -- no shared format table
// exists between backends, each one rolls its own.
static VkFormat agfxTextureFormatToVkFormat(agfxTextureFormat format)
{
    switch (format) {
        case AGFX_TEXTURE_FORMAT_R8_UNORM:          return VK_FORMAT_R8_UNORM;
        case AGFX_TEXTURE_FORMAT_RG8_UNORM:         return VK_FORMAT_R8G8_UNORM;
        case AGFX_TEXTURE_FORMAT_RGBA8_UNORM:       return VK_FORMAT_R8G8B8A8_UNORM;
        case AGFX_TEXTURE_FORMAT_BGRA8_UNORM:       return VK_FORMAT_B8G8R8A8_UNORM;
        case AGFX_TEXTURE_FORMAT_RGBA8_UNORM_SRGB:  return VK_FORMAT_R8G8B8A8_SRGB;
        case AGFX_TEXTURE_FORMAT_BGRA8_UNORM_SRGB:  return VK_FORMAT_B8G8R8A8_SRGB;
        case AGFX_TEXTURE_FORMAT_R16_UNORM:         return VK_FORMAT_R16_UNORM;
        case AGFX_TEXTURE_FORMAT_RG16_UNORM:        return VK_FORMAT_R16G16_UNORM;
        case AGFX_TEXTURE_FORMAT_RGBA16_UNORM:      return VK_FORMAT_R16G16B16A16_UNORM;
        case AGFX_TEXTURE_FORMAT_R16F:              return VK_FORMAT_R16_SFLOAT;
        case AGFX_TEXTURE_FORMAT_RG16F:             return VK_FORMAT_R16G16_SFLOAT;
        case AGFX_TEXTURE_FORMAT_RGBA16F:           return VK_FORMAT_R16G16B16A16_SFLOAT;
        case AGFX_TEXTURE_FORMAT_R32F:              return VK_FORMAT_R32_SFLOAT;
        case AGFX_TEXTURE_FORMAT_RG32F:              return VK_FORMAT_R32G32_SFLOAT;
        case AGFX_TEXTURE_FORMAT_RGBA32F:           return VK_FORMAT_R32G32B32A32_SFLOAT;
        case AGFX_TEXTURE_FORMAT_DEPTH32F:          return VK_FORMAT_D32_SFLOAT;
        case AGFX_TEXTURE_FORMAT_BC1_UNORM:         return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case AGFX_TEXTURE_FORMAT_BC1_UNORM_SRGB:    return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
        case AGFX_TEXTURE_FORMAT_BC3_UNORM:         return VK_FORMAT_BC3_UNORM_BLOCK;
        case AGFX_TEXTURE_FORMAT_BC3_UNORM_SRGB:    return VK_FORMAT_BC3_SRGB_BLOCK;
        case AGFX_TEXTURE_FORMAT_BC4_UNORM:         return VK_FORMAT_BC4_UNORM_BLOCK;
        case AGFX_TEXTURE_FORMAT_BC5_UNORM:         return VK_FORMAT_BC5_UNORM_BLOCK;
        case AGFX_TEXTURE_FORMAT_BC6H_UFLOAT:       return VK_FORMAT_BC6H_UFLOAT_BLOCK;
        case AGFX_TEXTURE_FORMAT_BC7_UNORM:         return VK_FORMAT_BC7_UNORM_BLOCK;
        case AGFX_TEXTURE_FORMAT_BC7_UNORM_SRGB:    return VK_FORMAT_BC7_SRGB_BLOCK;
        case AGFX_TEXTURE_FORMAT_ASTC_4X4_UNORM:      return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
        case AGFX_TEXTURE_FORMAT_ASTC_4X4_UNORM_SRGB: return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
        case AGFX_TEXTURE_FORMAT_ASTC_8X8_UNORM:      return VK_FORMAT_ASTC_8x8_UNORM_BLOCK;
        case AGFX_TEXTURE_FORMAT_ASTC_8X8_UNORM_SRGB: return VK_FORMAT_ASTC_8x8_SRGB_BLOCK;
        default:                                    return VK_FORMAT_UNDEFINED;
    }
}

// Bytes per texel block plus block dimensions, for converting the public API's bytesPerRow/
// bytesPerImage (bytes) into VkBufferImageCopy::bufferRowLength/bufferImageHeight (texels).
// Uncompressed formats are 1x1 blocks; BC/ASTC formats return their block byte size and dimensions.
static void agfxTextureFormatGetBlockInfo(agfxTextureFormat format, uint32_t* bytesPerBlock, uint32_t* blockWidth, uint32_t* blockHeight)
{
    *blockWidth = 1;
    *blockHeight = 1;
    switch (format) {
        case AGFX_TEXTURE_FORMAT_R8_UNORM:          *bytesPerBlock = 1;  break;
        case AGFX_TEXTURE_FORMAT_RG8_UNORM:         *bytesPerBlock = 2;  break;
        case AGFX_TEXTURE_FORMAT_RGBA8_UNORM:
        case AGFX_TEXTURE_FORMAT_BGRA8_UNORM:
        case AGFX_TEXTURE_FORMAT_RGBA8_UNORM_SRGB:
        case AGFX_TEXTURE_FORMAT_BGRA8_UNORM_SRGB:  *bytesPerBlock = 4;  break;
        case AGFX_TEXTURE_FORMAT_R16_UNORM:         *bytesPerBlock = 2;  break;
        case AGFX_TEXTURE_FORMAT_RG16_UNORM:        *bytesPerBlock = 4;  break;
        case AGFX_TEXTURE_FORMAT_RGBA16_UNORM:      *bytesPerBlock = 8;  break;
        case AGFX_TEXTURE_FORMAT_R16F:              *bytesPerBlock = 2;  break;
        case AGFX_TEXTURE_FORMAT_RG16F:             *bytesPerBlock = 4;  break;
        case AGFX_TEXTURE_FORMAT_RGBA16F:           *bytesPerBlock = 8;  break;
        case AGFX_TEXTURE_FORMAT_R32F:              *bytesPerBlock = 4;  break;
        case AGFX_TEXTURE_FORMAT_RG32F:             *bytesPerBlock = 8;  break;
        case AGFX_TEXTURE_FORMAT_RGBA32F:           *bytesPerBlock = 16; break;
        case AGFX_TEXTURE_FORMAT_DEPTH32F:          *bytesPerBlock = 4;  break;
        case AGFX_TEXTURE_FORMAT_BC1_UNORM:
        case AGFX_TEXTURE_FORMAT_BC1_UNORM_SRGB:
        case AGFX_TEXTURE_FORMAT_BC4_UNORM:         *bytesPerBlock = 8;  *blockWidth = 4; *blockHeight = 4; break;
        case AGFX_TEXTURE_FORMAT_BC3_UNORM:
        case AGFX_TEXTURE_FORMAT_BC3_UNORM_SRGB:
        case AGFX_TEXTURE_FORMAT_BC5_UNORM:
        case AGFX_TEXTURE_FORMAT_BC6H_UFLOAT:
        case AGFX_TEXTURE_FORMAT_BC7_UNORM:
        case AGFX_TEXTURE_FORMAT_BC7_UNORM_SRGB:    *bytesPerBlock = 16; *blockWidth = 4; *blockHeight = 4; break;
        case AGFX_TEXTURE_FORMAT_ASTC_4X4_UNORM:
        case AGFX_TEXTURE_FORMAT_ASTC_4X4_UNORM_SRGB: *bytesPerBlock = 16; *blockWidth = 4; *blockHeight = 4; break;
        case AGFX_TEXTURE_FORMAT_ASTC_8X8_UNORM:
        case AGFX_TEXTURE_FORMAT_ASTC_8X8_UNORM_SRGB: *bytesPerBlock = 16; *blockWidth = 8; *blockHeight = 8; break;
        default:                                    *bytesPerBlock = 4;  break;
    }
}

static VkImageUsageFlags agfxTextureUsageToVkImageUsageFlags(agfxTextureUsage usage)
{
    // Transfer usage is implicit on D3D12/Metal but must be requested explicitly in Vulkan -- every
    // texture needs it for agfxComputePassCopy*/agfxTextureReplaceRegion-style paths later.
    VkImageUsageFlags flags = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (usage & AGFX_TEXTURE_USAGE_SAMPLED) flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (usage & AGFX_TEXTURE_USAGE_STORAGE) flags |= VK_IMAGE_USAGE_STORAGE_BIT;
    if (usage & AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT) flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (usage & AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT) flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    return flags;
}

static VkBufferUsageFlags agfxBufferUsageToVkBufferUsageFlags(agfxDevice* device, agfxBufferUsage usage)
{
    // Every buffer is created eligible for device address use (the device already requires and
    // enables VkPhysicalDeviceVulkan12Features::bufferDeviceAddress) and for transfer, since Vulkan
    // requires explicit transfer usage unlike D3D12/Metal. Indirect usage is likewise unconditional:
    // agfxBufferUsage has no INDIRECT bit (D3D12/Metal need none), and indirect bundle buffers are
    // ordinary AGFX buffers underneath.
    VkBufferUsageFlags flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    // On RT devices any buffer may become BLAS geometry (D3D12/Metal need no opt-in there either),
    // and scratch buffers are plain storage buffers the app allocates itself.
    if (device->supportsRayTracing) flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    if (usage & AGFX_BUFFER_USAGE_INDEX) flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (usage & AGFX_BUFFER_USAGE_CONSTANT) flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (usage & (AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE)) flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    return flags;
}

static VkMemoryPropertyFlags agfxBufferMemoryTypeToVkMemoryPropertyFlags(agfxBufferMemoryType memoryType)
{
    switch (memoryType) {
        case AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY:
            return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        case AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        case AGFX_BUFFER_MEMORY_TYPE_GPU_TO_CPU:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        default:
            return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
}

// Mirrors the ID3D12Device::GetResourceAllocationInfo-driven searches in agfx_d3d12.cpp: pick a
// memory type index compatible with typeBits (a resource's VkMemoryRequirements::memoryTypeBits, or
// ~0u when there's no specific resource yet, e.g. sizing a heap) that has every bit in `required`.
static uint32_t agfxVkFindMemoryType(agfxDevice* device, uint32_t typeBits, VkMemoryPropertyFlags required)
{
    for (uint32_t i = 0; i < device->memoryProperties.memoryTypeCount; ++i) {
        if (!(typeBits & (1u << i))) continue;
        if ((device->memoryProperties.memoryTypes[i].propertyFlags & required) == required) return i;
    }
    return UINT32_MAX;
}

// Pipeline-state translation tables. No shared table exists between backends (same as the texture
// format table above) -- mirrors agfx_d3d12.cpp's/agfx_metal4.mm's own per-backend switches value-for-value.
static VkCullModeFlags agfxCullModeToVkCullMode(agfxCullMode mode)
{
    switch (mode) {
        case AGFX_CULL_MODE_NONE:  return VK_CULL_MODE_NONE;
        case AGFX_CULL_MODE_FRONT: return VK_CULL_MODE_FRONT_BIT;
        case AGFX_CULL_MODE_BACK:  return VK_CULL_MODE_BACK_BIT;
        default:                   return VK_CULL_MODE_NONE;
    }
}

static VkFrontFace agfxFrontFaceToVkFrontFace(agfxFrontFace face)
{
    return (face == AGFX_FRONT_FACE_COUNTER_CLOCKWISE) ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
}

static VkPolygonMode agfxFillModeToVkPolygonMode(agfxFillMode mode)
{
    return (mode == AGFX_FILL_MODE_WIREFRAME) ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
}

static VkCompareOp agfxComparisonFunctionToVkCompareOp(agfxComparisonFunction func)
{
    switch (func) {
        case AGFX_COMPARISON_FUNCTION_NEVER:          return VK_COMPARE_OP_NEVER;
        case AGFX_COMPARISON_FUNCTION_LESS:           return VK_COMPARE_OP_LESS;
        case AGFX_COMPARISON_FUNCTION_EQUAL:          return VK_COMPARE_OP_EQUAL;
        case AGFX_COMPARISON_FUNCTION_LESS_EQUAL:     return VK_COMPARE_OP_LESS_OR_EQUAL;
        case AGFX_COMPARISON_FUNCTION_GREATER:        return VK_COMPARE_OP_GREATER;
        case AGFX_COMPARISON_FUNCTION_NOT_EQUAL:      return VK_COMPARE_OP_NOT_EQUAL;
        case AGFX_COMPARISON_FUNCTION_GREATER_EQUAL:  return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case AGFX_COMPARISON_FUNCTION_ALWAYS:         return VK_COMPARE_OP_ALWAYS;
        default:                                      return VK_COMPARE_OP_ALWAYS;
    }
}

static VkFilter agfxSamplerFilterToVkFilter(agfxSamplerFilter filter)
{
    return (filter == AGFX_SAMPLER_FILTER_LINEAR) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
}

static VkSamplerMipmapMode agfxSamplerFilterToVkSamplerMipmapMode(agfxSamplerFilter filter)
{
    return (filter == AGFX_SAMPLER_FILTER_LINEAR) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
}

static VkSamplerAddressMode agfxSamplerAddressModeToVkSamplerAddressMode(agfxSamplerAddressMode mode)
{
    switch (mode) {
        case AGFX_SAMPLER_ADDRESS_MODE_REPEAT:          return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case AGFX_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:   return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        default:                                        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

static VkAttachmentLoadOp agfxLoadOperationToVkAttachmentLoadOp(agfxLoadOp op)
{
    switch (op) {
        case AGFX_LOAD_OPERATION_LOAD:      return VK_ATTACHMENT_LOAD_OP_LOAD;
        case AGFX_LOAD_OPERATION_CLEAR:     return VK_ATTACHMENT_LOAD_OP_CLEAR;
        case AGFX_LOAD_OPERATION_DONT_CARE: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        default:                            return VK_ATTACHMENT_LOAD_OP_LOAD;
    }
}

static VkAttachmentStoreOp agfxStoreOperationToVkAttachmentStoreOp(agfxStoreOp op)
{
    return (op == AGFX_STORE_OPERATION_DONT_CARE) ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
}

static VkPrimitiveTopology agfxTopologyToVkPrimitiveTopology(agfxTopology topology)
{
    switch (topology) {
        case AGFX_TOPOLOGY_TRIANGLES: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case AGFX_TOPOLOGY_LINES:     return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case AGFX_TOPOLOGY_POINTS:    return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        default:                      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

static VkBlendFactor agfxBlendFactorToVkBlendFactor(agfxBlendFactor factor)
{
    switch (factor) {
        case AGFX_BLEND_FACTOR_ZERO:                  return VK_BLEND_FACTOR_ZERO;
        case AGFX_BLEND_FACTOR_ONE:                   return VK_BLEND_FACTOR_ONE;
        case AGFX_BLEND_FACTOR_SRC_COLOR:             return VK_BLEND_FACTOR_SRC_COLOR;
        case AGFX_BLEND_FACTOR_ONE_MINUS_SRC_COLOR:   return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case AGFX_BLEND_FACTOR_DST_COLOR:             return VK_BLEND_FACTOR_DST_COLOR;
        case AGFX_BLEND_FACTOR_ONE_MINUS_DST_COLOR:   return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case AGFX_BLEND_FACTOR_SRC_ALPHA:             return VK_BLEND_FACTOR_SRC_ALPHA;
        case AGFX_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:   return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case AGFX_BLEND_FACTOR_DST_ALPHA:             return VK_BLEND_FACTOR_DST_ALPHA;
        case AGFX_BLEND_FACTOR_ONE_MINUS_DST_ALPHA:   return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        default:                                      return VK_BLEND_FACTOR_ONE;
    }
}

static VkBlendOp agfxBlendOperationToVkBlendOp(agfxBlendOperation op)
{
    switch (op) {
        case AGFX_BLEND_OPERATION_ADD:              return VK_BLEND_OP_ADD;
        case AGFX_BLEND_OPERATION_SUBTRACT:         return VK_BLEND_OP_SUBTRACT;
        case AGFX_BLEND_OPERATION_REVERSE_SUBTRACT: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case AGFX_BLEND_OPERATION_MIN:              return VK_BLEND_OP_MIN;
        case AGFX_BLEND_OPERATION_MAX:              return VK_BLEND_OP_MAX;
        default:                                    return VK_BLEND_OP_ADD;
    }
}

// Creates a VkPipelineCache from optional caller-provided initial data, shared by
// agfxRenderPipelineCreate and agfxComputePipelineCreate -- both keep it alive on their pipeline
// struct (rather than discarding it after creation, like a D3D12 PSO's implicit cache) because
// agfx*PipelineGetCache reads serialized data back from it later via vkGetPipelineCacheData.
static VkPipelineCache agfxVkCreatePipelineCache(agfxDevice* device, const uint8_t* initialData, uint64_t initialDataSize)
{
    VkPipelineCacheCreateInfo cacheInfo = { VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    cacheInfo.initialDataSize = initialDataSize;
    cacheInfo.pInitialData = initialData;

    VkPipelineCache cache = VK_NULL_HANDLE;
    if (vkCreatePipelineCache(device->device, &cacheInfo, nullptr, &cache) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxVkCreatePipelineCache: vkCreatePipelineCache failed");
        return VK_NULL_HANDLE;
    }
    return cache;
}

static uint8_t* agfxVkGetPipelineCacheData(agfxDevice* device, VkPipelineCache cache, uint64_t* outSize)
{
    *outSize = 0;
    if (!cache) return nullptr;

    size_t size = 0;
    if (vkGetPipelineCacheData(device->device, cache, &size, nullptr) != VK_SUCCESS || size == 0) {
        return nullptr;
    }

    uint8_t* data = (uint8_t*)device->createInfo.allocate(size, device->createInfo.userData);
    if (vkGetPipelineCacheData(device->device, cache, &size, data) != VK_SUCCESS) {
        device->createInfo.free(data, device->createInfo.userData);
        return nullptr;
    }

    *outSize = size;
    return data;
}

// Command buffer
struct agfxCommandBuffer {
    agfxDevice* device;
    agfxCommandQueueType queueType;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    bool isRecording;
};

agfxCommandBuffer* agfxCommandBufferCreate(agfxDevice* device, agfxCommandQueue* queue)
{
    agfxCommandBuffer* commandBuffer = AgfxAlloc<agfxCommandBuffer>(device);
    commandBuffer->device = device;
    commandBuffer->queueType = queue->type;

    VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queue->familyIndex;

    if (vkCreateCommandPool(device->device, &poolInfo, nullptr, &commandBuffer->commandPool) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxCommandBufferCreate: vkCreateCommandPool failed");
        AgfxFree(device, commandBuffer);
        return nullptr;
    }

    VkCommandBufferAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocateInfo.commandPool = commandBuffer->commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device->device, &allocateInfo, &commandBuffer->commandBuffer) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxCommandBufferCreate: vkAllocateCommandBuffers failed");
        vkDestroyCommandPool(device->device, commandBuffer->commandPool, nullptr);
        AgfxFree(device, commandBuffer);
        return nullptr;
    }

    return commandBuffer;
}

void agfxCommandBufferDestroy(agfxDevice* device, agfxCommandBuffer* commandBuffer)
{
    if (commandBuffer->commandPool) vkDestroyCommandPool(device->device, commandBuffer->commandPool, nullptr); // Frees the allocated VkCommandBuffer too.
    AgfxFree(device, commandBuffer);
}

void agfxCommandBufferReset(agfxCommandBuffer* commandBuffer)
{
    // Heavier than Begin(): reclaims the pool's memory instead of just reopening the command buffer.
    if (commandBuffer->isRecording) {
        vkEndCommandBuffer(commandBuffer->commandBuffer);
        commandBuffer->isRecording = false;
    }
    vkResetCommandPool(commandBuffer->device->device, commandBuffer->commandPool, 0);

    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(commandBuffer->commandBuffer, &beginInfo);
    commandBuffer->isRecording = true;
}

void agfxCommandBufferBegin(agfxCommandBuffer* commandBuffer)
{
    if (!commandBuffer->isRecording) {
        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(commandBuffer->commandBuffer, &beginInfo);
        commandBuffer->isRecording = true;
    }

    // Vulkan analog of D3D12's SetDescriptorHeaps: bind the one global bindless set at every
    // pipeline bind point AGFX shaders can run at, so any SetPipeline/Draw/Dispatch in this command
    // buffer can reference bindless resources without re-binding descriptor sets per draw. AGFX's
    // raytracing is inline (RayQuery from a compute shader), so there is no separate ray tracing
    // bind point to cover here. Each bind point is only legal on a queue family that supports it
    // (VUID-vkCmdBindDescriptorSets-pipelineBindPoint-00361) -- binding GRAPHICS on a compute-only
    // family is UB that hangs NVIDIA's compute queue outright.
    if (commandBuffer->queueType == AGFX_COMMAND_QUEUE_TYPE_GRAPHICS) {
        vkCmdBindDescriptorSets(commandBuffer->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, commandBuffer->device->globalPipelineLayout, 0, 1, &commandBuffer->device->globalSet, 0, nullptr);
    }
    if (commandBuffer->queueType != AGFX_COMMAND_QUEUE_TYPE_TRANSFER) {
        vkCmdBindDescriptorSets(commandBuffer->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, commandBuffer->device->globalPipelineLayout, 0, 1, &commandBuffer->device->globalSet, 0, nullptr);
    }
}

void agfxCommandBufferEnd(agfxCommandBuffer* commandBuffer)
{
    vkEndCommandBuffer(commandBuffer->commandBuffer);
    commandBuffer->isRecording = false;
}

// Defined down here (rather than in the Query pool section) because they need agfxCommandBuffer to
// be a complete type.
void agfxCommandBufferWriteTimestamp(agfxCommandBuffer* commandBuffer, agfxQueryPool* pool, uint32_t index)
{
    // Vulkan queries must be reset between uses (D3D12's EndQuery has no such requirement), so
    // reset just this slot right before rewriting it. Both commands are illegal inside a render
    // pass, matching where agfxCommandBufferWriteTimestamp already sits in the public API.
    vkCmdResetQueryPool(commandBuffer->commandBuffer, pool->vkQueryPool, index, 1);
    vkCmdWriteTimestamp2(commandBuffer->commandBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool->vkQueryPool, index);
}

void agfxCommandBufferResolveQueryPool(agfxCommandBuffer* commandBuffer, agfxQueryPool* pool, uint32_t firstIndex, uint32_t count)
{
    vkCmdCopyQueryPoolResults(commandBuffer->commandBuffer, pool->vkQueryPool, firstIndex, count,
        pool->readbackBuffer, (VkDeviceSize)firstIndex * sizeof(uint64_t), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
}

void agfxCommandQueueSubmit(agfxCommandQueue* queue, agfxCommandBuffer** commandBuffers, uint32_t commandBufferCount)
{
    std::vector<VkCommandBufferSubmitInfo> bufferInfos(commandBufferCount);
    for (uint32_t i = 0; i < commandBufferCount; ++i) {
        bufferInfos[i] = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        bufferInfos[i].commandBuffer = commandBuffers[i]->commandBuffer;
    }

    VkSubmitInfo2 submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = commandBufferCount;
    submitInfo.pCommandBufferInfos = bufferInfos.data();

    vkQueueSubmit2(queue->vkQueue, 1, &submitInfo, VK_NULL_HANDLE);
}

// Queue ownership: D3D12 Enhanced Barriers (and so AGFX's public barrier API, ported from it) have
// no queue-family-transfer concept at all -- none of the barrier functions below take a "from queue"
// parameter. Vulkan's VK_SHARING_MODE_EXCLUSIVE would need one (an explicit release on the source
// family's command buffer, matched with an acquire on the destination family's), which doesn't fit
// this signature without a breaking API addition. Resolution: AGFX-created Vulkan resources are
// created VK_SHARING_MODE_CONCURRENT across the graphics/compute/transfer families (set at resource
// creation, a later phase), so no ownership transfer is ever required and every barrier below can
// unconditionally use VK_QUEUE_FAMILY_IGNORED on both sides -- matching D3D12's queue-agnostic model
// exactly, at the cost of a little cross-queue synchronization overhead some drivers charge for
// CONCURRENT sharing. If that overhead ever matters, the alternative is a new, explicitly breaking
// agfxCommandBufferQueueOwnershipBarrier-style call -- not implemented here since CONCURRENT sharing
// avoids needing it at all.
void agfxCommandBufferTextureBarrier(agfxCommandBuffer* commandBuffer, agfxTexture* texture, agfxResourceState oldState, agfxResourceState newState, uint32_t mip, uint32_t layer, agfxBool agglomerate)
{
    bool allMips = (mip == (uint32_t)AGFX_SUBRESOURCE_ALL_MIPS);
    bool allLayers = (layer == (uint32_t)AGFX_SUBRESOURCE_ALL_LAYERS);

    VkImageSubresourceRange range = {};
    range.aspectMask = agfxTextureFormatToVkImageAspect(texture->createInfo.format);
    range.baseMipLevel = allMips ? 0 : mip;
    range.levelCount = allMips ? texture->createInfo.mipLevels : 1;
    range.baseArrayLayer = allLayers ? 0 : layer;
    range.layerCount = allLayers ? texture->createInfo.depthOrArrayLayers : 1;

    VkImageMemoryBarrier2 imageBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    imageBarrier.srcStageMask = agfxResourceStateToVkPipelineStage(oldState);
    imageBarrier.dstStageMask = agfxResourceStateToVkPipelineStage(newState);
    imageBarrier.srcAccessMask = agfxResourceStateToVkAccessFlags(oldState);
    imageBarrier.dstAccessMask = agfxResourceStateToVkAccessFlags(newState);
    // Transitioning OUT of PRESENT uses UNDEFINED rather than PRESENT_SRC_KHR: a freshly acquired
    // swap chain image has never been presented and is actually in UNDEFINED (claiming otherwise is
    // invalid), and a back buffer's prior contents at the start of a frame are discardable anyway.
    // COMMON gets the same treatment: the public contract is that textures start in COMMON, but the
    // VkImage underneath is created in UNDEFINED, and COMMON's GENERAL mapping would claim a layout
    // the image was never in. The cost is that COMMON -> X discards contents, which is fine as long
    // as COMMON only ever appears as a texture's initial, never-written state.
    imageBarrier.oldLayout = (oldState == AGFX_RESOURCE_STATE_PRESENT || oldState == AGFX_RESOURCE_STATE_COMMON)
        ? VK_IMAGE_LAYOUT_UNDEFINED : agfxResourceStateToVkImageLayout(oldState);
    imageBarrier.newLayout = agfxResourceStateToVkImageLayout(newState);
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = texture->vkImage;
    imageBarrier.subresourceRange = range;

    VkDependencyInfo dependencyInfo = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &imageBarrier;

    vkCmdPipelineBarrier2(commandBuffer->commandBuffer, &dependencyInfo);
}

void agfxCommandBufferMemoryBarrier(agfxCommandBuffer* commandBuffer, agfxResourceState oldState, agfxResourceState newState, agfxBool agglomerate)
{
    // Buffers and acceleration structures have no layout to transition, so -- exactly like the
    // D3D12 backend's CD3DX12_GLOBAL_BARRIER -- this is a global (memory-only) barrier rather than
    // one scoped to a specific resource.
    VkMemoryBarrier2 memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    memoryBarrier.srcStageMask = agfxResourceStateToVkPipelineStage(oldState);
    memoryBarrier.dstStageMask = agfxResourceStateToVkPipelineStage(newState);
    memoryBarrier.srcAccessMask = agfxResourceStateToVkAccessFlags(oldState);
    memoryBarrier.dstAccessMask = agfxResourceStateToVkAccessFlags(newState);

    VkDependencyInfo dependencyInfo = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dependencyInfo.memoryBarrierCount = 1;
    dependencyInfo.pMemoryBarriers = &memoryBarrier;

    vkCmdPipelineBarrier2(commandBuffer->commandBuffer, &dependencyInfo);
}

void agfxCommandBufferBufferBarrier(agfxCommandBuffer* commandBuffer, agfxBuffer* buffer, agfxResourceState oldState, agfxResourceState newState, agfxBool agglomerate)
{
    // Vulkan has no legacy-state validation quirk like D3D12's -- a global VkMemoryBarrier2 already
    // orders all matching access correctly. This exists purely to mirror agfxCommandBufferBufferBarrier's
    // cross-platform signature; scoping it to the actual buffer costs nothing and documents intent.
    VkBufferMemoryBarrier2 bufferBarrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    bufferBarrier.srcStageMask = agfxResourceStateToVkPipelineStage(oldState);
    bufferBarrier.dstStageMask = agfxResourceStateToVkPipelineStage(newState);
    bufferBarrier.srcAccessMask = agfxResourceStateToVkAccessFlags(oldState);
    bufferBarrier.dstAccessMask = agfxResourceStateToVkAccessFlags(newState);
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer->vkBuffer;
    bufferBarrier.offset = 0;
    bufferBarrier.size = VK_WHOLE_SIZE;

    VkDependencyInfo dependencyInfo = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dependencyInfo.bufferMemoryBarrierCount = 1;
    dependencyInfo.pBufferMemoryBarriers = &bufferBarrier;

    vkCmdPipelineBarrier2(commandBuffer->commandBuffer, &dependencyInfo);
}

void agfxCommandBufferAliasingBarrier(agfxCommandBuffer* commandBuffer, agfxTexture* incomingTexture, agfxResourceState outgoingState, agfxResourceState incomingState, agfxBool agglomerate)
{
    // Same two-part aliasing workflow as the D3D12 backend's pair of CD3DX12_GLOBAL_BARRIER +
    // CD3DX12_TEXTURE_BARRIER: a global barrier flushes the outgoing resource's pending writes (an
    // image barrier's access scope only ever covers its own image, and the incoming image has
    // nothing of its own to flush), then an image barrier activates the incoming resource with
    // oldLayout=UNDEFINED and srcAccessMask=NONE -- there is nothing worth preserving from before.
    // Vulkan's VkDependencyInfo can carry both barriers in one call, unlike D3D12's two BARRIER_GROUPs.
    VkMemoryBarrier2 memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    memoryBarrier.srcStageMask = agfxResourceStateToVkPipelineStage(outgoingState);
    memoryBarrier.dstStageMask = agfxResourceStateToVkPipelineStage(incomingState);
    memoryBarrier.srcAccessMask = agfxResourceStateToVkAccessFlags(outgoingState);
    memoryBarrier.dstAccessMask = agfxResourceStateToVkAccessFlags(incomingState);

    VkImageSubresourceRange range = {};
    range.aspectMask = agfxTextureFormatToVkImageAspect(incomingTexture->createInfo.format);
    range.baseMipLevel = 0;
    range.levelCount = incomingTexture->createInfo.mipLevels;
    range.baseArrayLayer = 0;
    range.layerCount = incomingTexture->createInfo.depthOrArrayLayers;

    VkImageMemoryBarrier2 imageBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    imageBarrier.srcStageMask = agfxResourceStateToVkPipelineStage(outgoingState);
    imageBarrier.dstStageMask = agfxResourceStateToVkPipelineStage(incomingState);
    imageBarrier.srcAccessMask = VK_ACCESS_2_NONE;
    imageBarrier.dstAccessMask = agfxResourceStateToVkAccessFlags(incomingState);
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageBarrier.newLayout = agfxResourceStateToVkImageLayout(incomingState);
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = incomingTexture->vkImage;
    imageBarrier.subresourceRange = range;

    VkDependencyInfo dependencyInfo = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dependencyInfo.memoryBarrierCount = 1;
    dependencyInfo.pMemoryBarriers = &memoryBarrier;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &imageBarrier;

    vkCmdPipelineBarrier2(commandBuffer->commandBuffer, &dependencyInfo);
}

// Acceleration structure
struct agfxAccelerationStructure {
    agfxAccelerationStructureCreateInfo createInfo{};
    // Geometry converted once at create time (like the D3D12 backend's geometryDescs) so sizing,
    // build, and update all describe identical inputs.
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<uint32_t> primitiveCounts;
    VkAccelerationStructureBuildSizesInfoKHR buildSizes = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    VkAccelerationStructureKHR vkAccelerationStructure = VK_NULL_HANDLE;
    // Cached at create: querying it per TLAS instance per frame is a driver call hot enough to show
    // up on the CPU profile.
    VkDeviceAddress deviceAddress = 0;
    // Backing storage the VkAccelerationStructureKHR lives in.
    VkBuffer storageBuffer = VK_NULL_HANDLE;
    VkDeviceMemory storageMemory = VK_NULL_HANDLE;
    // TLAS only: persistently mapped host-visible array of VkAccelerationStructureInstanceKHR.
    VkBuffer instanceBuffer = VK_NULL_HANDLE;
    VkDeviceMemory instanceMemory = VK_NULL_HANDLE;
    VkAccelerationStructureInstanceKHR* mappedInstances = nullptr;
    uint32_t currentInstanceCount = 0;
    // Binding 2 slot (TLAS only) -- what agfxAccelerationStructureGetHandle returns.
    uint64_t slot = UINT64_MAX;
    // Lazily created by agfxComputePassWriteCompactedSizeToBuffer; Vulkan needs a query pool where
    // D3D12's EmitRaytracingAccelerationStructurePostbuildInfo writes to a buffer directly.
    VkQueryPool compactedSizeQueryPool = VK_NULL_HANDLE;
    // Stashed at create time: AddInstances/ResetInstances take no agfxDevice parameter.
    agfxDevice* device = nullptr;
};

static VkDeviceAddress agfxVkBufferDeviceAddress(agfxDevice* device, VkBuffer buffer)
{
    VkBufferDeviceAddressInfo addressInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    addressInfo.buffer = buffer;
    return vkGetBufferDeviceAddress(device->device, &addressInfo);
}

// Small dedicated buffer+memory pair for AS-internal allocations (AS storage, TLAS instance
// arrays); these are raw Vulkan objects rather than agfxBuffers since they never cross the API.
static bool agfxVkCreateDedicatedBuffer(agfxDevice* device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryFlags, VkBuffer* outBuffer, VkDeviceMemory* outMemory)
{
    VkBufferCreateInfo bufferCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device->device, &bufferCreateInfo, nullptr, outBuffer) != VK_SUCCESS) return false;

    VkMemoryRequirements memoryRequirements = {};
    vkGetBufferMemoryRequirements(device->device, *outBuffer, &memoryRequirements);
    uint32_t memoryTypeIndex = agfxVkFindMemoryType(device, memoryRequirements.memoryTypeBits, memoryFlags);
    if (memoryTypeIndex == UINT32_MAX) {
        vkDestroyBuffer(device->device, *outBuffer, nullptr);
        *outBuffer = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateFlagsInfo allocateFlagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
    allocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocateInfo.pNext = &allocateFlagsInfo;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = memoryTypeIndex;
    if (vkAllocateMemory(device->device, &allocateInfo, nullptr, outMemory) != VK_SUCCESS ||
        vkBindBufferMemory(device->device, *outBuffer, *outMemory, 0) != VK_SUCCESS) {
        if (*outMemory) vkFreeMemory(device->device, *outMemory, nullptr);
        vkDestroyBuffer(device->device, *outBuffer, nullptr);
        *outBuffer = VK_NULL_HANDLE;
        *outMemory = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

static void agfxVkFillAccelerationStructureGeometries(agfxDevice* device, agfxAccelerationStructure* accelerationStructure)
{
    const agfxAccelerationStructureCreateInfo* createInfo = &accelerationStructure->createInfo;

    if (createInfo->type == AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL) {
        accelerationStructure->geometries.reserve(createInfo->bottomLevel.geometryCount);
        accelerationStructure->primitiveCounts.reserve(createInfo->bottomLevel.geometryCount);
        for (uint32_t i = 0; i < createInfo->bottomLevel.geometryCount; ++i) {
            const agfxAccelerationStructureGeometry* geometry = &createInfo->bottomLevel.geometries[i];

            VkAccelerationStructureGeometryKHR vkGeometry = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
            vkGeometry.flags = geometry->opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
            if (geometry->type == AGFX_ACCELERATION_STRUCTURE_GEOMETRY_TYPE_TRIANGLES) {
                vkGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                VkAccelerationStructureGeometryTrianglesDataKHR& triangles = vkGeometry.geometry.triangles;
                triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                // Same convention as D3D12: position is a float3 at the start of the vertex struct,
                // and vertexOffset/indexOffset are byte offsets.
                triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                triangles.vertexData.deviceAddress = agfxVkBufferDeviceAddress(device, geometry->triangles.vertexBuffer->vkBuffer) + geometry->triangles.vertexOffset;
                triangles.vertexStride = geometry->triangles.vertexBuffer->createInfo.stride;
                triangles.maxVertex = geometry->triangles.vertexCount > 0 ? geometry->triangles.vertexCount - 1 : 0;
                triangles.indexType = (geometry->triangles.indexBuffer->createInfo.stride == 2) ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
                triangles.indexData.deviceAddress = agfxVkBufferDeviceAddress(device, geometry->triangles.indexBuffer->vkBuffer) + geometry->triangles.indexOffset;
                accelerationStructure->primitiveCounts.push_back(geometry->triangles.indexCount / 3);
            } else {
                vkGeometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
                VkAccelerationStructureGeometryAabbsDataKHR& aabbs = vkGeometry.geometry.aabbs;
                aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
                aabbs.data.deviceAddress = agfxVkBufferDeviceAddress(device, geometry->aabbs.aabbBuffer->vkBuffer) + geometry->aabbs.aabbOffset;
                aabbs.stride = geometry->aabbs.aabbStride;
                accelerationStructure->primitiveCounts.push_back(geometry->aabbs.aabbCount);
            }
            accelerationStructure->geometries.push_back(vkGeometry);
        }
    } else {
        VkAccelerationStructureGeometryKHR vkGeometry = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        vkGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        VkAccelerationStructureGeometryInstancesDataKHR& instances = vkGeometry.geometry.instances;
        instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        instances.arrayOfPointers = VK_FALSE;
        instances.data.deviceAddress = agfxVkBufferDeviceAddress(device, accelerationStructure->instanceBuffer);
        accelerationStructure->geometries.push_back(vkGeometry);
        // Builds always cover the full maxInstanceCount range, matching the D3D12/Metal backends
        // (sizing and every rebuild agree on the count); the instance array is zero-filled, and a
        // zeroed instance has mask 0 -> always culled, so unused slots are inert.
        accelerationStructure->primitiveCounts.push_back(createInfo->topLevel.maxInstanceCount);
    }
}

static VkAccelerationStructureBuildGeometryInfoKHR agfxVkAccelerationStructureBuildInfo(agfxAccelerationStructure* accelerationStructure)
{
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    buildInfo.type = (accelerationStructure->createInfo.type == AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL)
        ? VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR : VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    // Same flag policy as the D3D12 backend: fast build + always compactable.
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    if (accelerationStructure->createInfo.allowUpdate) buildInfo.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = (uint32_t)accelerationStructure->geometries.size();
    buildInfo.pGeometries = accelerationStructure->geometries.data();
    return buildInfo;
}

// Shared tail of Create/CreateCompacted: backing buffer, the AS object itself, the TLAS bindless
// descriptor (binding 2), and the debug name.
static bool agfxVkFinalizeAccelerationStructure(agfxDevice* device, agfxAccelerationStructure* accelerationStructure, uint64_t size)
{
    if (!agfxVkCreateDedicatedBuffer(device, size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                     &accelerationStructure->storageBuffer, &accelerationStructure->storageMemory)) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxAccelerationStructureCreate: backing buffer creation failed");
        return false;
    }

    VkAccelerationStructureCreateInfoKHR accelCreateInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
    accelCreateInfo.buffer = accelerationStructure->storageBuffer;
    accelCreateInfo.offset = 0;
    accelCreateInfo.size = size;
    accelCreateInfo.type = (accelerationStructure->createInfo.type == AGFX_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL)
        ? VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR : VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    if (vkCreateAccelerationStructureKHR(device->device, &accelCreateInfo, nullptr, &accelerationStructure->vkAccelerationStructure) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxAccelerationStructureCreate: vkCreateAccelerationStructureKHR failed");
        return false;
    }

    VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
    deviceAddressInfo.accelerationStructure = accelerationStructure->vkAccelerationStructure;
    accelerationStructure->deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device->device, &deviceAddressInfo);

    // Only a TLAS is traced from shaders (__rt_as_array, set 0 binding 2), matching D3D12 where
    // only the TLAS gets an SRV.
    if (accelerationStructure->createInfo.type == AGFX_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL) {
        uint64_t slot = device->accelSlotAllocator.allocate();
        if (slot == UINT64_MAX) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxAccelerationStructureCreate: bindless acceleration structure slots exhausted");
            return false;
        }

        VkWriteDescriptorSetAccelerationStructureKHR accelWrite = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
        accelWrite.accelerationStructureCount = 1;
        accelWrite.pAccelerationStructures = &accelerationStructure->vkAccelerationStructure;

        VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.pNext = &accelWrite;
        write.dstSet = device->globalSet;
        write.dstBinding = 2;
        write.dstArrayElement = (uint32_t)slot;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        vkUpdateDescriptorSets(device->device, 1, &write, 0, nullptr);
        accelerationStructure->slot = slot;
    }

    if (accelerationStructure->createInfo.name && vkSetDebugUtilsObjectNameEXT) {
        VkDebugUtilsObjectNameInfoEXT nameInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
        nameInfo.objectType = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR;
        nameInfo.objectHandle = (uint64_t)accelerationStructure->vkAccelerationStructure;
        nameInfo.pObjectName = accelerationStructure->createInfo.name;
        vkSetDebugUtilsObjectNameEXT(device->device, &nameInfo);
    }

    return true;
}

agfxAccelerationStructure* agfxAccelerationStructureCreate(agfxDevice* device, const agfxAccelerationStructureCreateInfo* createInfo)
{
    agfxAccelerationStructure* accelerationStructure = AgfxAlloc<agfxAccelerationStructure>(device);
    accelerationStructure->createInfo = *createInfo;
    accelerationStructure->device = device;

    if (createInfo->type == AGFX_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL) {
        VkDeviceSize instanceBufferSize = (VkDeviceSize)createInfo->topLevel.maxInstanceCount * sizeof(VkAccelerationStructureInstanceKHR);
        if (!agfxVkCreateDedicatedBuffer(device, instanceBufferSize,
                                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         &accelerationStructure->instanceBuffer, &accelerationStructure->instanceMemory)) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxAccelerationStructureCreate: instance buffer creation failed");
            agfxAccelerationStructureDestroy(device, accelerationStructure);
            return nullptr;
        }
        if (vkMapMemory(device->device, accelerationStructure->instanceMemory, 0, instanceBufferSize, 0, (void**)&accelerationStructure->mappedInstances) != VK_SUCCESS) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxAccelerationStructureCreate: mapping the instance buffer failed");
            agfxAccelerationStructureDestroy(device, accelerationStructure);
            return nullptr;
        }
        memset(accelerationStructure->mappedInstances, 0, instanceBufferSize);
    }

    agfxVkFillAccelerationStructureGeometries(device, accelerationStructure);

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = agfxVkAccelerationStructureBuildInfo(accelerationStructure);
    vkGetAccelerationStructureBuildSizesKHR(device->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &buildInfo, accelerationStructure->primitiveCounts.data(), &accelerationStructure->buildSizes);

    if (!agfxVkFinalizeAccelerationStructure(device, accelerationStructure, accelerationStructure->buildSizes.accelerationStructureSize)) {
        agfxAccelerationStructureDestroy(device, accelerationStructure);
        return nullptr;
    }

    return accelerationStructure;
}

agfxAccelerationStructure* agfxAccelerationStructureCreateCompacted(agfxDevice* device, const agfxAccelerationStructureCreateInfo* createInfo, uint64_t compactedSize)
{
    // Compaction destination only -- no geometry conversion or instance buffer, matching D3D12: it
    // is filled by agfxComputePassCompactAccelerationStructure, never built directly.
    agfxAccelerationStructure* accelerationStructure = AgfxAlloc<agfxAccelerationStructure>(device);
    accelerationStructure->createInfo = *createInfo;
    accelerationStructure->device = device;
    accelerationStructure->buildSizes.accelerationStructureSize = compactedSize;

    if (!agfxVkFinalizeAccelerationStructure(device, accelerationStructure, compactedSize)) {
        agfxAccelerationStructureDestroy(device, accelerationStructure);
        return nullptr;
    }

    return accelerationStructure;
}

void agfxAccelerationStructureDestroy(agfxDevice* device, agfxAccelerationStructure* accelerationStructure)
{
    if (accelerationStructure->slot != UINT64_MAX) device->accelSlotAllocator.free(accelerationStructure->slot);
    if (accelerationStructure->compactedSizeQueryPool) vkDestroyQueryPool(device->device, accelerationStructure->compactedSizeQueryPool, nullptr);
    if (accelerationStructure->vkAccelerationStructure) vkDestroyAccelerationStructureKHR(device->device, accelerationStructure->vkAccelerationStructure, nullptr);
    if (accelerationStructure->storageBuffer) vkDestroyBuffer(device->device, accelerationStructure->storageBuffer, nullptr);
    if (accelerationStructure->storageMemory) vkFreeMemory(device->device, accelerationStructure->storageMemory, nullptr);
    if (accelerationStructure->mappedInstances) vkUnmapMemory(device->device, accelerationStructure->instanceMemory);
    if (accelerationStructure->instanceBuffer) vkDestroyBuffer(device->device, accelerationStructure->instanceBuffer, nullptr);
    if (accelerationStructure->instanceMemory) vkFreeMemory(device->device, accelerationStructure->instanceMemory, nullptr);
    AgfxFree(device, accelerationStructure);
}

void agfxAccelerationStructureGetSizes(agfxDevice* device, agfxAccelerationStructure* accelerationStructure, agfxAccelerationStructureSizes* sizes)
{
    sizes->scratchBufferSize = accelerationStructure->buildSizes.buildScratchSize;
    sizes->updateScratchBufferSize = accelerationStructure->buildSizes.updateScratchSize;
}

uint64_t agfxAccelerationStructureGetHandle(agfxAccelerationStructure* accelerationStructure)
{
    return accelerationStructure->slot;
}

void agfxAccelerationStructureAddInstances(agfxAccelerationStructure* accelerationStructure, const agfxAccelerationStructureInstance* instances, uint32_t instanceCount)
{
    for (uint32_t i = 0; i < instanceCount; ++i) {
        const agfxAccelerationStructureInstance* instance = &instances[i];
        VkAccelerationStructureInstanceKHR& vkInstance = accelerationStructure->mappedInstances[accelerationStructure->currentInstanceCount + i];
        memcpy(&vkInstance.transform, instance->transform, sizeof(float) * 12);
        vkInstance.instanceCustomIndex = instance->userID; // What CommittedInstanceID() returns.
        vkInstance.mask = 0xFF;
        vkInstance.instanceShaderBindingTableRecordOffset = 0;
        vkInstance.flags = instance->opaque ? VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR : VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
        vkInstance.accelerationStructureReference = instance->blas->deviceAddress;
    }
    accelerationStructure->currentInstanceCount += instanceCount;
}

void agfxAccelerationStructureResetInstances(agfxAccelerationStructure* accelerationStructure)
{
    accelerationStructure->currentInstanceCount = 0;
}

// Texture
static VkImageCreateInfo agfxTextureImageCreateInfo(const agfxTextureCreateInfo* createInfo)
{
    VkImageCreateInfo imageCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageCreateInfo.imageType = (createInfo->type == AGFX_TEXTURE_TYPE_1D) ? VK_IMAGE_TYPE_1D
        : (createInfo->type == AGFX_TEXTURE_TYPE_3D) ? VK_IMAGE_TYPE_3D
        : VK_IMAGE_TYPE_2D; // 2D, 2D_ARRAY and CUBE are all VK_IMAGE_TYPE_2D with varying arrayLayers.
    imageCreateInfo.format = agfxTextureFormatToVkFormat(createInfo->format);
    imageCreateInfo.extent = { createInfo->width, createInfo->height, (createInfo->type == AGFX_TEXTURE_TYPE_3D) ? createInfo->depthOrArrayLayers : 1 };
    imageCreateInfo.mipLevels = createInfo->mipLevels;
    imageCreateInfo.arrayLayers = (createInfo->type == AGFX_TEXTURE_TYPE_3D) ? 1 : createInfo->depthOrArrayLayers;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = agfxTextureUsageToVkImageUsageFlags(createInfo->usage);
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (createInfo->type == AGFX_TEXTURE_TYPE_CUBE) imageCreateInfo.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    // agfxTextureViewCreateInfo::format may reinterpret the texture's format (e.g. UNORM viewed as
    // SRGB) -- implicit on D3D12/Metal, but Vulkan only allows it when the image opts in.
    imageCreateInfo.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    return imageCreateInfo;
}

agfxTexture* agfxTextureCreate(agfxDevice* device, const agfxTextureCreateInfo* createInfo)
{
    agfxTexture* texture = AgfxAlloc<agfxTexture>(device);
    texture->createInfo = *createInfo;
    texture->device = device;

    VkImageCreateInfo imageCreateInfo = agfxTextureImageCreateInfo(createInfo);
    if (vkCreateImage(device->device, &imageCreateInfo, nullptr, &texture->vkImage) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxTextureCreate: vkCreateImage failed");
        AgfxFree(device, texture);
        return nullptr;
    }

    if (createInfo->heap) {
        if (createInfo->heap->createInfo.memoryType != AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxTextureCreate: textures may only be placed in AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY heaps");
            vkDestroyImage(device->device, texture->vkImage, nullptr);
            AgfxFree(device, texture);
            return nullptr;
        }
        if (vkBindImageMemory(device->device, texture->vkImage, createInfo->heap->vkMemory, createInfo->heapOffset) != VK_SUCCESS) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxTextureCreate: vkBindImageMemory (placed) failed");
            vkDestroyImage(device->device, texture->vkImage, nullptr);
            AgfxFree(device, texture);
            return nullptr;
        }
    } else {
        VkMemoryRequirements memoryRequirements = {};
        vkGetImageMemoryRequirements(device->device, texture->vkImage, &memoryRequirements);

        uint32_t memoryTypeIndex = agfxVkFindMemoryType(device, memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memoryTypeIndex == UINT32_MAX) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxTextureCreate: no device-local memory type compatible with this image");
            vkDestroyImage(device->device, texture->vkImage, nullptr);
            AgfxFree(device, texture);
            return nullptr;
        }

        VkMemoryAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        if (vkAllocateMemory(device->device, &allocateInfo, nullptr, &texture->vkMemory) != VK_SUCCESS) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxTextureCreate: vkAllocateMemory failed");
            vkDestroyImage(device->device, texture->vkImage, nullptr);
            AgfxFree(device, texture);
            return nullptr;
        }

        if (vkBindImageMemory(device->device, texture->vkImage, texture->vkMemory, 0) != VK_SUCCESS) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxTextureCreate: vkBindImageMemory (committed) failed");
            vkFreeMemory(device->device, texture->vkMemory, nullptr);
            vkDestroyImage(device->device, texture->vkImage, nullptr);
            AgfxFree(device, texture);
            return nullptr;
        }
    }

    return texture;
}

void agfxTextureDestroy(agfxDevice* device, agfxTexture* texture)
{
    if (texture->vkImage && texture->ownsImage) vkDestroyImage(device->device, texture->vkImage, nullptr);
    if (texture->vkMemory) vkFreeMemory(device->device, texture->vkMemory, nullptr);
    AgfxFree(device, texture);
}

void agfxTextureGetInfo(agfxTexture* texture, agfxTextureCreateInfo* info)
{
    *info = texture->createInfo;
}

void agfxTextureReplaceRegion(agfxDevice* device, agfxTexture* texture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, const void* data, uint32_t dataSize, uint32_t bytesPerRow, uint32_t bytesPerImage)
{
    // Unsupported, same as the D3D12 backend: use a staging buffer plus agfxComputePassCopyBufferToTexture instead.
    agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxTextureReplaceRegion: unsupported on the Vulkan backend, use a staging buffer and agfxComputePassCopyBufferToTexture instead");
}

void agfxTextureSetName(agfxTexture* texture, const char* name)
{
    if (!vkSetDebugUtilsObjectNameEXT || !texture->vkImage) return;

    VkDebugUtilsObjectNameInfoEXT nameInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
    nameInfo.objectType = VK_OBJECT_TYPE_IMAGE;
    nameInfo.objectHandle = (uint64_t)texture->vkImage;
    nameInfo.pObjectName = name;
    vkSetDebugUtilsObjectNameEXT(texture->device->device, &nameInfo);
}

VkImage agfxNativeGetVkImage(agfxTexture* texture) { return texture->vkImage; }

// Indirect bundle
struct agfxIndirectBundle {
    agfxIndirectBundleCreateInfo createInfo;
    agfxBuffer* commandsBuffer;
    agfxBufferView* commandsBufferView;
    agfxBuffer* countBuffer;
    agfxBufferView* countBufferView;
    uint32_t stride;
};

// Mirrors agfxIndirectBundleTypeStride in agfx_d3d12.cpp: the commands are laid out at the
// command struct's exact size, one per slot.
static uint32_t agfxIndirectBundleTypeStride(agfxIndirectBundleType type)
{
    switch (type) {
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW:         return sizeof(agfxDrawCommand);
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW_INDEXED: return sizeof(agfxDrawIndexedCommand);
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW_MESH:    return sizeof(agfxDrawMeshCommand);
        case AGFX_INDIRECT_BUNDLE_TYPE_DISPATCH:     return sizeof(agfxDispatchCommand);
        default:                                     return sizeof(agfxDrawIndexedCommand);
    }
}

agfxIndirectBundle* agfxIndirectBundleCreate(agfxDevice* device, const agfxIndirectBundleCreateInfo* createInfo)
{
    agfxIndirectBundle* bundle = AgfxAlloc<agfxIndirectBundle>(device);
    bundle->createInfo = *createInfo;
    bundle->stride = agfxIndirectBundleTypeStride(createInfo->type);

    agfxBufferCreateInfo commandsInfo = {};
    commandsInfo.size = (uint64_t)createInfo->maxCommandCount * bundle->stride;
    commandsInfo.stride = bundle->stride;
    commandsInfo.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
    commandsInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    bundle->commandsBuffer = agfxBufferCreate(device, &commandsInfo);

    agfxBufferCreateInfo countInfo = {};
    countInfo.size = sizeof(uint32_t) * createInfo->maxCountCount;
    countInfo.stride = sizeof(uint32_t);
    countInfo.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
    countInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    bundle->countBuffer = agfxBufferCreate(device, &countInfo);

    // The producing shader reaches both buffers bindlessly (AGFXIndirectDraw*Bundle unpacks the
    // packed handle into two RWByteAddressBuffer indices), so each needs a writeable raw view.
    agfxBufferViewCreateInfo commandsViewInfo = {};
    commandsViewInfo.buffer = bundle->commandsBuffer;
    commandsViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    commandsViewInfo.writeable = 1;
    bundle->commandsBufferView = agfxBufferViewCreate(device, &commandsViewInfo);

    agfxBufferViewCreateInfo countViewInfo = {};
    countViewInfo.buffer = bundle->countBuffer;
    countViewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    countViewInfo.writeable = 1;
    bundle->countBufferView = agfxBufferViewCreate(device, &countViewInfo);

    return bundle;
}

void agfxIndirectBundleDestroy(agfxDevice* device, agfxIndirectBundle* bundle)
{
    if (bundle->commandsBufferView) agfxBufferViewDestroy(device, bundle->commandsBufferView);
    if (bundle->countBufferView) agfxBufferViewDestroy(device, bundle->countBufferView);
    agfxBufferDestroy(device, bundle->commandsBuffer);
    agfxBufferDestroy(device, bundle->countBuffer);
    AgfxFree(device, bundle);
}

uint64_t agfxIndirectBundleGetHandle(agfxIndirectBundle* bundle)
{
    return ((uint64_t)agfxBufferViewGetHandle(bundle->countBufferView) << 32) | agfxBufferViewGetHandle(bundle->commandsBufferView);
}

agfxBuffer* agfxIndirectBundleGetCommandsBuffer(agfxIndirectBundle* bundle)
{
    return bundle->commandsBuffer;
}

agfxBuffer* agfxIndirectBundleGetCountBuffer(agfxIndirectBundle* bundle)
{
    return bundle->countBuffer;
}

void agfxComputePassPrepareIndirectBundle(agfxComputePass* computePass, agfxIndirectBundle* bundle, const agfxIndirectBundleExecuteInfo* executeInfo)
{
    // No-op on Vulkan, same as D3D12: vkCmdDraw*IndirectCount reads the commands/count buffers
    // directly at execute time. Kept as a real call so callers don't need to branch per backend
    // (Metal's ICB-conversion pass does the real work here).
    (void)computePass;
    (void)bundle;
    (void)executeInfo;
}

// agfxRenderPassExecuteIndirectBundle / agfxComputePassExecuteIndirectBundle are defined further
// down, after agfxRenderPass and agfxComputePass are complete types.

// Compute pass
struct agfxComputePass {
    agfxDevice* device;
    agfxCommandBuffer* commandBuffer;
};

agfxComputePass* agfxComputePassBegin(agfxCommandBuffer* commandBuffer, const char* name)
{
    agfxComputePass* pass = AgfxAlloc<agfxComputePass>(commandBuffer->device);
    pass->device = commandBuffer->device;
    pass->commandBuffer = commandBuffer;
    return pass;
}

void agfxComputePassTextureUAVBarrier(agfxComputePass* computePass, agfxTexture* texture)
{
    // Write-to-read hazard between dispatches on a storage image: no layout change (UAV state is
    // GENERAL on both sides), just execution + memory dependency -- the sync2 equivalent of the
    // D3D12 backend's matching-scope enhanced barrier.
    VkImageMemoryBarrier2 imageBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    imageBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = texture->vkImage;
    imageBarrier.subresourceRange.aspectMask = agfxTextureFormatToVkImageAspect(texture->createInfo.format);
    imageBarrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    imageBarrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

    VkDependencyInfo dependencyInfo = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &imageBarrier;

    vkCmdPipelineBarrier2(computePass->commandBuffer->commandBuffer, &dependencyInfo);
}

void agfxComputePassBufferUAVBarrier(agfxComputePass* computePass, agfxBuffer* buffer)
{
    VkBufferMemoryBarrier2 bufferBarrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    bufferBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    bufferBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    bufferBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer->vkBuffer;
    bufferBarrier.size = VK_WHOLE_SIZE;

    VkDependencyInfo dependencyInfo = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dependencyInfo.bufferMemoryBarrierCount = 1;
    dependencyInfo.pBufferMemoryBarriers = &bufferBarrier;

    vkCmdPipelineBarrier2(computePass->commandBuffer->commandBuffer, &dependencyInfo);
}

// Shared by the two buffer<->texture copies: the public API speaks bytes (bytesPerRow/bytesPerImage,
// like D3D12 footprints and Metal), Vulkan speaks texels (bufferRowLength/bufferImageHeight). 0 stays
// 0 ("tightly packed" in both vocabularies).
static VkBufferImageCopy agfxVkBufferImageCopy(agfxTexture* texture, uint64_t bufferOffset, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, uint32_t bytesPerRow, uint32_t bytesPerImage)
{
    uint32_t bytesPerBlock, blockWidth, blockHeight;
    agfxTextureFormatGetBlockInfo(texture->createInfo.format, &bytesPerBlock, &blockWidth, &blockHeight);

    VkBufferImageCopy copy = {};
    copy.bufferOffset = bufferOffset;
    copy.bufferRowLength = bytesPerRow ? (bytesPerRow / bytesPerBlock) * blockWidth : 0;
    copy.bufferImageHeight = (bytesPerImage && bytesPerRow) ? (bytesPerImage / bytesPerRow) * blockHeight : 0;
    copy.imageSubresource.aspectMask = agfxTextureFormatToVkImageAspect(texture->createInfo.format);
    copy.imageSubresource.mipLevel = mipLevel;
    copy.imageSubresource.baseArrayLayer = layer;
    copy.imageSubresource.layerCount = 1;
    copy.imageOffset = { (int32_t)region->x, (int32_t)region->y, (int32_t)region->z };
    copy.imageExtent = { region->width, region->height ? region->height : 1, region->depth ? region->depth : 1 };
    return copy;
}

void agfxComputePassCopyTextureToBuffer(agfxComputePass* computePass, agfxTexture* texture, agfxBuffer* buffer, uint64_t bufferOffset, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, uint32_t bytesPerRow, uint32_t bytesPerImage)
{
    VkBufferImageCopy copy = agfxVkBufferImageCopy(texture, bufferOffset, region, mipLevel, layer, bytesPerRow, bytesPerImage);
    // The public API contract has the texture in AGFX_RESOURCE_STATE_COPY_SOURCE/DEST here, which
    // agfxResourceStateToVkImageLayout maps to exactly these layouts.
    vkCmdCopyImageToBuffer(computePass->commandBuffer->commandBuffer, texture->vkImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer->vkBuffer, 1, &copy);
}

void agfxComputePassCopyBufferToTexture(agfxComputePass* computePass, agfxBuffer* buffer, uint64_t sourceOffset, agfxTexture* texture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer, uint32_t bytesPerRow, uint32_t bytesPerImage)
{
    VkBufferImageCopy copy = agfxVkBufferImageCopy(texture, sourceOffset, region, mipLevel, layer, bytesPerRow, bytesPerImage);
    vkCmdCopyBufferToImage(computePass->commandBuffer->commandBuffer, buffer->vkBuffer, texture->vkImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
}

void agfxComputePassCopyBufferToBuffer(agfxComputePass* computePass, agfxBuffer* srcBuffer, agfxBuffer* dstBuffer, uint64_t srcOffset, uint64_t dstOffset, uint64_t size)
{
    VkBufferCopy copy = { srcOffset, dstOffset, size };
    vkCmdCopyBuffer(computePass->commandBuffer->commandBuffer, srcBuffer->vkBuffer, dstBuffer->vkBuffer, 1, &copy);
}

void agfxComputePassCopyTextureToTexture(agfxComputePass* computePass, agfxTexture* srcTexture, agfxTexture* dstTexture, const agfxTextureRegion* region, uint32_t mipLevel, uint32_t layer)
{
    // A single mip/layer pair applies to both sides, same as the D3D12 backend's shared subresource index.
    VkImageCopy copy = {};
    copy.srcSubresource.aspectMask = agfxTextureFormatToVkImageAspect(srcTexture->createInfo.format);
    copy.srcSubresource.mipLevel = mipLevel;
    copy.srcSubresource.baseArrayLayer = layer;
    copy.srcSubresource.layerCount = 1;
    copy.dstSubresource.aspectMask = agfxTextureFormatToVkImageAspect(dstTexture->createInfo.format);
    copy.dstSubresource.mipLevel = mipLevel;
    copy.dstSubresource.baseArrayLayer = layer;
    copy.dstSubresource.layerCount = 1;
    copy.srcOffset = { (int32_t)region->x, (int32_t)region->y, (int32_t)region->z };
    copy.dstOffset = { (int32_t)region->x, (int32_t)region->y, (int32_t)region->z };
    copy.extent = { region->width, region->height ? region->height : 1, region->depth ? region->depth : 1 };
    vkCmdCopyImage(computePass->commandBuffer->commandBuffer, srcTexture->vkImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstTexture->vkImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
}

void agfxComputePassSetPipeline(agfxComputePass* computePass, agfxComputePipeline* pipeline)
{
    vkCmdBindPipeline(computePass->commandBuffer->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->vkPipeline);
}

void agfxComputePassPushConstants(agfxComputePass* computePass, const void* data, uint32_t size)
{
    vkCmdPushConstants(computePass->commandBuffer->commandBuffer, computePass->device->globalPipelineLayout, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data);
}

void agfxComputePassDispatch(agfxComputePass* computePass, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    vkCmdDispatch(computePass->commandBuffer->commandBuffer, groupCountX, groupCountY, groupCountZ);
}

void agfxComputePassBuildAccelerationStructure(agfxComputePass* computePass, agfxAccelerationStructure* accelerationStructure, agfxBuffer* scratchBuffer, uint64_t scratchBufferOffset)
{
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = agfxVkAccelerationStructureBuildInfo(accelerationStructure);
    buildInfo.dstAccelerationStructure = accelerationStructure->vkAccelerationStructure;
    buildInfo.scratchData.deviceAddress = agfxVkBufferDeviceAddress(computePass->device, scratchBuffer->vkBuffer) + scratchBufferOffset;

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(accelerationStructure->primitiveCounts.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
        ranges[i] = {};
        ranges[i].primitiveCount = accelerationStructure->primitiveCounts[i];
    }
    const VkAccelerationStructureBuildRangeInfoKHR* pRanges = ranges.data();
    vkCmdBuildAccelerationStructuresKHR(computePass->commandBuffer->commandBuffer, 1, &buildInfo, &pRanges);
}

void agfxComputePassUpdateAccelerationStructure(agfxComputePass* computePass, agfxAccelerationStructure* srcAccelerationStructure, agfxAccelerationStructure* dstAccelerationStructure, agfxBuffer* scratchBuffer, uint64_t scratchBufferOffset)
{
    if (!dstAccelerationStructure->createInfo.allowUpdate) return;

    // Mirrors D3D12's PERFORM_UPDATE path: the update reuses the source's geometry description.
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = agfxVkAccelerationStructureBuildInfo(srcAccelerationStructure);
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = srcAccelerationStructure->vkAccelerationStructure;
    buildInfo.dstAccelerationStructure = dstAccelerationStructure->vkAccelerationStructure;
    buildInfo.scratchData.deviceAddress = agfxVkBufferDeviceAddress(computePass->device, scratchBuffer->vkBuffer) + scratchBufferOffset;

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(srcAccelerationStructure->primitiveCounts.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
        ranges[i] = {};
        ranges[i].primitiveCount = srcAccelerationStructure->primitiveCounts[i];
    }
    const VkAccelerationStructureBuildRangeInfoKHR* pRanges = ranges.data();
    vkCmdBuildAccelerationStructuresKHR(computePass->commandBuffer->commandBuffer, 1, &buildInfo, &pRanges);
}

void agfxComputePassCopyAccelerationStructure(agfxComputePass* computePass, agfxAccelerationStructure* srcAccelerationStructure, agfxAccelerationStructure* dstAccelerationStructure)
{
    VkCopyAccelerationStructureInfoKHR copyInfo = { VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR };
    copyInfo.src = srcAccelerationStructure->vkAccelerationStructure;
    copyInfo.dst = dstAccelerationStructure->vkAccelerationStructure;
    copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR;
    vkCmdCopyAccelerationStructureKHR(computePass->commandBuffer->commandBuffer, &copyInfo);
}

void agfxComputePassWriteCompactedSizeToBuffer(agfxComputePass* computePass, agfxAccelerationStructure* accelerationStructure, agfxBuffer* dstBuffer, uint64_t dstBufferOffset)
{
    // D3D12's EmitRaytracingAccelerationStructurePostbuildInfo writes the uint64 size straight to a
    // buffer; Vulkan routes it through a query pool, which lives on the AS so it isn't destroyed
    // while this command buffer is still in flight.
    agfxDevice* device = computePass->device;
    if (!accelerationStructure->compactedSizeQueryPool) {
        VkQueryPoolCreateInfo queryPoolInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        queryPoolInfo.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        queryPoolInfo.queryCount = 1;
        if (vkCreateQueryPool(device->device, &queryPoolInfo, nullptr, &accelerationStructure->compactedSizeQueryPool) != VK_SUCCESS) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxComputePassWriteCompactedSizeToBuffer: vkCreateQueryPool failed");
            return;
        }
    }

    VkCommandBuffer cmd = computePass->commandBuffer->commandBuffer;
    vkCmdResetQueryPool(cmd, accelerationStructure->compactedSizeQueryPool, 0, 1);
    vkCmdWriteAccelerationStructuresPropertiesKHR(cmd, 1, &accelerationStructure->vkAccelerationStructure,
        VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, accelerationStructure->compactedSizeQueryPool, 0);
    vkCmdCopyQueryPoolResults(cmd, accelerationStructure->compactedSizeQueryPool, 0, 1,
        dstBuffer->vkBuffer, dstBufferOffset, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
}

void agfxComputePassCompactAccelerationStructure(agfxComputePass* computePass, agfxAccelerationStructure* srcAccelerationStructure, agfxAccelerationStructure* dstAccelerationStructure)
{
    VkCopyAccelerationStructureInfoKHR copyInfo = { VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR };
    copyInfo.src = srcAccelerationStructure->vkAccelerationStructure;
    copyInfo.dst = dstAccelerationStructure->vkAccelerationStructure;
    copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
    vkCmdCopyAccelerationStructureKHR(computePass->commandBuffer->commandBuffer, &copyInfo);
}

void agfxComputePassEnd(agfxComputePass* computePass)
{
    AgfxFree(computePass->device, computePass);
}

// Buffer (struct hoisted to the top of the file)
static VkBufferCreateInfo agfxBufferBufferCreateInfo(agfxDevice* device, const agfxBufferCreateInfo* createInfo)
{
    VkBufferCreateInfo bufferCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferCreateInfo.size = createInfo->size;
    bufferCreateInfo.usage = agfxBufferUsageToVkBufferUsageFlags(device, createInfo->usage);
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    return bufferCreateInfo;
}

agfxBuffer* agfxBufferCreate(agfxDevice* device, const agfxBufferCreateInfo* createInfo)
{
    agfxBuffer* buffer = AgfxAlloc<agfxBuffer>(device);
    buffer->createInfo = *createInfo;
    buffer->device = device;

    VkBufferCreateInfo bufferCreateInfo = agfxBufferBufferCreateInfo(device, createInfo);
    if (vkCreateBuffer(device->device, &bufferCreateInfo, nullptr, &buffer->vkBuffer) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxBufferCreate: vkCreateBuffer failed");
        AgfxFree(device, buffer);
        return nullptr;
    }

    if (createInfo->heap) {
        if (vkBindBufferMemory(device->device, buffer->vkBuffer, createInfo->heap->vkMemory, createInfo->heapOffset) != VK_SUCCESS) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxBufferCreate: vkBindBufferMemory (placed) failed");
            vkDestroyBuffer(device->device, buffer->vkBuffer, nullptr);
            AgfxFree(device, buffer);
            return nullptr;
        }
        buffer->boundMemory = createInfo->heap->vkMemory;
        buffer->boundOffset = createInfo->heapOffset;
    } else {
        VkMemoryRequirements memoryRequirements = {};
        vkGetBufferMemoryRequirements(device->device, buffer->vkBuffer, &memoryRequirements);

        VkMemoryPropertyFlags requiredFlags = agfxBufferMemoryTypeToVkMemoryPropertyFlags(createInfo->memoryType);
        uint32_t memoryTypeIndex = agfxVkFindMemoryType(device, memoryRequirements.memoryTypeBits, requiredFlags);
        if (memoryTypeIndex == UINT32_MAX && (requiredFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
            // Not every device exposes a host-cached type; retry without it, since HOST_VISIBLE|HOST_COHERENT is the only hard requirement for readback correctness.
            memoryTypeIndex = agfxVkFindMemoryType(device, memoryRequirements.memoryTypeBits, requiredFlags & ~VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        }
        if (memoryTypeIndex == UINT32_MAX) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxBufferCreate: no memory type compatible with this buffer and its requested memory type");
            vkDestroyBuffer(device->device, buffer->vkBuffer, nullptr);
            AgfxFree(device, buffer);
            return nullptr;
        }

        VkMemoryAllocateFlagsInfo allocateFlagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
        allocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocateInfo.pNext = &allocateFlagsInfo;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        if (vkAllocateMemory(device->device, &allocateInfo, nullptr, &buffer->vkMemory) != VK_SUCCESS) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxBufferCreate: vkAllocateMemory failed");
            vkDestroyBuffer(device->device, buffer->vkBuffer, nullptr);
            AgfxFree(device, buffer);
            return nullptr;
        }

        if (vkBindBufferMemory(device->device, buffer->vkBuffer, buffer->vkMemory, 0) != VK_SUCCESS) {
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxBufferCreate: vkBindBufferMemory (committed) failed");
            vkFreeMemory(device->device, buffer->vkMemory, nullptr);
            vkDestroyBuffer(device->device, buffer->vkBuffer, nullptr);
            AgfxFree(device, buffer);
            return nullptr;
        }
        buffer->boundMemory = buffer->vkMemory;
        buffer->boundOffset = 0;
    }

    return buffer;
}

void agfxBufferDestroy(agfxDevice* device, agfxBuffer* buffer)
{
    if (buffer->vkBuffer) vkDestroyBuffer(device->device, buffer->vkBuffer, nullptr);
    if (buffer->vkMemory) vkFreeMemory(device->device, buffer->vkMemory, nullptr);
    AgfxFree(device, buffer);
}

void* agfxBufferMap(agfxBuffer* buffer)
{
    void* data = nullptr;
    if (vkMapMemory(buffer->device->device, buffer->boundMemory, buffer->boundOffset, buffer->createInfo.size, 0, &data) != VK_SUCCESS) {
        return nullptr;
    }
    return data;
}

void agfxBufferUnmap(agfxBuffer* buffer)
{
    vkUnmapMemory(buffer->device->device, buffer->boundMemory);
}

void agfxBufferSetName(agfxBuffer* buffer, const char* name)
{
    if (!vkSetDebugUtilsObjectNameEXT || !buffer->vkBuffer) return;

    VkDebugUtilsObjectNameInfoEXT nameInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
    nameInfo.objectType = VK_OBJECT_TYPE_BUFFER;
    nameInfo.objectHandle = (uint64_t)buffer->vkBuffer;
    nameInfo.pObjectName = name;
    vkSetDebugUtilsObjectNameEXT(buffer->device->device, &nameInfo);
}

void agfxBufferGetInfo(agfxBuffer* buffer, agfxBufferCreateInfo* info)
{
    *info = buffer->createInfo;
}

VkBuffer agfxNativeGetVkBuffer(agfxBuffer* buffer) { return buffer->vkBuffer; }

// Heap
agfxHeap* agfxHeapCreate(agfxDevice* device, const agfxHeapCreateInfo* createInfo)
{
    agfxHeap* heap = AgfxAlloc<agfxHeap>(device);
    heap->createInfo = *createInfo;

    VkMemoryPropertyFlags requiredFlags = agfxBufferMemoryTypeToVkMemoryPropertyFlags(createInfo->memoryType);
    // No specific resource yet, so there's no memoryTypeBits to intersect with -- every backend's heap
    // model requires a memory type that accepts both buffers and textures for GPU_ONLY heaps, which
    // holds for the device-local type(s) on every GPU AGFX targets.
    uint32_t memoryTypeIndex = agfxVkFindMemoryType(device, ~0u, requiredFlags);
    if (memoryTypeIndex == UINT32_MAX) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxHeapCreate: no memory type satisfies the requested heap memory type");
        AgfxFree(device, heap);
        return nullptr;
    }

    VkMemoryAllocateFlagsInfo allocateFlagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
    allocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocateInfo.pNext = &allocateFlagsInfo;
    allocateInfo.allocationSize = createInfo->size;
    allocateInfo.memoryTypeIndex = memoryTypeIndex;
    if (vkAllocateMemory(device->device, &allocateInfo, nullptr, &heap->vkMemory) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxHeapCreate: vkAllocateMemory failed");
        AgfxFree(device, heap);
        return nullptr;
    }

    return heap;
}

void agfxHeapDestroy(agfxDevice* device, agfxHeap* heap)
{
    if (heap->vkMemory) vkFreeMemory(device->device, heap->vkMemory, nullptr);
    AgfxFree(device, heap);
}

VkDeviceMemory agfxNativeGetVkHeapMemory(agfxHeap* heap) { return heap->vkMemory; }

void agfxDeviceGetTextureAllocationInfo(agfxDevice* device, const agfxTextureCreateInfo* createInfo, agfxAllocationInfo* info)
{
    *info = {};

    VkImageCreateInfo imageCreateInfo = agfxTextureImageCreateInfo(createInfo);
    VkImage image = VK_NULL_HANDLE;
    if (vkCreateImage(device->device, &imageCreateInfo, nullptr, &image) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxDeviceGetTextureAllocationInfo: vkCreateImage failed");
        return;
    }

    VkMemoryRequirements memoryRequirements = {};
    vkGetImageMemoryRequirements(device->device, image, &memoryRequirements);
    info->size = memoryRequirements.size;
    info->alignment = memoryRequirements.alignment;

    vkDestroyImage(device->device, image, nullptr);
}

void agfxDeviceGetBufferAllocationInfo(agfxDevice* device, const agfxBufferCreateInfo* createInfo, agfxAllocationInfo* info)
{
    *info = {};

    VkBufferCreateInfo bufferCreateInfo = agfxBufferBufferCreateInfo(device, createInfo);
    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device->device, &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxDeviceGetBufferAllocationInfo: vkCreateBuffer failed");
        return;
    }

    VkMemoryRequirements memoryRequirements = {};
    vkGetBufferMemoryRequirements(device->device, buffer, &memoryRequirements);
    info->size = memoryRequirements.size;
    info->alignment = memoryRequirements.alignment;

    vkDestroyBuffer(device->device, buffer, nullptr);
}

// Texture view
struct agfxTextureView {
    VkImageView vkImageView = VK_NULL_HANDLE;
    uint64_t slot = UINT64_MAX;
};

static VkImageViewType agfxTextureTypeToVkImageViewType(agfxTextureType type, agfxBool writeable)
{
    switch (type) {
        case AGFX_TEXTURE_TYPE_1D:       return VK_IMAGE_VIEW_TYPE_1D;
        case AGFX_TEXTURE_TYPE_2D:       return VK_IMAGE_VIEW_TYPE_2D;
        case AGFX_TEXTURE_TYPE_2D_ARRAY: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        case AGFX_TEXTURE_TYPE_3D:       return VK_IMAGE_VIEW_TYPE_3D;
        // Storage cube views don't exist -- view the six faces as a 2D array instead, exactly like
        // the D3D12 backend's UAV_DIMENSION_TEXTURE2DARRAY fallback for CUBE.
        case AGFX_TEXTURE_TYPE_CUBE:     return writeable ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
        default:                         return VK_IMAGE_VIEW_TYPE_2D;
    }
}

agfxTextureView* agfxTextureViewCreate(agfxDevice* device, const agfxTextureViewCreateInfo* createInfo)
{
    agfxTextureFormat format = (createInfo->format != AGFX_TEXTURE_FORMAT_UNKNOWN) ? createInfo->format : createInfo->texture->createInfo.format;

    VkImageViewCreateInfo viewCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewCreateInfo.image = createInfo->texture->vkImage;
    viewCreateInfo.viewType = agfxTextureTypeToVkImageViewType(createInfo->type, createInfo->writeable);
    viewCreateInfo.format = agfxTextureFormatToVkFormat(format);
    viewCreateInfo.subresourceRange.aspectMask = agfxTextureFormatToVkImageAspect(format);
    viewCreateInfo.subresourceRange.baseMipLevel = createInfo->baseMipLevel;
    viewCreateInfo.subresourceRange.levelCount = (createInfo->mipLevelCount == (uint32_t)AGFX_SUBRESOURCE_ALL_MIPS) ? VK_REMAINING_MIP_LEVELS : createInfo->mipLevelCount;
    viewCreateInfo.subresourceRange.baseArrayLayer = createInfo->baseArrayLayer;
    viewCreateInfo.subresourceRange.layerCount = (createInfo->arrayLayerCount == (uint32_t)AGFX_SUBRESOURCE_ALL_LAYERS) ? VK_REMAINING_ARRAY_LAYERS : createInfo->arrayLayerCount;

    agfxTextureView* textureView = AgfxAlloc<agfxTextureView>(device);
    if (vkCreateImageView(device->device, &viewCreateInfo, nullptr, &textureView->vkImageView) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxTextureViewCreate: vkCreateImageView failed");
        AgfxFree(device, textureView);
        return nullptr;
    }

    // The descriptor's imageLayout must match the layout the texture is actually in when accessed:
    // agfxResourceStateToVkImageLayout maps UNORDERED_ACCESS to GENERAL and the *_SHADER_RESOURCE
    // states to SHADER_READ_ONLY_OPTIMAL.
    VkDescriptorImageInfo imageInfo = {};
    imageInfo.imageView = textureView->vkImageView;
    imageInfo.imageLayout = createInfo->writeable ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    textureView->slot = agfxVkWriteResourceDescriptor(device,
        createInfo->writeable ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        &imageInfo, nullptr);
    if (textureView->slot == UINT64_MAX) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxTextureViewCreate: bindless resource slots exhausted");
        vkDestroyImageView(device->device, textureView->vkImageView, nullptr);
        AgfxFree(device, textureView);
        return nullptr;
    }

    return textureView;
}

void agfxTextureViewDestroy(agfxDevice* device, agfxTextureView* textureView)
{
    if (textureView->slot != UINT64_MAX) device->resourceSlotAllocator.free(textureView->slot);
    if (textureView->vkImageView) vkDestroyImageView(device->device, textureView->vkImageView, nullptr);
    AgfxFree(device, textureView);
}

uint64_t agfxTextureViewGetHandle(agfxTextureView* textureView)
{
    return textureView->slot;
}

// Sampler
struct agfxSampler {
    VkSampler vkSampler = VK_NULL_HANDLE;
    uint64_t slot = UINT64_MAX;
};

agfxSampler* agfxSamplerCreate(agfxDevice* device, const agfxSamplerCreateInfo* createInfo)
{
    // Same convention as the D3D12 backend: ALWAYS means "not a comparison sampler" (the tests rely
    // on this -- see test_sampler_comparison.cpp), anything else enables comparison with that op.
    bool isComparison = createInfo->comparisonFunction != AGFX_COMPARISON_FUNCTION_ALWAYS;

    VkSamplerCreateInfo samplerCreateInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    samplerCreateInfo.magFilter = agfxSamplerFilterToVkFilter(createInfo->filter);
    samplerCreateInfo.minFilter = agfxSamplerFilterToVkFilter(createInfo->filter);
    samplerCreateInfo.mipmapMode = agfxSamplerFilterToVkSamplerMipmapMode(createInfo->filter);
    samplerCreateInfo.addressModeU = agfxSamplerAddressModeToVkSamplerAddressMode(createInfo->addressModeU);
    samplerCreateInfo.addressModeV = agfxSamplerAddressModeToVkSamplerAddressMode(createInfo->addressModeV);
    samplerCreateInfo.addressModeW = agfxSamplerAddressModeToVkSamplerAddressMode(createInfo->addressModeW);
    samplerCreateInfo.mipLodBias = createInfo->mipLodBias;
    // Anisotropy stays off: the D3D12 backend only ever emits the non-anisotropic MIN_MAG_MIP_*
    // filters, so enabling it here would diverge from the shared goldens.
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.compareEnable = isComparison ? VK_TRUE : VK_FALSE;
    samplerCreateInfo.compareOp = agfxComparisonFunctionToVkCompareOp(createInfo->comparisonFunction);
    samplerCreateInfo.minLod = createInfo->minLod;
    samplerCreateInfo.maxLod = createInfo->maxLod;
    // Matches D3D12's zero-initialized BorderColor ({0,0,0,0}).
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;

    agfxSampler* sampler = AgfxAlloc<agfxSampler>(device);
    if (vkCreateSampler(device->device, &samplerCreateInfo, nullptr, &sampler->vkSampler) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxSamplerCreate: vkCreateSampler failed");
        AgfxFree(device, sampler);
        return nullptr;
    }

    sampler->slot = agfxVkWriteSamplerDescriptor(device, sampler->vkSampler);
    if (sampler->slot == UINT64_MAX) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxSamplerCreate: bindless sampler slots exhausted");
        vkDestroySampler(device->device, sampler->vkSampler, nullptr);
        AgfxFree(device, sampler);
        return nullptr;
    }

    return sampler;
}

void agfxSamplerDestroy(agfxDevice* device, agfxSampler* sampler)
{
    if (sampler->slot != UINT64_MAX) device->samplerSlotAllocator.free(sampler->slot);
    if (sampler->vkSampler) vkDestroySampler(device->device, sampler->vkSampler, nullptr);
    AgfxFree(device, sampler);
}

uint64_t agfxSamplerGetHandle(agfxSampler* sampler)
{
    return sampler->slot;
}

// Buffer view
struct agfxBufferView {
    uint64_t slot = UINT64_MAX;
};

agfxBufferView* agfxBufferViewCreate(agfxDevice* device, const agfxBufferViewCreateInfo* createInfo)
{
    // HLSL's (RW)ByteAddressBuffer and (RW)StructuredBuffer both compile to SPIR-V storage buffers
    // (read-only-ness is a shader-side decoration, so RAW/STRUCTURED share one descriptor type
    // regardless of writeable) and ConstantBuffer to a uniform buffer.
    VkDescriptorType type = (createInfo->type == AGFX_BUFFER_VIEW_TYPE_CONSTANT)
        ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = createInfo->buffer->vkBuffer;
    bufferInfo.offset = createInfo->offset;
    // The public API has no size field: views always run to the end of the buffer.
    bufferInfo.range = VK_WHOLE_SIZE;

    agfxBufferView* bufferView = AgfxAlloc<agfxBufferView>(device);
    bufferView->slot = agfxVkWriteResourceDescriptor(device, type, nullptr, &bufferInfo);
    if (bufferView->slot == UINT64_MAX) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxBufferViewCreate: bindless resource slots exhausted");
        AgfxFree(device, bufferView);
        return nullptr;
    }
    return bufferView;
}

void agfxBufferViewDestroy(agfxDevice* device, agfxBufferView* bufferView)
{
    if (bufferView->slot != UINT64_MAX) device->resourceSlotAllocator.free(bufferView->slot);
    AgfxFree(device, bufferView);
}

uint64_t agfxBufferViewGetHandle(agfxBufferView* bufferView)
{
    return bufferView->slot;
}

// Render target
struct agfxRenderTarget {
    VkImageView vkImageView = VK_NULL_HANDLE;
};

agfxRenderTarget* agfxRenderTargetCreate(agfxDevice* device, const agfxRenderTargetCreateInfo* createInfo)
{
    agfxTextureFormat format = (createInfo->format != AGFX_TEXTURE_FORMAT_UNKNOWN) ? createInfo->format : createInfo->texture->createInfo.format;

    // Always a plain 2D view of exactly one mip and one layer -- a single cube face or array slice
    // is a valid 2D attachment (the D3D12 RTV/DSV descs make the same single-subresource choice).
    VkImageViewCreateInfo viewCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewCreateInfo.image = createInfo->texture->vkImage;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = agfxTextureFormatToVkFormat(format);
    viewCreateInfo.subresourceRange.aspectMask = createInfo->isDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    viewCreateInfo.subresourceRange.baseMipLevel = createInfo->mipLevel;
    viewCreateInfo.subresourceRange.levelCount = 1;
    viewCreateInfo.subresourceRange.baseArrayLayer = createInfo->arrayLayer;
    viewCreateInfo.subresourceRange.layerCount = 1;

    agfxRenderTarget* renderTarget = AgfxAlloc<agfxRenderTarget>(device);
    if (vkCreateImageView(device->device, &viewCreateInfo, nullptr, &renderTarget->vkImageView) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxRenderTargetCreate: vkCreateImageView failed");
        AgfxFree(device, renderTarget);
        return nullptr;
    }
    return renderTarget;
}

void agfxRenderTargetDestroy(agfxDevice* device, agfxRenderTarget* renderTarget)
{
    if (renderTarget->vkImageView) vkDestroyImageView(device->device, renderTarget->vkImageView, nullptr);
    AgfxFree(device, renderTarget);
}

// Render pass
struct agfxRenderPass {
    agfxDevice* device;
    agfxCommandBuffer* commandBuffer;
    bool hasDebugLabel;
};

static VkRenderingAttachmentInfo agfxRenderPassAttachmentToVkRenderingAttachmentInfo(const agfxRenderPassAttachment* attachment, agfxBool isDepth)
{
    VkRenderingAttachmentInfo info = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    info.imageView = attachment->renderTarget->vkImageView;
    // Must match the layout the attachment was transitioned into: agfxResourceStateToVkImageLayout
    // maps RENDER_TARGET -> COLOR_ATTACHMENT_OPTIMAL and DEPTH_WRITE -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
    info.imageLayout = isDepth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    info.loadOp = agfxLoadOperationToVkAttachmentLoadOp(attachment->loadOp);
    info.storeOp = agfxStoreOperationToVkAttachmentStoreOp(attachment->storeOp);
    if (isDepth) {
        info.clearValue.depthStencil = { attachment->clearDepth, 0 };
    } else {
        memcpy(info.clearValue.color.float32, attachment->clearColor, sizeof(attachment->clearColor));
    }
    return info;
}

agfxRenderPass* agfxRenderPassBegin(agfxCommandBuffer* cmdBuffer, const agfxRenderPassCreateInfo* createInfo)
{
    agfxRenderPass* pass = AgfxAlloc<agfxRenderPass>(cmdBuffer->device);
    pass->device = cmdBuffer->device;
    pass->commandBuffer = cmdBuffer;

    pass->hasDebugLabel = vkCmdBeginDebugUtilsLabelEXT && createInfo->name;
    if (pass->hasDebugLabel) {
        VkDebugUtilsLabelEXT label = { VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
        label.pLabelName = createInfo->name;
        vkCmdBeginDebugUtilsLabelEXT(cmdBuffer->commandBuffer, &label);
    }

    // Unlike the D3D12 backend (explicit ClearRenderTargetView + OMSetRenderTargets, storeOp
    // ignored), dynamic rendering carries load/store/clear natively per attachment.
    VkRenderingAttachmentInfo colorAttachments[8] = {};
    for (uint32_t i = 0; i < createInfo->colorAttachmentCount && i < 8; ++i) {
        colorAttachments[i] = agfxRenderPassAttachmentToVkRenderingAttachmentInfo(&createInfo->colorAttachments[i], 0);
    }
    VkRenderingAttachmentInfo depthAttachment = {};
    if (createInfo->hasDepthAttachment) {
        depthAttachment = agfxRenderPassAttachmentToVkRenderingAttachmentInfo(&createInfo->depthAttachment, 1);
    }

    VkRenderingInfo renderingInfo = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    renderingInfo.renderArea = { { 0, 0 }, { createInfo->width, createInfo->height } };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = createInfo->colorAttachmentCount;
    renderingInfo.pColorAttachments = colorAttachments;
    renderingInfo.pDepthAttachment = createInfo->hasDepthAttachment ? &depthAttachment : nullptr;
    vkCmdBeginRendering(cmdBuffer->commandBuffer, &renderingInfo);

    return pass;
}

void agfxRenderPassSetViewport(agfxRenderPass* renderPass, float x, float y, float width, float height, float minDepth, float maxDepth)
{
    // Negative-height viewport (core since Vulkan 1.1) flips the framebuffer Y axis to match D3D12/
    // Metal conventions -- the shader compiler passes no -fvk-invert-y, and the golden images are
    // shared across backends, so without this every rendered image comes out vertically mirrored.
    VkViewport viewport = {};
    viewport.x = x;
    viewport.y = y + height;
    viewport.width = width;
    viewport.height = -height;
    viewport.minDepth = minDepth;
    viewport.maxDepth = maxDepth;
    vkCmdSetViewport(renderPass->commandBuffer->commandBuffer, 0, 1, &viewport);
}

void agfxRenderPassSetScissor(agfxRenderPass* renderPass, uint32_t x, uint32_t y, uint32_t width, uint32_t height)
{
    VkRect2D scissor = { { (int32_t)x, (int32_t)y }, { width, height } };
    vkCmdSetScissor(renderPass->commandBuffer->commandBuffer, 0, 1, &scissor);
}

void agfxRenderPassSetPipeline(agfxRenderPass* renderPass, agfxRenderPipeline* pipeline)
{
    vkCmdBindPipeline(renderPass->commandBuffer->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->vkPipeline);
}

void agfxRenderPassPushConstants(agfxRenderPass* renderPass, const void* data, uint32_t size)
{
    vkCmdPushConstants(renderPass->commandBuffer->commandBuffer, renderPass->device->globalPipelineLayout, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data);
}

void agfxRenderPassDraw(agfxRenderPass* renderPass, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
    vkCmdDraw(renderPass->commandBuffer->commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
}

void agfxRenderPassDrawIndexed(agfxRenderPass* renderPass, agfxBuffer* indexBuffer, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, uint32_t vertexOffset, uint32_t firstInstance)
{
    // Index format inferred from the buffer's stride, exactly like the D3D12 backend's per-draw IBV.
    VkIndexType indexType = (indexBuffer->createInfo.stride == 2) ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    vkCmdBindIndexBuffer(renderPass->commandBuffer->commandBuffer, indexBuffer->vkBuffer, 0, indexType);
    vkCmdDrawIndexed(renderPass->commandBuffer->commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

void agfxRenderPassDrawMesh(agfxRenderPass* renderPass, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    vkCmdDrawMeshTasksEXT(renderPass->commandBuffer->commandBuffer, groupCountX, groupCountY, groupCountZ);
}

void agfxRenderPassEnd(agfxRenderPass* renderPass)
{
    vkCmdEndRendering(renderPass->commandBuffer->commandBuffer);
    if (renderPass->hasDebugLabel) {
        vkCmdEndDebugUtilsLabelEXT(renderPass->commandBuffer->commandBuffer);
    }
    AgfxFree(renderPass->device, renderPass);
}

// Defined down here (rather than with the rest of the Indirect bundle section) because they need
// agfxRenderPass/agfxComputePass to be complete types.
void agfxRenderPassExecuteIndirectBundle(agfxRenderPass* renderPass, agfxIndirectBundle* bundle, const agfxIndirectBundleExecuteInfo* executeInfo)
{
    VkCommandBuffer cmd = renderPass->commandBuffer->commandBuffer;
    agfxDevice* device = renderPass->device;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, executeInfo->renderPipeline->vkPipeline);
    vkCmdPushConstants(cmd, device->globalPipelineLayout, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(executeInfo->pushConstants), executeInfo->pushConstants);

    // The commands are D3D12-command-signature shaped: a leading uint32 drawID, then the native
    // draw arguments. Vulkan consumes the native arguments directly by starting 4 bytes in and
    // striding over the drawID; shaders recover their identity from the linear DrawIndex builtin
    // instead of D3D12's patched b1 constant (see AGFX_DRAW_ID in data/shaders/agfx.h), so apps
    // needing the culling shader's drawID must route it through their own indirection buffer.
    VkDeviceSize commandsOffset = (VkDeviceSize)executeInfo->commandOffset * bundle->stride + sizeof(uint32_t);
    VkDeviceSize countOffset = (VkDeviceSize)executeInfo->countIndex * sizeof(uint32_t);

    switch (bundle->createInfo.type) {
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW:
            vkCmdDrawIndirectCount(cmd, bundle->commandsBuffer->vkBuffer, commandsOffset, bundle->countBuffer->vkBuffer, countOffset, executeInfo->maxCommandCount, bundle->stride);
            break;
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW_INDEXED: {
            VkIndexType indexType = (executeInfo->indexBuffer->createInfo.stride == 2) ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
            vkCmdBindIndexBuffer(cmd, executeInfo->indexBuffer->vkBuffer, 0, indexType);
            vkCmdDrawIndexedIndirectCount(cmd, bundle->commandsBuffer->vkBuffer, commandsOffset, bundle->countBuffer->vkBuffer, countOffset, executeInfo->maxCommandCount, bundle->stride);
            break;
        }
        case AGFX_INDIRECT_BUNDLE_TYPE_DRAW_MESH:
            vkCmdDrawMeshTasksIndirectCountEXT(cmd, bundle->commandsBuffer->vkBuffer, commandsOffset, bundle->countBuffer->vkBuffer, countOffset, executeInfo->maxCommandCount, bundle->stride);
            break;
        default:
            agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxRenderPassExecuteIndirectBundle: DISPATCH bundles execute on a compute pass, not a render pass");
            break;
    }
}

void agfxComputePassExecuteIndirectBundle(agfxComputePass* computePass, agfxIndirectBundle* bundle, const agfxIndirectBundleExecuteInfo* executeInfo)
{
    VkCommandBuffer cmd = computePass->commandBuffer->commandBuffer;
    agfxDevice* device = computePass->device;

    if (bundle->createInfo.type != AGFX_INDIRECT_BUNDLE_TYPE_DISPATCH) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxComputePassExecuteIndirectBundle: only DISPATCH bundles execute on a compute pass");
        return;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, executeInfo->computePipeline->vkPipeline);
    vkCmdPushConstants(cmd, device->globalPipelineLayout, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(executeInfo->pushConstants), executeInfo->pushConstants);

    // Vulkan has no vkCmdDispatchIndirectCount, so unlike the draw bundle types the GPU-side count
    // cannot clamp anything here: all maxCommandCount slots are replayed. A producer that appends
    // fewer must leave the unused slots zeroed (a {0,0,0} dispatch is free), which costs nothing
    // since agfxDispatchCommand has no drawID and slots are position-independent.
    for (uint32_t i = 0; i < executeInfo->maxCommandCount; ++i) {
        vkCmdDispatchIndirect(cmd, bundle->commandsBuffer->vkBuffer, (VkDeviceSize)(executeInfo->commandOffset + i) * bundle->stride);
    }
}

// Swap chain
struct agfxSwapChain {
    agfxDevice* device = nullptr;
    agfxSwapChainCreateInfo createInfo{};
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR vkSwapChain = VK_NULL_HANDLE;
    VkQueue vkQueue = VK_NULL_HANDLE;
    agfxTextureFormat format = AGFX_TEXTURE_FORMAT_UNKNOWN;
    VkFormat vkFormat = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    std::vector<agfxTexture*> backBuffers;
    // Binary semaphores (timeline semaphores can't be used with acquire/present), rotated together:
    // acquire[i] is signaled by vkAcquireNextImageKHR and immediately absorbed into the queue's
    // submission order; present[i] is signaled by an empty submit after the frame's work and waited
    // on by vkQueuePresentKHR. imageCount+1 slots means a slot is only reused after a full cycle of
    // presents, by which point FIFO acquire backpressure guarantees its previous wait retired.
    std::vector<VkSemaphore> acquireSemaphores;
    std::vector<VkSemaphore> presentSemaphores;
    uint32_t semaphoreIndex = 0;
    uint32_t imageIndex = 0;
};

// Surface creation for the three Linux display protocols without requiring X11/XCB dev headers at
// build time: the create-info structs are re-declared here ABI-compatibly (their sType values are
// in vulkan_core.h) and the entry points fetched through vkGetInstanceProcAddr -- volk only
// declares/loads these functions when built with VK_USE_PLATFORM_* defines, which would drag the
// platform headers in.
struct agfxVkXlibSurfaceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    void* dpy;            // Display*
    unsigned long window; // Window (XID)
};
struct agfxVkXcbSurfaceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    void* connection; // xcb_connection_t*
    uint32_t window;  // xcb_window_t
};
struct agfxVkWaylandSurfaceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    void* display; // wl_display*
    void* surface; // wl_surface*
};

static VkSurfaceKHR agfxVkCreateSurface(agfxDevice* device, const agfxLinuxWindowHandle* handle)
{
    typedef VkResult (*agfxPfnCreateSurface)(VkInstance, const void*, const VkAllocationCallbacks*, VkSurfaceKHR*);

    const char* entryPointName = nullptr;
    agfxVkXlibSurfaceCreateInfo xlibInfo = {};
    agfxVkXcbSurfaceCreateInfo xcbInfo = {};
    agfxVkWaylandSurfaceCreateInfo waylandInfo = {};
    const void* surfaceCreateInfo = nullptr;

    switch (device->createInfo.displayServerProtocol) {
        case AGFX_DISPLAY_SERVER_PROTOCOL_X11:
            entryPointName = "vkCreateXlibSurfaceKHR";
            xlibInfo = { VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR, nullptr, 0, handle->display, (unsigned long)handle->window };
            surfaceCreateInfo = &xlibInfo;
            break;
        case AGFX_DISPLAY_SERVER_PROTOCOL_XCB:
            entryPointName = "vkCreateXcbSurfaceKHR";
            xcbInfo = { VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR, nullptr, 0, handle->display, (uint32_t)handle->window };
            surfaceCreateInfo = &xcbInfo;
            break;
        case AGFX_DISPLAY_SERVER_PROTOCOL_WAYLAND:
            entryPointName = "vkCreateWaylandSurfaceKHR";
            waylandInfo = { VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR, nullptr, 0, handle->display, (void*)(uintptr_t)handle->window };
            surfaceCreateInfo = &waylandInfo;
            break;
    }

    agfxPfnCreateSurface createSurface = (agfxPfnCreateSurface)vkGetInstanceProcAddr(device->instance, entryPointName);
    if (!createSurface) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxSwapChainCreate: %s unavailable -- was the device created with the right displayServerProtocol?", entryPointName);
        return VK_NULL_HANDLE;
    }

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (createSurface(device->instance, surfaceCreateInfo, nullptr, &surface) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxSwapChainCreate: surface creation via %s failed", entryPointName);
        return VK_NULL_HANDLE;
    }
    return surface;
}

// Mirrors the D3D12 backend's format policy: B8G8R8A8_UNORM for SDR, R16G16B16A16_SFLOAT + scRGB
// linear for HDR, silently staying SDR when the surface can't do HDR (D3D12's CheckHDRSupport path).
static void agfxVkPickSurfaceFormat(agfxDevice* device, agfxSwapChain* swapChain)
{
    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device->physicalDevice, swapChain->surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device->physicalDevice, swapChain->surface, &formatCount, formats.data());

    if (swapChain->createInfo.isHDR) {
        for (const VkSurfaceFormatKHR& format : formats) {
            if (format.format == VK_FORMAT_R16G16B16A16_SFLOAT && format.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) {
                swapChain->vkFormat = format.format;
                swapChain->colorSpace = format.colorSpace;
                swapChain->format = AGFX_TEXTURE_FORMAT_RGBA16F;
                return;
            }
        }
    }
    for (const VkSurfaceFormatKHR& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            swapChain->vkFormat = format.format;
            swapChain->colorSpace = format.colorSpace;
            swapChain->format = AGFX_TEXTURE_FORMAT_BGRA8_UNORM;
            return;
        }
    }
    for (const VkSurfaceFormatKHR& format : formats) {
        if (format.format == VK_FORMAT_R8G8B8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            swapChain->vkFormat = format.format;
            swapChain->colorSpace = format.colorSpace;
            swapChain->format = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
            return;
        }
    }
    // Last resort so creation still proceeds -- every implementation exposes at least one format.
    swapChain->vkFormat = formats[0].format;
    swapChain->colorSpace = formats[0].colorSpace;
    swapChain->format = AGFX_TEXTURE_FORMAT_BGRA8_UNORM;
}

static void agfxVkDestroySwapChainObjects(agfxDevice* device, agfxSwapChain* swapChain)
{
    for (agfxTexture* backBuffer : swapChain->backBuffers) {
        AgfxFree(device, backBuffer); // ownsImage == false: the VkImages die with the VkSwapchainKHR.
    }
    swapChain->backBuffers.clear();
    for (VkSemaphore semaphore : swapChain->acquireSemaphores) vkDestroySemaphore(device->device, semaphore, nullptr);
    for (VkSemaphore semaphore : swapChain->presentSemaphores) vkDestroySemaphore(device->device, semaphore, nullptr);
    swapChain->acquireSemaphores.clear();
    swapChain->presentSemaphores.clear();
}

// Shared by create and resize (which passes the previous swapchain as oldSwapchain and destroys it
// afterwards). The caller is responsible for the GPU being drained on the resize path, per the
// public API contract.
static bool agfxVkCreateSwapChainObjects(agfxDevice* device, agfxSwapChain* swapChain, uint32_t width, uint32_t height)
{
    VkSurfaceCapabilitiesKHR caps = {};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device->physicalDevice, swapChain->surface, &caps) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxSwapChainCreate: vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed");
        return false;
    }

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) { // Surface lets the swapchain decide; clamp the requested size.
        extent.width = width < caps.minImageExtent.width ? caps.minImageExtent.width : (width > caps.maxImageExtent.width ? caps.maxImageExtent.width : width);
        extent.height = height < caps.minImageExtent.height ? caps.minImageExtent.height : (height > caps.maxImageExtent.height ? caps.maxImageExtent.height : height);
    }

    uint32_t imageCount = swapChain->createInfo.imageCount;
    if (imageCount < caps.minImageCount) imageCount = caps.minImageCount;
    if (caps.maxImageCount != 0 && imageCount > caps.maxImageCount) imageCount = caps.maxImageCount;

    // vsync -> FIFO (always available); no vsync -> MAILBOX if the driver has it, IMMEDIATE otherwise.
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (!swapChain->createInfo.vsync) {
        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device->physicalDevice, swapChain->surface, &presentModeCount, nullptr);
        std::vector<VkPresentModeKHR> presentModes(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device->physicalDevice, swapChain->surface, &presentModeCount, presentModes.data());
        for (VkPresentModeKHR mode : presentModes) {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR) { presentMode = mode; break; }
            if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR) presentMode = mode;
        }
    }

    VkSwapchainCreateInfoKHR swapchainCreateInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    swapchainCreateInfo.surface = swapChain->surface;
    swapchainCreateInfo.minImageCount = imageCount;
    swapchainCreateInfo.imageFormat = swapChain->vkFormat;
    swapchainCreateInfo.imageColorSpace = swapChain->colorSpace;
    swapchainCreateInfo.imageExtent = extent;
    swapchainCreateInfo.imageArrayLayers = 1;
    swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        | (caps.supportedUsageFlags & (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainCreateInfo.preTransform = caps.currentTransform;
    swapchainCreateInfo.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
        ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    swapchainCreateInfo.presentMode = presentMode;
    swapchainCreateInfo.clipped = VK_TRUE;
    swapchainCreateInfo.oldSwapchain = swapChain->vkSwapChain;

    VkSwapchainKHR newSwapChain = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(device->device, &swapchainCreateInfo, nullptr, &newSwapChain) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxSwapChainCreate: vkCreateSwapchainKHR failed");
        return false;
    }
    if (swapChain->vkSwapChain) vkDestroySwapchainKHR(device->device, swapChain->vkSwapChain, nullptr);
    swapChain->vkSwapChain = newSwapChain;

    uint32_t actualImageCount = 0;
    vkGetSwapchainImagesKHR(device->device, swapChain->vkSwapChain, &actualImageCount, nullptr);
    std::vector<VkImage> images(actualImageCount);
    vkGetSwapchainImagesKHR(device->device, swapChain->vkSwapChain, &actualImageCount, images.data());

    swapChain->backBuffers.reserve(actualImageCount);
    for (uint32_t i = 0; i < actualImageCount; ++i) {
        agfxTexture* texture = AgfxAlloc<agfxTexture>(device);
        texture->vkImage = images[i];
        texture->device = device;
        texture->ownsImage = false;
        texture->createInfo.type = AGFX_TEXTURE_TYPE_2D;
        texture->createInfo.format = swapChain->format;
        texture->createInfo.usage = AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT;
        texture->createInfo.width = extent.width;
        texture->createInfo.height = extent.height;
        texture->createInfo.depthOrArrayLayers = 1;
        texture->createInfo.mipLevels = 1;
        swapChain->backBuffers.push_back(texture);
    }

    VkSemaphoreCreateInfo semaphoreCreateInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    for (uint32_t i = 0; i < actualImageCount + 1; ++i) {
        VkSemaphore acquireSemaphore = VK_NULL_HANDLE;
        VkSemaphore presentSemaphore = VK_NULL_HANDLE;
        vkCreateSemaphore(device->device, &semaphoreCreateInfo, nullptr, &acquireSemaphore);
        vkCreateSemaphore(device->device, &semaphoreCreateInfo, nullptr, &presentSemaphore);
        swapChain->acquireSemaphores.push_back(acquireSemaphore);
        swapChain->presentSemaphores.push_back(presentSemaphore);
    }
    swapChain->semaphoreIndex = 0;

    return true;
}

agfxSwapChain* agfxSwapChainCreate(agfxDevice* device, const agfxSwapChainCreateInfo* createInfo)
{
    agfxSwapChain* swapChain = AgfxAlloc<agfxSwapChain>(device);
    swapChain->device = device;
    swapChain->createInfo = *createInfo;
    swapChain->vkQueue = createInfo->queue ? createInfo->queue->vkQueue : device->graphicsQueue;

    swapChain->surface = agfxVkCreateSurface(device, (const agfxLinuxWindowHandle*)createInfo->handle);
    if (!swapChain->surface) {
        AgfxFree(device, swapChain);
        return nullptr;
    }

    VkBool32 presentSupported = VK_FALSE;
    uint32_t presentFamily = createInfo->queue ? createInfo->queue->familyIndex : device->graphicsFamily;
    vkGetPhysicalDeviceSurfaceSupportKHR(device->physicalDevice, presentFamily, swapChain->surface, &presentSupported);
    if (!presentSupported) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxSwapChainCreate: the given queue's family cannot present to this surface");
        vkDestroySurfaceKHR(device->instance, swapChain->surface, nullptr);
        AgfxFree(device, swapChain);
        return nullptr;
    }

    agfxVkPickSurfaceFormat(device, swapChain);

    if (!agfxVkCreateSwapChainObjects(device, swapChain, createInfo->width, createInfo->height)) {
        vkDestroySurfaceKHR(device->instance, swapChain->surface, nullptr);
        AgfxFree(device, swapChain);
        return nullptr;
    }

    return swapChain;
}

void agfxSwapChainDestroy(agfxDevice* device, agfxSwapChain* swapChain)
{
    agfxVkDestroySwapChainObjects(device, swapChain);
    if (swapChain->vkSwapChain) vkDestroySwapchainKHR(device->device, swapChain->vkSwapChain, nullptr);
    if (swapChain->surface) vkDestroySurfaceKHR(device->instance, swapChain->surface, nullptr);
    AgfxFree(device, swapChain);
}

void agfxSwapChainResize(agfxDevice* device, agfxSwapChain* swapChain, uint32_t width, uint32_t height)
{
    agfxVkDestroySwapChainObjects(device, swapChain);
    swapChain->createInfo.width = width;
    swapChain->createInfo.height = height;
    agfxVkCreateSwapChainObjects(device, swapChain, width, height); // Recreates via oldSwapchain.
}

agfxTextureFormat agfxSwapChainGetFormat(agfxSwapChain* swapChain)
{
    return swapChain->format;
}

agfxTexture* agfxSwapChainAcquireNextTexture(agfxSwapChain* swapChain)
{
    agfxDevice* device = swapChain->device;
    VkSemaphore acquireSemaphore = swapChain->acquireSemaphores[swapChain->semaphoreIndex];

    VkResult result = vkAcquireNextImageKHR(device->device, swapChain->vkSwapChain, UINT64_MAX, acquireSemaphore, VK_NULL_HANDLE, &swapChain->imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        agfxLog(device, AGFX_LOG_SEVERITY_WARNING, "agfxSwapChainAcquireNextTexture: swap chain out of date -- resize/recreate it");
        return nullptr;
    }

    // Absorb the binary acquire semaphore into the queue's submission order right away with an
    // empty submit: everything the caller submits afterwards (the frame's rendering) is ordered
    // behind it, so agfxCommandQueueSubmit never needs to know about swap chain semaphores.
    VkSemaphoreSubmitInfo waitInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    waitInfo.semaphore = acquireSemaphore;
    waitInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkSubmitInfo2 submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.waitSemaphoreInfoCount = 1;
    submitInfo.pWaitSemaphoreInfos = &waitInfo;
    vkQueueSubmit2(swapChain->vkQueue, 1, &submitInfo, VK_NULL_HANDLE);

    return swapChain->backBuffers[swapChain->imageIndex];
}

void agfxSwapChainPresent(agfxSwapChain* swapChain)
{
    VkSemaphore presentSemaphore = swapChain->presentSemaphores[swapChain->semaphoreIndex];

    // The mirror of the acquire-side empty submit: signal the binary present semaphore after all
    // previously submitted work on the queue (i.e. the frame that just rendered to the back buffer),
    // then hand it to vkQueuePresentKHR as its wait.
    VkSemaphoreSubmitInfo signalInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    signalInfo.semaphore = presentSemaphore;
    signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkSubmitInfo2 submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &signalInfo;
    vkQueueSubmit2(swapChain->vkQueue, 1, &submitInfo, VK_NULL_HANDLE);

    VkPresentInfoKHR presentInfo = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &presentSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapChain->vkSwapChain;
    presentInfo.pImageIndices = &swapChain->imageIndex;

    VkResult result = vkQueuePresentKHR(swapChain->vkQueue, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        agfxLog(swapChain->device, AGFX_LOG_SEVERITY_WARNING, "agfxSwapChainPresent: swap chain out of date -- resize/recreate it");
    }

    swapChain->semaphoreIndex = (swapChain->semaphoreIndex + 1) % (uint32_t)swapChain->acquireSemaphores.size();
}

// Shader module
struct agfxShaderModule {
    VkShaderModule vkModule = VK_NULL_HANDLE;
    char entryPoint[256];
    agfxShaderModuleType type;
};

agfxShaderModule* agfxShaderModuleCreate(agfxDevice* device, const agfxShaderModuleCreateInfo* createInfo)
{
    agfxShaderModule* shaderModule = AgfxAlloc<agfxShaderModule>(device);
    shaderModule->type = createInfo->type;
    agfxVkCopyBoundedString(shaderModule->entryPoint, sizeof(shaderModule->entryPoint), createInfo->entryPoint, sizeof(shaderModule->entryPoint));

    // The Linux shader compiler emits raw SPIR-V (DXC's -spirv target, agfx_shader_compiler_linux.cpp),
    // which is exactly what vkCreateShaderModule expects -- unlike D3D12, which just retains the DXIL
    // blob and hands it to the PSO desc later, Vulkan needs a real module object up front.
    VkShaderModuleCreateInfo moduleCreateInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    moduleCreateInfo.codeSize = createInfo->codeSize;
    moduleCreateInfo.pCode = (const uint32_t*)createInfo->code;

    if (vkCreateShaderModule(device->device, &moduleCreateInfo, nullptr, &shaderModule->vkModule) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxShaderModuleCreate: vkCreateShaderModule failed");
        AgfxFree(device, shaderModule);
        return nullptr;
    }

    return shaderModule;
}

void agfxShaderModuleDestroy(agfxDevice* device, agfxShaderModule* shaderModule)
{
    if (shaderModule->vkModule) vkDestroyShaderModule(device->device, shaderModule->vkModule, nullptr);
    AgfxFree(device, shaderModule);
}

// Render pipeline (struct hoisted to the top of the file)
agfxRenderPipeline* agfxRenderPipelineCreate(agfxDevice* device, const agfxRenderPipelineCreateInfo* createInfo)
{
    if (createInfo->meshShader && !device->supportsMeshShaders) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxRenderPipelineCreate: a mesh shader was requested but the device does not support VK_EXT_mesh_shader");
        return nullptr;
    }

    agfxRenderPipeline* pipeline = AgfxAlloc<agfxRenderPipeline>(device);
    pipeline->vkPipelineCache = agfxVkCreatePipelineCache(device, createInfo->cache, createInfo->cacheSize);
    if (!pipeline->vkPipelineCache) {
        AgfxFree(device, pipeline);
        return nullptr;
    }

    // Dynamic rendering (no VkRenderPass): the create info carries its own attachment formats rather
    // than looking them up from an agfxRenderTarget/agfxRenderPass.
    VkFormat colorFormats[8] = {};
    for (uint32_t i = 0; i < createInfo->colorAttachmentCount && i < 8; ++i) {
        colorFormats[i] = agfxTextureFormatToVkFormat(createInfo->colorFormats[i]);
    }
    VkPipelineRenderingCreateInfo renderingInfo = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    renderingInfo.colorAttachmentCount = createInfo->colorAttachmentCount;
    renderingInfo.pColorAttachmentFormats = colorFormats;
    renderingInfo.depthAttachmentFormat = (createInfo->depthFormat != AGFX_TEXTURE_FORMAT_UNKNOWN)
        ? agfxTextureFormatToVkFormat(createInfo->depthFormat) : VK_FORMAT_UNDEFINED;

    VkPipelineRasterizationStateCreateInfo rasterizationState = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterizationState.depthClampEnable = createInfo->depthClampEnable;
    rasterizationState.polygonMode = agfxFillModeToVkPolygonMode(createInfo->fillMode);
    rasterizationState.cullMode = agfxCullModeToVkCullMode(createInfo->cullMode);
    rasterizationState.frontFace = agfxFrontFaceToVkFrontFace(createInfo->frontFace);
    rasterizationState.lineWidth = 1.0f;

    VkPipelineDepthStencilStateCreateInfo depthStencilState = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depthStencilState.depthTestEnable = createInfo->depthTestEnable;
    depthStencilState.depthWriteEnable = createInfo->depthWriteEnable;
    depthStencilState.depthCompareOp = agfxComparisonFunctionToVkCompareOp(createInfo->depthCompareOp);
    depthStencilState.stencilTestEnable = VK_FALSE; // AGFX has no stencil API at all.

    VkPipelineColorBlendAttachmentState blendAttachments[8] = {};
    for (uint32_t i = 0; i < createInfo->colorAttachmentCount && i < 8; ++i) {
        VkPipelineColorBlendAttachmentState& blend = blendAttachments[i];
        blend.blendEnable = createInfo->blendEnable[i];
        blend.srcColorBlendFactor = agfxBlendFactorToVkBlendFactor(createInfo->srcColorBlendFactor[i]);
        blend.dstColorBlendFactor = agfxBlendFactorToVkBlendFactor(createInfo->dstColorBlendFactor[i]);
        blend.colorBlendOp = agfxBlendOperationToVkBlendOp(createInfo->colorBlendOp[i]);
        blend.srcAlphaBlendFactor = agfxBlendFactorToVkBlendFactor(createInfo->srcAlphaBlendFactor[i]);
        blend.dstAlphaBlendFactor = agfxBlendFactorToVkBlendFactor(createInfo->dstAlphaBlendFactor[i]);
        blend.alphaBlendOp = agfxBlendOperationToVkBlendOp(createInfo->alphaBlendOp[i]);
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }
    VkPipelineColorBlendStateCreateInfo colorBlendState = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlendState.attachmentCount = createInfo->colorAttachmentCount;
    colorBlendState.pAttachments = blendAttachments;

    VkPipelineMultisampleStateCreateInfo multisampleState = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Viewport/scissor are never baked into the pipeline (same as D3D12's RSSetViewports/
    // RSSetScissorRects) -- set per-draw once agfxRenderPassSetViewport/SetScissor exist.
    VkPipelineViewportStateCreateInfo viewportState = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineShaderStageCreateInfo stages[3] = {};
    uint32_t stageCount = 0;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipelineCreateInfo.pNext = &renderingInfo;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pColorBlendState = &colorBlendState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.layout = device->globalPipelineLayout;
    pipelineCreateInfo.renderPass = VK_NULL_HANDLE;

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    VkPipelineVertexInputStateCreateInfo vertexInputState = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

    if (createInfo->meshShader) {
        // Per the VK_EXT_mesh_shader spec, vertex input / input assembly state must not be provided
        // when a mesh shader stage is present -- topology is fully determined by the shader itself.
        pipelineCreateInfo.pVertexInputState = nullptr;
        pipelineCreateInfo.pInputAssemblyState = nullptr;

        if (createInfo->taskShader) {
            stages[stageCount++] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_TASK_BIT_EXT, createInfo->taskShader->vkModule, createInfo->taskShader->entryPoint, nullptr };
        }
        stages[stageCount++] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MESH_BIT_EXT, createInfo->meshShader->vkModule, createInfo->meshShader->entryPoint, nullptr };
        if (createInfo->fragmentShader) {
            stages[stageCount++] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, createInfo->fragmentShader->vkModule, createInfo->fragmentShader->entryPoint, nullptr };
        }
    } else {
        // Bindless: no vertex buffer bindings, shaders pull vertex data themselves (mirrors D3D12's
        // InputLayout = { nullptr, 0 }; there is no agfxVertexInputLayout anywhere in agfx.h).
        pipelineCreateInfo.pVertexInputState = &vertexInputState;
        inputAssemblyState.topology = agfxTopologyToVkPrimitiveTopology(createInfo->topology);
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;

        stages[stageCount++] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, createInfo->vertexShader->vkModule, createInfo->vertexShader->entryPoint, nullptr };
        if (createInfo->fragmentShader) {
            stages[stageCount++] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, createInfo->fragmentShader->vkModule, createInfo->fragmentShader->entryPoint, nullptr };
        }
    }

    pipelineCreateInfo.stageCount = stageCount;
    pipelineCreateInfo.pStages = stages;

    if (vkCreateGraphicsPipelines(device->device, pipeline->vkPipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline->vkPipeline) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxRenderPipelineCreate: vkCreateGraphicsPipelines failed");
        vkDestroyPipelineCache(device->device, pipeline->vkPipelineCache, nullptr);
        AgfxFree(device, pipeline);
        return nullptr;
    }

    return pipeline;
}

void agfxRenderPipelineDestroy(agfxDevice* device, agfxRenderPipeline* pipeline)
{
    if (pipeline->vkPipeline) vkDestroyPipeline(device->device, pipeline->vkPipeline, nullptr);
    if (pipeline->vkPipelineCache) vkDestroyPipelineCache(device->device, pipeline->vkPipelineCache, nullptr);
    AgfxFree(device, pipeline);
}

uint8_t* agfxRenderPipelineGetCache(agfxDevice* device, agfxRenderPipeline* pipeline, uint64_t* outSize)
{
    return agfxVkGetPipelineCacheData(device, pipeline->vkPipelineCache, outSize);
}

// Compute pipeline
agfxComputePipeline* agfxComputePipelineCreate(agfxDevice* device, const agfxComputePipelineCreateInfo* createInfo)
{
    agfxComputePipeline* pipeline = AgfxAlloc<agfxComputePipeline>(device);
    pipeline->vkPipelineCache = agfxVkCreatePipelineCache(device, createInfo->cache, createInfo->cacheSize);
    if (!pipeline->vkPipelineCache) {
        AgfxFree(device, pipeline);
        return nullptr;
    }

    VkPipelineShaderStageCreateInfo stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = createInfo->computeShader->vkModule;
    stage.pName = createInfo->computeShader->entryPoint;

    VkComputePipelineCreateInfo pipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipelineCreateInfo.stage = stage;
    pipelineCreateInfo.layout = device->globalPipelineLayout;

    if (vkCreateComputePipelines(device->device, pipeline->vkPipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline->vkPipeline) != VK_SUCCESS) {
        agfxLog(device, AGFX_LOG_SEVERITY_ERROR, "agfxComputePipelineCreate: vkCreateComputePipelines failed");
        vkDestroyPipelineCache(device->device, pipeline->vkPipelineCache, nullptr);
        AgfxFree(device, pipeline);
        return nullptr;
    }

    return pipeline;
}

void agfxComputePipelineDestroy(agfxDevice* device, agfxComputePipeline* pipeline)
{
    if (pipeline->vkPipeline) vkDestroyPipeline(device->device, pipeline->vkPipeline, nullptr);
    if (pipeline->vkPipelineCache) vkDestroyPipelineCache(device->device, pipeline->vkPipelineCache, nullptr);
    AgfxFree(device, pipeline);
}

uint8_t* agfxComputePipelineGetCache(agfxDevice* device, agfxComputePipeline* pipeline, uint64_t* outSize)
{
    return agfxVkGetPipelineCacheData(device, pipeline->vkPipelineCache, outSize);
}
