/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "buffer view handle not null".
//
// A buffer view's bindless handle is the index shaders use to pull the view out of
// ResourceDescriptorHeap, so a zero handle means every buffer read in every shader silently lands on
// the wrong descriptor. Walks all three view types in both the read-only and writeable flavors, plus
// a non-zero offset, to catch a backend that only populates the heap for the default configuration.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint64_t kBufferSize = 1024;
    constexpr uint64_t kStride = 16;
    /// @brief Constant views are bound at 256-byte granularity on both backends, so an offset test
    /// that has to stay legal for every view type uses that alignment.
    constexpr uint64_t kOffset = 256;

    /// @brief The permutations every flavor of this test walks.
    struct ViewVariant
    {
        const char* name;
        agfxBufferViewType type;
        bool writeable;
        uint64_t offset;
    };

    constexpr ViewVariant kVariants[] = {
        {"raw",                   AGFX_BUFFER_VIEW_TYPE_RAW,        false, 0},
        {"raw writeable",         AGFX_BUFFER_VIEW_TYPE_RAW,        true,  0},
        {"raw at offset",         AGFX_BUFFER_VIEW_TYPE_RAW,        false, kOffset},
        {"structured",            AGFX_BUFFER_VIEW_TYPE_STRUCTURED, false, 0},
        {"structured writeable",  AGFX_BUFFER_VIEW_TYPE_STRUCTURED, true,  0},
        {"structured at offset",  AGFX_BUFFER_VIEW_TYPE_STRUCTURED, true,  kOffset},
        {"constant",              AGFX_BUFFER_VIEW_TYPE_CONSTANT,   false, 0},
        {"constant at offset",    AGFX_BUFFER_VIEW_TYPE_CONSTANT,   false, kOffset},
    };

    agfxBufferCreateInfo BufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kBufferSize;
        info.stride = kStride;
        // Every usage the variants need, so one buffer backs all of them.
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE |
                                       AGFX_BUFFER_USAGE_CONSTANT);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxBufferViewCreateInfo ViewInfo(agfxBuffer* buffer, const ViewVariant& variant)
    {
        agfxBufferViewCreateInfo info{};
        info.buffer = buffer;
        info.type = variant.type;
        info.offset = variant.offset;
        info.writeable = variant.writeable ? 1 : 0;
        return info;
    }
} // namespace

AGFX_TEST_VALIDATION(BufferViewHandleNotNull, C)
{
    const agfxDeviceCreateInfo deviceInfo = DefaultDeviceCreateInfo();
    agfxDevice* device = agfxDeviceCreate(&deviceInfo);
    AGFX_EXPECT_NOT_NULL(device);

    const agfxBufferCreateInfo bufferInfo = BufferInfo();
    agfxBuffer* buffer = agfxBufferCreate(device, &bufferInfo);
    if (!buffer) {
        agfxDeviceDestroy(device);
        AGFX_FAIL("agfxBufferCreate returned null");
    }

    for (const ViewVariant& variant : kVariants) {
        const agfxBufferViewCreateInfo viewInfo = ViewInfo(buffer, variant);
        agfxBufferView* view = agfxBufferViewCreate(device, &viewInfo);
        if (!view) {
            agfxBufferDestroy(device, buffer);
            agfxDeviceDestroy(device);
            AGFX_FAIL(std::string("agfxBufferViewCreate returned null for ") + variant.name);
        }

        const uint64_t handle = agfxBufferViewGetHandle(view);
        agfxBufferViewDestroy(device, view);
        if (handle == 0) {
            agfxBufferDestroy(device, buffer);
            agfxDeviceDestroy(device);
            AGFX_FAIL(std::string("buffer view handle was 0 for ") + variant.name);
        }
    }

    agfxBufferDestroy(device, buffer);
    agfxDeviceDestroy(device);
}

AGFX_TEST_VALIDATION(BufferViewHandleNotNull, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::Buffer buffer = device.CreateBuffer(BufferInfo());
    AGFX_EXPECT_NOT_NULL(buffer.Get());

    for (const ViewVariant& variant : kVariants) {
        agfx::BufferView view = device.CreateBufferView(ViewInfo(buffer, variant));
        AGFX_EXPECT_MSG(view.Get() != nullptr, variant.name);
        AGFX_EXPECT_MSG(view.GetHandle() != 0, variant.name);
    }
}

AGFX_TEST_VALIDATION(BufferViewHandleNotNull, Ez)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless
    contextInfo.width = 128;
    contextInfo.height = 128;

    agfx::ez::Context context(contextInfo);

    const std::vector<uint8_t> zeros((size_t)kBufferSize, 0u);
    agfx::ez::Buffer buffer =
        context.CreateStructuredBuffer(zeros.data(), kBufferSize, kStride, /*shaderWritable*/ true);

    // The ez path: Buffer::View() lazily creates and caches one view per (type, writeable) pair.
    // Every one of them has to come back with a live handle, and asking twice must not mint a
    // second, different descriptor.
    for (const ViewVariant& variant : kVariants) {
        if (variant.offset != 0) {
            continue; // ez's cached views are always offset zero; the offset cases are C/C++ only.
        }

        agfx::BufferView& view = buffer.View(variant.type, variant.writeable);
        AGFX_EXPECT_MSG(view.Get() != nullptr, variant.name);
        AGFX_EXPECT_MSG(view.GetHandle() != 0, variant.name);

        agfx::BufferView& again = buffer.View(variant.type, variant.writeable);
        AGFX_EXPECT_MSG(again.GetHandle() == view.GetHandle(), variant.name);

        // And it must survive the narrowing ShaderBindings applies when packing it into push
        // constants — that cast is what shaders actually receive.
        agfx::ez::ShaderBindings bindings;
        bindings.BindBuffer(view);
        AGFX_EXPECT_MSG(*(const uint32_t*)bindings.Data() != 0u, variant.name);
    }
}
