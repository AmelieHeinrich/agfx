/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "copy buffer to buffer", "copy buffer to buffer offset" and "upload readback
// roundtrip".
//
// No shaders involved: a host pattern goes up through an upload buffer, is copied buffer-to-buffer
// twice — once at offset zero, once with both a source and a destination offset — and comes back
// down through a readback buffer. The two copies write disjoint halves of the destination, so a
// backend that ignores srcOffset or dstOffset produces a destination whose halves are duplicates or
// misaligned rather than the interleaving the golden records.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kSourceWords = 64;
    constexpr uint32_t kDestWords = 64;
    constexpr uint64_t kSourceSize = kSourceWords * sizeof(uint32_t);
    constexpr uint64_t kDestSize = kDestWords * sizeof(uint32_t);

    // Copy 1: the first half of the source onto the first half of the destination, no offsets.
    constexpr uint64_t kHalfSize = kDestSize / 2;
    // Copy 2: a window starting mid-source onto the destination's second half. Both offsets are
    // non-zero and different from each other, so swapping them is detectable.
    constexpr uint64_t kOffsetSrc = 16 * sizeof(uint32_t);
    constexpr uint64_t kOffsetDst = kHalfSize;

    constexpr const char* kGolden = "copy_buffer_to_buffer.bin";

    /// @brief The host pattern. Each word encodes its own index so a shifted copy is obvious.
    std::vector<uint32_t> SourcePattern()
    {
        std::vector<uint32_t> data(kSourceWords);
        for (uint32_t i = 0; i < kSourceWords; ++i) {
            data[i] = 0xC0DE0000u | (i * 7u + 1u);
        }
        return data;
    }

    agfxBufferCreateInfo SourceInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kSourceSize;
        info.stride = sizeof(uint32_t);
        info.usage = AGFX_BUFFER_USAGE_SHADER_READ;
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_UPLOAD;
        return info;
    }

    agfxBufferCreateInfo DestInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kDestSize;
        info.stride = sizeof(uint32_t);
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }
} // namespace

AGFX_TEST_BUFFER(CopyBufferToBuffer, C)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const agfxBufferCreateInfo sourceInfo = SourceInfo();
    const agfxBufferCreateInfo destInfo = DestInfo();
    agfxBuffer* source = agfxBufferCreate(device, &sourceInfo);
    agfxBuffer* dest = agfxBufferCreate(device, &destInfo);
    AGFX_EXPECT_NOT_NULL(source);
    AGFX_EXPECT_NOT_NULL(dest);

    agfxDeviceMakeResourcesResident(device);

    // The upload half of the roundtrip: write the pattern straight into the mapped upload buffer.
    const std::vector<uint32_t> pattern = SourcePattern();
    void* mapped = agfxBufferMap(source);
    if (!mapped) {
        agfxBufferDestroy(device, dest);
        agfxBufferDestroy(device, source);
        AGFX_FAIL("failed to map the upload buffer");
    }
    memcpy(mapped, pattern.data(), (size_t)kSourceSize);
    agfxBufferUnmap(source);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferBufferBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_COPY_DEST, 0);

        agfxComputePass* pass = agfxComputePassBegin(cmd, "copy buffer to buffer");
        agfxComputePassCopyBufferToBuffer(pass, source, dest, 0, 0, kHalfSize);
        agfxComputePassCopyBufferToBuffer(pass, source, dest, kOffsetSrc, kOffsetDst, kHalfSize);
        agfxComputePassEnd(pass);
    });

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device, gpu.Queue(), dest, kDestSize,
                                       AGFX_RESOURCE_STATE_COPY_DEST, bytes);

    agfxBufferDestroy(device, dest);
    agfxBufferDestroy(device, source);

    AGFX_EXPECT_MSG(readOk, "buffer readback failed");
    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(CopyBufferToBuffer, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    agfx::Buffer source = device.CreateBuffer(SourceInfo());
    agfx::Buffer dest = device.CreateBuffer(DestInfo());
    AGFX_EXPECT_NOT_NULL(source.Get());
    AGFX_EXPECT_NOT_NULL(dest.Get());

    device.MakeResourcesResident();

    const std::vector<uint32_t> pattern = SourcePattern();
    {
        // MappedBuffer unmaps on scope exit, which is the whole point of the C++ wrapper here.
        agfx::MappedBuffer mapping(source);
        AGFX_EXPECT_NOT_NULL(mapping.Get());
        memcpy(mapping.Get(), pattern.data(), (size_t)kSourceSize);
    }

    cmd.Begin();
    cmd.BufferBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_DEST, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("copy buffer to buffer");
        pass.CopyBufferToBuffer(source, dest, 0, 0, kHalfSize);
        pass.CopyBufferToBuffer(source, dest, kOffsetSrc, kOffsetDst, kHalfSize);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), queue, dest, kDestSize,
                                       AGFX_RESOURCE_STATE_COPY_DEST, bytes);
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");

    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(CopyBufferToBuffer, Ez)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = 128;
    contextInfo.height = 128;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    // ez's CreateStructuredBuffer runs the upload through the Context's own uploader, which is the
    // upload half of the roundtrip as an ez user would write it.
    const std::vector<uint32_t> pattern = SourcePattern();
    agfx::ez::Buffer source = context.CreateStructuredBuffer(pattern.data(), kSourceSize, sizeof(uint32_t));

    const std::vector<uint32_t> zeros(kDestWords, 0u);
    agfx::ez::Buffer dest =
        context.CreateStructuredBuffer(zeros.data(), kDestSize, sizeof(uint32_t), /*shaderWritable*/ true);

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionBuffer(source, AGFX_RESOURCE_STATE_COPY_SOURCE);
        context.TransitionBuffer(dest, AGFX_RESOURCE_STATE_COPY_DEST);

        // ez has no copy sugar of its own; drop to the frame's raw command buffer.
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("copy buffer to buffer");
        pass.CopyBufferToBuffer(source.Raw(), dest.Raw(), 0, 0, kHalfSize);
        pass.CopyBufferToBuffer(source.Raw(), dest.Raw(), kOffsetSrc, kOffsetDst, kHalfSize);
    }
    context.DrainGPU();

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), context.GetGraphicsQueue(), dest.Raw(), kDestSize,
                                       dest.State(), bytes);
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");

    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}
