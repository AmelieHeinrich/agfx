/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "copy texture to buffer".
//
// The opposite direction from CopyBufferToTexture, and checked as raw bytes rather than as an image
// so the row pitch is actually under test: a known pattern is uploaded into a texture, copied back
// into a GPU buffer, and memcmp'd against the golden. A backend that writes rows at the texture's
// native pitch instead of the requested bytesPerRow, or that drops the buffer offset, produces
// shifted bytes that an image comparison with a tolerance might have waved through.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 64;
    constexpr uint32_t kHeight = 64;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr uint32_t kBytesPerPixel = 4;
    constexpr uint32_t kRowBytes = kWidth * kBytesPerPixel; // 256: the D3D12 row-pitch alignment.
    constexpr uint64_t kImageBytes = uint64_t(kRowBytes) * kHeight;

    // The destination buffer is copied into at a non-zero offset, with the leading bytes left at a
    // known filler value so a copy that ignores bufferOffset overwrites them and is caught.
    constexpr uint64_t kDestOffset = 256;
    constexpr uint64_t kDestSize = kDestOffset + kImageBytes;
    constexpr uint8_t kFiller = 0xAB;

    constexpr const char* kGolden = "copy_texture_to_buffer.bin";

    std::vector<uint8_t> SourcePixels()
    {
        std::vector<uint8_t> pixels((size_t)kImageBytes);
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                texel[0] = (uint8_t)(x * 4u);
                texel[1] = (uint8_t)(y * 4u);
                texel[2] = (uint8_t)((x ^ y) * 4u); // XOR: not symmetric under a row/column shift.
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    /// @brief The destination's initial contents: filler everywhere, so any byte the copy touches
    /// outside its intended window shows up.
    std::vector<uint8_t> FillerBytes()
    {
        return std::vector<uint8_t>((size_t)kDestSize, kFiller);
    }

    agfxTextureCreateInfo SourceInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kFormat;
        info.usage = AGFX_TEXTURE_USAGE_SAMPLED;
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    agfxBufferCreateInfo DestInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kDestSize;
        info.stride = kBytesPerPixel;
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxTextureRegion FullRegion()
    {
        agfxTextureRegion region{};
        region.width = kWidth;
        region.height = kHeight;
        region.depth = 1;
        return region;
    }
} // namespace

AGFX_TEST_BUFFER(CopyTextureToBuffer, C)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const agfxTextureCreateInfo sourceInfo = SourceInfo();
    const agfxBufferCreateInfo destInfo = DestInfo();
    agfxTexture* source = agfxTextureCreate(device, &sourceInfo);
    agfxBuffer* dest = agfxBufferCreate(device, &destInfo);
    AGFX_EXPECT_NOT_NULL(source);
    AGFX_EXPECT_NOT_NULL(dest);

    agfxDeviceMakeResourcesResident(device);

    const std::vector<uint8_t> pixels = SourcePixels();
    const std::vector<uint8_t> filler = FillerBytes();
    const bool seededTexture = UploadTexture2D(device, gpu.Queue(), source, kWidth, kHeight, kFormat,
                                               pixels.data(), AGFX_RESOURCE_STATE_COMMON);
    const bool seededBuffer = UploadBuffer(device, gpu.Queue(), dest, filler.data(), kDestSize,
                                           AGFX_RESOURCE_STATE_COMMON);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_COPY_SOURCE, 0, 0, 0);
        agfxCommandBufferBufferBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_COPY_DEST, 0);

        const agfxTextureRegion region = FullRegion();
        agfxComputePass* pass = agfxComputePassBegin(cmd, "copy texture to buffer");
        agfxComputePassCopyTextureToBuffer(pass, source, dest, kDestOffset, &region, 0, 0, kRowBytes,
                                           (uint32_t)kImageBytes);
        agfxComputePassEnd(pass);
    });

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device, gpu.Queue(), dest, kDestSize,
                                       AGFX_RESOURCE_STATE_COPY_DEST, bytes);

    agfxBufferDestroy(device, dest);
    agfxTextureDestroy(device, source);

    AGFX_EXPECT_MSG(seededTexture, "failed to seed the source texture");
    AGFX_EXPECT_MSG(seededBuffer, "failed to seed the destination buffer");
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");
    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(CopyTextureToBuffer, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    agfx::Texture source = device.CreateTexture(SourceInfo());
    agfx::Buffer dest = device.CreateBuffer(DestInfo());
    AGFX_EXPECT_NOT_NULL(source.Get());
    AGFX_EXPECT_NOT_NULL(dest.Get());

    device.MakeResourcesResident();

    const std::vector<uint8_t> pixels = SourcePixels();
    const std::vector<uint8_t> filler = FillerBytes();
    AGFX_EXPECT_MSG(UploadTexture2D(device.Get(), queue, source, kWidth, kHeight, kFormat, pixels.data(),
                                    AGFX_RESOURCE_STATE_COMMON),
                    "failed to seed the source texture");
    AGFX_EXPECT_MSG(UploadBuffer(device.Get(), queue, dest, filler.data(), kDestSize,
                                 AGFX_RESOURCE_STATE_COMMON),
                    "failed to seed the destination buffer");

    cmd.Begin();
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_SOURCE,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    cmd.BufferBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_DEST, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("copy texture to buffer");
        pass.CopyTextureToBuffer(source, dest, kDestOffset, FullRegion(), 0, 0, kRowBytes,
                                 (uint32_t)kImageBytes);
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

AGFX_TEST_BUFFER(CopyTextureToBuffer, Ez)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    const std::vector<uint8_t> pixels = SourcePixels();
    agfx::ez::Texture2D source =
        context.CreateTexture2D(kWidth, kHeight, kFormat, AGFX_TEXTURE_USAGE_SAMPLED, pixels.data(), kRowBytes);

    const std::vector<uint8_t> filler = FillerBytes();
    agfx::ez::Buffer dest =
        context.CreateStructuredBuffer(filler.data(), kDestSize, kBytesPerPixel, /*shaderWritable*/ true);

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(source, AGFX_RESOURCE_STATE_COPY_SOURCE);
        context.TransitionBuffer(dest, AGFX_RESOURCE_STATE_COPY_DEST);

        // ez has no texture-to-buffer sugar; drop to the frame's raw command buffer.
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("copy texture to buffer");
        pass.CopyTextureToBuffer(source.Raw(), dest.Raw(), kDestOffset, FullRegion(), 0, 0, kRowBytes,
                                 (uint32_t)kImageBytes);
    }
    context.DrainGPU();

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), context.GetGraphicsQueue(), dest.Raw(), kDestSize,
                                       dest.State(), bytes);
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");

    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}
