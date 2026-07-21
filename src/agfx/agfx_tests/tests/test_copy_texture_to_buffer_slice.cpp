/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "copy texture to buffer slice".
//
// The array-layer twin of CopyTextureToBufferMip. All four layers of a 2D array texture get distinct
// patterns, then layer 2 is copied out. Layer 2 is neither the first nor the last, so a backend that
// clamps the layer argument, ignores it, or walks off the end of the array lands on a pattern the
// test can name. As in the rest of this family the copied bytes are checked against the expected
// layer directly, not only against the golden.

#include "../test_gpu.h"

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 64; // 64 * 4 bytes = 256: D3D12's row-pitch alignment.
    constexpr uint32_t kHeight = 64;
    constexpr uint32_t kLayerCount = 4;
    constexpr uint32_t kCopyLayer = 2;

    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr uint32_t kBytesPerPixel = 4;
    constexpr uint32_t kRowBytes = kWidth * kBytesPerPixel;
    constexpr uint64_t kLayerBytes = uint64_t(kRowBytes) * kHeight;

    constexpr const char* kGolden = "copy_texture_to_buffer_slice.bin";

    /// @brief A per-layer pattern, distinct in every channel across layers.
    std::vector<uint8_t> LayerPixels(uint32_t layer)
    {
        std::vector<uint8_t> pixels((size_t)kLayerBytes);
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                texel[0] = (uint8_t)(x * 4u + layer * 61u);
                texel[1] = (uint8_t)(y * 4u + layer * 29u);
                texel[2] = (uint8_t)((x ^ y) * 4u + layer * 113u);
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    agfxTextureCreateInfo SourceInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D_ARRAY;
        info.format = kFormat;
        info.usage = AGFX_TEXTURE_USAGE_SAMPLED;
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = kLayerCount;
        info.mipLevels = 1;
        return info;
    }

    agfxBufferCreateInfo DestInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kLayerBytes;
        info.stride = kBytesPerPixel;
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxTextureRegion LayerRegion()
    {
        agfxTextureRegion region{};
        region.width = kWidth;
        region.height = kHeight;
        region.depth = 1;
        return region;
    }
} // namespace

AGFX_TEST_BUFFER(CopyTextureToBufferSlice, C)
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

    bool seeded = true;
    for (uint32_t layer = 0; layer < kLayerCount; ++layer) {
        const std::vector<uint8_t> pixels = LayerPixels(layer);
        seeded &= UploadTextureSubresource(device, gpu.Queue(), source, kWidth, kHeight, kFormat,
                                           pixels.data(), AGFX_RESOURCE_STATE_COMMON, 0, layer);
    }

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_COPY_SOURCE, 0, kCopyLayer, 0);
        agfxCommandBufferBufferBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_COPY_DEST, 0);

        const agfxTextureRegion region = LayerRegion();
        agfxComputePass* pass = agfxComputePassBegin(cmd, "copy texture slice to buffer");
        agfxComputePassCopyTextureToBuffer(pass, source, dest, 0, &region, 0, kCopyLayer, kRowBytes,
                                           (uint32_t)kLayerBytes);
        agfxComputePassEnd(pass);
    });

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device, gpu.Queue(), dest, kLayerBytes,
                                       AGFX_RESOURCE_STATE_COPY_DEST, bytes);

    agfxBufferDestroy(device, dest);
    agfxTextureDestroy(device, source);

    AGFX_EXPECT_MSG(seeded, "failed to seed the array layers");
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");
    AGFX_EXPECT_MSG(bytes == LayerPixels(kCopyLayer), "copied bytes are not layer 2's pattern");
    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(CopyTextureToBufferSlice, Cpp)
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

    for (uint32_t layer = 0; layer < kLayerCount; ++layer) {
        const std::vector<uint8_t> pixels = LayerPixels(layer);
        AGFX_EXPECT_MSG(UploadTextureSubresource(device.Get(), queue, source, kWidth, kHeight, kFormat,
                                                 pixels.data(), AGFX_RESOURCE_STATE_COMMON, 0, layer),
                        "failed to seed an array layer");
    }

    cmd.Begin();
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_SOURCE,
                       AGFX_SUBRESOURCE_ALL_MIPS, kCopyLayer, false);
    cmd.BufferBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_DEST, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("copy texture slice to buffer");
        pass.CopyTextureToBuffer(source, dest, 0, LayerRegion(), 0, kCopyLayer, kRowBytes,
                                 (uint32_t)kLayerBytes);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), queue, dest, kLayerBytes,
                                       AGFX_RESOURCE_STATE_COPY_DEST, bytes);
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");
    AGFX_EXPECT_MSG(bytes == LayerPixels(kCopyLayer), "copied bytes are not layer 2's pattern");

    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}
