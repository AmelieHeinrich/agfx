/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "copy buffer to texture slice".
//
// The array-layer twin of CopyBufferToTextureMip: a buffer is copied into layer 2 of a four-layer 2D
// array whose layers were all pre-seeded. Layer 2 must take the buffer's contents and every other
// layer must be untouched — the second half is what actually tests the layer argument, and it is
// checked against all three neighbours rather than just one, so an off-by-one in either direction is
// caught.

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

    constexpr const char* kGolden = "copy_buffer_to_texture_slice.png";

    /// @brief The per-layer seed. Distinct per layer, so "layer N was disturbed" is diagnosable.
    std::vector<uint8_t> SeedPixels(uint32_t layer)
    {
        std::vector<uint8_t> pixels((size_t)kLayerBytes);
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                texel[0] = (uint8_t)(layer * 60u);
                texel[1] = (uint8_t)(y * 3u);
                texel[2] = 180u;
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    /// @brief The buffer's contents, and layer 2's expected result.
    std::vector<uint8_t> SourcePixels()
    {
        std::vector<uint8_t> pixels((size_t)kLayerBytes);
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                texel[0] = (uint8_t)(x * 4u);
                texel[1] = 25u;
                texel[2] = (uint8_t)((x ^ y) * 4u);
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    agfxTextureCreateInfo DestInfo()
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

    agfxBufferCreateInfo SourceInfo()
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

AGFX_TEST_TEXTURE(CopyBufferToTextureSlice, C, kWidth, kHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const agfxTextureCreateInfo destInfo = DestInfo();
    const agfxBufferCreateInfo sourceInfo = SourceInfo();
    agfxTexture* dest = agfxTextureCreate(device, &destInfo);
    agfxBuffer* source = agfxBufferCreate(device, &sourceInfo);
    AGFX_EXPECT_NOT_NULL(dest);
    AGFX_EXPECT_NOT_NULL(source);

    agfxDeviceMakeResourcesResident(device);

    const std::vector<uint8_t> sourcePixels = SourcePixels();
    bool seeded = true;
    for (uint32_t layer = 0; layer < kLayerCount; ++layer) {
        const std::vector<uint8_t> pixels = SeedPixels(layer);
        seeded &= UploadTextureSubresource(device, gpu.Queue(), dest, kWidth, kHeight, kFormat,
                                           pixels.data(), AGFX_RESOURCE_STATE_COMMON, 0, layer);
    }
    seeded &= UploadBuffer(device, gpu.Queue(), source, sourcePixels.data(), kLayerBytes,
                           AGFX_RESOURCE_STATE_COMMON);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferBufferBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_COPY_SOURCE, 0);
        agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_COPY_DEST, 0, kCopyLayer, 0);

        const agfxTextureRegion region = LayerRegion();
        agfxComputePass* pass = agfxComputePassBegin(cmd, "copy buffer to texture slice");
        agfxComputePassCopyBufferToTexture(pass, source, dest, &region, 0, kCopyLayer, kRowBytes,
                                           (uint32_t)kLayerBytes);
        agfxComputePassEnd(pass);
    });

    Image target;
    const bool readOk = ReadbackTextureSubresource(device, gpu.Queue(), dest, kWidth, kHeight, kFormat,
                                                   AGFX_RESOURCE_STATE_COPY_DEST, 0, kCopyLayer, target);

    bool neighboursIntact = true;
    for (uint32_t layer = 0; layer < kLayerCount; ++layer) {
        if (layer == kCopyLayer) {
            continue;
        }
        Image neighbour;
        neighboursIntact &= ReadbackTextureSubresource(device, gpu.Queue(), dest, kWidth, kHeight,
                                                       kFormat, AGFX_RESOURCE_STATE_COMMON, 0, layer,
                                                       neighbour);
        neighboursIntact &= ImageEqualsRgba8(neighbour, SeedPixels(layer));
    }

    agfxBufferDestroy(device, source);
    agfxTextureDestroy(device, dest);

    AGFX_EXPECT_MSG(seeded, "failed to seed the array layers or the source buffer");
    AGFX_EXPECT_MSG(readOk, "layer readback failed");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(target, sourcePixels),
                    "layer 2 does not hold the buffer's contents");
    AGFX_EXPECT_MSG(neighboursIntact, "the copy disturbed a layer other than layer 2");
    ExpectImageMatchesGolden(ctx, kGolden, target);
}

AGFX_TEST_TEXTURE(CopyBufferToTextureSlice, Cpp, kWidth, kHeight)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    agfx::Texture dest = device.CreateTexture(DestInfo());
    agfx::Buffer source = device.CreateBuffer(SourceInfo());
    AGFX_EXPECT_NOT_NULL(dest.Get());
    AGFX_EXPECT_NOT_NULL(source.Get());

    device.MakeResourcesResident();

    const std::vector<uint8_t> sourcePixels = SourcePixels();
    for (uint32_t layer = 0; layer < kLayerCount; ++layer) {
        const std::vector<uint8_t> pixels = SeedPixels(layer);
        AGFX_EXPECT_MSG(UploadTextureSubresource(device.Get(), queue, dest, kWidth, kHeight, kFormat,
                                                 pixels.data(), AGFX_RESOURCE_STATE_COMMON, 0, layer),
                        "failed to seed an array layer");
    }
    AGFX_EXPECT_MSG(UploadBuffer(device.Get(), queue, source, sourcePixels.data(), kLayerBytes,
                                 AGFX_RESOURCE_STATE_COMMON),
                    "failed to seed the source buffer");

    cmd.Begin();
    cmd.BufferBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_SOURCE, false);
    cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_DEST,
                       AGFX_SUBRESOURCE_ALL_MIPS, kCopyLayer, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("copy buffer to texture slice");
        pass.CopyBufferToTexture(source, dest, LayerRegion(), 0, kCopyLayer, kRowBytes,
                                 (uint32_t)kLayerBytes);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image target;
    AGFX_EXPECT_MSG(ReadbackTextureSubresource(device.Get(), queue, dest, kWidth, kHeight, kFormat,
                                               AGFX_RESOURCE_STATE_COPY_DEST, 0, kCopyLayer, target),
                    "layer readback failed");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(target, sourcePixels),
                    "layer 2 does not hold the buffer's contents");

    for (uint32_t layer = 0; layer < kLayerCount; ++layer) {
        if (layer == kCopyLayer) {
            continue;
        }
        Image neighbour;
        AGFX_EXPECT_MSG(ReadbackTextureSubresource(device.Get(), queue, dest, kWidth, kHeight, kFormat,
                                                   AGFX_RESOURCE_STATE_COMMON, 0, layer, neighbour),
                        "neighbour layer readback failed");
        AGFX_EXPECT_MSG(ImageEqualsRgba8(neighbour, SeedPixels(layer)),
                        "the copy disturbed a layer other than layer 2");
    }

    ExpectImageMatchesGolden(ctx, kGolden, target);
}
