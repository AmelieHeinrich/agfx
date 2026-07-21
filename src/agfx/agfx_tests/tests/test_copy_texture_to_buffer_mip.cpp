/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "copy texture to buffer mip".
//
// CopyTextureToBuffer already covers mip 0; this pins down the mipLevel argument itself. Both mips of
// a 128x128 texture are seeded with different patterns, then mip 1 is copied out. A backend that
// ignores mipLevel and copies mip 0 produces the other pattern at the wrong extent.
//
// The golden alone cannot catch that — it is generated from whatever the backend produced — so the
// copied bytes are also compared against the CPU-computed mip 1 pattern directly. The golden is the
// regression net; the in-test comparison is the oracle. Every test in this mip/slice family does the
// same.

#include "../test_gpu.h"

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kBaseWidth = 128;
    constexpr uint32_t kBaseHeight = 128;
    constexpr uint32_t kMipLevels = 2;
    constexpr uint32_t kCopyMip = 1;

    // Mip 1 of a 128-wide RGBA8 texture is 64 texels => 256 bytes per row, which is exactly D3D12's
    // required row-pitch alignment for buffer/texture copies. Mip 2 would be 128 bytes and would pass
    // on Metal while failing on Windows, so this test deliberately stops at mip 1.
    constexpr uint32_t kMipWidth = kBaseWidth >> kCopyMip;
    constexpr uint32_t kMipHeight = kBaseHeight >> kCopyMip;

    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr uint32_t kBytesPerPixel = 4;
    constexpr uint32_t kMipRowBytes = kMipWidth * kBytesPerPixel;
    constexpr uint64_t kMipBytes = uint64_t(kMipRowBytes) * kMipHeight;

    constexpr const char* kGolden = "copy_texture_to_buffer_mip.bin";

    /// @brief A per-mip pattern. `mip` is folded into every channel, so mip 0's and mip 1's contents
    /// have no texel in common — reading the wrong level is never mistakable for the right one.
    std::vector<uint8_t> MipPixels(uint32_t mip, uint32_t width, uint32_t height)
    {
        std::vector<uint8_t> pixels((size_t)width * height * kBytesPerPixel);
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint8_t* texel = &pixels[((size_t)y * width + x) * kBytesPerPixel];
                texel[0] = (uint8_t)(x * 3u + mip * 97u);
                texel[1] = (uint8_t)(y * 3u + mip * 53u);
                texel[2] = (uint8_t)((x ^ y) + mip * 131u); // XOR: survives no row/column shift.
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    agfxTextureCreateInfo SourceInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kFormat;
        info.usage = AGFX_TEXTURE_USAGE_SAMPLED;
        info.width = kBaseWidth;
        info.height = kBaseHeight;
        info.depthOrArrayLayers = 1;
        info.mipLevels = kMipLevels;
        return info;
    }

    agfxBufferCreateInfo DestInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kMipBytes;
        info.stride = kBytesPerPixel;
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxTextureRegion MipRegion()
    {
        agfxTextureRegion region{};
        region.width = kMipWidth;
        region.height = kMipHeight;
        region.depth = 1;
        return region;
    }
} // namespace

AGFX_TEST_BUFFER(CopyTextureToBufferMip, C)
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

    const std::vector<uint8_t> mip0 = MipPixels(0, kBaseWidth, kBaseHeight);
    const std::vector<uint8_t> mip1 = MipPixels(1, kMipWidth, kMipHeight);
    const bool seeded0 = UploadTextureSubresource(device, gpu.Queue(), source, kBaseWidth, kBaseHeight,
                                                  kFormat, mip0.data(), AGFX_RESOURCE_STATE_COMMON, 0, 0);
    const bool seeded1 = UploadTextureSubresource(device, gpu.Queue(), source, kMipWidth, kMipHeight,
                                                  kFormat, mip1.data(), AGFX_RESOURCE_STATE_COMMON,
                                                  kCopyMip, 0);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_COPY_SOURCE, kCopyMip, 0, 0);
        agfxCommandBufferBufferBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_COPY_DEST, 0);

        const agfxTextureRegion region = MipRegion();
        agfxComputePass* pass = agfxComputePassBegin(cmd, "copy texture mip to buffer");
        agfxComputePassCopyTextureToBuffer(pass, source, dest, 0, &region, kCopyMip, 0, kMipRowBytes,
                                           (uint32_t)kMipBytes);
        agfxComputePassEnd(pass);
    });

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device, gpu.Queue(), dest, kMipBytes,
                                       AGFX_RESOURCE_STATE_COPY_DEST, bytes);

    agfxBufferDestroy(device, dest);
    agfxTextureDestroy(device, source);

    AGFX_EXPECT_MSG(seeded0, "failed to seed mip 0");
    AGFX_EXPECT_MSG(seeded1, "failed to seed mip 1");
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");
    AGFX_EXPECT_MSG(bytes == mip1, "copied bytes are not mip 1's pattern");
    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(CopyTextureToBufferMip, Cpp)
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

    const std::vector<uint8_t> mip0 = MipPixels(0, kBaseWidth, kBaseHeight);
    const std::vector<uint8_t> mip1 = MipPixels(1, kMipWidth, kMipHeight);
    AGFX_EXPECT_MSG(UploadTextureSubresource(device.Get(), queue, source, kBaseWidth, kBaseHeight,
                                             kFormat, mip0.data(), AGFX_RESOURCE_STATE_COMMON, 0, 0),
                    "failed to seed mip 0");
    AGFX_EXPECT_MSG(UploadTextureSubresource(device.Get(), queue, source, kMipWidth, kMipHeight, kFormat,
                                             mip1.data(), AGFX_RESOURCE_STATE_COMMON, kCopyMip, 0),
                    "failed to seed mip 1");

    cmd.Begin();
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_SOURCE, kCopyMip,
                       AGFX_SUBRESOURCE_ALL_LAYERS, false);
    cmd.BufferBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_DEST, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("copy texture mip to buffer");
        pass.CopyTextureToBuffer(source, dest, 0, MipRegion(), kCopyMip, 0, kMipRowBytes,
                                 (uint32_t)kMipBytes);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), queue, dest, kMipBytes,
                                       AGFX_RESOURCE_STATE_COPY_DEST, bytes);
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");
    AGFX_EXPECT_MSG(bytes == mip1, "copied bytes are not mip 1's pattern");

    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}
