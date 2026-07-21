/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "copy buffer to texture mip".
//
// The write direction: a buffer is copied into mip 1 of a two-mip texture whose mips were both
// pre-seeded. Two things have to hold, and a single golden of the destination can only show the
// first — so both are asserted here. Mip 1 must end up holding the buffer's contents, and mip 0 must
// be left exactly as seeded. A backend that ignores mipLevel fails the second check loudly rather
// than quietly producing a plausible-looking mip 1.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kBaseWidth = 128;
    constexpr uint32_t kBaseHeight = 128;
    constexpr uint32_t kMipLevels = 2;
    constexpr uint32_t kCopyMip = 1;

    // Mip 1 is 64 texels wide => 256 bytes per row, D3D12's required copy row-pitch alignment.
    constexpr uint32_t kMipWidth = kBaseWidth >> kCopyMip;
    constexpr uint32_t kMipHeight = kBaseHeight >> kCopyMip;

    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr uint32_t kBytesPerPixel = 4;
    constexpr uint32_t kMipRowBytes = kMipWidth * kBytesPerPixel;
    constexpr uint64_t kMipBytes = uint64_t(kMipRowBytes) * kMipHeight;

    constexpr const char* kGolden = "copy_buffer_to_texture_mip.png";

    /// @brief What mip 0 is seeded with and must still hold afterwards.
    std::vector<uint8_t> SeedPixels(uint32_t width, uint32_t height)
    {
        std::vector<uint8_t> pixels((size_t)width * height * kBytesPerPixel);
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint8_t* texel = &pixels[((size_t)y * width + x) * kBytesPerPixel];
                texel[0] = 20u;
                texel[1] = (uint8_t)(y * 2u);
                texel[2] = 200u;
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    /// @brief The buffer's contents, and therefore mip 1's expected result. Shares no channel
    /// behaviour with the seed, so the two are never confusable.
    std::vector<uint8_t> SourcePixels()
    {
        std::vector<uint8_t> pixels((size_t)kMipBytes);
        for (uint32_t y = 0; y < kMipHeight; ++y) {
            for (uint32_t x = 0; x < kMipWidth; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kMipRowBytes + (size_t)x * kBytesPerPixel];
                texel[0] = (uint8_t)(x * 4u);
                texel[1] = 30u;
                texel[2] = (uint8_t)((x ^ y) * 4u);
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    agfxTextureCreateInfo DestInfo()
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

    agfxBufferCreateInfo SourceInfo()
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

AGFX_TEST_TEXTURE(CopyBufferToTextureMip, C, kMipWidth, kMipHeight)
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

    const std::vector<uint8_t> seed0 = SeedPixels(kBaseWidth, kBaseHeight);
    const std::vector<uint8_t> seed1 = SeedPixels(kMipWidth, kMipHeight);
    const std::vector<uint8_t> sourcePixels = SourcePixels();
    bool seeded = UploadTextureSubresource(device, gpu.Queue(), dest, kBaseWidth, kBaseHeight, kFormat,
                                           seed0.data(), AGFX_RESOURCE_STATE_COMMON, 0, 0);
    seeded &= UploadTextureSubresource(device, gpu.Queue(), dest, kMipWidth, kMipHeight, kFormat,
                                       seed1.data(), AGFX_RESOURCE_STATE_COMMON, kCopyMip, 0);
    seeded &= UploadBuffer(device, gpu.Queue(), source, sourcePixels.data(), kMipBytes,
                           AGFX_RESOURCE_STATE_COMMON);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferBufferBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_COPY_SOURCE, 0);
        agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_COPY_DEST, kCopyMip, 0, 0);

        const agfxTextureRegion region = MipRegion();
        agfxComputePass* pass = agfxComputePassBegin(cmd, "copy buffer to texture mip");
        agfxComputePassCopyBufferToTexture(pass, source, dest, &region, kCopyMip, 0, kMipRowBytes,
                                           (uint32_t)kMipBytes);
        agfxComputePassEnd(pass);
    });

    Image mip1;
    Image mip0;
    const bool readMip1 = ReadbackTextureSubresource(device, gpu.Queue(), dest, kMipWidth, kMipHeight,
                                                     kFormat, AGFX_RESOURCE_STATE_COPY_DEST, kCopyMip,
                                                     0, mip1);
    const bool readMip0 = ReadbackTextureSubresource(device, gpu.Queue(), dest, kBaseWidth, kBaseHeight,
                                                     kFormat, AGFX_RESOURCE_STATE_COMMON, 0, 0, mip0);

    agfxBufferDestroy(device, source);
    agfxTextureDestroy(device, dest);

    AGFX_EXPECT_MSG(seeded, "failed to seed the texture mips or the source buffer");
    AGFX_EXPECT_MSG(readMip1, "mip 1 readback failed");
    AGFX_EXPECT_MSG(readMip0, "mip 0 readback failed");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(mip1, sourcePixels), "mip 1 does not hold the buffer's contents");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(mip0, seed0), "the copy disturbed mip 0");
    ExpectImageMatchesGolden(ctx, kGolden, mip1);
}

AGFX_TEST_TEXTURE(CopyBufferToTextureMip, Cpp, kMipWidth, kMipHeight)
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

    const std::vector<uint8_t> seed0 = SeedPixels(kBaseWidth, kBaseHeight);
    const std::vector<uint8_t> seed1 = SeedPixels(kMipWidth, kMipHeight);
    const std::vector<uint8_t> sourcePixels = SourcePixels();
    AGFX_EXPECT_MSG(UploadTextureSubresource(device.Get(), queue, dest, kBaseWidth, kBaseHeight, kFormat,
                                             seed0.data(), AGFX_RESOURCE_STATE_COMMON, 0, 0),
                    "failed to seed mip 0");
    AGFX_EXPECT_MSG(UploadTextureSubresource(device.Get(), queue, dest, kMipWidth, kMipHeight, kFormat,
                                             seed1.data(), AGFX_RESOURCE_STATE_COMMON, kCopyMip, 0),
                    "failed to seed mip 1");
    AGFX_EXPECT_MSG(UploadBuffer(device.Get(), queue, source, sourcePixels.data(), kMipBytes,
                                 AGFX_RESOURCE_STATE_COMMON),
                    "failed to seed the source buffer");

    cmd.Begin();
    cmd.BufferBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_SOURCE, false);
    cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_DEST, kCopyMip,
                       AGFX_SUBRESOURCE_ALL_LAYERS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("copy buffer to texture mip");
        pass.CopyBufferToTexture(source, dest, MipRegion(), kCopyMip, 0, kMipRowBytes,
                                 (uint32_t)kMipBytes);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image mip1;
    Image mip0;
    AGFX_EXPECT_MSG(ReadbackTextureSubresource(device.Get(), queue, dest, kMipWidth, kMipHeight, kFormat,
                                               AGFX_RESOURCE_STATE_COPY_DEST, kCopyMip, 0, mip1),
                    "mip 1 readback failed");
    AGFX_EXPECT_MSG(ReadbackTextureSubresource(device.Get(), queue, dest, kBaseWidth, kBaseHeight,
                                               kFormat, AGFX_RESOURCE_STATE_COMMON, 0, 0, mip0),
                    "mip 0 readback failed");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(mip1, sourcePixels), "mip 1 does not hold the buffer's contents");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(mip0, seed0), "the copy disturbed mip 0");

    ExpectImageMatchesGolden(ctx, kGolden, mip1);
}

AGFX_TEST_TEXTURE(CopyBufferToTextureMip, Ez, kMipWidth, kMipHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kBaseWidth;
    contextInfo.height = kBaseHeight;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    agfx::ez::Texture2D dest = context.CreateTexture2D(kBaseWidth, kBaseHeight, kFormat,
                                                       AGFX_TEXTURE_USAGE_SAMPLED, nullptr, 0, kMipLevels);

    const std::vector<uint8_t> seed0 = SeedPixels(kBaseWidth, kBaseHeight);
    const std::vector<uint8_t> seed1 = SeedPixels(kMipWidth, kMipHeight);
    const std::vector<uint8_t> sourcePixels = SourcePixels();
    context.UploadTexture(dest, agfxTextureRegion{0, 0, 0, kBaseWidth, kBaseHeight, 1}, 0, 0, seed0.data(),
                          (uint32_t)seed0.size(), kBaseWidth * kBytesPerPixel);
    context.UploadTexture(dest, MipRegion(), kCopyMip, 0, seed1.data(), (uint32_t)seed1.size(), kMipRowBytes);
    agfx::ez::Buffer source = context.CreateStructuredBuffer(sourcePixels.data(), kMipBytes, kBytesPerPixel,
                                                             /*shaderWritable*/ true);

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionBuffer(source, AGFX_RESOURCE_STATE_COPY_SOURCE);
        context.TransitionTexture(dest, AGFX_RESOURCE_STATE_COPY_DEST, kCopyMip, 0);
        context.CopyBufferToTexture(source, dest, MipRegion(), kCopyMip, 0, kMipRowBytes, (uint32_t)kMipBytes);
    }
    context.DrainGPU();

    Image mip1;
    Image mip0;
    AGFX_EXPECT_MSG(ReadbackTextureSubresource(device.Get(), context.GetGraphicsQueue(), dest.Raw(), kMipWidth,
                                               kMipHeight, kFormat, dest.StateAt(kCopyMip, 0), kCopyMip, 0, mip1),
                    "mip 1 readback failed");
    AGFX_EXPECT_MSG(ReadbackTextureSubresource(device.Get(), context.GetGraphicsQueue(), dest.Raw(), kBaseWidth,
                                               kBaseHeight, kFormat, dest.StateAt(0, 0), 0, 0, mip0),
                    "mip 0 readback failed");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(mip1, sourcePixels), "mip 1 does not hold the buffer's contents");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(mip0, seed0), "the copy disturbed mip 0");

    ExpectImageMatchesGolden(ctx, kGolden, mip1);
}
