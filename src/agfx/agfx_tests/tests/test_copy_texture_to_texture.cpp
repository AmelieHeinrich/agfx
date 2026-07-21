/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "copy texture to texture".
//
// Seeds a source texture with a known pattern, seeds the destination with a different one, then
// copies a sub-region of the source over the destination. Because both surfaces start out populated
// and distinct, the golden pins down exactly which texels the copy was allowed to touch — a copy
// that overreaches its region, or that silently blits the whole surface, changes texels the golden
// says should still hold the destination's own pattern.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 128;
    constexpr uint32_t kHeight = 128;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr uint32_t kBytesPerPixel = 4;
    constexpr uint32_t kRowBytes = kWidth * kBytesPerPixel;
    constexpr uint64_t kImageBytes = uint64_t(kRowBytes) * kHeight;

    // The copied window. Square and centred would be indistinguishable under a transpose, so it is
    // neither.
    constexpr uint32_t kCopyX = 32;
    constexpr uint32_t kCopyY = 8;
    constexpr uint32_t kCopyW = 64;
    constexpr uint32_t kCopyH = 32;

    constexpr const char* kGolden = "copy_texture_to_texture.png";

    /// @brief Source: warm horizontal ramp with fine vertical banding.
    std::vector<uint8_t> SourcePixels()
    {
        std::vector<uint8_t> pixels((size_t)kImageBytes);
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                texel[0] = (uint8_t)(x * 255u / (kWidth - 1u));
                texel[1] = (y % 8u) < 4u ? 200u : 60u;
                texel[2] = 0u;
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    /// @brief Destination: cool diagonal stripes. Nothing in common with the source, so the copied
    /// window's boundary is unambiguous.
    std::vector<uint8_t> DestPixels()
    {
        std::vector<uint8_t> pixels((size_t)kImageBytes);
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                texel[0] = 0u;
                texel[1] = 0u;
                texel[2] = ((x + y) / 8u) % 2u ? 255u : 80u;
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    agfxTextureCreateInfo TextureInfo()
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

    agfxTextureRegion CopyRegion()
    {
        agfxTextureRegion region{};
        region.x = kCopyX;
        region.y = kCopyY;
        region.width = kCopyW;
        region.height = kCopyH;
        region.depth = 1;
        return region;
    }
} // namespace

AGFX_TEST_TEXTURE(CopyTextureToTexture, C, kWidth, kHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const agfxTextureCreateInfo textureInfo = TextureInfo();
    agfxTexture* source = agfxTextureCreate(device, &textureInfo);
    agfxTexture* dest = agfxTextureCreate(device, &textureInfo);
    AGFX_EXPECT_NOT_NULL(source);
    AGFX_EXPECT_NOT_NULL(dest);

    agfxDeviceMakeResourcesResident(device);

    const std::vector<uint8_t> sourcePixels = SourcePixels();
    const std::vector<uint8_t> destPixels = DestPixels();
    const bool seededSource = UploadTexture2D(device, gpu.Queue(), source, kWidth, kHeight, kFormat,
                                              sourcePixels.data(), AGFX_RESOURCE_STATE_COMMON);
    const bool seededDest = UploadTexture2D(device, gpu.Queue(), dest, kWidth, kHeight, kFormat,
                                            destPixels.data(), AGFX_RESOURCE_STATE_COMMON);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_COPY_SOURCE, 0, 0, 0);
        agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_COPY_DEST, 0, 0, 0);

        const agfxTextureRegion region = CopyRegion();
        agfxComputePass* pass = agfxComputePassBegin(cmd, "copy texture to texture");
        agfxComputePassCopyTextureToTexture(pass, source, dest, &region, 0, 0);
        agfxComputePassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), dest, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_COPY_DEST, image);

    agfxTextureDestroy(device, dest);
    agfxTextureDestroy(device, source);

    AGFX_EXPECT_MSG(seededSource, "failed to seed the source texture");
    AGFX_EXPECT_MSG(seededDest, "failed to seed the destination texture");
    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(CopyTextureToTexture, Cpp, kWidth, kHeight)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    agfx::Texture source = device.CreateTexture(TextureInfo());
    agfx::Texture dest = device.CreateTexture(TextureInfo());
    AGFX_EXPECT_NOT_NULL(source.Get());
    AGFX_EXPECT_NOT_NULL(dest.Get());

    device.MakeResourcesResident();

    const std::vector<uint8_t> sourcePixels = SourcePixels();
    const std::vector<uint8_t> destPixels = DestPixels();
    AGFX_EXPECT_MSG(UploadTexture2D(device.Get(), queue, source, kWidth, kHeight, kFormat,
                                    sourcePixels.data(), AGFX_RESOURCE_STATE_COMMON),
                    "failed to seed the source texture");
    AGFX_EXPECT_MSG(UploadTexture2D(device.Get(), queue, dest, kWidth, kHeight, kFormat,
                                    destPixels.data(), AGFX_RESOURCE_STATE_COMMON),
                    "failed to seed the destination texture");

    cmd.Begin();
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_SOURCE,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_DEST,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("copy texture to texture");
        pass.CopyTextureToTexture(source, dest, CopyRegion(), 0, 0);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), queue, dest, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_COPY_DEST, image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(CopyTextureToTexture, Ez, kWidth, kHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    const std::vector<uint8_t> sourcePixels = SourcePixels();
    const std::vector<uint8_t> destPixels = DestPixels();
    agfx::ez::Texture2D source = context.CreateTexture2D(kWidth, kHeight, kFormat,
                                                         AGFX_TEXTURE_USAGE_SAMPLED, sourcePixels.data(), kRowBytes);
    agfx::ez::Texture2D dest = context.CreateTexture2D(kWidth, kHeight, kFormat,
                                                       AGFX_TEXTURE_USAGE_SAMPLED, destPixels.data(), kRowBytes);

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(source, AGFX_RESOURCE_STATE_COPY_SOURCE);
        context.TransitionTexture(dest, AGFX_RESOURCE_STATE_COPY_DEST);

        context.CopyTextureToTexture(source, dest, CopyRegion(), 0, 0);
    }
    context.DrainGPU();

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), dest.Raw(),
                                          kWidth, kHeight, kFormat, dest.State(), image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
