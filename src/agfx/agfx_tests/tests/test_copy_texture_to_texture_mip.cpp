/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "copy texture to texture mip".
//
// agfxComputePassCopyTextureToTexture takes one mipLevel for both sides, so this copies mip 1 of a
// two-mip source over mip 1 of a two-mip destination. Every mip on both textures is seeded, and with
// a sub-region rather than the whole surface, three failure modes stay separable: reading the source's
// mip 0, writing the destination's mip 0, and overrunning the region. Mip 0 of the destination is
// checked to be untouched alongside the mip 1 golden.

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

    // A window that is neither square nor centred, so a transposed or recentred copy is visible.
    constexpr uint32_t kCopyX = 16;
    constexpr uint32_t kCopyY = 4;
    constexpr uint32_t kCopyW = 32;
    constexpr uint32_t kCopyH = 16;

    constexpr const char* kGolden = "copy_texture_to_texture_mip.png";

    std::vector<uint8_t> SourcePixels(uint32_t mip, uint32_t width, uint32_t height)
    {
        std::vector<uint8_t> pixels((size_t)width * height * kBytesPerPixel);
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint8_t* texel = &pixels[((size_t)y * width + x) * kBytesPerPixel];
                texel[0] = (uint8_t)(x * 4u + mip * 89u);
                texel[1] = (uint8_t)(y * 4u + mip * 41u);
                texel[2] = 0u;
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    std::vector<uint8_t> DestPixels(uint32_t mip, uint32_t width, uint32_t height)
    {
        std::vector<uint8_t> pixels((size_t)width * height * kBytesPerPixel);
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                uint8_t* texel = &pixels[((size_t)y * width + x) * kBytesPerPixel];
                texel[0] = 0u;
                texel[1] = 0u;
                texel[2] = (uint8_t)(((x + y) / 8u) % 2u ? 255u : 80u - mip * 40u);
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    /// @brief What mip 1 of the destination must look like afterwards: its own seed everywhere
    /// except the copied window, which holds the source's mip 1 texels from the same coordinates.
    std::vector<uint8_t> ExpectedMip1()
    {
        std::vector<uint8_t> expected = DestPixels(kCopyMip, kMipWidth, kMipHeight);
        const std::vector<uint8_t> source = SourcePixels(kCopyMip, kMipWidth, kMipHeight);
        for (uint32_t y = kCopyY; y < kCopyY + kCopyH; ++y) {
            const size_t rowStart = ((size_t)y * kMipWidth + kCopyX) * kBytesPerPixel;
            memcpy(&expected[rowStart], &source[rowStart], (size_t)kCopyW * kBytesPerPixel);
        }
        return expected;
    }

    agfxTextureCreateInfo TextureInfo()
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

AGFX_TEST_TEXTURE(CopyTextureToTextureMip, C, kMipWidth, kMipHeight)
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

    const std::vector<uint8_t> destMip0 = DestPixels(0, kBaseWidth, kBaseHeight);
    bool seeded = true;
    for (uint32_t mip = 0; mip < kMipLevels; ++mip) {
        const uint32_t width = kBaseWidth >> mip;
        const uint32_t height = kBaseHeight >> mip;
        const std::vector<uint8_t> sourcePixels = SourcePixels(mip, width, height);
        const std::vector<uint8_t> destSeed = DestPixels(mip, width, height);
        seeded &= UploadTextureSubresource(device, gpu.Queue(), source, width, height, kFormat,
                                           sourcePixels.data(), AGFX_RESOURCE_STATE_COMMON, mip, 0);
        seeded &= UploadTextureSubresource(device, gpu.Queue(), dest, width, height, kFormat,
                                           destSeed.data(), AGFX_RESOURCE_STATE_COMMON, mip, 0);
    }

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_COPY_SOURCE, kCopyMip, 0, 0);
        agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_COPY_DEST, kCopyMip, 0, 0);

        const agfxTextureRegion region = CopyRegion();
        agfxComputePass* pass = agfxComputePassBegin(cmd, "copy texture mip to texture mip");
        agfxComputePassCopyTextureToTexture(pass, source, dest, &region, kCopyMip, 0);
        agfxComputePassEnd(pass);
    });

    Image mip1;
    Image mip0;
    const bool readMip1 = ReadbackTextureSubresource(device, gpu.Queue(), dest, kMipWidth, kMipHeight,
                                                     kFormat, AGFX_RESOURCE_STATE_COPY_DEST, kCopyMip,
                                                     0, mip1);
    const bool readMip0 = ReadbackTextureSubresource(device, gpu.Queue(), dest, kBaseWidth, kBaseHeight,
                                                     kFormat, AGFX_RESOURCE_STATE_COMMON, 0, 0, mip0);

    agfxTextureDestroy(device, dest);
    agfxTextureDestroy(device, source);

    AGFX_EXPECT_MSG(seeded, "failed to seed the source or destination mips");
    AGFX_EXPECT_MSG(readMip1, "mip 1 readback failed");
    AGFX_EXPECT_MSG(readMip0, "mip 0 readback failed");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(mip1, ExpectedMip1()), "mip 1 is not the seed with the copied window");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(mip0, destMip0), "the copy disturbed the destination's mip 0");
    ExpectImageMatchesGolden(ctx, kGolden, mip1);
}

AGFX_TEST_TEXTURE(CopyTextureToTextureMip, Cpp, kMipWidth, kMipHeight)
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

    const std::vector<uint8_t> destMip0 = DestPixels(0, kBaseWidth, kBaseHeight);
    for (uint32_t mip = 0; mip < kMipLevels; ++mip) {
        const uint32_t width = kBaseWidth >> mip;
        const uint32_t height = kBaseHeight >> mip;
        const std::vector<uint8_t> sourcePixels = SourcePixels(mip, width, height);
        const std::vector<uint8_t> destSeed = DestPixels(mip, width, height);
        AGFX_EXPECT_MSG(UploadTextureSubresource(device.Get(), queue, source, width, height, kFormat,
                                                 sourcePixels.data(), AGFX_RESOURCE_STATE_COMMON, mip, 0),
                        "failed to seed a source mip");
        AGFX_EXPECT_MSG(UploadTextureSubresource(device.Get(), queue, dest, width, height, kFormat,
                                                 destSeed.data(), AGFX_RESOURCE_STATE_COMMON, mip, 0),
                        "failed to seed a destination mip");
    }

    cmd.Begin();
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_SOURCE, kCopyMip,
                       AGFX_SUBRESOURCE_ALL_LAYERS, false);
    cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_DEST, kCopyMip,
                       AGFX_SUBRESOURCE_ALL_LAYERS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("copy texture mip to texture mip");
        pass.CopyTextureToTexture(source, dest, CopyRegion(), kCopyMip, 0);
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
    AGFX_EXPECT_MSG(ImageEqualsRgba8(mip1, ExpectedMip1()), "mip 1 is not the seed with the copied window");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(mip0, destMip0), "the copy disturbed the destination's mip 0");

    ExpectImageMatchesGolden(ctx, kGolden, mip1);
}

AGFX_TEST_TEXTURE(CopyTextureToTextureMip, Ez, kMipWidth, kMipHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kBaseWidth;
    contextInfo.height = kBaseHeight;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    agfx::ez::Texture2D source = context.CreateTexture2D(kBaseWidth, kBaseHeight, kFormat,
                                                         AGFX_TEXTURE_USAGE_SAMPLED, nullptr, 0, kMipLevels);
    agfx::ez::Texture2D dest = context.CreateTexture2D(kBaseWidth, kBaseHeight, kFormat,
                                                       AGFX_TEXTURE_USAGE_SAMPLED, nullptr, 0, kMipLevels);

    // Context::UploadTexture is the general form of CreateTexture2D's pixels argument -- it is what
    // lets ez seed a mip other than 0, which is the whole reason this test now has an Ez flavor.
    const std::vector<uint8_t> destMip0 = DestPixels(0, kBaseWidth, kBaseHeight);
    for (uint32_t mip = 0; mip < kMipLevels; ++mip) {
        const uint32_t width = kBaseWidth >> mip;
        const uint32_t height = kBaseHeight >> mip;
        const uint32_t rowBytes = width * kBytesPerPixel;
        const std::vector<uint8_t> sourcePixels = SourcePixels(mip, width, height);
        const std::vector<uint8_t> destSeed = DestPixels(mip, width, height);
        const agfxTextureRegion region{0, 0, 0, width, height, 1};
        context.UploadTexture(source, region, mip, 0, sourcePixels.data(), rowBytes * height, rowBytes);
        context.UploadTexture(dest, region, mip, 0, destSeed.data(), rowBytes * height, rowBytes);
    }

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        // Transitioning one mip rather than the whole texture: the tracker splits, so mip 0 stays in
        // COMMON and only mip 1 moves. Reading both back below with their own StateAt() is what
        // proves the split is tracked rather than smeared across the resource.
        context.TransitionTexture(source, AGFX_RESOURCE_STATE_COPY_SOURCE, kCopyMip, 0);
        context.TransitionTexture(dest, AGFX_RESOURCE_STATE_COPY_DEST, kCopyMip, 0);
        context.CopyTextureToTexture(source, dest, CopyRegion(), kCopyMip, 0);
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
    AGFX_EXPECT_MSG(ImageEqualsRgba8(mip1, ExpectedMip1()), "mip 1 is not the seed with the copied window");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(mip0, destMip0), "the copy disturbed the destination's mip 0");

    ExpectImageMatchesGolden(ctx, kGolden, mip1);
}
