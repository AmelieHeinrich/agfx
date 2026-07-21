/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "copy texture to texture slice".
//
// The last of the mip/slice copy family: a sub-region of layer 2 of a four-layer array is copied onto
// layer 2 of another. Both arrays have every layer seeded distinctly, so the check that the other
// three destination layers are untouched is what actually holds the layer argument to account —
// without it, a backend that wrote every layer would still produce a correct-looking layer 2.

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

    // Neither square nor centred: a transposed or recentred copy shows up.
    constexpr uint32_t kCopyX = 16;
    constexpr uint32_t kCopyY = 4;
    constexpr uint32_t kCopyW = 32;
    constexpr uint32_t kCopyH = 16;

    constexpr const char* kGolden = "copy_texture_to_texture_slice.png";

    std::vector<uint8_t> SourcePixels(uint32_t layer)
    {
        std::vector<uint8_t> pixels((size_t)kLayerBytes);
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                texel[0] = (uint8_t)(x * 4u + layer * 47u);
                texel[1] = (uint8_t)(y * 4u + layer * 83u);
                texel[2] = 0u;
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    std::vector<uint8_t> DestPixels(uint32_t layer)
    {
        std::vector<uint8_t> pixels((size_t)kLayerBytes);
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                texel[0] = 0u;
                texel[1] = (uint8_t)(layer * 55u);
                texel[2] = ((x + y) / 8u) % 2u ? 255u : 80u;
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    /// @brief Destination layer 2 afterwards: its seed, with the copied window taken from the
    /// source's layer 2 at the same coordinates.
    std::vector<uint8_t> ExpectedLayer()
    {
        std::vector<uint8_t> expected = DestPixels(kCopyLayer);
        const std::vector<uint8_t> source = SourcePixels(kCopyLayer);
        for (uint32_t y = kCopyY; y < kCopyY + kCopyH; ++y) {
            const size_t rowStart = (size_t)y * kRowBytes + (size_t)kCopyX * kBytesPerPixel;
            memcpy(&expected[rowStart], &source[rowStart], (size_t)kCopyW * kBytesPerPixel);
        }
        return expected;
    }

    agfxTextureCreateInfo TextureInfo()
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

AGFX_TEST_TEXTURE(CopyTextureToTextureSlice, C, kWidth, kHeight)
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

    bool seeded = true;
    for (uint32_t layer = 0; layer < kLayerCount; ++layer) {
        const std::vector<uint8_t> sourcePixels = SourcePixels(layer);
        const std::vector<uint8_t> destSeed = DestPixels(layer);
        seeded &= UploadTextureSubresource(device, gpu.Queue(), source, kWidth, kHeight, kFormat,
                                           sourcePixels.data(), AGFX_RESOURCE_STATE_COMMON, 0, layer);
        seeded &= UploadTextureSubresource(device, gpu.Queue(), dest, kWidth, kHeight, kFormat,
                                           destSeed.data(), AGFX_RESOURCE_STATE_COMMON, 0, layer);
    }

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, source, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_COPY_SOURCE, 0, kCopyLayer, 0);
        agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_COPY_DEST, 0, kCopyLayer, 0);

        const agfxTextureRegion region = CopyRegion();
        agfxComputePass* pass = agfxComputePassBegin(cmd, "copy texture slice to texture slice");
        agfxComputePassCopyTextureToTexture(pass, source, dest, &region, 0, kCopyLayer);
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
        neighboursIntact &= ImageEqualsRgba8(neighbour, DestPixels(layer));
    }

    agfxTextureDestroy(device, dest);
    agfxTextureDestroy(device, source);

    AGFX_EXPECT_MSG(seeded, "failed to seed the source or destination layers");
    AGFX_EXPECT_MSG(readOk, "layer readback failed");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(target, ExpectedLayer()),
                    "layer 2 is not the seed with the copied window");
    AGFX_EXPECT_MSG(neighboursIntact, "the copy disturbed a destination layer other than layer 2");
    ExpectImageMatchesGolden(ctx, kGolden, target);
}

AGFX_TEST_TEXTURE(CopyTextureToTextureSlice, Cpp, kWidth, kHeight)
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

    for (uint32_t layer = 0; layer < kLayerCount; ++layer) {
        const std::vector<uint8_t> sourcePixels = SourcePixels(layer);
        const std::vector<uint8_t> destSeed = DestPixels(layer);
        AGFX_EXPECT_MSG(UploadTextureSubresource(device.Get(), queue, source, kWidth, kHeight, kFormat,
                                                 sourcePixels.data(), AGFX_RESOURCE_STATE_COMMON, 0, layer),
                        "failed to seed a source layer");
        AGFX_EXPECT_MSG(UploadTextureSubresource(device.Get(), queue, dest, kWidth, kHeight, kFormat,
                                                 destSeed.data(), AGFX_RESOURCE_STATE_COMMON, 0, layer),
                        "failed to seed a destination layer");
    }

    cmd.Begin();
    cmd.TextureBarrier(source, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_SOURCE,
                       AGFX_SUBRESOURCE_ALL_MIPS, kCopyLayer, false);
    cmd.TextureBarrier(dest, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_DEST,
                       AGFX_SUBRESOURCE_ALL_MIPS, kCopyLayer, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("copy texture slice to texture slice");
        pass.CopyTextureToTexture(source, dest, CopyRegion(), 0, kCopyLayer);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image target;
    AGFX_EXPECT_MSG(ReadbackTextureSubresource(device.Get(), queue, dest, kWidth, kHeight, kFormat,
                                               AGFX_RESOURCE_STATE_COPY_DEST, 0, kCopyLayer, target),
                    "layer readback failed");
    AGFX_EXPECT_MSG(ImageEqualsRgba8(target, ExpectedLayer()),
                    "layer 2 is not the seed with the copied window");

    for (uint32_t layer = 0; layer < kLayerCount; ++layer) {
        if (layer == kCopyLayer) {
            continue;
        }
        Image neighbour;
        AGFX_EXPECT_MSG(ReadbackTextureSubresource(device.Get(), queue, dest, kWidth, kHeight, kFormat,
                                                   AGFX_RESOURCE_STATE_COMMON, 0, layer, neighbour),
                        "neighbour layer readback failed");
        AGFX_EXPECT_MSG(ImageEqualsRgba8(neighbour, DestPixels(layer)),
                        "the copy disturbed a destination layer other than layer 2");
    }

    ExpectImageMatchesGolden(ctx, kGolden, target);
}
