/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "copy buffer to texture" and "copy buffer to texture region".
//
// No shaders: a CPU-built RGBA8 image goes into an upload buffer and is copied into a texture, first
// whole-surface and then as a smaller offset region overwriting part of it. The region copy uses a
// different pattern from the base, so a backend that ignores the region origin, mixes up
// width/height, or mis-computes the row pitch leaves the second pattern in the wrong place — visible
// in the golden rather than merely off by a row.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 128;
    constexpr uint32_t kHeight = 128;
    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr uint32_t kBytesPerPixel = 4;
    constexpr uint32_t kRowBytes = kWidth * kBytesPerPixel; // 512: already 256-aligned for D3D12.
    constexpr uint64_t kImageBytes = uint64_t(kRowBytes) * kHeight;

    // The region copy: a 32x32 patch dropped at a deliberately asymmetric origin, so swapping x/y
    // or width/height moves it somewhere the golden doesn't have it.
    constexpr uint32_t kRegionX = 16;
    constexpr uint32_t kRegionY = 64;
    constexpr uint32_t kRegionW = 32;
    constexpr uint32_t kRegionH = 32;
    constexpr uint32_t kRegionRowBytes = kRegionW * kBytesPerPixel;
    constexpr uint64_t kRegionBytes = uint64_t(kRegionRowBytes) * kRegionH;

    constexpr const char* kGolden = "copy_buffer_to_texture.png";

    /// @brief The full-surface pattern: a two-axis gradient with a coarse checker in blue.
    std::vector<uint8_t> BasePixels()
    {
        std::vector<uint8_t> pixels((size_t)kImageBytes);
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kRowBytes + (size_t)x * kBytesPerPixel];
                texel[0] = (uint8_t)(x * 255u / (kWidth - 1u));
                texel[1] = (uint8_t)(y * 255u / (kHeight - 1u));
                texel[2] = ((x / 16u) + (y / 16u)) % 2u ? 255u : 0u;
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    /// @brief The patch pattern: deliberately unlike the base so its placement is unambiguous.
    std::vector<uint8_t> RegionPixels()
    {
        std::vector<uint8_t> pixels((size_t)kRegionBytes);
        for (uint32_t y = 0; y < kRegionH; ++y) {
            for (uint32_t x = 0; x < kRegionW; ++x) {
                uint8_t* texel = &pixels[(size_t)y * kRegionRowBytes + (size_t)x * kBytesPerPixel];
                // A diagonal wedge: asymmetric under both transpose and flip.
                const bool inside = x >= y;
                texel[0] = inside ? 255u : 32u;
                texel[1] = 0u;
                texel[2] = inside ? 0u : 255u;
                texel[3] = 255u;
            }
        }
        return pixels;
    }

    agfxTextureCreateInfo TargetInfo()
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

    agfxBufferCreateInfo StagingInfo(uint64_t size)
    {
        agfxBufferCreateInfo info{};
        info.size = size;
        info.stride = kBytesPerPixel;
        info.usage = AGFX_BUFFER_USAGE_SHADER_READ;
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_UPLOAD;
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

    agfxTextureRegion PatchRegion()
    {
        agfxTextureRegion region{};
        region.x = kRegionX;
        region.y = kRegionY;
        region.width = kRegionW;
        region.height = kRegionH;
        region.depth = 1;
        return region;
    }
} // namespace

AGFX_TEST_TEXTURE(CopyBufferToTexture, C, kWidth, kHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const agfxTextureCreateInfo targetInfo = TargetInfo();
    agfxTexture* target = agfxTextureCreate(device, &targetInfo);
    AGFX_EXPECT_NOT_NULL(target);

    const agfxBufferCreateInfo baseInfo = StagingInfo(kImageBytes);
    const agfxBufferCreateInfo patchInfo = StagingInfo(kRegionBytes);
    agfxBuffer* baseBuffer = agfxBufferCreate(device, &baseInfo);
    agfxBuffer* patchBuffer = agfxBufferCreate(device, &patchInfo);
    AGFX_EXPECT_NOT_NULL(baseBuffer);
    AGFX_EXPECT_NOT_NULL(patchBuffer);

    agfxDeviceMakeResourcesResident(device);

    const std::vector<uint8_t> base = BasePixels();
    const std::vector<uint8_t> patch = RegionPixels();
    void* baseMapped = agfxBufferMap(baseBuffer);
    void* patchMapped = agfxBufferMap(patchBuffer);
    if (!baseMapped || !patchMapped) {
        agfxBufferDestroy(device, patchBuffer);
        agfxBufferDestroy(device, baseBuffer);
        agfxTextureDestroy(device, target);
        AGFX_FAIL("failed to map an upload buffer");
    }
    memcpy(baseMapped, base.data(), base.size());
    memcpy(patchMapped, patch.data(), patch.size());
    agfxBufferUnmap(baseBuffer);
    agfxBufferUnmap(patchBuffer);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_COPY_DEST, 0, 0, 0);

        const agfxTextureRegion full = FullRegion();
        const agfxTextureRegion patchRegion = PatchRegion();

        agfxComputePass* pass = agfxComputePassBegin(cmd, "copy buffer to texture");
        agfxComputePassCopyBufferToTexture(pass, baseBuffer, target, &full, 0, 0, kRowBytes,
                                           (uint32_t)kImageBytes);
        agfxComputePassCopyBufferToTexture(pass, patchBuffer, target, &patchRegion, 0, 0, kRegionRowBytes,
                                           (uint32_t)kRegionBytes);
        agfxComputePassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), target, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_COPY_DEST, image);

    agfxBufferDestroy(device, patchBuffer);
    agfxBufferDestroy(device, baseBuffer);
    agfxTextureDestroy(device, target);

    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(CopyBufferToTexture, Cpp, kWidth, kHeight)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    agfx::Texture target = device.CreateTexture(TargetInfo());
    agfx::Buffer baseBuffer = device.CreateBuffer(StagingInfo(kImageBytes));
    agfx::Buffer patchBuffer = device.CreateBuffer(StagingInfo(kRegionBytes));
    AGFX_EXPECT_NOT_NULL(target.Get());
    AGFX_EXPECT_NOT_NULL(baseBuffer.Get());
    AGFX_EXPECT_NOT_NULL(patchBuffer.Get());

    device.MakeResourcesResident();

    const std::vector<uint8_t> base = BasePixels();
    const std::vector<uint8_t> patch = RegionPixels();
    {
        agfx::MappedBuffer mapping(baseBuffer);
        AGFX_EXPECT_NOT_NULL(mapping.Get());
        memcpy(mapping.Get(), base.data(), base.size());
    }
    {
        agfx::MappedBuffer mapping(patchBuffer);
        AGFX_EXPECT_NOT_NULL(mapping.Get());
        memcpy(mapping.Get(), patch.data(), patch.size());
    }

    cmd.Begin();
    cmd.TextureBarrier(target, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_DEST,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("copy buffer to texture");
        pass.CopyBufferToTexture(baseBuffer, target, FullRegion(), 0, 0, kRowBytes, (uint32_t)kImageBytes);
        pass.CopyBufferToTexture(patchBuffer, target, PatchRegion(), 0, 0, kRegionRowBytes,
                                 (uint32_t)kRegionBytes);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), queue, target, kWidth, kHeight, kFormat,
                                          AGFX_RESOURCE_STATE_COPY_DEST, image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(CopyBufferToTexture, Ez, kWidth, kHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    // The base image goes in through ez's own uploader at creation time — that is the ez spelling of
    // "copy a buffer into a texture", staging buffer and all.
    const std::vector<uint8_t> base = BasePixels();
    agfx::ez::Texture2D target =
        context.CreateTexture2D(kWidth, kHeight, kFormat, AGFX_TEXTURE_USAGE_SAMPLED, base.data(), kRowBytes);

    // The region copy has no ez sugar, so it goes through a staging buffer and the raw command
    // buffer, exercising the offset path the uploader never takes.
    const std::vector<uint8_t> patch = RegionPixels();
    agfx::ez::Buffer patchBuffer =
        context.CreateStructuredBuffer(patch.data(), kRegionBytes, kBytesPerPixel);

    device.MakeResourcesResident();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionBuffer(patchBuffer, AGFX_RESOURCE_STATE_COPY_SOURCE);
        context.TransitionTexture(target, AGFX_RESOURCE_STATE_COPY_DEST);

        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("copy buffer to texture");
        pass.CopyBufferToTexture(patchBuffer.Raw(), target.Raw(), PatchRegion(), 0, 0, kRegionRowBytes,
                                 (uint32_t)kRegionBytes);
    }
    context.DrainGPU();

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), target.Raw(),
                                          kWidth, kHeight, kFormat, target.State(), image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");

    ExpectImageMatchesGolden(ctx, kGolden, image);
}
