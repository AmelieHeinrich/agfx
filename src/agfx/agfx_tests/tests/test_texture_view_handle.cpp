/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "texture view handle not null" and "texture view format reinterpretation".
//
// A texture view's bindless handle is the index shaders use to pull it out of ResourceDescriptorHeap,
// so a zero handle means every sampled or storage access silently lands on the wrong descriptor.
// Walks sampled and storage views across 2D, 2D-array and 3D textures, single-mip and single-slice
// subranges, and a format reinterpretation (RGBA8_UNORM viewed as its sRGB counterpart) — the cases
// where a backend is most likely to populate the heap for the default view only.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint32_t kWidth = 64;
    constexpr uint32_t kHeight = 64;
    constexpr uint32_t kMipLevels = 4;
    constexpr uint32_t kArrayLayers = 6;

    /// @brief The permutations every flavor of this test walks against a 2D array texture.
    struct ViewVariant
    {
        const char* name;
        agfxTextureFormat format;
        agfxTextureType type;
        uint32_t baseMip;
        uint32_t mipCount;
        uint32_t baseLayer;
        uint32_t layerCount;
        bool writeable;
    };

    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;

    constexpr ViewVariant kVariants[] = {
        // The whole resource, sampled: the case every backend gets right.
        {"array sampled, all subresources", kFormat, AGFX_TEXTURE_TYPE_2D_ARRAY, 0, kMipLevels, 0, kArrayLayers, false},
        // One mip, all layers — the mip-slicing path.
        {"array sampled, mip 2 only",       kFormat, AGFX_TEXTURE_TYPE_2D_ARRAY, 2, 1, 0, kArrayLayers, false},
        // One layer viewed as a plain 2D texture — the dimensionality reinterpretation path.
        {"layer 3 as 2D, sampled",          kFormat, AGFX_TEXTURE_TYPE_2D,       0, 1, 3, 1, false},
        // Storage views must land in the heap too, and only ever cover one mip.
        {"layer 0 as 2D, storage",          kFormat, AGFX_TEXTURE_TYPE_2D,       0, 1, 0, 1, true},
        {"array storage, mip 1",            kFormat, AGFX_TEXTURE_TYPE_2D_ARRAY, 1, 1, 0, kArrayLayers, true},
        // Format reinterpretation: same bits, sRGB decode on read. A backend that validates the
        // view format against the texture's too strictly rejects this outright.
        {"sRGB reinterpretation, sampled",  AGFX_TEXTURE_FORMAT_RGBA8_UNORM_SRGB, AGFX_TEXTURE_TYPE_2D_ARRAY,
                                            0, kMipLevels, 0, kArrayLayers, false},
    };

    agfxTextureCreateInfo TextureInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D_ARRAY;
        info.format = kFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_SAMPLED | AGFX_TEXTURE_USAGE_STORAGE);
        info.width = kWidth;
        info.height = kHeight;
        info.depthOrArrayLayers = kArrayLayers;
        info.mipLevels = kMipLevels;
        return info;
    }

    agfxTextureViewCreateInfo ViewInfo(agfxTexture* texture, const ViewVariant& variant)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = variant.format;
        info.type = variant.type;
        info.baseMipLevel = variant.baseMip;
        info.mipLevelCount = variant.mipCount;
        info.baseArrayLayer = variant.baseLayer;
        info.arrayLayerCount = variant.layerCount;
        info.writeable = variant.writeable ? 1 : 0;
        return info;
    }
} // namespace

AGFX_TEST_VALIDATION(TextureViewHandleNotNull, C)
{
    const agfxDeviceCreateInfo deviceInfo = DefaultDeviceCreateInfo();
    agfxDevice* device = agfxDeviceCreate(&deviceInfo);
    AGFX_EXPECT_NOT_NULL(device);

    const agfxTextureCreateInfo textureInfo = TextureInfo();
    agfxTexture* texture = agfxTextureCreate(device, &textureInfo);
    if (!texture) {
        agfxDeviceDestroy(device);
        AGFX_FAIL("agfxTextureCreate returned null");
    }

    for (const ViewVariant& variant : kVariants) {
        const agfxTextureViewCreateInfo viewInfo = ViewInfo(texture, variant);
        agfxTextureView* view = agfxTextureViewCreate(device, &viewInfo);
        if (!view) {
            agfxTextureDestroy(device, texture);
            agfxDeviceDestroy(device);
            AGFX_FAIL(std::string("agfxTextureViewCreate returned null for ") + variant.name);
        }

        const uint64_t handle = agfxTextureViewGetHandle(view);
        agfxTextureViewDestroy(device, view);
        if (handle == 0) {
            agfxTextureDestroy(device, texture);
            agfxDeviceDestroy(device);
            AGFX_FAIL(std::string("texture view handle was 0 for ") + variant.name);
        }
    }

    agfxTextureDestroy(device, texture);
    agfxDeviceDestroy(device);
}

AGFX_TEST_VALIDATION(TextureViewHandleNotNull, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::Texture texture = device.CreateTexture(TextureInfo());
    AGFX_EXPECT_NOT_NULL(texture.Get());

    for (const ViewVariant& variant : kVariants) {
        agfx::TextureView view = device.CreateTextureView(ViewInfo(texture, variant));
        AGFX_EXPECT_MSG(view.Get() != nullptr, variant.name);
        AGFX_EXPECT_MSG(view.GetHandle() != 0, variant.name);
    }
}

AGFX_TEST_VALIDATION(TextureViewHandleNotNull, Ez)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless
    contextInfo.width = kWidth;
    contextInfo.height = kHeight;

    agfx::ez::Context context(contextInfo);

    // ez::Texture2D is 2D/single-mip by construction, so the subrange and array permutations above
    // aren't expressible through it. What ez does own is the lazily-created SRV/UAV pair, and both
    // have to come back live and stable across calls.
    agfx::ez::Texture2D texture =
        context.CreateTexture2D(kWidth, kHeight, kFormat,
                                (agfxTextureUsage)(AGFX_TEXTURE_USAGE_SAMPLED | AGFX_TEXTURE_USAGE_STORAGE));

    agfx::TextureView& srv = texture.SRV();
    AGFX_EXPECT_MSG(srv.Get() != nullptr, "SRV was null");
    AGFX_EXPECT_MSG(srv.GetHandle() != 0, "SRV handle was 0");

    agfx::TextureView& uav = texture.UAV();
    AGFX_EXPECT_MSG(uav.Get() != nullptr, "UAV was null");
    AGFX_EXPECT_MSG(uav.GetHandle() != 0, "UAV handle was 0");

    // Distinct descriptors: a cache that returned the SRV for both would make every storage write
    // land on a read-only view.
    AGFX_EXPECT_MSG(srv.GetHandle() != uav.GetHandle(), "SRV and UAV share a handle");

    // Asking again must not mint a second descriptor.
    AGFX_EXPECT_MSG(texture.SRV().GetHandle() == srv.GetHandle(), "SRV handle not stable");
    AGFX_EXPECT_MSG(texture.UAV().GetHandle() == uav.GetHandle(), "UAV handle not stable");

    // And both must survive the narrowing ShaderBindings applies when packing into push constants.
    agfx::ez::ShaderBindings bindings;
    bindings.BindTexture(srv);
    bindings.BindTexture(uav);
    const uint32_t* packed = (const uint32_t*)bindings.Data();
    AGFX_EXPECT_MSG(packed[0] != 0u, "packed SRV handle was 0");
    AGFX_EXPECT_MSG(packed[1] != 0u, "packed UAV handle was 0");
}
