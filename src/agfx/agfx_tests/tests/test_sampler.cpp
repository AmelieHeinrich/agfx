/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "sampler handle not null".
//
// A sampler's bindless handle is the index shaders use to pull it out of SamplerDescriptorHeap, so
// a zero/garbage handle means every sampled read in every shader silently reads the wrong state.
// Covers a few filter/address permutations to catch a backend that only populates the heap for the
// default configuration.

#include "../test_gpu.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    /// @brief The permutations every flavor of this test walks.
    struct SamplerVariant
    {
        const char* name;
        agfxSamplerFilter filter;
        agfxSamplerAddressMode addressMode;
        float maxAnisotropy;
    };

    constexpr SamplerVariant kVariants[] = {
        {"linear/repeat",         AGFX_SAMPLER_FILTER_LINEAR,  AGFX_SAMPLER_ADDRESS_MODE_REPEAT,          1.0f},
        {"nearest/clamp",         AGFX_SAMPLER_FILTER_NEAREST, AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,   1.0f},
        {"linear/mirror/aniso16", AGFX_SAMPLER_FILTER_LINEAR,  AGFX_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT, 16.0f},
    };

    agfxSamplerCreateInfo MakeSamplerInfo(const SamplerVariant& variant)
    {
        agfxSamplerCreateInfo info{};
        info.filter = variant.filter;
        info.addressModeU = variant.addressMode;
        info.addressModeV = variant.addressMode;
        info.addressModeW = variant.addressMode;
        info.maxAnisotropy = variant.maxAnisotropy;
        info.comparisonFunction = AGFX_COMPARISON_FUNCTION_ALWAYS;
        info.minLod = 0.0f;
        info.maxLod = 16.0f;
        return info;
    }
} // namespace

AGFX_TEST_VALIDATION(SamplerHandleNotNull, C)
{
    const agfxDeviceCreateInfo deviceInfo = DefaultDeviceCreateInfo();
    agfxDevice* device = agfxDeviceCreate(&deviceInfo);
    AGFX_EXPECT_NOT_NULL(device);

    for (const SamplerVariant& variant : kVariants) {
        const agfxSamplerCreateInfo info = MakeSamplerInfo(variant);
        agfxSampler* sampler = agfxSamplerCreate(device, &info);
        if (!sampler) {
            agfxDeviceDestroy(device);
            AGFX_FAIL(std::string("agfxSamplerCreate returned null for ") + variant.name);
        }

        const uint64_t handle = agfxSamplerGetHandle(sampler);
        agfxSamplerDestroy(device, sampler);
        if (handle == 0) {
            agfxDeviceDestroy(device);
            AGFX_FAIL(std::string("sampler handle was 0 for ") + variant.name);
        }
    }

    agfxDeviceDestroy(device);
}

AGFX_TEST_VALIDATION(SamplerHandleNotNull, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    for (const SamplerVariant& variant : kVariants) {
        agfx::Sampler sampler = device.CreateSampler(MakeSamplerInfo(variant));
        AGFX_EXPECT_MSG(sampler.Get() != nullptr, variant.name);
        AGFX_EXPECT_MSG(sampler.GetHandle() != 0, variant.name);
    }
}

AGFX_TEST_VALIDATION(SamplerHandleNotNull, Ez)
{
    // ez has no sampler wrapper of its own — samplers are stateless and created straight off the
    // Context's device, then bound via ShaderBindings::BindSampler.
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless
    contextInfo.width = 128;
    contextInfo.height = 128;

    agfx::ez::Context context(contextInfo);

    for (const SamplerVariant& variant : kVariants) {
        agfx::Sampler sampler = context.GetDevice().CreateSampler(MakeSamplerInfo(variant));
        AGFX_EXPECT_MSG(sampler.Get() != nullptr, variant.name);
        AGFX_EXPECT_MSG(sampler.GetHandle() != 0, variant.name);

        // The handle must also survive the narrowing ShaderBindings applies when it packs the
        // handle into push constants.
        agfx::ez::ShaderBindings bindings;
        bindings.BindSampler(sampler);
        AGFX_EXPECT_MSG(*(const uint32_t*)bindings.Data() != 0u, variant.name);
    }
}
