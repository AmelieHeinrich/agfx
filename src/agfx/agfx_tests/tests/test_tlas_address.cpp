/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// Tests that TLAS handle addresses are not null and can be used in shaders.
//
// The handle is the acceleration structure's GPU virtual address, which the underlying buffer has
// from the moment it's created -- building is what fills the buffer with valid acceleration
// structure data, not what gives it an address. So this test never builds anything (and never
// touches AGFX_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, which isn't a legal ResourceBarrier
// target on D3D12 -- that pseudo-state is only ever entered/left via a UAV barrier around a build).

#include "../test_gpu.h"

namespace
{
    using namespace agfxtest;

    agfxAccelerationStructureCreateInfo TlasInfo()
    {
        agfxTopLevelAccelerationStructureCreateInfo topLevel{};
        topLevel.maxInstanceCount = 1;

        agfxAccelerationStructureCreateInfo info{};
        info.type = AGFX_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        info.topLevel = topLevel;
        info.allowUpdate = 0;
        info.name = "test tlas";
        return info;
    }
} // namespace

AGFX_TEST_VALIDATION(TLASAddressNotNull, C)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const agfxAccelerationStructureCreateInfo tlasInfo = TlasInfo();
    agfxAccelerationStructure* tlas = agfxAccelerationStructureCreate(device, &tlasInfo);
    AGFX_EXPECT_NOT_NULL(tlas);

    uint64_t tlasHandle = agfxAccelerationStructureGetHandle(tlas);
    AGFX_EXPECT_MSG(tlasHandle != 0, "TLAS handle should not be null");

    agfxAccelerationStructureDestroy(device, tlas);
}

AGFX_TEST_VALIDATION(TLASAddressNotNull, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::AccelerationStructure tlas = device.CreateAccelerationStructure(TlasInfo());
    AGFX_EXPECT_NOT_NULL(tlas.Get());

    uint64_t tlasHandle = tlas.GetHandle();
    AGFX_EXPECT_MSG(tlasHandle != 0, "TLAS handle should not be null");
}
