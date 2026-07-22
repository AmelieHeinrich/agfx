//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: wave (subgroup) intrinsics -- WaveGetLaneCount, WaveGetLaneIndex, WaveActiveSum,
// WaveActiveMax, WaveIsFirstLane, and WaveReadLaneFirst.
//
// Wave width is not a portable constant: Metal SIMD groups are commonly 32 lanes on Apple GPUs,
// D3D12 warps are commonly 32 on NVIDIA and 64 on AMD, and AGFX exposes no way to pin it. So this
// test cannot check the *count* the wave ops see, only that they are internally consistent with each
// other and with the launch geometry -- which is what actually pins the intrinsics down without
// baking in a wave size the test would break on a different GPU.
//
// One thread group of 64 lanes is dispatched; on any wave width that divides 64 evenly (32, 64, 16,
// ...) the group is covered by a whole number of waves, so WaveActiveSum(1) over the group's threads
// sums to the group size in multiples of the wave size -- not asserted directly, since the group can
// itself span multiple waves, but each wave's own partial sum divides the group evenly, which is
// what ExpectedPattern checks per-thread against WaveGetLaneCount reported by that same thread.

#include "data/shaders/agfx.h"

struct WaveOpsPushConstants
{
    ResourceHandle destination;       // AGFXRWStructuredBuffer<uint4>, one element per thread.
    ResourceHandle firstLaneIDBuffer; // AGFXRWStructuredBuffer<uint>, one element per thread.
    uint threadCount;
    uint padding0;
};

AGFX_PUSH_CONSTANTS(WaveOpsPushConstants, g_Constants);

[WaveSize(32)]
[numthreads(64, 1, 1)]
void main_wave_ops_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Constants.threadCount) {
        return;
    }

    AGFXRWStructuredBuffer<uint4> dst = AGFXRWStructuredBuffer<uint4>::Create(g_Constants.destination);
    AGFXRWStructuredBuffer<uint> firstLaneIDs = AGFXRWStructuredBuffer<uint>::Create(g_Constants.firstLaneIDBuffer);

    const uint laneCount = WaveGetLaneCount();
    const uint laneIndex = WaveGetLaneIndex();

    // Every lane contributes 1: the wave's own size, computed the hard way. Must equal laneCount.
    const uint activeSum = WaveActiveSum(1u);

    // Every lane contributes its own lane index: the max across the wave is laneCount - 1.
    const uint maxLaneIndex = WaveActiveMax(laneIndex);

    // Packed so three checks travel through one write per thread: x = lane count as this lane
    // computed it, y = the active-sum cross-check, z = the active-max cross-check (+1, so a zero
    // entry is distinguishable from "never written"), w = 1 if this lane agrees with
    // WaveIsFirstLane() about being lane 0, else 0.
    const uint isFirstMatches = (WaveIsFirstLane() == (laneIndex == 0)) ? 1u : 0u;
    dst.Store(id.x, uint4(laneCount, activeSum, maxLaneIndex + 1u, isFirstMatches));

    // WaveReadLaneFirst broadcasts lane 0's value; every lane sharing a wave must read back the
    // same dispatch-thread-ID for it, since lane 0 is one fixed lane within the wave. Written
    // separately from the packed record above so the host can group threads by wave membership
    // purely from this value, without having to first decode which wave each thread belonged to.
    firstLaneIDs.Store(id.x, WaveReadLaneFirst(id.x));
}
