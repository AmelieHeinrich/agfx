/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <agfx/agfx.h>

class GpuProfiler {
public:
    static const uint32_t kMaxScopes = 16;
    static const uint32_t kFramesInFlight = 3; // must match agfx_demo_main's kFramesInFlight
    static const uint32_t kHistoryLength = 128;

    GpuProfiler() = default;
    GpuProfiler(const GpuProfiler&) = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;

    void Init(agfxDevice* device, agfxCommandQueue* queue);
    void Shutdown(agfxDevice* device);

    // Reads back the previous use of this frame slot's queries (if any) and resets the scope list
    // for the new frame. Call right after the caller has waited on this slot's fence.
    void BeginFrame(agfxDevice* device, uint32_t frameSlot);

    void BeginScope(agfxCommandBuffer* cmd, const char* name);
    void EndScope(agfxCommandBuffer* cmd);

    // Records the GPU-side resolve of this frame's queries. Call once, after the last
    // EndScope and before the command buffer is ended.
    void EndFrame(agfxCommandBuffer* cmd);

    void DrawUI();

private:
    agfxQueryPool* pools[kFramesInFlight] = {};
    uint32_t scopeCountAtSlot[kFramesInFlight] = {}; // scopes written into each slot's pool last time it was used
    uint32_t currentSlot = 0;

    char scopeNames[kMaxScopes][64] = {};
    uint32_t scopeCount = 0;

    float lastDurationsMs[kMaxScopes] = {};
    uint32_t lastScopeCount = 0;

    float totalHistoryMs[kHistoryLength] = {};
    uint32_t historyWriteIndex = 0;
};
