/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-19 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "gpu_profiler.h"

#include <cstring>
#include <imgui.h>

void GpuProfiler::Init(agfxDevice* device, agfxCommandQueue* queue)
{
    agfxQueryPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.count = kMaxScopes * 2; // begin + end per scope
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        pools[i] = agfxQueryPoolCreate(device, queue, &poolCreateInfo);
    }
}

void GpuProfiler::Shutdown(agfxDevice* device)
{
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        agfxQueryPoolDestroy(device, pools[i]);
        pools[i] = nullptr;
    }
}

void GpuProfiler::BeginFrame(agfxDevice* device, uint32_t frameSlot)
{
    currentSlot = frameSlot;

    uint32_t previousScopeCount = scopeCountAtSlot[frameSlot];
    if (previousScopeCount > 0) {
        uint64_t timestampsNs[kMaxScopes * 2];
        agfxQueryPoolReadback(device, pools[frameSlot], 0, previousScopeCount * 2, timestampsNs);

        for (uint32_t i = 0; i < previousScopeCount; ++i) {
            uint64_t beginNs = timestampsNs[i * 2 + 0];
            uint64_t endNs = timestampsNs[i * 2 + 1];
            lastDurationsMs[i] = (float)((double)(endNs - beginNs) / 1000000.0);
        }
        lastScopeCount = previousScopeCount;

        float total = 0.0f;
        for (uint32_t i = 0; i < previousScopeCount; ++i) {
            total += lastDurationsMs[i];
        }
        totalHistoryMs[historyWriteIndex % kHistoryLength] = total;
        historyWriteIndex++;
    }

    scopeCount = 0;
}

void GpuProfiler::BeginScope(agfxCommandBuffer* cmd, const char* name)
{
    if (scopeCount >= kMaxScopes) {
        return;
    }

    uint32_t scopeIndex = scopeCount;
    strncpy(scopeNames[scopeIndex], name, sizeof(scopeNames[scopeIndex]) - 1);
    agfxCommandBufferWriteTimestamp(cmd, pools[currentSlot], scopeIndex * 2 + 0);
}

void GpuProfiler::EndScope(agfxCommandBuffer* cmd)
{
    if (scopeCount >= kMaxScopes) {
        return;
    }

    uint32_t scopeIndex = scopeCount;
    agfxCommandBufferWriteTimestamp(cmd, pools[currentSlot], scopeIndex * 2 + 1);
    scopeCount++;
    scopeCountAtSlot[currentSlot] = scopeCount;
}

void GpuProfiler::EndFrame(agfxCommandBuffer* cmd)
{
    if (scopeCount > 0) {
        agfxCommandBufferResolveQueryPool(cmd, pools[currentSlot], 0, scopeCount * 2);
    }
}

void GpuProfiler::DrawUI()
{
    ImGui::Begin("GPU Profiler");
    float total = 0.0f;
    for (uint32_t i = 0; i < lastScopeCount; ++i) {
        ImGui::Text("%-12s %.3f ms", scopeNames[i], lastDurationsMs[i]);
        total += lastDurationsMs[i];
    }
    ImGui::Separator();
    ImGui::Text("%-12s %.3f ms", "Total", total);
    ImGui::End();
}
