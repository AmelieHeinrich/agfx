---
name: agfx-synchronization
description: ALWAYS use when working with AGFX fences, frames-in-flight pacing, queue signal/wait, resource-state barriers, or the barrier "agglomerate" flag. Trigger for agfxFenceCreate/Wait/Signal/GetCompletedValue, agfxCommandQueueSignal/Wait, agfxCommandBufferTextureBarrier/BufferBarrier, agfxComputePassTextureUAVBarrier/BufferUAVBarrier, agfxResourceState transitions, "GPU/CPU sync", "drain the GPU", "frames in flight", "hazard tracking", agglomerate flag, resize/HDR-toggle-before-recreate ordering. Do NOT trigger for swap chain acquire/present mechanics themselves — use agfx-presentation-and-swapchain. Do NOT trigger for render pass attachment authoring — use agfx-render-targets-and-passes.
---

# AGFX Synchronization: Fences, Barriers & Frame Pacing

## Overview

AGFX exposes one fence type (`agfxFence`) used identically for CPU↔GPU and GPU↔GPU synchronization, and one barrier API (`agfxCommandBufferTextureBarrier`/`agfxCommandBufferBufferBarrier`) used for resource-state transitions. The tricky part is that AGFX's Metal backend has no per-resource hazard tracking the way D3D12 does — barriers on Metal are merged into a single pending barrier flushed automatically at the next pass boundary, controlled by the `agglomerate` parameter. Getting `agglomerate` wrong is the single most common cross-platform bug in AGFX code: it can look completely correct on D3D12 (which ignores the flag and always transitions immediately) while producing missing barriers or corrupted output on Metal.

## Ownership

**Owns:**
- `agfxFenceCreate`/`Destroy`/`Wait`/`Signal`/`GetCompletedValue`
- `agfxCommandQueueSignal`/`agfxCommandQueueWait` (GPU-side queue signal/wait)
- Frames-in-flight pacing pattern (per-slot fence values, `kFramesInFlight`)
- `agfxCommandBufferTextureBarrier`/`agfxCommandBufferBufferBarrier` and the `agglomerate` flag's Metal-vs-D3D12 semantics
- `agfxComputePassTextureUAVBarrier`/`agfxComputePassBufferUAVBarrier` (UAV read/write hazard barriers within a compute pass)
- The "drain the GPU before destroying/recreating a resource still referenced by in-flight work" rule

**Doesn't own:**
- Swap chain acquire/present call sequence itself → `agfx-presentation-and-swapchain`
- Render pass attachment state requirements (which states attachments must be in before `agfxRenderPassBegin`) → `agfx-render-targets-and-passes` (though the barriers to reach those states are documented here)

## References

Read `agfx/agfx.h`'s `agfxCommandBufferTextureBarrier` doc comment carefully — it's the authoritative explanation of `agglomerate`. `agfx_demo/agfx_demo_main.cpp`'s `drainGPU` lambda and per-frame fence-slot logic is the reference frame-pacing implementation; `deferred_renderer.cpp` shows barrier usage across a full G-buffer/SSAO/shadow/lighting pipeline.

## Design Patterns

### Frames-in-flight pacing

Don't block on every frame — only block when about to reuse a command-buffer slot whose GPU work may still be in flight:

```cpp
agfxCommandBuffer* commandBuffers[kFramesInFlight];
agfxFence* frameFence = agfxFenceCreate(device);
uint64_t fenceValue = 0;
uint64_t slotFenceValues[kFramesInFlight] = {};

// Each frame:
uint32_t frameSlot = (uint32_t)(frameIndex % kFramesInFlight);
agfxFenceWait(frameFence, slotFenceValues[frameSlot], UINT64_MAX); // block only if that slot hasn't finished
agfxCommandBufferReset(commandBuffers[frameSlot]);
agfxCommandBufferBegin(commandBuffers[frameSlot]);
// ... record + submit ...
agfxCommandQueueSubmit(queue, &commandBuffers[frameSlot], 1);
agfxSwapChainPresent(swapChain);

slotFenceValues[frameSlot] = ++fenceValue;
agfxCommandQueueSignal(queue, frameFence, fenceValue); // GPU signals this value once this frame's work completes
frameIndex++;
```

Every `agfxCommandQueueSignal` call must use a fresh monotonically increasing value — reusing a value (or signaling out of order) means a later wait can be satisfied by an earlier, unrelated signal.

### Draining the GPU before destructive operations

Any operation that destroys or recreates a resource an in-flight (submitted but not yet completed) command buffer might still reference — swap chain resize, HDR toggle, shared G-buffer/depth target resize — must fully drain first:

```cpp
auto drainGPU = [&]() {
    agfxCommandQueueSignal(queue, frameFence, ++fenceValue);
    agfxFenceWait(frameFence, fenceValue, UINT64_MAX);
};

// Before: swap chain resize, swap chain recreation (HDR toggle), shared render target resize
drainGPU();
agfxSwapChainResize(device, swapChain, newWidth, newHeight);
```

Skipping this drain is a use-after-free/GPU-fault waiting to happen: the old resource can be destroyed while a previously-submitted, still-executing command buffer references it.

### The `agglomerate` barrier flag

`agfxCommandBufferTextureBarrier`/`agfxCommandBufferBufferBarrier` take an `agglomerate` bool with backend-divergent meaning:

- **Metal**: no per-resource hazard tracking exists. `agglomerate = true` merges this transition's producer/consumer stages into a single pending barrier, automatically flushed at the start of the next `agfxComputePassBegin` or `agfxRenderPassBegin`. `agglomerate = false` is a no-op on Metal — the barrier is effectively skipped.
- **D3D12**: the flag is ignored entirely; the transition is always emitted immediately regardless of its value.

Rule of thumb:
- **Ordinary resource transitions** (e.g. depth texture from `PIXEL_SHADER_RESOURCE` to `DEPTH_WRITE` before a shadow pass) → pass `true`, so Metal actually tracks the hazard.
- **Swap chain PRESENT↔RENDER_TARGET transitions around acquire/present** → pass `false`. Presentable drawables need no explicit barrier on Metal (the display server/compositor handles it), but D3D12 still requires the transition, which happens unconditionally regardless of the flag — see `agfx-presentation-and-swapchain`.

```cpp
// Ordinary transition: track the hazard on Metal
agfxCommandBufferTextureBarrier(cmd, depthTexture, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_DEPTH_WRITE, 0, 0, /*agglomerate=*/true);

// Present-adjacent transition: no-op on Metal, unconditional on D3D12
agfxCommandBufferTextureBarrier(cmd, backBuffer, AGFX_RESOURCE_STATE_PRESENT, AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, /*agglomerate=*/false);
```

Both `agfxRenderPassBegin` and `agfxComputePassBegin` flush any pending agglomerated barriers automatically — you don't need (and can't) manually flush them.

### UAV hazard barriers within a compute pass

Between two dispatches that both read/write the same UAV texture or buffer within one `agfxComputePass`, insert an explicit UAV barrier — these are separate from the state-transition barriers above and needed even when the resource state itself doesn't change:

```cpp
agfxComputePass* pass = agfxComputePassBegin(cmd, "Mipgen");
agfxComputePassSetPipeline(pass, mipgenPipeline);
agfxComputePassDispatch(pass, groupsX, groupsY, 1);
agfxComputePassTextureUAVBarrier(pass, mipTexture); // ensure the write above completes before the next dispatch reads it
agfxComputePassDispatch(pass, groupsX2, groupsY2, 1);
agfxComputePassEnd(pass);
```

### Subresource barrier granularity

`agfxCommandBufferTextureBarrier` takes explicit `mip`/`layer` parameters, or `AGFX_SUBRESOURCE_ALL_MIPS`/`AGFX_SUBRESOURCE_ALL_LAYERS` to target every mip/layer at once. Use per-subresource barriers (not "all") when only a specific mip is being written (e.g. mip-chain generation writing one mip at a time from the previous), otherwise other subresources get unnecessarily serialized.
