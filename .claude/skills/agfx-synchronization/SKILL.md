---
name: agfx-synchronization
description: ALWAYS use when working with AGFX fences, frames-in-flight pacing, queue signal/wait, resource-state barriers, or the barrier "agglomerate" flag. Trigger for agfxFenceCreate/Wait/Signal/GetCompletedValue, agfxCommandQueueSignal/Wait, agfxCommandBufferTextureBarrier/BufferBarrier, agfxComputePassTextureUAVBarrier/BufferUAVBarrier, agfxResourceState transitions, "GPU/CPU sync", "drain the GPU", "frames in flight", "hazard tracking", agglomerate flag, resize/HDR-toggle-before-recreate ordering. Do NOT trigger for swap chain acquire/present mechanics themselves — use agfx-presentation-and-swapchain. Do NOT trigger for render pass attachment authoring — use agfx-render-targets-and-passes.
---

# AGFX Synchronization: Fences, Barriers & Frame Pacing

## Overview

AGFX exposes one fence type (`agfxFence`) used identically for CPU↔GPU and GPU↔GPU synchronization, and two barrier APIs: **state transitions** (`agfxCommandBufferTextureBarrier`/`agfxCommandBufferBufferBarrier`) and **UAV hazard barriers** (`agfxComputePassTextureUAVBarrier`/`BufferUAVBarrier`). The tricky part is that AGFX's Metal backend has no per-resource hazard tracking the way D3D12 does — barriers on Metal are stage-based and get merged into a single pending barrier, controlled by the `agglomerate` parameter. Getting `agglomerate` wrong is the single most common cross-platform bug in AGFX code: it looks completely correct on D3D12 (which ignores the flag and always transitions immediately) while producing missing barriers or corrupted output on Metal.

The second thing to internalize is **barrier scope**. Metal 4 distinguishes barriers that order work against *prior encoders* from barriers that order work *within the current encoder*, and they are not interchangeable — using the wrong one produces a barrier that silently orders nothing. The two AGFX barrier APIs map onto that split: **state transitions are cross-encoder, UAV barriers are intra-encoder.** Pick by where the producer and consumer actually live, not by which call is more convenient.

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

- **Metal**: no per-resource hazard tracking exists. `agglomerate = true` merges this transition's producer/consumer stages into a single pending barrier, flushed automatically at the start of the next `agfxComputePassBegin`/`agfxRenderPassBegin` — or immediately, if a pass is already open when the call is made. `agglomerate = false` is a no-op on Metal — the barrier is **silently dropped**, not emitted some other way.
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

Both `agfxRenderPassBegin` and `agfxComputePassBegin` flush any pending agglomerated barriers automatically — you don't need (and can't) manually flush them. Calling a transition barrier *while a pass is already open* also works: the backend flushes it inline against the open encoder. This is what makes a mid-pass transition (e.g. a copy into a buffer, then a dispatch that reads it, both inside one compute pass) actually order correctly.

**Transitions out of a read-only state are real barriers.** A `PIXEL_SHADER_RESOURCE → UNORDERED_ACCESS` or `INDIRECT_ARGUMENT → UNORDERED_ACCESS` transition is a write-after-read hazard: the readers must finish before the new writer starts. The backend derives the "wait for" stages from whoever *read* the resource when the old state writes nothing (and symmetrically, the "must wait" stages from whoever *writes* when the new state only writes, e.g. `RENDER_TARGET`). Don't skip these on the assumption that "nothing wrote it, so there's nothing to synchronize" — that reasoning is what leaves a GPU-driven pass overwriting buffers the previous frame is still reading.

### UAV hazard barriers — **between two dispatches in the same pass**

`agfxComputePassTextureUAVBarrier`/`BufferUAVBarrier` order dispatches **within one open encoder**. They are separate from the state-transition barriers above and are needed even when the resource state itself doesn't change:

```cpp
agfxComputePass* pass = agfxComputePassBegin(cmd, "Cull + Prepare");
// dispatch A writes the buffer
agfxComputePassDispatch(pass, groupsX, groupsY, 1);
agfxComputePassBufferUAVBarrier(pass, buffer);   // A completes before B reads it
// dispatch B reads what A wrote
agfxComputePassDispatch(pass, groupsX2, groupsY2, 1);
agfxComputePassEnd(pass);
```

**A UAV barrier is only meaningful between two dispatches in the same pass.** Placing one at the *end* of a pass, expecting it to order the *next* pass's work, does nothing — it is an intra-encoder barrier and there is no subsequent work in that encoder for it to apply to. Cross-pass hazards are the state-transition barriers' job:

```cpp
// WRONG: orders nothing. The consumer is in the next encoder.
agfxComputePassDispatch(pass, gx, gy, 1);
agfxComputePassTextureUAVBarrier(pass, texture);
agfxComputePassEnd(pass);

// RIGHT: a state transition between the passes, with agglomerate = true.
agfxComputePassDispatch(pass, gx, gy, 1);
agfxComputePassEnd(pass);
agfxCommandBufferTextureBarrier(cmd, texture,
    AGFX_RESOURCE_STATE_UNORDERED_ACCESS, AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, mip, 0, /*agglomerate=*/true);
```

A mip-chain generator that runs one pass per mip is the canonical example of the second shape: each mip is written by one pass's dispatch and read by the next pass's, so the whole chain rides on agglomerated transitions, not UAV barriers. See `agfx_demo/agfx_mipgen.cpp`.

### Common mistakes

- **Passing `agglomerate = false` for an ordinary transition.** It is documented as a no-op on Metal and the barrier vanishes entirely. Reserve `false` for the swap chain PRESENT↔RENDER_TARGET pair only. A whole dependency chain built from `false` barriers runs correctly on D3D12 and has *zero* synchronization on Metal.
- **Using a UAV barrier to order across passes.** See above — it orders nothing outside its own encoder.
- **Omitting a write-after-read transition** because the old state "doesn't write anything". Read→write is a real hazard; this is how a shared resource gets clobbered by the next frame while the current one still reads it.
- **Assuming a resource shared across frames in flight is safe because each frame has its own command buffer.** Command buffers on a queue are not implicitly serialized against each other on Metal 4. A resource rebuilt every frame and consumed in the same frame still needs a barrier at the top of the rebuild, or per-frame-in-flight copies.
- **Reusing or decreasing a fence value in `agfxCommandQueueSignal`.** Values must be strictly monotonic; a stale value lets a later wait pass immediately.

### Subresource barrier granularity

`agfxCommandBufferTextureBarrier` takes explicit `mip`/`layer` parameters, or `AGFX_SUBRESOURCE_ALL_MIPS`/`AGFX_SUBRESOURCE_ALL_LAYERS` to target every mip/layer at once. Use per-subresource barriers (not "all") when only a specific mip is being written (e.g. mip-chain generation writing one mip at a time from the previous), otherwise other subresources get unnecessarily serialized.
