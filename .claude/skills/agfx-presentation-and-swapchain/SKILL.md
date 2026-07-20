---
name: agfx-presentation-and-swapchain
description: ALWAYS use when creating, resizing, or presenting through an agfxSwapChain in AGFX — window/layer handle setup, HDR toggling, vsync, resize handling, or back buffer acquire/present. Trigger for agfxSwapChainCreate/Destroy/Resize/Present, agfxSwapChainAcquireNextTexture, agfxSwapChainGetFormat, AGFX_RESOURCE_STATE_PRESENT, CAMetalLayer/HWND handle, "blank window", "swap chain out of date", HDR toggle, resize crash. Do NOT trigger for render pass/attachment authoring against the acquired back buffer texture — use agfx-render-targets-and-passes. Do NOT trigger for fence/frame-in-flight synchronization details — use agfx-synchronization.
---

# AGFX Presentation & Swap Chain

## Overview

`agfxSwapChain` is AGFX's cross-platform presentation object: on macOS it wraps a `CAMetalLayer*`, on Windows an `HWND`. The contract is deliberately simple — acquire a back buffer texture, render into it via a render pass, barrier it back to `AGFX_RESOURCE_STATE_PRESENT`, submit, then present. AGFX hides the Metal-4-vs-D3D12 differences in the queue-level present sequence (`waitForDrawable`/`signalDrawable` vs. `IDXGISwapChain::Present`) behind `agfxSwapChainPresent`.

Anything that changes the swap chain's underlying resources — resize, or an HDR toggle — requires the GPU to be fully idle first (see `agfx-synchronization` for `drainGPU`). AGFX does not do this drain internally; the caller must drain before resizing or recreating.

## Ownership

**Owns:**
- `agfxSwapChainCreateInfo` / `agfxSwapChainCreate` / `agfxSwapChainDestroy`
- `agfxSwapChainResize`
- `agfxSwapChainGetFormat`
- `agfxSwapChainAcquireNextTexture` / `agfxSwapChainPresent`
- HDR (`isHDR`) and vsync (`vsync`) swap chain configuration
- The native handle contract (`CAMetalLayer*` on macOS, `HWND` on Windows)
- `AGFX_RESOURCE_STATE_PRESENT` as the required pre/post state for the acquired texture

**Doesn't own:**
- Building the render pass that renders into the acquired back buffer texture → `agfx-render-targets-and-passes`
- Draining the GPU / fence semantics that must precede resize or swap chain recreation → `agfx-synchronization`
- Window creation, SDL/AppKit event handling, mouse/keyboard input — engine-level, not part of AGFX

## References

Read `agfx/agfx.h`'s `// Swap chain` section for authoritative struct/enum definitions. `agfx_demo/agfx_demo_main.cpp` is the full reference implementation: swap chain creation from an SDL window, per-frame acquire/present, resize handling, and HDR toggle.

## Design Patterns

### Creating the swap chain

```cpp
agfxSwapChainCreateInfo swapChainCreateInfo = {};
swapChainCreateInfo.queue = graphicsQueue;      // must be a graphics queue
swapChainCreateInfo.imageCount = 2;             // double buffering; 3 for triple buffering
swapChainCreateInfo.width = drawableWidth;
swapChainCreateInfo.height = drawableHeight;
swapChainCreateInfo.isHDR = false;
swapChainCreateInfo.vsync = true;
#if GAME_MAC
swapChainCreateInfo.handle = metalLayer;        // CAMetalLayer* from SDL_Metal_GetLayer or similar
#else
swapChainCreateInfo.handle = hwnd;
#endif
agfxSwapChain* swapChain = agfxSwapChainCreate(device, &swapChainCreateInfo);
```

Query the back buffer's actual pixel format with `agfxSwapChainGetFormat(swapChain)` rather than assuming a fixed format — it changes when `isHDR` changes, and any render target/pipeline built against the back buffer must use this queried format (`AGFX_TEXTURE_FORMAT_UNKNOWN` in the render target's `format` field lets it inherit automatically; see `agfx-render-targets-and-passes`).

### Per-frame acquire → render → present

```cpp
agfxTexture* backBuffer = agfxSwapChainAcquireNextTexture(swapChain);
// backBuffer starts in AGFX_RESOURCE_STATE_PRESENT
agfxCommandBufferTextureBarrier(cmd, backBuffer, AGFX_RESOURCE_STATE_PRESENT, AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, /*agglomerate=*/false);

// ... wrap backBuffer in an agfxRenderTarget, run a render pass, draw ...

agfxCommandBufferTextureBarrier(cmd, backBuffer, AGFX_RESOURCE_STATE_RENDER_TARGET, AGFX_RESOURCE_STATE_PRESENT, 0, 0, /*agglomerate=*/false);
agfxCommandBufferEnd(cmd);
agfxCommandQueueSubmit(queue, &cmd, 1);
agfxSwapChainPresent(swapChain);
```

Pass `agglomerate = false` specifically for these PRESENT↔RENDER_TARGET transitions around acquire/present (see `agfx-synchronization` for why: presentable drawables need no explicit hazard tracking on Metal, but D3D12 still needs the transition emitted unconditionally). `agfxSwapChainPresent` must be called only after the command buffer that renders to and barriers the back buffer has already been submitted via `agfxCommandQueueSubmit` — calling it before submit is a synchronization bug.

### Resize handling

Resizing is destructive to in-flight GPU work referencing swap chain resources, so always drain first:

```cpp
// On a window resize event:
drainGPU(); // see agfx-synchronization — signal + wait until the frame fence catches up
agfxSwapChainResize(device, swapChain, newWidth, newHeight);
renderer.Resize(device, newWidth, newHeight); // resize any render targets sized to match the window
```

### HDR toggle

AGFX has no in-place HDR reconfiguration API — toggling HDR means destroying and recreating the swap chain, then recreating anything downstream whose pipeline format depended on the old back buffer format:

```cpp
drainGPU();
agfxSwapChainDestroy(device, swapChain);
swapChainCreateInfo.isHDR = wantHDR;
swapChain = agfxSwapChainCreate(device, &swapChainCreateInfo);
renderer.RecreateTonemapPipeline(device, agfxSwapChainGetFormat(swapChain));
imguiBackend.RecreatePipeline(device, agfxSwapChainGetFormat(swapChain));
```

Skipping the pipeline recreation step is the most common HDR-toggle bug: the render pass will still bind fine, but color-attachment format mismatch between the pipeline and the actual back buffer produces validation errors on D3D12 and garbage/black output on Metal.
