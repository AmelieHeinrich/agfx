---
name: agfx-render-targets-and-passes
description: ALWAYS use when creating agfxRenderTarget or agfxRenderPass objects in AGFX, or wiring up color/depth attachments, load/store ops, viewport/scissor, or draw calls within a render pass. Trigger for agfxRenderTargetCreate, agfxRenderPassBegin/End, agfxRenderPassAttachment, AGFX_LOAD_OPERATION_*, AGFX_STORE_OPERATION_*, agfxRenderPassSetViewport/SetScissor, agfxRenderPassDraw/DrawIndexed/DrawMesh, GBuffer/HDR/depth target setup, "wrong render target format", "render pass validation error". Do NOT trigger for swap chain acquire/present or CAMetalLayer-equivalent presentation — use agfx-presentation-and-swapchain. Do NOT trigger for fences, frames-in-flight, or barrier agglomeration semantics — use agfx-synchronization.
---

# AGFX Render Targets & Render Passes

## Overview

AGFX render targets and render passes mirror the way Metal 4 and D3D12 both express "attachment then pass" but AGFX flattens both backends behind one API. An `agfxRenderTarget` is a thin, cheap-to-create wrapper around a texture (or texture subresource) that tells the backend how to attach it — as a color or depth-stencil target, at a given mip/layer. An `agfxRenderPass` is the actual encoding scope: it groups up to 8 color attachments and one optional depth attachment, and every draw call in AGFX must happen inside one.

Render targets are intentionally lightweight: create one right before `agfxRenderPassBegin` and destroy it right after `agfxRenderPassEnd`. Don't cache them across frames — cache the underlying `agfxTexture` instead and re-wrap it each pass (see `deferred_renderer.cpp` for the GBuffer/HDR/depth targets recreated only on resize, but their `agfxRenderTarget` wrappers created per-pass).

## Ownership

**Owns:**
- `agfxRenderTargetCreateInfo` / `agfxRenderTargetCreate` / `agfxRenderTargetDestroy`
- `agfxRenderPassCreateInfo` / `agfxRenderPassBegin` / `agfxRenderPassEnd`
- Color/depth attachment wiring, `agfxLoadOp` / `agfxStoreOp`, clear colors
- Viewport/scissor state (`agfxRenderPassSetViewport`, `agfxRenderPassSetScissor`)
- Draw calls within a pass (`agfxRenderPassDraw`, `agfxRenderPassDrawIndexed`, `agfxRenderPassDrawMesh`) and push constants
- Render pipeline attachment-format matching (`colorFormats[]` / `depthFormat` in `agfxRenderPipelineCreateInfo` must match the pass it's used with)

**Doesn't own:**
- Swap chain back buffer acquisition/present, HDR toggling → `agfx-presentation-and-swapchain`
- Resource state barrier semantics, agglomeration, fences, frame pacing → `agfx-synchronization`
- Compute/copy passes (`agfxComputePass*`) — those are a separate pass type for compute dispatch and copies, not covered here
- GPU-driven replay inside a pass (`agfxRenderPassExecuteIndirectBundle`) and everything feeding it → `agfx-mdi`; the pass setup around it is ordinary and covered here, but the bundle itself, its barriers, and the `supportsIndirect` pipeline flag are not
- Mesh-shader pipeline setup behind `agfxRenderPassDrawMesh` (`meshShader`/`taskShader`, the reflected group sizes) → `agfx-writing-bindless-shaders`

## References

Read `agfx/agfx.h` before writing render-target/pass code — it's the single source of truth for every struct field and enum value in this API (search for `// Render target` and `// Render pass` section markers). `agfx_demo/deferred_renderer.cpp` and `agfx_demo/agfx_demo_main.cpp` show real per-frame usage: GBuffer, SSAO, shadow map, lighting, and backbuffer passes.

## Design Patterns

### Creating a render target from a texture

```cpp
agfxRenderTargetCreateInfo rtCreateInfo = {};
rtCreateInfo.texture = colorTexture;                 // must have AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT
rtCreateInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;    // inherit the texture's own format
rtCreateInfo.mipLevel = 0;
rtCreateInfo.arrayLayer = 0;
rtCreateInfo.isDepth = false;
agfxRenderTarget* colorRT = agfxRenderTargetCreate(device, &rtCreateInfo);

// Depth target
agfxRenderTargetCreateInfo depthRtCreateInfo = {};
depthRtCreateInfo.texture = depthTexture;             // must have AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT
depthRtCreateInfo.isDepth = true;
agfxRenderTarget* depthRT = agfxRenderTargetCreate(device, &depthRtCreateInfo);
```

`AGFX_TEXTURE_FORMAT_UNKNOWN` is the correct choice whenever the render target wraps a texture whose format is already fixed (e.g. a swap chain back buffer, queried via `agfxSwapChainGetFormat`) — don't hardcode the format in that case, it can silently mismatch on an HDR toggle.

### Required state before the pass begins

Every color attachment's underlying texture must already be in `AGFX_RESOURCE_STATE_RENDER_TARGET`, and every depth attachment's texture in `AGFX_RESOURCE_STATE_DEPTH_WRITE`, **before** calling `agfxRenderPassBegin`. Barrier into that state first:

```cpp
agfxCommandBufferTextureBarrier(cmd, colorTexture, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, /*agglomerate=*/true);
```

See `agfx-synchronization` for the `agglomerate` flag's Metal-vs-D3D12 semantics — get this wrong and you'll get correct D3D12 output with garbage or validation errors on Metal.

### Beginning a render pass with multiple attachments

```cpp
agfxRenderPassCreateInfo passInfo = {};
passInfo.colorAttachmentCount = 1;
passInfo.colorAttachments[0].renderTarget = colorRT;
passInfo.colorAttachments[0].loadOp = AGFX_LOAD_OPERATION_CLEAR;
passInfo.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
passInfo.colorAttachments[0].clearColor[0] = 0.05f;
passInfo.colorAttachments[0].clearColor[1] = 0.05f;
passInfo.colorAttachments[0].clearColor[2] = 0.1f;
passInfo.colorAttachments[0].clearColor[3] = 1.0f;

passInfo.hasDepthAttachment = true;
passInfo.depthAttachment.renderTarget = depthRT;
passInfo.depthAttachment.loadOp = AGFX_LOAD_OPERATION_CLEAR;
passInfo.depthAttachment.storeOp = AGFX_STORE_OPERATION_DONT_CARE; // transient depth, not read afterwards

passInfo.width = width;
passInfo.height = height;
passInfo.name = "GBuffer";

agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
agfxRenderPassSetViewport(pass, 0, 0, (float)width, (float)height, 0.0f, 1.0f);
agfxRenderPassSetScissor(pass, 0, 0, width, height);
agfxRenderPassSetPipeline(pass, pipeline);
agfxRenderPassPushConstants(pass, &pushData, sizeof(pushData));
agfxRenderPassDrawIndexed(pass, indexBuffer, indexCount, 1, 0, 0, 0);
agfxRenderPassEnd(pass);

agfxRenderTargetDestroy(device, colorRT);
agfxRenderTargetDestroy(device, depthRT);
```

### Load/store op choices

- `AGFX_LOAD_OPERATION_CLEAR` + `AGFX_STORE_OPERATION_STORE` — standard color/depth attachment that's cleared then read later (e.g. GBuffer albedo, depth used for SSAO)
- `AGFX_LOAD_OPERATION_DONT_CARE` + `AGFX_STORE_OPERATION_STORE` — every pixel will be fully overwritten this pass and the result is needed afterwards; skip the clear cost
- `AGFX_LOAD_OPERATION_LOAD` + `AGFX_STORE_OPERATION_STORE` — accumulating into an attachment across multiple passes without a full clear (e.g. compositing UI on top of tonemapped output)
- Any load op + `AGFX_STORE_OPERATION_DONT_CARE` — transient attachment discarded at pass end (e.g. an MSAA resolve source, or depth that's only used within the pass and never sampled afterward)

### Pipeline/pass format matching

`agfxRenderPipelineCreateInfo::colorFormats[]`/`colorAttachmentCount` and `depthFormat` must match the actual formats of the render pass's attachments the pipeline is bound in. When a pass renders to the swap chain back buffer and the app supports toggling HDR (see `agfx-presentation-and-swapchain`), the pipeline bound in that pass must be recreated whenever the back buffer format changes — `agfx_demo_main.cpp`'s `RecreateTonemapPipeline` on the HDR toggle is the reference pattern.

### Mesh-shading vs classic draws

A pipeline is either classic (`vertexShader`/`fragmentShader`, drawn with `agfxRenderPassDraw`/`agfxRenderPassDrawIndexed`) or mesh-shading (`meshShader`, optional `taskShader`, drawn with `agfxRenderPassDrawMesh`). Don't mix: calling `agfxRenderPassDrawMesh` with a classic pipeline bound (or vice versa) is undefined — match the draw call to how the bound pipeline was created.
