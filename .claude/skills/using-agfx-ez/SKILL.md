---
name: using-agfx-ez
description: ALWAYS use when writing or modifying code against agfx::ez (agfx_ez.hpp) — the D3D11/OpenGL-style immediate-context convenience layer built on top of raw AGFX. Trigger on agfx::ez::Context, agfx::ez::Texture2D, agfx::ez::Buffer, agfx::ez::ShaderBindings, agfx::ez::PipelineDesc, Context::BeginFrame/EndFrame, Context::SetRenderTargets/SetPipeline/Draw, Context::CreateTexture2D/CreateVertexBuffer/CreateStructuredBuffer, Context::AllocateConstants, Context::TransitionTexture/TransitionBuffer/TransitionIndirectBundle, Context::CreateIndirectBundle/ExecuteIndirectBundle, agfx::ez::IndirectBundle, agfx::ez::AccelerationStructure, "agfx ez", "immediate mode AGFX", "simple AGFX API", or a decision about whether a new AGFX-based tool/prototype/small app should use raw AGFX or the ez layer. Do NOT trigger for raw agfx.h/agfx.hpp API usage with no agfx_ez.hpp involvement — use the relevant agfx-* subsystem skill instead (agfx-presentation-and-swapchain, agfx-render-targets-and-passes, agfx-synchronization, agfx-writing-bindless-shaders, agfx-mdi, agfx-raytracing). Do NOT trigger for porting an existing engine from another graphics API — use the relevant agfx-porting-from-* skill, which will itself decide whether to recommend ez.
---

# Using agfx::ez

## Overview

`agfx::ez` (`src/agfx/agfx/agfx_ez.hpp`) is a header-only, D3D11/OpenGL-style immediate-context layer built entirely on top of raw AGFX (`agfx.hpp`/`agfx.h`). It owns its own frame loop, caches render pipelines and resource views, does one-call texture/buffer creation with synchronous upload, provides a ring-buffered per-frame dynamic constant allocator, and a 128-byte bindless push-constant builder (`ShaderBindings`). The entire point is to remove the ceremony raw AGFX requires for simple cases (frame pacing, render-target/pass boilerplate, pipeline caching, upload staging) while still producing real AGFX resources underneath — `Texture2D::Raw()`/`Buffer::Raw()`/`Context::GetDevice()` give an escape hatch back to raw AGFX whenever ez's simplifications don't fit.

**This is convenience, not automatic hazard tracking.** AGFX is fully bindless — a shader can read a resource through a handle at any point, with no host-visible hook to observe that. `Context::TransitionTexture`/`TransitionBuffer` (and the automatic transitions inside `SetRenderTargets`) only track the last state *the ez layer itself* moved a resource into via its own calls. They do **not**: see inside shader bindless reads, track resources created directly through raw `agfx.hpp`/`agfx.h` (mixing raw and ez creation for the same resource silently defeats tracking for it), or infer intra-pass UAV read/write hazards (use the re-exposed `TextureUAVBarrier`/`BufferUAVBarrier` on `agfx::ComputePass` for that, same as raw AGFX — see `agfx-synchronization`). Keep this scope in mind before assuming ez has "solved" synchronization.

Tracking *is* subresource-granular: `ResourceStateTracker` stores one state while every (mip, layer) agrees and splits into per-subresource states on the first subresource transition, re-collapsing when they agree again. So `TransitionTexture(tex, state, mip, layer)` and per-mip render targets are expressible — mip-chain generation works, and whole-resource transitions still cost a single barrier in the common case.

**Coverage today: ez does essentially everything raw AGFX does**, with these known exclusions — multi-queue (one graphics queue, one command buffer per frame slot), per-attachment blend state (`PipelineDesc` blend is uniform across color attachments), compute-pass sugar (open a raw `agfx::ComputePass` via `GetCurrentCommandBuffer()`), and async/batched upload (creation uploads block). Everything else — all four texture types, mip/layer-addressed render targets and views, copies, mesh shading, indirect bundles, ray tracing, headless operation — has an ez path. Don't recommend dropping to raw AGFX for anything outside that exclusion list without checking the header first.

## Ownership

**Owns:**
- `agfx::ez::Context`: construction (`ContextCreateInfo`), the `BeginFrame`/`EndFrame` (or RAII `Frame` guard) loop, resize/HDR toggle, immediate-mode render calls, one-call resource creation, per-frame dynamic constants, the explicit-transition escape hatch, and raw-AGFX object access
- `agfx::ez::Texture2D`/`Texture2DArray`/`Texture3D`/`TextureCube` (all sharing `detail::TextureBase`) and `agfx::ez::Buffer`: lazily-created cached views (SRV/UAV keyed on the full mip/layer subrange, RTV keyed on (mip, layer), and per-`agfxBufferViewType` buffer views) and their subresource state tracker
- `RenderTargetBinding`: naming a mip and array layer per attachment in `SetRenderTargets`
- Headless contexts (`windowHandle == nullptr`), and which entry points are unavailable in one
- `agfx::ez::ShaderBindings`: the 128-byte push-constant builder and its `BindTexture`/`BindBuffer`/`BindSampler`/`Write` calls
- `agfx::ez::PipelineDesc` + the internal pipeline cache: the simplified render-pipeline description and when a new `agfx::RenderPipeline` gets built vs. reused
- `agfx::ez::IndirectBundle`/`AccelerationStructure` **as ez objects**: their state tracking, `Context::CreateIndirectBundle`/`ExecuteIndirectBundle`/`TransitionIndirectBundle`, and the raw-compute-pass escape hatch they require
- The explicit scope limits of ez's barrier tracking (see Overview) — when to reach for raw AGFX barriers instead
- When to recommend `agfx::ez` over raw AGFX for a new tool/prototype, and the reverse (when an existing ez-based app has outgrown it)

**Doesn't own:**
- Raw AGFX API detail for calls ez wraps — the four `agfx-*` subsystem skills own the underlying `agfxRenderPassBegin`/`agfxCommandBufferTextureBarrier`/etc. semantics that `Context` methods are thin sugar over
- HLSL shader authoring itself (`AGFX_PUSH_CONSTANTS`, `AGFXTexture2D`, etc.) — `agfx-writing-bindless-shaders`; ez's `ShaderBindings` only builds the *host-side* constant blob, it doesn't change what the shader itself looks like
- GPU-driven submission semantics themselves (bundle layout, prepare/execute ordering, `supportsIndirect`, the HLSL append helpers) — **agfx-mdi**; ez only sugars bundle creation, the lockstep transition, and the replay call
- Acceleration-structure build semantics — **agfx-raytracing**; ez only sugars the one-call synchronous build
- Deciding whether a *port* from another API should target ez or raw AGFX — each `agfx-porting-from-*` skill makes that call itself and links back here for the mechanics once decided

## References

`agfx/agfx_ez.hpp` is the entire implementation and is short enough to read in full before writing ez code — the top-of-file comment is the authoritative statement of what barrier tracking does and doesn't cover. `agfx_ez_demo/agfx_ez_demo_main.cpp` is a complete small program against this API — the reference shape for a new ez-based tool. Compare against `agfx_demo/` (raw AGFX) when deciding whether a given piece of code has outgrown ez.

## Design Patterns

### Setting up a Context

```cpp
agfx::ez::ContextCreateInfo info{};
info.deviceInfo.enableValidation = true; // or leave device fields default and set existingDevice instead
info.windowHandle = metalLayerOrHwnd;    // CAMetalLayer* on macOS, HWND on Windows
info.width = width;
info.height = height;
info.vsync = true;
info.hdr = false;
info.framesInFlight = 3;
agfx::ez::Context ctx(info);
```

Pass an existing `agfx::Device*` via `existingDevice` when a larger app already owns device creation elsewhere (e.g. an editor tool embedding an ez-based viewport panel inside a raw-AGFX-driven app) — `Context` won't destroy a device it doesn't own.

### The frame loop

```cpp
{
    agfx::ez::Frame frame = ctx.BeginFrame(); // RAII: calls EndFrame() on scope exit
    ctx.SetBackBufferRenderTarget(); // or ctx.SetRenderTargets({...}, depthTarget) for off-screen targets

    ctx.SetPipeline(myPipelineDesc);
    agfx::ez::ShaderBindings bindings;
    bindings.BindTexture(myTexture.SRV());
    bindings.BindSampler(mySampler);
    ctx.PushShaderBindings(bindings);
    ctx.Draw(3);
}
```

`BeginFrame`/`EndFrame` can also be called directly (without the RAII guard) if control flow doesn't fit a single scope — both drive the same state machine, so pick one style per call site rather than mixing them on the same frame. `BeginFrame` already waits on the right frame-in-flight fence slot and acquires+barriers the back buffer; don't add manual fence waits or back-buffer barriers alongside it.

### Render targets, off-screen and back buffer

```cpp
ctx.SetRenderTargets({&gbufferAlbedo, &gbufferNormal}, &depthTarget,
                      AGFX_LOAD_OPERATION_CLEAR, myClearColor,
                      AGFX_LOAD_OPERATION_CLEAR, /*clearDepth=*/1.0f);
// ... draw calls ...
ctx.SetBackBufferRenderTarget(AGFX_LOAD_OPERATION_CLEAR, myClearColor);
// ... tonemap/composite draw calls ...
```

`SetRenderTargets`/`SetBackBufferRenderTarget` each implicitly end any previously active render pass and transition every passed `Texture2D`/depth target into the correct state first — this is exactly the scope of ez's automatic tracking described in Overview. Call `EndActivePass()` before starting a compute pass on the same command buffer (`ctx.GetCurrentCommandBuffer().BeginComputePass(...)`), since a render pass and compute pass can't be active simultaneously.

Attachments can name a mip and layer, which is what makes mip chains and cube faces expressible. The pass dimensions come from the *bound subresource*, so rendering into mip N gets an N-sized pass:

```cpp
ctx.SetRenderTargets({{cubemap, /*mip*/ 0, /*layer*/ face}});          // one cube face (+X,-X,+Y,-Y,+Z,-Z)
ctx.SetRenderTargets({{mipChain, /*mip*/ n}});                          // one level of a mip chain
```

Only the bound subresource is transitioned — the other mips/layers keep their state, so mip N-1 can sit in a shader-read state while mip N is the render target. A 3D texture's depth slices are *not* layers and cannot be bound this way; address them through an `agfxTextureRegion`'s z/depth in a copy or upload.

### Viewport and scissor

`SetViewportScissor(x, y, w, h)` sets both to the same rect — the common case. When they must disagree (squeezing the image into a sub-rect is a viewport job; cropping it is a scissor job), call `SetViewport` and `SetScissor` separately.

### One-call resource creation

```cpp
agfx::ez::Texture2D albedo = ctx.CreateTexture2D(width, height, AGFX_TEXTURE_FORMAT_RGBA8_UNORM,
                                                  AGFX_TEXTURE_USAGE_SAMPLED, pixelData, bytesPerRow);
agfx::ez::Texture2DArray layers = ctx.CreateTexture2DArray(w, h, arrayLayers, format, usage, mipLevels);
agfx::ez::Texture3D volume = ctx.CreateTexture3D(w, h, depth, format, usage);
agfx::ez::TextureCube cube = ctx.CreateTextureCube(size, format, usage);
agfx::ez::Buffer vbo = ctx.CreateVertexBuffer(vertexData, vertexDataSize, sizeof(Vertex));
agfx::ez::Buffer ibo = ctx.CreateIndexBuffer(indexData, indexDataSize);
agfx::ez::Buffer sceneBuf = ctx.CreateStructuredBuffer(sceneData, sceneDataSize, sizeof(SceneVertex));
```

These upload synchronously (map a staging buffer, copy, submit, wait) and call `agfxDeviceMakeResourcesResident` before returning — safe to call outside the frame loop (asset loading), but each call is a blocking round trip to the GPU, so batch asset loading rather than calling these per-frame or in a tight loop. For a render-target-only or storage-only texture (no initial pixel data), pass `pixels = nullptr`; residency is still established. `AGFX_TEXTURE_USAGE_SAMPLED` is force-ORed into every texture's usage, so it never needs passing explicitly.

`CreateTexture2D`'s `pixels` only seeds mip 0 of layer 0. `UploadTexture(dst, region, mipLevel, layer, data, dataSize, bytesPerRow, bytesPerImage)` is the general form — use it to seed lower mips, array layers, cube faces, and 3D slices (3D slices via `region.z`/`region.depth`, since a 3D texture has no layers). It blocks and establishes residency, same as creation.

Uploaded resources are deliberately left in `AGFX_RESOURCE_STATE_COMMON`, and the tracker agrees. Don't "fix" this with a post-upload barrier: COMMON's producer stages are 0 like every read state's, so it changes nothing on D3D12, and on Metal4 a 0-producer/non-zero-consumer pair yields `barrierAfterQueueStages:0`, which Metal's validation layer rejects.

### Copies (inside the frame loop)

```cpp
ctx.CopyTextureToTexture(src, dst, region, mipLevel, layer);
ctx.CopyBufferToTexture(src, dst, region, mipLevel, layer, bytesPerRow);
ctx.CopyTextureToBuffer(src, dst, bufferOffset, region, mipLevel, layer, bytesPerRow);
```

Each opens and closes its own scoped compute pass on the frame's command buffer, so call them between `BeginFrame`/`EndFrame` and after `EndActivePass()` if a render pass is open. **These do not transition for you** — issue the `COPY_SOURCE`/`COPY_DEST` transitions yourself via `TransitionTexture`/`TransitionBuffer`, matching the rest of ez's explicit barrier model.

### Views: lazily created, cached, not re-created per frame

```cpp
agfx::TextureView& srv = albedo.SRV(); // whole resource by default; created on first call, cached thereafter
agfx::TextureView& mipSrv = albedo.SRV(baseMip, mipCount, baseLayer, layerCount); // any subrange
agfx::TextureView& uav = albedo.UAV(mip, 1); // single-mip UAV, what mip-chain generation needs
agfx::RenderTarget& rtv = albedo.RTV(mipLevel, arrayLayer); // COLOR_ATTACHMENT or DEPTH_STENCIL_ATTACHMENT bit required
agfx::BufferView& structured = sceneBuf.View(AGFX_BUFFER_VIEW_TYPE_STRUCTURED);
```

Call `SRV()`/`UAV()`/`RTV()`/`View()` each time a handle is needed rather than caching the return value yourself across frames — they're cheap after the first call (return the cached view), and this avoids holding a reference that outlives the owning `Texture2D`/`Buffer`.

### Pipelines: describe, don't manage objects

```cpp
agfx::ez::PipelineDesc desc;
desc.name = "gbuffer";
desc.vertexShader = &gbufferVS;
desc.fragmentShader = &gbufferFS;
desc.cullMode = AGFX_CULL_MODE_BACK;
desc.depthTestEnable = true;
ctx.SetPipeline(desc); // hashed and cached internally -- same desc == same agfx::RenderPipeline reused
```

`SetPipeline` must be called after `SetRenderTargets`/`SetBackBufferRenderTarget` in the same pass — the cache key includes the currently bound color/depth formats, which aren't known until targets are set. `PipelineDesc` is intentionally limited (uniform blend state across all color attachments, no per-attachment arrays) — drop to raw AGFX (`agfxRenderPipelineCreateInfo` directly, via `ctx.GetDevice()`) for anything needing per-attachment blend state.

Mesh shading is supported: set `desc.meshShader` (and optionally `desc.taskShader`) instead of `desc.vertexShader`, keep `desc.fragmentShader` set, then draw with `ctx.DrawMesh(groupCountX, groupCountY, groupCountZ)` instead of `ctx.Draw`/`ctx.DrawIndexed`. A `PipelineDesc` must set exactly one of `vertexShader` or `meshShader` — mixing both (or setting neither) asserts. Also set `desc.meshGroupSizeX/Y/Z` (and `taskGroupSizeX/Y/Z` if using a task shader) from `agfxShaderCompilerResult::meshSizeX/Y/Z`/`taskSizeX/Y/Z` — the reflected thread-group size `agfxCompileShader` extracts from the shader's `[numthreads(...)]` — since ez has no hook into the compiler itself and can't infer these; the Metal backend needs them to dispatch correctly and they default to `1,1,1`, which is wrong for almost every real shader.

### Push constants via ShaderBindings

```cpp
agfx::ez::ShaderBindings bindings;
bindings.BindTexture(albedo.SRV());
bindings.BindBuffer(sceneBuf.View(AGFX_BUFFER_VIEW_TYPE_CONSTANT));
bindings.BindSampler(linearSampler);
bindings.Write(worldMatrix); // raw POD data, same 128-byte budget
ctx.PushShaderBindings(bindings);
```

Write order in `ShaderBindings` must match the field order in the shader's push-constant struct (`AGFX_PUSH_CONSTANTS` — see `agfx-writing-bindless-shaders`) exactly, same as building the struct by hand in raw AGFX. `kCapacity` is 128 bytes; `Write` asserts on overflow — if a shader needs more, route the extra data through a `ResourceHandle` to a structured buffer (`AllocateConstants`, below) rather than growing past the budget.

### Per-frame dynamic constants

```cpp
struct FrameConstants { float4x4 viewProj; };
FrameConstants fc{ viewProj };
agfx::BufferView& cbView = ctx.AllocateConstants(fc); // ring-buffered, valid for this frame slot only
bindings.BindBuffer(cbView);
```

`AllocateConstants` hands back a view into a ring buffer sized by `dynamicConstantsBudgetPerFrame` (default 4 MiB) and rotated per frame-in-flight slot — safe to call freely within a frame for per-draw/per-pass constant data, but the returned view is only valid for the current frame (the slot gets reused once `BeginFrame` wraps back around after `framesInFlight` frames). Don't hold onto a returned `agfx::BufferView&` past `EndFrame`.

### GPU-driven draws (indirect bundles)

```cpp
ez::IndirectBundle bundle = ctx.CreateIndirectBundle(AGFX_INDIRECT_BUNDLE_TYPE_DRAW_INDEXED, maxDraws);

// Cull + prepare on a raw compute pass -- ez has no compute-pass sugar, by design.
ctx.EndActivePass();
ctx.TransitionIndirectBundle(bundle, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);
{
    agfx::ComputePass pass = ctx.GetCurrentCommandBuffer().BeginComputePass("Culling");
    // ... dispatch the culling shader with bundle.GetHandle() in its push constants ...
    pass.BufferUAVBarrier(bundle.Raw().CommandsBuffer());
    pass.BufferUAVBarrier(bundle.Raw().CountBuffer());
    pass.PrepareIndirectBundle(bundle.Raw(), prepareInfo);
}
ctx.TransitionIndirectBundle(bundle, AGFX_RESOURCE_STATE_INDIRECT_ARGUMENT);

ctx.SetBackBufferRenderTarget();
ctx.SetPipeline(desc);                     // desc.supportsIndirect must be true
ctx.ExecuteIndirectBundle(bundle, executeInfo);
```

`TransitionIndirectBundle` moves the commands and count buffers together — they always transition in lockstep, so the bundle carries a single shared state tracker — and no-ops when already in the target state. `Context::ExecuteIndirectBundle` asserts an active render pass and handles DRAW/DRAW_INDEXED/DRAW_MESH bundles; **DISPATCH bundles have no ez wrapper** and go through `agfx::ComputePass::ExecuteIndirectBundle` on a raw compute pass, same as prepare.

Two requirements ez cannot check for you, both silent-on-D3D12 / fatal-on-Metal: the pipeline needs `supportsIndirect`, and `pushConstants` must be filled on the *prepare* info as well as the execute info. See **agfx-mdi** for both, and for the bundle layout and ordering rules this sugar sits on top of.

### Explicit transitions (the escape hatch)

```cpp
ctx.TransitionTexture(shadowMap, AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE); // whole resource; no-op if already there
ctx.TransitionTexture(chain, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, mip, layer); // one subresource
```

Whole-resource transitions after a split are still correct: the subresources are re-grouped by current state and one barrier is emitted per distinct old state, since a barrier can only name one. `State()` on a texture asserts while the tracker is split — use `StateAt(mip, layer)` in code that transitions subresources individually.

Use this when a resource needs a state ez's automatic `SetRenderTargets` tracking doesn't cover — e.g. transitioning a texture into a shader-readable state before binding it via `ShaderBindings::BindTexture`, since ez only auto-transitions render/depth targets on `SetRenderTargets`, not arbitrary sampled textures. Remember the tracking-scope caveat from Overview: this only works correctly if the resource has never been touched by raw AGFX calls outside ez.

### Resize and HDR toggle

```cpp
ctx.Resize(newWidth, newHeight);       // drains GPU internally, then resizes the swap chain
ctx.SetHDR(wantHDR, vsyncSetting);     // drains GPU internally, destroys+recreates the swap chain
```

Both already drain the GPU — don't add a manual drain around these calls. After either, any `PipelineDesc` bound against the back buffer format is still fine (the cache keys on bound formats, and `SetBackBufferRenderTarget` re-queries `GetSwapChainFormat()` each call), but off-screen render targets sized to match the window need to be resized/recreated by the caller — ez doesn't own those.

### Ray tracing

```cpp
if (ctx.SupportsRayTracing()) {
    ez::AccelerationStructure blas = ctx.CreateBottomLevel(vbo, vertexCount, vertexOffset,
                                                           ibo, indexCount, indexOffset);
    ez::AccelerationStructure tlas = ctx.CreateTopLevel(instances, instanceCount);
    bindings.Write(tlas.Bind()); // bindless handle to trace against
}
```

Both calls create *and* synchronously build in one go, hiding scratch allocation, residency, and the AS→AS build barrier. `vertexOffset`/`indexOffset` are element indices (scaled by the buffer's stride internally), and the first vertex attribute must be `float3` position. On a non-RT device they return an invalid structure rather than failing — `Valid()` is false and `Bind()` gives 0 — so always guard with `SupportsRayTracing()` instead of relying on that. Instance transforms are row-major 3x4, and a referenced BLAS must already be built before `CreateTopLevel`. Tracing itself is inline `RayQuery` HLSL — see **agfx-raytracing**.

### Headless contexts

Leave `windowHandle` null in `ContextCreateInfo` and no swap chain is created — the frame loop, resource creation, passes, copies and readback all work normally. `IsHeadless()` reports it, and `SetBackBufferRenderTarget`, `Resize`, `SetHDR`, and `GetSwapChainFormat` all assert if called. This is the shape for offline bakers, test harnesses, and render-to-file tools: render into an ez texture, `CopyTextureToBuffer`, drain, read back.

### Dropping to raw AGFX

```cpp
agfx::Device& device = ctx.GetDevice();
agfx::CommandQueue& queue = ctx.GetGraphicsQueue();
agfx::CommandBuffer& cmd = ctx.GetCurrentCommandBuffer(); // e.g. to hand to ImGui_ImplAGFX_RenderDrawData
```

Use these when a specific piece of code needs raw AGFX control (custom barrier scheduling, resource types `Texture2D`/`Buffer` don't model, multiple command buffers per frame) without abandoning ez for the rest of the app — `GetActiveRenderPass()` similarly exposes the raw `agfx::RenderPass&` mid-frame for cases like handing it to a third-party ImGui backend.

## Common Mistakes

- **Mixing raw AGFX and ez creation for the same logical resource.** Creating a texture via `agfxTextureCreate` directly, then trying to track its state through `ctx.TransitionTexture` (which expects an `ez::Texture2D`), defeats tracking silently — pick one creation path per resource.
- **Caching a `ShaderBindings` or `AllocateConstants` view across frames.** Both are frame-scoped; the ring buffer slot backing `AllocateConstants`'s return value gets overwritten once the frame-in-flight slot recycles.
- **Calling `SetPipeline` before `SetRenderTargets`/`SetBackBufferRenderTarget`.** The pipeline cache key depends on the currently bound attachment formats — this ordering isn't optional.
- **Assuming `TransitionTexture`/`SetRenderTargets` tracking covers UAV read/write hazards within a compute pass.** It doesn't — use the re-exposed `TextureUAVBarrier`/`BufferUAVBarrier` on the raw `agfx::ComputePass` for that (see `agfx-synchronization`), same requirement as raw AGFX.
- **Reaching for ez on a codebase that needs per-attachment blend state, or manual multi-command-buffer scheduling.** `PipelineDesc` and `Context`'s single-command-buffer-per-frame-slot model don't cover these — use raw AGFX instead (or ez for the parts that fit, raw AGFX for the parts that don't, via the escape hatches above). Mesh shading is covered by ez (`PipelineDesc::meshShader`/`taskShader`, `Context::DrawMesh`).
- **Assuming `ctx.ExecuteIndirectBundle` is the whole indirect story.** It is only the replay call. Creating the bundle, resetting the count slot, culling, preparing, and the surrounding transitions are all on you — and a bundle shared across frames in flight needs a write-after-read barrier before each rebuild. See **agfx-mdi**.
- **Setting both `vertexShader` and `meshShader` on a `PipelineDesc` (or neither).** Exactly one must be set — `SetPipeline` asserts otherwise.
