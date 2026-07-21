---
name: agfx-porting-from-metal
description: ALWAYS use when porting an existing Metal (Metal 3 or Metal 4) engine or renderer to AGFX — translating MTLDevice/MTLCommandBuffer/MTLRenderPipelineState/MTLArgumentEncoder code to agfxDevice/agfxCommandBuffer/agfxRenderPipeline calls, converting Metal Shading Language (MSL) argument-buffer/bindless shaders to AGFX's HLSL bindless model, or mapping Metal concepts (command queues/buffers/encoders, MTLFence/MTLEvent, MTLResourceUsage/useResource residency, CAMetalLayer) onto their AGFX equivalents. Trigger on MTLDevice/MTLCommandBuffer/MTLRenderCommandEncoder/MTLComputeCommandEncoder/MTLRenderPipelineState, id<MTLTexture>/id<MTLBuffer>, MTLArgumentEncoder, CAMetalLayer, "port to AGFX", "port from Metal". Do NOT trigger for porting a non-Metal engine to raw Metal (that's game-porting-skills, not AGFX). Do NOT trigger for AGFX-native questions with no Metal source code involved — use the specific agfx-* skill for that subsystem instead (agfx-presentation-and-swapchain, agfx-render-targets-and-passes, agfx-synchronization, agfx-writing-bindless-shaders, agfx-raytracing, agfx-mdi).
---

# Porting a Metal Engine to AGFX

## Overview

This skill is for porting an existing **Metal-based engine to AGFX** (making it cross-platform, since AGFX also has a D3D12 backend) — the reverse direction from `game-porting-skills`, which ports non-Metal engines *to* raw Metal. If the request is instead "port my D3D12/Vulkan engine to Metal directly, not through AGFX," that's `game-porting-skills`, not this skill.

Metal is, of all AGFX's source APIs, the one whose *backend implementation* AGFX itself is built on for macOS (`agfx/agfx_metal4.mm` uses Metal 4). Porting a Metal engine to AGFX is largely exposing/renaming an already-similar shape: command buffers, encoders-as-passes, explicit resource barriers (if already using Metal 4's explicit barrier API) or `useResource`-style residency (Metal 3), and — if the engine already uses argument buffers with bindless-style indexing — a binding model close to AGFX's own. The main new work is making the engine's Metal-only code paths cross-platform-shaped (AGFX API calls instead of direct `id<MTLDevice>` calls) and, if the source still uses classic per-encoder `setTexture:`/`setBuffer:`/`setSamplerState:` binding rather than argument buffers, rewriting that to bindless.

## Ownership

**Owns:**
- The Metal → AGFX concept translation table below
- Metal 3 vs. Metal 4 API differences that matter for this port (AGFX's own backend is Metal 4; a Metal-3-based source engine has a slightly bigger step to take)
- What gets deleted (classic per-encoder resource binding, if present) vs. what maps closely (command buffers, encoders, barriers, residency, argument-buffer-based bindless code)
- Recommended porting order

**Doesn't own:**
- Metal API mechanics unrelated to AGFX (writing/debugging raw Metal 4 code, MetalFX, GPU capture/validation tooling) — that's the `game-porting-skills` domain skills (`translating-to-metal4-api`, `managing-metal4-resources`, `managing-metal4-synchronization`, `presenting-metal-drawables`, etc.)
- Subsystem-specific AGFX API detail once the call is identified — the four `agfx-*` subsystem skills
- D3D11/D3D12/OpenGL/Vulkan-specific concept mapping — the sibling `agfx-porting-from-*` skills

## References

`agfx/agfx.h` is the entire public API surface — read it top to bottom once before starting. `agfx/agfx_metal4.mm` is AGFX's own Metal 4 backend implementation — the authoritative answer for "what does this AGFX call actually do in Metal terms," and useful for confirming a proposed translation is correct rather than guessing. `agfx_demo/` (`deferred_renderer.cpp`, `agfx_demo_main.cpp`) is a complete cross-platform reference engine already written against AGFX. If the source engine's Metal code needs *deeper* Metal-API-level understanding before it can be mapped (e.g. what a given Metal 3 call's Metal 4 equivalent is), consult the relevant `game-porting-skills` domain skill (e.g. `translating-to-metal4-api`) alongside this one — that skill owns raw Metal API detail, this one owns the AGFX mapping.

## Concept Translation Table

| Metal | AGFX | Notes |
|---|---|---|
| `id<MTLDevice>` (`MTLCreateSystemDefaultDevice`) | `agfxDevice*` | `agfxDeviceCreate`; AGFX also handles D3D12 device creation behind the same call on Windows |
| `id<MTLCommandQueue>` | `agfxCommandQueue*` | `agfxCommandQueueCreate`, typed via `agfxCommandQueueType` |
| `id<MTLCommandBuffer>` | `agfxCommandBuffer*` | `agfxCommandBufferCreate`/`Begin`/`End`; AGFX command buffers are explicitly reset/reused per frame slot rather than one-shot-and-discard like typical Metal 3 usage — see `agfx-synchronization` for the frame-in-flight pattern |
| `id<MTLFence>` (Metal 3, intra-queue) / `MTLSharedEvent`/`MTL4Fence`/`MTL4Event` (Metal 4, cross-queue/CPU sync) | `agfxFence*` | AGFX unifies both roles behind one fence object with monotonically increasing values, used the same way for CPU↔GPU (`agfxFenceWait`) and GPU↔GPU (`agfxCommandQueueSignal`/`Wait`) sync — see `agfx-synchronization` |
| `MTLResourceUsage`/`useResource:usage:`/`useHeap:` (Metal 3 residency-and-hazard declaration) or explicit `MTL4Barrier`/`MTLBarrierScope` (Metal 4) | `agfxResourceState` + `agfxCommandBufferTextureBarrier`/`BufferBarrier` | AGFX's barrier states are D3D12-shaped (`AGFX_RESOURCE_STATE_RENDER_TARGET`, `PIXEL_SHADER_RESOURCE`, etc.) rather than Metal's usage-flag model; the `agglomerate` parameter is specifically for Metal's lack of per-resource hazard tracking outside explicit Metal-4-style barriers — see `agfx-synchronization` for exactly how it's used under the hood in `agfx_metal4.mm` |
| `agfxDeviceMakeResourcesResident` | — | AGFX exposes this directly (`agfxDeviceMakeResourcesResident`, doc'd as the `MTLResidencySet.commit` equivalent) — call it after creating/uploading new resources and before submitting work that references them, same requirement Metal residency sets already impose |
| `MTLTextureDescriptor` + `newTextureWithDescriptor:` | `agfxTexture*` | `agfxTextureCreate`; `agfxTextureUsage` bitflags replace `MTLTextureUsage` |
| `MTLBufferDescriptor`/`newBufferWithLength:options:` | `agfxBuffer*` | `agfxBufferUsage` replaces buffer usage intent; `agfxBufferMemoryType` replaces `MTLStorageMode` (`.private` → `GPU_ONLY`, `.shared`/`.managed` → `CPU_TO_GPU`/`GPU_TO_CPU` depending on direction) |
| `-[MTLBuffer contents]` (persistently mapped for shared/managed storage) | `agfxBufferMap`/`agfxBufferUnmap` | AGFX requires explicit map/unmap even though the underlying Metal buffer may already be persistently mapped — call both around each CPU write, don't hold the pointer across frames |
| Texture/sampler views via `newTextureViewWithPixelFormat:` or argument-buffer entries | `agfxTextureView*`/`agfxSampler*` | `agfxTextureViewCreate`/`agfxSamplerCreate`; both return a bindless `ResourceHandle` via `agfxTextureViewGetHandle`/`agfxSamplerGetHandle` rather than an argument-buffer slot you manage yourself |
| Argument buffers (`MTLArgumentEncoder`, or Metal 3's manual `IRDescriptorTableEntry`-style bindless) | AGFX's own bindless heap (`ResourceDescriptorHeap`/`SamplerDescriptorHeap` on the HLSL side) | if the source engine already builds its own bindless argument buffer with GPU-resident texture/buffer handles indexed by an integer, this is close to a rename — the handle concept already exists, it's just re-obtained through `agfxTextureViewGetHandle`/`agfxBufferViewGetHandle`/`agfxSamplerGetHandle` instead of a hand-rolled encoder; see `agfx-writing-bindless-shaders` |
| Classic per-encoder `setVertexTexture:atIndex:`/`setFragmentBuffer:offset:atIndex:`/`setFragmentSamplerState:atIndex:` | **deleted — bindless** | if the source engine binds resources this way rather than through argument buffers, delete this pattern outright and route through push constants + `ResourceHandle` instead — see `agfx-writing-bindless-shaders` |
| `setVertexBytes:length:atIndex:`/`setFragmentBytes:length:atIndex:` (small inline constant data) | `agfxRenderPassPushConstants`/`agfxComputePassPushConstants` | closest 1:1 mapping in the whole table — both are small inline per-draw/per-dispatch constant blocks; AGFX's is fixed at `register(b0)` |
| `[MTLVertexDescriptor]` + per-vertex-buffer binding | **deleted — vertex pulling** | AGFX vertex shaders take `SV_VertexID` and manually load from an `AGFXStructuredBuffer`; if the source engine already does manual vertex pulling in MSL (common in bindless Metal engines), this is close to a rename — see `agfx-writing-bindless-shaders` |
| `MTLRenderPassDescriptor` + `-[MTLCommandBuffer renderCommandEncoderWithDescriptor:]` | `agfxRenderTarget` + `agfxRenderPassBegin`/`agfxRenderPassCreateInfo` | `MTLLoadAction`/`MTLStoreAction` map directly to `agfxLoadOp`/`agfxStoreOp`; the encoder itself becomes the `agfxRenderPass` object returned by `agfxRenderPassBegin` — see `agfx-render-targets-and-passes` |
| `id<MTLRenderCommandEncoder>` draw calls (`drawPrimitives:`/`drawIndexedPrimitives:`) | `agfxRenderPassDraw`/`agfxRenderPassDrawIndexed` | recorded on the `agfxRenderPass` object, same as the encoder |
| `id<MTLComputeCommandEncoder>` + `dispatchThreadgroups:threadsPerThreadgroup:` | `agfxComputePass` + `agfxComputePassDispatch` | AGFX's `agfxComputePassDispatch` takes thread**group** counts like Metal's `dispatchThreadgroups:`, not the total-thread-count form of `dispatchThreads:threadsPerThreadgroup:` — check which the source uses and convert accordingly |
| `MTLRenderPipelineDescriptor`/`newRenderPipelineStateWithDescriptor:` (or `MTL4RenderPipelineDescriptor`/`MTL4Compiler` on Metal 4) | `agfxRenderPipelineCreate` | blend/raster/depth state plus attachment pixel formats folded into one `agfxRenderPipelineCreateInfo`; AGFX has no stencil support — flag stencil-dependent pipelines |
| `MTLComputePipelineDescriptor`/`newComputePipelineStateWithDescriptor:` | `agfxComputePipelineCreate` | straightforward 1:1; `[numthreads(x,y,z)]` equivalent comes from the HLSL shader itself, not a separate threadgroup-size descriptor field |
| MSL source (`.metal` files, compiled via `xcrun metal`/`MTLCompileOptions`) | HLSL, compiled via `agfxCompileShader` (DXC → DXIL, translated to Metal IR on macOS by AGFX internally) | source-level rewrite required: AGFX shaders are authored in HLSL, not MSL, even though the macOS backend ultimately still runs on Metal — see `agfx-writing-bindless-shaders`. Resource-access syntax (`[[texture(n)]]`, `[[buffer(n)]]` attributes) has no equivalent; everything routes through `ResourceHandle` |
| `CAMetalLayer` + `nextDrawable`/`present`/`presentAfterMinimumDuration:` | `agfxSwapChain*` | `agfxSwapChainCreate` takes the `CAMetalLayer*` as its `handle` on macOS; `agfxSwapChainAcquireNextTexture`/`agfxSwapChainPresent` replace `nextDrawable`/`present` — see `agfx-presentation-and-swapchain` |
| `wantsExtendedDynamicRangeContent`/EDR headroom (HDR) | `agfxSwapChainCreateInfo::isHDR` | destroy+recreate to toggle, same as every other AGFX backend — no in-place reconfiguration, unlike some direct `CAMetalLayer` HDR toggling code might assume — see `agfx-presentation-and-swapchain` |

## Recommended Porting Order

Because Metal and AGFX's Metal 4 backend already share deep structural similarity, the riskiest part of this port is usually the binding model (classic per-encoder binding vs. argument-buffer bindless), not the command-buffer/pass/barrier plumbing.

1. **Assess the source binding model first**, before planning phases: grep for `setVertexTexture:`/`setFragmentBuffer:` classic binding vs. `MTLArgumentEncoder`/hand-rolled bindless argument buffers. This determines whether the shader-porting phase is a rename (already bindless) or a rewrite (classic binding) — size the plan accordingly.
2. **Device, queue, swap chain.** Replace `MTLCreateSystemDefaultDevice`/`newCommandQueue`/`CAMetalLayer` setup with `agfxDeviceCreate`/`agfxCommandQueueCreate`/`agfxSwapChainCreate`. Get a cleared back buffer presenting before porting rendering logic.
3. **Frame pacing and fences.** Port `MTLFence`/`MTLSharedEvent`/Metal-4-fence-based frame pacing to `agfxFence` + per-slot command buffers (`agfx-synchronization`) — this is close to a rename if the source already tracks frame-in-flight command buffer reuse explicitly.
4. **Resources.** Port `newTextureWithDescriptor:`/`newBufferWithLength:options:` to `agfxTextureCreate`/`agfxBufferCreate`. If the source manages its own argument-buffer-backed bindless heap, this step also means retiring that in favor of AGFX's built-in one — keep the returned `ResourceHandle`s, drop the custom heap bookkeeping.
5. **One render pass end-to-end.** Convert one `MTLRenderPassDescriptor` + encoder sequence to `agfxRenderPassBegin`/draw/`End` (`agfx-render-targets-and-passes`), with barriers into the right states beforehand (`agfx-synchronization`) — this is where binding-model mistakes surface fastest.
6. **Shaders.** Rewrite each `.metal` shader to HLSL: remove `[[texture(n)]]`/`[[buffer(n)]]` attributes and any classic-binding assumptions, replace with `AGFX_PUSH_CONSTANTS` + `ResourceHandle` fields and `AGFXTexture2D`/`AGFXStructuredBuffer`/`AGFXSampler::Create(handle)` calls; if the source already does manual vertex pulling in MSL, port that logic directly rather than reinventing it (`agfx-writing-bindless-shaders`). Port a shader and its host-side push-constant struct together.
7. **Remaining passes**, then **cross-cutting**: stencil-dependent logic (unsupported, flag it), MetalFX upscaling/frame interpolation (no AGFX equivalent — flag as a feature the port drops or needs a separate integration), HDR/resize (`agfx-presentation-and-swapchain`). Mesh shaders, ray tracing, and indirect command buffers all have AGFX equivalents — see "Advanced features" below.

## Advanced features: mesh shaders, ray tracing, GPU-driven draws

All three are supported as of **AGFX v1.2.0** (ray tracing landed in v1.1.0, multi-draw indirect in v1.2.0). Each is capability-gated — query once and keep the fallback path alive, since none are universal (Apple silicon needs M3+ for ray tracing and mesh shaders):

```cpp
agfxDeviceInfo info = {};
agfxDeviceGetInfo(device, &info);
// info.supportsRayTracing / info.supportsMeshShaders / info.supportsMultiDrawIndirect
```

| Metal | AGFX | Notes |
|---|---|---|
| `drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:` | `agfxRenderPassDrawMesh` | AGFX takes the threadgroup sizes from the pipeline (`meshGroupSizeX/Y/Z`, `taskGroupSizeX/Y/Z`), not the draw call |
| `MTLAccelerationStructure` + acceleration-structure encoder builds | `agfxAccelerationStructureCreate` + `agfxComputePassBuildAccelerationStructure` | AGFX builds inside a compute pass on both backends |
| `intersection_query` in MSL | `RayQuery` in HLSL | direct equivalent — Metal's inline model is the one AGFX exposes |
| `MTLIndirectCommandBuffer` + `executeCommandsInBuffer:` + a GPU encoding kernel | `agfxIndirectBundle` + `PrepareIndirectBundle`/`ExecuteIndirectBundle` | **AGFX already owns the ICB and its encoding kernel.** Delete the engine's hand-rolled ICB encoding; write D3D12-shaped commands via the `AGFXIndirectDraw*Bundle` HLSL helpers and let `PrepareIndirectBundle` build the ICB |

**The one structural mismatch to flag early:** AGFX supports **inline ray tracing only** — `RayQuery`/`TraceRayInline` from a compute shader. There is no ray-generation/any-hit/closest-hit pipeline, no hit groups, and no shader binding table. A source engine built around a ray-tracing *pipeline* needs those passes restructured into compute dispatches that trace inline and shade at the hit point themselves; that is a redesign, not a translation, and is worth surfacing to the user before starting.

Delegate the actual work: **agfx-raytracing** (acceleration structures, inline tracing), **agfx-mdi** (indirect bundles, GPU culling), **agfx-writing-bindless-shaders** (`main_ms`/`main_as` entry points, reflected group sizes) with **agfx-render-targets-and-passes** for `agfxRenderPassDrawMesh`.

## Common Porting Pitfalls

- **Assuming Metal 3 residency (`useResource:`/`useHeap:`) needs no explicit barrier equivalent.** It still needs an `agfxCommandBufferTextureBarrier`/`BufferBarrier` call with the right `agglomerate` value — Metal 3's `useResource:` declares residency/usage, it isn't the same thing as AGFX's state-transition barrier, even though both ultimately affect what's valid to access when.
- **Porting `dispatchThreads:threadsPerThreadgroup:` (total-thread-count) dispatch math as if it were `dispatchThreadgroups:threadsPerThreadgroup:` (group-count).** `agfxComputePassDispatch` takes group counts — divide/round up total thread counts by the threadgroup size before calling it if the source used the total-thread-count form.
- **Treating a source engine's already-bindless MSL as done.** Even a fully argument-buffer-bindless Metal shader still needs its resource-access syntax rewritten to HLSL and its handle type changed to `ResourceHandle` — the *concept* transfers, the *code* doesn't compile as-is.
- **Porting stencil-dependent logic.** AGFX's pipeline depth state has no stencil fields — flag stencil-dependent Metal code to the user rather than silently dropping it.
- **Assuming MetalFX (temporal upscaling, frame interpolation) has an AGFX equivalent.** It doesn't — AGFX has no upscaling/frame-generation API. Flag this to the user; the engine either drops the feature in the ported build or needs a separate, AGFX-external MetalFX integration.
