---
name: agfx-porting-from-d3d11
description: ALWAYS use when porting an existing D3D11 engine or renderer to AGFX — translating ID3D11Device/DeviceContext/InputLayout/state-object code to agfxDevice/agfxCommandBuffer/agfxRenderPipeline calls, converting HLSL from D3D11's `register(t/b/u/s)` immediate-context binding model to AGFX bindless, or mapping D3D11 concepts (immediate/deferred context, input layouts, state objects, swap chain) onto their AGFX equivalents. Trigger on ID3D11*, D3D11_*, HRESULT device/context calls, VSSetShaderResources/PSSetConstantBuffers/IASetInputLayout, CreateBlendState/CreateRasterizerState/CreateDepthStencilState, "port to AGFX", "port from D3D11". Do NOT trigger for D3D12 sources — use agfx-porting-from-d3d12. Do NOT trigger for AGFX-native questions with no D3D11 source code involved — use the specific agfx-* skill for that subsystem instead (agfx-presentation-and-swapchain, agfx-render-targets-and-passes, agfx-synchronization, agfx-writing-bindless-shaders).
---

# Porting a D3D11 Engine to AGFX

## Overview

D3D11 is an immediate-mode, implicit-synchronization API: one `ID3D11DeviceContext` records and submits state changes call-by-call, resource lifetime and GPU/CPU hazard tracking are handled by the driver, and shader resource binding happens through slot-based `VSSetShaderResources`/`PSSetConstantBuffers`/`PSSetSamplers` calls tied to `register(tN)`/`register(bN)`/`register(sN)` in HLSL. AGFX is the opposite in every one of these respects: commands are recorded into explicit `agfxCommandBuffer`s and submitted explicitly, resource-state transitions and GPU/CPU sync are the caller's job (barriers, fences), and shader resource access is fully bindless through `ResourceHandle` — there is no slot to bind to.

This is a bigger conceptual gap than porting from D3D12 or Vulkan (both of which already use explicit command buffers/lists and know about resource states), so treat this as "rebuild the frame around AGFX's model," not "translate call-by-call." If the codebase's calling style already looks like D3D11 (single global immediate context, no manual barriers) and the goal is to preserve that programming model, consider porting to `agfx::ez` (`using-agfx-ez`) instead of raw AGFX — it's an immediate-mode C++ wrapper deliberately designed to feel like D3D11's context, and removes most of the work described below.

## Ownership

**Owns:**
- The D3D11 → AGFX concept translation table below
- What has no AGFX equivalent at all (input layouts, deferred contexts, D3D11 state objects) vs. what maps onto something AGFX-shaped
- Recommended porting order and where the D3D11→AGFX gap is widest
- When to recommend `agfx::ez` instead of raw AGFX for this kind of port

**Doesn't own:**
- Subsystem-specific API detail once you know which AGFX call replaces a given D3D11 call — that's the four `agfx-*` subsystem skills, or `using-agfx-ez` if porting to the ez layer
- D3D12-specific concept mapping (root signatures, descriptor heaps, explicit barriers already present in the source) → `agfx-porting-from-d3d12`, since a D3D11 codebase won't have these to begin with

## References

`agfx/agfx.h` is the entire public API surface — read it top to bottom once before starting. `agfx/agfx_ez.hpp` is the immediate-mode convenience layer (`using-agfx-ez`) — read it if recommending/using the ez path. `agfx_ez_demo/agfx_ez_demo_main.cpp` is a complete small program against the ez layer, structurally closer to a typical D3D11 app than `agfx_demo/`. `agfx_demo/` (particularly `deferred_renderer.cpp`) is the raw-AGFX reference for larger/more structured engines.

## Concept Translation Table

| D3D11 | AGFX | Notes |
|---|---|---|
| `ID3D11Device` | `agfxDevice*` | `agfxDeviceCreate` |
| `ID3D11DeviceContext` (immediate) | `agfxCommandBuffer*` (+ `agfxCommandQueue*`) | D3D11's single implicit context becomes an explicitly recorded-and-submitted command buffer; there is no implicit "current context" |
| `ID3D11DeviceContext` (deferred) + `FinishCommandList`/`ExecuteCommandList` | multiple `agfxCommandBuffer`s submitted via `agfxCommandQueueSubmit` | AGFX command buffers are always "deferred" in D3D11 terms — recorded, then explicitly submitted; there's no separate deferred-context concept |
| Swap chain (`IDXGISwapChain`, `CreateDeviceAndSwapChain`) | `agfxSwapChain*` | `agfxSwapChainCreate/AcquireNextTexture/Present/Resize` — see `agfx-presentation-and-swapchain` |
| `ID3D11Texture2D` + `CreateShaderResourceView`/`CreateRenderTargetView`/`CreateUnorderedAccessView`/`CreateDepthStencilView` | `agfxTexture*` + `agfxTextureView*`/`agfxRenderTarget*` | one texture object; separate typed views (SRV/UAV analog is `agfxTextureView`, RTV/DSV analog is `agfxRenderTarget`) created explicitly, not implicitly bound |
| `ID3D11Buffer` (vertex/index/constant, `D3D11_BIND_*` flags) | `agfxBuffer*` | `agfxBufferUsage` bitflags replace `D3D11_BIND_VERTEX_BUFFER`/`INDEX_BUFFER`/`CONSTANT_BUFFER`/`SHADER_RESOURCE`/`UNORDERED_ACCESS`; `agfxBufferMemoryType` replaces `D3D11_USAGE_DYNAMIC`/`DEFAULT`/`STAGING` |
| `Map`/`Unmap` (`D3D11_MAP_WRITE_DISCARD`) | `agfxBufferMap`/`agfxBufferUnmap` | only valid on `AGFX_BUFFER_MEMORY_TYPE_CPU_TO_GPU`/`GPU_TO_CPU` buffers, same "map, write, unmap, submit" shape as D3D11 dynamic buffers |
| `ID3D11InputLayout` + `IASetInputLayout`/`IASetVertexBuffers` | **deleted — vertex pulling** | there is no input layout or vertex-buffer binding in AGFX; vertex shaders take `SV_VertexID` and manually load from an `AGFXStructuredBuffer` — see `agfx-writing-bindless-shaders` |
| `VSSetShaderResources`/`PSSetShaderResources`/`CSSetShaderResources`, `*SetConstantBuffers`, `*SetSamplers` | **deleted — bindless** | no slot-based binding of any kind; every resource is accessed via `ResourceHandle` passed through push constants — see `agfx-writing-bindless-shaders` |
| `register(t0)`/`register(b0)`/`register(s0)` in HLSL, tied to slot-binding calls | AGFX's bindless header + `ResourceHandle` | shader HLSL needs rewriting, not just host code — see `agfx-writing-bindless-shaders` |
| `ID3D11VertexShader`/`PixelShader`/`ComputeShader` + `CreateVertexShader` etc. (from precompiled `.cso`/`ID3DBlob`) | `agfxShaderModule*` | AGFX compiles HLSL itself via `agfxCompileShader` (DXC → DXIL, translated to Metal IR on macOS); if the D3D11 engine compiles offline, switch to AGFX's runtime/build-time compiler pipeline — see `agfx-writing-bindless-shaders` |
| `ID3D11BlendState`/`CreateBlendState` | fields on `agfxRenderPipelineCreateInfo` (`blendEnable[]`, `srcColorBlendFactor[]`, etc.) | baked into the pipeline object at creation, not a separately bound state object |
| `ID3D11RasterizerState`/`CreateRasterizerState` | fields on `agfxRenderPipelineCreateInfo` (`fillMode`, `cullMode`, `frontFace`) | same — baked into the pipeline, no separate bind call |
| `ID3D11DepthStencilState`/`CreateDepthStencilState` | fields on `agfxRenderPipelineCreateInfo` (`depthTestEnable`, `depthWriteEnable`, `depthCompareOp`) | same; AGFX has no separate stencil support to map — drop stencil-only logic or flag it as unsupported |
| `ID3D11SamplerState`/`CreateSamplerState` | `agfxSampler*` | `agfxSamplerCreate`; obtained as a bindless handle like any other resource, not bound to a slot |
| `OMSetRenderTargets` | `agfxRenderTarget` + `agfxRenderPassBegin` | D3D11's implicit "currently bound RTVs" becomes an explicit render pass with attachments and load/store ops — see `agfx-render-targets-and-passes` |
| `ClearRenderTargetView`/`ClearDepthStencilView` | `agfxLoadOp` = `AGFX_LOAD_OPERATION_CLEAR` on the render pass attachment | clears are declared as part of starting the pass, not a separate call |
| `Draw`/`DrawIndexed`/`DrawInstanced` | `agfxRenderPassDraw`/`agfxRenderPassDrawIndexed` | 1:1 in spirit; called on the render pass object, not the context |
| `Dispatch` | `agfxComputePassDispatch` | recorded within an `agfxComputePass`, not directly on the context |
| No explicit resource-state concept (driver-managed hazard tracking) | `agfxResourceState` + `agfxCommandBufferTextureBarrier`/`BufferBarrier` | this is the single biggest new burden a D3D11 port takes on: every render target write → shader read transition, every UAV write → next-read ordering, must now be barriered by hand — see `agfx-synchronization` |
| No explicit CPU/GPU sync (driver-managed) | `agfxFence` + frame-in-flight command buffer rotation | D3D11 never required manual fencing; AGFX does — see `agfx-synchronization` |

## Recommended Porting Order

1. **Decide raw AGFX vs. `agfx::ez`.** If the D3D11 code is architecturally simple (single render loop, no deferred contexts, resources created ad hoc rather than pooled/tracked), porting to `agfx::ez` (`using-agfx-ez`) preserves the D3D11-like "immediate calls, no manual barrier bookkeeping for the common case" feel and is significantly less work. Reserve raw AGFX for engines that need explicit control the ez layer doesn't expose (custom barrier scheduling, more than one command buffer per frame, non-2D texture types).
2. **Device, queue, swap chain.** Wire up `agfxDeviceCreate`/`agfxCommandQueueCreate`/`agfxSwapChainCreate` (or `agfx::ez::Context` if using ez) and get a cleared back buffer presenting — validates build/link before any rendering logic moves over.
3. **Frame pacing and fences** (raw AGFX only — `agfx::ez::Context` does this internally). D3D11 had no fence concept to port; this is new code, not translated code. See `agfx-synchronization`.
4. **Resources.** Port `ID3D11Buffer`/`ID3D11Texture2D` creation to `agfxBufferCreate`/`agfxTextureCreate` (or `agfx::ez::Context::CreateVertexBuffer`/`CreateTexture2D` etc.). Drop `Map`/`Unmap`-per-frame dynamic-buffer patterns in favor of AGFX's ring-buffered constants if using ez, or a hand-rolled equivalent if using raw AGFX.
5. **One render pass end-to-end.** Convert one `OMSetRenderTargets` + draw sequence to an `agfxRenderPassBegin`/draw/`End` (`agfx-render-targets-and-passes`), including the barriers D3D11 never required (`agfx-synchronization`, or `agfx::ez::Context::TransitionTexture`/`SetRenderTargets`, which tracks this automatically for ez-created resources).
6. **Shaders.** Rewrite each HLSL shader's binding section: remove `register(tN/bN/sN)` slot declarations, replace with `AGFX_PUSH_CONSTANTS` + `ResourceHandle` fields and `AGFXTexture2D`/`AGFXStructuredBuffer`/`AGFXSampler::Create(handle)` calls, replace input-layout vertex fetch with vertex pulling from `SV_VertexID` (`agfx-writing-bindless-shaders`). A shader rewritten without its host-side push-constant struct updated in the same pass won't compile — do these together.
7. **Remaining passes**, then **cross-cutting** concerns: any stencil-dependent logic (unsupported — flag to the user), MSAA (check current AGFX support before assuming it maps), HDR/resize (`agfx-presentation-and-swapchain`).

## Common Porting Pitfalls

- **Assuming the driver still tracks hazards.** The single most common bug: D3D11 code has zero barrier calls because the driver inserted them implicitly. Every render-target-write → shader-read, and every UAV write ordered against another dispatch, needs an explicit barrier now (`agfx-synchronization`) — missing one doesn't fail to compile, it produces silently wrong or flickering output.
- **Trying to preserve `Map`-every-frame dynamic buffer patterns 1:1.** It still works (`agfxBufferMap` on a `CPU_TO_GPU` buffer), but without frame-in-flight fencing around it, per-frame constant writes race the GPU reading the previous frame's data — either drive it through `agfx::ez`'s ring buffer or replicate that pattern by hand in raw AGFX.
- **Leaving slot-based binding calls (`PSSetShaderResources` etc.) as dead code instead of deleting them along with the input layout and state objects** — there's nothing on the AGFX side to bind to; delete outright rather than translate.
- **Porting stencil logic.** AGFX's pipeline depth state has no stencil fields. Flag any stencil-dependent D3D11 code to the user rather than silently dropping it.
