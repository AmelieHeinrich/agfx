---
name: agfx-porting-from-d3d12
description: ALWAYS use when porting an existing D3D12 engine or renderer to AGFX — translating ID3D12Device/CommandList/PipelineState/RootSignature/DescriptorHeap code to agfxDevice/agfxCommandBuffer/agfxRenderPipeline calls, converting HLSL from a D3D12 root-signature binding model to AGFX bindless, or mapping D3D12 concepts (fences, resource states, swap chain, descriptor heaps) onto their AGFX equivalents. Trigger on ID3D12*, D3D12_*, root signature, descriptor heap/table, CreateCommittedResource, ResourceBarrier, ExecuteCommandLists, "port to AGFX", "port from D3D12". Do NOT trigger for AGFX-native questions with no D3D12 source code involved — use the specific agfx-* skill for that subsystem instead (agfx-presentation-and-swapchain, agfx-render-targets-and-passes, agfx-synchronization, agfx-writing-bindless-shaders).
---

# Porting a D3D12 Engine to AGFX

## Overview

AGFX is a thin (~5000 LOC) bindless-first wrapper over D3D12 and Metal 4. Porting a D3D12 engine to it is mostly a *simplification*: AGFX collapses root signatures, descriptor tables, and per-draw descriptor binding into one bindless model (`ResourceHandle` + `ResourceDescriptorHeap`), so a lot of D3D12 binding machinery is deleted outright rather than translated. The parts that do translate 1:1 are resource creation, command recording, barriers, and pipeline state.

This skill is the *index* into the four subsystem skills — read it first to get the map, then dispatch into the specific skill for the subsystem you're touching:
- `agfx-presentation-and-swapchain` — swap chain / back buffer
- `agfx-render-targets-and-passes` — render targets, render passes, draws
- `agfx-synchronization` — fences, barriers, frame pacing
- `agfx-writing-bindless-shaders` — HLSL shader changes (root signature → bindless)

## Ownership

**Owns:**
- The D3D12 → AGFX concept translation table below
- What gets deleted outright (root signatures, descriptor tables, per-draw `SetGraphicsRootDescriptorTable` calls) vs. what gets translated
- Recommended porting order and how to validate each stage
- Where to find AGFX's own D3D12 backend as a reference for "what does AGFX do internally for X"

**Doesn't own:**
- Subsystem-specific API detail once you know which AGFX call replaces a given D3D12 call — that's the four skills above
- Metal-specific porting concerns (this repo doesn't touch Metal directly; AGFX's Metal 4 backend is `agfx/agfx_metal4.mm`, useful only as a curiosity, not something the porting engine author should touch)

## References

`agfx/agfx.h` is the entire public API surface — read it top to bottom once before starting; it's short enough to hold in full context. `agfx/agfx_d3d12.cpp` is AGFX's own D3D12 backend implementation — when unsure how a given AGFX call should behave in D3D12 terms, this is the authoritative answer (e.g. what resource states/flags it sets, how it maps `agglomerate`). `agfx_demo/` (particularly `deferred_renderer.cpp`, `agfx_demo_main.cpp`) is a complete reference engine already written against AGFX — structurally a good target shape for what the ported engine should look like.

## Concept Translation Table

| D3D12 | AGFX | Notes |
|---|---|---|
| `ID3D12Device` | `agfxDevice*` | `agfxDeviceCreate` also takes the allocator callbacks and picks the backend (D3D12/Metal4) at compile time |
| `ID3D12CommandQueue` | `agfxCommandQueue*` | `agfxCommandQueueCreate`, typed via `agfxCommandQueueType` |
| `ID3D12GraphicsCommandList` | `agfxCommandBuffer*` | `agfxCommandBufferCreate/Begin/End/Reset` |
| `ID3D12Fence` + `SetEventOnCompletion`/`WaitForSingleObject` | `agfxFence*` | one fence type for both CPU↔GPU and GPU↔GPU waits — see `agfx-synchronization` |
| `ID3D12RootSignature` + descriptor tables | **deleted** | AGFX is bindless-first; there is no root signature to author. Resources are accessed via `ResourceHandle` pulled from `ResourceDescriptorHeap`/`SamplerDescriptorHeap` — see `agfx-writing-bindless-shaders` |
| `ID3D12DescriptorHeap` (CBV/SRV/UAV, sampler) | **implicit / internal** | AGFX manages the bindless heap itself; you get handles back from view creation, you don't manage heap slots yourself |
| Root constants (`SetGraphicsRoot32BitConstants`) | `agfxRenderPassPushConstants`/`agfxComputePassPushConstants` | bound at `register(b0)`; the one binding mechanism that survives — see `AGFX_PUSH_CONSTANTS` in `agfx-writing-bindless-shaders` |
| `CreateCommittedResource`/`CreatePlacedResource` (texture) | `agfxTextureCreate` | `agfxTextureUsage` replaces D3D12 resource flags (`RENDER_TARGET`, `UNORDERED_ACCESS`, etc.) |
| `CreateCommittedResource` (buffer) | `agfxBufferCreate` | `agfxBufferMemoryType` replaces heap type (DEFAULT/UPLOAD/READBACK) |
| SRV/UAV/CBV descriptor (`CreateShaderResourceView` etc.) | `agfxTextureView`/`agfxBufferView` | creation returns a `ResourceHandle` for bindless shader access, rather than writing into a caller-managed heap slot |
| Sampler descriptor | `agfxSampler` | `agfxSamplerCreate`, handle obtained the same way as texture/buffer views |
| `ResourceBarrier` (transition) | `agfxCommandBufferTextureBarrier`/`BufferBarrier` | states map closely (`agfxResourceState` mirrors `D3D12_RESOURCE_STATES`); has an extra `agglomerate` flag D3D12 doesn't need — ignored on the D3D12 backend, meaningful on Metal — see `agfx-synchronization` |
| UAV barrier | `agfxComputePassTextureUAVBarrier`/`BufferUAVBarrier` | scoped to within a compute pass |
| `OMSetRenderTargets` + manual load/clear | `agfxRenderTarget` + `agfxRenderPassBegin`/`agfxRenderPassCreateInfo` attachments | load/store ops are explicit (`agfxLoadOperation`/`agfxStoreOperation`), similar to D3D12's render-pass API (`BeginRenderPass`) if the engine already uses that, otherwise new relative to bare `OMSetRenderTargets` |
| `IASetVertexBuffers`/input layout | **deleted — vertex pulling** | AGFX shaders take `SV_VertexID` and manually load from a `AGFXStructuredBuffer` in the vertex shader; there is no input-assembler vertex buffer binding — see `agfx-writing-bindless-shaders` |
| `CreateGraphicsPipelineState` | `agfxRenderPipelineCreate` | `agfxRenderPipelineCreateInfo` folds in blend/raster/depth-stencil state plus attachment formats (must match the render pass it's used in) |
| `CreateComputePipelineState` | `agfxComputePipelineCreate` | straightforward 1:1 |
| Root signature + `.hlsl` `register(t/b/u/s, spaceN)` | AGFX's `agfx.h` bindless header + `ResourceHandle` | shader HLSL itself needs rewriting, not just host code — see `agfx-writing-bindless-shaders` |
| `IDXGISwapChain` | `agfxSwapChain*` | `agfxSwapChainCreate/AcquireNextTexture/Present/Resize`; HDR toggle is destroy+recreate, not in-place — see `agfx-presentation-and-swapchain` |
| Frame-in-flight command allocator/list rotation | per-slot `agfxCommandBuffer` + one `agfxFence` | pattern is the same shape as most D3D12 engines already use — see `agfx-synchronization` |

## Recommended Porting Order

Porting shader-and-binding code before device/resource plumbing compiles but can't be tested; the reverse order lets you validate incrementally against a triangle/simple scene before tackling the whole renderer.

1. **Device, queue, swap chain.** Get `agfxDeviceCreate`/`agfxCommandQueueCreate`/`agfxSwapChainCreate` wired up and clearing the back buffer to a solid color and presenting. This validates the build/link setup (see the repo's top-level `README.md` for libraries to link on each platform) before any rendering logic is ported.
2. **Frame pacing and fences.** Port the D3D12 fence/frame-in-flight loop to `agfxFence` + per-slot command buffers (`agfx-synchronization`). Do this before resource porting — later steps assume a working drain/wait mechanism, since resource creation/destruction during porting will otherwise race in-flight GPU work.
3. **Resources.** Port texture/buffer creation calls (`CreateCommittedResource` → `agfxTextureCreate`/`agfxBufferCreate`). Drop descriptor-heap-slot management entirely; keep the returned view handles instead.
4. **One render pass end-to-end**, e.g. a single G-buffer or forward pass: render target creation, barriers into the right states, `agfxRenderPassBegin`/draw/`End` (`agfx-render-targets-and-passes`). Get one pass rendering a real mesh with the actual shaders before porting the rest of the pipeline — this is where root-signature-to-bindless mistakes surface fastest.
5. **Shaders.** Rewrite each HLSL shader's binding section: remove `register(t/b/u/s, spaceN)` declarations tied to the old root signature, replace with `AGFX_PUSH_CONSTANTS` + `ResourceHandle` fields, replace `register(t0)`-style texture/buffer/sampler declarations with `AGFXTexture2D`/`AGFXStructuredBuffer`/`AGFXSampler` `::Create(handle)` calls, replace input-assembler vertex fetch with vertex pulling from `SV_VertexID` (`agfx-writing-bindless-shaders`). Do this pass-by-pass alongside step 4, not as one giant shader-only pass — a shader rewritten without its host-side push-constant struct updated to match won't compile.
6. **Remaining passes.** Repeat steps 4–5 for every other render/compute pass in the engine.
7. **Cross-cutting**: HDR toggle, resize handling, mip generation, any raytracing/indirect-draw usage (AGFX v1.0.0 does not yet support raytracing or indirect draw — see the top-level `README.md` changelog; flag these to the user rather than silently dropping the feature or attempting an unsupported translation).

## Common Porting Pitfalls

- **Trying to preserve the descriptor-heap-slot-management code.** There's nothing to preserve — delete it. AGFX hands back a `ResourceHandle` at view-creation time; there's no caller-managed heap index arithmetic.
- **Leaving a second root CBV in the shader.** AGFX shaders only have `register(b0)` (push constants). Any additional per-frame/per-scene constant buffer must go through the "handle nested in push constants, loaded as a one-element `AGFXStructuredBuffer`" pattern in `agfx-writing-bindless-shaders`, not a second binding slot.
- **Getting `agglomerate` backwards.** It's silently correct on the D3D12 side of a port (the flag is ignored there) and only breaks once the same code path is exercised on Metal — see `agfx-synchronization` before assuming a barrier port is done just because it builds and runs on Windows.
- **Assuming raytracing/indirect draw are portable.** They're not yet implemented in AGFX v1.0.0. Surface this to the user early rather than discovering it mid-port.
