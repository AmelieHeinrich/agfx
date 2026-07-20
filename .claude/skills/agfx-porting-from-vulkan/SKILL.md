---
name: agfx-porting-from-vulkan
description: ALWAYS use when porting an existing Vulkan engine or renderer to AGFX — translating VkDevice/VkCommandBuffer/VkPipeline/VkDescriptorSet/VkRenderPass code to agfxDevice/agfxCommandBuffer/agfxRenderPipeline calls, converting GLSL/HLSL from Vulkan's descriptor-set binding model (or VK_EXT_descriptor_buffer/bindless) to AGFX bindless HLSL, or mapping Vulkan concepts (descriptor sets/pools/layouts, image layouts, pipeline barriers, render passes/dynamic rendering, timeline semaphores) onto their AGFX equivalents. Trigger on VkDevice/VkCommandBuffer/VkImage/VkBuffer/VkPipeline/VkDescriptorSet*, vkCmdPipelineBarrier, VkImageLayout, vkQueueSubmit, VkRenderPass/vkCmdBeginRendering, "port to AGFX", "port from Vulkan". Do NOT trigger for AGFX-native questions with no Vulkan source code involved — use the specific agfx-* skill for that subsystem instead (agfx-presentation-and-swapchain, agfx-render-targets-and-passes, agfx-synchronization, agfx-writing-bindless-shaders).
---

# Porting a Vulkan Engine to AGFX

## Overview

Vulkan is, of all the source APIs, structurally the closest to AGFX: both use explicit command buffers, explicit resource-state/layout transitions, explicit GPU/CPU and GPU/GPU synchronization primitives, and (once the engine already uses descriptor indexing / `VK_EXT_descriptor_buffer` / bindless-style descriptor sets) a similar "handle-based" resource access story. Porting from Vulkan is mostly a *simplification and renaming* exercise rather than a structural rewrite: AGFX collapses descriptor sets, layouts, and pools into one implicit bindless heap, and folds `VkRenderPass`/`VkFramebuffer` (or dynamic rendering's `vkCmdBeginRendering`) into `agfxRenderTarget`/`agfxRenderPassBegin`. The parts that map almost 1:1 are command buffers, barriers/layout transitions, fences/semaphores, and pipeline creation.

If the source engine still uses classic per-material `VkDescriptorSet`s (not already bindless/descriptor-indexed), expect the binding-model rewrite to be the most invasive part of the port — same shape of work as a D3D12-root-signature port, just starting from descriptor sets instead of root signatures.

## Ownership

**Owns:**
- The Vulkan → AGFX concept translation table below
- What gets deleted outright (descriptor sets/pools/layouts, `VkFramebuffer`) vs. what maps closely (barriers, fences/semaphores, render pass shape)
- Recommended porting order and how to validate each stage
- Where to find AGFX's own backends as a reference for "what does AGFX do internally for X"

**Doesn't own:**
- Subsystem-specific API detail once you know which AGFX call replaces a given Vulkan call — that's the four `agfx-*` subsystem skills
- D3D12/D3D11/Metal/OpenGL-specific concept mapping — the sibling `agfx-porting-from-*` skills

## References

`agfx/agfx.h` is the entire public API surface — read it top to bottom once before starting; it's short enough to hold in full context. `agfx/agfx_d3d12.cpp` and `agfx/agfx_metal4.mm` are AGFX's own backend implementations — useful when unsure how a given AGFX call should behave in terms closer to Vulkan's explicit model (e.g. what a barrier actually does under the hood). `agfx_demo/` (particularly `deferred_renderer.cpp`, `agfx_demo_main.cpp`) is a complete reference engine already written against AGFX — a good target shape for a ported Vulkan renderer.

## Concept Translation Table

| Vulkan | AGFX | Notes |
|---|---|---|
| `VkInstance` + `VkPhysicalDevice` + `VkDevice` | `agfxDevice*` | `agfxDeviceCreate` collapses instance/physical-device selection/logical-device creation into one call; no separate instance/extension enumeration step |
| `VkQueue` | `agfxCommandQueue*` | `agfxCommandQueueCreate`, typed via `agfxCommandQueueType` (graphics/compute/transfer), roughly matching queue family specialization |
| `VkCommandPool` + `VkCommandBuffer` | `agfxCommandBuffer*` | `agfxCommandBufferCreate`/`Begin`/`End`/`Reset` — no separate pool object; allocation/reset is folded into the command buffer itself |
| `VkFence` (CPU wait) / `VkSemaphore` (binary or timeline, GPU wait) | `agfxFence*` | one fence type unifies both roles — timeline-semaphore-style monotonically increasing values for both CPU↔GPU (`agfxFenceWait`) and GPU↔GPU (`agfxCommandQueueSignal`/`Wait`) sync; binary semaphores have no direct analog, model them as a timeline value instead — see `agfx-synchronization` |
| `vkQueueSubmit` (with wait/signal semaphore arrays) | `agfxCommandQueueSubmit` + `agfxCommandQueueSignal`/`agfxCommandQueueWait` | submission and GPU-side signal/wait are separate calls in AGFX rather than one submit-info struct |
| `VkDescriptorSetLayout`/`VkDescriptorPool`/`VkDescriptorSet`/`vkUpdateDescriptorSets` | **deleted** | AGFX is bindless-first; there is no descriptor set to allocate or update. Resources are accessed via `ResourceHandle` pulled from an implicit bindless heap — see `agfx-writing-bindless-shaders`. If the source engine already uses `VK_EXT_descriptor_buffer`/descriptor indexing with a bindless heap of its own, this step is mostly deleting bookkeeping that AGFX now does internally |
| `VkPipelineLayout` (descriptor set layouts + push constant ranges) | **implicit**, push-constant range only | AGFX's only "pipeline layout" concept is the fixed push-constant block (`register(b0)`, `AGFX_PUSH_CONSTANTS`) plus the optional draw-ID constant (`register(b1)`) — see `agfx-writing-bindless-shaders` |
| `vkCmdPushConstants` | `agfxRenderPassPushConstants`/`agfxComputePassPushConstants` | same purpose, recorded on the pass object rather than the raw command buffer |
| `VkImage` + `vkAllocateMemory`/VMA + `VkImageView` | `agfxTexture*` + `agfxTextureView*` | `agfxTextureCreate` folds allocation into creation (no separate memory-binding step to manage); `agfxTextureUsage` bitflags replace `VkImageUsageFlags` |
| `VkBuffer` + `vkAllocateMemory`/VMA | `agfxBuffer*` | `agfxBufferUsage` replaces `VkBufferUsageFlags`; `agfxBufferMemoryType` replaces the memory-type/heap selection (`DEVICE_LOCAL` → `GPU_ONLY`, `HOST_VISIBLE|HOST_COHERENT` → `CPU_TO_GPU`/`GPU_TO_CPU`) |
| `vkMapMemory`/`vkUnmapMemory` | `agfxBufferMap`/`agfxBufferUnmap` | only valid on `CPU_TO_GPU`/`GPU_TO_CPU` buffers, same map-write-unmap shape |
| `VkImageLayout` (`vkCmdPipelineBarrier`/`vkCmdPipelineBarrier2` image barriers) | `agfxResourceState` + `agfxCommandBufferTextureBarrier` | states map closely conceptually (`AGFX_RESOURCE_STATE_RENDER_TARGET` ≈ `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`, `AGFX_RESOURCE_STATE_PIXEL_SHADER_RESOURCE` ≈ `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`, `AGFX_RESOURCE_STATE_PRESENT` ≈ `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`); AGFX has an extra `agglomerate` flag with no Vulkan equivalent — see `agfx-synchronization` |
| Buffer memory barriers (`vkCmdPipelineBarrier2` buffer barriers) | `agfxCommandBufferBufferBarrier` | same state-transition model as texture barriers |
| `VkMemoryBarrier`/`VK_ACCESS_SHADER_WRITE_BIT` → `VK_ACCESS_SHADER_READ_BIT` between compute dispatches | `agfxComputePassTextureUAVBarrier`/`agfxComputePassBufferUAVBarrier` | scoped to within a compute pass, purpose-built for the read-after-write-same-resource case |
| `VkRenderPass`+`VkFramebuffer` (classic) or `vkCmdBeginRendering`/`VkRenderingInfo` (dynamic rendering) | `agfxRenderTarget` + `agfxRenderPassBegin`/`agfxRenderPassCreateInfo` | AGFX's model is closest to dynamic rendering — attachments and load/store ops specified per-pass, no separate framebuffer-compatibility object to manage; `VkAttachmentLoadOp`/`StoreOp` map directly to `agfxLoadOp`/`agfxStoreOp` — see `agfx-render-targets-and-passes` |
| `VkPipeline` (graphics) + `VkGraphicsPipelineCreateInfo` | `agfxRenderPipelineCreate` | blend/raster/depth-stencil state (minus stencil, which AGFX doesn't support) plus attachment formats folded into one `agfxRenderPipelineCreateInfo`, same as Vulkan's monolithic pipeline object |
| `VkPipeline` (compute) | `agfxComputePipelineCreate` | straightforward 1:1 |
| `VkShaderModule` + SPIR-V (from GLSL/HLSL via glslang/DXC) | `agfxShaderModule*` | AGFX shaders are HLSL compiled through `agfxCompileShader` (DXC → DXIL, translated to Metal IR on macOS) rather than SPIR-V — if the source is GLSL, it needs rewriting to HLSL, not just recompiling; if already HLSL compiled to SPIR-V, the binding-model rewrite is still required (bindless push constants, no `layout(set=,binding=)`) — see `agfx-writing-bindless-shaders` |
| `layout(set=N, binding=M)` (descriptor-set-bound) or `layout(binding=M) uniform` (bindless-heap-indexed, if already using descriptor indexing) | `ResourceHandle` fields on the push-constant struct + `ResourceDescriptorHeap`/`SamplerDescriptorHeap` | if the source shader is already bindless-style (indexing a big `sampler2D descriptors[]` array with a push-constant index), this is largely a rename; if it's classic per-set binding, it's a structural rewrite — see `agfx-writing-bindless-shaders` |
| `VkVertexInputBindingDescription`/`VkVertexInputAttributeDescription` (vertex input state) | **deleted — vertex pulling** | AGFX shaders take `SV_VertexID` and manually load from an `AGFXStructuredBuffer`; there is no vertex input state to configure — see `agfx-writing-bindless-shaders` |
| `VkSwapchainKHR` + `vkAcquireNextImageKHR`/`vkQueuePresentKHR` | `agfxSwapChain*` | `agfxSwapChainCreate/AcquireNextTexture/Present/Resize`; AGFX handles the acquire/present semaphore bookkeeping internally — see `agfx-presentation-and-swapchain` |
| `VkSampler` | `agfxSampler*` | `agfxSamplerCreate`, handle obtained the same way as texture/buffer views (bindless), not bound into a descriptor set |
| NDC depth range `[0, 1]` (Vulkan already matches AGFX), clip-space Y sign flip conventions | matches AGFX directly | no depth-range fix needed (unlike GL/D3D11 ports); double-check any existing Y-flip workaround the engine has for swap chain vs. off-screen targets still makes sense under AGFX's own convention |

## Recommended Porting Order

Because Vulkan and AGFX share the same broad shape (explicit command buffers, explicit barriers, explicit fences), porting shader-and-binding code before device/resource plumbing is somewhat less risky here than for other source APIs — but still validate incrementally rather than attempting the whole renderer at once.

1. **Device, queue, swap chain.** Replace instance/physical-device/logical-device/queue creation with `agfxDeviceCreate`/`agfxCommandQueueCreate`, and `VkSwapchainKHR` setup with `agfxSwapChainCreate`. Get a cleared back buffer presenting before porting rendering logic — validates the build/link setup (no more Vulkan loader/extension/validation-layer bootstrapping).
2. **Frame pacing and fences.** Port the frame-in-flight fence/semaphore rotation to `agfxFence` + per-slot command buffers (`agfx-synchronization`) — if the engine already uses timeline semaphores this is close to a rename; if it uses binary semaphores + per-frame fences, consolidate onto AGFX's single monotonic-value fence model.
3. **Resources.** Port `VkImage`/`VkBuffer` (+ VMA/manual allocation) creation to `agfxTextureCreate`/`agfxBufferCreate`. Drop descriptor-set/descriptor-pool bookkeeping; keep the returned view/handle objects instead — see `agfx-writing-bindless-shaders`.
4. **One render pass end-to-end.** Convert one `vkCmdBeginRenderPass`/`vkCmdBeginRendering` sequence (with its layout transitions) to `agfxRenderPassBegin`/draw/`End` (`agfx-render-targets-and-passes`), including barriers into the right states beforehand (`agfx-synchronization`) — this is where descriptor-set-to-bindless mistakes surface fastest if the source wasn't already bindless.
5. **Shaders.** Rewrite each shader's binding section: remove `layout(set=,binding=)` descriptor declarations, replace with `AGFX_PUSH_CONSTANTS` + `ResourceHandle` fields and `AGFXTexture2D`/`AGFXStructuredBuffer`/`AGFXSampler::Create(handle)` calls, replace vertex-input-state fetch with vertex pulling from `SV_VertexID` (`agfx-writing-bindless-shaders`). Do this pass-by-pass alongside step 4 — a shader rewritten without its host-side push-constant struct updated in the same pass won't compile.
6. **Remaining passes.** Repeat steps 4–5 for every other render/compute pass.
7. **Cross-cutting**: any stencil-dependent logic (unsupported in AGFX — flag to the user), raytracing (`VkAccelerationStructureKHR`/ray tracing pipelines — AGFX v1.0.0 doesn't support raytracing; flag rather than silently drop), multi-draw indirect (`vkCmdDrawIndirect`/`MultiIndirect` — also unsupported), HDR/resize handling (`agfx-presentation-and-swapchain`).

## Common Porting Pitfalls

- **Trying to preserve descriptor set/pool/layout management.** If the source is classic (non-bindless) Vulkan, there's real deletion work here, not just a rename — `agfx-writing-bindless-shaders` for the replacement pattern.
- **Assuming `agglomerate` has a Vulkan equivalent.** It doesn't — Vulkan already requires explicit per-barrier scope/stage masks, so every AGFX barrier from a Vulkan port should generally pass `agglomerate = true` (ordinary transitions) to get Metal-side hazard tracking; see `agfx-synchronization` for the one exception (present-adjacent transitions).
- **Assuming raytracing or indirect draw are portable.** Neither is implemented in AGFX v1.0.0. Surface this to the user early if the source engine uses `VkAccelerationStructureKHR` or `vkCmdDrawIndirect`/`MultiIndirect`, rather than discovering it mid-port.
- **Porting stencil-dependent logic** (`VK_FORMAT_D24_UNORM_S8_UINT`-based effects, stencil-buffer masking). AGFX's pipeline depth state has no stencil fields — flag it rather than silently dropping it.
- **Missing that AGFX's depth range already matches Vulkan's `[0,1]`** and re-applying a GL-style depth-range fix that isn't needed, or missing a genuine Y-flip discrepancy between the source's off-screen-vs-swapchain convention and AGFX's own.
