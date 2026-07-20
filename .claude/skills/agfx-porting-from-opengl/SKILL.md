---
name: agfx-porting-from-opengl
description: ALWAYS use when porting an existing OpenGL (or OpenGL ES) engine or renderer to AGFX — translating glGenTextures/glBindTexture/glBufferData/glUseProgram/FBO code to agfxDevice/agfxCommandBuffer/agfxRenderPipeline calls, converting GLSL uniform/sampler-binding shaders to AGFX bindless HLSL, or mapping OpenGL concepts (the global bind-to-slot state machine, VAOs, FBOs, uniform blocks) onto their AGFX equivalents. Trigger on glGenBuffers/glBindBuffer/glTexImage2D/glUseProgram/glDrawArrays/glDrawElements, GLSL `uniform`/`layout(binding=...)`, VAO/VBO/EBO/FBO, "port to AGFX", "port from OpenGL". Do NOT trigger for AGFX-native questions with no OpenGL source code involved — use the specific agfx-* skill for that subsystem instead (agfx-presentation-and-swapchain, agfx-render-targets-and-passes, agfx-synchronization, agfx-writing-bindless-shaders).
---

# Porting an OpenGL Engine to AGFX

## Overview

OpenGL is a global, implicit state machine: objects are created, then made "current" by binding them to a target (`glBindTexture`, `glBindBuffer`, `glBindFramebuffer`), and later calls implicitly act on whatever is currently bound. There is no command buffer, no resource-state/barrier concept (outside of `glMemoryBarrier` for compute/image access, which most engines under-use), and shader resource binding goes through named uniforms/samplers resolved by string or explicit `layout(binding=N)` slots. AGFX has none of this: everything is an explicit object handle passed as a function argument, commands are recorded into an `agfxCommandBuffer` and submitted explicitly, resource-state transitions are the caller's job, and shaders access resources by `ResourceHandle` through a fully bindless model rather than named/slot-bound uniforms.

This is the widest conceptual gap of any of the porting skills — OpenGL's implicit binding and lack of GPU-driven synchronization means most of what a GL renderer does is either deleted (bind calls) or newly authored (barriers, fences, explicit render passes) rather than translated line-for-line. If the target engine's rendering style is otherwise simple (immediate draw calls, no manual multithreaded command recording), consider porting to `agfx::ez` (`using-agfx-ez`) instead of raw AGFX — its immediate-mode API and automatic-for-the-common-case barrier tracking are the closest thing AGFX has to GL's implicit model.

## Ownership

**Owns:**
- The OpenGL → AGFX concept translation table below
- What gets deleted outright (bind-to-target calls, VAOs) vs. what maps onto an AGFX-shaped equivalent (FBOs → render targets/passes, uniform blocks → constant buffers)
- Recommended porting order, and where GL engines most often hide bugs once ported (missing barriers, alpha-flipped textures, NDC/clip-space differences)
- When to recommend `agfx::ez` for this kind of port

**Doesn't own:**
- Subsystem-specific API detail once the AGFX call is identified — the four `agfx-*` subsystem skills, or `using-agfx-ez` if porting to the ez layer
- D3D11/D3D12/Vulkan/Metal-specific concept mapping — the other `agfx-porting-from-*` skills

## References

`agfx/agfx.h` is the entire public API surface — read it top to bottom once before starting. `agfx/agfx_ez.hpp` (`using-agfx-ez`) if recommending/using the immediate-mode path. `agfx_demo/deferred_renderer.cpp` and `agfx_demo/agfx_demo_main.cpp` are the raw-AGFX reference; `agfx_ez_demo/agfx_ez_demo_main.cpp` the ez-layer reference, structurally closer to a small GL app.

## Concept Translation Table

| OpenGL | AGFX | Notes |
|---|---|---|
| GL context (implicit, created by the windowing layer) | `agfxDevice*` | `agfxDeviceCreate`; no "make current" — the device/command buffer is passed explicitly everywhere |
| Default framebuffer / `glfwSwapBuffers`/`SDL_GL_SwapWindow` | `agfxSwapChain*` | `agfxSwapChainCreate/AcquireNextTexture/Present` — see `agfx-presentation-and-swapchain`; GL's implicit default framebuffer becomes an explicitly acquired back-buffer texture |
| `glGenTextures`/`glBindTexture`/`glTexImage2D`/`glTexStorage2D` | `agfxTexture*` | `agfxTextureCreate`; no bind-to-target — the texture object is passed directly to whatever needs it |
| `glTexParameteri` (`GL_TEXTURE_MIN_FILTER` etc.) | `agfxSampler*` | GL couples filtering to the texture object; AGFX splits it into a separate `agfxSamplerCreate` object, closer to Vulkan/D3D12/Metal — a texture can be sampled with different samplers without duplicating it |
| `glGenBuffers`/`glBindBuffer`/`glBufferData` (`GL_ARRAY_BUFFER`, `GL_ELEMENT_ARRAY_BUFFER`, `GL_UNIFORM_BUFFER`, `GL_SHADER_STORAGE_BUFFER`) | `agfxBuffer*` | `agfxBufferUsage` bitflags replace the bind target; `GL_STATIC_DRAW`/`GL_DYNAMIC_DRAW` roughly map to `agfxBufferMemoryType` `GPU_ONLY` vs `CPU_TO_GPU` |
| `glMapBuffer`/`glMapBufferRange`/`glUnmapBuffer` | `agfxBufferMap`/`agfxBufferUnmap` | only valid on `CPU_TO_GPU`/`GPU_TO_CPU` buffers; no coherent/persistent-mapping distinction to preserve, map-write-unmap per frame |
| `glGenVertexArrays`/`glBindVertexArray`/`glVertexAttribPointer` (VAO) | **deleted — vertex pulling** | there is no vertex-attribute binding at all; vertex shaders take `SV_VertexID` and manually load from an `AGFXStructuredBuffer` — see `agfx-writing-bindless-shaders` |
| `glGenFramebuffers`/`glBindFramebuffer`/`glFramebufferTexture2D` (FBO) | `agfxRenderTarget*` + `agfxRenderPassBegin` | GL's "bind an FBO, its attachments are already configured" becomes an explicit per-pass render target + attachment list with load/store ops — see `agfx-render-targets-and-passes` |
| `glClear`/`glClearColor`/`glClearDepth` | `agfxLoadOp` = `AGFX_LOAD_OPERATION_CLEAR` + `clearColor` on the render pass attachment | clearing is declared when starting the pass, not a separate call |
| `glUseProgram` + separately linked/compiled `.vert`/`.frag` GLSL | `agfxShaderModule*` + `agfxRenderPipelineCreateInfo` | AGFX shaders are HLSL, compiled via `agfxCompileShader` (DXC → DXIL, translated on macOS) — GLSL source needs rewriting to HLSL, not just recompiling; see `agfx-writing-bindless-shaders` |
| `uniform sampler2D`/`layout(binding=N) uniform texture...` + `glUniform1i`/`glBindTextureUnit` | **deleted — bindless** | no named uniforms and no texture units; every resource access goes through a `ResourceHandle` passed via push constants — see `agfx-writing-bindless-shaders` |
| Uniform blocks (`layout(std140) uniform`) + `glBindBufferBase(GL_UNIFORM_BUFFER, ...)` | `ResourceHandle` to a constant buffer, nested in push constants, loaded as `AGFXStructuredBuffer` | no separate uniform-block binding point — see the "scene/per-frame constants" pattern in `agfx-writing-bindless-shaders` |
| SSBOs (`layout(std430) buffer`) + `glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ...)` | `AGFXStructuredBuffer<T>`/`AGFXRWStructuredBuffer<T>` via `ResourceHandle` | same bindless pattern; read-only vs. read/write picks the wrapper, matching `AGFX_BUFFER_USAGE_SHADER_READ`/`SHADER_WRITE` on the host buffer |
| Blend state (`glEnable(GL_BLEND)`, `glBlendFunc`, `glBlendEquation`) | fields on `agfxRenderPipelineCreateInfo` (`blendEnable[]`, `srcColorBlendFactor[]`, `colorBlendOp[]`) | baked into the pipeline object at creation, not toggled per draw |
| Raster state (`glEnable(GL_CULL_FACE)`, `glCullFace`, `glFrontFace`, `glPolygonMode`) | fields on `agfxRenderPipelineCreateInfo` (`cullMode`, `frontFace`, `fillMode`) | same — no global toggled state, it's part of the pipeline |
| Depth state (`glEnable(GL_DEPTH_TEST)`, `glDepthFunc`, `glDepthMask`) | fields on `agfxRenderPipelineCreateInfo` (`depthTestEnable`, `depthCompareOp`, `depthWriteEnable`) | same; AGFX has no stencil support to map — flag stencil-dependent GL code to the user |
| `glDrawArrays`/`glDrawElements`/`glDrawArraysInstanced` | `agfxRenderPassDraw`/`agfxRenderPassDrawIndexed` | recorded on the render pass object, indices/vertex counts map directly |
| `glDispatchCompute` | `agfxComputePassDispatch` | recorded within an `agfxComputePass`, not a bare context call |
| `glMemoryBarrier`, or (commonly) nothing at all | `agfxCommandBufferTextureBarrier`/`BufferBarrier`, `agfxComputePassTextureUAVBarrier`/`BufferUAVBarrier` | GL engines frequently under-barrier since the driver often papers over missing `glMemoryBarrier` calls in practice; every render-target-write → shader-read and compute read/write ordering AGFX needs an *explicit* barrier for — see `agfx-synchronization` |
| No explicit CPU/GPU fence (implicit via `glFinish`/driver-managed, or `glFenceSync` if used) | `agfxFence` + frame-in-flight command buffer rotation | if the engine used `glFenceSync`/`glClientWaitSync` for CPU readback pacing, that maps onto `agfxFence`; if not (most GL engines rely on implicit driver pacing), this is new code — see `agfx-synchronization` |
| NDC depth range `[-1, 1]`, texture origin bottom-left | NDC depth range `[0, 1]` (matches D3D12/Metal), texture origin top-left | not a call-level translation but a math/data one — see Common Pitfalls below |

## Recommended Porting Order

1. **Decide raw AGFX vs. `agfx::ez`.** GL's implicit global-state model is the closest thing AGFX has an equivalent for in `agfx::ez` (`using-agfx-ez`): immediate `SetRenderTargets`/`SetPipeline`/`Draw` calls, automatic-for-the-common-case barrier tracking, one-call texture/buffer creation. Recommend it unless the engine needs multithreaded command recording or resource types the ez layer doesn't cover.
2. **Device, queue, swap chain.** Wire up `agfxDeviceCreate`/`agfxCommandQueueCreate`/`agfxSwapChainCreate` (or `agfx::ez::Context`) and get a cleared back buffer presenting before porting any rendering logic — this also validates the build/link setup replacing GL context creation (GLFW/SDL/EGL/WGL) and function-pointer loading (GLAD/GLEW), which AGFX doesn't need.
3. **NDC/depth-range and texture-origin fixes.** Before porting shaders or texture uploads, decide how the engine's projection matrices and UV/texture-load code will be adjusted for AGFX's `[0,1]` depth range and top-left texture origin (both differ from GL's `[-1,1]`/bottom-left convention) — get this right once, early, rather than debugging it per-shader later.
4. **Frame pacing and fences** (raw AGFX only — `agfx::ez::Context` handles this internally). Most GL engines have nothing to port here since the driver paced frames implicitly; this is new code. See `agfx-synchronization`.
5. **Resources.** Port `glGenTextures`/`glTexImage2D` and `glGenBuffers`/`glBufferData` calls to `agfxTextureCreate`/`agfxBufferCreate` (or the `agfx::ez::Context` one-call equivalents). Delete VAO setup entirely — there's nothing to port it to.
6. **One render pass end-to-end.** Convert one FBO-bind-and-draw sequence to `agfxRenderPassBegin`/draw/`End` (`agfx-render-targets-and-passes`), adding the barriers GL never required by hand (`agfx-synchronization`, or `agfx::ez::Context::SetRenderTargets`/`TransitionTexture` for ez-tracked resources).
7. **Shaders.** Rewrite GLSL to HLSL per shader (not a source-level auto-translation — different language, different binding model): remove `uniform`/`layout(binding=...)` declarations, replace with `AGFX_PUSH_CONSTANTS` + `ResourceHandle` fields and `AGFXTexture2D`/`AGFXStructuredBuffer`/`AGFXSampler::Create(handle)` calls, replace VAO-driven vertex fetch with vertex pulling from `SV_VertexID` (`agfx-writing-bindless-shaders`). Port a shader and update its host-side push-constant struct together, not as separate passes.
8. **Remaining passes**, then **cross-cutting**: stencil-dependent logic (unsupported, flag it), MSAA (verify current AGFX support before assuming it maps), HDR/resize (`agfx-presentation-and-swapchain`).

## Common Porting Pitfalls

- **Assuming the driver still inserts barriers.** GL engines frequently have zero or near-zero `glMemoryBarrier` calls because drivers are lenient; porting that silence to AGFX produces missing barriers, not a compile error — every write-then-read across passes needs one explicitly (`agfx-synchronization`).
- **Leaving the GL depth-range/texture-origin convention baked into shader math.** Projection matrices tuned for `[-1,1]` NDC depth and UV math assuming a bottom-left texture origin will render with inverted or squashed depth and flipped textures on AGFX until adjusted for `[0,1]`/top-left.
- **Trying to preserve bind-to-target calls (`glBindTexture`, `glBindBuffer`, `glBindFramebuffer`) as dead code alongside VAOs** — delete them along with vertex attribute setup; there is nothing on the AGFX side to bind to.
- **Porting GLSL source mechanically instead of rewriting to HLSL.** Uniform/sampler declarations, matrix multiply order (GLSL is column-major by convention, HLSL row-major by default), and the binding model are different enough that a shader needs to be rewritten against `agfx-writing-bindless-shaders`, not transliterated keyword-by-keyword.
- **Porting stencil-dependent logic.** AGFX's pipeline depth state has no stencil fields — flag stencil-dependent GL code to the user rather than silently dropping it.
