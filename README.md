# AGFX : Amelie's graphics library

![](.github/image.png)

AGFX is a small (<10k LOC) wrapper over D3D12 and Metal 4, designed to make it easier for indie developers to ship game on multiple platforms. It's MIT licensed.

It is designed to be bindless first, which is pretty reasonable considering any GPU that isn't older than the library's author should support it (I was born in 2006!).

## Design

AGFX takes the coding style of Vulkan/WebGPU -- C compliant, that way it's easy to make bindings for other languages.

```c
agfxDeviceCreateInfo deviceCreateInfo = {};
deviceCreateInfo.allocate = agfxAlloc;
deviceCreateInfo.free = agfxDealloc;
deviceCreateInfo.tempAllocate = agfxAlloc;
deviceCreateInfo.tempFree = agfxDealloc;
deviceCreateInfo.enableValidation = true;

agfxDevice* device = agfxDeviceCreate(&deviceCreateInfo);

//
agfxCommandBufferReset(commandBuffer);
agfxCommandBufferBegin(commandBuffer);

agfxRenderPassCreateInfo renderPassCreateInfo = {};
renderPassCreateInfo.colorAttachmentCount = 1;
renderPassCreateInfo.colorAttachments[0].renderTarget = backBufferRenderTarget;
renderPassCreateInfo.colorAttachments[0].loadOp = AGFX_LOAD_OPERATION_CLEAR;
renderPassCreateInfo.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
renderPassCreateInfo.width = (uint32_t)drawableWidth;
renderPassCreateInfo.height = (uint32_t)drawableHeight;

uint32_t uniformHandle = agfxBufferViewGetHandle(constantView);

agfxRenderPass* renderPass = agfxRenderPassBegin(commandBuffer, &renderPassCreateInfo);
agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)drawableWidth, (float)drawableHeight, 0.0f, 1.0f);
agfxRenderPassSetScissor(pass, 0, 0, width, height);
agfxRenderPassSetPipeline(pass, pipeline);
agfxRenderPassPushConstants(pass, &uniformHandle, sizeof(uniformHandle));
agfxRenderPassDrawIndexed(pass, indexBuffer, indexCount, 1, indexOffset, 0, 0);
agfxRenderPassEnd(renderPass);
```

The library also ships a C++17 header `agfx.hpp` with RAII, and agfx_ez.hpp, a library meant to facilitate D3D11/OpenGL users to switch to AGFX.

## What's included

In this repository you can find a few different folders:
- .claude : a porting agent and claude skills to facilitate porting your game to AGFX
- agfx: The main library
- agfx_demo: A self-contained demo using SDL3 and AGFX with physically based rendering, cascaded shadow maps, SSAO, HDR, raytraced mirror reflections, GPU driven frustum culling
- agfx_ez_demo: Simple self-contained demo showcasing agfx_ez.hpp
- agfx_imgui: ImGui backend for AGFX
- agfx_shader: Simple shader compiler library to go from HLSL -> AGFX bytecode
- agfx_shader_cli: A CLI to compile AGFX shaders
- data/shaders: Helpful HLSL header include, and shaders for agfx_imgui and the demo app

## AI Notice

This project was written using AI tools (Claude Code) -- I wrote most of the backend code by hand, and the LLM wrote most of the upper layer stuff (ez, demo etc). Keep it in mind when using this library.

## Getting started

### Requirements
- Mac: Apple Silicon M1 and above (M3+ for mesh shading and raytracing) and macOS 26+
- Windows: Any GPU that supports bindless

### Compiling
- Install [xmake](https://xmake.io)
- xmake
- xmake run agfx_demo
- Done!

### Libraries to link (if you're using your own build system)
On Mac:
- agfx: Metal, QuartzCore, CoreGraphics
- agfx_shader: dxcompiler, metalirconverter (Metal Shader Converter dylib)

On Windows:
- agfx: d3d12, dxgi
- agfx_shader: dxcompiler.lib

## Changelogs
- v1.2.0: Multi draw indirect update
- v1.1.0: Raytracing update
- v1.0.0: Base Metal4/D3D12 backends with basic features, missing raytracing/draw indirect. Fully usable for a video game.

## Projects that use AGFX

- Eclipse (Amélie Heinrich) : Adventure puzzle game where you set out to vanquish an evil force trying to plunge the world into darkness.
- Voxel Game (RyDawgE) : Adventure voxel game
