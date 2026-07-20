/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 00:07:40
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include <cstdlib>
#include <iostream>
#include <agfx/agfx.h>
#include <agfx_shader/agfx_shader_compiler.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_metal.h>
#include <imgui.h>
#include <glm/gtc/type_ptr.hpp>
#include "imgui_impl_sdl3.h"
#include <agfx_imgui/imgui_impl_agfx.h>
#include "camera.h"
#include "gltf_scene.h"
#include "deferred_renderer.h"
#include "gpu_profiler.h"
#include "demo_file_utils.h"

static void* agfxAlloc(uint64_t size)
{
    return malloc(size);
}

static void agfxDealloc(void* ptr)
{
    free(ptr);
}

static agfxShaderModule* CompileImGuiShader(agfxDevice* device, const std::string& source, agfxShaderStage stage, const char* entryPoint, agfxShaderModuleType moduleType)
{
    agfxShaderCompilerOptions options = {};
    options.stage = stage;
    strncpy(options.entryPoint, entryPoint, sizeof(options.entryPoint) - 1);
    options.sourceCode = source.empty() ? nullptr : const_cast<char*>(&source[0]);
    options.sourceCodeSize = (uint32_t)source.size();

    agfxShaderCompilerResult result = {};
    agfxCompileShader(&options, &result);

    if (result.compiledCode == nullptr || result.compiledSize == 0) {
        return nullptr;
    }

    agfxShaderModuleCreateInfo moduleInfo = {};
    moduleInfo.code = result.compiledCode;
    moduleInfo.codeSize = result.compiledSize;
    moduleInfo.entryPoint = entryPoint;
    moduleInfo.type = moduleType;
    agfxShaderModule* module = agfxShaderModuleCreate(device, &moduleInfo);

    free(result.compiledCode);
    return module;
}

int main()
{
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD) == 0) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return 1;
    }

#if GAME_MAC
    SDL_Window* window = SDL_CreateWindow("agfx_demo", 1280, 720, SDL_WINDOW_METAL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
#else
    SDL_Window* window = SDL_CreateWindow("agfx_demo", 1280, 720, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
#endif
    if (!window) {
        std::cerr << "Failed to create SDL window: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

#if GAME_MAC
    SDL_MetalView metalView = SDL_Metal_CreateView(window);
    void* metalLayer = SDL_Metal_GetLayer(metalView);
#endif

    int drawableWidth = 0, drawableHeight = 0;
    SDL_GetWindowSizeInPixels(window, &drawableWidth, &drawableHeight);

    agfxDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.allocate = agfxAlloc;
    deviceCreateInfo.free = agfxDealloc;
    deviceCreateInfo.tempAllocate = agfxAlloc;
    deviceCreateInfo.tempFree = agfxDealloc;
    deviceCreateInfo.enableValidation = true;
    agfxDevice* device = agfxDeviceCreate(&deviceCreateInfo);

    agfxDeviceInfo deviceInfo = {};
    agfxDeviceGetInfo(device, &deviceInfo);
    const bool supportsRayTracing = deviceInfo.supportsRayTracing != 0;

    agfxCommandQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.type = AGFX_COMMAND_QUEUE_TYPE_GRAPHICS;
    agfxCommandQueue* queue = agfxCommandQueueCreate(device, &queueCreateInfo);

    bool isHDR = false;
    agfxSwapChainCreateInfo swapChainCreateInfo = {};
    swapChainCreateInfo.queue = queue;
    swapChainCreateInfo.imageCount = 2;
    swapChainCreateInfo.width = (uint32_t)drawableWidth;
    swapChainCreateInfo.height = (uint32_t)drawableHeight;
    swapChainCreateInfo.isHDR = isHDR;
    swapChainCreateInfo.vsync = true;
#if GAME_MAC
    swapChainCreateInfo.handle = metalLayer;
#else
    swapChainCreateInfo.handle = SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
#endif  
    agfxSwapChain* swapChain = agfxSwapChainCreate(device, &swapChainCreateInfo);

    agfxCommandBuffer* commandBuffers[kFramesInFlight];
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        commandBuffers[i] = agfxCommandBufferCreate(device, queue);
    }
    agfxFence* frameFence = agfxFenceCreate(device);
    uint64_t frameIndex = 0;
    uint64_t fenceValue = 0;
    uint64_t slotFenceValues[kFramesInFlight] = {};

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
    ImGui_ImplSDL3_InitForMetal(window);

    std::string imguiShaderSource = ReadFile((std::string(kDataDir) + "shaders/agfx_imgui/imgui.hlsl").c_str());
    agfxShaderModule* imguiVertexShader = CompileImGuiShader(device, imguiShaderSource, AGFX_SHADER_STAGE_VERTEX, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
    agfxShaderModule* imguiFragmentShader = CompileImGuiShader(device, imguiShaderSource, AGFX_SHADER_STAGE_FRAGMENT, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
    if (!imguiVertexShader || !imguiFragmentShader) {
        std::cerr << "Failed to compile ImGui shaders" << std::endl;
        return 1;
    }

    ImGui_ImplAGFX_InitInfo imguiInitInfo = {};
    imguiInitInfo.Device = device;
    imguiInitInfo.CommandQueue = queue;
    imguiInitInfo.ColorAttachmentFormat = agfxSwapChainGetFormat(swapChain);
    imguiInitInfo.VertexShaderModule = imguiVertexShader;
    imguiInitInfo.FragmentShaderModule = imguiFragmentShader;
    imguiInitInfo.FramesInFlight = kFramesInFlight;
    if (!ImGui_ImplAGFX_Init(&imguiInitInfo)) {
        std::cerr << "Failed to initialize ImGui AGFX backend" << std::endl;
        return 1;
    }

    GpuProfiler profiler;
    profiler.Init(device, queue);

    GltfScene scene;
    std::string sponzaPath = std::string(kDataDir) + "models/Sponza/Sponza.gltf";
    if (!scene.Load(device, queue, sponzaPath.c_str())) {
        std::cerr << "Failed to load Sponza scene" << std::endl;
        return 1;
    }

    DeferredRenderer renderer;
    renderer.Init(device, agfxSwapChainGetFormat(swapChain), (uint32_t)drawableWidth, (uint32_t)drawableHeight);

    Camera camera;
    camera.position = glm::vec3(0.0f, 2.0f, 0.0f);
    camera.aspect = (float)drawableWidth / (float)drawableHeight;

    LightSettings light;

    bool rightMouseDown = false;
    bool wantHDR = false;

    // Drains all in-flight GPU work. Required before destroying/recreating any resource that
    // an already-submitted (but not yet completed) frame's command buffer might still reference,
    // e.g. the shared GBuffer/HDR/depth targets on resize, or the swapchain on the HDR toggle.
    // Every signal consumes a fresh monotonic fence value so a drain can never
    // collide with (and pre-satisfy) a later frame's completion signal.
    auto drainGPU = [&]() {
        agfxCommandQueueSignal(queue, frameFence, ++fenceValue);
        agfxFenceWait(frameFence, fenceValue, UINT64_MAX);
    };

    bool running = true;
    uint64_t lastTicks = SDL_GetTicks();
    while (running) {
#if GAME_MAC
        @autoreleasepool {
#endif
            uint64_t nowTicks = SDL_GetTicks();
            float dt = (float)(nowTicks - lastTicks) / 1000.0f;
            lastTicks = nowTicks;

            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                ImGui_ImplSDL3_ProcessEvent(&event);
                if (event.type == SDL_EVENT_QUIT) {
                    running = false;
                } else if (event.type == SDL_EVENT_WINDOW_RESIZED || event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
                    SDL_GetWindowSizeInPixels(window, &drawableWidth, &drawableHeight);
                    drainGPU();
                    agfxSwapChainResize(device, swapChain, (uint32_t)drawableWidth, (uint32_t)drawableHeight);
                    renderer.Resize(device, (uint32_t)drawableWidth, (uint32_t)drawableHeight);
                    camera.aspect = (float)drawableWidth / (float)drawableHeight;
                } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_RIGHT && !io.WantCaptureMouse) {
                    rightMouseDown = true;
                    SDL_SetWindowRelativeMouseMode(window, true);
                } else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.button == SDL_BUTTON_RIGHT) {
                    rightMouseDown = false;
                    SDL_SetWindowRelativeMouseMode(window, false);
                }
            }

            ImGui_ImplSDL3_NewFrame();
            ImGui::NewFrame();

            profiler.DrawUI();

            ImGui::Begin("Renderer");
            ImGui::Checkbox("HDR", &wantHDR);
            ImGui::SliderFloat3("Light Direction", glm::value_ptr(light.direction), -1.0f, 1.0f);
            ImGui::ColorEdit3("Light Color", glm::value_ptr(light.color));
            ImGui::SliderFloat("Light Intensity", &light.intensity, 0.0f, 10.0f);
            ImGui::Separator();
            ImGui::Text("Shadows (CSM + PCSS)");
            ImGui::SliderInt("Cascade Count", (int*)&renderer.csm.cascadeCount, 1, (int)kMaxCascades);
            ImGui::SliderFloat("Split Lambda", &renderer.csm.splitLambda, 0.0f, 1.0f);
            ImGui::SliderFloat("Shadow Distance", &renderer.csm.shadowDistance, 10.0f, 300.0f);
            static int shadowResIndex = 1; // 0=1024, 1=2048, 2=4096
            const char* shadowResLabels[] = { "1024", "2048", "4096" };
            if (ImGui::Combo("Shadow Map Resolution", &shadowResIndex, shadowResLabels, 3)) {
                drainGPU();
                renderer.RecreateShadowTargets(device, 1024u << shadowResIndex);
            }
            ImGui::SliderFloat("PCSS Light Size (UV)", &renderer.csm.lightSizeUV, 0.001f, 0.05f);
            ImGui::SliderFloat("PCSS Max Penumbra (UV)", &renderer.csm.pcssMaxPenumbraUV, 0.001f, 0.1f);
            ImGui::SliderFloat("Depth Bias Constant", &renderer.csm.depthBiasConstant, 0.0f, 0.01f);
            ImGui::Checkbox("Visualize Cascades", &renderer.csm.visualizeCascades);
            ImGui::Separator();
            ImGui::Text("SSAO");
            ImGui::Checkbox("Enabled", &renderer.ssaoSettings.enabled);
            ImGui::SliderFloat("Radius", &renderer.ssaoSettings.radius, 0.05f, 2.0f);
            ImGui::SliderFloat("Bias", &renderer.ssaoSettings.bias, 0.0f, 0.1f);
            ImGui::SliderFloat("Power", &renderer.ssaoSettings.power, 0.5f, 4.0f);
            ImGui::Separator();
            ImGui::Text("Raytraced Reflections");
            if (supportsRayTracing) {
                ImGui::Checkbox("Enabled##Reflections", &renderer.reflectionSettings.enabled);
                ImGui::BeginDisabled(!renderer.reflectionSettings.enabled);
                ImGui::SliderFloat("Metallic Threshold", &renderer.reflectionSettings.metallicThreshold, 0.0f, 1.0f);
                ImGui::SliderFloat("Roughness Threshold", &renderer.reflectionSettings.roughnessThreshold, 0.0f, 0.5f);
                ImGui::EndDisabled();
            } else {
                ImGui::TextDisabled("Ray tracing not supported on this device");
            }
            ImGui::Separator();
            ImGui::Text("Hold Right Mouse Button to look around");
            ImGui::Text("WASD to move, Shift to sprint, Q/E for down/up");
            ImGui::End();

            float mouseDx = 0.0f, mouseDy = 0.0f;
            SDL_GetRelativeMouseState(&mouseDx, &mouseDy);
            if (!io.WantCaptureKeyboard) {
                const bool* keyState = SDL_GetKeyboardState(nullptr);
                if (!rightMouseDown) {
                    mouseDx = 0.0f;
                    mouseDy = 0.0f;
                }
                camera.Update(keyState, mouseDx, mouseDy, dt);
            }

            ImGui::Render();

            if (wantHDR != isHDR) {
                // Swapchain/pipeline lifecycle changes need the GPU fully idle first.
                drainGPU();

                agfxSwapChainDestroy(device, swapChain);
                isHDR = wantHDR;
                swapChainCreateInfo.isHDR = isHDR;
                swapChain = agfxSwapChainCreate(device, &swapChainCreateInfo);
                renderer.RecreateTonemapPipeline(device, agfxSwapChainGetFormat(swapChain));
                ImGui_ImplAGFX_RecreatePipeline(agfxSwapChainGetFormat(swapChain));
            }

            // Frames-in-flight synchronization (mirrors the Metal4/D3D12 pattern): only block
            // when we're about to reuse a slot whose GPU work may not have finished yet.
            uint32_t frameSlot = (uint32_t)(frameIndex % kFramesInFlight);
            agfxFenceWait(frameFence, slotFenceValues[frameSlot], UINT64_MAX);
            profiler.BeginFrame(device, frameSlot);
            agfxCommandBuffer* commandBuffer = commandBuffers[frameSlot];

            agfxCommandBufferReset(commandBuffer);
            agfxCommandBufferBegin(commandBuffer);

            profiler.BeginScope(commandBuffer, "GBuffer");
            renderer.RenderGBuffer(device, commandBuffer, scene, camera, frameSlot);
            profiler.EndScope(commandBuffer);

            profiler.BeginScope(commandBuffer, "SSAO");
            renderer.RenderSSAO(device, commandBuffer, camera, frameSlot);
            profiler.EndScope(commandBuffer);

            profiler.BeginScope(commandBuffer, "Shadows");
            renderer.RenderShadows(device, commandBuffer, scene, camera, light, frameSlot);
            profiler.EndScope(commandBuffer);

            profiler.BeginScope(commandBuffer, "Reflections");
            renderer.RenderReflections(device, commandBuffer, scene, camera, light, frameSlot);
            profiler.EndScope(commandBuffer);

            profiler.BeginScope(commandBuffer, "Lighting");
            renderer.RenderLighting(device, commandBuffer, light, camera, frameSlot);
            profiler.EndScope(commandBuffer);

            agfxTexture* backBuffer = agfxSwapChainAcquireNextTexture(swapChain);
            agfxCommandBufferTextureBarrier(commandBuffer, backBuffer, AGFX_RESOURCE_STATE_PRESENT, AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, false);

            agfxRenderTargetCreateInfo renderTargetCreateInfo = {};
            renderTargetCreateInfo.texture = backBuffer;
            renderTargetCreateInfo.format = AGFX_TEXTURE_FORMAT_UNKNOWN;
            renderTargetCreateInfo.mipLevel = 0;
            renderTargetCreateInfo.arrayLayer = 0;
            renderTargetCreateInfo.isDepth = false;
            agfxRenderTarget* backBufferRenderTarget = agfxRenderTargetCreate(device, &renderTargetCreateInfo);

            agfxRenderPassCreateInfo renderPassCreateInfo = {};
            renderPassCreateInfo.colorAttachmentCount = 1;
            renderPassCreateInfo.colorAttachments[0].renderTarget = backBufferRenderTarget;
            renderPassCreateInfo.colorAttachments[0].loadOp = AGFX_LOAD_OPERATION_CLEAR;
            renderPassCreateInfo.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
            renderPassCreateInfo.colorAttachments[0].clearColor[0] = 0.05f;
            renderPassCreateInfo.colorAttachments[0].clearColor[1] = 0.05f;
            renderPassCreateInfo.colorAttachments[0].clearColor[2] = 0.1f;
            renderPassCreateInfo.colorAttachments[0].clearColor[3] = 1.0f;
            renderPassCreateInfo.hasDepthAttachment = false;
            renderPassCreateInfo.width = (uint32_t)drawableWidth;
            renderPassCreateInfo.height = (uint32_t)drawableHeight;
            renderPassCreateInfo.name = "Backbuffer";

            profiler.BeginScope(commandBuffer, "Backbuffer");
            agfxRenderPass* renderPass = agfxRenderPassBegin(commandBuffer, &renderPassCreateInfo);
            renderer.RenderTonemap(device, renderPass, isHDR);
            ImGui_ImplAGFX_RenderDrawData(ImGui::GetDrawData(), renderPass, (uint32_t)drawableWidth, (uint32_t)drawableHeight, frameSlot);
            agfxRenderPassEnd(renderPass);
            profiler.EndScope(commandBuffer);

            agfxRenderTargetDestroy(device, backBufferRenderTarget);

            agfxCommandBufferTextureBarrier(commandBuffer, backBuffer, AGFX_RESOURCE_STATE_RENDER_TARGET, AGFX_RESOURCE_STATE_PRESENT, 0, 0, false);
            profiler.EndFrame(commandBuffer);
            agfxCommandBufferEnd(commandBuffer);
            agfxCommandQueueSubmit(queue, &commandBuffer, 1);
            agfxSwapChainPresent(swapChain);

            slotFenceValues[frameSlot] = ++fenceValue;
            agfxCommandQueueSignal(queue, frameFence, fenceValue);
            frameIndex++;
#if GAME_MAC
        }
#endif
    }

    // Drain all frames-in-flight before tearing anything down.
    drainGPU();

    profiler.Shutdown(device);
    ImGui_ImplAGFX_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    renderer.Shutdown(device);
    scene.Destroy(device);

    agfxFenceDestroy(device, frameFence);
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        agfxCommandBufferDestroy(device, commandBuffers[i]);
    }
    agfxSwapChainDestroy(device, swapChain);
    agfxCommandQueueDestroy(device, queue);
    agfxDeviceDestroy(device);

#if GAME_MAC
    SDL_Metal_DestroyView(metalView);
#endif
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
