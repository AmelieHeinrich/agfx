-- 
-- @ Author: Amélie Heinrich @ Amélie Heinrich
-- @ Create Time: 2026-07-18 00:06:40
-- @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
-- 

add_rules("mode.debug", "mode.release", "mode.releasedbg")

set_languages("cxx17")

add_requires("libsdl3", "cgltf", "stb", "glm", "nlohmann_json")
add_requires("imgui v1.92.7-docking", { configs = { } })

add_includedirs("src", "src/agfx", {public = true})
add_linkdirs("data/dlls", {public = true})
set_rundir(".", {public = true})

-- RPaths for macOS
-- "data/dlls" is cwd-relative and only works when running from the repo root (see set_rundir).
-- "@executable_path" / "@loader_path" are binary-relative, so a copied-out binary stays loadable.
if is_plat("macosx") then
    add_rpathdirs("@executable_path", "@loader_path", "data/dlls", { public = true })
end

-- Compile flags and platform specific packages
if is_plat("windows") then
    add_defines("WIN32_LEAN_AND_MEAN", "GAME_WINDOWS", { public = true })
elseif is_plat("macosx") then
    add_defines("GAME_MAC", { public = true })
    add_cxxflags("-fobjc-arc", "-x objective-c++", { public = true })
end

includes("src/agfx")