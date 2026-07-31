# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

set(ncnn_vulkan_uses_webgpu OFF)
if(NCNN_VULKAN AND CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    if(NOT NCNN_SIMPLEVK)
        message(FATAL_ERROR "Emscripten Vulkan requires NCNN_SIMPLEVK")
    endif()
    if(NCNN_SYSTEM_GLSLANG)
        message(FATAL_ERROR "Emscripten Vulkan requires the bundled glslang")
    endif()

    set(ncnn_vulkan_uses_webgpu ON)
endif()

if(ncnn_vulkan_uses_webgpu)
    if(CMAKE_VERSION VERSION_LESS "3.22")
        message(FATAL_ERROR "Emscripten Vulkan requires CMake 3.22 or newer")
    endif()

    include(${CMAKE_CURRENT_LIST_DIR}/vkwebgpu_versions.cmake)
    set(VKWEBGPU_NODE_EXECUTABLE "" CACHE FILEPATH "Node.js executable for Emscripten Vulkan tests")
    set(NCNN_TEST_CHROME "" CACHE FILEPATH "Chrome executable for ncnn WebGPU tests")
    set(NCNN_TEST_WEBGPU_ADAPTER "swiftshader" CACHE STRING "WebGPU browser test adapter (swiftshader or hardware)")
    set_property(CACHE NCNN_TEST_WEBGPU_ADAPTER PROPERTY STRINGS swiftshader hardware)
    if(NOT NCNN_TEST_WEBGPU_ADAPTER STREQUAL "swiftshader" AND NOT NCNN_TEST_WEBGPU_ADAPTER STREQUAL "hardware")
        message(FATAL_ERROR "NCNN_TEST_WEBGPU_ADAPTER must be swiftshader or hardware")
    endif()

    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        OUTPUT_VARIABLE VKWEBGPU_EMCC_VERSION_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    string(FIND "${VKWEBGPU_EMCC_VERSION_OUTPUT}" "${VKWEBGPU_EMSDK_VERSION}" VKWEBGPU_EMCC_VERSION_POS)
    if(VKWEBGPU_EMCC_VERSION_POS EQUAL -1)
        message(FATAL_ERROR "Emscripten Vulkan requires emsdk ${VKWEBGPU_EMSDK_VERSION}, but ${CMAKE_CXX_COMPILER} reports:\n${VKWEBGPU_EMCC_VERSION_OUTPUT}")
    endif()

    set(VKWEBGPU_EMDAWNWEBGPU_PORT "${CMAKE_SOURCE_DIR}/cmake/ports/${VKWEBGPU_EMDAWNWEBGPU_PORT_FILENAME}")
    set(VKWEBGPU_CLOSURE_EXTERNS "${CMAKE_SOURCE_DIR}/cmake/ports/webgpu_immediates.externs.js")
    if(NOT EXISTS "${VKWEBGPU_EMDAWNWEBGPU_PORT}")
        message(FATAL_ERROR "Missing frozen Emdawnwebgpu port: ${VKWEBGPU_EMDAWNWEBGPU_PORT}")
    endif()
    if(NOT EXISTS "${VKWEBGPU_CLOSURE_EXTERNS}")
        message(FATAL_ERROR "Missing WebGPU immediates Closure externs: ${VKWEBGPU_CLOSURE_EXTERNS}")
    endif()
    file(SHA256 "${VKWEBGPU_EMDAWNWEBGPU_PORT}" VKWEBGPU_ACTUAL_PORT_SHA256)
    if(NOT VKWEBGPU_ACTUAL_PORT_SHA256 STREQUAL VKWEBGPU_EMDAWNWEBGPU_PORT_SHA256)
        message(FATAL_ERROR "Emdawnwebgpu port SHA256 mismatch: expected ${VKWEBGPU_EMDAWNWEBGPU_PORT_SHA256}, got ${VKWEBGPU_ACTUAL_PORT_SHA256}")
    endif()

    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/dawn/CMakeLists.txt")
        message(FATAL_ERROR "The Dawn submodule is missing. Run \"git submodule update --init dawn\" and \"python3 tools/webgpu/bootstrap_dawn.py\".")
    endif()
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/dawn/out/.ncnn-webgpu-deps-stamp")
        message(FATAL_ERROR "Dawn dependencies are missing. Run \"python3 tools/webgpu/bootstrap_dawn.py\".")
    endif()

    message(STATUS "vkwebgpu Dawn/Tint commit: ${VKWEBGPU_DAWN_COMMIT}")
    message(STATUS "vkwebgpu Emdawnwebgpu port: ${VKWEBGPU_EMDAWNWEBGPU_PORT_FILENAME} (${VKWEBGPU_EMDAWNWEBGPU_PORT_SHA256})")
    message(STATUS "vkwebgpu emsdk: ${VKWEBGPU_EMSDK_VERSION}")
endif()

function(ncnn_add_vkwebgpu_dependencies)
    set(CMAKE_CXX_STANDARD 20)

    set(DAWN_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_MONOLITHIC_LIBRARY OFF CACHE STRING "" FORCE)
    set(DAWN_BUILD_NODE_BINDINGS OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_PROTOBUF OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DAWN_FETCH_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_D3D11 OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_D3D12 OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_DESKTOP_GL OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_METAL OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_NULL OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_OPENGLES OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_SPIRV_VALIDATION OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
    set(DAWN_USE_GLFW OFF CACHE BOOL "" FORCE)
    set(DAWN_USE_WAYLAND OFF CACHE BOOL "" FORCE)
    set(DAWN_USE_X11 OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_GLSL_VALIDATOR OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_GLSL_WRITER OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_HLSL_WRITER OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_IR_BINARY OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_MSL_WRITER OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_NULL_WRITER OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_SPV_READER ON CACHE BOOL "" FORCE)
    set(TINT_BUILD_SPV_WRITER OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_TINTD OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_WGSL_READER OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_WGSL_WRITER ON CACHE BOOL "" FORCE)
    set(TINT_ENABLE_IR_DUMPING OFF CACHE BOOL "" FORCE)

    set(SPIRV-Headers_SOURCE_DIR ${CMAKE_SOURCE_DIR}/dawn/third_party/spirv-headers/src)
    set(SPIRV_HEADERS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(SPIRV_HEADERS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        ${SPIRV-Headers_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/spirv-headers
        EXCLUDE_FROM_ALL)

    # Tint parses SPIR-V through spvtools::opt. vkwebgpu also freezes Vulkan
    # specialization constants before Tint so inactive interfaces can be removed.
    set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
    set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
    set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
    set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        ${CMAKE_SOURCE_DIR}/dawn/third_party/spirv-tools/src
        ${CMAKE_BINARY_DIR}/spirv-tools
        EXCLUDE_FROM_ALL)

    add_subdirectory(${CMAKE_SOURCE_DIR}/dawn ${CMAKE_BINARY_DIR}/dawn EXCLUDE_FROM_ALL)
endfunction()

function(ncnn_configure_vkwebgpu_target target)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_link_libraries(${target} PRIVATE "$<BUILD_INTERFACE:emdawnwebgpu_c_include>")
    target_compile_options(${target} PUBLIC
        "$<BUILD_INTERFACE:--use-port=${VKWEBGPU_EMDAWNWEBGPU_PORT}:cpp_bindings=false>")
    target_link_options(${target} PUBLIC
        "$<BUILD_INTERFACE:--use-port=${VKWEBGPU_EMDAWNWEBGPU_PORT}:cpp_bindings=false>"
        "-sASYNCIFY=1"
        "-sASYNCIFY_STACK_SIZE=${VKWEBGPU_ASYNCIFY_STACK_SIZE}"
        "-sSTACK_SIZE=${VKWEBGPU_WASM_STACK_SIZE}"
        "$<$<CONFIG:Release>:--closure=1>"
        "$<$<CONFIG:Release>:--closure-args=--externs=${VKWEBGPU_CLOSURE_EXTERNS}>")
    target_include_directories(${target} PRIVATE
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/dawn/third_party/spirv-headers/src/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/dawn/third_party/spirv-tools/src/include>")
    target_link_libraries(${target} PRIVATE
        "$<BUILD_INTERFACE:SPIRV-Tools-opt>"
        "$<BUILD_INTERFACE:tint_api>"
        "$<BUILD_INTERFACE:tint_lang_core_ir_transform>")
endfunction()
