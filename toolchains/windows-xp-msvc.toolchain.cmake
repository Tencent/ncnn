# Windows XP MSVC Toolchain
# Supports Visual Studio 2017 (v141) and later with XP compatibility
# Contributors: @Sugar-Baby and @AtomAlpaca

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_VERSION 5.1)
set(CMAKE_C_COMPILER cl)
set(CMAKE_CXX_COMPILER cl)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D_WIN32_WINNT=0x0501 /DWINVER=0x0501 /EHsc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_WIN32_WINNT=0x0501 /DWINVER=0x0501 /EHsc")
set(CMAKE_GENERATOR_TOOLSET "v141_xp" CACHE STRING "Platform Toolset" FORCE)

# Disabling AVX will automatically disable newer instruction sets (AVX2, AVX512, etc.)
set(NCNN_VULKAN OFF CACHE BOOL "Disable Vulkan for Windows XP" FORCE)
set(CMAKE_GENERATOR_PLATFORM "Win32" CACHE STRING "Platform" FORCE)
set(NCNN_SIMPLEOCV ON CACHE BOOL "Use simple OpenCV implementation" FORCE)

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

message(STATUS "Windows XP MSVC Toolchain configured")
message(STATUS "  Platform Toolset: ${CMAKE_GENERATOR_TOOLSET}")
message(STATUS "  Platform: ${CMAKE_GENERATOR_PLATFORM}")
message(STATUS "  Runtime Library: ${CMAKE_MSVC_RUNTIME_LIBRARY}")
message(STATUS "  Vulkan: ${NCNN_VULKAN}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")