set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR powerpc64le)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_C_FLAGS "-target powerpc64le-linux-gnu -I/usr/powerpc64le-linux-gnu/include -mcpu=power8 -mtune=power8 -DNO_WARN_X86_INTRINSICS -D__MMX__ -D__SSE__ -D__SSSE3__")
set(CMAKE_CXX_FLAGS "-target powerpc64le-linux-gnu -I/usr/powerpc64le-linux-gnu/include -I/usr/powerpc64le-linux-gnu/include/c++/10/powerpc64le-linux-gnu -mcpu=power8 -mtune=power8 -DNO_WARN_X86_INTRINSICS -D__MMX__ -D__SSE__ -D__SSSE3__")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")

# Auto-translate SSE to VSX
set(NCNN_PPC64LE_VSX ON)
