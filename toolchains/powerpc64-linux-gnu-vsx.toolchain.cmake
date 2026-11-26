set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR powerpc64)

set(CMAKE_C_COMPILER "powerpc64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "powerpc64-linux-gnu-g++")

if(NOT CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
endif()
if(NOT CMAKE_FIND_ROOT_PATH_MODE_LIBRARY)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
endif()
if(NOT CMAKE_FIND_ROOT_PATH_MODE_INCLUDE)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
endif()
if(NOT CMAKE_FIND_ROOT_PATH_MODE_PACKAGE)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
endif()

# powerpc64 big endian, power8 with vsx
# vsx is backward compatible with vmx and adds vector div/sqrt, natural alignment
# load/store and the upper 32 vector registers (vsr32..63)
# vsx requires power7 or newer, it is not available on the ps3 cell ppe
set(CMAKE_C_FLAGS "-mcpu=power8 -mtune=power8 -mvsx")
set(CMAKE_CXX_FLAGS "-mcpu=power8 -mtune=power8 -mvsx")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
