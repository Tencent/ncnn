set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR mips32el)

if(DEFINED ENV{MIPS_ROOT_PATH})
    file(TO_CMAKE_PATH $ENV{MIPS_ROOT_PATH} MIPS_ROOT_PATH)
else()
    message(FATAL_ERROR "MIPS_ROOT_PATH env must be defined")
endif()

set(MIPS_ROOT_PATH ${MIPS_ROOT_PATH} CACHE STRING "root path to mips toolchain")

set(CMAKE_C_COMPILER "${MIPS_ROOT_PATH}/bin/mips-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${MIPS_ROOT_PATH}/bin/mips-linux-gnu-g++")

set(CMAKE_FIND_ROOT_PATH "${MIPS_ROOT_PATH}/mips-linux-gnu")

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

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=xburst2 -mtune=xburst2 -mfp64 -mnan=2008")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=xburst2 -mtune=xburst2 -mfp64 -mnan=2008")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")

# export MIPS_ROOT_PATH=/home/nihui/osd/ingenic-linux-kernel4.4.94-x2000_v12-v8.0-20220125/prebuilts/toolchains/mips-gcc720-glibc229
# export MIPS_ROOT_PATH=/home/nihui/osd/君正X2000开发板资料发布/03_SDK/sdk/prebuilts/toolchains/mips-gcc720-glibc229
# cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ingenic-x2000.toolchain.cmake ..
