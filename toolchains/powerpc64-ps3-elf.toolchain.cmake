set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR powerpc64)

set(PS3DEV "/usr/local/ps3dev" CACHE PATH "ps3dev sdk root")

set(CMAKE_C_COMPILER "${PS3DEV}/ppu/bin/powerpc64-ps3-elf-gcc")
set(CMAKE_CXX_COMPILER "${PS3DEV}/ppu/bin/powerpc64-ps3-elf-g++")

# the bare-metal ps3 elf toolchain links against newlib (no glibc)
# cmake feature probes must not try to link a hosted executable
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_FIND_ROOT_PATH "${PS3DEV}/ppu/powerpc64-ps3-elf" "${PS3DEV}/ppu")

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

# PS3 Cell PPE: powerpc64 big endian with VMX (Altivec), no VSX
set(CMAKE_C_FLAGS "-mcpu=cell -maltivec -mabi=altivec")
set(CMAKE_CXX_FLAGS "-mcpu=cell -maltivec -mabi=altivec")

# NOTE the produced ELF is not directly runnable by rpcs3/ps3
# the ps3 elf toolchain links newlib with empty import stub metadata (.lib.stub)
# the import table must be generated after linking by running the sdk sprxlinker on the ELF
# (PSL1GHT's own Makefile does this in its %.self rule)
#
#   sprxlinker app.elf
#
# the ps3dev host tools (sprxlinker, bin2s) are built for x86-64-v3 hosts
#
# SPE (SPU) kernels are built separately with spu-gcc (-lsputhread), converted to an
# object with bin2s, and loaded on the PPE with sysSpuImageImport
#   spu-gcc -O2 -lsputhread kernel.c -o kernel.elf
#   bin2s -a 64 kernel.elf | ppu-as -o kernel_bin.o
#
# newlib provides no pthread.h, so configure with -DNCNN_THREADS=OFF -DNCNN_OPENMP=OFF
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
