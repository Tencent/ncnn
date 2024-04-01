set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(DEFINED ENV{RISCV_ROOT_PATH})
    file(TO_CMAKE_PATH $ENV{RISCV_ROOT_PATH} RISCV_ROOT_PATH)
else()
    message(FATAL_ERROR "RISCV_ROOT_PATH env must be defined")
endif()

set(RISCV_ROOT_PATH ${RISCV_ROOT_PATH} CACHE STRING "root path to riscv gnu toolchain")

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_SYSROOT "${RISCV_ROOT_PATH}/sysroot")

set(CMAKE_C_COMPILER_TARGET "riscv64-unknown-linux-gnu")
set(CMAKE_CXX_COMPILER_TARGET "riscv64-unknown-linux-gnu")

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

# add --ld-path=${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-ld or append $RISCV_ROOT_PATH/bin to PATH.
set(CMAKE_C_FLAGS "--gcc-toolchain=${RISCV_ROOT_PATH} -march=rv64gc")
set(CMAKE_CXX_FLAGS "--gcc-toolchain=${RISCV_ROOT_PATH} -march=rv64gc")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
