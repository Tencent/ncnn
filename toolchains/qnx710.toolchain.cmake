# before invoking this script, set the required environment variables:
# export QNX_HOST=/home/zz/soft/qnx710/host/linux/x86_64
# export QNX_TARGET=/home/zz/soft/qnx710/target/qnx7

# create ld to solve 'cannot find ld' issue
# cd ${QNX_HOST}/usr/bin/
# ln -s aarch64-unknown-nto-qnx7.1.0-ld ld

set(CMAKE_SYSTEM_NAME QNX)
set(CMAKE_SYSTEM_PROCESSOR aarch64le)

if(NOT DEFINED ENV{QNX_HOST})
    message(FATAL_ERROR "Please set env var `QNX_HOST` first")
endif()
set(QNX_HOST "$ENV{QNX_HOST}")

if(NOT DEFINED ENV{QNX_TARGET})
    message(FATAL_ERROR "Please set env var `QNX_TARGET` first")
endif()
set(QNX_TARGET "$ENV{QNX_TARGET}")

set(CMAKE_C_COMPILER "${QNX_HOST}/usr/bin/aarch64-unknown-nto-qnx7.1.0-gcc-8.3.0")
set(CMAKE_CXX_COMPILER "${QNX_HOST}/usr/bin/aarch64-unknown-nto-qnx7.1.0-g++-8.3.0")

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
