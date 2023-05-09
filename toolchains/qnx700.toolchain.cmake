# before invoking this script, set the required environment variables:
# export QNX_HOST=/home/zz/soft/qnx700_sdp/host/linux/x86_64
# export QNX_TARGET=/home/zz/soft/qnx700_sdp/target/qnx7

set(CMAKE_SYSTEM_NAME QNX)
set(CMAKE_SYSTEM_PROCESSOR arm)

if(NOT DEFINED ENV{QNX_HOST})
  message(FATAL_ERROR "Please set env var `QNX_HOST` first")
endif()
set(QNX_HOST "$ENV{QNX_HOST}")

if(NOT DEFINED ENV{QNX_TARGET})
  message(FATAL_ERROR "Please set env var `QNX_TARGET` first")
endif()
set(QNX_TARGET "$ENV{QNX_TARGET}")

set(CMAKE_C_COMPILER "${QNX_HOST}/usr/bin/aarch64-unknown-nto-qnx7.0.0-gcc-5.4.0")
set(CMAKE_CXX_COMPILER "${QNX_HOST}/usr/bin/aarch64-unknown-nto-qnx7.0.0-g++-5.4.0")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
