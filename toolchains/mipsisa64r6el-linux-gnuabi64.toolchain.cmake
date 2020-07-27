set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR mipsisa64r6el)

set(CMAKE_C_COMPILER "mipsisa64r6el-linux-gnuabi64-gcc")
set(CMAKE_CXX_COMPILER "mipsisa64r6el-linux-gnuabi64-g++")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_C_FLAGS "-march=mips64r6 -mmsa -mhard-float -mfp64 -mnan=2008")
set(CMAKE_CXX_FLAGS "-march=mips64r6 -mmsa -mhard-float -mfp64 -mnan=2008")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
