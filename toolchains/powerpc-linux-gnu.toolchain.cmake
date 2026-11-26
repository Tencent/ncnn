set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR powerpc)

set(CMAKE_C_COMPILER "powerpc-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "powerpc-linux-gnu-g++")

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

# 32-bit big endian, locked to the G4 (PowerPC 7400) VMX subset. G4 is the
# baseline that has altivec; the generic -mcpu=powerpc target has no vector
# unit at all, so the compiler emits the altivec round and truncate opcodes
# for it and they fault at runtime. VSX is power7 and newer, so it stays out
# of this toolchain.
set(CMAKE_C_FLAGS "-mcpu=G4 -maltivec -mabi=altivec")
set(CMAKE_CXX_FLAGS "-mcpu=G4 -maltivec -mabi=altivec")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
