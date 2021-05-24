# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR x86 )
# if gcc/g++ was installed:
SET ( CMAKE_C_COMPILER "gcc" )
SET ( CMAKE_CXX_COMPILER "g++" )

# set searching rules
SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

# set ${CMAKE_C_FLAGS} and ${CMAKE_CXX_FLAGS}flag
SET ( CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}" )

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nodefaultlibs -fno-builtin -nostdinc++ -lc" )

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
