# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR mips )
# make sure mipsisa32r6-linux-gnu-gcc and mipsisa32r6-linux-gnu-g++ can be found in $PATH:
SET ( CMAKE_C_COMPILER "mipsisa32r6-linux-gnu-gcc" )
SET ( CMAKE_CXX_COMPILER "mipsisa32r6-linux-gnu-g++" )

# set searching rules for cross-compiler
SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

# set ${CMAKE_C_FLAGS} and ${CMAKE_CXX_FLAGS}flag for cross-compiled process
SET ( CMAKE_CXX_FLAGS "-std=c++11 -march=mips32r6 -mmsa -fopenmp ${CMAKE_CXX_FLAGS}" )

# other settings
add_definitions(-D__MIPS_MSA)
add_definitions(-DLINUX)
SET ( LINUX true)
