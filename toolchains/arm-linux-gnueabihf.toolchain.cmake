# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR arm )
# when hislicon SDK was installed, toolchain was installed in the path as below: 
SET ( CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc" )
SET ( CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++" )

# set searching rules for cross-compiler
SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

# set ${CMAKE_C_FLAGS} and ${CMAKE_CXX_FLAGS}flag for cross-compiled process
SET ( CMAKE_CXX_FLAGS "-std=c++11 -march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 -fopenmp ${CMAKE_CXX_FLAGS}" )

# other settings
add_definitions(-D__ARM_NEON)
add_definitions(-D__ANDROID__)
SET ( ANDROID true)
