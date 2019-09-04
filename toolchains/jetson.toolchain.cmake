# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR aarch64 )
# for the reason of aarch64-linux-gnu-gcc DONOT need to be installed, make sure aarch64-linux-gnu-gcc and aarch64-linux-gnu-g++ can be found in $PATH: 
SET ( CMAKE_C_COMPILER "aarch64-linux-gnu-gcc" )
SET ( CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++" )

# set ${CMAKE_C_FLAGS} and ${CMAKE_CXX_FLAGS}flag for cross-compiled process
# -march=armv8-a could work on Jetson, but will compile without some extra cpu features
SET ( CMAKE_CXX_FLAGS "-std=c++11 -march=native -fopenmp ${CMAKE_CXX_FLAGS}" )

# other settings
# Jetson CPU supports asimd
add_definitions ( -D__ARM_NEON)
# Jetson does NOT run ANDROID
# but `__ANDROID__` marco is tested before `__aarch64__`
# and currently no negative effect is caused by this marco
add_definitions( -D__ANDROID__)
SET ( ANDROID true)
