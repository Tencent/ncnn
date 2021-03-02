# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR aarch64 )

# 注意这里的编译器改为clang
SET ( CMAKE_C_COMPILER "clang" )
SET ( CMAKE_CXX_COMPILER "clang++" )

# set searching rules for cross-compiler
SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

# set ${CMAKE_C_FLAGS} and ${CMAKE_CXX_FLAGS}flag for cross-compiled process
SET ( CMAKE_CXX_FLAGS "-std=c++11 -march=armv8-a -fopenmp ${CMAKE_CXX_FLAGS}" )

# other settings
add_definitions(-D__ARM_NEON)
add_definitions(-D__ANDROID__)
SET ( ANDROID true)
#————————————————
#版权声明：本文为CSDN博主「yuanlulu」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
#原文链接：https://blog.csdn.net/yuanlulu/article/details/87952370