project(pnnx)
cmake_minimum_required(VERSION 3.10)

#set(CMAKE_BUILD_TYPE debug)
#set(CMAKE_BUILD_TYPE relwithdebinfo)
#set(CMAKE_BUILD_TYPE release)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
