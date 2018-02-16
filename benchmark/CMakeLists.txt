
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

add_executable(benchncnn benchncnn.cpp)
set_property(TARGET benchncnn PROPERTY COMPILE_FLAGS "-fpie")
set_property(TARGET benchncnn PROPERTY LINK_FLAGS "-pie")
target_link_libraries(benchncnn ncnn)
