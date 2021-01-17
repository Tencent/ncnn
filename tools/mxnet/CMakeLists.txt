
add_library(libmxnet2ncnn mxnet2ncnn.cpp mxnet2ncnn.h)
target_include_directories(libmxnet2ncnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(mxnet2ncnn mxnet2ncnn_exe.cpp)
target_link_libraries(mxnet2ncnn libmxnet2ncnn)

# add all mxnet2ncnn tool to a virtual project group
set_property(TARGET mxnet2ncnn libmxnet2ncnn PROPERTY FOLDER "tools/converter")
