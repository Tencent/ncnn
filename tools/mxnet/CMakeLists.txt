
add_executable(mxnet2ncnn mxnet2ncnn.cpp)

# add all mxnet2ncnn tool to a virtual project group
set_property(TARGET mxnet2ncnn PROPERTY FOLDER "tools/converter")
