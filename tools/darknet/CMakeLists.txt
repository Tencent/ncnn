add_executable(darknet2ncnn darknet2ncnn.cpp)
target_compile_options(darknet2ncnn PRIVATE -std=c++11)
set_property(TARGET darknet2ncnn PROPERTY FOLDER "tools/converter")
