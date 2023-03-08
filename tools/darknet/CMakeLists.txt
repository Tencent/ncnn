add_executable(darknet2ncnn darknet2ncnn.cpp)
set_property(TARGET darknet2ncnn PROPERTY FOLDER "tools/converter")
ncnn_install_tool(darknet2ncnn)
