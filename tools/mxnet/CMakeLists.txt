
add_executable(mxnet2ncnn mxnet2ncnn.cpp)

if (EMSCRIPTEN AND NCNN_EMSCRIPTEN_MODULARIZE)
    set_target_properties(mxnet2ncnn PROPERTIES LINK_FLAGS "-s 'EXPORT_NAME=\"create_mxnet2ncnn\"'")
endif()

# add all mxnet2ncnn tool to a virtual project group
set_property(TARGET mxnet2ncnn PROPERTY FOLDER "tools/converter")
ncnn_install_tool(mxnet2ncnn)
