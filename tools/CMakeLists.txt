
add_subdirectory(caffe)
add_subdirectory(mxnet)
add_subdirectory(onnx)
# add_subdirectory(quantize)

add_executable(ncnn2mem ncnn2mem.cpp)

target_link_libraries(ncnn2mem PRIVATE ncnn)

if(NCNN_VULKAN)
    target_link_libraries(ncnn2mem PRIVATE ${Vulkan_LIBRARY})
endif()

add_executable(ncnnoptimize ncnnoptimize.cpp)

target_link_libraries(ncnnoptimize PRIVATE ncnn)

if(NCNN_VULKAN)
    target_link_libraries(ncnnoptimize PRIVATE ${Vulkan_LIBRARY})
endif()
