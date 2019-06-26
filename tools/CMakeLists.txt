
add_subdirectory(caffe)
add_subdirectory(mxnet)
add_subdirectory(onnx)
# add_subdirectory(quantize)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

include_directories(${CMAKE_SOURCE_DIR}/src)

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
