
add_subdirectory(caffe)
add_subdirectory(mxnet)
add_subdirectory(onnx)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

include_directories(${CMAKE_SOURCE_DIR}/src)

if(NCNN_VULKAN)
    include_directories(${Vulkan_INCLUDE_DIR})
endif()

add_executable(ncnn2mem ncnn2mem.cpp)

target_link_libraries(ncnn2mem PRIVATE ncnn)

if(COVERAGE)
    target_link_libraries(ncnn2mem PRIVATE --coverage)
endif()

if(NCNN_VULKAN)
    target_link_libraries(ncnn2mem PRIVATE ${Vulkan_LIBRARY})
endif()
