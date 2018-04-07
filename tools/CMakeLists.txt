
add_subdirectory(caffe)
add_subdirectory(mxnet)
add_subdirectory(onnx)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(ncnn2mem ncnn2mem.cpp)

target_link_libraries(ncnn2mem ncnn)
