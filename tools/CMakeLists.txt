
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

find_package(Protobuf REQUIRED)

include_directories(${PROTOBUF_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HDRS caffe.proto)

add_executable(caffe2ncnn caffe2ncnn.cpp ${CAFFE_PROTO_SRCS} ${CAFFE_PROTO_HDRS})
add_executable(caffe2ncnn_mtcnn caffe2ncnn_mtcnn.cpp ${CAFFE_PROTO_SRCS} ${CAFFE_PROTO_HDRS})

target_link_libraries(caffe2ncnn ${PROTOBUF_LIBRARIES})
target_link_libraries(caffe2ncnn_mtcnn ${PROTOBUF_LIBRARIES})

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(ncnn2mem ncnn2mem.cpp)

target_link_libraries(ncnn2mem ncnn)
