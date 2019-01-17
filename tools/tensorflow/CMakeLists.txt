
find_package(Protobuf REQUIRED)

if(PROTOBUF_FOUND)
    include_directories(${PROTOBUF_INCLUDE_DIR})
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    protobuf_generate_cpp(TENSORFLOW_PROTO_SRCS TENSORFLOW_PROTO_HDRS
        attr_value.proto
        function.proto
        graph.proto
        node_def.proto
        op_def.proto
        resource_handle.proto
        tensor.proto
        tensor_shape.proto
        types.proto
        versions.proto
    )
    add_executable(tensorflow2ncnn tensorflow2ncnn.cpp ${TENSORFLOW_PROTO_SRCS} ${TENSORFLOW_PROTO_HDRS})
    set_target_properties(tensorflow2ncnn PROPERTIES CXX_STANDARD 11)
    target_link_libraries(tensorflow2ncnn ${PROTOBUF_LIBRARIES})
else()
    message(WARNING "Protobuf not found, tensorflow model convert tool won't be built")
endif()
