
find_package(Protobuf)

if(NOT PROTOBUF_FOUND)
    # fallback to config mode
    find_package(protobuf CONFIG)
    set(PROTOBUF_FOUND ${protobuf_FOUND})
    set(PROTOBUF_VERSION ${protobuf_VERSION})
endif()

if(PROTOBUF_FOUND)
    if(DEFINED PROTOBUF_VERSION AND PROTOBUF_VERSION VERSION_GREATER_EQUAL 3.22)
        set(CMAKE_CXX_STANDARD 17)
    endif()

    if(protobuf_MODULE_COMPATIBLE)
        protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx.proto)
        add_executable(onnx2ncnn onnx2ncnn.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
        target_include_directories(onnx2ncnn PRIVATE ${PROTOBUF_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
        target_link_libraries(onnx2ncnn PRIVATE ${PROTOBUF_LIBRARIES})
    else()
        add_executable(onnx2ncnn onnx2ncnn.cpp onnx.proto)
        target_include_directories(onnx2ncnn PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
        protobuf_generate(TARGET onnx2ncnn)
        target_link_libraries(onnx2ncnn PRIVATE protobuf::libprotobuf)
    endif()

    # add all onnx2ncnn tool to a virtual project group
    set_property(TARGET onnx2ncnn PROPERTY FOLDER "tools/converter")
    ncnn_install_tool(onnx2ncnn)
else()
    message(WARNING "Protobuf not found, onnx model convert tool won't be built")
endif()
