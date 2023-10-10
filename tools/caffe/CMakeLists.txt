
find_package(protobuf CONFIG)

if(protobuf_FOUND)
    set(PROTOBUF_FOUND ${protobuf_FOUND})
    set(PROTOBUF_VERSION ${protobuf_VERSION})
else()
    # fallback to system
    find_package(Protobuf)
    set(PROTOBUF_FOUND ${Protobuf_FOUND})
    set(PROTOBUF_VERSION ${Protobuf_VERSION})
    if(TARGET protobuf::protoc)
        set_target_properties(protobuf::protoc PROPERTIES IMPORTED_LOCATION_RELEASE "${PROTOBUF_PROTOC_EXECUTABLE}")
    endif()
endif()

if(PROTOBUF_FOUND)
    if(DEFINED PROTOBUF_VERSION AND PROTOBUF_VERSION VERSION_GREATER_EQUAL 3.22)
        set(CMAKE_CXX_STANDARD 17)
    endif()

    if(Protobuf_FOUND OR protobuf_MODULE_COMPATIBLE)
        protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HDRS caffe.proto)
        add_executable(caffe2ncnn caffe2ncnn.cpp ${CAFFE_PROTO_SRCS} ${CAFFE_PROTO_HDRS})
        target_include_directories(caffe2ncnn PRIVATE ${PROTOBUF_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
        target_link_libraries(caffe2ncnn PRIVATE ${PROTOBUF_LIBRARIES})
    else()
        add_executable(caffe2ncnn caffe2ncnn.cpp caffe.proto)
        target_include_directories(caffe2ncnn PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
        protobuf_generate(TARGET caffe2ncnn)
        target_link_libraries(caffe2ncnn PRIVATE protobuf::libprotobuf)
    endif()

    # add all caffe2ncnn tool to a virtual project group
    set_property(TARGET caffe2ncnn PROPERTY FOLDER "tools/converter")
    ncnn_install_tool(caffe2ncnn)
else()
    message(WARNING "Protobuf not found, caffe model convert tool won't be built")
endif()
