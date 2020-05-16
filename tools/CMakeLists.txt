cmake_minimum_required(VERSION 3.1) # for CMAKE_CXX_STANDARD
set(CMAKE_CXX_STANDARD 11)

if(MSVC)
    # warning C4018: '<': signed/unsigned mismatch
    # warning C4244: 'argument': conversion from 'uint64_t' to 'const unsigned int', possible loss of data
    # warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    add_definitions(/wd4018 /wd4244 /wd4996)
endif()

add_subdirectory(caffe)
add_subdirectory(mxnet)
add_subdirectory(onnx)
add_subdirectory(darknet)
add_subdirectory(quantize)

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

# add all tools to a virtual project group
set_property(TARGET ncnn2mem PROPERTY FOLDER "tools")
set_property(TARGET ncnnoptimize PROPERTY FOLDER "tools")
