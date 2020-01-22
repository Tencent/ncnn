
add_subdirectory(caffe)
add_subdirectory(mxnet)
add_subdirectory(onnx)

find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
endif()

if(OpenCV_FOUND)
    add_subdirectory(quantize)
else()
    message(WARNING "OpenCV not found, quantize tools won't be built")
endif()

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
