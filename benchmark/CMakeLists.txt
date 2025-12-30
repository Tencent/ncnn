
if(MSVC)
    # warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    add_definitions(/wd4996)
endif()

add_executable(benchncnn benchncnn.cpp)
target_link_libraries(benchncnn PRIVATE ncnn)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_libraries(benchncnn PRIVATE nodefs.js)
endif()

# add benchncnn to a virtual project group
set_property(TARGET benchncnn PROPERTY FOLDER "benchmark")

# generate model data
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_generate_model_data_header.cmake)

set(embedded_model_param_includes "")

set(PARAM_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/squeezenet.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/squeezenet_int8.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/mobilenet.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/mobilenet_int8.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/mobilenet_v2.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/mobilenet_v3.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/shufflenet.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/shufflenet_v2.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/mnasnet.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/proxylessnasnet.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/efficientnet_b0.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/efficientnetv2_b0.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/regnety_400m.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/blazeface.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/googlenet.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/googlenet_int8.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/resnet18.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/resnet18_int8.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/alexnet.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/vgg16.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/vgg16_int8.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/resnet50.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_int8.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/squeezenet_ssd.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/squeezenet_ssd_int8.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/mobilenet_ssd.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/mobilenet_ssd_int8.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/mobilenet_yolo.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/mobilenetv2_yolov3.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/yolov4-tiny.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/nanodet_m.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/yolo-fastest-1.1.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/yolo-fastestv2.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/vision_transformer.param"
    "${CMAKE_CURRENT_SOURCE_DIR}/FastestDet.param"
)

foreach(PARAM_FILE ${PARAM_FILES})
    ncnn_convert_param_file(${PARAM_FILE})
endforeach()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/embedded_model_params.h "${embedded_model_param_includes}")