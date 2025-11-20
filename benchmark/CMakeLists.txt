
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

file(GLOB PARAM_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.param")

foreach(PARAM_FILE ${PARAM_FILES})
    ncnn_convert_param_file(${PARAM_FILE})
endforeach()

configure_file(model_param_registry.h.in ${CMAKE_CURRENT_BINARY_DIR}/model_param_registry.h)
configure_file(model_param_spv_data.h.in ${CMAKE_CURRENT_BINARY_DIR}/model_param_spv_data.h)