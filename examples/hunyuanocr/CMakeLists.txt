cmake_minimum_required(VERSION 3.15)
project(hunyuanocr)

# In-tree builds already have the ncnn target; standalone builds need find_package.
if(NOT TARGET ncnn)
    find_package(ncnn REQUIRED)
    set(_ncnn_target ncnn::ncnn)
else()
    set(_ncnn_target ncnn)
endif()

# OpenCV or ncnn SimpleOCV fallback
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)

add_executable(hunyuanocr hunyuanocr.cpp)
target_link_libraries(hunyuanocr PRIVATE ${_ncnn_target})

if(OpenCV_FOUND)
    target_include_directories(hunyuanocr PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(hunyuanocr PRIVATE ${OpenCV_LIBS})
else()
    target_compile_definitions(hunyuanocr PRIVATE USE_NCNN_SIMPLEOCV)
endif()

set_property(TARGET hunyuanocr PROPERTY FOLDER "examples")
