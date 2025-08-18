add_library(usermod_ncnn INTERFACE)
find_package(OpenMP REQUIRED)

find_library(NCNN_LIBRARY 
    NAMES ncnn
    PATHS 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../build_micropython/install/lib64
        ${CMAKE_CURRENT_SOURCE_DIR}/../../build_micropython/install/lib
    NO_DEFAULT_PATH
)

target_compile_definitions(usermod_ncnn INTERFACE
    NCNN_STRING=1
    NCNN_STDIO=1
    NCNN_PIXEL=1
    NCNN_PIXEL_DRAWING=1
)

target_sources(usermod_ncnn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/c_api.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ncnn_module.c
)

target_include_directories(usermod_ncnn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../../build_micropython/install/include
)

target_link_directories(usermod_ncnn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/../../build_micropython/install/lib
    OpenMP::OpenMP_CXX
)

target_link_libraries(usermod_ncnn INTERFACE
    ncnn
)

target_link_libraries(usermod INTERFACE usermod_ncnn)