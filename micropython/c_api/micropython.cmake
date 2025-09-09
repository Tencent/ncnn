add_library(usermod_ncnn INTERFACE)

set(NCNN_INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}/../../build_micropython/install)

find_library(NCNN_LIBRARY 
    NAMES ncnn
    PATHS 
        ${NCNN_INSTALL_DIR}/lib64
        ${NCNN_INSTALL_DIR}/lib
    NO_DEFAULT_PATH
)

if(NOT NCNN_LIBRARY)
    message(FATAL_ERROR "NCNN library not found in ${NCNN_INSTALL_DIR}")
endif()

if(NOT EXISTS ${NCNN_INSTALL_DIR}/include)
    message(FATAL_ERROR "NCNN headers not found at ${NCNN_INSTALL_DIR}/include")
endif()

target_sources(usermod_ncnn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/src/core/ncnn_module.c
    ${CMAKE_CURRENT_LIST_DIR}/src/api/version.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/allocator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/option.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/mat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/version.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/allocator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/option.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/mat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/mat_pixel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/mat_pixel_drawing.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/mat_process.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/extractor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/layer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/net.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/datareader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/paramdict.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/modelbin.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/api/blob.cpp
)

if(DEFINED IDF_TARGET)
    target_link_options(usermod_ncnn INTERFACE
        -Wl,--allow-multiple-definition
    )
endif()

target_include_directories(usermod_ncnn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${NCNN_INSTALL_DIR}/include
)

target_link_libraries(usermod_ncnn INTERFACE
    ${NCNN_LIBRARY}
)

target_link_libraries(usermod INTERFACE usermod_ncnn)