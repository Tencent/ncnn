add_library(usermod_ncnn INTERFACE)
find_package(OpenMP REQUIRED)

target_sources(usermod_ncnn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/ncnn_module.cpp
)

target_include_directories(usermod_ncnn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../../../build_micropython/install/include
)

target_link_directories(usermod_ncnn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/../../../build_micropython/install/lib
    OpenMP::OpenMP_CXX
)

target_link_libraries(usermod_ncnn INTERFACE
    ncnn
)

target_link_libraries(usermod INTERFACE usermod_ncnn)