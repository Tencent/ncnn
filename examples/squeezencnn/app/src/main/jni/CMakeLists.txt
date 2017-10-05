cmake_minimum_required(VERSION 3.6)
project(SQUEEZENCNN)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -std=c++11 -fopenmp")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -fopenmp")

set(NCNN_INSTALL_PATH
     ${CMAKE_SOURCE_DIR}/../../../../../../build-android-armv7/install)

add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn
        PROPERTIES IMPORTED_LOCATION ${NCNN_INSTALL_PATH}/lib/libncnn.a)


include_directories(
    ${NCNN_INSTALL_PATH}/include
)

set(SQUEEZENCNN_FILES squeezencnn_jni.cpp)

add_library(squeezencnn SHARED ${SQUEEZENCNN_FILES})
target_link_libraries(squeezencnn
                      ncnn
                      log
                      z
                      jnigraphics)