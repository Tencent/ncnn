add_library(ncnn_mpy INTERFACE)

set(NCNN_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/../ncnn/src)

file(GLOB NCNN_CORE_SRC
    "${NCNN_SRC_DIR}/*.cpp"
)
# some hw may not support vulkan
list(FILTER NCNN_CORE_SRC EXCLUDE REGEX ".*vulkan.*")

file(GLOB_RECURSE NCNN_ALL_LAYER_SRC
    "${NCNN_SRC_DIR}/layer/*.cpp"
)
# exclude non-embedded sources
list(FILTER NCNN_ALL_LAYER_SRC EXCLUDE REGEX "/vulkan/")
list(FILTER NCNN_ALL_LAYER_SRC EXCLUDE REGEX "/x86/")
# list(FILTER NCNN_ALL_LAYER_SRC EXCLUDE REGEX "/mips/")

target_sources(ncnn_mpy INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/ncnn_mp.c
    ${NCNN_CORE_SRC}
    ${NCNN_ALL_LAYER_SRC}
)

target_include_directories(ncnn_mpy INTERFACE
    ${NCNN_SRC_DIR}
)

target_compile_definitions(ncnn_mpy INTERFACE
    # Disable features not needed in MicroPython
    NCNN_VULKAN=0
    NCNN_BUILD_TOOLS=0
    NCNN_BUILD_EXAMPLES=0
    NCNN_BUILD_BENCHMARK=0
    NCNN_BUILD_TESTS=0
    NCNN_PYTHON=0
    NCNN_STDIO=1
    NCNN_STRING=1
    NCNN_PIXEL=1
    NCNN_SIMPLEOMP=1
    NCNN_SIMPLESTL=1
    NCNN_DISABLE_RTTI=1
    NCNN_DISABLE_EXCEPTION=1
)

# Link the main 'usermod' target.
target_link_libraries(usermod INTERFACE ncnn_mpy)
