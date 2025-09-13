# Create an INTERFACE library for our C module.
add_library(usermod_ncnn INTERFACE)

# Add our source files to the library.
target_sources(usermod_ncnn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/ncnn_module.c
)

# Add ncnn include path and define C API flag
target_include_directories(usermod_ncnn INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/../../src
    ${CMAKE_CURRENT_LIST_DIR}/../../build/src
)

target_compile_definitions(usermod_ncnn INTERFACE
    NCNN_C_API=1
    NCNN_STDIO=1
    NCNN_STRING=1
    NCNN_PIXEL=1
)

# Find and link ncnn library
find_library(NCNN_LIB 
    NAMES ncnn
    PATHS ${CMAKE_CURRENT_LIST_DIR}/../../build
    NO_DEFAULT_PATH
)

if(NCNN_LIB)
    target_link_libraries(usermod_ncnn INTERFACE ${NCNN_LIB})
else()
    message(WARNING "ncnn library not found, please build ncnn first")
endif()

# Enable C++11 support
set_property(TARGET usermod_ncnn PROPERTY CXX_STANDARD 11)

# Link the module with the main binary.
target_link_libraries(usermod INTERFACE usermod_ncnn)