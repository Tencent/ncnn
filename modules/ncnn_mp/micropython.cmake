add_library(ncnn_mpy INTERFACE)

# Add the binding file to the library.
target_sources(ncnn_mpy INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/ncnn_mp.c
)

# Adding following link option to avoid multiple definition of `fesetround`.
if(DEFINED IDF_TARGET)
    message(STATUS "ESP-IDF target detected: ${IDF_TARGET}.")
    target_link_options(ncnn_mpy INTERFACE
        -Wl,--allow-multiple-definition
    )
endif()

find_package(ncnn CONFIG QUIET)

if(ncnn_FOUND)
    message(STATUS "Found ncnn via find_package.")
    target_include_directories(ncnn_mpy INTERFACE
        ${ncnn_INCLUDE_DIRS}
    )
    target_link_libraries(ncnn_mpy INTERFACE
        ${ncnn_LIBRARIES}
    )
else()
    message(STATUS "ncnn not found via find_package.")
    if(NOT DEFINED NCNN_INSTALL_PREFIX)
        message(STATUS "NCNN_INSTALL_PREFIX not specified, using ncnn/build/install")
        set(NCNN_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../ncnn/build/install" CACHE PATH "Path to the ncnn installation directory.")
    endif()

    get_filename_component(NCNN_ABSOLUTE_INSTALL_DIR ${NCNN_INSTALL_PREFIX} ABSOLUTE)
    message(STATUS "ncnn path is: ${NCNN_ABSOLUTE_INSTALL_DIR}")

    target_include_directories(ncnn_mpy INTERFACE
        "${NCNN_ABSOLUTE_INSTALL_DIR}/include"
    )
    # Link the pre-built ncnn static library.
    target_link_libraries(ncnn_mpy INTERFACE
        "${NCNN_ABSOLUTE_INSTALL_DIR}/lib/libncnn.a"
    )
endif()

# Link our user module to the main 'usermod' target.
target_link_libraries(usermod INTERFACE ncnn_mpy)
