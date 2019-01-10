
find_program(GLSLANGVALIDATOR_EXECUTABLE NAMES glslangValidator)
message(STATUS "Found glslangValidator: ${GLSLANGVALIDATOR_EXECUTABLE}")

find_program(SPIRV-OPT_EXECUTABLE NAMES spirv-opt)
message(STATUS "Found spirv-opt: ${SPIRV-OPT_EXECUTABLE}")

find_program(SPIRV-LINK_EXECUTABLE NAMES spirv-link)
message(STATUS "Found spirv-link: ${SPIRV-LINK_EXECUTABLE}")

# glslangValidator -> spirv-opt -> spirv-link -> binary2hex
function(compile_compute_shader class shader_module_hex_file)
    string(TOLOWER ${class} name)

    file(GLOB_RECURSE SHADER_SRCS "layer/shader/${name}.comp")
    file(GLOB_RECURSE SHADER_SUBSRCS "layer/shader/${name}_*.comp")
    list(APPEND SHADER_SRCS ${SHADER_SUBSRCS})

    if(NOT SHADER_SRCS)
        return()
    endif()

    set(SHADER_OPT_SPV_FILES)

    foreach(SHADER_SRC ${SHADER_SRCS})

        get_filename_component(SHADER_SRC_NAME ${SHADER_SRC} NAME)
        get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)

        set(SHADER_SPV_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME}.spv)
        add_custom_command(
            OUTPUT ${SHADER_SPV_FILE}
            COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
            ARGS -V -s -e ${SHADER_SRC_NAME_WE} --source-entrypoint main -o ${SHADER_SPV_FILE} ${SHADER_SRC}
            DEPENDS ${SHADER_SRC}
            COMMENT "Compiling ${SHADER_SRC_NAME}"
            VERBATIM
        )
        set_source_files_properties(${SHADER_SPV_FILE} PROPERTIES GENERATED TRUE)

        set(SHADER_OPT_SPV_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME}.opt.spv)
        add_custom_command(
            OUTPUT ${SHADER_OPT_SPV_FILE}
            COMMAND ${SPIRV-OPT_EXECUTABLE}
            ARGS -O -o ${SHADER_OPT_SPV_FILE} ${SHADER_SPV_FILE}
            DEPENDS ${SHADER_SPV_FILE}
            COMMENT "Optimizing ${SHADER_SRC_NAME}.spv"
            VERBATIM
        )
        set_source_files_properties(${SHADER_OPT_SPV_FILE} PROPERTIES GENERATED TRUE)

        list(APPEND SHADER_OPT_SPV_FILES ${SHADER_OPT_SPV_FILE})

    endforeach()

    set(SHADER_MODULE_FILE ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv)
    add_custom_command(
        OUTPUT ${SHADER_MODULE_FILE}
        COMMAND ${SPIRV-LINK_EXECUTABLE}
        ARGS -o ${SHADER_MODULE_FILE} ${SHADER_OPT_SPV_FILES}
        DEPENDS ${SHADER_OPT_SPV_FILES}
        COMMENT "Linking ${name}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_MODULE_FILE} PROPERTIES GENERATED TRUE)

    set(SHADER_MODULE_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_MODULE_HEX_FILE}
        COMMAND ${CMAKE_COMMAND}
        ARGS -DBINARY_FILE=${SHADER_MODULE_FILE} -P ${CMAKE_SOURCE_DIR}/binary2hex.cmake
        DEPENDS ${SHADER_MODULE_FILE}
        COMMENT "Converting ${name}.spv to hex code"
        VERBATIM
    )
    set_source_files_properties(${SHADER_MODULE_HEX_FILE} PROPERTIES GENERATED TRUE)

    set(${shader_module_hex_file} ${SHADER_MODULE_HEX_FILE} PARENT_SCOPE)

endfunction()
