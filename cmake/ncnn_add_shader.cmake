
macro(ncnn_add_shader NCNN_SHADER_SRC)
    get_filename_component(NCNN_SHADER_SRC_NAME_WE ${NCNN_SHADER_SRC} NAME_WE)
    set(NCNN_SHADER_COMP_HEADER ${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/${NCNN_SHADER_SRC_NAME_WE}.comp.hex.h)

    add_custom_command(
        OUTPUT ${NCNN_SHADER_COMP_HEADER}
        COMMAND ${CMAKE_COMMAND} -DSHADER_SRC=${NCNN_SHADER_SRC} -DSHADER_COMP_HEADER=${NCNN_SHADER_COMP_HEADER} -DNCNN_COVERAGE=${NCNN_COVERAGE} -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_generate_shader_comp_header.cmake"
        DEPENDS ${NCNN_SHADER_SRC} "${CMAKE_CURRENT_BINARY_DIR}/platform.h" "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_generate_shader_comp_header.cmake"
        COMMENT "Preprocessing shader source ${NCNN_SHADER_SRC_NAME_WE}.comp"
        VERBATIM
    )
    set_source_files_properties(${NCNN_SHADER_COMP_HEADER} PROPERTIES GENERATED TRUE)

    get_filename_component(NCNN_SHADER_COMP_HEADER_NAME ${NCNN_SHADER_COMP_HEADER} NAME)
    string(APPEND layer_shader_spv_data "#include \"layer/vulkan/shader/${NCNN_SHADER_COMP_HEADER_NAME}\"\n")

    get_filename_component(NCNN_SHADER_SRC_NAME_WE ${NCNN_SHADER_SRC} NAME_WE)
    string(APPEND layer_shader_registry "{${NCNN_SHADER_SRC_NAME_WE}_comp_data,sizeof(${NCNN_SHADER_SRC_NAME_WE}_comp_data)},\n")

    if(NCNN_COVERAGE)
        # prefix sums use the original files and are shared by all shader variants
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${NCNN_SHADER_SRC}")
        file(READ "${NCNN_SHADER_SRC}" coverage_source)
        string(REGEX REPLACE "[^\n]" "" coverage_newlines "${coverage_source}")
        string(LENGTH "${coverage_newlines}" coverage_lines)
        math(EXPR coverage_lines "${coverage_lines} + 1")
        if(NOT DEFINED shader_coverage_line_count)
            set(shader_coverage_line_count 0)
        endif()
        file(RELATIVE_PATH coverage_path "${PROJECT_SOURCE_DIR}" "${NCNN_SHADER_SRC}")
        file(SHA256 "${NCNN_SHADER_SRC}" coverage_hash)
        string(APPEND shader_coverage_identity "${coverage_path}:${coverage_hash}:${shader_coverage_line_count}:${coverage_lines}\n")
        string(APPEND shader_coverage_entries "{\"${coverage_path}\", ${shader_coverage_line_count}, ${coverage_lines}},\n")
        math(EXPR shader_coverage_line_count "${shader_coverage_line_count} + ${coverage_lines}")
    endif()

    list(APPEND NCNN_SHADER_SPV_HEX_FILES ${NCNN_SHADER_COMP_HEADER})

    # generate layer_shader_type_enum file
    set(layer_shader_type_enum "${layer_shader_type_enum}${NCNN_SHADER_SRC_NAME_WE} = ${__LAYER_SHADER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_SHADER_TYPE_ENUM_INDEX "${__LAYER_SHADER_TYPE_ENUM_INDEX}+1")
endmacro()
