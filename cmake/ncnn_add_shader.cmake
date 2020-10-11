
macro(ncnn_add_shader NCNN_SHADER_SRC)
    get_filename_component(NCNN_SHADER_SRC_NAME_WE ${NCNN_SHADER_SRC} NAME_WE)
    set(NCNN_SHADER_COMP_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${NCNN_SHADER_SRC_NAME_WE}.comp.hex.h)

    add_custom_command(
        OUTPUT ${NCNN_SHADER_COMP_HEADER}
        COMMAND ${CMAKE_COMMAND} -DSHADER_SRC=${NCNN_SHADER_SRC} -DSHADER_COMP_HEADER=${NCNN_SHADER_COMP_HEADER} -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_generate_shader_comp_header.cmake"
        DEPENDS ${NCNN_SHADER_SRC}
        COMMENT "Preprocessing shader source ${NCNN_SHADER_SRC_NAME_WE}.comp"
        VERBATIM
    )
    set_source_files_properties(${NCNN_SHADER_COMP_HEADER} PROPERTIES GENERATED TRUE)

    get_filename_component(NCNN_SHADER_COMP_HEADER_NAME ${NCNN_SHADER_COMP_HEADER} NAME)
    string(APPEND layer_shader_spv_data "#include \"${NCNN_SHADER_COMP_HEADER_NAME}\"\n")

    get_filename_component(NCNN_SHADER_SRC_NAME_WE ${NCNN_SHADER_SRC} NAME_WE)
    string(APPEND layer_shader_registry "{${NCNN_SHADER_SRC_NAME_WE}_comp_data,sizeof(${NCNN_SHADER_SRC_NAME_WE}_comp_data)},\n")

    list(APPEND NCNN_SHADER_SPV_HEX_FILES ${NCNN_SHADER_COMP_HEADER})

    # generate layer_shader_type_enum file
    set(layer_shader_type_enum "${layer_shader_type_enum}${NCNN_SHADER_SRC_NAME_WE} = ${__LAYER_SHADER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_SHADER_TYPE_ENUM_INDEX "${__LAYER_SHADER_TYPE_ENUM_INDEX}+1")
endmacro()

