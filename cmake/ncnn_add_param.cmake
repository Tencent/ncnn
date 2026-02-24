
macro(ncnn_add_param NCNN_PARAM_SRC)
    # Get the file name with extension
    get_filename_component(NCNN_PARAM_SRC_NAME_WE ${NCNN_PARAM_SRC} NAME)
    # Manually remove ".param" since NAME_WE treats ".1.param" as a multi-extension
    string(REPLACE ".param" "" NCNN_PARAM_SRC_NAME_WE "${NCNN_PARAM_SRC_NAME_WE}")
    # Replace characters invalid in C identifiers ('.' and '-') with underscores
    string(REPLACE ".param" "" NCNN_PARAM_SRC_NAME_WE "${NCNN_PARAM_SRC_NAME_WE}")
    # Replace characters invalid in C identifiers ('.' and '-') with underscores
    string(REPLACE "." "_" NCNN_PARAM_SRC_NAME_WE "${NCNN_PARAM_SRC_NAME_WE}")
    string(REPLACE "-" "_" NCNN_PARAM_SRC_NAME_WE "${NCNN_PARAM_SRC_NAME_WE}")
    # Check if the result is empty
    if (NOT NCNN_PARAM_SRC_NAME_WE)
        message(FATAL_ERROR "Failed to extract valid filename from '${NCNN_PARAM_SRC}'")
    endif()
    # Check if the extracted filename is a valid C identifier
    string(REGEX MATCH "^[A-Za-z_][A-Za-z0-9_]*$" is_valid "${NCNN_PARAM_SRC_NAME_WE}")
    if (NOT is_valid)
        message(FATAL_ERROR "Extracted filename '${NCNN_PARAM_SRC_NAME_WE}' is not a valid C identifier")
    endif()

    set(NCNN_PARAM_HEADER ${CMAKE_CURRENT_BINARY_DIR}/param/${NCNN_PARAM_SRC_NAME_WE}.hex.h)

    add_custom_command(
        OUTPUT ${NCNN_PARAM_HEADER}
        COMMAND ${CMAKE_COMMAND} -DPARAM_SRC=${NCNN_PARAM_SRC} -DPARAM_SRC_NAME_WE=${NCNN_PARAM_SRC_NAME_WE} -DPARAM_HEADER=${NCNN_PARAM_HEADER} -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_generate_param_header.cmake"
        DEPENDS ${NCNN_PARAM_SRC}
        COMMENT "Preprocessing param source ${NCNN_PARAM_SRC_NAME_WE}.param"
        VERBATIM
    )
    set_source_files_properties(${NCNN_PARAM_HEADER} PROPERTIES GENERATED TRUE)

    get_filename_component(NCNN_PARAM_HEADER_NAME ${NCNN_PARAM_HEADER} NAME)
    string(APPEND param_header_data "#include \"param/${NCNN_PARAM_HEADER_NAME}\"\n")

    list(APPEND NCNN_PARAM_HEX_FILES ${NCNN_PARAM_HEADER})
endmacro()
