
macro(ncnn_convert_param_file PARAM_FILE)
    # Use a macro to convert the contents of a single parameter file into memory content
    file(READ ${PARAM_FILE} param_file_data)

    # remove whitespace
    string(REGEX REPLACE "\n +" "\n" param_file_data ${param_file_data})

    # remove empty line
    string(REGEX REPLACE "\n\n" "\n" param_file_data ${param_file_data})

    # Get the file name with extension
    get_filename_component(PARAM_FILE_NAME ${PARAM_FILE} NAME)
    # Manually remove ".param" since NAME_WE treats ".1.param" as a multi-extension
    string(REPLACE ".param" "" PARAM_FILE_NAME_WE "${PARAM_FILE_NAME}")
    # Replace characters invalid in C identifiers ('.' and '-') with underscores
    string(REPLACE "." "_" PARAM_FILE_NAME_WE "${PARAM_FILE_NAME_WE}")
    string(REPLACE "-" "_" PARAM_FILE_NAME_WE "${PARAM_FILE_NAME_WE}")
    # Check if the result is empty
    if (NOT PARAM_FILE_NAME_WE)
        message(FATAL_ERROR "Failed to extract valid filename from '${PARAM_FILE}'")
    endif()
    # Check if the extracted filename is a valid C identifier
    string(REGEX MATCH "^[A-Za-z_][A-Za-z0-9_]*$" is_valid "${PARAM_FILE_NAME_WE}")
    if (NOT is_valid)
        message(FATAL_ERROR "Extracted filename '${PARAM_FILE_NAME_WE}' is not a valid C identifier")
    endif()

    # text to hex
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/param_hex_data/${PARAM_FILE_NAME_WE}.text2hex.txt "${param_file_data}")
    file(READ ${CMAKE_CURRENT_BINARY_DIR}/param_hex_data/${PARAM_FILE_NAME_WE}.text2hex.txt param_file_data_hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," param_file_data_hex ${param_file_data_hex})
    string(FIND "${param_file_data_hex}" "," tail_comma REVERSE)
    string(SUBSTRING "${param_file_data_hex}" 0 ${tail_comma} param_file_data_hex)

    # generate model param header file
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/model_param_header/${PARAM_FILE_NAME_WE}.comp.hex.h "static const char ${PARAM_FILE_NAME_WE}_param_data[] = {${param_file_data_hex},0x00};\n")

    # append include line to a CMake variable for later output
    string(APPEND model_param_spv_data "#include \"model_param_header/${PARAM_FILE_NAME_WE}.comp.hex.h\"\n")
endmacro()