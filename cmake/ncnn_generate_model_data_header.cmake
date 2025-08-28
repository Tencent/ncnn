
macro(ncnn_convert_model_file MODEL_FILE)
    # Use macro to convert single model file to mem content
    file(READ ${MODEL_FILE} model_file_data)

    # remove whitespace
    string(REGEX REPLACE "\n +" "\n" model_file_data ${model_file_data})

    # remove empty line
    string(REGEX REPLACE "\n\n" "\n" model_file_data ${model_file_data})

    get_filename_component(MODEL_FILE_NAME_WE ${MODEL_FILE} NAME_WE)

    # text to hex
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/model_hex_data/${MODEL_FILE_NAME_WE}.text2hex.txt "${model_file_data}")
    file(READ ${CMAKE_CURRENT_BINARY_DIR}/model_hex_data/${MODEL_FILE_NAME_WE}.text2hex.txt model_file_data_hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," model_file_data_hex ${model_file_data_hex})
    string(FIND "${model_file_data_hex}" "," tail_comma REVERSE)
    string(SUBSTRING "${model_file_data_hex}" 0 ${tail_comma} model_file_data_hex)

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/model_data_header/${MODEL_FILE_NAME_WE}.comp.hex.h "static const char ${MODEL_FILE_NAME_WE}_param_data[] = {${model_file_data_hex},0x00};\n")
    string(APPEND model_data_spv_data "#include \"model_data_header/${MODEL_FILE_NAME_WE}.comp.hex.h\"\n")
    string(APPEND model_data_registry "{\"${MODEL_FILE_NAME_WE}\", ${MODEL_FILE_NAME_WE}_param_data},\n")
endmacro()