
# must define PARAM_HEADER PARAM_SRC PARAM_SRC_NAME_WE

file(READ ${PARAM_SRC} param_data)

# remove whitespace
string(REGEX REPLACE "\n +" "\n" param_data ${param_data})

# replace more spaces to one space
string(REGEX REPLACE "[ \t]+" " " param_data "${param_data}")

# remove empty line
string(REGEX REPLACE "\n[\n]+" "\n" comp_data "${comp_data}")

# text to hex
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/param/${PARAM_SRC_NAME_WE}.text2hex.txt "${param_data}")
file(READ ${CMAKE_CURRENT_BINARY_DIR}/param/${PARAM_SRC_NAME_WE}.text2hex.txt param_data_hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," param_data_hex ${param_data_hex})
string(FIND "${param_data_hex}" "," tail_comma REVERSE)
string(SUBSTRING "${param_data_hex}" 0 ${tail_comma} param_data_hex)

# generate model param header file
file(WRITE ${PARAM_HEADER} "static const char ${PARAM_SRC_NAME_WE}_param_data[] = {${param_data_hex},0x00};\n")
