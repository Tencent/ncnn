# must define SHADER_COMP_HEADER SHADER_SRC

file(READ ${SHADER_SRC} comp_data)

# skip leading comment
string(FIND "${comp_data}" "#version" version_start)
if(NOT ${version_start} EQUAL -1)
    string(SUBSTRING "${comp_data}" ${version_start} -1 comp_data)
endif()

# WebGPU transformation: convert push constants to uniform bindings
# Transform: layout (push_constant) uniform parameter { ... } p;
# To: layout (binding = 1) uniform parameter_blob { parameter p; };

# Find push_constant blocks and transform them
string(REGEX REPLACE 
    "layout \\(push_constant\\) uniform ([a-zA-Z_][a-zA-Z0-9_]*)\n\\{\n([^}]*)\n\\} ([a-zA-Z_][a-zA-Z0-9_]*);"
    "struct \\1\n{\n\\2\n};\nlayout (binding = 1) uniform \\1_blob { \\1 \\3; };"
    comp_data "${comp_data}")

# remove whitespace
string(REGEX REPLACE "\n +" "\n" comp_data "${comp_data}")

# remove empty line
string(REGEX REPLACE "\n\n" "\n" comp_data "${comp_data}")

get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)

# text to hex
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/${SHADER_SRC_NAME_WE}.webgpu.text2hex.txt "${comp_data}")
file(READ ${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/${SHADER_SRC_NAME_WE}.webgpu.text2hex.txt comp_data_hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," comp_data_hex ${comp_data_hex})
string(FIND "${comp_data_hex}" "," tail_comma REVERSE)
string(SUBSTRING "${comp_data_hex}" 0 ${tail_comma} comp_data_hex)

file(WRITE ${SHADER_COMP_HEADER} "static const char ${SHADER_SRC_NAME_WE}_comp_data[] = {${comp_data_hex}};\n")