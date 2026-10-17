
# must define SRC DST CLASS

file(READ ${SRC} source_data)

# MXU2 reuses MSA kernels, but must not include the compiler's MSA header.
# Both the generated layer source and the non-runtime base source use this.
string(REPLACE "#include <msa.h>" "#include \"mips_usability.h\"" source_data "${source_data}")

if(NOT NCNN_MXU2_BASE)
# replace
string(TOUPPER ${CLASS} CLASS_UPPER)
string(TOLOWER ${CLASS} CLASS_LOWER)

string(REGEX REPLACE "LAYER_${CLASS_UPPER}_MIPS_H" "LAYER_${CLASS_UPPER}_MIPS_MXU2_H" source_data "${source_data}")
string(REGEX REPLACE "${CLASS}_mips" "${CLASS}_mips_mxu2" source_data "${source_data}")
string(REGEX REPLACE "#include \"${CLASS_LOWER}_mips.h\"" "#include \"${CLASS_LOWER}_mips_mxu2.h\"" source_data "${source_data}")

endif()

get_filename_component(DST_DIR "${DST}" DIRECTORY)
file(MAKE_DIRECTORY "${DST_DIR}")
file(WRITE "${DST}" "${source_data}")
