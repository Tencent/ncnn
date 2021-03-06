
# must define SRC DST CLASS

file(READ ${SRC} source_data)

# replace
string(TOUPPER ${CLASS} CLASS_UPPER)
string(TOLOWER ${CLASS} CLASS_LOWER)

string(REGEX REPLACE "LAYER_${CLASS_UPPER}_MIPS_H" "LAYER_${CLASS_UPPER}_MIPS_MMI_H" source_data "${source_data}")
string(REGEX REPLACE "${CLASS}_mips" "${CLASS}_mips_mmi" source_data "${source_data}")
string(REGEX REPLACE "#include \"${CLASS_LOWER}_mips.h\"" "#include \"${CLASS_LOWER}_mips_mmi.h\"" source_data "${source_data}")

file(WRITE ${DST} "${source_data}")
