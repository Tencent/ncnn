
# must define SRC DST CLASS

file(READ ${SRC} source_data)

# replace
string(TOUPPER ${CLASS} CLASS_UPPER)
string(TOLOWER ${CLASS} CLASS_LOWER)

string(REGEX REPLACE "LAYER_${CLASS_UPPER}_LOONGARCH_H" "LAYER_${CLASS_UPPER}_LOONGARCH_LSX_H" source_data "${source_data}")
string(REGEX REPLACE "${CLASS}_loongarch" "${CLASS}_loongarch_lsx" source_data "${source_data}")
string(REGEX REPLACE "#include \"${CLASS_LOWER}_loongarch.h\"" "#include \"${CLASS_LOWER}_loongarch_lsx.h\"" source_data "${source_data}")

file(WRITE ${DST} "${source_data}")
