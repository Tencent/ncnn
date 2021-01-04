
# must define SRC DST CLASS

file(READ ${SRC} source_data)

# replace
string(TOUPPER ${CLASS} CLASS_UPPER)
string(TOLOWER ${CLASS} CLASS_LOWER)

string(REGEX REPLACE "LAYER_${CLASS_UPPER}_X86_H" "LAYER_${CLASS_UPPER}_X86_AVX2_H" source_data "${source_data}")
string(REGEX REPLACE "${CLASS}_x86" "${CLASS}_x86_avx2" source_data "${source_data}")
string(REGEX REPLACE "#include \"${CLASS_LOWER}_x86.h\"" "#include \"${CLASS_LOWER}_x86_avx2.h\"" source_data "${source_data}")

file(WRITE ${DST} "${source_data}")
