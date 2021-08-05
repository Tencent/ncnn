
# must define SRC DST CLASS

file(READ ${SRC} source_data)

# replace
string(TOUPPER ${CLASS} CLASS_UPPER)
string(TOLOWER ${CLASS} CLASS_LOWER)

string(REGEX REPLACE "LAYER_${CLASS_UPPER}_ARM_H" "LAYER_${CLASS_UPPER}_ARM_ARM82DOT_H" source_data "${source_data}")
string(REGEX REPLACE "${CLASS}_arm" "${CLASS}_arm_arm82dot" source_data "${source_data}")
string(REGEX REPLACE "#include \"${CLASS_LOWER}_arm.h\"" "#include \"${CLASS_LOWER}_arm_arm82dot.h\"" source_data "${source_data}")

file(WRITE ${DST} "${source_data}")
