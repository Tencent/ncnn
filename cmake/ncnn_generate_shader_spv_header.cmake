
function(ncnn_generate_shader_spv_header SHADER_SPV_HEADER SHADER_SPV_HEX_HEADERS SHADER_SRC)

    # fp32
    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)

    set(SHADER_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float -Dsfpvec2=vec2 -Dsfpvec4=vec4 -Dsfpmat4=mat4 -Dafp=float -Dafpvec2=vec2 -Dafpvec4=vec4 -Dafpmat4=mat4 -Dsfp2afp\(v\)=v -Dafp2sfp\(v\)=v -Dsfp2afpvec2\(v\)=v -Dafp2sfpvec2\(v\)=v -Dsfp2afpvec4\(v\)=v -Dafp2sfpvec4\(v\)=v -Dsfp2afpmat4\(v\)=v -Dafp2sfpmat4\(v\)=v -V -s -e ${SHADER_SRC_NAME_WE} --source-entrypoint main -x -o ${SHADER_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    # fp16 packed
    set(SHADER_fp16p_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16p")

    set(SHADER_fp16p_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16p_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_fp16p_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float -Dsfpvec2=uint -Dsfpvec4=uvec2 -Dafp=float -Dafpvec2=vec2 -Dafpvec4=vec4 -Dafpmat4=mat4 -Dsfp2afp\(v\)=v -Dafp2sfp\(v\)=v -Dsfp2afpvec2\(v\)=unpackHalf2x16\(v\) -Dafp2sfpvec2\(v\)=packHalf2x16\(v\) -Dsfp2afpvec4\(v\)=vec4\(unpackHalf2x16\(v.x\),unpackHalf2x16\(v.y\)\) -Dafp2sfpvec4\(v\)=uvec2\(packHalf2x16\(v.rg\),packHalf2x16\(v.ba\)\) -DNCNN_fp16_packed=1 -V -s -e ${SHADER_fp16p_SRC_NAME_WE} --source-entrypoint main -x -o ${SHADER_fp16p_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_fp16p_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_fp16p_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    # fp16 storage
    set(SHADER_fp16s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16s")

    set(SHADER_fp16s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_fp16s_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float16_t -Dsfpvec2=f16vec2 -Dsfpvec4=f16vec4 -Dafp=float -Dafpvec2=vec2 -Dafpvec4=vec4 -Dafpmat4=mat4 -Dsfp2afp\(v\)=float\(v\) -Dafp2sfp\(v\)=float16_t\(v\) -Dsfp2afpvec2\(v\)=vec2\(v\) -Dafp2sfpvec2\(v\)=f16vec2\(v\) -Dsfp2afpvec4\(v\)=vec4\(v\) -Dafp2sfpvec4\(v\)=f16vec4\(v\) -DNCNN_fp16_storage=1 -V -s -e ${SHADER_fp16s_SRC_NAME_WE} --source-entrypoint main -x -o ${SHADER_fp16s_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_fp16s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_fp16s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    # fp16 storage + fp16 arithmetic
    set(SHADER_fp16a_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16a")

    set(SHADER_fp16a_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16a_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_fp16a_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -Dsfp=float16_t -Dsfpvec2=f16vec2 -Dsfpvec4=f16vec4 -Dsfpmat4=f16mat4 -Dafp=float16_t -Dafpvec2=f16vec2 -Dafpvec4=f16vec4 -Dafpmat4=f16mat4 -Dsfp2afp\(v\)=v -Dafp2sfp\(v\)=v -Dsfp2afpvec2\(v\)=v -Dafp2sfpvec2\(v\)=v -Dsfp2afpvec4\(v\)=v -Dafp2sfpvec4\(v\)=v -Dsfp2afpmat4\(v\)=v -Dafp2sfpmat4\(v\)=v -DNCNN_fp16_storage=1 -DNCNN_fp16_arithmetic=1 -V -s -e ${SHADER_fp16a_SRC_NAME_WE} --source-entrypoint main -x -o ${SHADER_fp16a_SPV_HEX_FILE} ${SHADER_SRC}
        DEPENDS ${SHADER_SRC}
        COMMENT "Building SPIR-V module ${SHADER_fp16a_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_fp16a_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)

    set(LOCAL_SHADER_SPV_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.spv.h)

    file(WRITE ${LOCAL_SHADER_SPV_HEADER}
        "static const uint32_t ${SHADER_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_SRC_NAME_WE}.spv.hex.h\"\n};\n"
        "static const uint32_t ${SHADER_fp16p_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_fp16p_SRC_NAME_WE}.spv.hex.h\"\n};\n"
        "static const uint32_t ${SHADER_fp16s_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_fp16s_SRC_NAME_WE}.spv.hex.h\"\n};\n"
        "static const uint32_t ${SHADER_fp16a_SRC_NAME_WE}_spv_data[] = {\n#include \"${SHADER_fp16a_SRC_NAME_WE}.spv.hex.h\"\n};\n"
    )

    set_source_files_properties(${LOCAL_SHADER_SPV_HEADER} PROPERTIES GENERATED TRUE)

    set(LOCAL_SHADER_SPV_HEX_HEADERS
        ${SHADER_SPV_HEX_FILE}
        ${SHADER_fp16p_SPV_HEX_FILE}
        ${SHADER_fp16s_SPV_HEX_FILE}
        ${SHADER_fp16a_SPV_HEX_FILE}
    )

    set(${SHADER_SPV_HEADER} ${LOCAL_SHADER_SPV_HEADER} PARENT_SCOPE)
    set(${SHADER_SPV_HEX_HEADERS} ${LOCAL_SHADER_SPV_HEX_HEADERS} PARENT_SCOPE)

endfunction()
