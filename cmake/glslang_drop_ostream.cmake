# this cmake script remove ostream usage in glslang

function(glslang_drop_ostream GLSLANG_ROOT)

    # patch glslang/SPIRV/GlslangToSpv.h
    file(STRINGS ${GLSLANG_ROOT}/SPIRV/GlslangToSpv.h GlslangToSpv_HDR)
    file(WRITE ${GLSLANG_ROOT}/SPIRV/GlslangToSpv.h "")
    foreach(LINE IN LISTS GlslangToSpv_HDR)
        if(LINE MATCHES "^bool OutputSpvBin(.*);" OR LINE MATCHES "^bool OutputSpvHex(.*);")
            continue()
        endif()

        file(APPEND ${GLSLANG_ROOT}/SPIRV/GlslangToSpv.h "${LINE}\n")
    endforeach()

    # patch glslang/SPIRV/GlslangToSpv.cpp
    file(STRINGS ${GLSLANG_ROOT}/SPIRV/GlslangToSpv.cpp GlslangToSpv_SRC)
    file(WRITE ${GLSLANG_ROOT}/SPIRV/GlslangToSpv.cpp "")
    foreach(LINE IN LISTS GlslangToSpv_SRC)
        if(LINE MATCHES "^bool OutputSpvBin(.*)")
            set(GlslangToSpv_SRC_OutputSpvBin TRUE)
        endif()
        if(LINE MATCHES "^bool OutputSpvHex(.*)")
            set(GlslangToSpv_SRC_OutputSpvHex TRUE)
        endif()

        if(GlslangToSpv_SRC_OutputSpvBin OR GlslangToSpv_SRC_OutputSpvHex)
            if(LINE MATCHES "^}$")
                if(GlslangToSpv_SRC_OutputSpvBin)
                    unset(GlslangToSpv_SRC_OutputSpvBin)
                endif()
                if(GlslangToSpv_SRC_OutputSpvHex)
                    unset(GlslangToSpv_SRC_OutputSpvHex)
                endif()
            endif()
            continue()
        endif()

        file(APPEND ${GLSLANG_ROOT}/SPIRV/GlslangToSpv.cpp "${LINE}\n")
    endforeach()

endfunction()
