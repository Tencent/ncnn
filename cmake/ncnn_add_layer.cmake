
macro(ncnn_add_arch_opt_layer class NCNN_TARGET_ARCH_OPT NCNN_TARGET_ARCH_OPT_CFLAGS)
    set(NCNN_${NCNN_TARGET_ARCH}_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}.h)
    set(NCNN_${NCNN_TARGET_ARCH}_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}.cpp)

    if(WITH_LAYER_${name} AND EXISTS ${NCNN_${NCNN_TARGET_ARCH}_HEADER} AND EXISTS ${NCNN_${NCNN_TARGET_ARCH}_SOURCE})

        set(NCNN_${NCNN_TARGET_ARCH_OPT}_HEADER ${CMAKE_CURRENT_BINARY_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}_${NCNN_TARGET_ARCH_OPT}.h)
        set(NCNN_${NCNN_TARGET_ARCH_OPT}_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}_${NCNN_TARGET_ARCH_OPT}.cpp)

        add_custom_command(
            OUTPUT ${NCNN_${NCNN_TARGET_ARCH_OPT}_HEADER}
            COMMAND ${CMAKE_COMMAND} -DSRC=${NCNN_${NCNN_TARGET_ARCH}_HEADER} -DDST=${NCNN_${NCNN_TARGET_ARCH_OPT}_HEADER} -DCLASS=${class} -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_generate_${NCNN_TARGET_ARCH_OPT}_source.cmake"
            DEPENDS ${NCNN_${NCNN_TARGET_ARCH}_HEADER}
            COMMENT "Generating source ${name}_${NCNN_TARGET_ARCH}_${NCNN_TARGET_ARCH_OPT}.h"
            VERBATIM
        )
        set_source_files_properties(${NCNN_${NCNN_TARGET_ARCH_OPT}_HEADER} PROPERTIES GENERATED TRUE)

        add_custom_command(
            OUTPUT ${NCNN_${NCNN_TARGET_ARCH_OPT}_SOURCE}
            COMMAND ${CMAKE_COMMAND} -DSRC=${NCNN_${NCNN_TARGET_ARCH}_SOURCE} -DDST=${NCNN_${NCNN_TARGET_ARCH_OPT}_SOURCE} -DCLASS=${class} -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_generate_${NCNN_TARGET_ARCH_OPT}_source.cmake"
            DEPENDS ${NCNN_${NCNN_TARGET_ARCH}_SOURCE}
            COMMENT "Generating source ${name}_${NCNN_TARGET_ARCH}_${NCNN_TARGET_ARCH_OPT}.cpp"
            VERBATIM
        )
        set_source_files_properties(${NCNN_${NCNN_TARGET_ARCH_OPT}_SOURCE} PROPERTIES GENERATED TRUE)

        set_source_files_properties(${NCNN_${NCNN_TARGET_ARCH_OPT}_SOURCE} PROPERTIES COMPILE_FLAGS ${NCNN_TARGET_ARCH_OPT_CFLAGS})

        list(APPEND ncnn_SRCS ${NCNN_${NCNN_TARGET_ARCH_OPT}_HEADER} ${NCNN_${NCNN_TARGET_ARCH_OPT}_SOURCE})

        # generate layer_declaration and layer_registry file
        set(layer_declaration "${layer_declaration}#include \"layer/${name}.h\"\n")
        set(layer_declaration_class "class ${class}_final_${NCNN_TARGET_ARCH_OPT} : virtual public ${class}")
        set(create_pipeline_content "        { int ret = ${class}::create_pipeline(opt); if (ret) return ret; }\n")
        set(destroy_pipeline_content "        { int ret = ${class}::destroy_pipeline(opt); if (ret) return ret; }\n")

        set(layer_declaration "${layer_declaration}#include \"layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}_${NCNN_TARGET_ARCH_OPT}.h\"\n")
        set(layer_declaration_class "${layer_declaration_class}, virtual public ${class}_${NCNN_TARGET_ARCH}_${NCNN_TARGET_ARCH_OPT}")
        set(create_pipeline_content "${create_pipeline_content}        { int ret = ${class}_${NCNN_TARGET_ARCH}_${NCNN_TARGET_ARCH_OPT}::create_pipeline(opt); if (ret) return ret; }\n")
        set(destroy_pipeline_content "        { int ret = ${class}_${NCNN_TARGET_ARCH}_${NCNN_TARGET_ARCH_OPT}::destroy_pipeline(opt); if (ret) return ret; }\n${destroy_pipeline_content}")

        if(WITH_LAYER_${name}_vulkan)
            set(layer_declaration "${layer_declaration}#include \"layer/vulkan/${name}_vulkan.h\"\n")
            set(layer_declaration_class "${layer_declaration_class}, virtual public ${class}_vulkan")
            set(create_pipeline_content "${create_pipeline_content}        if (vkdev) { int ret = ${class}_vulkan::create_pipeline(opt); if (ret) return ret; }\n")
            set(destroy_pipeline_content "        if (vkdev) { int ret = ${class}_vulkan::destroy_pipeline(opt); if (ret) return ret; }\n${destroy_pipeline_content}")
        endif()

        set(layer_declaration "${layer_declaration}namespace ncnn {\n${layer_declaration_class}\n{\n")
        set(layer_declaration "${layer_declaration}public:\n")
        set(layer_declaration "${layer_declaration}    virtual int create_pipeline(const Option& opt) {\n${create_pipeline_content}        return 0;\n    }\n")
        set(layer_declaration "${layer_declaration}    virtual int destroy_pipeline(const Option& opt) {\n${destroy_pipeline_content}        return 0;\n    }\n")
        set(layer_declaration "${layer_declaration}};\n")
        set(layer_declaration "${layer_declaration}DEFINE_LAYER_CREATOR(${class}_final_${NCNN_TARGET_ARCH_OPT})\n} // namespace ncnn\n\n")

        set(layer_registry_${NCNN_TARGET_ARCH_OPT} "${layer_registry_${NCNN_TARGET_ARCH_OPT}}#if NCNN_STRING\n{\"${class}\", ${class}_final_${NCNN_TARGET_ARCH_OPT}_layer_creator},\n#else\n{${class}_final_${NCNN_TARGET_ARCH_OPT}_layer_creator},\n#endif\n")
    else()
        # no arm optimized version
        if(WITH_LAYER_${name})
            set(layer_registry_${NCNN_TARGET_ARCH_OPT} "${layer_registry_${NCNN_TARGET_ARCH_OPT}}#if NCNN_STRING\n{\"${class}\", ${class}_final_layer_creator},\n#else\n{${class}_final_layer_creator},\n#endif\n")
        else()
            set(layer_registry_${NCNN_TARGET_ARCH_OPT} "${layer_registry_${NCNN_TARGET_ARCH_OPT}}#if NCNN_STRING\n{\"${class}\", 0},\n#else\n{0},\n#endif\n")
        endif()
    endif()

endmacro()

macro(ncnn_add_arch_opt_source class NCNN_TARGET_ARCH_OPT NCNN_TARGET_ARCH_OPT_CFLAGS)
    set(NCNN_${NCNN_TARGET_ARCH_OPT}_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}_${NCNN_TARGET_ARCH_OPT}.cpp)

    if(WITH_LAYER_${name} AND EXISTS ${NCNN_${NCNN_TARGET_ARCH_OPT}_SOURCE})
        set_source_files_properties(${NCNN_${NCNN_TARGET_ARCH_OPT}_SOURCE} PROPERTIES COMPILE_FLAGS ${NCNN_TARGET_ARCH_OPT_CFLAGS})
        list(APPEND ncnn_SRCS ${NCNN_${NCNN_TARGET_ARCH_OPT}_SOURCE})
    endif()
endmacro()

macro(ncnn_add_layer class)
    string(TOLOWER ${class} name)

    # WITH_LAYER_xxx option
    if(${ARGC} EQUAL 2)
        option(WITH_LAYER_${name} "build with layer ${name}" ${ARGV1})
    else()
        option(WITH_LAYER_${name} "build with layer ${name}" ON)
    endif()

    if(NCNN_CMAKE_VERBOSE)
        message(STATUS "WITH_LAYER_${name} = ${WITH_LAYER_${name}}")
    endif()

    if(WITH_LAYER_${name})
        list(APPEND ncnn_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/layer/${name}.cpp)

        # look for arch specific implementation and append source
        # optimized implementation for armv7, aarch64 or x86
        set(LAYER_ARCH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}.cpp)
        if(EXISTS ${LAYER_ARCH_SRC})
            set(WITH_LAYER_${name}_${NCNN_TARGET_ARCH} 1)
            list(APPEND ncnn_SRCS ${LAYER_ARCH_SRC})
        endif()

        set(LAYER_VULKAN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/layer/vulkan/${name}_vulkan.cpp)
        if(NCNN_VULKAN AND EXISTS ${LAYER_VULKAN_SRC})
            set(WITH_LAYER_${name}_vulkan 1)
            list(APPEND ncnn_SRCS ${LAYER_VULKAN_SRC})
        endif()
    endif()

    # generate layer_declaration and layer_registry file
    if(WITH_LAYER_${name})
        set(layer_declaration "${layer_declaration}#include \"layer/${name}.h\"\n")
        set(layer_declaration_class "class ${class}_final : virtual public ${class}")
        set(create_pipeline_content "        { int ret = ${class}::create_pipeline(opt); if (ret) return ret; }\n")
        set(destroy_pipeline_content "        { int ret = ${class}::destroy_pipeline(opt); if (ret) return ret; }\n")

        source_group ("sources\\\\layers" FILES "${CMAKE_CURRENT_SOURCE_DIR}/layer/${name}.cpp")
    endif()

    if(WITH_LAYER_${name}_${NCNN_TARGET_ARCH})
        set(layer_declaration "${layer_declaration}#include \"layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}.h\"\n")
        set(layer_declaration_class "${layer_declaration_class}, virtual public ${class}_${NCNN_TARGET_ARCH}")
        set(create_pipeline_content "${create_pipeline_content}        { int ret = ${class}_${NCNN_TARGET_ARCH}::create_pipeline(opt); if (ret) return ret; }\n")
        set(destroy_pipeline_content "        { int ret = ${class}_${NCNN_TARGET_ARCH}::destroy_pipeline(opt); if (ret) return ret; }\n${destroy_pipeline_content}")

        source_group ("sources\\\\layers\\\\${NCNN_TARGET_ARCH}" FILES "${CMAKE_CURRENT_SOURCE_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}.cpp")
    endif()

    if(WITH_LAYER_${name}_vulkan)
        set(layer_declaration "${layer_declaration}#include \"layer/vulkan/${name}_vulkan.h\"\n")
        set(layer_declaration_class "${layer_declaration_class}, virtual public ${class}_vulkan")
        set(create_pipeline_content "${create_pipeline_content}        if (vkdev) { int ret = ${class}_vulkan::create_pipeline(opt); if (ret) return ret; }\n")
        set(destroy_pipeline_content "        if (vkdev) { int ret = ${class}_vulkan::destroy_pipeline(opt); if (ret) return ret; }\n${destroy_pipeline_content}")

        file(GLOB_RECURSE NCNN_SHADER_SRCS "layer/vulkan/shader/${name}.comp")
        file(GLOB_RECURSE NCNN_SHADER_SUBSRCS "layer/vulkan/shader/${name}_*.comp")
        list(APPEND NCNN_SHADER_SRCS ${NCNN_SHADER_SUBSRCS})
        foreach(NCNN_SHADER_SRC ${NCNN_SHADER_SRCS})
            ncnn_add_shader(${NCNN_SHADER_SRC})
        endforeach()

        source_group ("sources\\\\layers\\\\vulkan" FILES "${CMAKE_CURRENT_SOURCE_DIR}/layer/vulkan/${name}_vulkan.cpp")
    endif()

    if(WITH_LAYER_${name})
        set(layer_declaration "${layer_declaration}namespace ncnn {\n${layer_declaration_class}\n{\n")
        set(layer_declaration "${layer_declaration}public:\n")
        set(layer_declaration "${layer_declaration}    virtual int create_pipeline(const Option& opt) {\n${create_pipeline_content}        return 0;\n    }\n")
        set(layer_declaration "${layer_declaration}    virtual int destroy_pipeline(const Option& opt) {\n${destroy_pipeline_content}        return 0;\n    }\n")
        set(layer_declaration "${layer_declaration}};\n")
        set(layer_declaration "${layer_declaration}DEFINE_LAYER_CREATOR(${class}_final)\n} // namespace ncnn\n\n")
    endif()

    if(WITH_LAYER_${name})
        set(layer_registry "${layer_registry}#if NCNN_STRING\n{\"${class}\", ${class}_final_layer_creator},\n#else\n{${class}_final_layer_creator},\n#endif\n")
    else()
        set(layer_registry "${layer_registry}#if NCNN_STRING\n{\"${class}\", 0},\n#else\n{0},\n#endif\n")
    endif()

    if(NCNN_RUNTIME_CPU AND NCNN_TARGET_ARCH STREQUAL "x86")
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            if(NCNN_AVX512)
                ncnn_add_arch_opt_layer(${class} avx512 "/arch:AVX512 /D__FMA__ /D__F16C__")
            endif()
            if(NCNN_FMA)
                ncnn_add_arch_opt_layer(${class} fma "/arch:AVX /D__FMA__ /D__F16C__")
            endif()
            if(NCNN_AVX)
                ncnn_add_arch_opt_layer(${class} avx "/arch:AVX")
            endif()
            if(NCNN_AVX512VNNI)
                ncnn_add_arch_opt_source(${class} avx512vnni "/arch:AVX512 /D__FMA__ /D__F16C__ /D__AVX512VNNI__")
            endif()
            if(NCNN_AVXVNNI)
                ncnn_add_arch_opt_source(${class} avxvnni "/arch:AVX2 /D__FMA__ /D__F16C__ /D__AVXVNNI__")
            endif()
            if(NCNN_AVX2)
                ncnn_add_arch_opt_source(${class} avx2 "/arch:AVX2 /D__FMA__ /D__F16C__")
            endif()
            if(NCNN_XOP)
                ncnn_add_arch_opt_source(${class} xop "/arch:AVX /D__XOP__")
            endif()
            if(NCNN_F16C)
                ncnn_add_arch_opt_source(${class} f16c "/arch:AVX /D__F16C__")
            endif()
        else()
            if(NCNN_AVX512)
                ncnn_add_arch_opt_layer(${class} avx512 "-mavx512f -mavx512bw -mavx512vl -mfma -mf16c")
            endif()
            if(NCNN_FMA)
                ncnn_add_arch_opt_layer(${class} fma "-mavx -mfma")
            endif()
            if(NCNN_AVX)
                ncnn_add_arch_opt_layer(${class} avx "-mavx")
            endif()
            if(NCNN_AVX512VNNI)
                ncnn_add_arch_opt_source(${class} avx512vnni "-mavx512f -mavx512bw -mavx512vl -mfma -mf16c -mavx512vnni")
            endif()
            if(NCNN_AVXVNNI)
                ncnn_add_arch_opt_source(${class} avxvnni "-mavx2 -mfma -mf16c -mavxvnni")
            endif()
            if(NCNN_AVX2)
                ncnn_add_arch_opt_source(${class} avx2 "-mavx2 -mfma -mf16c")
            endif()
            if(NCNN_XOP)
                ncnn_add_arch_opt_source(${class} xop "-mavx -mxop")
            endif()
            if(NCNN_F16C)
                ncnn_add_arch_opt_source(${class} f16c "-mavx -mf16c")
            endif()
        endif()
    endif()

    if(NCNN_RUNTIME_CPU AND ((IOS AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64") OR (APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)")))
        if(NCNN_ARM82)
            ncnn_add_arch_opt_layer(${class} arm82 "-march=armv8.2-a+fp16")
        endif()
        if(NCNN_ARM82DOT)
            ncnn_add_arch_opt_source(${class} arm82dot "-march=armv8.2-a+fp16+dotprod")
        endif()
    endif()

    if(NCNN_RUNTIME_CPU AND NCNN_TARGET_ARCH STREQUAL "mips")
        if(NCNN_MSA)
            ncnn_add_arch_opt_layer(${class} msa "-mmsa")
        endif()
        if(NCNN_MMI)
            ncnn_add_arch_opt_source(${class} mmi "-mloongson-mmi")
        endif()
    endif()

    if(NCNN_RUNTIME_CPU AND NCNN_RVV AND NCNN_TARGET_ARCH STREQUAL "riscv")
        if(NCNN_COMPILER_SUPPORT_RVV_FP16)
            ncnn_add_arch_opt_layer(${class} rvv "-march=rv64gcv_zfh")
        elseif(NCNN_COMPILER_SUPPORT_RVV)
            ncnn_add_arch_opt_layer(${class} rvv "-march=rv64gcv")
        endif()
    endif()

    # generate layer_type_enum file
    set(layer_type_enum "${layer_type_enum}${class} = ${__LAYER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_TYPE_ENUM_INDEX "${__LAYER_TYPE_ENUM_INDEX}+1")
endmacro()
