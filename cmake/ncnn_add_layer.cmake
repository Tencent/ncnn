
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
        set(layer_registry "${layer_registry}#if NCNN_STRING\n{\"${class}\",${class}_final_layer_creator},\n#else\n{${class}_final_layer_creator},\n#endif\n")
    else()
        set(layer_registry "${layer_registry}#if NCNN_STRING\n{\"${class}\",0},\n#else\n{0},\n#endif\n")
    endif()


    if(NCNN_RUNTIME_CPU AND NCNN_AVX2 AND NCNN_TARGET_ARCH STREQUAL "x86")
        # enable avx2
        set(NCNN_X86_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}.h)
        set(NCNN_X86_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}.cpp)

        if(WITH_LAYER_${name} AND EXISTS ${NCNN_X86_HEADER} AND EXISTS ${NCNN_X86_SOURCE})

            set(NCNN_AVX2_HEADER ${CMAKE_CURRENT_BINARY_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}_avx2.h)
            set(NCNN_AVX2_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}_avx2.cpp)

            add_custom_command(
                OUTPUT ${NCNN_AVX2_HEADER}
                COMMAND ${CMAKE_COMMAND} -DSRC=${NCNN_X86_HEADER} -DDST=${NCNN_AVX2_HEADER} -DCLASS=${class} -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_generate_avx2_source.cmake"
                DEPENDS ${NCNN_X86_HEADER}
                COMMENT "Generating source ${name}_${NCNN_TARGET_ARCH}_avx2.h"
                VERBATIM
            )
            set_source_files_properties(${NCNN_AVX2_HEADER} PROPERTIES GENERATED TRUE)

            add_custom_command(
                OUTPUT ${NCNN_AVX2_SOURCE}
                COMMAND ${CMAKE_COMMAND} -DSRC=${NCNN_X86_SOURCE} -DDST=${NCNN_AVX2_SOURCE} -DCLASS=${class} -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_generate_avx2_source.cmake"
                DEPENDS ${NCNN_X86_SOURCE}
                COMMENT "Generating source ${name}_${NCNN_TARGET_ARCH}_avx2.cpp"
                VERBATIM
            )
            set_source_files_properties(${NCNN_AVX2_SOURCE} PROPERTIES GENERATED TRUE)

            if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC"))
                set_source_files_properties(${NCNN_AVX2_SOURCE} PROPERTIES COMPILE_FLAGS "/arch:AVX2 /DAVX2 /fp:strict")
            else()
                set_source_files_properties(${NCNN_AVX2_SOURCE} PROPERTIES COMPILE_FLAGS "-mfma -mf16c -mavx2")
            endif()

            list(APPEND ncnn_SRCS ${NCNN_AVX2_HEADER} ${NCNN_AVX2_SOURCE})

            # generate layer_declaration and layer_registry_avx2 file
            set(layer_declaration "${layer_declaration}#include \"layer/${name}.h\"\n")
            set(layer_declaration_class "class ${class}_final_avx2 : virtual public ${class}")
            set(create_pipeline_content "        { int ret = ${class}::create_pipeline(opt); if (ret) return ret; }\n")
            set(destroy_pipeline_content "        { int ret = ${class}::destroy_pipeline(opt); if (ret) return ret; }\n")

            set(layer_declaration "${layer_declaration}#include \"layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}_avx2.h\"\n")
            set(layer_declaration_class "${layer_declaration_class}, virtual public ${class}_${NCNN_TARGET_ARCH}_avx2")
            set(create_pipeline_content "${create_pipeline_content}        { int ret = ${class}_${NCNN_TARGET_ARCH}_avx2::create_pipeline(opt); if (ret) return ret; }\n")
            set(destroy_pipeline_content "        { int ret = ${class}_${NCNN_TARGET_ARCH}_avx2::destroy_pipeline(opt); if (ret) return ret; }\n${destroy_pipeline_content}")

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
            set(layer_declaration "${layer_declaration}DEFINE_LAYER_CREATOR(${class}_final_avx2)\n} // namespace ncnn\n\n")

            set(layer_registry_avx2 "${layer_registry_avx2}#if NCNN_STRING\n{\"${class}\",${class}_final_avx2_layer_creator},\n#else\n{${class}_final_avx2_layer_creator},\n#endif\n")
        else()
            # no arm optimized version
            if(WITH_LAYER_${name})
                set(layer_registry_avx2 "${layer_registry_avx2}#if NCNN_STRING\n{\"${class}\",${class}_final_layer_creator},\n#else\n{${class}_final_layer_creator},\n#endif\n")
            else()
                set(layer_registry_avx2 "${layer_registry_avx2}#if NCNN_STRING\n{\"${class}\",0},\n#else\n{0},\n#endif\n")
            endif()
        endif()
    endif()

    if(NCNN_RUNTIME_CPU AND NCNN_ARM82 AND ((IOS AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64)")))
        # enable armv8.2a+fp16
        set(NCNN_ARM_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}.h)
        set(NCNN_ARM_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}.cpp)

        if(WITH_LAYER_${name} AND EXISTS ${NCNN_ARM_HEADER} AND EXISTS ${NCNN_ARM_SOURCE})

            set(NCNN_ARM82_HEADER ${CMAKE_CURRENT_BINARY_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}_arm82.h)
            set(NCNN_ARM82_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}_arm82.cpp)

            add_custom_command(
                OUTPUT ${NCNN_ARM82_HEADER}
                COMMAND ${CMAKE_COMMAND} -DSRC=${NCNN_ARM_HEADER} -DDST=${NCNN_ARM82_HEADER} -DCLASS=${class} -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_generate_arm82_source.cmake"
                DEPENDS ${NCNN_ARM_HEADER}
                COMMENT "Generating source ${name}_${NCNN_TARGET_ARCH}_arm82.h"
                VERBATIM
            )
            set_source_files_properties(${NCNN_ARM82_HEADER} PROPERTIES GENERATED TRUE)

            add_custom_command(
                OUTPUT ${NCNN_ARM82_SOURCE}
                COMMAND ${CMAKE_COMMAND} -DSRC=${NCNN_ARM_SOURCE} -DDST=${NCNN_ARM82_SOURCE} -DCLASS=${class} -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ncnn_generate_arm82_source.cmake"
                DEPENDS ${NCNN_ARM_SOURCE}
                COMMENT "Generating source ${name}_${NCNN_TARGET_ARCH}_arm82.cpp"
                VERBATIM
            )
            set_source_files_properties(${NCNN_ARM82_SOURCE} PROPERTIES GENERATED TRUE)

            set_source_files_properties(${NCNN_ARM82_SOURCE} PROPERTIES COMPILE_FLAGS "-march=armv8.2-a+fp16")

            list(APPEND ncnn_SRCS ${NCNN_ARM82_HEADER} ${NCNN_ARM82_SOURCE})

            # generate layer_declaration and layer_registry_arm82 file
            set(layer_declaration "${layer_declaration}#include \"layer/${name}.h\"\n")
            set(layer_declaration_class "class ${class}_final_arm82 : virtual public ${class}")
            set(create_pipeline_content "        { int ret = ${class}::create_pipeline(opt); if (ret) return ret; }\n")
            set(destroy_pipeline_content "        { int ret = ${class}::destroy_pipeline(opt); if (ret) return ret; }\n")

            set(layer_declaration "${layer_declaration}#include \"layer/${NCNN_TARGET_ARCH}/${name}_${NCNN_TARGET_ARCH}_arm82.h\"\n")
            set(layer_declaration_class "${layer_declaration_class}, virtual public ${class}_${NCNN_TARGET_ARCH}_arm82")
            set(create_pipeline_content "${create_pipeline_content}        { int ret = ${class}_${NCNN_TARGET_ARCH}_arm82::create_pipeline(opt); if (ret) return ret; }\n")
            set(destroy_pipeline_content "        { int ret = ${class}_${NCNN_TARGET_ARCH}_arm82::destroy_pipeline(opt); if (ret) return ret; }\n${destroy_pipeline_content}")

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
            set(layer_declaration "${layer_declaration}DEFINE_LAYER_CREATOR(${class}_final_arm82)\n} // namespace ncnn\n\n")

            set(layer_registry_arm82 "${layer_registry_arm82}#if NCNN_STRING\n{\"${class}\",${class}_final_arm82_layer_creator},\n#else\n{${class}_final_arm82_layer_creator},\n#endif\n")
        else()
            # no arm optimized version
            if(WITH_LAYER_${name})
                set(layer_registry_arm82 "${layer_registry_arm82}#if NCNN_STRING\n{\"${class}\",${class}_final_layer_creator},\n#else\n{${class}_final_layer_creator},\n#endif\n")
            else()
                set(layer_registry_arm82 "${layer_registry_arm82}#if NCNN_STRING\n{\"${class}\",0},\n#else\n{0},\n#endif\n")
            endif()
        endif()
    endif()

    # generate layer_type_enum file
    set(layer_type_enum "${layer_type_enum}${class} = ${__LAYER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_TYPE_ENUM_INDEX "${__LAYER_TYPE_ENUM_INDEX}+1")
endmacro()

