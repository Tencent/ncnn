# must define SHADER_COMP_HEADER SHADER_SRC

file(READ ${SHADER_SRC} comp_data)

if(NCNN_SHADER_COMPRESS)
    set(WORDLIST_FILE "${CMAKE_SOURCE_DIR}/../../src/layer/vulkan/shader/shader_compress_dict.in")
    if(EXISTS ${WORDLIST_FILE})
        file(STRINGS ${WORDLIST_FILE} RAW_LIST)

        # 处理转义序列并创建新列表
        set(PROCESSED_LIST)
        foreach(line IN LISTS RAW_LIST)
            string(REPLACE "\\n" "\n" processed_line "${line}")
            list(APPEND PROCESSED_LIST "${processed_line}")
        endforeach()

        # 按长度降序排序 (优先匹配长字符串)
        list(SORT PROCESSED_LIST COMPARE STRING ORDER DESCENDING)
        set(WORD_LIST ${PROCESSED_LIST})
    else()
        message(FATAL_ERROR "shader_compress_dict is missing: ${WORDLIST_FILE}")
    endif()
endif()

# skip leading comment
string(FIND "${comp_data}" "#version" version_start)
if(NOT ${version_start} EQUAL -1)
    string(SUBSTRING "${comp_data}" ${version_start} -1 comp_data)
endif()

if(NCNN_SHADER_COMPRESS)
    # use
    list(LENGTH WORD_LIST word_count)
    if(word_count GREATER 128)
        message(WARNING "词表过大(超过128词)，将截断")
        math(EXPR word_count "128")
    endif()

    set(idx 0)
    foreach(word IN LISTS WORD_LIST)
        if(idx LESS 128)  # 确保不超过128个词
            # 生成替换字符(128 + idx)
            math(EXPR char_code "128 + ${idx}")
            string(ASCII ${char_code} replace_char)

            # 执行全局替换
            string(REPLACE "${word}" "${replace_char}" comp_data "${comp_data}")

            # 索引递增
            math(EXPR idx "${idx} + 1")
        endif()
    endforeach()
endif ()

# remove whitespace
string(REGEX REPLACE "\n +" "\n" comp_data "${comp_data}")

# remove empty line
string(REGEX REPLACE "\n\n" "\n" comp_data "${comp_data}")

get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)

# text to hex
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/${SHADER_SRC_NAME_WE}.text2hex.txt "${comp_data}")
file(READ ${CMAKE_CURRENT_BINARY_DIR}/layer/vulkan/shader/${SHADER_SRC_NAME_WE}.text2hex.txt comp_data_hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," comp_data_hex ${comp_data_hex})
string(FIND "${comp_data_hex}" "," tail_comma REVERSE)
string(SUBSTRING "${comp_data_hex}" 0 ${tail_comma} comp_data_hex)

if(NCNN_SHADER_COMPRESS)
    file(WRITE ${SHADER_COMP_HEADER} "static const unsigned char ${SHADER_SRC_NAME_WE}_comp_data[] = {${comp_data_hex}};\n")
else ()
    file(WRITE ${SHADER_COMP_HEADER} "static const char ${SHADER_SRC_NAME_WE}_comp_data[] = {${comp_data_hex}};\n")
endif ()