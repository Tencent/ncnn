// Tencent 2026
// SPDX-License-Identifier: BSD-3-Clause

// json.hpp 边界单测（独立 harness，不进 CMake 构建）。
// 编译（在 tests/ 目录下）：
//     g++ -O2 -std=c++11 -Wall -Wextra -I../src test_json.cpp -o test_json
// 运行：./test_json   （全部通过退出码 0）

#include "json.hpp"

#include <stdio.h>
#include <string>

static int test_failed = 0;
static int test_count = 0;

#define CHECK(cond)                                                 \
    do                                                              \
    {                                                               \
        test_count++;                                               \
        if (!(cond))                                                \
        {                                                           \
            fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond); \
            test_failed++;                                          \
        }                                                           \
    } while (0)

#define CHECK_PARSE_ERROR(text)                                                          \
    do                                                                                   \
    {                                                                                    \
        test_count++;                                                                    \
        try                                                                              \
        {                                                                                \
            pnnx::parse_json(text);                                                      \
            fprintf(stderr, "FAIL line %d: expected parse error: %s\n", __LINE__, text); \
            test_failed++;                                                               \
        }                                                                                \
        catch (const std::exception&)                                                    \
        {                                                                                \
        }                                                                                \
    } while (0)

static void test_scalars()
{
    pnnx::JsonValue v = pnnx::parse_json("null");
    CHECK(v.isNull());

    v = pnnx::parse_json("true");
    CHECK(v.isBool() && v.asBool() == true);

    v = pnnx::parse_json("false");
    CHECK(v.isBool() && v.asBool() == false);

    v = pnnx::parse_json("42");
    CHECK(v.isInt() && v.asInt() == 42 && !v.isDouble());

    v = pnnx::parse_json("-7");
    CHECK(v.isInt() && v.asInt() == -7);

    v = pnnx::parse_json("0");
    CHECK(v.isInt() && v.asInt() == 0);

    v = pnnx::parse_json("3.5");
    CHECK(v.isDouble() && v.asDouble() == 3.5 && !v.isInt());

    v = pnnx::parse_json("-2.25");
    CHECK(v.isDouble() && v.asDouble() == -2.25);

    // 指数记法 → double；asDouble() 对 int 也成立
    v = pnnx::parse_json("1e-3");
    CHECK(v.isDouble() && v.asDouble() > 0.00099 && v.asDouble() < 0.00101);

    v = pnnx::parse_json("2.5E+4");
    CHECK(v.isDouble() && v.asDouble() == 25000.0);

    v = pnnx::parse_json("10");
    CHECK(v.isNumber() && v.asDouble() == 10.0);

    // 大整数走 int 通道精确保持（double 在 2^53 附近会失真）
    v = pnnx::parse_json("9007199254740993");
    CHECK(v.isInt() && v.asInt() == 9007199254740993LL);

    v = pnnx::parse_json("-9223372036854775807");
    CHECK(v.isInt() && v.asInt() == -9223372036854775807LL);

    // double 精度往返
    v = pnnx::parse_json("3.141592653589793");
    CHECK(v.isDouble() && v.asDouble() == 3.141592653589793);
}

static void test_strings()
{
    pnnx::JsonValue v = pnnx::parse_json("\"hello\"");
    CHECK(v.isString() && v.asString() == "hello");

    v = pnnx::parse_json("\"\"");
    CHECK(v.isString() && v.asString().empty());

    // 全部转义序列
    v = pnnx::parse_json("\"a\\\"b\\\\c\\/d\\be\\ff\\ng\\rh\\ti\"");
    CHECK(v.asString() == "a\"b\\c/d\be\ff\ng\rh\ti");

    // \u 基本平面：1~3 字节 UTF-8
    v = pnnx::parse_json("\"\\u0041\"");
    CHECK(v.asString() == "A");

    v = pnnx::parse_json("\"\\u00e9\"");
    CHECK(v.asString() == "\xC3\xA9");

    v = pnnx::parse_json("\"\\u4e2d\"");
    CHECK(v.asString() == "\xE4\xB8\xAD");

    // 代理对展开成 4 字节 UTF-8（U+1F600 = \uD83D\uDE00）
    v = pnnx::parse_json("\"\\ud83d\\ude00\"");
    CHECK(v.asString() == "\xF0\x9F\x98\x80");

    // 孤立高代理：至少不能产出非法 UTF-8 后静默错位——此处按字面 BMP 编码保留
    v = pnnx::parse_json("\"\\ud83d\"");
    CHECK(v.isString() && v.asString().size() == 3);

    // 非 ASCII 字节透传（torch 导出 json 的 ensure_ascii=false 场景）
    v = pnnx::parse_json("\"\xE4\xB8\xAD\xE6\x96\x87\"");
    CHECK(v.asString() == "\xE4\xB8\xAD\xE6\x96\x87");
}

static void test_containers()
{
    pnnx::JsonValue v = pnnx::parse_json("[]");
    CHECK(v.isArray() && v.size() == 0);

    v = pnnx::parse_json("{}");
    CHECK(v.isObject() && v.size() == 0);

    v = pnnx::parse_json("[1,\"two\",null,true,{\"k\":-1}]");
    CHECK(v.isArray() && v.size() == 5);
    CHECK(v[0].isInt() && v[0].asInt() == 1);
    CHECK(v[1].asString() == "two");
    CHECK(v[2].isNull());
    CHECK(v[3].isBool() && v[3].asBool() == true);
    CHECK(v[4].isObject() && v[4]["k"].asInt() == -1);

    // .pt2 真实形态：as_ints / as_tensors / sizes 列表元素是单键对象
    v = pnnx::parse_json("{\"as_ints\":[1,1],\"kind\":1}");
    CHECK(v["as_ints"].isArray() && v["as_ints"].size() == 2 && v["as_ints"][1].asInt() == 1);
    CHECK(v["kind"].asInt() == 1);

    v = pnnx::parse_json("{\"sizes\":[{\"as_int\":4},{\"as_int\":3}]}");
    CHECK(v["sizes"][1]["as_int"].asInt() == 3);

    v = pnnx::parse_json("{\"as_tensors\":[{\"name\":\"flatten\"},{\"name\":\"flatten_1\"}]}");
    CHECK(v["as_tensors"][0]["name"].asString() == "flatten");

    v = pnnx::parse_json("{\"as_none\":true}");
    CHECK(v["as_none"].isBool() && v["as_none"].asBool() == true);

    // 键顺序无关；重复键后者覆盖（std::map 语义）
    v = pnnx::parse_json("{\"a\":1,\"b\":2,\"a\":3}");
    CHECK(v["a"].asInt() == 3 && v["b"].asInt() == 2);

    // 深嵌套 500 层
    {
        std::string deep;
        for (int i = 0; i < 500; i++)
            deep += "[";
        deep += "1";
        for (int i = 0; i < 500; i++)
            deep += "]";
        v = pnnx::parse_json(deep);
        for (int i = 0; i < 500; i++)
            v = v[0];
        CHECK(v.isInt() && v.asInt() == 1);
    }
}

static void test_accessors()
{
    pnnx::JsonValue v = pnnx::parse_json("{\"a\":{\"b\":[10,20]}}");

    CHECK(v.hasMember("a"));
    CHECK(!v.hasMember("x"));
    CHECK(v["x"].isNull());
    CHECK(v["a"]["b"][1].asInt() == 20);
    // 越界数组下标 / 对非容器取下标 → null（容错访问，不 crash）
    CHECK(v["a"]["b"][5].isNull());
    CHECK(v["a"]["b"]["k"].isNull());
    CHECK(v["a"]["x"].isNull());
}

static void test_whitespace_and_errors()
{
    // 前后空白合法
    pnnx::JsonValue v = pnnx::parse_json("  \r\n [1, 2] \t\n ");
    CHECK(v.isArray() && v.size() == 2);

    // 非法输入必须报错
    CHECK_PARSE_ERROR("");
    CHECK_PARSE_ERROR("   ");
    CHECK_PARSE_ERROR("{");
    CHECK_PARSE_ERROR("[");
    CHECK_PARSE_ERROR("[1,2");
    CHECK_PARSE_ERROR("[1,]");
    CHECK_PARSE_ERROR("{\"a\":1,}");
    CHECK_PARSE_ERROR("{\"a\"}");
    CHECK_PARSE_ERROR("{\"a\" 1}");
    CHECK_PARSE_ERROR("{1:2}");
    CHECK_PARSE_ERROR("[1 2]");
    CHECK_PARSE_ERROR("tru");
    CHECK_PARSE_ERROR("nul");
    CHECK_PARSE_ERROR("hello");
    CHECK_PARSE_ERROR("\"abc");       // 未闭合字符串
    CHECK_PARSE_ERROR("\"ab\\xc\"");  // 非法转义
    CHECK_PARSE_ERROR("\"\\u12g4\""); // 非法 hex
    CHECK_PARSE_ERROR("[1-2]");       // 数字内垃圾字节必须终止解析
    CHECK_PARSE_ERROR("[1 2] 3");     // 顶层多个值
    CHECK_PARSE_ERROR("01");          // 前导零非 JSON（宽容/严格均可，此处按严格）
    CHECK_PARSE_ERROR("1.");
    CHECK_PARSE_ERROR(".5");
    CHECK_PARSE_ERROR("1e");
    CHECK_PARSE_ERROR("+1");
}

int main()
{
    test_scalars();
    test_strings();
    test_containers();
    test_accessors();
    test_whitespace_and_errors();

    fprintf(stderr, "test_json: %d checks, %d failed\n", test_count, test_failed);
    return test_failed == 0 ? 0 : 1;
}
