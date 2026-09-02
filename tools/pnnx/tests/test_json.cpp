#include <stdio.h>

#include <limits>
#include <string>
#include <vector>

#include "json.h"

static int test_failures = 0;

static void expect_true(bool value, const char* message)
{
    if (value)
        return;

    fprintf(stderr, "FAILED: %s\n", message);
    test_failures++;
}

static void test_value_types()
{
    const std::string document = "{\"null\":null,\"bool\":true,\"int\":-42,\"number\":1.25e2,\"string\":\"line\\n\\u4f60\\u597d\",\"array\":[1,false,{}]}";

    pnnx::JsonValue root;
    std::string error;
    expect_true(pnnx::parse_json(document, root, error), error.c_str());
    expect_true(root.type() == pnnx::JsonValue::Object, "root is an object");

    const pnnx::JsonValue* null_value = root.get("null");
    expect_true(null_value && null_value->is_null(), "null value");

    bool boolean_value = false;
    const pnnx::JsonValue* bool_value = root.get("bool");
    expect_true(bool_value && bool_value->get_bool(boolean_value) && boolean_value, "boolean value");

    int64_t integer_value = 0;
    const pnnx::JsonValue* int_value = root.get("int");
    expect_true(int_value && int_value->get_int(integer_value) && integer_value == -42, "integer value");

    double number_value = 0.0;
    const pnnx::JsonValue* number = root.get("number");
    expect_true(number && number->get_number(number_value) && number_value == 125.0, "number value");

    const pnnx::JsonValue* string_value = root.get("string");
    const std::string* string = string_value ? string_value->get_string() : 0;
    expect_true(string && *string == "line\n\xe4\xbd\xa0\xe5\xa5\xbd", "escaped string value");

    const pnnx::JsonValue* array_value = root.get("array");
    const std::vector<pnnx::JsonValue>* array = array_value ? array_value->get_array() : 0;
    expect_true(array && array->size() == 3, "array value");
    expect_true(root.get("missing") == 0, "missing object member");
}

static void test_integer_boundaries()
{
    pnnx::JsonValue value;
    std::string error;
    int64_t integer_value = 0;

    expect_true(pnnx::parse_json("9223372036854775807", value, error), "maximum int64 parses");
    expect_true(value.get_int(integer_value) && integer_value == std::numeric_limits<int64_t>::max(), "maximum int64 value");

    expect_true(pnnx::parse_json("-9223372036854775808", value, error), "minimum int64 parses");
    expect_true(value.get_int(integer_value) && integer_value == std::numeric_limits<int64_t>::min(), "minimum int64 value");

    double number_value = 0.0;
    expect_true(value.get_number(number_value), "integer can be read as number");
}

static void test_invalid_documents()
{
    const char* invalid_documents[] = {
        "",
        "true false",
        "[1,]",
        "{\"key\":1,\"key\":2}",
        "01",
        "\"\\uD800\""};

    for (size_t i = 0; i < sizeof(invalid_documents) / sizeof(invalid_documents[0]); i++)
    {
        pnnx::JsonValue value;
        std::string error;
        expect_true(!pnnx::parse_json(invalid_documents[i], value, error), "invalid document is rejected");
        expect_true(!error.empty(), "invalid document reports an error");
    }
}

int main()
{
    test_value_types();
    test_integer_boundaries();
    test_invalid_documents();

    if (test_failures != 0)
    {
        fprintf(stderr, "%d json test(s) failed\n", test_failures);
        return 1;
    }

    return 0;
}