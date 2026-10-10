// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_JSON_READER_H
#define PNNX_JSON_READER_H

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <string>
#include <vector>

namespace pnnx {

enum JsonType
{
    JSON_NULL,
    JSON_BOOL,
    JSON_INT64,
    JSON_UINT64,
    JSON_DOUBLE,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
};

struct JsonParseOptions
{
    JsonParseOptions();

    size_t max_depth;
    size_t max_nodes;
    size_t max_string_length;
};

struct JsonParseError
{
    size_t byte_offset;
    size_t line;
    size_t column;
    std::string message;
};

class JsonValue
{
public:
    JsonValue();

    JsonType type() const;

    bool as_bool() const;
    int64_t as_int64() const;
    uint64_t as_uint64() const;
    double as_double() const;
    const std::string& as_string() const;
    const std::vector<JsonValue>& as_array() const;
    const std::map<std::string, JsonValue>& as_object() const;

    const JsonValue* find(const std::string& key) const;

private:
    JsonType type_;
    bool bool_value_;
    int64_t int64_value_;
    uint64_t uint64_value_;
    double double_value_;
    std::string string_value_;
    std::vector<JsonValue> array_value_;
    std::map<std::string, JsonValue> object_value_;

    friend class JsonParser;
};

int parse_json(const char* data, size_t size, JsonValue& value, JsonParseError& error, const JsonParseOptions& options);

} // namespace pnnx

#endif // PNNX_JSON_READER_H
