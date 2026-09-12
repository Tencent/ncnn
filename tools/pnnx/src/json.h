// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_JSON_H
#define PNNX_JSON_H

#include <map>
#include <string>
#include <vector>

namespace pnnx {

class JsonValue
{
public:
    enum Type
    {
        T_NULL = 0,
        T_BOOL,
        T_NUMBER,
        T_STRING,
        T_ARRAY,
        T_OBJECT
    };

    Type type;
    bool b;
    double n;
    std::string s;
    std::vector<JsonValue> a;
    std::map<std::string, JsonValue> o;

    JsonValue()
        : type(T_NULL), b(false), n(0.0)
    {
    }

    bool is_null() const { return type == T_NULL; }
    bool is_bool() const { return type == T_BOOL; }
    bool is_number() const { return type == T_NUMBER; }
    bool is_string() const { return type == T_STRING; }
    bool is_array() const { return type == T_ARRAY; }
    bool is_object() const { return type == T_OBJECT; }

    const JsonValue* find(const std::string& key) const;
    const JsonValue& get(const std::string& key) const;

    std::string as_string() const;
    bool as_bool() const;
    int as_int() const;
    double as_double() const;
};

int parse_json(const char* data, size_t size, JsonValue& out);

} // namespace pnnx

#endif // PNNX_JSON_H
