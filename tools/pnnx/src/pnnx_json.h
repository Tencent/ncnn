// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_JSON_H
#define PNNX_JSON_H

#include <stdint.h>
#include <map>
#include <string>
#include <vector>

namespace pnnx {

class Parser;

// a minimal zero-dependency JSON DOM parser for parsing the torch
// exported program serialized graph (models/model.json) and the
// payload config files (weights/constants config).
class JsonValue
{
public:
    enum Type
    {
        NULL_TYPE = 0,
        BOOL_TYPE = 1,
        INT_TYPE = 2,
        DOUBLE_TYPE = 3,
        STRING_TYPE = 4,
        ARRAY_TYPE = 5,
        OBJECT_TYPE = 6
    };

    JsonValue();

    Type type() const
    {
        return t;
    }

    bool is_null() const
    {
        return t == NULL_TYPE;
    }
    bool is_bool() const
    {
        return t == BOOL_TYPE;
    }
    bool is_int() const
    {
        return t == INT_TYPE;
    }
    bool is_double() const
    {
        return t == DOUBLE_TYPE;
    }
    bool is_number() const
    {
        return t == INT_TYPE || t == DOUBLE_TYPE;
    }
    bool is_string() const
    {
        return t == STRING_TYPE;
    }
    bool is_array() const
    {
        return t == ARRAY_TYPE;
    }
    bool is_object() const
    {
        return t == OBJECT_TYPE;
    }

    bool as_bool() const
    {
        return b;
    }
    int64_t as_int() const;
    double as_double() const;
    const std::string& as_string() const;
    const std::vector<JsonValue>& as_array() const;
    const std::map<std::string, JsonValue>& as_object() const;

    // object helpers : return the static null value when the key is absent
    bool has(const std::string& key) const;
    const JsonValue& get(const std::string& key) const;
    const JsonValue& operator[](const std::string& key) const;

    // array helpers
    size_t size() const;
    const JsonValue& operator[](size_t i) const;

private:
    Type t;
    bool b;
    int64_t i;
    double f;
    std::string s;
    std::vector<JsonValue> a;
    std::map<std::string, JsonValue> o;

    friend class Parser;
    friend class JsonParser;
};

class JsonParser
{
public:
    // parse the whole buffer, returns false on syntax error
    static bool parse(const char* text, size_t len, JsonValue& out);
    static bool parse(const std::string& text, JsonValue& out);
};

} // namespace pnnx

#endif // PNNX_JSON_H
