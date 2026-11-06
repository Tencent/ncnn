// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_JSON_H
#define PNNX_JSON_H

#include <stddef.h>
#include <stdint.h>
#include <map>
#include <string>
#include <vector>

namespace pnnx {

class JsonValue
{
public:
    enum Type
    {
        Null,
        Bool,
        Int,
        Float,
        String,
        Array,
        Object
    };

    JsonValue()
        : type(Null), b(false), i(0), f(0)
    {
    }
    explicit JsonValue(bool v)
        : type(Bool), b(v), i(0), f(0)
    {
    }
    explicit JsonValue(int64_t v)
        : type(Int), b(false), i(v), f(0)
    {
    }
    explicit JsonValue(double v)
        : type(Float), b(false), i(0), f(v)
    {
    }
    explicit JsonValue(const std::string& v)
        : type(String), b(false), i(0), f(0), s(v)
    {
    }

    const JsonValue* get(const std::string& key) const
    {
        if (type != Object) return 0;
        std::map<std::string, JsonValue>::const_iterator it = o.find(key);
        return it == o.end() ? 0 : &it->second;
    }

    Type type;
    bool b;
    int64_t i;
    double f;
    std::string s;
    std::vector<JsonValue> a;
    std::map<std::string, JsonValue> o;
};

int parse_json(const char* data, size_t size, JsonValue& value, std::string& error);

} // namespace pnnx

#endif // PNNX_JSON_H
