// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PT2_JSON_H
#define PNNX_PT2_JSON_H

#include <stddef.h>

#include <map>
#include <string>
#include <vector>

namespace pnnx {

struct Pt2JsonValue
{
    enum Type
    {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object
    };

    Pt2JsonValue();

    Type type;
    bool boolean;
    size_t offset;
    std::string value;
    std::vector<Pt2JsonValue> array;
    std::map<std::string, Pt2JsonValue> object;
};

int parse_pt2_json(const unsigned char* data, size_t size, Pt2JsonValue& value, std::string& error);

} // namespace pnnx

#endif // PNNX_PT2_JSON_H
