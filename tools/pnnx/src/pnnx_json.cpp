// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pnnx_json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace pnnx {

static const JsonValue& null_value()
{
    static const JsonValue v;
    return v;
}

JsonValue::JsonValue()
    : t(NULL_TYPE), b(false), i(0), f(0.f)
{
}

int64_t JsonValue::as_int() const
{
    if (t == INT_TYPE)
        return i;
    if (t == DOUBLE_TYPE)
        return (int64_t)f;
    if (t == BOOL_TYPE)
        return b ? 1 : 0;
    return 0;
}

double JsonValue::as_double() const
{
    if (t == DOUBLE_TYPE)
        return f;
    if (t == INT_TYPE)
        return (double)i;
    return 0.f;
}

const std::string& JsonValue::as_string() const
{
    if (t == STRING_TYPE)
        return s;

    static const std::string empty;
    return empty;
}

const std::vector<JsonValue>& JsonValue::as_array() const
{
    if (t == ARRAY_TYPE)
        return a;

    static const std::vector<JsonValue> empty;
    return empty;
}

const std::map<std::string, JsonValue>& JsonValue::as_object() const
{
    if (t == OBJECT_TYPE)
        return o;

    static const std::map<std::string, JsonValue> empty;
    return empty;
}

bool JsonValue::has(const std::string& key) const
{
    if (t != OBJECT_TYPE)
        return false;

    return o.find(key) != o.end();
}

const JsonValue& JsonValue::get(const std::string& key) const
{
    if (t != OBJECT_TYPE)
        return null_value();

    std::map<std::string, JsonValue>::const_iterator it = o.find(key);
    if (it == o.end())
        return null_value();

    return it->second;
}

const JsonValue& JsonValue::operator[](const std::string& key) const
{
    return get(key);
}

size_t JsonValue::size() const
{
    if (t == ARRAY_TYPE)
        return a.size();
    if (t == OBJECT_TYPE)
        return o.size();
    return 0;
}

const JsonValue& JsonValue::operator[](size_t i) const
{
    if (t != ARRAY_TYPE)
        return null_value();

    if (i >= a.size())
        return null_value();

    return a[i];
}

class Parser
{
public:
    Parser(const char* text, size_t len)
        : cur(text), end(text + len), ok(true)
    {
    }

    bool parse(JsonValue& out)
    {
        skip_ws();

        if (!parse_value(out))
            return false;

        skip_ws();

        return cur == end;
    }

    bool failed() const
    {
        return !ok;
    }

private:
    const char* cur;
    const char* end;
    bool ok;

    void skip_ws()
    {
        while (cur < end)
        {
            char c = *cur;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                cur++;
            else
                break;
        }
    }

    bool parse_value(JsonValue& out)
    {
        if (cur >= end)
            return false;

        char c = *cur;

        if (c == '{')
            return parse_object(out);
        if (c == '[')
            return parse_array(out);
        if (c == '"')
            return parse_string(out);
        if (c == 't')
            return parse_literal("true", JsonValue::BOOL_TYPE, out);
        if (c == 'f')
            return parse_literal("false", JsonValue::BOOL_TYPE, out);
        if (c == 'n')
            return parse_literal("null", JsonValue::NULL_TYPE, out);
        if (c == '-' || (c >= '0' && c <= '9'))
            return parse_number(out);

        return false;
    }

    bool parse_literal(const char* lit, JsonValue::Type type, JsonValue& out)
    {
        size_t len = strlen(lit);
        if ((size_t)(end - cur) < len || strncmp(cur, lit, len) != 0)
            return false;

        cur += len;

        if (type == JsonValue::BOOL_TYPE)
        {
            out.t = JsonValue::BOOL_TYPE;
            out.b = (lit[0] == 't');
        }
        else
        {
            out.t = JsonValue::NULL_TYPE;
        }

        return true;
    }

    bool parse_number(JsonValue& out)
    {
        const char* p = cur;
        if (p < end && *p == '-')
            p++;

        bool is_double = false;
        while (p < end && *p >= '0' && *p <= '9')
            p++;

        if (p < end && *p == '.')
        {
            is_double = true;
            p++;
            while (p < end && *p >= '0' && *p <= '9')
                p++;
        }

        if (p < end && (*p == 'e' || *p == 'E'))
        {
            is_double = true;
            p++;
            if (p < end && (*p == '+' || *p == '-'))
                p++;
            while (p < end && *p >= '0' && *p <= '9')
                p++;
        }

        size_t len = p - cur;
        std::string num(cur, len);

        if (is_double)
        {
            out.t = JsonValue::DOUBLE_TYPE;
            out.f = strtod(num.c_str(), 0);
        }
        else
        {
            out.t = JsonValue::INT_TYPE;
            out.i = strtoll(num.c_str(), 0, 10);
        }

        cur = p;
        return true;
    }

    bool parse_string(JsonValue& out)
    {
        // cur points at the opening quote
        cur++;

        out.t = JsonValue::STRING_TYPE;
        out.s.clear();

        while (cur < end)
        {
            char c = *cur++;

            if (c == '"')
                return true;

            if (c == '\\')
            {
                if (cur >= end)
                    return false;

                char e = *cur++;

                if (e == '"')
                    out.s.push_back('"');
                else if (e == '\\')
                    out.s.push_back('\\');
                else if (e == '/')
                    out.s.push_back('/');
                else if (e == 'b')
                    out.s.push_back('\b');
                else if (e == 'f')
                    out.s.push_back('\f');
                else if (e == 'n')
                    out.s.push_back('\n');
                else if (e == 'r')
                    out.s.push_back('\r');
                else if (e == 't')
                    out.s.push_back('\t');
                else if (e == 'u')
                {
                    // \uXXXX (basic multilingual plane only)
                    if ((size_t)(end - cur) < 4)
                        return false;

                    unsigned int cp = 0;
                    for (int i = 0; i < 4; i++)
                    {
                        char h = *cur++;
                        cp <<= 4;
                        if (h >= '0' && h <= '9')
                            cp |= h - '0';
                        else if (h >= 'a' && h <= 'f')
                            cp |= h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F')
                            cp |= h - 'A' + 10;
                        else
                            return false;
                    }

                    // encode as utf-8
                    if (cp < 0x80)
                    {
                        out.s.push_back((char)cp);
                    }
                    else if (cp < 0x800)
                    {
                        out.s.push_back((char)(0xc0 | (cp >> 6)));
                        out.s.push_back((char)(0x80 | (cp & 0x3f)));
                    }
                    else
                    {
                        out.s.push_back((char)(0xe0 | (cp >> 12)));
                        out.s.push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
                        out.s.push_back((char)(0x80 | (cp & 0x3f)));
                    }
                }
                else
                {
                    return false;
                }
            }
            else
            {
                out.s.push_back(c);
            }
        }

        return false;
    }

    bool parse_array(JsonValue& out)
    {
        // cur points at '['
        cur++;

        out.t = JsonValue::ARRAY_TYPE;
        out.a.clear();

        skip_ws();

        if (cur < end && *cur == ']')
        {
            cur++;
            return true;
        }

        for (;;)
        {
            JsonValue v;
            if (!parse_value(v))
                return false;

            out.a.push_back(v);

            skip_ws();

            if (cur >= end)
                return false;

            char c = *cur++;
            if (c == ',')
            {
                skip_ws();
                continue;
            }
            if (c == ']')
                return true;

            return false;
        }
    }

    bool parse_object(JsonValue& out)
    {
        // cur points at '{'
        cur++;

        out.t = JsonValue::OBJECT_TYPE;
        out.o.clear();

        skip_ws();

        if (cur < end && *cur == '}')
        {
            cur++;
            return true;
        }

        for (;;)
        {
            skip_ws();

            if (cur >= end || *cur != '"')
                return false;

            JsonValue key;
            if (!parse_string(key))
                return false;

            skip_ws();

            if (cur >= end || *cur != ':')
                return false;
            cur++;

            skip_ws();

            JsonValue v;
            if (!parse_value(v))
                return false;

            out.o[key.s] = v;

            skip_ws();

            if (cur >= end)
                return false;

            char c = *cur++;
            if (c == ',')
                continue;
            if (c == '}')
                return true;

            return false;
        }
    }
};

bool JsonParser::parse(const char* text, size_t len, JsonValue& out)
{
    Parser p(text, len);
    return p.parse(out);
}

bool JsonParser::parse(const std::string& text, JsonValue& out)
{
    return parse(text.c_str(), text.size(), out);
}

} // namespace pnnx
