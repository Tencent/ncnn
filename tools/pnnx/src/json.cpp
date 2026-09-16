// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "json.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace pnnx {

static JsonValue g_null;

const JsonValue* JsonValue::find(const std::string& key) const
{
    if (type != T_OBJECT)
        return 0;
    std::map<std::string, JsonValue>::const_iterator it = o.find(key);
    if (it == o.end())
        return 0;
    return &it->second;
}

const JsonValue& JsonValue::get(const std::string& key) const
{
    const JsonValue* v = find(key);
    return v ? *v : g_null;
}

std::string JsonValue::as_string() const
{
    if (type == T_STRING)
        return s;
    if (type == T_NUMBER)
    {
        char buf[64];
        if (n == (double)(long long)n)
            sprintf(buf, "%lld", (long long)n);
        else
            sprintf(buf, "%g", n);
        return std::string(buf);
    }
    if (type == T_ARRAY)
    {
        // torch 2.13 may encode op targets as ["aten","relu","default"]
        std::string out;
        for (size_t i = 0; i < a.size(); i++)
        {
            if (i)
                out += ".";
            out += a[i].as_string();
        }
        return out;
    }
    return std::string();
}

bool JsonValue::as_bool() const
{
    if (type == T_BOOL)
        return b;
    if (type == T_NUMBER)
        return n != 0;
    return false;
}

int JsonValue::as_int() const
{
    if (type == T_NUMBER)
        return (int)n;
    if (type == T_BOOL)
        return b ? 1 : 0;
    if (type == T_STRING)
        return atoi(s.c_str());
    return 0;
}

double JsonValue::as_double() const
{
    if (type == T_NUMBER)
        return n;
    if (type == T_STRING)
        return atof(s.c_str());
    return 0;
}

struct Parser
{
    const char* p;
    const char* end;

    void skip_ws()
    {
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t'))
            p++;
    }

    bool parse_value(JsonValue& out)
    {
        skip_ws();
        if (p >= end)
            return false;

        if (*p == 'n')
            return parse_null(out);
        if (*p == 't' || *p == 'f')
            return parse_bool(out);
        if (*p == '"')
            return parse_string(out);
        if (*p == '[')
            return parse_array(out);
        if (*p == '{')
            return parse_object(out);
        if (*p == '-' || (*p >= '0' && *p <= '9'))
            return parse_number(out);
        return false;
    }

    bool parse_null(JsonValue& out)
    {
        if (end - p < 4 || strncmp(p, "null", 4) != 0)
            return false;
        p += 4;
        out = JsonValue();
        return true;
    }

    bool parse_bool(JsonValue& out)
    {
        out = JsonValue();
        out.type = JsonValue::T_BOOL;
        if (end - p >= 4 && strncmp(p, "true", 4) == 0)
        {
            p += 4;
            out.b = true;
            return true;
        }
        if (end - p >= 5 && strncmp(p, "false", 5) == 0)
        {
            p += 5;
            out.b = false;
            return true;
        }
        return false;
    }

    bool parse_number(JsonValue& out)
    {
        const char* start = p;
        if (*p == '-')
            p++;
        if (p >= end || !isdigit((unsigned char)*p))
            return false;
        while (p < end && isdigit((unsigned char)*p))
            p++;
        if (p < end && *p == '.')
        {
            p++;
            while (p < end && isdigit((unsigned char)*p))
                p++;
        }
        if (p < end && (*p == 'e' || *p == 'E'))
        {
            p++;
            if (p < end && (*p == '+' || *p == '-'))
                p++;
            while (p < end && isdigit((unsigned char)*p))
                p++;
        }
        out = JsonValue();
        out.type = JsonValue::T_NUMBER;
        out.n = strtod(start, 0);
        return true;
    }

    bool parse_string(JsonValue& out)
    {
        if (*p != '"')
            return false;
        p++;
        std::string s;
        while (p < end && *p != '"')
        {
            if (*p == '\\')
            {
                p++;
                if (p >= end)
                    return false;
                char c = *p++;
                if (c == '"' || c == '\\' || c == '/')
                    s += c;
                else if (c == 'b')
                    s += '\b';
                else if (c == 'f')
                    s += '\f';
                else if (c == 'n')
                    s += '\n';
                else if (c == 'r')
                    s += '\r';
                else if (c == 't')
                    s += '\t';
                else if (c == 'u')
                {
                    if (end - p < 4)
                        return false;
                    unsigned int cp = 0;
                    for (int i = 0; i < 4; i++)
                    {
                        char h = *p++;
                        cp <<= 4;
                        if (h >= '0' && h <= '9')
                            cp += (unsigned int)(h - '0');
                        else if (h >= 'a' && h <= 'f')
                            cp += (unsigned int)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F')
                            cp += (unsigned int)(h - 'A' + 10);
                        else
                            return false;
                    }
                    if (cp < 0x80)
                    {
                        s += (char)cp;
                    }
                    else if (cp < 0x800)
                    {
                        s += (char)(0xc0 | (cp >> 6));
                        s += (char)(0x80 | (cp & 0x3f));
                    }
                    else
                    {
                        s += (char)(0xe0 | (cp >> 12));
                        s += (char)(0x80 | ((cp >> 6) & 0x3f));
                        s += (char)(0x80 | (cp & 0x3f));
                    }
                }
                else
                    return false;
            }
            else
            {
                s += *p++;
            }
        }
        if (p >= end || *p != '"')
            return false;
        p++;
        out = JsonValue();
        out.type = JsonValue::T_STRING;
        out.s = s;
        return true;
    }

    bool parse_array(JsonValue& out)
    {
        if (*p != '[')
            return false;
        p++;
        out = JsonValue();
        out.type = JsonValue::T_ARRAY;
        skip_ws();
        if (p < end && *p == ']')
        {
            p++;
            return true;
        }
        for (;;)
        {
            JsonValue item;
            if (!parse_value(item))
                return false;
            out.a.push_back(item);
            skip_ws();
            if (p < end && *p == ',')
            {
                p++;
                continue;
            }
            if (p < end && *p == ']')
            {
                p++;
                return true;
            }
            return false;
        }
    }

    bool parse_object(JsonValue& out)
    {
        if (*p != '{')
            return false;
        p++;
        out = JsonValue();
        out.type = JsonValue::T_OBJECT;
        skip_ws();
        if (p < end && *p == '}')
        {
            p++;
            return true;
        }
        for (;;)
        {
            skip_ws();
            JsonValue key;
            if (!parse_string(key))
                return false;
            skip_ws();
            if (p >= end || *p != ':')
                return false;
            p++;
            JsonValue val;
            if (!parse_value(val))
                return false;
            out.o[key.s] = val;
            skip_ws();
            if (p < end && *p == ',')
            {
                p++;
                continue;
            }
            if (p < end && *p == '}')
            {
                p++;
                return true;
            }
            return false;
        }
    }
};

int parse_json(const char* data, size_t size, JsonValue& out)
{
    Parser ps;
    ps.p = data;
    ps.end = data + size;
    if (!ps.parse_value(out))
    {
        fprintf(stderr, "parse_json failed\n");
        return -1;
    }
    ps.skip_ws();
    if (ps.p != ps.end)
    {
        // allow trailing whitespace only
        return 0;
    }
    return 0;
}

} // namespace pnnx
