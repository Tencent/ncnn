// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pt2_json.h"

#include <stdio.h>
#include <string.h>

#include <utility>

namespace pnnx {

Pt2JsonValue::Pt2JsonValue()
{
    type = Null;
    boolean = false;
    offset = 0;
}

class Pt2JsonParser
{
public:
    Pt2JsonParser(const unsigned char* _data, size_t _size)
        : data(_data), size(_size), pos(0)
    {
    }

    int parse(Pt2JsonValue& value, std::string& _error)
    {
        if (!data)
        {
            _error = "json offset 0: invalid input";
            return -1;
        }

        skip_space();
        if (!parse_value(value))
        {
            _error = error;
            return -1;
        }
        skip_space();
        if (pos != size)
        {
            fail("trailing data");
            _error = error;
            return -1;
        }

        return 0;
    }

private:
    bool fail(const char* message)
    {
        if (error.empty())
        {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), "json offset %lu: %s", (unsigned long)pos, message);
            error = buffer;
        }
        return false;
    }

    void skip_space()
    {
        while (pos < size && (data[pos] == ' ' || data[pos] == '\t' || data[pos] == '\r' || data[pos] == '\n'))
            pos++;
    }

    bool parse_value(Pt2JsonValue& value)
    {
        if (pos == size)
            return fail("unexpected end of input");

        value.offset = pos;
        if (data[pos] == 'n')
            return parse_literal("null", Pt2JsonValue::Null, value);
        if (data[pos] == 't')
        {
            value.boolean = true;
            return parse_literal("true", Pt2JsonValue::Bool, value);
        }
        if (data[pos] == 'f')
        {
            value.boolean = false;
            return parse_literal("false", Pt2JsonValue::Bool, value);
        }
        if (data[pos] == '"')
        {
            value.type = Pt2JsonValue::String;
            return parse_string(value.value);
        }
        if (data[pos] == '[')
            return parse_array(value);
        if (data[pos] == '{')
            return parse_object(value);
        if (data[pos] == '-' || (data[pos] >= '0' && data[pos] <= '9'))
            return parse_number(value);

        return fail("unexpected character");
    }

    bool parse_literal(const char* literal, Pt2JsonValue::Type type, Pt2JsonValue& value)
    {
        const size_t len = strlen(literal);
        if (len > size - pos || memcmp(data + pos, literal, len) != 0)
            return fail("invalid literal");
        pos += len;
        value.type = type;
        return true;
    }

    static void append_utf8(uint32_t codepoint, std::string& value)
    {
        if (codepoint <= 0x7f)
            value.push_back((char)codepoint);
        else if (codepoint <= 0x7ff)
        {
            value.push_back((char)(0xc0 | (codepoint >> 6)));
            value.push_back((char)(0x80 | (codepoint & 0x3f)));
        }
        else if (codepoint <= 0xffff)
        {
            value.push_back((char)(0xe0 | (codepoint >> 12)));
            value.push_back((char)(0x80 | ((codepoint >> 6) & 0x3f)));
            value.push_back((char)(0x80 | (codepoint & 0x3f)));
        }
        else
        {
            value.push_back((char)(0xf0 | (codepoint >> 18)));
            value.push_back((char)(0x80 | ((codepoint >> 12) & 0x3f)));
            value.push_back((char)(0x80 | ((codepoint >> 6) & 0x3f)));
            value.push_back((char)(0x80 | (codepoint & 0x3f)));
        }
    }

    bool parse_utf8(unsigned char first, std::string& value)
    {
        const size_t begin = pos - 1;
        int trailing;
        uint32_t codepoint;
        uint32_t minimum;
        if (first >= 0xc2 && first <= 0xdf)
        {
            trailing = 1;
            codepoint = first & 0x1f;
            minimum = 0x80;
        }
        else if (first >= 0xe0 && first <= 0xef)
        {
            trailing = 2;
            codepoint = first & 0x0f;
            minimum = 0x800;
        }
        else if (first >= 0xf0 && first <= 0xf4)
        {
            trailing = 3;
            codepoint = first & 0x07;
            minimum = 0x10000;
        }
        else
        {
            return fail("invalid utf-8");
        }

        if (size - pos < (size_t)trailing)
            return fail("incomplete utf-8");
        for (int i = 0; i < trailing; i++)
        {
            const unsigned char ch = data[pos++];
            if ((ch & 0xc0) != 0x80)
                return fail("invalid utf-8");
            codepoint = (codepoint << 6) | (ch & 0x3f);
        }
        if (codepoint < minimum || codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff))
            return fail("invalid utf-8");
        value.append((const char*)data + begin, pos - begin);
        return true;
    }

    bool parse_hex4(uint32_t& codepoint)
    {
        if (size - pos < 4)
            return fail("incomplete unicode escape");

        codepoint = 0;
        for (int i = 0; i < 4; i++)
        {
            const unsigned char ch = data[pos++];
            codepoint <<= 4;
            if (ch >= '0' && ch <= '9')
                codepoint |= ch - '0';
            else if (ch >= 'a' && ch <= 'f')
                codepoint |= ch - 'a' + 10;
            else if (ch >= 'A' && ch <= 'F')
                codepoint |= ch - 'A' + 10;
            else
                return fail("invalid unicode escape");
        }
        return true;
    }

    bool parse_string(std::string& value)
    {
        value.clear();
        pos++;

        while (pos < size)
        {
            const unsigned char ch = data[pos++];
            if (ch == '"')
                return true;
            if (ch < 0x20)
                return fail("unescaped control character");
            if (ch != '\\')
            {
                if (ch < 0x80)
                    value.push_back((char)ch);
                else if (!parse_utf8(ch, value))
                    return false;
            }
            else
            {
                if (pos == size)
                    return fail("incomplete escape");
                const unsigned char escaped = data[pos++];
                if (escaped == '"' || escaped == '\\' || escaped == '/')
                    value.push_back((char)escaped);
                else if (escaped == 'b')
                    value.push_back('\b');
                else if (escaped == 'f')
                    value.push_back('\f');
                else if (escaped == 'n')
                    value.push_back('\n');
                else if (escaped == 'r')
                    value.push_back('\r');
                else if (escaped == 't')
                    value.push_back('\t');
                else if (escaped == 'u')
                {
                    uint32_t codepoint;
                    if (!parse_hex4(codepoint))
                        return false;
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff)
                    {
                        if (size - pos < 6 || data[pos] != '\\' || data[pos + 1] != 'u')
                            return fail("missing low surrogate");
                        pos += 2;
                        uint32_t low;
                        if (!parse_hex4(low) || low < 0xdc00 || low > 0xdfff)
                            return fail("invalid low surrogate");
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + low - 0xdc00;
                    }
                    else if (codepoint >= 0xdc00 && codepoint <= 0xdfff)
                    {
                        return fail("unexpected low surrogate");
                    }
                    append_utf8(codepoint, value);
                }
                else
                {
                    return fail("invalid escape");
                }
            }
        }

        return fail("unterminated string");
    }

    bool parse_number(Pt2JsonValue& value)
    {
        const size_t begin = pos;
        if (data[pos] == '-')
        {
            pos++;
            if (pos == size)
                return fail("incomplete number");
        }

        if (data[pos] == '0')
            pos++;
        else
        {
            if (data[pos] < '1' || data[pos] > '9')
                return fail("invalid number");
            while (pos < size && data[pos] >= '0' && data[pos] <= '9')
                pos++;
        }

        if (pos < size && data[pos] == '.')
        {
            pos++;
            if (pos == size || data[pos] < '0' || data[pos] > '9')
                return fail("invalid fraction");
            while (pos < size && data[pos] >= '0' && data[pos] <= '9')
                pos++;
        }

        if (pos < size && (data[pos] == 'e' || data[pos] == 'E'))
        {
            pos++;
            if (pos < size && (data[pos] == '+' || data[pos] == '-'))
                pos++;
            if (pos == size || data[pos] < '0' || data[pos] > '9')
                return fail("invalid exponent");
            while (pos < size && data[pos] >= '0' && data[pos] <= '9')
                pos++;
        }

        value.type = Pt2JsonValue::Number;
        value.value.assign((const char*)data + begin, pos - begin);
        return true;
    }

    bool parse_array(Pt2JsonValue& value)
    {
        value.type = Pt2JsonValue::Array;
        pos++;
        skip_space();
        if (pos < size && data[pos] == ']')
        {
            pos++;
            return true;
        }

        while (true)
        {
            value.array.push_back(Pt2JsonValue());
            if (!parse_value(value.array.back()))
                return false;
            skip_space();
            if (pos == size)
                return fail("unterminated array");
            if (data[pos] == ']')
            {
                pos++;
                return true;
            }
            if (data[pos++] != ',')
                return fail("expected ',' or ']'");
            skip_space();
        }
    }

    bool parse_object(Pt2JsonValue& value)
    {
        value.type = Pt2JsonValue::Object;
        pos++;
        skip_space();
        if (pos < size && data[pos] == '}')
        {
            pos++;
            return true;
        }

        while (true)
        {
            if (pos == size || data[pos] != '"')
                return fail("expected object key");
            std::string key;
            if (!parse_string(key))
                return false;
            skip_space();
            if (pos == size || data[pos++] != ':')
                return fail("expected ':'");
            skip_space();

            Pt2JsonValue child;
            if (!parse_value(child))
                return false;
            if (!value.object.insert(std::make_pair(key, std::move(child))).second)
                return fail("duplicate object key");

            skip_space();
            if (pos == size)
                return fail("unterminated object");
            if (data[pos] == '}')
            {
                pos++;
                return true;
            }
            if (data[pos++] != ',')
                return fail("expected ',' or '}'");
            skip_space();
        }
    }

    const unsigned char* data;
    size_t size;
    size_t pos;
    std::string error;
};

int parse_pt2_json(const unsigned char* data, size_t size, Pt2JsonValue& value, std::string& error)
{
    value = Pt2JsonValue();
    error.clear();
    Pt2JsonParser parser(data, size);
    return parser.parse(value, error);
}

} // namespace pnnx
