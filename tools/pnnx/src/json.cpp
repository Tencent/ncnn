// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "json.h"

#include <cmath>
#include <limits>
#include <locale>
#include <sstream>
#include <utility>

namespace pnnx {

class JsonParser
{
public:
    JsonParser(const char* data, size_t size) : data(data), size(size), pos(0)
    {
    }

    bool fail(const char* message)
    {
        if (error.empty())
            error = "byte " + std::to_string(pos) + ": " + message;
        return false;
    }

    void whitespace()
    {
        while (pos < size && (data[pos] == ' ' || data[pos] == '\t' || data[pos] == '\r' || data[pos] == '\n'))
            pos++;
    }

    bool literal(const char* text)
    {
        for (size_t i = 0; text[i]; i++)
            if (pos == size || data[pos++] != text[i])
                return fail("invalid literal");
        return true;
    }

    bool hex4(unsigned int& value)
    {
        value = 0;
        for (int i = 0; i < 4; i++)
        {
            if (pos == size)
                return fail("truncated Unicode escape");
            unsigned char c = data[pos++];
            unsigned int n;
            if (c >= '0' && c <= '9')
                n = c - '0';
            else if (c >= 'a' && c <= 'f')
                n = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F')
                n = c - 'A' + 10;
            else
                return fail("invalid Unicode escape");
            value = (value << 4) | n;
        }
        return true;
    }

    void utf8(unsigned int value, std::string& text)
    {
        if (value <= 0x7f)
            text.push_back((char)value);
        else if (value <= 0x7ff)
        {
            text.push_back((char)(0xc0 | (value >> 6)));
            text.push_back((char)(0x80 | (value & 63)));
        }
        else if (value <= 0xffff)
        {
            text.push_back((char)(0xe0 | (value >> 12)));
            text.push_back((char)(0x80 | ((value >> 6) & 63)));
            text.push_back((char)(0x80 | (value & 63)));
        }
        else
        {
            text.push_back((char)(0xf0 | (value >> 18)));
            text.push_back((char)(0x80 | ((value >> 12) & 63)));
            text.push_back((char)(0x80 | ((value >> 6) & 63)));
            text.push_back((char)(0x80 | (value & 63)));
        }
    }

    bool string(std::string& text)
    {
        if (pos == size || data[pos++] != '"')
            return fail("expected string");
        while (pos < size)
        {
            unsigned char c = data[pos++];
            if (c == '"')
                return true;
            if (c < 32)
                return fail("unescaped control character");
            if (c == '\\')
            {
                if (pos == size)
                    return fail("truncated string escape");
                c = data[pos++];
                if (c == '"' || c == '\\' || c == '/')
                    text.push_back((char)c);
                else if (c == 'b')
                    text.push_back('\b');
                else if (c == 'f')
                    text.push_back('\f');
                else if (c == 'n')
                    text.push_back('\n');
                else if (c == 'r')
                    text.push_back('\r');
                else if (c == 't')
                    text.push_back('\t');
                else if (c == 'u')
                {
                    unsigned int value;
                    if (!hex4(value))
                        return false;
                    if (value >= 0xd800 && value <= 0xdbff)
                    {
                        if (size - pos < 2 || data[pos] != '\\' || data[pos + 1] != 'u')
                            return fail("missing low surrogate");
                        pos += 2;
                        unsigned int low;
                        if (!hex4(low))
                            return false;
                        if (low < 0xdc00 || low > 0xdfff)
                            return fail("invalid low surrogate");
                        value = 0x10000 + ((value - 0xd800) << 10) + low - 0xdc00;
                    }
                    else if (value >= 0xdc00 && value <= 0xdfff)
                        return fail("unpaired low surrogate");
                    utf8(value, text);
                }
                else
                    return fail("invalid string escape");
            }
            else if (c < 0x80)
                text.push_back((char)c);
            else
            {
                int count;
                unsigned int value;
                unsigned int minimum;
                if (c >= 0xc2 && c <= 0xdf)
                {
                    count = 1;
                    value = c & 31;
                    minimum = 0x80;
                }
                else if (c >= 0xe0 && c <= 0xef)
                {
                    count = 2;
                    value = c & 15;
                    minimum = 0x800;
                }
                else if (c >= 0xf0 && c <= 0xf4)
                {
                    count = 3;
                    value = c & 7;
                    minimum = 0x10000;
                }
                else
                    return fail("invalid UTF-8 sequence");
                for (int i = 0; i < count; i++)
                {
                    if (pos == size)
                        return fail("truncated UTF-8 sequence");
                    unsigned char next = data[pos++];
                    if ((next & 0xc0) != 0x80)
                        return fail("invalid UTF-8 continuation");
                    value = (value << 6) | (next & 63);
                }
                if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff))
                    return fail("invalid UTF-8 code point");
                utf8(value, text);
            }
        }
        return fail("unterminated string");
    }

    bool digit() const
    {
        return pos < size && data[pos] >= '0' && data[pos] <= '9';
    }

    bool number(JsonValue& value)
    {
        const size_t start = pos;
        bool negative = false;
        if (data[pos] == '-')
        {
            negative = true;
            pos++;
        }
        if (!digit())
            return fail("expected digit");
        if (data[pos] == '0')
        {
            pos++;
            if (digit())
                return fail("leading zero");
        }
        else
            while (digit())
                pos++;
        bool floating = false;
        if (pos < size && data[pos] == '.')
        {
            floating = true;
            pos++;
            if (!digit())
                return fail("missing fractional digits");
            while (digit())
                pos++;
        }
        if (pos < size && (data[pos] == 'e' || data[pos] == 'E'))
        {
            floating = true;
            pos++;
            if (pos < size && (data[pos] == '+' || data[pos] == '-'))
                pos++;
            if (!digit())
                return fail("missing exponent digits");
            while (digit())
                pos++;
        }
        if (floating)
        {
            std::istringstream stream(std::string(data + start, pos - start));
            stream.imbue(std::locale::classic());
            double f;
            stream >> f;
            if (stream.fail() || !std::isfinite(f))
                return fail("floating point overflow");
            value = JsonValue(f);
        }
        else
        {
            uint64_t n = 0;
            const uint64_t limit = negative ? (uint64_t)INT64_MAX + 1 : (uint64_t)INT64_MAX;
            for (size_t i = start + (negative ? 1 : 0); i < pos; i++)
            {
                unsigned int d = data[i] - '0';
                if (n > (limit - d) / 10)
                    return fail("integer overflow");
                n = n * 10 + d;
            }
            int64_t result;
            if (!negative)
                result = (int64_t)n;
            else if (n == (uint64_t)INT64_MAX + 1)
                result = INT64_MIN;
            else
                result = -(int64_t)n;
            value = JsonValue(result);
        }
        return true;
    }

    bool parse(JsonValue& value, unsigned int depth)
    {
        if (depth > 128)
            return fail("nesting limit exceeded");
        whitespace();
        if (pos == size)
            return fail("unexpected end of input");
        const char c = data[pos];
        if (c == '"')
        {
            value.type = JsonValue::String;
            return string(value.s);
        }
        if (c == 'n')
            return literal("null");
        if (c == 't')
        {
            value = JsonValue(true);
            return literal("true");
        }
        if (c == 'f')
        {
            value = JsonValue(false);
            return literal("false");
        }
        if (c == '-' || digit())
            return number(value);
        if (c == '[')
        {
            value.type = JsonValue::Array;
            pos++;
            whitespace();
            if (pos < size && data[pos] == ']')
            {
                pos++;
                return true;
            }
            for (;;)
            {
                JsonValue item;
                if (!parse(item, depth + 1))
                    return false;
                value.a.push_back(std::move(item));
                whitespace();
                if (pos == size)
                    return fail("unterminated array");
                char delimiter = data[pos++];
                if (delimiter == ']')
                    return true;
                if (delimiter != ',')
                    return fail("expected array comma");
            }
        }
        if (c == '{')
        {
            value.type = JsonValue::Object;
            pos++;
            whitespace();
            if (pos < size && data[pos] == '}')
            {
                pos++;
                return true;
            }
            for (;;)
            {
                whitespace();
                std::string key;
                if (!string(key))
                    return false;
                if (value.o.count(key))
                    return fail("duplicate object key");
                whitespace();
                if (pos == size || data[pos++] != ':')
                    return fail("expected object colon");
                JsonValue item;
                if (!parse(item, depth + 1))
                    return false;
                value.o.insert(std::make_pair(key, std::move(item)));
                whitespace();
                if (pos == size)
                    return fail("unterminated object");
                char delimiter = data[pos++];
                if (delimiter == '}')
                    return true;
                if (delimiter != ',')
                    return fail("expected object comma");
            }
        }
        return fail("unexpected character");
    }

    const char* data;
    size_t size;
    size_t pos;
    std::string error;
};

int parse_json(const char* data, size_t size, JsonValue& value, std::string& error)
{
    value = JsonValue();
    JsonParser parser(data, size);
    JsonValue result;
    if (!parser.parse(result, 0))
    {
        error = parser.error;
        return -1;
    }
    parser.whitespace();
    if (parser.pos != size)
    {
        parser.fail("trailing data");
        error = parser.error;
        return -1;
    }
    value = std::move(result);
    error.clear();
    return 0;
}

} // namespace pnnx
