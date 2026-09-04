// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_JSON_HPP
#define PNNX_JSON_HPP

#include <cctype>
#include <cstdio>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// 最小 JSON 解析器（课题 2：pnnx 支持 torch.export .pt2 所需）。
// 设计约束：不引入任何第三方库。仅依赖 STL。
// 支持：null / bool / int / double / string(含转义与\u) / array / object。
// 用法：
//     pnnx::JsonValue root = pnnx::parse_json(text);
//     if (root.isObject()) { const JsonValue& v = root["some_key"]; ... }

namespace pnnx {

class JsonValue
{
public:
    enum Type
    {
        JSON_NULL,
        JSON_BOOL,
        JSON_INT,
        JSON_DOUBLE,
        JSON_STRING,
        JSON_ARRAY,
        JSON_OBJECT
    };

    JsonValue()
        : type(JSON_NULL), bool_value(false), int_value(0), double_value(0.0)
    {
    }

    bool isNull() const
    {
        return type == JSON_NULL;
    }
    bool isBool() const
    {
        return type == JSON_BOOL;
    }
    bool isInt() const
    {
        return type == JSON_INT;
    }
    bool isDouble() const
    {
        return type == JSON_DOUBLE;
    }
    bool isNumber() const
    {
        return type == JSON_INT || type == JSON_DOUBLE;
    }
    bool isString() const
    {
        return type == JSON_STRING;
    }
    bool isArray() const
    {
        return type == JSON_ARRAY;
    }
    bool isObject() const
    {
        return type == JSON_OBJECT;
    }

    bool asBool() const
    {
        return bool_value;
    }
    long long asInt() const
    {
        return int_value;
    }
    double asDouble() const
    {
        return type == JSON_INT ? static_cast<double>(int_value) : double_value;
    }
    const std::string& asString() const
    {
        return string_value;
    }

    size_t size() const
    {
        if (type == JSON_ARRAY)
            return array_value.size();
        if (type == JSON_OBJECT)
            return object_value.size();
        return 0;
    }

    const JsonValue& operator[](size_t i) const
    {
        static const JsonValue null_value;
        if (type == JSON_ARRAY && i < array_value.size())
            return array_value[i];
        return null_value;
    }

    bool hasMember(const std::string& key) const
    {
        return type == JSON_OBJECT && object_value.find(key) != object_value.end();
    }

    const JsonValue& operator[](const std::string& key) const
    {
        static const JsonValue null_value;
        if (type == JSON_OBJECT)
        {
            std::map<std::string, JsonValue>::const_iterator it = object_value.find(key);
            if (it != object_value.end())
                return it->second;
        }
        return null_value;
    }

    Type type;
    bool bool_value;
    long long int_value;
    double double_value;
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::map<std::string, JsonValue> object_value;
};

class JsonParser
{
public:
    explicit JsonParser(const std::string& text)
        : s(text), pos(0)
    {
    }

    JsonValue parse()
    {
        skip_whitespace();
        JsonValue v = parse_value();
        skip_whitespace();
        if (pos != s.size())
            throw std::runtime_error("json parse error: trailing characters at " + pos_str());
        return v;
    }

private:
    const std::string& s;
    size_t pos;

    std::string pos_str() const
    {
        char buf[32];
        snprintf(buf, sizeof(buf), "%zu", pos);
        return std::string(buf);
    }

    void skip_whitespace()
    {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
            ++pos;
    }

    char peek() const
    {
        return pos < s.size() ? s[pos] : '\0';
    }

    JsonValue parse_value()
    {
        skip_whitespace();
        char c = peek();
        if (c == '{')
            return parse_object();
        if (c == '[')
            return parse_array();
        if (c == '"')
            return parse_string();
        if (c == 't' || c == 'f' || c == 'n')
            return parse_literal();
        if (c == '-' || (c >= '0' && c <= '9'))
            return parse_number();
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "json parse error at pos %zu: unexpected byte 0x%02x", pos, (unsigned char)c);
            throw std::runtime_error(std::string(buf));
        }
    }

    JsonValue parse_object()
    {
        JsonValue v;
        v.type = JsonValue::JSON_OBJECT;
        ++pos; // consume '{'
        skip_whitespace();
        if (peek() == '}')
        {
            ++pos;
            return v;
        }
        while (true)
        {
            skip_whitespace();
            if (peek() != '"')
                throw std::runtime_error("json parse error: expected key string at " + pos_str());
            JsonValue key = parse_string();
            skip_whitespace();
            if (peek() != ':')
                throw std::runtime_error("json parse error: expected ':' at " + pos_str());
            ++pos;
            JsonValue val = parse_value();
            v.object_value[key.string_value] = val;
            skip_whitespace();
            char c = peek();
            if (c == ',')
            {
                ++pos;
                continue;
            }
            if (c == '}')
            {
                ++pos;
                break;
            }
            throw std::runtime_error("json parse error: expected ',' or '}' at " + pos_str());
        }
        return v;
    }

    JsonValue parse_array()
    {
        JsonValue v;
        v.type = JsonValue::JSON_ARRAY;
        ++pos; // consume '['
        skip_whitespace();
        if (peek() == ']')
        {
            ++pos;
            return v;
        }
        while (true)
        {
            JsonValue val = parse_value();
            v.array_value.push_back(val);
            skip_whitespace();
            char c = peek();
            if (c == ',')
            {
                ++pos;
                continue;
            }
            if (c == ']')
            {
                ++pos;
                break;
            }
            throw std::runtime_error("json parse error: expected ',' or ']' at " + pos_str());
        }
        return v;
    }

    JsonValue parse_string()
    {
        JsonValue v;
        v.type = JsonValue::JSON_STRING;
        ++pos; // consume opening quote
        std::string out;
        bool closed = false;
        while (pos < s.size())
        {
            char c = s[pos++];
            if (c == '"')
            {
                closed = true;
                break;
            }
            if (c == '\\')
            {
                if (pos >= s.size())
                    throw std::runtime_error("json parse error: unterminated escape");
                char e = s[pos++];
                switch (e)
                {
                case '"':
                    out.push_back('"');
                    break;
                case '\\':
                    out.push_back('\\');
                    break;
                case '/':
                    out.push_back('/');
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u':
                {
                    if (pos + 4 > s.size())
                        throw std::runtime_error("json parse error: bad \\u escape");
                    unsigned int cp = 0;
                    for (int i = 0; i < 4; ++i)
                    {
                        char h = s[pos++];
                        cp <<= 4;
                        if (h >= '0' && h <= '9')
                            cp |= (h - '0');
                        else if (h >= 'a' && h <= 'f')
                            cp |= (h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F')
                            cp |= (h - 'A' + 10);
                        else
                            throw std::runtime_error("json parse error: bad hex in \\u");
                    }
                    // 高代理后紧跟合法低代理 → 合并码点展开为 4 字节 UTF-8
                    if (cp >= 0xD800 && cp <= 0xDBFF && pos + 6 <= s.size()
                        && s[pos] == '\\' && s[pos + 1] == 'u')
                    {
                        size_t save = pos;
                        pos += 2;
                        unsigned int low = 0;
                        bool low_ok = true;
                        for (int i = 0; i < 4; ++i)
                        {
                            char h = s[pos++];
                            low <<= 4;
                            if (h >= '0' && h <= '9')
                                low |= (h - '0');
                            else if (h >= 'a' && h <= 'f')
                                low |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F')
                                low |= (h - 'A' + 10);
                            else
                            {
                                low_ok = false;
                                break;
                            }
                        }
                        if (low_ok && low >= 0xDC00 && low <= 0xDFFF)
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                        else
                            pos = save; // 不是合法低代理，回退按孤立高代理处理
                    }
                    // Basic Multilingual Plane 直接转 UTF-8；孤立代理按 3 字节保留
                    if (cp < 0x80)
                        out.push_back(static_cast<char>(cp));
                    else if (cp < 0x800)
                    {
                        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    }
                    else if (cp < 0x10000)
                    {
                        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    }
                    else
                    {
                        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    }
                    break;
                }
                default:
                    throw std::runtime_error(std::string("json parse error: bad escape '") + e + "'");
                }
            }
            else
            {
                out.push_back(c);
            }
        }
        if (!closed)
            throw std::runtime_error("json parse error: unterminated string");
        v.string_value = out;
        return v;
    }

    JsonValue parse_number()
    {
        size_t start = pos;
        bool is_double = false;

        if (peek() == '-')
            ++pos;

        // 整数部分：至少一位数字；不允许前导零（"0" 本身除外）
        size_t int_digits_start = pos;
        while (pos < s.size() && isdigit_s(s[pos]))
            ++pos;
        if (pos == int_digits_start)
            throw std::runtime_error("json parse error: bad number at " + pos_str());
        if (s[int_digits_start] == '0' && pos - int_digits_start > 1)
            throw std::runtime_error("json parse error: leading zero in number at " + pos_str());

        if (peek() == '.')
        {
            is_double = true;
            ++pos;
            if (!isdigit_s(peek()))
                throw std::runtime_error("json parse error: digit required after '.' at " + pos_str());
            while (pos < s.size() && isdigit_s(s[pos]))
                ++pos;
        }

        if (peek() == 'e' || peek() == 'E')
        {
            is_double = true;
            ++pos;
            if (peek() == '+' || peek() == '-')
                ++pos;
            if (!isdigit_s(peek()))
                throw std::runtime_error("json parse error: digit required in exponent at " + pos_str());
            while (pos < s.size() && isdigit_s(s[pos]))
                ++pos;
        }

        std::string num = s.substr(start, pos - start);
        JsonValue v;
        if (!is_double)
        {
            v.type = JsonValue::JSON_INT;
            v.int_value = strtoll(num.c_str(), 0, 10);
        }
        else
        {
            v.type = JsonValue::JSON_DOUBLE;
            v.double_value = strtod(num.c_str(), 0);
        }
        return v;
    }

    JsonValue parse_literal()
    {
        if (s.compare(pos, 4, "true") == 0)
        {
            pos += 4;
            JsonValue v;
            v.type = JsonValue::JSON_BOOL;
            v.bool_value = true;
            return v;
        }
        if (s.compare(pos, 5, "false") == 0)
        {
            pos += 5;
            JsonValue v;
            v.type = JsonValue::JSON_BOOL;
            v.bool_value = false;
            return v;
        }
        if (s.compare(pos, 4, "null") == 0)
        {
            pos += 4;
            JsonValue v;
            v.type = JsonValue::JSON_NULL;
            return v;
        }
        throw std::runtime_error("json parse error: bad literal at " + pos_str());
    }

    static bool isdigit_s(char c)
    {
        return c >= '0' && c <= '9';
    }
};

inline JsonValue parse_json(const std::string& text)
{
    JsonParser parser(text);
    return parser.parse();
}

} // namespace pnnx

#endif // PNNX_JSON_HPP
