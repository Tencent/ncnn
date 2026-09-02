#include "json.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>

#include <limits>

namespace pnnx {

JsonValue::JsonValue()
    : value_type(Null), boolean_value(false), integer_value(0), number_value(0.0)
{
}

JsonValue::Type JsonValue::type() const
{
    return value_type;
}

bool JsonValue::is_null() const
{
    return value_type == Null;
}

bool JsonValue::get_bool(bool& value) const
{
    if (value_type != Boolean)
        return false;

    value = boolean_value;
    return true;
}

bool JsonValue::get_int(int64_t& value) const
{
    if (value_type != Integer)
        return false;

    value = integer_value;
    return true;
}

bool JsonValue::get_number(double& value) const
{
    if (value_type == Integer)
    {
        value = (double)integer_value;
        return true;
    }

    if (value_type != Number)
        return false;

    value = number_value;
    return true;
}

const std::string* JsonValue::get_string() const
{
    if (value_type != String)
        return 0;

    return &string_value;
}

const std::vector<JsonValue>* JsonValue::get_array() const
{
    if (value_type != Array)
        return 0;

    return &array_value;
}

const std::map<std::string, JsonValue>* JsonValue::get_object() const
{
    if (value_type != Object)
        return 0;

    return &object_value;
}

const JsonValue* JsonValue::get(const std::string& key) const
{
    if (value_type != Object)
        return 0;

    std::map<std::string, JsonValue>::const_iterator it = object_value.find(key);
    if (it == object_value.end())
        return 0;

    return &it->second;
}

class JsonParser
{
public:
    JsonParser(const std::string& text, std::string& error)
        : text(text), error(error), offset(0)
    {
        error.clear();
    }

    bool parse(JsonValue& value)
    {
        skip_whitespace();
        if (!parse_value(value))
            return false;

        skip_whitespace();
        if (offset != text.size())
            return fail("unexpected trailing character");

        return true;
    }

private:
    bool parse_value(JsonValue& value)
    {
        if (offset == text.size())
            return fail("expected value");

        const char ch = text[offset];
        if (ch == 'n')
            return parse_literal("null", JsonValue::Null, value);
        if (ch == 't')
            return parse_literal("true", JsonValue::Boolean, value, true);
        if (ch == 'f')
            return parse_literal("false", JsonValue::Boolean, value, false);
        if (ch == '"')
        {
            value = JsonValue();
            value.value_type = JsonValue::String;
            return parse_string(value.string_value);
        }
        if (ch == '[')
            return parse_array(value);
        if (ch == '{')
            return parse_object(value);
        if (ch == '-' || (ch >= '0' && ch <= '9'))
            return parse_number(value);

        return fail("unexpected character while parsing value");
    }

    bool parse_literal(const char* literal, JsonValue::Type type, JsonValue& value, bool boolean_value = false)
    {
        size_t length = 0;
        while (literal[length] != '\0')
            length++;

        if (text.compare(offset, length, literal) != 0)
            return fail("invalid literal");

        offset += length;
        value = JsonValue();
        value.value_type = type;
        value.boolean_value = boolean_value;
        return true;
    }

    bool parse_number(JsonValue& value)
    {
        const size_t begin = offset;

        if (text[offset] == '-')
        {
            offset++;
            if (offset == text.size())
                return fail("expected digit after minus sign");
        }

        if (text[offset] == '0')
        {
            offset++;
        }
        else if (text[offset] >= '1' && text[offset] <= '9')
        {
            while (offset < text.size() && text[offset] >= '0' && text[offset] <= '9')
                offset++;
        }
        else
        {
            return fail("expected digit");
        }

        bool is_integer = true;
        if (offset < text.size() && text[offset] == '.')
        {
            is_integer = false;
            offset++;
            if (offset == text.size() || text[offset] < '0' || text[offset] > '9')
                return fail("expected digit after decimal point");

            while (offset < text.size() && text[offset] >= '0' && text[offset] <= '9')
                offset++;
        }

        if (offset < text.size() && (text[offset] == 'e' || text[offset] == 'E'))
        {
            is_integer = false;
            offset++;
            if (offset < text.size() && (text[offset] == '+' || text[offset] == '-'))
                offset++;
            if (offset == text.size() || text[offset] < '0' || text[offset] > '9')
                return fail("expected digit in exponent");

            while (offset < text.size() && text[offset] >= '0' && text[offset] <= '9')
                offset++;
        }

        const std::string token = text.substr(begin, offset - begin);
        char* endptr = 0;
        errno = 0;

        value = JsonValue();
        if (is_integer)
        {
            const long long parsed = strtoll(token.c_str(), &endptr, 10);
            if (errno == ERANGE || endptr != token.c_str() + token.size())
                return fail("integer is out of range");

            value.value_type = JsonValue::Integer;
            value.integer_value = (int64_t)parsed;
            return true;
        }

        const double parsed = strtod(token.c_str(), &endptr);
        if (errno == ERANGE || endptr != token.c_str() + token.size() || !isfinite(parsed))
            return fail("number is out of range");

        value.value_type = JsonValue::Number;
        value.number_value = parsed;
        return true;
    }

    bool parse_string(std::string& value)
    {
        offset++;
        value.clear();

        while (offset < text.size())
        {
            const unsigned char ch = (unsigned char)text[offset++];
            if (ch == '"')
                return true;
            if (ch == '\\')
            {
                if (!parse_escape(value))
                    return false;
                continue;
            }
            if (ch < 0x20)
                return fail("unescaped control character in string");

            value.push_back((char)ch);
        }

        return fail("unterminated string");
    }

    bool parse_escape(std::string& value)
    {
        if (offset == text.size())
            return fail("unterminated escape sequence");

        const char escaped = text[offset++];
        if (escaped == '"' || escaped == '\\' || escaped == '/')
        {
            value.push_back(escaped);
            return true;
        }
        if (escaped == 'b')
        {
            value.push_back('\b');
            return true;
        }
        if (escaped == 'f')
        {
            value.push_back('\f');
            return true;
        }
        if (escaped == 'n')
        {
            value.push_back('\n');
            return true;
        }
        if (escaped == 'r')
        {
            value.push_back('\r');
            return true;
        }
        if (escaped == 't')
        {
            value.push_back('\t');
            return true;
        }
        if (escaped != 'u')
            return fail("invalid escape sequence");

        uint32_t codepoint = 0;
        if (!parse_hex4(codepoint))
            return false;

        if (codepoint >= 0xd800 && codepoint <= 0xdbff)
        {
            if (offset + 2 > text.size() || text[offset] != '\\' || text[offset + 1] != 'u')
                return fail("missing low surrogate");
            offset += 2;

            uint32_t low = 0;
            if (!parse_hex4(low))
                return false;
            if (low < 0xdc00 || low > 0xdfff)
                return fail("invalid low surrogate");

            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
        }
        else if (codepoint >= 0xdc00 && codepoint <= 0xdfff)
        {
            return fail("unexpected low surrogate");
        }

        append_utf8(codepoint, value);
        return true;
    }

    bool parse_hex4(uint32_t& value)
    {
        if (text.size() - offset < 4)
            return fail("incomplete unicode escape");

        value = 0;
        for (int i = 0; i < 4; i++)
        {
            const char ch = text[offset++];
            value <<= 4;
            if (ch >= '0' && ch <= '9')
                value |= ch - '0';
            else if (ch >= 'a' && ch <= 'f')
                value |= ch - 'a' + 10;
            else if (ch >= 'A' && ch <= 'F')
                value |= ch - 'A' + 10;
            else
                return fail("invalid unicode escape");
        }

        return true;
    }

    static void append_utf8(uint32_t codepoint, std::string& value)
    {
        if (codepoint <= 0x7f)
        {
            value.push_back((char)codepoint);
        }
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

    bool parse_array(JsonValue& value)
    {
        offset++;
        value = JsonValue();
        value.value_type = JsonValue::Array;

        skip_whitespace();
        if (offset < text.size() && text[offset] == ']')
        {
            offset++;
            return true;
        }

        while (true)
        {
            JsonValue element;
            if (!parse_value(element))
                return false;
            value.array_value.push_back(element);

            skip_whitespace();
            if (offset == text.size())
                return fail("unterminated array");
            if (text[offset] == ']')
            {
                offset++;
                return true;
            }
            if (text[offset] != ',')
                return fail("expected comma in array");

            offset++;
            skip_whitespace();
        }
    }

    bool parse_object(JsonValue& value)
    {
        offset++;
        value = JsonValue();
        value.value_type = JsonValue::Object;

        skip_whitespace();
        if (offset < text.size() && text[offset] == '}')
        {
            offset++;
            return true;
        }

        while (true)
        {
            if (offset == text.size() || text[offset] != '"')
                return fail("expected object key");

            std::string key;
            if (!parse_string(key))
                return false;

            skip_whitespace();
            if (offset == text.size() || text[offset] != ':')
                return fail("expected colon after object key");
            offset++;
            skip_whitespace();

            JsonValue member;
            if (!parse_value(member))
                return false;
            if (!value.object_value.insert(std::make_pair(key, member)).second)
                return fail("duplicate object key");

            skip_whitespace();
            if (offset == text.size())
                return fail("unterminated object");
            if (text[offset] == '}')
            {
                offset++;
                return true;
            }
            if (text[offset] != ',')
                return fail("expected comma in object");

            offset++;
            skip_whitespace();
        }
    }

    void skip_whitespace()
    {
        while (offset < text.size())
        {
            const char ch = text[offset];
            if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r')
                break;
            offset++;
        }
    }

    bool fail(const char* message)
    {
        if (error.empty())
            error = "json parse error at offset " + std::to_string(offset) + ": " + message;
        return false;
    }

private:
    const std::string& text;
    std::string& error;
    size_t offset;
};

bool parse_json(const std::string& text, JsonValue& value, std::string& error)
{
    JsonParser parser(text, error);
    return parser.parse(value);
}

} // namespace pnnx