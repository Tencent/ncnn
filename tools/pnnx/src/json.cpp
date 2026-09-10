#include "json.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>

#include <limits>

namespace pnnx {

JsonParseOptions::JsonParseOptions()
    : max_document_size(512 * 1024 * 1024), max_depth(256), max_values(16 * 1024 * 1024), max_string_size(256 * 1024 * 1024), max_number_size(256)
{
}

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
    JsonParser(const std::string& text, std::string& error, const JsonParseOptions& options)
        : text(text), error(error), options(options), offset(0), value_count(0)
    {
        error.clear();
    }

    bool parse(JsonValue& value)
    {
        if (text.size() > options.max_document_size)
            return fail("document exceeds size limit");

        skip_whitespace();
        if (!parse_value(value, 0))
            return false;

        skip_whitespace();
        if (offset != text.size())
            return fail("unexpected trailing character");

        return true;
    }

private:
    bool parse_value(JsonValue& value, size_t depth)
    {
        if (offset == text.size())
            return fail("expected value");

        if (value_count == options.max_values)
            return fail("value count exceeds limit");
        value_count++;

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
        {
            if (depth == options.max_depth)
                return fail("nesting depth exceeds limit");
            return parse_array(value, depth);
        }
        if (ch == '{')
        {
            if (depth == options.max_depth)
                return fail("nesting depth exceeds limit");
            return parse_object(value, depth);
        }
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

        const size_t token_size = offset - begin;
        if (token_size > options.max_number_size)
            return fail("number exceeds size limit");

        const std::string token = text.substr(begin, token_size);
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
            const unsigned char ch = (unsigned char)text[offset];
            if (ch == '"')
            {
                offset++;
                return true;
            }
            if (ch == '\\')
            {
                offset++;
                if (!parse_escape(value))
                    return false;
                continue;
            }
            if (ch < 0x20)
                return fail("unescaped control character in string");

            if (ch < 0x80)
            {
                offset++;
                if (!append_char((char)ch, value))
                    return false;
                continue;
            }

            if (!parse_utf8(value))
                return false;
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
            return append_char(escaped, value);
        }
        if (escaped == 'b')
        {
            return append_char('\b', value);
        }
        if (escaped == 'f')
        {
            return append_char('\f', value);
        }
        if (escaped == 'n')
        {
            return append_char('\n', value);
        }
        if (escaped == 'r')
        {
            return append_char('\r', value);
        }
        if (escaped == 't')
        {
            return append_char('\t', value);
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

        return append_utf8(codepoint, value);
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

    bool parse_utf8(std::string& value)
    {
        const size_t begin = offset;
        const unsigned char first = (unsigned char)text[offset];
        size_t length = 0;

        if (first >= 0xc2 && first <= 0xdf)
            length = 2;
        else if (first >= 0xe0 && first <= 0xef)
            length = 3;
        else if (first >= 0xf0 && first <= 0xf4)
            length = 4;
        else
            return fail("invalid utf-8 sequence");

        if (text.size() - offset < length)
            return fail("incomplete utf-8 sequence");

        for (size_t i = 1; i < length; i++)
        {
            const unsigned char continuation = (unsigned char)text[offset + i];
            if (continuation < 0x80 || continuation > 0xbf)
                return fail("invalid utf-8 continuation byte");
        }

        const unsigned char second = (unsigned char)text[offset + 1];
        if ((first == 0xe0 && second < 0xa0) || (first == 0xed && second > 0x9f) || (first == 0xf0 && second < 0x90) || (first == 0xf4 && second > 0x8f))
            return fail("invalid utf-8 code point");

        offset += length;
        return append_bytes(text.data() + begin, length, value);
    }

    bool append_char(char ch, std::string& value)
    {
        return append_bytes(&ch, 1, value);
    }

    bool append_bytes(const char* data, size_t size, std::string& value)
    {
        if (value.size() > options.max_string_size || size > options.max_string_size - value.size())
            return fail("string exceeds size limit");

        value.append(data, size);
        return true;
    }

    bool append_utf8(uint32_t codepoint, std::string& value)
    {
        char utf8[4];
        size_t length = 0;

        if (codepoint <= 0x7f)
        {
            utf8[0] = (char)codepoint;
            length = 1;
        }
        else if (codepoint <= 0x7ff)
        {
            utf8[0] = (char)(0xc0 | (codepoint >> 6));
            utf8[1] = (char)(0x80 | (codepoint & 0x3f));
            length = 2;
        }
        else if (codepoint <= 0xffff)
        {
            utf8[0] = (char)(0xe0 | (codepoint >> 12));
            utf8[1] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
            utf8[2] = (char)(0x80 | (codepoint & 0x3f));
            length = 3;
        }
        else
        {
            utf8[0] = (char)(0xf0 | (codepoint >> 18));
            utf8[1] = (char)(0x80 | ((codepoint >> 12) & 0x3f));
            utf8[2] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
            utf8[3] = (char)(0x80 | (codepoint & 0x3f));
            length = 4;
        }

        return append_bytes(utf8, length, value);
    }

    bool parse_array(JsonValue& value, size_t depth)
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
            if (!parse_value(element, depth + 1))
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

    bool parse_object(JsonValue& value, size_t depth)
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
            if (!parse_value(member, depth + 1))
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
    const JsonParseOptions& options;
    size_t offset;
    size_t value_count;
};

bool parse_json(const std::string& text, JsonValue& value, std::string& error)
{
    const JsonParseOptions options;
    JsonParser parser(text, error, options);
    return parser.parse(value);
}

bool parse_json(const std::string& text, JsonValue& value, std::string& error, const JsonParseOptions& options)
{
    JsonParser parser(text, error, options);
    return parser.parse(value);
}

} // namespace pnnx