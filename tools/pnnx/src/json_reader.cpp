// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "json_reader.h"

#include <string.h>

#include <cmath>
#include <limits>
#include <locale>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace pnnx {

JsonParseOptions::JsonParseOptions()
{
    max_depth = 256;
    max_nodes = 16 * 1024 * 1024;
    max_string_length = 64 * 1024 * 1024;
}

JsonValue::JsonValue()
{
    type_ = JSON_NULL;
    bool_value_ = false;
    int64_value_ = 0;
    uint64_value_ = 0;
    double_value_ = 0.0;
}

JsonType JsonValue::type() const
{
    return type_;
}

bool JsonValue::as_bool() const
{
    return bool_value_;
}

int64_t JsonValue::as_int64() const
{
    return int64_value_;
}

uint64_t JsonValue::as_uint64() const
{
    return uint64_value_;
}

double JsonValue::as_double() const
{
    return double_value_;
}

const std::string& JsonValue::as_string() const
{
    return string_value_;
}

const std::vector<JsonValue>& JsonValue::as_array() const
{
    return array_value_;
}

const std::map<std::string, JsonValue>& JsonValue::as_object() const
{
    return object_value_;
}

const JsonValue* JsonValue::find(const std::string& key) const
{
    if (type_ != JSON_OBJECT)
        return 0;

    std::map<std::string, JsonValue>::const_iterator it = object_value_.find(key);
    if (it == object_value_.end())
        return 0;

    return &it->second;
}

class JsonParser
{
public:
    JsonParser(const char* data, size_t size, JsonParseError& error, const JsonParseOptions& options)
        : data_(data), size_(size), error_(error), options_(options)
    {
        position_ = 0;
        node_count_ = 0;
    }

    int parse(JsonValue& value)
    {
        error_.byte_offset = 0;
        error_.line = 1;
        error_.column = 1;
        error_.message.clear();

        value = JsonValue();

        if (!data_ && size_ != 0)
        {
            fail(0, "json data is null");
            return -1;
        }

        skip_whitespace();
        if (position_ == size_)
        {
            fail(position_, "expected json value");
            return -1;
        }

        if (!parse_value(value, 1))
            return -1;

        skip_whitespace();
        if (position_ != size_)
        {
            fail(position_, "trailing characters after json value");
            return -1;
        }

        return 0;
    }

private:
    void skip_whitespace()
    {
        while (position_ < size_)
        {
            const char ch = data_[position_];
            if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n')
                break;

            position_++;
        }
    }

    bool parse_value(JsonValue& value, size_t depth)
    {
        if (position_ == size_)
            return fail(position_, "expected json value");
        if (depth > options_.max_depth)
            return fail(position_, "json depth limit exceeded");
        if (node_count_ >= options_.max_nodes)
            return fail(position_, "json node limit exceeded");

        node_count_++;

        const char ch = data_[position_];
        if (ch == 'n')
            return parse_literal("null", value, JSON_NULL, false);
        if (ch == 't')
            return parse_literal("true", value, JSON_BOOL, true);
        if (ch == 'f')
            return parse_literal("false", value, JSON_BOOL, false);
        if (ch == '"')
            return parse_string(value);
        if (ch == '-' || (ch >= '0' && ch <= '9'))
            return parse_number(value);
        if (ch == '[')
            return parse_array(value, depth);
        if (ch == '{')
            return parse_object(value, depth);

        return fail(position_, "expected json value");
    }

    bool parse_literal(const char* literal, JsonValue& value, JsonType type, bool bool_value)
    {
        const size_t literal_size = strlen(literal);
        if (literal_size > size_ - position_ || memcmp(data_ + position_, literal, literal_size) != 0)
            return fail(position_, "invalid literal");

        position_ += literal_size;
        value.type_ = type;
        value.bool_value_ = bool_value;
        return true;
    }

    bool append_string_bytes(std::string& text, const char* bytes, size_t byte_count, size_t character_offset)
    {
        if (text.size() > options_.max_string_length || byte_count > options_.max_string_length - text.size())
            return fail(character_offset, "json string length limit exceeded");

        text.append(bytes, byte_count);
        return true;
    }

    bool append_string_character(std::string& text, char ch, size_t character_offset)
    {
        return append_string_bytes(text, &ch, 1, character_offset);
    }

    bool append_utf8_code_point(std::string& text, uint32_t code_point, size_t character_offset)
    {
        char bytes[4];
        size_t byte_count = 0;
        if (code_point <= 0x7f)
        {
            bytes[0] = (char)code_point;
            byte_count = 1;
        }
        else if (code_point <= 0x7ff)
        {
            bytes[0] = (char)(0xc0 | (code_point >> 6));
            bytes[1] = (char)(0x80 | (code_point & 0x3f));
            byte_count = 2;
        }
        else if (code_point >= 0xd800 && code_point <= 0xdfff)
        {
            return fail(character_offset, "unicode surrogate code point is invalid");
        }
        else if (code_point <= 0xffff)
        {
            bytes[0] = (char)(0xe0 | (code_point >> 12));
            bytes[1] = (char)(0x80 | ((code_point >> 6) & 0x3f));
            bytes[2] = (char)(0x80 | (code_point & 0x3f));
            byte_count = 3;
        }
        else if (code_point <= 0x10ffff)
        {
            bytes[0] = (char)(0xf0 | (code_point >> 18));
            bytes[1] = (char)(0x80 | ((code_point >> 12) & 0x3f));
            bytes[2] = (char)(0x80 | ((code_point >> 6) & 0x3f));
            bytes[3] = (char)(0x80 | (code_point & 0x3f));
            byte_count = 4;
        }
        else
        {
            return fail(character_offset, "unicode code point out of range");
        }

        return append_string_bytes(text, bytes, byte_count, character_offset);
    }

    static int hex_digit(char ch)
    {
        if (ch >= '0' && ch <= '9')
            return ch - '0';
        if (ch >= 'a' && ch <= 'f')
            return ch - 'a' + 10;
        if (ch >= 'A' && ch <= 'F')
            return ch - 'A' + 10;

        return -1;
    }

    bool parse_unicode_quad(uint16_t& code_unit)
    {
        uint16_t value = 0;
        for (int i = 0; i < 4; i++)
        {
            if (position_ == size_)
                return fail(position_, "incomplete unicode escape");

            const int digit = hex_digit(data_[position_]);
            if (digit < 0)
                return fail(position_, "invalid unicode escape");

            value = (uint16_t)(value * 16 + digit);
            position_++;
        }

        code_unit = value;
        return true;
    }

    bool parse_raw_utf8(std::string& text)
    {
        const size_t sequence_offset = position_;
        const unsigned char leading_byte = (unsigned char)data_[sequence_offset];

        size_t sequence_length = 0;
        uint32_t code_point = 0;
        uint32_t minimum_code_point = 0;
        if (leading_byte >= 0xc2 && leading_byte <= 0xdf)
        {
            sequence_length = 2;
            code_point = leading_byte & 0x1f;
            minimum_code_point = 0x80;
        }
        else if (leading_byte >= 0xe0 && leading_byte <= 0xef)
        {
            sequence_length = 3;
            code_point = leading_byte & 0x0f;
            minimum_code_point = 0x800;
        }
        else if (leading_byte >= 0xf0 && leading_byte <= 0xf4)
        {
            sequence_length = 4;
            code_point = leading_byte & 0x07;
            minimum_code_point = 0x10000;
        }
        else
        {
            return fail(sequence_offset, "invalid utf-8 leading byte in json string");
        }

        if (sequence_length > size_ - sequence_offset)
            return fail(size_, "truncated utf-8 sequence in json string");

        for (size_t i = 1; i < sequence_length; i++)
        {
            const unsigned char continuation_byte = (unsigned char)data_[sequence_offset + i];
            if ((continuation_byte & 0xc0) != 0x80)
                return fail(sequence_offset + i, "invalid utf-8 continuation byte in json string");

            code_point = (code_point << 6) | (continuation_byte & 0x3f);
        }

        if (code_point < minimum_code_point)
            return fail(sequence_offset, "overlong utf-8 sequence in json string");
        if (code_point >= 0xd800 && code_point <= 0xdfff)
            return fail(sequence_offset, "utf-8 surrogate code point in json string");
        if (code_point > 0x10ffff)
            return fail(sequence_offset, "utf-8 code point out of range");

        if (!append_string_bytes(text, data_ + sequence_offset, sequence_length, sequence_offset))
            return false;

        position_ += sequence_length;
        return true;
    }

    bool parse_string(JsonValue& value)
    {
        position_++;
        std::string text;

        while (position_ < size_)
        {
            const size_t character_offset = position_;
            const unsigned char ch = (unsigned char)data_[position_];

            if (ch == '"')
            {
                position_++;
                value.type_ = JSON_STRING;
                value.string_value_.swap(text);
                return true;
            }

            if (ch < 0x20)
                return fail(character_offset, "control character in json string");
            if (ch >= 0x80)
            {
                if (!parse_raw_utf8(text))
                    return false;
                continue;
            }

            position_++;

            if (ch != '\\')
            {
                if (!append_string_character(text, (char)ch, character_offset))
                    return false;
                continue;
            }

            if (position_ == size_)
                return fail(position_, "unterminated json escape sequence");

            const size_t escape_offset = position_;
            const char escape = data_[position_++];
            char decoded = 0;
            if (escape == '"' || escape == '\\' || escape == '/')
                decoded = escape;
            else if (escape == 'b')
                decoded = '\b';
            else if (escape == 'f')
                decoded = '\f';
            else if (escape == 'n')
                decoded = '\n';
            else if (escape == 'r')
                decoded = '\r';
            else if (escape == 't')
                decoded = '\t';
            else if (escape == 'u')
            {
                uint16_t first_code_unit = 0;
                if (!parse_unicode_quad(first_code_unit))
                    return false;

                uint32_t code_point = first_code_unit;
                if (first_code_unit >= 0xd800 && first_code_unit <= 0xdbff)
                {
                    if (size_ - position_ < 2 || data_[position_] != '\\' || data_[position_ + 1] != 'u')
                        return fail(position_, "unpaired high surrogate");

                    position_ += 2;
                    const size_t low_surrogate_offset = position_ - 1;
                    uint16_t second_code_unit = 0;
                    if (!parse_unicode_quad(second_code_unit))
                        return false;
                    if (second_code_unit < 0xdc00 || second_code_unit > 0xdfff)
                        return fail(low_surrogate_offset, "invalid low surrogate");

                    code_point = 0x10000 + (((uint32_t)first_code_unit - 0xd800) << 10) + ((uint32_t)second_code_unit - 0xdc00);
                }
                else if (first_code_unit >= 0xdc00 && first_code_unit <= 0xdfff)
                {
                    return fail(escape_offset, "unpaired low surrogate");
                }

                if (!append_utf8_code_point(text, code_point, escape_offset))
                    return false;
                continue;
            }
            else
                return fail(escape_offset, "invalid escape in json string");

            if (!append_string_character(text, decoded, escape_offset))
                return false;
        }

        return fail(position_, "unterminated string");
    }

    bool parse_array(JsonValue& value, size_t depth)
    {
        value.type_ = JSON_ARRAY;
        value.array_value_.clear();
        position_++;
        skip_whitespace();

        if (position_ == size_)
            return fail(position_, "unterminated array");
        if (data_[position_] == ']')
        {
            position_++;
            return true;
        }

        while (true)
        {
            JsonValue element;
            if (!parse_value(element, depth + 1))
                return false;
            value.array_value_.push_back(std::move(element));

            skip_whitespace();
            if (position_ == size_)
                return fail(position_, "unterminated array");
            if (data_[position_] == ']')
            {
                position_++;
                return true;
            }
            if (data_[position_] != ',')
                return fail(position_, "expected comma or closing bracket in json array");

            position_++;
            skip_whitespace();
            if (position_ == size_)
                return fail(position_, "unterminated array");
            if (data_[position_] == ']')
                return fail(position_, "trailing comma in json array");
        }
    }

    bool parse_object(JsonValue& value, size_t depth)
    {
        value.type_ = JSON_OBJECT;
        value.object_value_.clear();
        position_++;
        skip_whitespace();

        if (position_ == size_)
            return fail(position_, "unterminated object");
        if (data_[position_] == '}')
        {
            position_++;
            return true;
        }

        while (true)
        {
            if (data_[position_] != '"')
                return fail(position_, "expected object key string");

            JsonValue key_value;
            if (!parse_string(key_value))
                return false;
            std::string key;
            key.swap(key_value.string_value_);
            if (value.object_value_.find(key) != value.object_value_.end())
                return fail(position_, "duplicate key " + key);

            skip_whitespace();
            if (position_ == size_ || data_[position_] != ':')
                return fail(position_, "expected colon after json object key");

            position_++;
            skip_whitespace();
            JsonValue member;
            if (!parse_value(member, depth + 1))
                return false;
            value.object_value_.insert(std::make_pair(std::move(key), std::move(member)));

            skip_whitespace();
            if (position_ == size_)
                return fail(position_, "unterminated object");
            if (data_[position_] == '}')
            {
                position_++;
                return true;
            }
            if (data_[position_] != ',')
                return fail(position_, "expected comma or closing brace in json object");

            position_++;
            skip_whitespace();
            if (position_ == size_)
                return fail(position_, "unterminated object");
            if (data_[position_] == '}')
                return fail(position_, "trailing comma in json object");
        }
    }

    bool parse_number(JsonValue& value)
    {
        const size_t number_start = position_;
        bool negative = false;
        if (data_[position_] == '-')
        {
            negative = true;
            position_++;
            if (position_ == size_)
                return fail(position_, "json number requires digit");
        }

        const size_t integer_start = position_;
        if (data_[position_] == '0')
        {
            position_++;
            if (position_ < size_ && data_[position_] >= '0' && data_[position_] <= '9')
                return fail(position_, "leading zero in json number");
        }
        else if (data_[position_] >= '1' && data_[position_] <= '9')
        {
            while (position_ < size_ && data_[position_] >= '0' && data_[position_] <= '9')
                position_++;
        }
        else
        {
            return fail(position_, "json number requires digit");
        }

        bool floating_point = false;
        if (position_ < size_ && data_[position_] == '.')
        {
            floating_point = true;
            position_++;
            if (position_ == size_ || data_[position_] < '0' || data_[position_] > '9')
                return fail(position_, "json number fraction requires digit");

            while (position_ < size_ && data_[position_] >= '0' && data_[position_] <= '9')
                position_++;
        }

        if (position_ < size_ && (data_[position_] == 'e' || data_[position_] == 'E'))
        {
            floating_point = true;
            position_++;
            if (position_ < size_ && (data_[position_] == '+' || data_[position_] == '-'))
                position_++;
            if (position_ == size_ || data_[position_] < '0' || data_[position_] > '9')
                return fail(position_, "json number exponent requires digit");

            while (position_ < size_ && data_[position_] >= '0' && data_[position_] <= '9')
                position_++;
        }

        if (floating_point)
        {
            const std::string number_text(data_ + number_start, position_ - number_start);
            std::istringstream stream(number_text);
            stream.imbue(std::locale::classic());

            double double_value = 0.0;
            stream >> double_value;
            if (!stream || stream.peek() != std::char_traits<char>::eof() || !std::isfinite(double_value))
                return fail(position_, "non-finite number");

            value.type_ = JSON_DOUBLE;
            value.double_value_ = double_value;
            return true;
        }

        uint64_t magnitude = 0;
        bool overflow = false;
        for (size_t i = integer_start; i < position_; i++)
        {
            const uint64_t digit = data_[i] - '0';
            if (magnitude > (std::numeric_limits<uint64_t>::max() - digit) / 10)
            {
                overflow = true;
                break;
            }

            magnitude = magnitude * 10 + digit;
        }

        if (negative)
        {
            const uint64_t int64_min_magnitude = (uint64_t)std::numeric_limits<int64_t>::max() + 1;
            if (overflow || magnitude > int64_min_magnitude)
                return fail(position_, "json integer overflow");

            value.type_ = JSON_INT64;
            if (magnitude == int64_min_magnitude)
                value.int64_value_ = std::numeric_limits<int64_t>::min();
            else
                value.int64_value_ = -(int64_t)magnitude;
            return true;
        }

        if (overflow)
            return fail(position_, "json integer overflow");

        if (magnitude <= (uint64_t)std::numeric_limits<int64_t>::max())
        {
            value.type_ = JSON_INT64;
            value.int64_value_ = (int64_t)magnitude;
        }
        else
        {
            value.type_ = JSON_UINT64;
            value.uint64_value_ = magnitude;
        }

        return true;
    }

    bool fail(size_t byte_offset, const std::string& message)
    {
        if (!error_.message.empty())
            return false;

        if (byte_offset > size_)
            byte_offset = size_;

        error_.byte_offset = byte_offset;
        error_.line = 1;
        error_.column = 1;
        error_.message = message;

        size_t offset = 0;
        while (offset < byte_offset)
        {
            const char ch = data_[offset];
            if (ch == '\r')
            {
                if (offset + 1 < byte_offset && data_[offset + 1] == '\n')
                    offset++;

                error_.line++;
                error_.column = 1;
            }
            else if (ch == '\n')
            {
                error_.line++;
                error_.column = 1;
            }
            else
            {
                error_.column++;
            }

            offset++;
        }

        return false;
    }

private:
    const char* data_;
    size_t size_;
    JsonParseError& error_;
    const JsonParseOptions& options_;
    size_t position_;
    size_t node_count_;
};

int parse_json(const char* data, size_t size, JsonValue& value, JsonParseError& error, const JsonParseOptions& options)
{
    try
    {
        JsonParser parser(data, size, error, options);
        return parser.parse(value);
    }
    catch (const std::length_error&)
    {
        error.byte_offset = 0;
        error.line = 1;
        error.column = 1;
        error.message = "json allocation failed";
        return -1;
    }
    catch (const std::bad_alloc&)
    {
        error.byte_offset = 0;
        error.line = 1;
        error.column = 1;
        error.message = "json allocation failed";
        return -1;
    }
}

} // namespace pnnx
