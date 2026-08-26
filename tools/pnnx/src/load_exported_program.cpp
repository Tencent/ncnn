// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exported_program.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/script.h>
#include <torch/csrc/jit/serialization/import_read.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

#include "storezip.h"

namespace pnnx {

namespace {

class JsonValue
{
public:
    enum Type
    {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object
    };

    JsonValue()
        : type(Null), boolean(false), number(0.0), integer_valid(false), integer(0)
    {
    }

    const JsonValue* find(const std::string& key) const
    {
        if (type != Object)
            return 0;

        std::map<std::string, JsonValue>::const_iterator it = object.find(key);
        if (it == object.end())
            return 0;

        return &it->second;
    }

    Type type;
    bool boolean;
    double number;
    bool integer_valid;
    int64_t integer;
    std::string string;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;
};

class JsonParser
{
public:
    JsonParser(const std::string& text)
        : begin(text.data()), cur(text.data()), end(text.data() + text.size())
    {
    }

    bool parse(JsonValue& value, std::string& error)
    {
        skip_space();
        if (!parse_value(value, 0))
        {
            std::ostringstream ss;
            ss << message << " at byte " << (cur - begin);
            error = ss.str();
            return false;
        }

        skip_space();
        if (cur != end)
        {
            error = "trailing data at byte " + std::to_string(cur - begin);
            return false;
        }

        return true;
    }

private:
    void skip_space()
    {
        while (cur != end && (*cur == ' ' || *cur == '\t' || *cur == '\r' || *cur == '\n'))
            cur++;
    }

    bool fail(const char* text)
    {
        message = text;
        return false;
    }

    bool parse_value(JsonValue& value, int depth)
    {
        if (depth > 256)
            return fail("json nesting is too deep");
        if (cur == end)
            return fail("unexpected end of json");

        if (*cur == 'n')
        {
            if (end - cur < 4 || memcmp(cur, "null", 4) != 0)
                return fail("invalid null literal");
            cur += 4;
            value.type = JsonValue::Null;
            return true;
        }
        if (*cur == 't')
        {
            if (end - cur < 4 || memcmp(cur, "true", 4) != 0)
                return fail("invalid true literal");
            cur += 4;
            value.type = JsonValue::Bool;
            value.boolean = true;
            return true;
        }
        if (*cur == 'f')
        {
            if (end - cur < 5 || memcmp(cur, "false", 5) != 0)
                return fail("invalid false literal");
            cur += 5;
            value.type = JsonValue::Bool;
            value.boolean = false;
            return true;
        }
        // Python's json.dumps(), used by the ExportedProgram serializer,
        // emits these non-standard numeric tokens when a scalar is non-finite.
        if (*cur == 'N' && end - cur >= 3 && memcmp(cur, "NaN", 3) == 0)
        {
            cur += 3;
            value.type = JsonValue::Number;
            value.number = std::numeric_limits<double>::quiet_NaN();
            return true;
        }
        if (*cur == 'I' && end - cur >= 8 && memcmp(cur, "Infinity", 8) == 0)
        {
            cur += 8;
            value.type = JsonValue::Number;
            value.number = std::numeric_limits<double>::infinity();
            return true;
        }
        if (*cur == '-' && end - cur >= 9 && memcmp(cur, "-Infinity", 9) == 0)
        {
            cur += 9;
            value.type = JsonValue::Number;
            value.number = -std::numeric_limits<double>::infinity();
            return true;
        }
        if (*cur == '"')
        {
            value.type = JsonValue::String;
            return parse_string(value.string);
        }
        if (*cur == '[')
            return parse_array(value, depth + 1);
        if (*cur == '{')
            return parse_object(value, depth + 1);
        if (*cur == '-' || (*cur >= '0' && *cur <= '9'))
            return parse_number(value);

        return fail("invalid json value");
    }

    static void append_utf8(std::string& s, uint32_t codepoint)
    {
        if (codepoint <= 0x7f)
        {
            s.push_back((char)codepoint);
        }
        else if (codepoint <= 0x7ff)
        {
            s.push_back((char)(0xc0 | (codepoint >> 6)));
            s.push_back((char)(0x80 | (codepoint & 0x3f)));
        }
        else if (codepoint <= 0xffff)
        {
            s.push_back((char)(0xe0 | (codepoint >> 12)));
            s.push_back((char)(0x80 | ((codepoint >> 6) & 0x3f)));
            s.push_back((char)(0x80 | (codepoint & 0x3f)));
        }
        else
        {
            s.push_back((char)(0xf0 | (codepoint >> 18)));
            s.push_back((char)(0x80 | ((codepoint >> 12) & 0x3f)));
            s.push_back((char)(0x80 | ((codepoint >> 6) & 0x3f)));
            s.push_back((char)(0x80 | (codepoint & 0x3f)));
        }
    }

    bool parse_hex4(uint32_t& value)
    {
        if (end - cur < 4)
            return fail("incomplete unicode escape");

        value = 0;
        for (int i = 0; i < 4; i++)
        {
            const char ch = *cur++;
            value <<= 4;
            if (ch >= '0' && ch <= '9') value += ch - '0';
            else if (ch >= 'a' && ch <= 'f') value += ch - 'a' + 10;
            else if (ch >= 'A' && ch <= 'F') value += ch - 'A' + 10;
            else return fail("invalid unicode escape");
        }
        return true;
    }

    bool parse_string(std::string& value)
    {
        if (cur == end || *cur != '"')
            return fail("expected string");

        cur++;
        value.clear();
        while (cur != end)
        {
            unsigned char ch = (unsigned char)*cur++;
            if (ch == '"')
                return true;
            if (ch < 0x20)
                return fail("control character in string");
            if (ch != '\\')
            {
                value.push_back((char)ch);
                continue;
            }

            if (cur == end)
                return fail("incomplete string escape");

            const char escape = *cur++;
            if (escape == '"') value.push_back('"');
            else if (escape == '\\') value.push_back('\\');
            else if (escape == '/') value.push_back('/');
            else if (escape == 'b') value.push_back('\b');
            else if (escape == 'f') value.push_back('\f');
            else if (escape == 'n') value.push_back('\n');
            else if (escape == 'r') value.push_back('\r');
            else if (escape == 't') value.push_back('\t');
            else if (escape == 'u')
            {
                uint32_t codepoint = 0;
                if (!parse_hex4(codepoint))
                    return false;

                if (codepoint >= 0xd800 && codepoint <= 0xdbff)
                {
                    if (end - cur < 2 || cur[0] != '\\' || cur[1] != 'u')
                        return fail("missing low unicode surrogate");
                    cur += 2;
                    uint32_t low = 0;
                    if (!parse_hex4(low))
                        return false;
                    if (low < 0xdc00 || low > 0xdfff)
                        return fail("invalid low unicode surrogate");
                    codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                }
                else if (codepoint >= 0xdc00 && codepoint <= 0xdfff)
                {
                    return fail("unexpected low unicode surrogate");
                }

                append_utf8(value, codepoint);
            }
            else
            {
                return fail("invalid string escape");
            }
        }

        return fail("unterminated string");
    }

    bool parse_number(JsonValue& value)
    {
        const char* number_begin = cur;
        bool integer_syntax = true;
        if (*cur == '-')
        {
            cur++;
            if (cur == end)
                return fail("incomplete number");
        }

        if (*cur == '0')
        {
            cur++;
            if (cur != end && *cur >= '0' && *cur <= '9')
                return fail("leading zero in number");
        }
        else
        {
            if (*cur < '1' || *cur > '9')
                return fail("invalid number");
            while (cur != end && *cur >= '0' && *cur <= '9')
                cur++;
        }

        if (cur != end && *cur == '.')
        {
            integer_syntax = false;
            cur++;
            if (cur == end || *cur < '0' || *cur > '9')
                return fail("invalid number fraction");
            while (cur != end && *cur >= '0' && *cur <= '9')
                cur++;
        }

        if (cur != end && (*cur == 'e' || *cur == 'E'))
        {
            integer_syntax = false;
            cur++;
            if (cur != end && (*cur == '+' || *cur == '-'))
                cur++;
            if (cur == end || *cur < '0' || *cur > '9')
                return fail("invalid number exponent");
            while (cur != end && *cur >= '0' && *cur <= '9')
                cur++;
        }

        const std::string text(number_begin, cur);
        value.integer_valid = false;
        if (integer_syntax)
        {
            char* integer_end = 0;
            errno = 0;
            const long long integer = strtoll(text.c_str(), &integer_end, 10);
            if (errno != ERANGE && integer_end == text.c_str() + text.size())
            {
                value.integer_valid = true;
                value.integer = integer;
            }
        }
        char* number_end = 0;
        value.number = strtod(text.c_str(), &number_end);
        if (!number_end || number_end != text.c_str() + text.size() || !std::isfinite(value.number))
            return fail("number is out of range");

        value.type = JsonValue::Number;
        return true;
    }

    bool parse_array(JsonValue& value, int depth)
    {
        cur++;
        value.type = JsonValue::Array;
        value.array.clear();
        skip_space();
        if (cur != end && *cur == ']')
        {
            cur++;
            return true;
        }

        for (;;)
        {
            JsonValue item;
            if (!parse_value(item, depth))
                return false;
            value.array.push_back(item);
            skip_space();
            if (cur == end)
                return fail("unterminated array");
            if (*cur == ']')
            {
                cur++;
                return true;
            }
            if (*cur != ',')
                return fail("expected comma in array");
            cur++;
            skip_space();
        }
    }

    bool parse_object(JsonValue& value, int depth)
    {
        cur++;
        value.type = JsonValue::Object;
        value.object.clear();
        skip_space();
        if (cur != end && *cur == '}')
        {
            cur++;
            return true;
        }

        for (;;)
        {
            std::string key;
            if (!parse_string(key))
                return false;
            if (value.object.find(key) != value.object.end())
                return fail("duplicate object key");
            skip_space();
            if (cur == end || *cur != ':')
                return fail("expected colon in object");
            cur++;
            skip_space();
            JsonValue item;
            if (!parse_value(item, depth))
                return false;
            value.object[key] = item;
            skip_space();
            if (cur == end)
                return fail("unterminated object");
            if (*cur == '}')
            {
                cur++;
                return true;
            }
            if (*cur != ',')
                return fail("expected comma in object");
            cur++;
            skip_space();
        }
    }

private:
    const char* begin;
    const char* cur;
    const char* end;
    std::string message;
};

static bool json_integer(const JsonValue& value, int64_t& result)
{
    if (value.type != JsonValue::Number)
        return false;
    if (value.integer_valid)
    {
        result = value.integer;
        return true;
    }
    if (std::floor(value.number) != value.number)
        return false;
    // INT64_MAX is not exactly representable as double and rounds to 2^63.
    if (value.number < -9223372036854775808.0 || value.number >= 9223372036854775808.0)
        return false;

    result = (int64_t)value.number;
    return true;
}

static bool ends_with_path(const std::string& name, const std::string& suffix)
{
    if (name == suffix)
        return true;
    if (name.size() <= suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;

    return name[name.size() - suffix.size() - 1] == '/';
}

class ExportArchive
{
public:
    bool open(const std::string& path, bool quiet = false)
    {
        if (zip.open(path, quiet) != 0)
            return false;
        names = zip.get_names();
        return true;
    }

    std::string find(const std::string& suffix) const
    {
        for (size_t i = 0; i < names.size(); i++)
        {
            if (ends_with_path(names[i], suffix))
                return names[i];
        }
        return std::string();
    }

    bool read(const std::string& name, std::string& data)
    {
        if (name.empty() || std::find(names.begin(), names.end(), name) == names.end())
            return false;

        const uint64_t size = zip.get_file_size(name);
        if (size > (uint64_t)std::numeric_limits<size_t>::max())
        {
            fprintf(stderr, "pt2 archive member is too large %s\n", name.c_str());
            return false;
        }

        data.resize((size_t)size);
        if (size != 0 && zip.read_file(name, &data[0]) != 0)
            return false;

        return true;
    }

    const std::vector<std::string>& get_names() const
    {
        return names;
    }

private:
    StoreZipReader zip;
    std::vector<std::string> names;
};

struct TensorPayload
{
    std::map<std::string, at::Tensor> named;
    std::vector<at::Tensor> ordered;
};

struct PickleTensor
{
    std::string storage_key;
    int64_t dtype;
    int64_t storage_offset;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
};

struct PickleValue
{
    enum Kind
    {
        Unknown,
        Mark,
        None,
        Boolean,
        Integer,
        String,
        Global,
        Tuple,
        List,
        Dict,
        Storage,
        Tensor
    };

    PickleValue(Kind kind = Unknown)
        : kind(kind), integer(0)
    {
    }

    Kind kind;
    int64_t integer;
    std::string string;
    std::vector<PickleValue> sequence;
    std::map<std::string, PickleValue> dictionary;
    PickleTensor tensor;
};

class StateDictPickleParser
{
public:
    StateDictPickleParser(const char* data, size_t size)
        : cur((const unsigned char*)data), end((const unsigned char*)data + size), next_memo(0)
    {
    }

    bool parse(PickleValue& root, std::string& error)
    {
        while (cur < end)
        {
            const unsigned char opcode = *cur++;
            if (opcode == 0x80) // PROTO
            {
                uint8_t protocol = 0;
                if (!read(protocol) || protocol > 5) return fail(error, "unsupported pickle protocol");
            }
            else if (opcode == 0x95) // FRAME
            {
                uint64_t frame_size = 0;
                if (!read(frame_size) || frame_size > (uint64_t)(end - cur)) return fail(error, "invalid pickle frame");
            }
            else if (opcode == '(') stack.push_back(PickleValue(PickleValue::Mark));
            else if (opcode == 'N') stack.push_back(PickleValue(PickleValue::None));
            else if (opcode == 0x88 || opcode == 0x89)
            {
                PickleValue value(PickleValue::Boolean);
                value.integer = opcode == 0x88;
                stack.push_back(value);
            }
            else if (opcode == 'K')
            {
                uint8_t number = 0;
                if (!read(number)) return fail(error, "truncated BININT1");
                push_integer(number);
            }
            else if (opcode == 'M')
            {
                uint16_t number = 0;
                if (!read(number)) return fail(error, "truncated BININT2");
                push_integer(number);
            }
            else if (opcode == 'J')
            {
                int32_t number = 0;
                if (!read(number)) return fail(error, "truncated BININT");
                push_integer(number);
            }
            else if (opcode == 0x8a) // LONG1
            {
                uint8_t size = 0;
                if (!read(size) || size > 8 || (size_t)(end - cur) < size) return fail(error, "unsupported LONG1");
                uint64_t bits = 0;
                for (size_t i = 0; i < size; i++) bits |= (uint64_t)cur[i] << (i * 8);
                if (size > 0 && size < 8 && (cur[size - 1] & 0x80)) bits |= std::numeric_limits<uint64_t>::max() << (size * 8);
                int64_t number = 0;
                memcpy(&number, &bits, sizeof(number));
                cur += size;
                push_integer(number);
            }
            else if (opcode == 'X' || opcode == 'T' || opcode == 'B')
            {
                uint32_t size = 0;
                if (!read(size) || (size_t)(end - cur) < size) return fail(error, "truncated pickle string");
                push_string((const char*)cur, size);
                cur += size;
            }
            else if (opcode == 0x8c || opcode == 'U' || opcode == 'C')
            {
                uint8_t size = 0;
                if (!read(size) || (size_t)(end - cur) < size) return fail(error, "truncated short pickle string");
                push_string((const char*)cur, size);
                cur += size;
            }
            else if (opcode == 0x8d || opcode == 0x8e)
            {
                uint64_t size = 0;
                if (!read(size) || size > (uint64_t)(end - cur) || size > (uint64_t)std::numeric_limits<size_t>::max()) return fail(error, "truncated long pickle string");
                push_string((const char*)cur, (size_t)size);
                cur += (size_t)size;
            }
            else if (opcode == 'c')
            {
                std::string module;
                std::string name;
                if (!read_line(module) || !read_line(name)) return fail(error, "truncated GLOBAL");
                PickleValue value(PickleValue::Global);
                value.string = module + "." + name;
                stack.push_back(value);
            }
            else if (opcode == 0x93) // STACK_GLOBAL
            {
                if (stack.size() < 2 || stack.back().kind != PickleValue::String || stack[stack.size() - 2].kind != PickleValue::String) return fail(error, "invalid STACK_GLOBAL");
                PickleValue value(PickleValue::Global);
                value.string = stack[stack.size() - 2].string + "." + stack.back().string;
                stack.resize(stack.size() - 2);
                stack.push_back(value);
            }
            else if (opcode == ')') stack.push_back(PickleValue(PickleValue::Tuple));
            else if (opcode == ']') stack.push_back(PickleValue(PickleValue::List));
            else if (opcode == '}') stack.push_back(PickleValue(PickleValue::Dict));
            else if (opcode == 't' || opcode == 'l' || opcode == 'd')
            {
                const size_t mark = find_mark();
                if (mark == (size_t)-1) return fail(error, "pickle MARK is missing");
                PickleValue value(opcode == 't' ? PickleValue::Tuple : (opcode == 'l' ? PickleValue::List : PickleValue::Dict));
                if (value.kind == PickleValue::Dict)
                {
                    if ((stack.size() - mark - 1) % 2 != 0) return fail(error, "invalid DICT");
                    for (size_t i = mark + 1; i < stack.size(); i += 2)
                    {
                        if (stack[i].kind != PickleValue::String) return fail(error, "non-string state dict key");
                        value.dictionary[stack[i].string] = stack[i + 1];
                    }
                }
                else
                {
                    value.sequence.assign(stack.begin() + mark + 1, stack.end());
                }
                stack.resize(mark);
                stack.push_back(value);
            }
            else if (opcode == 0x85 || opcode == 0x86 || opcode == 0x87)
            {
                const size_t count = opcode - 0x84;
                if (stack.size() < count) return fail(error, "invalid short TUPLE");
                PickleValue value(PickleValue::Tuple);
                value.sequence.assign(stack.end() - count, stack.end());
                stack.resize(stack.size() - count);
                stack.push_back(value);
            }
            else if (opcode == 'a')
            {
                if (stack.size() < 2 || stack[stack.size() - 2].kind != PickleValue::List) return fail(error, "invalid APPEND");
                PickleValue item = stack.back();
                stack.pop_back();
                stack.back().sequence.push_back(item);
            }
            else if (opcode == 'e')
            {
                const size_t mark = find_mark();
                if (mark == (size_t)-1 || mark == 0 || stack[mark - 1].kind != PickleValue::List) return fail(error, "invalid APPENDS");
                stack[mark - 1].sequence.insert(stack[mark - 1].sequence.end(), stack.begin() + mark + 1, stack.end());
                stack.resize(mark);
            }
            else if (opcode == 's')
            {
                if (stack.size() < 3 || stack[stack.size() - 3].kind != PickleValue::Dict || stack[stack.size() - 2].kind != PickleValue::String) return fail(error, "invalid SETITEM");
                const std::string key = stack[stack.size() - 2].string;
                const PickleValue value = stack.back();
                stack.resize(stack.size() - 2);
                stack.back().dictionary[key] = value;
            }
            else if (opcode == 'u')
            {
                const size_t mark = find_mark();
                if (mark == (size_t)-1 || mark == 0 || stack[mark - 1].kind != PickleValue::Dict || (stack.size() - mark - 1) % 2 != 0) return fail(error, "invalid SETITEMS");
                for (size_t i = mark + 1; i < stack.size(); i += 2)
                {
                    if (stack[i].kind != PickleValue::String) return fail(error, "non-string state dict key");
                    stack[mark - 1].dictionary[stack[i].string] = stack[i + 1];
                }
                stack.resize(mark);
            }
            else if (opcode == 'Q')
            {
                if (stack.empty() || !make_storage(stack.back())) return fail(error, "invalid storage persistent id");
            }
            else if (opcode == 'R')
            {
                if (stack.size() < 2) return fail(error, "invalid REDUCE");
                const PickleValue arguments = stack.back();
                const PickleValue callable = stack[stack.size() - 2];
                stack.resize(stack.size() - 2);
                PickleValue result;
                if (!reduce(callable, arguments, result)) return fail(error, "unsupported pickle reducer");
                stack.push_back(result);
            }
            else if (opcode == 'b')
            {
                if (stack.size() < 2) return fail(error, "invalid BUILD");
                stack.pop_back();
            }
            else if (opcode == 'q')
            {
                uint8_t index = 0;
                if (!read(index) || stack.empty()) return fail(error, "invalid BINPUT");
                memo[index] = stack.back();
                next_memo = std::max(next_memo, (size_t)index + 1);
            }
            else if (opcode == 'r')
            {
                uint32_t index = 0;
                if (!read(index) || stack.empty()) return fail(error, "invalid LONG_BINPUT");
                memo[index] = stack.back();
                next_memo = std::max(next_memo, (size_t)index + 1);
            }
            else if (opcode == 0x94)
            {
                if (stack.empty()) return fail(error, "invalid MEMOIZE");
                memo[next_memo++] = stack.back();
            }
            else if (opcode == 'h')
            {
                uint8_t index = 0;
                if (!read(index) || memo.find(index) == memo.end()) return fail(error, "invalid BINGET");
                stack.push_back(memo[index]);
            }
            else if (opcode == 'j')
            {
                uint32_t index = 0;
                if (!read(index) || memo.find(index) == memo.end()) return fail(error, "invalid LONG_BINGET");
                stack.push_back(memo[index]);
            }
            else if (opcode == '.')
            {
                if (stack.empty()) return fail(error, "empty pickle result");
                root = stack.back();
                return true;
            }
            else
            {
                std::ostringstream ss;
                ss << "unsupported pickle opcode 0x" << std::hex << (int)opcode;
                return fail(error, ss.str());
            }
        }

        return fail(error, "pickle STOP is missing");
    }

private:
    template<typename T>
    bool read(T& value)
    {
        if ((size_t)(end - cur) < sizeof(T)) return false;
        memcpy(&value, cur, sizeof(T));
        cur += sizeof(T);
        return true;
    }

    bool read_line(std::string& line)
    {
        const unsigned char* start = cur;
        while (cur < end && *cur != '\n') cur++;
        if (cur == end) return false;
        line.assign((const char*)start, (const char*)cur);
        cur++;
        return true;
    }

    bool fail(std::string& error, const std::string& text)
    {
        error = text;
        return false;
    }

    void push_integer(int64_t number)
    {
        PickleValue value(PickleValue::Integer);
        value.integer = number;
        stack.push_back(value);
    }

    void push_string(const char* data, size_t size)
    {
        PickleValue value(PickleValue::String);
        value.string.assign(data, size);
        stack.push_back(value);
    }

    size_t find_mark() const
    {
        for (size_t i = stack.size(); i > 0; i--)
            if (stack[i - 1].kind == PickleValue::Mark) return i - 1;
        return (size_t)-1;
    }

    static int64_t storage_dtype(const std::string& name)
    {
        if (name == "torch.ByteStorage") return 1;
        if (name == "torch.CharStorage") return 2;
        if (name == "torch.ShortStorage") return 3;
        if (name == "torch.IntStorage") return 4;
        if (name == "torch.LongStorage") return 5;
        if (name == "torch.HalfStorage") return 6;
        if (name == "torch.FloatStorage") return 7;
        if (name == "torch.DoubleStorage") return 8;
        if (name == "torch.ComplexHalfStorage") return 9;
        if (name == "torch.ComplexFloatStorage") return 10;
        if (name == "torch.ComplexDoubleStorage") return 11;
        if (name == "torch.BoolStorage") return 12;
        if (name == "torch.BFloat16Storage") return 13;
        return 0;
    }

    static bool sequence_ints(const PickleValue& value, std::vector<int64_t>& values)
    {
        if (value.kind != PickleValue::Tuple && value.kind != PickleValue::List) return false;
        values.clear();
        for (size_t i = 0; i < value.sequence.size(); i++)
        {
            if (value.sequence[i].kind != PickleValue::Integer) return false;
            values.push_back(value.sequence[i].integer);
        }
        return true;
    }

    static bool make_storage(PickleValue& value)
    {
        if (value.kind != PickleValue::Tuple || value.sequence.size() < 5 || value.sequence[0].kind != PickleValue::String || value.sequence[0].string != "storage" || value.sequence[1].kind != PickleValue::Global || value.sequence[2].kind != PickleValue::String)
            return false;
        const int64_t dtype = storage_dtype(value.sequence[1].string);
        if (dtype == 0) return false;
        PickleValue storage(PickleValue::Storage);
        storage.tensor.storage_key = value.sequence[2].string;
        storage.tensor.dtype = dtype;
        value = storage;
        return true;
    }

    static bool reduce(const PickleValue& callable, const PickleValue& arguments, PickleValue& result)
    {
        if (callable.kind != PickleValue::Global || arguments.kind != PickleValue::Tuple) return false;
        if (callable.string == "collections.OrderedDict")
        {
            result = PickleValue(PickleValue::Dict);
            if (!arguments.sequence.empty() && arguments.sequence[0].kind == PickleValue::List)
            {
                for (const PickleValue& item : arguments.sequence[0].sequence)
                {
                    if (item.kind != PickleValue::Tuple || item.sequence.size() != 2 || item.sequence[0].kind != PickleValue::String) return false;
                    result.dictionary[item.sequence[0].string] = item.sequence[1];
                }
            }
            return true;
        }
        if (callable.string == "torch._utils._rebuild_parameter" || callable.string == "torch._utils._rebuild_parameter_with_state")
        {
            if (arguments.sequence.empty() || arguments.sequence[0].kind != PickleValue::Tensor) return false;
            result = arguments.sequence[0];
            return true;
        }
        if (callable.string == "torch._utils._rebuild_tensor" || callable.string == "torch._utils._rebuild_tensor_v2" || callable.string == "torch._utils._rebuild_tensor_v3")
        {
            if (arguments.sequence.size() < 4 || arguments.sequence[0].kind != PickleValue::Storage || arguments.sequence[1].kind != PickleValue::Integer) return false;
            result = PickleValue(PickleValue::Tensor);
            result.tensor = arguments.sequence[0].tensor;
            result.tensor.storage_offset = arguments.sequence[1].integer;
            return sequence_ints(arguments.sequence[2], result.tensor.sizes) && sequence_ints(arguments.sequence[3], result.tensor.strides);
        }
        return false;
    }

private:
    const unsigned char* cur;
    const unsigned char* end;
    std::vector<PickleValue> stack;
    std::map<size_t, PickleValue> memo;
    size_t next_memo;
};

static bool read_torch_save(const std::string& data, TensorPayload& payload);

static bool serialized_dtype(int64_t value, at::ScalarType& dtype, int& pnnx_type, size_t& elemsize)
{
    if (value == 1) { dtype = at::kByte; pnnx_type = 8; elemsize = 1; return true; }
    if (value == 2) { dtype = at::kChar; pnnx_type = 7; elemsize = 1; return true; }
    if (value == 3) { dtype = at::kShort; pnnx_type = 6; elemsize = 2; return true; }
    if (value == 4) { dtype = at::kInt; pnnx_type = 4; elemsize = 4; return true; }
    if (value == 5) { dtype = at::kLong; pnnx_type = 5; elemsize = 8; return true; }
    if (value == 6) { dtype = at::kHalf; pnnx_type = 3; elemsize = 2; return true; }
    if (value == 7) { dtype = at::kFloat; pnnx_type = 1; elemsize = 4; return true; }
    if (value == 8) { dtype = at::kDouble; pnnx_type = 2; elemsize = 8; return true; }
    if (value == 9) { dtype = at::kComplexHalf; pnnx_type = 12; elemsize = 4; return true; }
    if (value == 10) { dtype = at::kComplexFloat; pnnx_type = 10; elemsize = 8; return true; }
    if (value == 11) { dtype = at::kComplexDouble; pnnx_type = 11; elemsize = 16; return true; }
    if (value == 12) { dtype = at::kBool; pnnx_type = 9; elemsize = 1; return true; }
    if (value == 13) { dtype = at::kBFloat16; pnnx_type = 13; elemsize = 2; return true; }

    return false;
}

static bool argument_int(const JsonValue& value, int64_t& result)
{
    if (json_integer(value, result))
        return true;
    if (value.type != JsonValue::Object)
        return false;

    const JsonValue* as_int = value.find("as_int");
    if (as_int && json_integer(*as_int, result))
        return true;

    const JsonValue* as_sym_int = value.find("as_sym_int");
    if (as_sym_int)
        return argument_int(*as_sym_int, result);

    return false;
}

static bool parse_int_array(const JsonValue& value, std::vector<int64_t>& result, bool allow_symbolic)
{
    if (value.type != JsonValue::Array)
        return false;

    result.clear();
    for (size_t i = 0; i < value.array.size(); i++)
    {
        int64_t number = 0;
        if (!argument_int(value.array[i], number))
        {
            if (!allow_symbolic)
                return false;
            number = -1;
        }
        result.push_back(number);
    }

    return true;
}

struct TensorMeta
{
    at::ScalarType dtype;
    int pnnx_type;
    size_t elemsize;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    int64_t storage_offset;
};

static bool parse_tensor_meta(const JsonValue& value, TensorMeta& meta, bool allow_symbolic)
{
    if (value.type != JsonValue::Object)
        return false;

    const JsonValue* dtype = value.find("dtype");
    const JsonValue* sizes = value.find("sizes");
    const JsonValue* strides = value.find("strides");
    const JsonValue* storage_offset = value.find("storage_offset");
    const JsonValue* layout = value.find("layout");
    int64_t dtype_number = 0;
    int64_t layout_number = 7;
    if (!dtype || !json_integer(*dtype, dtype_number) || !serialized_dtype(dtype_number, meta.dtype, meta.pnnx_type, meta.elemsize))
        return false;
    if (!sizes || !parse_int_array(*sizes, meta.sizes, allow_symbolic))
        return false;
    if (strides && !parse_int_array(*strides, meta.strides, allow_symbolic))
        return false;
    if (meta.strides.empty())
    {
        meta.strides.resize(meta.sizes.size());
        int64_t stride = 1;
        for (int i = (int)meta.sizes.size() - 1; i >= 0; i--)
        {
            meta.strides[i] = stride;
            if (meta.sizes[i] > 0)
            {
                if (stride > std::numeric_limits<int64_t>::max() / meta.sizes[i])
                    return false;
                stride *= meta.sizes[i];
            }
        }
    }
    meta.storage_offset = 0;
    if (storage_offset && !argument_int(*storage_offset, meta.storage_offset))
        return false;
    if (layout && (!json_integer(*layout, layout_number) || layout_number != 7))
        return false;

    return meta.sizes.size() == meta.strides.size();
}

static bool host_is_little_endian()
{
    const uint16_t value = 1;
    return *(const unsigned char*)&value == 1;
}

static bool tensor_from_raw(const std::string& data, const TensorMeta& meta, const std::string& byteorder, at::Tensor& tensor)
{
    if (byteorder != "little" && byteorder != "big")
    {
        fprintf(stderr, "invalid pt2 tensor byte order %s\n", byteorder.c_str());
        return false;
    }
    if ((byteorder == "little") != host_is_little_endian() && meta.elemsize != 1)
    {
        fprintf(stderr, "pt2 tensor byte order %s is not supported on this platform\n", byteorder.c_str());
        return false;
    }
    if (meta.elemsize == 0 || data.size() % meta.elemsize != 0 || meta.storage_offset < 0)
        return false;

    const size_t storage_elements = data.size() / meta.elemsize;
    if (storage_elements > (size_t)std::numeric_limits<int64_t>::max())
        return false;
    uint64_t required_elements = (uint64_t)meta.storage_offset;
    bool empty_tensor = false;
    for (size_t i = 0; i < meta.sizes.size(); i++)
    {
        if (meta.sizes[i] < 0 || meta.strides[i] < 0)
            return false;
        if (meta.sizes[i] == 0)
        {
            empty_tensor = true;
            break;
        }
        const uint64_t extent = (uint64_t)(meta.sizes[i] - 1) * (uint64_t)meta.strides[i];
        if (meta.strides[i] != 0 && extent / (uint64_t)meta.strides[i] != (uint64_t)(meta.sizes[i] - 1))
            return false;
        if (extent > std::numeric_limits<uint64_t>::max() - required_elements)
            return false;
        required_elements += extent;
    }
    if (empty_tensor)
        required_elements = 0;
    else
        required_elements++;
    if (required_elements > storage_elements)
        return false;

    at::Tensor storage = torch::empty({(int64_t)storage_elements}, torch::TensorOptions().dtype(meta.dtype).device(torch::kCPU));
    if (!data.empty())
        memcpy(storage.data_ptr(), data.data(), data.size());
    tensor = storage.as_strided(meta.sizes, meta.strides, meta.storage_offset).clone();
    return true;
}

static void collect_pickle_tensors(const PickleValue& value, std::vector<std::pair<std::string, PickleTensor> >& tensors, const std::string& name = std::string())
{
    if (value.kind == PickleValue::Tensor)
    {
        tensors.push_back(std::make_pair(name, value.tensor));
        return;
    }

    if (value.kind == PickleValue::Dict)
    {
        for (std::map<std::string, PickleValue>::const_iterator it = value.dictionary.begin(); it != value.dictionary.end(); ++it)
            collect_pickle_tensors(it->second, tensors, it->first);
        return;
    }

    if (value.kind == PickleValue::Tuple || value.kind == PickleValue::List)
    {
        for (size_t i = 0; i < value.sequence.size(); i++)
            collect_pickle_tensors(value.sequence[i], tensors);
    }
}

static bool read_torch_save(const std::string& data, TensorPayload& payload)
{
    try
    {
        std::istringstream stream(data, std::ios::in | std::ios::binary);
        caffe2::serialize::PyTorchStreamReader reader(&stream);
        if (!reader.hasRecord("data.pkl"))
        {
            fprintf(stderr, "pt2 tensor payload has no data.pkl\n");
            return false;
        }

        std::tuple<at::DataPtr, size_t> pickle_record = reader.getRecord("data.pkl");
        PickleValue root;
        std::string error;
        StateDictPickleParser parser((const char*)std::get<0>(pickle_record).get(), std::get<1>(pickle_record));
        if (!parser.parse(root, error))
        {
            fprintf(stderr, "parse pt2 tensor pickle failed: %s\n", error.c_str());
            return false;
        }

        std::string byteorder = "little";
        if (reader.hasRecord("byteorder"))
        {
            std::tuple<at::DataPtr, size_t> byteorder_record = reader.getRecord("byteorder");
            byteorder.assign((const char*)std::get<0>(byteorder_record).get(), std::get<1>(byteorder_record));
        }

        std::vector<std::pair<std::string, PickleTensor> > tensors;
        collect_pickle_tensors(root, tensors);
        for (size_t i = 0; i < tensors.size(); i++)
        {
            const PickleTensor& descriptor = tensors[i].second;
            TensorMeta meta;
            if (!serialized_dtype(descriptor.dtype, meta.dtype, meta.pnnx_type, meta.elemsize))
            {
                fprintf(stderr, "unsupported pt2 pickle tensor dtype %lld\n", (long long)descriptor.dtype);
                return false;
            }
            meta.storage_offset = descriptor.storage_offset;
            meta.sizes = descriptor.sizes;
            meta.strides = descriptor.strides;

            const std::string record_name = "data/" + descriptor.storage_key;
            if (!reader.hasRecord(record_name))
            {
                fprintf(stderr, "pt2 tensor storage %s is missing\n", descriptor.storage_key.c_str());
                return false;
            }
            std::tuple<at::DataPtr, size_t> storage_record = reader.getRecord(record_name);
            std::string storage((const char*)std::get<0>(storage_record).get(), std::get<1>(storage_record));
            at::Tensor tensor;
            if (!tensor_from_raw(storage, meta, byteorder, tensor))
            {
                fprintf(stderr, "load pt2 tensor storage %s failed\n", descriptor.storage_key.c_str());
                return false;
            }

            payload.ordered.push_back(tensor);
            if (!tensors[i].first.empty())
                payload.named[tensors[i].first] = tensor;
        }
        return true;
    }
    catch (const c10::Error& e)
    {
        fprintf(stderr, "load pt2 tensor payload failed: %s\n", e.what());
        return false;
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "load pt2 tensor payload failed: %s\n", e.what());
        return false;
    }
}

static bool parse_json(const std::string& data, JsonValue& root, const char* what)
{
    std::string error;
    JsonParser parser(data);
    if (!parser.parse(root, error))
    {
        fprintf(stderr, "parse %s failed: %s\n", what, error.c_str());
        return false;
    }
    return true;
}

static bool load_nested_payload(ExportArchive& archive, const std::string& name, TensorPayload& payload)
{
    if (name.empty())
        return true;

    std::string data;
    if (!archive.read(name, data))
        return false;

    return read_torch_save(data, payload);
}

static bool load_payload_config(ExportArchive& archive, const std::string& config_name, const std::string& byteorder, TensorPayload& payload)
{
    if (config_name.empty())
        return true;

    std::string config_data;
    if (!archive.read(config_name, config_data))
        return false;

    JsonValue config_root;
    if (!parse_json(config_data, config_root, "pt2 tensor config"))
        return false;
    const JsonValue* config = config_root.find("config");
    if (!config || config->type != JsonValue::Object)
    {
        fprintf(stderr, "pt2 tensor config has no config object\n");
        return false;
    }

    const size_t slash = config_name.find_last_of('/');
    const std::string directory = slash == std::string::npos ? std::string() : config_name.substr(0, slash + 1);
    std::map<std::string, std::string> file_cache;

    for (std::map<std::string, JsonValue>::const_iterator it = config->object.begin(); it != config->object.end(); ++it)
    {
        const std::string& tensor_name = it->first;
        const JsonValue& entry = it->second;
        const JsonValue* path_name = entry.find("path_name");
        const JsonValue* use_pickle = entry.find("use_pickle");
        const JsonValue* tensor_meta = entry.find("tensor_meta");
        if (!path_name || path_name->type != JsonValue::String || !use_pickle || use_pickle->type != JsonValue::Bool)
        {
            fprintf(stderr, "invalid pt2 tensor config entry %s\n", tensor_name.c_str());
            return false;
        }

        const std::string member_name = directory + path_name->string;
        std::string& tensor_data = file_cache[member_name];
        if (tensor_data.empty() && !archive.read(member_name, tensor_data))
            return false;

        at::Tensor tensor;
        if (use_pickle->boolean)
        {
            TensorPayload pickled;
            if (!read_torch_save(tensor_data, pickled) || pickled.ordered.empty())
            {
                fprintf(stderr, "invalid pickled pt2 tensor %s\n", tensor_name.c_str());
                return false;
            }
            tensor = pickled.ordered[0];
        }
        else
        {
            TensorMeta meta;
            if (!tensor_meta || !parse_tensor_meta(*tensor_meta, meta, false) || !tensor_from_raw(tensor_data, meta, byteorder, tensor))
            {
                fprintf(stderr, "invalid raw pt2 tensor %s\n", tensor_name.c_str());
                return false;
            }
        }

        payload.named[tensor_name] = tensor;
        payload.ordered.push_back(tensor);
    }

    return true;
}

static const at::Tensor* find_tensor(const TensorPayload& payload, const std::string& name)
{
    std::map<std::string, at::Tensor>::const_iterator it = payload.named.find(name);
    if (it != payload.named.end())
        return &it->second;
    if (payload.named.empty() && payload.ordered.size() == 1)
        return &payload.ordered[0];

    return 0;
}

static int input_type_to_pnnx_type(const std::string& type)
{
    if (type == "f32") return 1;
    if (type == "f64") return 2;
    if (type == "f16") return 3;
    if (type == "i32") return 4;
    if (type == "i64") return 5;
    if (type == "i16") return 6;
    if (type == "i8") return 7;
    if (type == "u8") return 8;
    if (type == "bool") return 9;
    if (type == "c64") return 10;
    if (type == "c128") return 11;
    if (type == "c32") return 12;
    if (type == "bf16") return 13;
    return 0;
}

static bool set_operand_tensor_meta(const JsonValue& tensor_values, const std::string& name, Operand* operand)
{
    const JsonValue* value = tensor_values.find(name);
    if (!value)
        return false;

    TensorMeta meta;
    if (!parse_tensor_meta(*value, meta, true))
        return false;

    operand->type = meta.pnnx_type;
    operand->shape.clear();
    for (size_t i = 0; i < meta.sizes.size(); i++)
    {
        if (meta.sizes[i] < 0 || meta.sizes[i] > INT_MAX)
            operand->shape.push_back(-1);
        else
            operand->shape.push_back((int)meta.sizes[i]);
    }
    return true;
}

static bool parameter_from_ivalue(const torch::jit::IValue& value, Parameter& parameter)
{
    if (value.isNone()) { parameter = Parameter(); return true; }
    if (value.isBool()) { parameter = Parameter(value.toBool()); return true; }
    if (value.isInt()) { parameter = Parameter((long long)value.toInt()); return true; }
    if (value.isDouble())
    {
        const double number = value.toDouble();
        if (!std::isfinite(number)) return false;
        parameter = Parameter(number);
        return true;
    }
    if (value.isString()) { parameter = Parameter(value.toStringRef()); return true; }
    if (value.isIntList()) { parameter = Parameter(value.toIntVector()); return true; }
    if (value.isDoubleList())
    {
        const std::vector<double> values = value.toDoubleVector();
        for (size_t i = 0; i < values.size(); i++) if (!std::isfinite(values[i])) return false;
        parameter = Parameter(values);
        return true;
    }
    if (value.isBoolList())
    {
        std::vector<int> values;
        for (bool item : value.toBoolList()) values.push_back(item ? 1 : 0);
        parameter = Parameter(values);
        return true;
    }
    if (value.isDevice()) { parameter = Parameter(value.toDevice().str()); return true; }
    if (value.isScalar())
    {
        const at::Scalar scalar = value.toScalar();
        if (scalar.isIntegral(false)) parameter = Parameter((long long)scalar.toLong());
        else if (scalar.isFloatingPoint())
        {
            const double number = scalar.toDouble();
            if (!std::isfinite(number)) return false;
            parameter = Parameter(number);
        }
        else return false;
        return true;
    }
    return false;
}

static std::string serialized_dtype_string(int64_t value)
{
    if (value == 1) return "torch.uint8";
    if (value == 2) return "torch.int8";
    if (value == 3) return "torch.int16";
    if (value == 4) return "torch.int32";
    if (value == 5) return "torch.int64";
    if (value == 6) return "torch.float16";
    if (value == 7) return "torch.float32";
    if (value == 8) return "torch.float64";
    if (value == 9) return "torch.complex32";
    if (value == 10) return "torch.complex64";
    if (value == 11) return "torch.complex128";
    if (value == 12) return "torch.bool";
    if (value == 13) return "torch.bfloat16";
    return std::string();
}

static bool parameter_from_json(const JsonValue& argument, Parameter& parameter)
{
    if (argument.type != JsonValue::Object)
        return false;

    if (argument.find("as_none")) { parameter = Parameter(); return true; }

    const JsonValue* value = argument.find("as_bool");
    if (value && value->type == JsonValue::Bool) { parameter = Parameter(value->boolean); return true; }

    value = argument.find("as_int");
    int64_t integer = 0;
    if (value && json_integer(*value, integer)) { parameter = Parameter((long long)integer); return true; }

    value = argument.find("as_float");
    if (value && value->type == JsonValue::Number && std::isfinite(value->number)) { parameter = Parameter(value->number); return true; }

    value = argument.find("as_string");
    if (value && value->type == JsonValue::String) { parameter = Parameter(value->string); return true; }

    value = argument.find("as_ints");
    if (value)
    {
        std::vector<int64_t> values;
        if (!parse_int_array(*value, values, false)) return false;
        parameter = Parameter(values);
        return true;
    }

    value = argument.find("as_floats");
    if (value && value->type == JsonValue::Array)
    {
        std::vector<double> values;
        for (size_t i = 0; i < value->array.size(); i++)
        {
            if (value->array[i].type != JsonValue::Number || !std::isfinite(value->array[i].number)) return false;
            values.push_back(value->array[i].number);
        }
        parameter = Parameter(values);
        return true;
    }

    value = argument.find("as_strings");
    if (value && value->type == JsonValue::Array)
    {
        std::vector<std::string> values;
        for (size_t i = 0; i < value->array.size(); i++)
        {
            if (value->array[i].type != JsonValue::String) return false;
            values.push_back(value->array[i].string);
        }
        parameter = Parameter(values);
        return true;
    }

    value = argument.find("as_bools");
    if (value && value->type == JsonValue::Array)
    {
        std::vector<int> values;
        for (size_t i = 0; i < value->array.size(); i++)
        {
            if (value->array[i].type != JsonValue::Bool) return false;
            values.push_back(value->array[i].boolean ? 1 : 0);
        }
        parameter = Parameter(values);
        return true;
    }

    value = argument.find("as_sym_int");
    if (value) return parameter_from_json(*value, parameter);

    value = argument.find("as_sym_bool");
    if (value) return parameter_from_json(*value, parameter);

    value = argument.find("as_sym_float");
    if (value) return parameter_from_json(*value, parameter);

    value = argument.find("as_scalar_type");
    if (value && json_integer(*value, integer))
    {
        const std::string dtype = serialized_dtype_string(integer);
        if (dtype.empty()) return false;
        parameter = Parameter(dtype);
        return true;
    }

    value = argument.find("as_device");
    if (value && value->type == JsonValue::Object)
    {
        const JsonValue* type = value->find("type");
        const JsonValue* index = value->find("index");
        if (!type || type->type != JsonValue::String) return false;
        std::string device = type->string;
        if (index && index->type == JsonValue::Number && json_integer(*index, integer)) device += ":" + std::to_string(integer);
        parameter = Parameter(device);
        return true;
    }

    value = argument.find("as_layout");
    if (value && json_integer(*value, integer))
    {
        if (integer != 7) return false;
        parameter = Parameter("torch.strided");
        return true;
    }

    value = argument.find("as_memory_format");
    if (value && json_integer(*value, integer))
    {
        if (integer == 1) parameter = Parameter("torch.contiguous_format");
        else if (integer == 2) parameter = Parameter("torch.channels_last");
        else if (integer == 3) parameter = Parameter("torch.channels_last_3d");
        else if (integer == 4) parameter = Parameter("torch.preserve_format");
        else return false;
        return true;
    }

    value = argument.find("as_operator");
    if (value && value->type == JsonValue::String) { parameter = Parameter(value->string); return true; }

    value = argument.find("as_complex");
    if (value && value->type == JsonValue::Object)
    {
        const JsonValue* real = value->find("real");
        const JsonValue* imag = value->find("imag");
        if (!real || real->type != JsonValue::Number || !std::isfinite(real->number) || !imag || imag->type != JsonValue::Number || !std::isfinite(imag->number)) return false;
        parameter = Parameter(std::complex<double>(real->number, imag->number));
        return true;
    }

    return false;
}

static std::string argument_name(const JsonValue& argument)
{
    if (argument.type != JsonValue::Object)
        return std::string();

    const JsonValue* tensor = argument.find("as_tensor");
    if (!tensor) tensor = &argument;
    if (tensor->type == JsonValue::Object)
    {
        const JsonValue* name = tensor->find("name");
        if (name && name->type == JsonValue::String)
            return name->string;
        const JsonValue* as_name = tensor->find("as_name");
        if (as_name && as_name->type == JsonValue::String)
            return as_name->string;
    }

    const JsonValue* sym_int = argument.find("as_sym_int");
    if (sym_int) return argument_name(*sym_int);
    const JsonValue* sym_bool = argument.find("as_sym_bool");
    if (sym_bool) return argument_name(*sym_bool);
    const JsonValue* sym_float = argument.find("as_sym_float");
    if (sym_float) return argument_name(*sym_float);
    return std::string();
}

static bool parse_target(const std::string& target, std::string& schema_name, std::string& overload, std::string& operator_type)
{
    if (target == "_operator.eq" || target == "_operator.ne" || target == "_operator.lt" || target == "_operator.le" || target == "_operator.gt" || target == "_operator.ge")
    {
        schema_name.clear();
        overload.clear();
        operator_type = "aten::" + target.substr(10);
        return true;
    }

    const std::string prefix = "torch.ops.";
    if (target.compare(0, prefix.size(), prefix) != 0)
        return false;

    const std::string qualified = target.substr(prefix.size());
    const size_t namespace_dot = qualified.find('.');
    const size_t overload_dot = qualified.find('.', namespace_dot == std::string::npos ? 0 : namespace_dot + 1);
    if (namespace_dot == std::string::npos || overload_dot == std::string::npos)
        return false;

    schema_name = qualified.substr(0, namespace_dot) + "::" + qualified.substr(namespace_dot + 1, overload_dot - namespace_dot - 1);
    overload = qualified.substr(overload_dot + 1);
    if (overload == "default") overload.clear();
    operator_type = schema_name;
    return true;
}

class ExportGraphBuilder
{
public:
    ExportGraphBuilder(Graph& graph, const JsonValue& tensor_values)
        : graph(graph), tensor_values(tensor_values), index(0)
    {
    }

    Operand* add_input(const std::string& name)
    {
        Operator* op = graph.new_operator("pnnx.Input", next_name("pnnx_input"));
        Operand* operand = graph.new_operand(name);
        operand->producer = op;
        op->outputs.push_back(operand);
        set_operand_tensor_meta(tensor_values, name, operand);
        values[name] = operand;
        return operand;
    }

    Operand* add_attribute(const std::string& value_name, const std::string& attribute_name, const at::Tensor& tensor)
    {
        Operator* op = graph.new_operator("pnnx.Attribute", attribute_name);
        op->attrs["data"] = tensor;
        Operand* operand = graph.new_operand(value_name);
        operand->producer = op;
        operand->type = op->attrs["data"].type;
        operand->shape = op->attrs["data"].shape;
        op->outputs.push_back(operand);
        values[value_name] = operand;
        return operand;
    }

    Operand* add_constant(const Parameter& parameter)
    {
        const std::string name = next_name("pnnx");
        Operator* op = graph.new_operator("prim::Constant", name);
        op->params["value"] = parameter;
        Operand* operand = graph.new_operand(name);
        operand->producer = op;
        op->outputs.push_back(operand);
        return operand;
    }

    Operand* argument_operand(const JsonValue& argument)
    {
        const std::string name = argument_name(argument);
        if (!name.empty())
        {
            std::map<std::string, Operand*>::const_iterator it = values.find(name);
            if (it == values.end())
            {
                fprintf(stderr, "pt2 graph references unknown value %s\n", name.c_str());
                return 0;
            }
            return it->second;
        }

        Parameter parameter;
        if (parameter_from_json(argument, parameter))
            return add_constant(parameter);

        const JsonValue* optional_tensor = argument.find("as_optional_tensor");
        if (optional_tensor)
            return argument_operand(*optional_tensor);

        const JsonValue* tensors = argument.find("as_tensors");
        if (!tensors) tensors = argument.find("as_optional_tensors");
        if (!tensors) tensors = argument.find("as_sym_ints");
        if (!tensors) tensors = argument.find("as_sym_bools");
        if (!tensors) tensors = argument.find("as_sym_floats");
        if (tensors)
            return list_operand(*tensors);

        const JsonValue* nested = argument.find("as_nested_tensors");
        const char* primitive_key = 0;
        if (!nested) { nested = argument.find("as_int_lists"); primitive_key = "as_int"; }
        if (!nested) { nested = argument.find("as_float_lists"); primitive_key = "as_float"; }
        if (nested && nested->type == JsonValue::Array)
        {
            std::vector<Operand*> items;
            for (size_t i = 0; i < nested->array.size(); i++)
            {
                Operand* input = list_operand(nested->array[i], primitive_key);
                if (!input) return 0;
                items.push_back(input);
            }
            return list_operand(items);
        }

        const JsonValue* dictionary = argument.find("as_string_to_argument");
        if (dictionary && dictionary->type == JsonValue::Object)
        {
            std::vector<Operand*> items;
            for (std::map<std::string, JsonValue>::const_iterator it = dictionary->object.begin(); it != dictionary->object.end(); ++it)
            {
                items.push_back(add_constant(Parameter(it->first)));
                Operand* value = argument_operand(it->second);
                if (!value) return 0;
                items.push_back(value);
            }
            return container_operand("prim::DictConstruct", items);
        }

        fprintf(stderr, "unsupported pt2 argument\n");
        return 0;
    }

    bool add_node(const JsonValue& node)
    {
        const JsonValue* target = node.find("target");
        const JsonValue* inputs = node.find("inputs");
        const JsonValue* outputs = node.find("outputs");
        if (!target || target->type != JsonValue::String || !inputs || inputs->type != JsonValue::Array || !outputs || outputs->type != JsonValue::Array)
            return false;

        std::string schema_name;
        std::string overload;
        std::string operator_type;
        if (!parse_target(target->string, schema_name, overload, operator_type))
        {
            fprintf(stderr, "unsupported pt2 operator target %s\n", target->string.c_str());
            return false;
        }

        std::string operator_name = next_name("pnnx");
        const JsonValue* serialized_name = node.find("name");
        if (serialized_name && serialized_name->type == JsonValue::String)
            operator_name = serialized_name->string;

        std::map<std::string, const JsonValue*> provided;
        std::vector<std::string> provided_order;
        for (size_t i = 0; i < inputs->array.size(); i++)
        {
            const JsonValue* name = inputs->array[i].find("name");
            const JsonValue* argument = inputs->array[i].find("arg");
            if (!name || name->type != JsonValue::String || !argument || provided.find(name->string) != provided.end())
                return false;
            provided[name->string] = argument;
            provided_order.push_back(name->string);
        }

        std::set<std::string> consumed;
        std::vector<std::pair<std::string, Operand*> > ordered_inputs;
        std::optional<c10::OperatorHandle> handle;
        if (!schema_name.empty())
            handle = c10::Dispatcher::singleton().findSchema(c10::OperatorName(schema_name, overload));
        if (handle)
        {
            for (const c10::Argument& schema_argument : handle->schema().arguments())
            {
                Operand* operand = 0;
                std::map<std::string, const JsonValue*>::const_iterator it = provided.find(schema_argument.name());
                if (it != provided.end())
                {
                    operand = argument_operand(*it->second);
                    consumed.insert(schema_argument.name());
                }
                else if (schema_argument.default_value())
                {
                    Parameter parameter;
                    if (!parameter_from_ivalue(*schema_argument.default_value(), parameter))
                    {
                        fprintf(stderr, "unsupported default value for %s.%s\n", schema_name.c_str(), schema_argument.name().c_str());
                        return false;
                    }
                    operand = add_constant(parameter);
                }
                else
                {
                    fprintf(stderr, "missing required pt2 operator argument %s.%s\n", schema_name.c_str(), schema_argument.name().c_str());
                    return false;
                }
                if (!operand) return false;
                ordered_inputs.push_back(std::make_pair(schema_argument.name(), operand));
            }
        }

        for (size_t i = 0; i < provided_order.size(); i++)
        {
            if (consumed.find(provided_order[i]) != consumed.end()) continue;
            Operand* operand = argument_operand(*provided[provided_order[i]]);
            if (!operand) return false;
            ordered_inputs.push_back(std::make_pair(provided_order[i], operand));
        }

        Operator* op = graph.new_operator(operator_type, operator_name);
        for (size_t i = 0; i < ordered_inputs.size(); i++)
        {
            ordered_inputs[i].second->consumers.push_back(op);
            op->inputs.push_back(ordered_inputs[i].second);
        }

        for (size_t i = 0; i < outputs->array.size(); i++)
        {
            const JsonValue& output = outputs->array[i];
            const std::string name = argument_name(output);
            const JsonValue* sequence = output.find("as_tensors");
            if (!sequence) sequence = output.find("as_optional_tensors");
            if (sequence)
            {
                if (sequence->type != JsonValue::Array)
                    return false;

                Operand* list = graph.new_operand(next_name("pnnx"));
                list->producer = op;
                op->outputs.push_back(list);

                Operator* unpack = graph.new_operator("prim::ListUnpack", next_name("pnnx"));
                list->consumers.push_back(unpack);
                unpack->inputs.push_back(list);
                for (size_t j = 0; j < sequence->array.size(); j++)
                {
                    const std::string item_name = argument_name(sequence->array[j]);
                    if (item_name.empty())
                    {
                        fprintf(stderr, "unsupported unnamed pt2 sequence output\n");
                        return false;
                    }
                    Operand* operand = graph.new_operand(item_name);
                    operand->producer = unpack;
                    set_operand_tensor_meta(tensor_values, item_name, operand);
                    unpack->outputs.push_back(operand);
                    values[item_name] = operand;
                }
                continue;
            }
            if (name.empty())
            {
                fprintf(stderr, "unsupported unnamed pt2 operator output\n");
                return false;
            }
            Operand* operand = graph.new_operand(name);
            operand->producer = op;
            set_operand_tensor_meta(tensor_values, name, operand);
            op->outputs.push_back(operand);
            values[name] = operand;
        }
        return true;
    }

    bool add_output(const std::string& value_name, int output_index)
    {
        std::map<std::string, Operand*>::const_iterator it = values.find(value_name);
        if (it == values.end())
            return false;
        Operator* op = graph.new_operator("pnnx.Output", "pnnx_output_" + std::to_string(output_index));
        it->second->consumers.push_back(op);
        op->inputs.push_back(it->second);
        return true;
    }

private:
    Operand* list_operand(const JsonValue& array, const char* primitive_key = 0)
    {
        if (array.type != JsonValue::Array)
            return 0;

        std::vector<Operand*> items;
        for (size_t i = 0; i < array.array.size(); i++)
        {
            const JsonValue* item = &array.array[i];
            JsonValue wrapped;
            if (primitive_key)
            {
                wrapped.type = JsonValue::Object;
                wrapped.object[primitive_key] = *item;
                item = &wrapped;
            }
            Operand* input = argument_operand(*item);
            if (!input) return 0;
            items.push_back(input);
        }
        return list_operand(items);
    }

    Operand* list_operand(const std::vector<Operand*>& items)
    {
        return container_operand("prim::ListConstruct", items);
    }

    Operand* container_operand(const char* type, const std::vector<Operand*>& items)
    {
        Operator* op = graph.new_operator(type, next_name("pnnx"));
        for (size_t i = 0; i < items.size(); i++)
        {
            items[i]->consumers.push_back(op);
            op->inputs.push_back(items[i]);
        }
        Operand* output = graph.new_operand(op->name);
        output->producer = op;
        op->outputs.push_back(output);
        return output;
    }

    std::string next_name(const char* prefix)
    {
        return std::string(prefix) + "_" + std::to_string(index++);
    }

private:
    Graph& graph;
    const JsonValue& tensor_values;
    int index;
    std::map<std::string, Operand*> values;
};

static std::string find_config(const ExportArchive& archive, const std::string& directory, const std::string& suffix)
{
    const std::vector<std::string>& names = archive.get_names();
    for (size_t i = 0; i < names.size(); i++)
    {
        if (names[i].find(directory) != std::string::npos && names[i].size() >= suffix.size() && names[i].compare(names[i].size() - suffix.size(), suffix.size(), suffix) == 0)
            return names[i];
    }
    return std::string();
}

static const JsonValue* required_object(const JsonValue& object, const char* key)
{
    const JsonValue* value = object.find(key);
    return value && value->type == JsonValue::Object ? value : 0;
}

static const JsonValue* required_array(const JsonValue& object, const char* key)
{
    const JsonValue* value = object.find(key);
    return value && value->type == JsonValue::Array ? value : 0;
}

} // namespace

bool model_file_is_exported_program(const std::string& path)
{
    ExportArchive archive;
    if (!archive.open(path, true))
        return false;
    if (!archive.find("serialized_exported_program.json").empty())
        return true;

    const std::string format_name = archive.find("archive_format");
    const std::string model_name = archive.find("models/model.json");
    if (format_name.empty() || model_name.empty())
        return false;

    std::string format;
    return archive.read(format_name, format) && format == "pt2";
}

int load_exported_program(const std::string& ptpath, Graph& graph,
                          const std::vector<std::vector<int64_t> >& input_shapes,
                          const std::vector<std::string>& input_types)
{
    ExportArchive archive;
    if (!archive.open(ptpath))
        return -1;

    std::string model_json_name = archive.find("serialized_exported_program.json");
    const bool packaged = model_json_name.empty();
    if (packaged)
        model_json_name = archive.find("models/model.json");
    if (model_json_name.empty())
    {
        fprintf(stderr, "pt2 archive does not contain an exported program\n");
        return -1;
    }

    if (packaged)
    {
        std::string format;
        std::string version;
        if (!archive.read(archive.find("archive_format"), format) || format != "pt2")
        {
            fprintf(stderr, "invalid pt2 archive format\n");
            return -1;
        }
        const std::string version_name = archive.find("archive_version");
        if (!version_name.empty() && (!archive.read(version_name, version) || (version != "0" && version != "1")))
        {
            fprintf(stderr, "unsupported pt2 archive version %s\n", version.c_str());
            return -1;
        }
    }

    std::string model_json_data;
    JsonValue root;
    if (!archive.read(model_json_name, model_json_data) || !parse_json(model_json_data, root, "exported program json"))
        return -1;

    const JsonValue* schema_version = root.find("schema_version");
    int64_t schema_major = 0;
    int64_t schema_minor = 0;
    if (!schema_version || (schema_version->type == JsonValue::Object && (!schema_version->find("major") || !schema_version->find("minor") || !json_integer(*schema_version->find("major"), schema_major) || !json_integer(*schema_version->find("minor"), schema_minor))) || (schema_version->type != JsonValue::Object && !json_integer(*schema_version, schema_major)))
    {
        fprintf(stderr, "pt2 exported program has no valid schema version\n");
        return -1;
    }
    if (schema_major < 2 || schema_major > 8)
    {
        fprintf(stderr, "unsupported pt2 exported program schema %lld.%lld (supported major versions 2 through 8)\n", (long long)schema_major, (long long)schema_minor);
        return -1;
    }

    const JsonValue* graph_module = required_object(root, "graph_module");
    const JsonValue* serialized_graph = graph_module ? required_object(*graph_module, "graph") : 0;
    const JsonValue* signature = graph_module ? required_object(*graph_module, "signature") : 0;
    const JsonValue* graph_inputs = serialized_graph ? required_array(*serialized_graph, "inputs") : 0;
    const JsonValue* graph_nodes = serialized_graph ? required_array(*serialized_graph, "nodes") : 0;
    const JsonValue* tensor_values = serialized_graph ? required_object(*serialized_graph, "tensor_values") : 0;
    const JsonValue* input_specs = signature ? required_array(*signature, "input_specs") : 0;
    const JsonValue* output_specs = signature ? required_array(*signature, "output_specs") : 0;
    if (!graph_inputs || !graph_nodes || !tensor_values || !input_specs || !output_specs || graph_inputs->array.size() != input_specs->array.size())
    {
        fprintf(stderr, "invalid pt2 graph or graph signature\n");
        return -1;
    }

    bool need_state_dict = false;
    bool need_constants = false;
    for (size_t i = 0; i < input_specs->array.size(); i++)
    {
        need_state_dict = need_state_dict || input_specs->array[i].find("parameter") || input_specs->array[i].find("buffer");
        need_constants = need_constants || input_specs->array[i].find("tensor_constant");
    }

    TensorPayload state_dict;
    TensorPayload constants;
    std::string byteorder = "little";
    const std::string byteorder_name = archive.find("byteorder");
    if (!byteorder_name.empty() && !archive.read(byteorder_name, byteorder))
        return -1;

    if (!packaged)
    {
        std::string state_dict_name = archive.find("serialized_state_dict.pt");
        if (state_dict_name.empty()) state_dict_name = archive.find("serialized_state_dict.json");
        std::string constants_name = archive.find("serialized_constants.pt");
        if (constants_name.empty()) constants_name = archive.find("serialized_constants.json");
        if (need_state_dict && (state_dict_name.empty() || !load_nested_payload(archive, state_dict_name, state_dict)))
            return -1;
        if (need_constants && (constants_name.empty() || !load_nested_payload(archive, constants_name, constants)))
            return -1;
    }
    else if (need_state_dict || need_constants)
    {
        const std::string nested_weights = archive.find("data/weights/model.pt");
        const std::string nested_constants = archive.find("data/constants/model.pt");
        if (!nested_weights.empty() || !nested_constants.empty())
        {
            if ((need_state_dict && (nested_weights.empty() || !load_nested_payload(archive, nested_weights, state_dict))) || (need_constants && (nested_constants.empty() || !load_nested_payload(archive, nested_constants, constants))))
                return -1;
        }
        else
        {
            const std::string weights_config = find_config(archive, "/data/weights/", "_weights_config.json");
            const std::string constants_config = find_config(archive, "/data/constants/", "_constants_config.json");
            if ((need_state_dict && weights_config.empty()) || (need_constants && constants_config.empty()))
            {
                fprintf(stderr, "pt2 archive has no supported tensor payload\n");
                return -1;
            }
            if ((need_state_dict && !load_payload_config(archive, weights_config, byteorder, state_dict)) || (need_constants && !load_payload_config(archive, constants_config, byteorder, constants)))
                return -1;
        }
    }

    fprintf(stderr, "############# pass_level0\n");
    fprintf(stderr, "############# pass_level1\n");

    ExportGraphBuilder builder(graph, *tensor_values);
    size_t user_input_index = 0;
    for (size_t i = 0; i < input_specs->array.size(); i++)
    {
        const JsonValue& spec = input_specs->array[i];
        const JsonValue* user_input = spec.find("user_input");
        const JsonValue* parameter = spec.find("parameter");
        const JsonValue* buffer = spec.find("buffer");
        const JsonValue* tensor_constant = spec.find("tensor_constant");
        const JsonValue* body = user_input ? user_input : (parameter ? parameter : (buffer ? buffer : tensor_constant));
        if (!body || body->type != JsonValue::Object)
        {
            fprintf(stderr, "unsupported pt2 graph input kind\n");
            return -1;
        }

        const JsonValue* arg = body->find("arg");
        if (!arg)
            return -1;
        const std::string value_name = argument_name(*arg);
        if (value_name.empty())
        {
            fprintf(stderr, "non-tensor pt2 graph inputs are not supported\n");
            return -1;
        }

        if (user_input)
        {
            Operand* operand = builder.add_input(value_name);
            if (!input_shapes.empty())
            {
                if (user_input_index >= input_shapes.size())
                {
                    fprintf(stderr, "inputshape has fewer tensors than the pt2 graph\n");
                    return -1;
                }
                operand->shape.clear();
                for (size_t j = 0; j < input_shapes[user_input_index].size(); j++) operand->shape.push_back((int)input_shapes[user_input_index][j]);
                if (user_input_index < input_types.size()) operand->type = input_type_to_pnnx_type(input_types[user_input_index]);
            }
            user_input_index++;
            continue;
        }

        const char* target_key = parameter ? "parameter_name" : (buffer ? "buffer_name" : "tensor_constant_name");
        const JsonValue* target = body->find(target_key);
        if (!target && tensor_constant) target = body->find("constant_name");
        if (!target || target->type != JsonValue::String)
            return -1;
        const TensorPayload& payload = tensor_constant ? constants : state_dict;
        const at::Tensor* tensor = find_tensor(payload, target->string);
        if (!tensor)
        {
            fprintf(stderr, "pt2 tensor %s is missing from payload\n", target->string.c_str());
            return -1;
        }
        builder.add_attribute(value_name, target->string, *tensor);
    }
    if (!input_shapes.empty() && user_input_index != input_shapes.size())
    {
        fprintf(stderr, "inputshape has more tensors than the pt2 graph\n");
        return -1;
    }

    for (size_t i = 0; i < graph_nodes->array.size(); i++)
    {
        if (!builder.add_node(graph_nodes->array[i]))
        {
            fprintf(stderr, "load pt2 graph node %d failed\n", (int)i);
            return -1;
        }
    }

    int user_output_index = 0;
    for (size_t i = 0; i < output_specs->array.size(); i++)
    {
        const JsonValue* user_output = output_specs->array[i].find("user_output");
        if (!user_output || user_output->type != JsonValue::Object)
            continue;
        const JsonValue* arg = user_output->find("arg");
        const std::string value_name = arg ? argument_name(*arg) : std::string();
        if (value_name.empty() || !builder.add_output(value_name, user_output_index++))
        {
            fprintf(stderr, "invalid pt2 user output\n");
            return -1;
        }
    }
    if (user_output_index == 0)
    {
        fprintf(stderr, "pt2 graph has no user outputs\n");
        return -1;
    }

    return 0;
}

} // namespace pnnx
