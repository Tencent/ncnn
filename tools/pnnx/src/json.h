#ifndef PNNX_JSON_H
#define PNNX_JSON_H

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <string>
#include <vector>

namespace pnnx {

class JsonParseOptions
{
public:
    JsonParseOptions();

    size_t max_document_size;
    size_t max_depth;
    size_t max_values;
    size_t max_string_size;
    size_t max_number_size;
};

class JsonValue
{
public:
    enum Type
    {
        Null,
        Boolean,
        Integer,
        Number,
        String,
        Array,
        Object
    };

    JsonValue();

    Type type() const;

    bool is_null() const;
    bool get_bool(bool& value) const;
    bool get_int(int64_t& value) const;
    bool get_number(double& value) const;
    const std::string* get_string() const;
    const std::vector<JsonValue>* get_array() const;
    const std::map<std::string, JsonValue>* get_object() const;
    const JsonValue* get(const std::string& key) const;

private:
    friend class JsonParser;

    Type value_type;
    bool boolean_value;
    int64_t integer_value;
    double number_value;
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::map<std::string, JsonValue> object_value;
};

bool parse_json(const std::string& text, JsonValue& value, std::string& error);
bool parse_json(const std::string& text, JsonValue& value, std::string& error, const JsonParseOptions& options);

} // namespace pnnx

#endif // PNNX_JSON_H