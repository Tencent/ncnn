// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pt2_program.h"

#include <errno.h>
#include <stdlib.h>

#include <cmath>
#include <limits>
#include <set>
#include <utility>

#include "pt2_archive.h"
#include "pt2_json.h"

namespace pnnx {

Pt2SymInt::Pt2SymInt()
{
    symbolic = false;
    value = 0;
    has_hint = false;
    hint = 0;
}

Pt2Argument::Pt2Argument()
{
    type = None;
    i = 0;
    f = 0.f;
    b = false;
}

Pt2Program::Pt2Program()
{
    schema_major = 0;
    schema_minor = 0;
}

class Pt2ProgramDecoder
{
public:
    Pt2ProgramDecoder(Pt2Program& _program)
        : program(_program)
    {
    }

    int decode(const Pt2JsonValue& root)
    {
        if (!require_type(root, Pt2JsonValue::Object, "$", "object"))
            return -1;

        const Pt2JsonValue* schema = field(root, "schema_version", "$", true);
        if (!schema || !decode_schema_version(*schema))
            return -1;

        const Pt2JsonValue* opsets = field(root, "opset_version", "$", true);
        if (!opsets || !decode_opsets(*opsets))
            return -1;
        if (program.opset_versions.find("aten") == program.opset_versions.end())
            return fail("$.opset_version", opsets, "missing aten opset");
        if (program.opset_versions["aten"] != 10)
            return fail("$.opset_version.aten", opsets, "unsupported aten opset");

        const Pt2JsonValue* graph_module = field(root, "graph_module", "$", true);
        if (!graph_module || !decode_graph_module(*graph_module))
            return -1;

        const Pt2JsonValue* ranges = field(root, "range_constraints", "$", true);
        if (!ranges || !decode_ranges(*ranges))
            return -1;

        const Pt2JsonValue* torch_version = field(root, "torch_version", "$", false);
        if (torch_version && !get_string(*torch_version, "$.torch_version", program.torch_version))
            return -1;

        const Pt2JsonValue* verifiers = field(root, "verifiers", "$", false);
        const Pt2JsonValue* guards_code = field(root, "guards_code", "$", false);
        if ((verifiers && !decode_string_array(*verifiers, "$.verifiers")) ||
            (guards_code && !decode_string_array(*guards_code, "$.guards_code")) ||
            !reject_unknown(root, "$", "graph_module", "opset_version", "range_constraints", "schema_version", "torch_version", "verifiers", "guards_code"))
            return -1;
        return verify();
    }

    int decode_tensor_meta(const Pt2JsonValue& value, const std::string& path, Pt2Tensor& tensor)
    {
        return decode_tensor(value, path, tensor) ? 0 : -1;
    }

private:
    int fail(const std::string& path, const Pt2JsonValue* value, const std::string& message)
    {
        program.error = path + " at json offset " + std::to_string(value ? value->offset : 0) + ": " + message;
        return -1;
    }

    bool require_type(const Pt2JsonValue& value, Pt2JsonValue::Type type, const std::string& path, const char* expected)
    {
        if (value.type == type)
            return true;
        fail(path, &value, std::string("expected ") + expected);
        return false;
    }

    const Pt2JsonValue* field(const Pt2JsonValue& object, const char* name, const std::string& path, bool required)
    {
        if (!require_type(object, Pt2JsonValue::Object, path, "object"))
            return 0;
        std::map<std::string, Pt2JsonValue>::const_iterator it = object.object.find(name);
        if (it != object.object.end())
            return &it->second;
        if (required)
            fail(path + "." + name, &object, "missing required field");
        return 0;
    }

    bool get_string(const Pt2JsonValue& value, const std::string& path, std::string& result)
    {
        if (!require_type(value, Pt2JsonValue::String, path, "string"))
            return false;
        result = value.value;
        return true;
    }

    bool get_bool(const Pt2JsonValue& value, const std::string& path, bool& result)
    {
        if (!require_type(value, Pt2JsonValue::Bool, path, "boolean"))
            return false;
        result = value.boolean;
        return true;
    }

    bool get_int(const Pt2JsonValue& value, const std::string& path, int64_t& result)
    {
        if (!require_type(value, Pt2JsonValue::Number, path, "integer"))
            return false;
        if (value.value.find_first_of(".eE") != std::string::npos)
        {
            fail(path, &value, "expected integer");
            return false;
        }

        errno = 0;
        char* end = 0;
        const long long parsed = strtoll(value.value.c_str(), &end, 10);
        if (errno == ERANGE || !end || *end)
        {
            fail(path, &value, "integer is out of range");
            return false;
        }
        result = (int64_t)parsed;
        return true;
    }

    bool get_double(const Pt2JsonValue& value, const std::string& path, double& result)
    {
        if (!require_type(value, Pt2JsonValue::Number, path, "number"))
            return false;
        errno = 0;
        char* end = 0;
        result = strtod(value.value.c_str(), &end);
        if (errno == ERANGE || !end || *end)
        {
            fail(path, &value, "number is out of range");
            return false;
        }
        return true;
    }

    bool get_float(const Pt2JsonValue& value, const std::string& path, double& result)
    {
        if (value.type != Pt2JsonValue::String)
            return get_double(value, path, result);
        if (value.value == "Infinity") result = std::numeric_limits<double>::infinity();
        else if (value.value == "-Infinity") result = -std::numeric_limits<double>::infinity();
        else if (value.value == "NaN") result = std::numeric_limits<double>::quiet_NaN();
        else
        {
            fail(path, &value, "invalid special float");
            return false;
        }
        return true;
    }

    bool decode_string_array(const Pt2JsonValue& value, const std::string& path)
    {
        if (!require_type(value, Pt2JsonValue::Array, path, "string array"))
            return false;
        for (size_t i = 0; i < value.array.size(); i++)
        {
            std::string ignored;
            if (!get_string(value.array[i], path + "[" + std::to_string(i) + "]", ignored))
                return false;
        }
        return true;
    }

    bool reject_unknown(const Pt2JsonValue& value, const std::string& path, const char* a, const char* b = 0, const char* c = 0,
                        const char* d = 0, const char* e = 0, const char* f = 0, const char* g = 0,
                        const char* h = 0, const char* i = 0)
    {
        const char* known[] = {a, b, c, d, e, f, g, h, i};
        for (std::map<std::string, Pt2JsonValue>::const_iterator it = value.object.begin(); it != value.object.end(); ++it)
        {
            bool found = false;
            for (size_t i = 0; i < sizeof(known) / sizeof(known[0]) && known[i]; i++)
            {
                if (it->first == known[i])
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                fail(path + "." + it->first, &it->second, "unknown field");
                return false;
            }
        }
        return true;
    }

    bool decode_schema_version(const Pt2JsonValue& value)
    {
        const Pt2JsonValue* major = field(value, "major", "$.schema_version", true);
        const Pt2JsonValue* minor = field(value, "minor", "$.schema_version", true);
        int64_t major_value;
        int64_t minor_value;
        if (!major || !minor || !get_int(*major, "$.schema_version.major", major_value) || !get_int(*minor, "$.schema_version.minor", minor_value))
            return false;
        if (major_value < 0 || major_value > INT32_MAX || minor_value < 0 || minor_value > INT32_MAX)
        {
            fail("$.schema_version", &value, "schema version is out of range");
            return false;
        }
        program.schema_major = (int)major_value;
        program.schema_minor = (int)minor_value;
        if (program.schema_major != 8)
        {
            fail("$.schema_version.major", major, "unsupported schema major");
            return false;
        }
        if (program.schema_minor != 2 && program.schema_minor != 7 && program.schema_minor != 8 && program.schema_minor != 14 && program.schema_minor != 15 && program.schema_minor != 17 && program.schema_minor != 20)
        {
            fail("$.schema_version.minor", minor, "untested schema minor");
            return false;
        }
        return reject_unknown(value, "$.schema_version", "major", "minor");
    }

    bool decode_opsets(const Pt2JsonValue& value)
    {
        if (!require_type(value, Pt2JsonValue::Object, "$.opset_version", "object"))
            return false;
        for (std::map<std::string, Pt2JsonValue>::const_iterator it = value.object.begin(); it != value.object.end(); ++it)
        {
            int64_t version;
            if (!get_int(it->second, "$.opset_version." + it->first, version) || version < 0 || version > INT32_MAX)
                return false;
            program.opset_versions[it->first] = (int)version;
        }
        return true;
    }

    bool decode_sym_int(const Pt2JsonValue& value, const std::string& path, Pt2SymInt& result)
    {
        if (!require_type(value, Pt2JsonValue::Object, path, "SymInt union") || value.object.size() != 1)
        {
            fail(path, &value, "SymInt union must contain exactly one field");
            return false;
        }

        const std::string& tag = value.object.begin()->first;
        const Pt2JsonValue& payload = value.object.begin()->second;
        if (tag == "as_int")
            return get_int(payload, path + ".as_int", result.value);
        if (tag != "as_expr")
        {
            fail(path, &value, "unknown SymInt union tag " + tag);
            return false;
        }

        result.symbolic = true;
        const Pt2JsonValue* expression = field(payload, "expr_str", path + ".as_expr", true);
        if (!expression || !get_string(*expression, path + ".as_expr.expr_str", result.expression))
            return false;
        const Pt2JsonValue* hint = field(payload, "hint", path + ".as_expr", false);
        if (hint && hint->type != Pt2JsonValue::Null)
        {
            if (!require_type(*hint, Pt2JsonValue::Object, path + ".as_expr.hint", "SymExprHint union") || hint->object.size() != 1)
                return false;
            if (hint->object.begin()->first != "as_int")
            {
                fail(path + ".as_expr.hint", hint, "SymInt hint must use as_int");
                return false;
            }
            if (!get_int(hint->object.begin()->second, path + ".as_expr.hint.as_int", result.hint))
                return false;
            result.has_hint = true;
        }
        return reject_unknown(payload, path + ".as_expr", "expr_str", "hint");
    }

    bool decode_device(const Pt2JsonValue& value, const std::string& path, std::string& type, int& index)
    {
        const Pt2JsonValue* type_value = field(value, "type", path, true);
        if (!type_value || !get_string(*type_value, path + ".type", type))
            return false;
        index = -1;
        const Pt2JsonValue* index_value = field(value, "index", path, false);
        if (index_value && index_value->type != Pt2JsonValue::Null)
        {
            int64_t parsed;
            if (!get_int(*index_value, path + ".index", parsed) || parsed < 0 || parsed > INT32_MAX)
                return false;
            index = (int)parsed;
        }
        return reject_unknown(value, path, "type", "index");
    }

    bool decode_tensor_argument(const Pt2JsonValue& value, const std::string& path, std::string& name)
    {
        const Pt2JsonValue* name_value = field(value, "name", path, true);
        if (!name_value || !get_string(*name_value, path + ".name", name))
            return false;
        if (name.empty())
        {
            fail(path + ".name", name_value, "tensor name is empty");
            return false;
        }
        return reject_unknown(value, path, "name");
    }

    bool decode_optional_tensor_argument(const Pt2JsonValue& value, const std::string& path, std::string& name, bool& has_tensor)
    {
        if (!require_type(value, Pt2JsonValue::Object, path, "OptionalTensorArgument union") || value.object.size() != 1)
        {
            fail(path, &value, "OptionalTensorArgument union must contain exactly one field");
            return false;
        }

        has_tensor = value.object.begin()->first == "as_tensor";
        if (has_tensor)
            return decode_tensor_argument(value.object.begin()->second, path + ".as_tensor", name);
        if (value.object.begin()->first == "as_none")
        {
            bool marker;
            if (!get_bool(value.object.begin()->second, path + ".as_none", marker))
                return false;
            if (!marker)
            {
                fail(path + ".as_none", &value.object.begin()->second, "none union marker must be true");
                return false;
            }
            return true;
        }

        fail(path, &value, "unknown OptionalTensorArgument union tag " + value.object.begin()->first);
        return false;
    }

    bool decode_argument(const Pt2JsonValue& value, const std::string& path, Pt2Argument& result)
    {
        if (!require_type(value, Pt2JsonValue::Object, path, "Argument union") || value.object.size() != 1)
        {
            fail(path, &value, "Argument union must contain exactly one field");
            return false;
        }

        const std::string& tag = value.object.begin()->first;
        const Pt2JsonValue& payload = value.object.begin()->second;
        const std::string payload_path = path + "." + tag;

        if (tag == "as_none")
        {
            result.type = Pt2Argument::None;
            if (!get_bool(payload, payload_path, result.b))
                return false;
            if (!result.b)
            {
                fail(payload_path, &payload, "none union marker must be true");
                return false;
            }
            return true;
        }
        if (tag == "as_tensor")
        {
            result.type = Pt2Argument::Tensor;
            return decode_tensor_argument(payload, payload_path, result.s);
        }
        if (tag == "as_tensors")
        {
            result.type = Pt2Argument::Tensors;
            if (!require_type(payload, Pt2JsonValue::Array, payload_path, "array"))
                return false;
            for (size_t i = 0; i < payload.array.size(); i++)
            {
                std::string name;
                if (!decode_tensor_argument(payload.array[i], payload_path + "[" + std::to_string(i) + "]", name))
                    return false;
                result.as.push_back(std::move(name));
            }
            return true;
        }
        if (tag == "as_optional_tensor")
        {
            result.type = Pt2Argument::OptionalTensor;
            return decode_optional_tensor_argument(payload, payload_path, result.s, result.b);
        }
        if (tag == "as_optional_tensors")
        {
            result.type = Pt2Argument::OptionalTensors;
            if (!require_type(payload, Pt2JsonValue::Array, payload_path, "array"))
                return false;
            for (size_t i = 0; i < payload.array.size(); i++)
            {
                std::string name;
                bool has_tensor;
                if (!decode_optional_tensor_argument(payload.array[i], payload_path + "[" + std::to_string(i) + "]", name, has_tensor))
                    return false;
                result.as.push_back(has_tensor ? std::move(name) : std::string());
            }
            return true;
        }
        if (tag == "as_int" || tag == "as_scalar_type" || tag == "as_memory_format" || tag == "as_layout")
        {
            result.type = tag == "as_int" ? Pt2Argument::Int : tag == "as_scalar_type" ? Pt2Argument::ScalarType : tag == "as_memory_format" ? Pt2Argument::MemoryFormat : Pt2Argument::Layout;
            return get_int(payload, payload_path, result.i);
        }
        if (tag == "as_ints")
        {
            result.type = Pt2Argument::Ints;
            if (!require_type(payload, Pt2JsonValue::Array, payload_path, "array"))
                return false;
            for (size_t i = 0; i < payload.array.size(); i++)
            {
                int64_t item;
                if (!get_int(payload.array[i], payload_path + "[" + std::to_string(i) + "]", item))
                    return false;
                result.ai.push_back(item);
            }
            return true;
        }
        if (tag == "as_float")
        {
            result.type = Pt2Argument::Float;
            return get_float(payload, payload_path, result.f);
        }
        if (tag == "as_floats")
        {
            result.type = Pt2Argument::Floats;
            if (!require_type(payload, Pt2JsonValue::Array, payload_path, "array"))
                return false;
            for (size_t i = 0; i < payload.array.size(); i++)
            {
                double item;
                if (!get_float(payload.array[i], payload_path + "[" + std::to_string(i) + "]", item))
                    return false;
                result.af.push_back(item);
            }
            return true;
        }
        if (tag == "as_complex")
        {
            result.type = Pt2Argument::Complex;
            const Pt2JsonValue* real = field(payload, "real", payload_path, true);
            const Pt2JsonValue* imag = field(payload, "imag", payload_path, true);
            if (!real || !imag)
                return false;
            double r;
            double i;
            if (!get_float(*real, payload_path + ".real", r) || !get_float(*imag, payload_path + ".imag", i))
                return false;
            result.af.push_back(r);
            result.af.push_back(i);
            return reject_unknown(payload, payload_path, "real", "imag");
        }
        if (tag == "as_string")
        {
            result.type = Pt2Argument::String;
            return get_string(payload, payload_path, result.s);
        }
        if (tag == "as_strings")
        {
            result.type = Pt2Argument::Strings;
            if (!require_type(payload, Pt2JsonValue::Array, payload_path, "array"))
                return false;
            for (size_t i = 0; i < payload.array.size(); i++)
            {
                std::string item;
                if (!get_string(payload.array[i], payload_path + "[" + std::to_string(i) + "]", item))
                    return false;
                result.as.push_back(std::move(item));
            }
            return true;
        }
        if (tag == "as_bool")
        {
            result.type = Pt2Argument::Bool;
            return get_bool(payload, payload_path, result.b);
        }
        if (tag == "as_bools")
        {
            result.type = Pt2Argument::Bools;
            if (!require_type(payload, Pt2JsonValue::Array, payload_path, "array"))
                return false;
            for (size_t i = 0; i < payload.array.size(); i++)
            {
                bool item;
                if (!get_bool(payload.array[i], payload_path + "[" + std::to_string(i) + "]", item))
                    return false;
                result.ab.push_back(item);
            }
            return true;
        }
        if (tag == "as_device")
        {
            result.type = Pt2Argument::Device;
            int index;
            if (!decode_device(payload, payload_path, result.s, index))
                return false;
            result.i = index;
            return true;
        }
        if (tag == "as_sym_int")
        {
            result.type = Pt2Argument::SymInt;
            if (!require_type(payload, Pt2JsonValue::Object, payload_path, "SymIntArgument union") || payload.object.size() != 1)
            {
                fail(payload_path, &payload, "SymIntArgument union must contain exactly one field");
                return false;
            }
            if (payload.object.begin()->first == "as_name")
            {
                result.b = true;
                return get_string(payload.object.begin()->second, payload_path + ".as_name", result.s);
            }
            if (payload.object.begin()->first == "as_int")
                return get_int(payload.object.begin()->second, payload_path + ".as_int", result.i);
            fail(payload_path, &payload, "unknown SymIntArgument union tag " + payload.object.begin()->first);
            return false;
        }
        if (tag == "as_sym_ints")
        {
            result.type = Pt2Argument::SymInts;
            if (!require_type(payload, Pt2JsonValue::Array, payload_path, "symbolic integer array"))
                return false;
            for (size_t i = 0; i < payload.array.size(); i++)
            {
                const Pt2JsonValue& item = payload.array[i];
                const std::string item_path = payload_path + "[" + std::to_string(i) + "]";
                if (!require_type(item, Pt2JsonValue::Object, item_path, "symbolic integer union"))
                    return false;
                if (item.object.size() != 1)
                {
                    fail(item_path, &item, "symbolic integer union must contain exactly one field");
                    return false;
                }
                Pt2Argument arg;
                arg.type = Pt2Argument::SymInt;
                if (item.object.begin()->first == "as_name")
                {
                    arg.b = true;
                    if (!get_string(item.object.begin()->second, item_path + ".as_name", arg.s))
                        return false;
                }
                else if (item.object.begin()->first == "as_int")
                {
                    if (!get_int(item.object.begin()->second, item_path + ".as_int", arg.i))
                        return false;
                }
                else
                {
                    fail(item_path, &item, "unknown symbolic integer union tag " + item.object.begin()->first);
                    return false;
                }
                result.args.push_back(std::move(arg));
            }
            return true;
        }
        if (tag == "as_sym_bool")
        {
            result.type = Pt2Argument::SymBool;
            if (!require_type(payload, Pt2JsonValue::Object, payload_path, "SymBoolArgument union"))
                return false;
            if (payload.object.size() != 1 || payload.object.begin()->first != "as_name")
            {
                fail(payload_path, &payload, "SymBoolArgument union must contain exactly one as_name field");
                return false;
            }
            result.b = true;
            return get_string(payload.object.begin()->second, payload_path + ".as_name", result.s);
        }

        fail(path, &value, "unsupported Argument union tag " + tag);
        return false;
    }

    bool decode_tensor(const Pt2JsonValue& value, const std::string& path, Pt2Tensor& tensor)
    {
        const Pt2JsonValue* dtype = field(value, "dtype", path, true);
        const Pt2JsonValue* sizes = field(value, "sizes", path, true);
        const Pt2JsonValue* requires_grad = field(value, "requires_grad", path, true);
        const Pt2JsonValue* device = field(value, "device", path, true);
        const Pt2JsonValue* strides = field(value, "strides", path, true);
        const Pt2JsonValue* storage_offset = field(value, "storage_offset", path, true);
        const Pt2JsonValue* layout = field(value, "layout", path, true);
        int64_t dtype_value;
        int64_t layout_value;
        if (!dtype || !sizes || !requires_grad || !device || !strides || !storage_offset || !layout ||
            !get_int(*dtype, path + ".dtype", dtype_value) || !get_int(*layout, path + ".layout", layout_value) ||
            !get_bool(*requires_grad, path + ".requires_grad", tensor.requires_grad) ||
            !decode_device(*device, path + ".device", tensor.device, tensor.device_index) ||
            !require_type(*sizes, Pt2JsonValue::Array, path + ".sizes", "array") ||
            !require_type(*strides, Pt2JsonValue::Array, path + ".strides", "array") ||
            !decode_sym_int(*storage_offset, path + ".storage_offset", tensor.storage_offset))
            return false;

        if (dtype_value < 0 || dtype_value > INT32_MAX || layout_value < 0 || layout_value > INT32_MAX)
        {
            fail(path, &value, "tensor enum value is out of range");
            return false;
        }
        tensor.dtype = (int)dtype_value;
        tensor.layout = (int)layout_value;
        for (size_t i = 0; i < sizes->array.size(); i++)
        {
            Pt2SymInt item;
            if (!decode_sym_int(sizes->array[i], path + ".sizes[" + std::to_string(i) + "]", item))
                return false;
            tensor.sizes.push_back(std::move(item));
        }
        for (size_t i = 0; i < strides->array.size(); i++)
        {
            Pt2SymInt item;
            if (!decode_sym_int(strides->array[i], path + ".strides[" + std::to_string(i) + "]", item))
                return false;
            tensor.strides.push_back(std::move(item));
        }
        return reject_unknown(value, path, "dtype", "sizes", "requires_grad", "device", "strides", "storage_offset", "layout");
    }

    bool decode_graph(const Pt2JsonValue& value)
    {
        const std::string path = "$.graph_module.graph";
        const Pt2JsonValue* inputs = field(value, "inputs", path, true);
        const Pt2JsonValue* outputs = field(value, "outputs", path, true);
        const Pt2JsonValue* nodes = field(value, "nodes", path, true);
        const Pt2JsonValue* tensors = field(value, "tensor_values", path, true);
        if (!inputs || !outputs || !nodes || !tensors ||
            !require_type(*inputs, Pt2JsonValue::Array, path + ".inputs", "array") ||
            !require_type(*outputs, Pt2JsonValue::Array, path + ".outputs", "array") ||
            !require_type(*nodes, Pt2JsonValue::Array, path + ".nodes", "array") ||
            !require_type(*tensors, Pt2JsonValue::Object, path + ".tensor_values", "object"))
            return false;

        for (size_t i = 0; i < inputs->array.size(); i++)
        {
            Pt2Argument arg;
            if (!decode_argument(inputs->array[i], path + ".inputs[" + std::to_string(i) + "]", arg))
                return false;
            program.inputs.push_back(std::move(arg));
        }
        for (size_t i = 0; i < outputs->array.size(); i++)
        {
            Pt2Argument arg;
            if (!decode_argument(outputs->array[i], path + ".outputs[" + std::to_string(i) + "]", arg))
                return false;
            program.outputs.push_back(std::move(arg));
        }
        for (std::map<std::string, Pt2JsonValue>::const_iterator it = tensors->object.begin(); it != tensors->object.end(); ++it)
        {
            Pt2Tensor tensor;
            if (!decode_tensor(it->second, path + ".tensor_values." + it->first, tensor))
                return false;
            program.tensors[it->first] = std::move(tensor);
        }

        const Pt2JsonValue* sym_ints = field(value, "sym_int_values", path, false);
        if (sym_ints)
        {
            if (!require_type(*sym_ints, Pt2JsonValue::Object, path + ".sym_int_values", "object"))
                return false;
            for (std::map<std::string, Pt2JsonValue>::const_iterator it = sym_ints->object.begin(); it != sym_ints->object.end(); ++it)
            {
                Pt2SymInt sym;
                if (!decode_sym_int(it->second, path + ".sym_int_values." + it->first, sym))
                    return false;
                program.sym_ints[it->first] = std::move(sym);
            }
        }

        const Pt2JsonValue* sym_bools = field(value, "sym_bool_values", path, false);
        if (sym_bools && !require_type(*sym_bools, Pt2JsonValue::Object, path + ".sym_bool_values", "object"))
            return false;
        if (sym_bools)
        {
            for (std::map<std::string, Pt2JsonValue>::const_iterator it = sym_bools->object.begin(); it != sym_bools->object.end(); ++it)
            {
                const std::string symbol_path = path + ".sym_bool_values." + it->first;
                if (!require_type(it->second, Pt2JsonValue::Object, symbol_path, "SymBool union"))
                    return false;
                if (it->second.object.size() != 1 || it->second.object.begin()->first != "as_expr")
                {
                    fail(symbol_path, &it->second, "SymBool union must contain exactly one as_expr field");
                    return false;
                }
                const Pt2JsonValue& expr = it->second.object.begin()->second;
                const Pt2JsonValue* expression = field(expr, "expr_str", symbol_path + ".as_expr", true);
                std::string ignored;
                if (!expression || !get_string(*expression, symbol_path + ".as_expr.expr_str", ignored) ||
                    !reject_unknown(expr, symbol_path + ".as_expr", "expr_str", "hint"))
                    return false;
                const Pt2JsonValue* hint = field(expr, "hint", symbol_path + ".as_expr", false);
                if (hint && hint->type != Pt2JsonValue::Null)
                {
                    if (!require_type(*hint, Pt2JsonValue::Object, symbol_path + ".as_expr.hint", "SymExprHint union"))
                        return false;
                    if (hint->object.size() != 1 || hint->object.begin()->first != "as_bool")
                    {
                        fail(symbol_path + ".as_expr.hint", hint, "SymBool hint must use as_bool");
                        return false;
                    }
                    bool ignored_hint;
                    if (!get_bool(hint->object.begin()->second, symbol_path + ".as_expr.hint.as_bool", ignored_hint))
                        return false;
                }
                program.sym_bools.insert(it->first);
            }
        }
        const char* unsupported_maps[] = {"sym_float_values", "custom_obj_values"};
        for (size_t i = 0; i < 2; i++)
        {
            const Pt2JsonValue* unsupported = field(value, unsupported_maps[i], path, false);
            if (unsupported && !require_type(*unsupported, Pt2JsonValue::Object, path + "." + unsupported_maps[i], "object"))
                return false;
            if (unsupported && !unsupported->object.empty())
            {
                fail(path + "." + unsupported_maps[i], unsupported, "unsupported non-empty value table");
                return false;
            }
        }
        const Pt2JsonValue* single_return = field(value, "is_single_tensor_return", path, false);
        if (single_return && single_return->type != Pt2JsonValue::Null)
        {
            bool enabled;
            if (!get_bool(*single_return, path + ".is_single_tensor_return", enabled))
                return false;
            if (enabled)
            {
                fail(path + ".is_single_tensor_return", single_return, "higher-order single tensor return is unsupported");
                return false;
            }
        }

        for (size_t i = 0; i < nodes->array.size(); i++)
        {
            const Pt2JsonValue& node_value = nodes->array[i];
            const std::string node_path = path + ".nodes[" + std::to_string(i) + "]";
            const Pt2JsonValue* target = field(node_value, "target", node_path, true);
            const Pt2JsonValue* node_inputs = field(node_value, "inputs", node_path, true);
            const Pt2JsonValue* node_outputs = field(node_value, "outputs", node_path, true);
            const Pt2JsonValue* metadata = field(node_value, "metadata", node_path, true);
            if (!target || !node_inputs || !node_outputs ||
                !require_type(*node_inputs, Pt2JsonValue::Array, node_path + ".inputs", "array") ||
                !require_type(*node_outputs, Pt2JsonValue::Array, node_path + ".outputs", "array") ||
                !metadata || !require_type(*metadata, Pt2JsonValue::Object, node_path + ".metadata", "object"))
                return false;

            Pt2Node node;
            if (!get_string(*target, node_path + ".target", node.target))
                return false;
            if (node.target.compare(0, 23, "torch.ops.higher_order.") == 0)
            {
                fail(node_path + ".target", target, node.target == "torch.ops.higher_order.wrap_with_autocast" ? "wrap_with_autocast is unsupported" : "higher-order operators are unsupported");
                return false;
            }
            const Pt2JsonValue* node_name = field(node_value, "name", node_path, false);
            if (node_name && !get_string(*node_name, node_path + ".name", node.name))
                return false;
            const Pt2JsonValue* single_return = field(node_value, "is_hop_single_tensor_return", node_path, false);
            if (single_return && single_return->type != Pt2JsonValue::Null)
            {
                bool enabled;
                if (!get_bool(*single_return, node_path + ".is_hop_single_tensor_return", enabled))
                    return false;
                if (enabled)
                {
                    fail(node_path + ".is_hop_single_tensor_return", single_return, "higher-order single tensor return is unsupported");
                    return false;
                }
            }

            for (size_t j = 0; j < node_inputs->array.size(); j++)
            {
                const Pt2JsonValue& named_value = node_inputs->array[j];
                const std::string named_path = node_path + ".inputs[" + std::to_string(j) + "]";
                const Pt2JsonValue* name = field(named_value, "name", named_path, true);
                const Pt2JsonValue* arg = field(named_value, "arg", named_path, true);
                Pt2NamedArgument named;
                named.kind = 0;
                if (!name || !arg || !get_string(*name, named_path + ".name", named.name) || !decode_argument(*arg, named_path + ".arg", named.arg))
                    return false;
                const Pt2JsonValue* kind = field(named_value, "kind", named_path, false);
                if (kind)
                {
                    int64_t kind_value;
                    if (!get_int(*kind, named_path + ".kind", kind_value) || kind_value < 0 || kind_value > INT32_MAX)
                        return false;
                    named.kind = (int)kind_value;
                }
                if (!reject_unknown(named_value, named_path, "name", "arg", "kind"))
                    return false;
                node.inputs.push_back(std::move(named));
            }
            for (size_t j = 0; j < node_outputs->array.size(); j++)
            {
                Pt2Argument arg;
                if (!decode_argument(node_outputs->array[j], node_path + ".outputs[" + std::to_string(j) + "]", arg))
                    return false;
                node.outputs.push_back(std::move(arg));
            }

            if (!reject_unknown(node_value, node_path, "target", "inputs", "outputs", "metadata", "name", "is_hop_single_tensor_return"))
                return false;
            program.nodes.push_back(std::move(node));
        }

        return reject_unknown(value, path, "inputs", "outputs", "nodes", "tensor_values", "sym_int_values", "sym_bool_values", "is_single_tensor_return", "custom_obj_values", "sym_float_values");
    }

    bool decode_signature(const Pt2JsonValue& value)
    {
        const std::string path = "$.graph_module.signature";
        const Pt2JsonValue* inputs = field(value, "input_specs", path, true);
        const Pt2JsonValue* outputs = field(value, "output_specs", path, true);
        if (!inputs || !outputs || !require_type(*inputs, Pt2JsonValue::Array, path + ".input_specs", "array") ||
            !require_type(*outputs, Pt2JsonValue::Array, path + ".output_specs", "array"))
            return false;

        for (size_t i = 0; i < inputs->array.size(); i++)
        {
            const Pt2JsonValue& spec_value = inputs->array[i];
            const std::string spec_path = path + ".input_specs[" + std::to_string(i) + "]";
            if (!require_type(spec_value, Pt2JsonValue::Object, spec_path, "InputSpec union") || spec_value.object.size() != 1)
            {
                fail(spec_path, &spec_value, "InputSpec union must contain exactly one field");
                return false;
            }
            const std::string& tag = spec_value.object.begin()->first;
            const Pt2JsonValue& payload = spec_value.object.begin()->second;
            Pt2InputSpec spec;
            spec.persistent = false;
            if (tag == "user_input")
            {
                spec.kind = Pt2InputSpec::UserInput;
                const Pt2JsonValue* arg = field(payload, "arg", spec_path + ".user_input", true);
                if (!arg || !decode_argument(*arg, spec_path + ".user_input.arg", spec.arg) ||
                    !reject_unknown(payload, spec_path + ".user_input", "arg"))
                    return false;
            }
            else if (tag == "parameter" || tag == "buffer" || tag == "tensor_constant")
            {
                spec.kind = tag == "parameter" ? Pt2InputSpec::Parameter : tag == "buffer" ? Pt2InputSpec::Buffer : Pt2InputSpec::TensorConstant;
                const std::string base = spec_path + "." + tag;
                const Pt2JsonValue* arg = field(payload, "arg", base, true);
                const char* target_name = tag == "parameter" ? "parameter_name" : tag == "buffer" ? "buffer_name" : "tensor_constant_name";
                const Pt2JsonValue* target = field(payload, target_name, base, true);
                spec.arg.type = Pt2Argument::Tensor;
                if (!arg || !target || !decode_tensor_argument(*arg, base + ".arg", spec.arg.s) || !get_string(*target, base + "." + target_name, spec.target))
                    return false;
                if (spec.target.empty())
                {
                    fail(base + "." + target_name, target, "weight name is empty");
                    return false;
                }
                if (tag == "buffer")
                {
                    const Pt2JsonValue* persistent = field(payload, "persistent", base, true);
                    if (!persistent || !get_bool(*persistent, base + ".persistent", spec.persistent) ||
                        !reject_unknown(payload, base, "arg", "buffer_name", "persistent"))
                        return false;
                }
                else if (!reject_unknown(payload, base, "arg", target_name))
                    return false;
            }
            else if (tag == "constant_input")
            {
                spec.kind = Pt2InputSpec::ConstantInput;
                const std::string base = spec_path + ".constant_input";
                const Pt2JsonValue* name = field(payload, "name", base, true);
                const Pt2JsonValue* constant = field(payload, "value", base, true);
                if (!name || !constant || !get_string(*name, base + ".name", spec.target) || !decode_argument(*constant, base + ".value", spec.arg) ||
                    !reject_unknown(payload, base, "name", "value"))
                    return false;
                if (spec.target.empty())
                {
                    fail(base + ".name", name, "constant input name is empty");
                    return false;
                }
            }
            else
            {
                fail(spec_path, &spec_value, "unsupported InputSpec union tag " + tag);
                return false;
            }
            program.input_specs.push_back(std::move(spec));
        }

        for (size_t i = 0; i < outputs->array.size(); i++)
        {
            const Pt2JsonValue& spec_value = outputs->array[i];
            const std::string spec_path = path + ".output_specs[" + std::to_string(i) + "]";
            if (!require_type(spec_value, Pt2JsonValue::Object, spec_path, "OutputSpec union") || spec_value.object.size() != 1)
            {
                fail(spec_path, &spec_value, "OutputSpec union must contain exactly one field");
                return false;
            }
            if (spec_value.object.begin()->first != "user_output")
            {
                fail(spec_path, &spec_value, "training or mutation output is unsupported");
                return false;
            }
            const Pt2JsonValue& payload = spec_value.object.begin()->second;
            const Pt2JsonValue* arg = field(payload, "arg", spec_path + ".user_output", true);
            Pt2OutputSpec spec;
            if (!arg || !decode_argument(*arg, spec_path + ".user_output.arg", spec.arg) ||
                !reject_unknown(payload, spec_path + ".user_output", "arg"))
                return false;
            program.output_specs.push_back(std::move(spec));
        }
        return reject_unknown(value, path, "input_specs", "output_specs");
    }

    bool decode_graph_module(const Pt2JsonValue& value)
    {
        const std::string path = "$.graph_module";
        const Pt2JsonValue* graph = field(value, "graph", path, true);
        const Pt2JsonValue* signature = field(value, "signature", path, true);
        const Pt2JsonValue* module_call_graph = field(value, "module_call_graph", path, true);
        const Pt2JsonValue* metadata = field(value, "metadata", path, false);
        const Pt2JsonValue* treespec = field(value, "treespec_namedtuple_fields", path, false);
        if (!graph || !signature || !module_call_graph || !require_type(*module_call_graph, Pt2JsonValue::Array, path + ".module_call_graph", "array") ||
            (metadata && !require_type(*metadata, Pt2JsonValue::Object, path + ".metadata", "object")) ||
            (treespec && !require_type(*treespec, Pt2JsonValue::Object, path + ".treespec_namedtuple_fields", "object")) ||
            !decode_graph(*graph) || !decode_signature(*signature) ||
            !reject_unknown(value, path, "graph", "signature", "module_call_graph", "metadata", "treespec_namedtuple_fields"))
            return false;
        return true;
    }

    bool decode_ranges(const Pt2JsonValue& value)
    {
        const std::string path = "$.range_constraints";
        if (!require_type(value, Pt2JsonValue::Object, path, "object"))
            return false;
        for (std::map<std::string, Pt2JsonValue>::const_iterator it = value.object.begin(); it != value.object.end(); ++it)
        {
            const std::string item_path = path + "." + it->first;
            const Pt2JsonValue* min_value = field(it->second, "min_val", item_path, true);
            const Pt2JsonValue* max_value = field(it->second, "max_val", item_path, true);
            if (!min_value || !max_value)
                return false;
            Pt2RangeConstraint range;
            range.has_min = min_value->type != Pt2JsonValue::Null;
            range.has_max = max_value->type != Pt2JsonValue::Null;
            range.min = 0;
            range.max = 0;
            if ((range.has_min && !get_int(*min_value, item_path + ".min_val", range.min)) ||
                (range.has_max && !get_int(*max_value, item_path + ".max_val", range.max)) ||
                !reject_unknown(it->second, item_path, "min_val", "max_val"))
                return false;
            program.range_constraints[it->first] = range;
        }
        return true;
    }

    bool same_argument(const Pt2Argument& a, const Pt2Argument& b) const
    {
        if (a.type != b.type)
            return false;
        if (a.type == Pt2Argument::Tensor || a.type == Pt2Argument::String)
            return a.s == b.s;
        if (a.type == Pt2Argument::Device)
            return a.s == b.s && a.i == b.i;
        if (a.type == Pt2Argument::OptionalTensor)
            return a.b == b.b && (!a.b || a.s == b.s);
        if (a.type == Pt2Argument::Int || a.type == Pt2Argument::ScalarType || a.type == Pt2Argument::MemoryFormat || a.type == Pt2Argument::Layout)
            return a.i == b.i;
        if (a.type == Pt2Argument::Float)
            return a.f == b.f || (std::isnan(a.f) && std::isnan(b.f));
        if (a.type == Pt2Argument::Bool || a.type == Pt2Argument::None)
            return a.b == b.b;
        if (a.type == Pt2Argument::Tensors || a.type == Pt2Argument::OptionalTensors || a.type == Pt2Argument::Strings)
            return a.as == b.as;
        if (a.type == Pt2Argument::Ints)
            return a.ai == b.ai;
        if (a.type == Pt2Argument::Floats || a.type == Pt2Argument::Complex)
        {
            if (a.af.size() != b.af.size())
                return false;
            for (size_t i = 0; i < a.af.size(); i++)
            {
                if (a.af[i] != b.af[i] && !(std::isnan(a.af[i]) && std::isnan(b.af[i])))
                    return false;
            }
            return true;
        }
        if (a.type == Pt2Argument::Bools)
            return a.ab == b.ab;
        if (a.type == Pt2Argument::SymInt || a.type == Pt2Argument::SymBool)
            return a.b == b.b && (a.b ? a.s == b.s : a.i == b.i);
        if (a.type == Pt2Argument::SymInts)
        {
            if (a.args.size() != b.args.size())
                return false;
            for (size_t i = 0; i < a.args.size(); i++)
            {
                if (!same_argument(a.args[i], b.args[i]))
                    return false;
            }
            return true;
        }
        return true;
    }

    bool verify_argument(const Pt2Argument& arg, const std::set<std::string>& values, const std::string& path)
    {
        if (arg.type == Pt2Argument::Tensor && values.find(arg.s) == values.end())
        {
            fail(path, 0, "unknown tensor value " + arg.s);
            return false;
        }
        if (arg.type == Pt2Argument::OptionalTensor && arg.b && values.find(arg.s) == values.end())
        {
            fail(path, 0, "unknown tensor value " + arg.s);
            return false;
        }
        if ((arg.type == Pt2Argument::SymInt || arg.type == Pt2Argument::SymBool) && arg.b && values.find(arg.s) == values.end())
        {
            fail(path, 0, "unknown symbolic value " + arg.s);
            return false;
        }
        if (arg.type == Pt2Argument::SymInts)
        {
            for (size_t i = 0; i < arg.args.size(); i++)
            {
                if (!verify_argument(arg.args[i], values, path + "[" + std::to_string(i) + "]"))
                    return false;
            }
        }
        if (arg.type == Pt2Argument::Tensors || arg.type == Pt2Argument::OptionalTensors)
        {
            for (size_t i = 0; i < arg.as.size(); i++)
            {
                if (!arg.as[i].empty() && values.find(arg.as[i]) == values.end())
                {
                    fail(path, 0, "unknown tensor value " + arg.as[i]);
                    return false;
                }
            }
        }
        return true;
    }

    int verify()
    {
        if (program.inputs.size() != program.input_specs.size())
            return fail("$.graph_module.signature.input_specs", 0, "input signature count does not match graph inputs");
        if (program.outputs.size() != program.output_specs.size())
            return fail("$.graph_module.signature.output_specs", 0, "output signature count does not match graph outputs");

        std::set<std::string> values;
        for (size_t i = 0; i < program.inputs.size(); i++)
        {
            const Pt2Argument& arg = program.inputs[i];
            if (arg.type == Pt2Argument::Tensor)
            {
                if (!values.insert(arg.s).second || program.tensors.find(arg.s) == program.tensors.end())
                    return fail("$.graph_module.graph.inputs[" + std::to_string(i) + "]", 0, "invalid tensor input " + arg.s);
            }
            if (!same_argument(arg, program.input_specs[i].arg))
                return fail("$.graph_module.signature.input_specs[" + std::to_string(i) + "]", 0, "signature input does not match graph input");
        }

        for (size_t i = 0; i < program.nodes.size(); i++)
        {
            const Pt2Node& node = program.nodes[i];
            const std::string path = "$.graph_module.graph.nodes[" + std::to_string(i) + "]";
            if (node.target.compare(0, 10, "torch.ops.") != 0 && node.target.compare(0, 10, "_operator.") != 0)
                return fail(path + ".target", 0, "unsupported operator target");
            for (size_t j = 0; j < node.inputs.size(); j++)
            {
                if (!verify_argument(node.inputs[j].arg, values, path + ".inputs[" + std::to_string(j) + "]"))
                    return -1;
            }
            for (size_t j = 0; j < node.outputs.size(); j++)
            {
                const Pt2Argument& output = node.outputs[j];
                if (output.type == Pt2Argument::Tensor)
                {
                    if (!values.insert(output.s).second || program.tensors.find(output.s) == program.tensors.end())
                        return fail(path + ".outputs[" + std::to_string(j) + "]", 0, "invalid tensor output");
                }
                else if (output.type == Pt2Argument::Tensors)
                {
                    for (size_t k = 0; k < output.as.size(); k++)
                    {
                        if (!values.insert(output.as[k]).second || program.tensors.find(output.as[k]) == program.tensors.end())
                            return fail(path + ".outputs[" + std::to_string(j) + "]", 0, "invalid tensor list output");
                    }
                }
                else if (output.type == Pt2Argument::SymInt && output.b)
                {
                    if (!values.insert(output.s).second || program.sym_ints.find(output.s) == program.sym_ints.end())
                        return fail(path + ".outputs[" + std::to_string(j) + "]", 0, "invalid symbolic integer output");
                }
                else if (output.type == Pt2Argument::SymBool && output.b)
                {
                    if (!values.insert(output.s).second || program.sym_bools.find(output.s) == program.sym_bools.end())
                        return fail(path + ".outputs[" + std::to_string(j) + "]", 0, "invalid symbolic bool output");
                }
                else if (output.type != Pt2Argument::None)
                    return fail(path + ".outputs[" + std::to_string(j) + "]", 0, "invalid tensor output");
            }
        }
        for (size_t i = 0; i < program.outputs.size(); i++)
        {
            if (!verify_argument(program.outputs[i], values, "$.graph_module.graph.outputs[" + std::to_string(i) + "]"))
                return -1;
            if (!same_argument(program.outputs[i], program.output_specs[i].arg))
                return fail("$.graph_module.signature.output_specs[" + std::to_string(i) + "]", 0, "signature output does not match graph output");
        }

        for (std::map<std::string, Pt2Tensor>::const_iterator it = program.tensors.begin(); it != program.tensors.end(); ++it)
        {
            const Pt2Tensor& tensor = it->second;
            const std::string path = "$.graph_module.graph.tensor_values." + it->first;
            if (tensor.dtype < 1 || tensor.dtype > 13)
                return fail(path + ".dtype", 0, "unsupported scalar type");
            if (tensor.layout != 7 || tensor.device != "cpu" || tensor.device_index != -1)
                return fail(path, 0, "only dense strided CPU tensors are supported");
            if (tensor.sizes.size() != tensor.strides.size())
                return fail(path, 0, "size and stride rank mismatch");
            if ((!tensor.storage_offset.symbolic && tensor.storage_offset.value < 0))
                return fail(path + ".storage_offset", 0, "negative storage offset");
            for (size_t i = 0; i < tensor.sizes.size(); i++)
            {
                if ((!tensor.sizes[i].symbolic && tensor.sizes[i].value < 0) || (!tensor.strides[i].symbolic && tensor.strides[i].value < 0))
                    return fail(path, 0, "negative size or stride");
            }
        }
        for (std::map<std::string, Pt2RangeConstraint>::const_iterator it = program.range_constraints.begin(); it != program.range_constraints.end(); ++it)
        {
            if (it->second.has_min && it->second.has_max && it->second.min > it->second.max)
                return fail("$.range_constraints." + it->first, 0, "minimum exceeds maximum");
        }
        return 0;
    }

    Pt2Program& program;
};

int parse_pt2_program(const unsigned char* data, size_t size, Pt2Program& program)
{
    program = Pt2Program();
    Pt2JsonValue root;
    if (parse_pt2_json(data, size, root, program.error) != 0)
        return -1;
    Pt2ProgramDecoder decoder(program);
    return decoder.decode(root);
}

int decode_pt2_tensor_meta(const Pt2JsonValue& value, const std::string& path, Pt2Tensor& tensor, std::string& error)
{
    Pt2Program program;
    Pt2ProgramDecoder decoder(program);
    const int result = decoder.decode_tensor_meta(value, path, tensor);
    error = program.error;
    return result;
}

int load_pt2_program(Pt2ArchiveReader& archive, Pt2Program& program)
{
    std::vector<unsigned char> data;
    if (archive.read_file(archive.model_record, data) != 0)
    {
        program.error = archive.error;
        return -1;
    }
    if (parse_pt2_program(data.empty() ? 0 : &data[0], data.size(), program) != 0)
    {
        program.error = archive.model_record + ": " + program.error;
        return -1;
    }
    if (archive.container_kind == Pt2ContainerLegacyExportedProgram &&
        archive.archive_version != std::to_string(program.schema_major) + "." + std::to_string(program.schema_minor))
    {
        program.error = archive.model_record + ": schema version does not match legacy archive version " + archive.archive_version;
        return -1;
    }
    return 0;
}

} // namespace pnnx
