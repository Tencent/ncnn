// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pt2_operator.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>

#include <cmath>
#include <exception>
#include <utility>

#include <stdio.h>

namespace pnnx {

static bool is_identifier(const std::string& s)
{
    if (s.empty())
        return false;

    for (size_t i = 0; i < s.size(); i++)
    {
        const char c = s[i];
        if (c != '_' && !(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') && !(i > 0 && c >= '0' && c <= '9'))
            return false;
    }
    return true;
}

static JsonValue argument_value(const std::string& tag, const JsonValue& value)
{
    JsonValue arg;
    arg.type = JsonValue::Object;
    arg.o[tag] = value;
    return arg;
}

static bool is_none(const JsonValue& arg)
{
    const JsonValue* value = arg.get("as_none");
    return arg.type == JsonValue::Object && arg.o.size() == 1 && value && value->type == JsonValue::Bool && value->b;
}

static bool is_float(const JsonValue& value)
{
    return value.type == JsonValue::Float || value.type == JsonValue::Int
           || (value.type == JsonValue::String && (value.s == "NaN" || value.s == "Infinity" || value.s == "-Infinity"));
}

static bool is_tensor(const JsonValue& value)
{
    const JsonValue* name = value.get("name");
    return value.type == JsonValue::Object && value.o.size() == 1 && name && name->type == JsonValue::String && !name->s.empty();
}

static bool is_symbol(const JsonValue& value, const char* literal, JsonValue::Type type)
{
    if (value.type != JsonValue::Object || value.o.size() != 1)
        return false;
    const JsonValue* name = value.get("as_name");
    if (name)
        return name->type == JsonValue::String && !name->s.empty();
    const JsonValue* v = value.get(literal);
    return v && (type == JsonValue::Float ? is_float(*v) : v->type == type);
}

static bool is_number(const std::string& tag, const JsonValue& value)
{
    if (tag == "as_int")
        return value.type == JsonValue::Int;
    if (tag == "as_float")
        return is_float(value);
    if (tag == "as_bool")
        return value.type == JsonValue::Bool;
    if (tag == "as_sym_int")
        return is_symbol(value, "as_int", JsonValue::Int);
    if (tag == "as_sym_float")
        return is_symbol(value, "as_float", JsonValue::Float);
    if (tag == "as_sym_bool")
        return is_symbol(value, "as_bool", JsonValue::Bool);
    if (tag == "as_complex")
    {
        const JsonValue* real = value.get("real");
        const JsonValue* imag = value.get("imag");
        return value.type == JsonValue::Object && value.o.size() == 2 && real && imag && is_float(*real) && is_float(*imag);
    }
    return false;
}

static bool allows_numbers_as_tensors(const std::string& name)
{
    // Python bindings allow numbers for these operator bases, including Tensor overloads.
    static const char* const names[] = {
        "aten::_conj", "aten::_to_copy", "aten::add", "aten::add_", "aten::copy", "aten::copy_",
        "aten::div", "aten::div_", "aten::divide", "aten::divide_", "aten::floor_divide", "aten::floor_divide_",
        "aten::mul", "aten::mul_", "aten::multiply", "aten::multiply_", "aten::sub", "aten::sub_",
        "aten::subtract", "aten::subtract_", "aten::to", "aten::true_divide", "aten::true_divide_"
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
    {
        if (name == names[i])
            return true;
    }
    return false;
}

static const char* list_element_tag(const std::string& tag)
{
    if (tag == "as_tensors")
        return "as_tensor";
    if (tag == "as_optional_tensors")
        return "";
    if (tag == "as_ints")
        return "as_int";
    if (tag == "as_floats")
        return "as_float";
    if (tag == "as_bools")
        return "as_bool";
    if (tag == "as_strings")
        return "as_string";
    if (tag == "as_sym_ints")
        return "as_sym_int";
    if (tag == "as_sym_floats")
        return "as_sym_float";
    if (tag == "as_sym_bools")
        return "as_sym_bool";
    return 0;
}

static bool matches_type(const JsonValue& arg, const c10::TypePtr& type, bool allow_number, bool output, int depth)
{
    if (depth > 16 || arg.type != JsonValue::Object || arg.o.size() != 1)
        return false;

    const std::string& tag = arg.o.begin()->first;
    const JsonValue& value = arg.o.begin()->second;
    const c10::TypeKind kind = type->kind();

    if (tag == "as_optional_tensor")
    {
        if (kind != c10::TypeKind::OptionalType || type->cast<c10::OptionalType>()->getElementType()->kind() != c10::TypeKind::TensorType)
            return false;
        if (!value.get("as_tensor") && !value.get("as_none"))
            return false;
        return matches_type(value, type, false, output, depth + 1);
    }
    if (kind == c10::TypeKind::OptionalType)
        return is_none(arg) || matches_type(arg, type->cast<c10::OptionalType>()->getElementType(), allow_number, output, depth + 1);
    if (kind == c10::TypeKind::UnionType)
    {
        for (const c10::TypePtr& t : type->cast<c10::UnionType>()->containedTypes())
        {
            if (matches_type(arg, t, false, output, depth + 1))
                return true;
        }
        return false;
    }
    if (kind == c10::TypeKind::ListType)
    {
        const char* element_tag = list_element_tag(tag);
        if (!element_tag || value.type != JsonValue::Array)
            return false;
        const c10::TypePtr element = type->cast<c10::ListType>()->getElementType();
        // Check the tag even for an empty list; [] does not erase its element type.
        if (tag == "as_optional_tensors" && (element->kind() != c10::TypeKind::OptionalType || element->cast<c10::OptionalType>()->getElementType()->kind() != c10::TypeKind::TensorType))
            return false;
        if (value.a.empty())
        {
            if (tag == "as_optional_tensors")
                return true;
            JsonValue sample;
            if (tag == "as_tensors")
                sample = argument_value("name", JsonValue(std::string("_")));
            if (tag == "as_ints")
                sample = JsonValue((int64_t)0);
            if (tag == "as_floats")
                sample = JsonValue(0.0);
            if (tag == "as_bools")
                sample = JsonValue(false);
            if (tag == "as_strings")
                sample = JsonValue(std::string());
            if (tag == "as_sym_ints" || tag == "as_sym_floats" || tag == "as_sym_bools")
                sample = argument_value("as_name", JsonValue(std::string("_")));
            return matches_type(argument_value(element_tag, sample), element, false, output, depth + 1);
        }
        for (const JsonValue& v : value.a)
        {
            if (tag == "as_optional_tensors" && !v.get("as_tensor") && !v.get("as_none"))
                return false;
            if (!matches_type(*element_tag ? argument_value(element_tag, v) : v, element, false, output, depth + 1))
                return false;
        }
        return true;
    }
    if (kind == c10::TypeKind::NoneType)
        return is_none(arg);
    if (kind == c10::TypeKind::TensorType)
        return (tag == "as_tensor" && is_tensor(value)) || (allow_number && is_number(tag, value)) || (output && is_none(arg));
    if (kind == c10::TypeKind::NumberType)
        return is_number(tag, value) || (output && is_none(arg));
    if (kind == c10::TypeKind::IntType || kind == c10::TypeKind::SymIntType)
        return (tag == "as_int" && value.type == JsonValue::Int) || (tag == "as_sym_int" && is_symbol(value, "as_int", JsonValue::Int));
    if (kind == c10::TypeKind::FloatType || kind == c10::TypeKind::SymFloatType)
        return (tag == "as_float" && is_float(value)) || (tag == "as_sym_float" && is_symbol(value, "as_float", JsonValue::Float));
    if (kind == c10::TypeKind::BoolType || kind == c10::TypeKind::SymBoolType)
        return (tag == "as_bool" && value.type == JsonValue::Bool) || (tag == "as_sym_bool" && is_symbol(value, "as_bool", JsonValue::Bool));
    if (kind == c10::TypeKind::ComplexType)
        return tag == "as_complex" && is_number(tag, value);
    if (kind == c10::TypeKind::StringType)
        return tag == "as_string" && value.type == JsonValue::String;
    if (kind == c10::TypeKind::ScalarTypeType)
        return tag == "as_scalar_type" && value.type == JsonValue::Int && value.i >= 1 && value.i <= 13;
    if (kind == c10::TypeKind::MemoryFormatType)
        return tag == "as_memory_format" && value.type == JsonValue::Int && value.i >= 1 && value.i <= 4;
    if (kind == c10::TypeKind::LayoutType)
        return tag == "as_layout" && value.type == JsonValue::Int && value.i >= 1 && value.i <= 7;
    if (kind == c10::TypeKind::DeviceObjType && tag == "as_device")
    {
        const JsonValue* device_type = value.get("type");
        const JsonValue* index = value.get("index");
        if (value.type != JsonValue::Object || !device_type || device_type->type != JsonValue::String || !is_identifier(device_type->s)
                || value.o.size() != (index ? 2 : 1))
            return false;
        if (index && index->type != JsonValue::Null && (index->type != JsonValue::Int || index->i < 0 || index->i > 127))
            return false;
        const std::string device_name = device_type->s + (index && index->type == JsonValue::Int ? ":" + std::to_string(index->i) : "");
        const c10::Device device(device_name);
        return !device_type->s.empty() && device.type() != c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES;
    }
    return false;
}

static JsonValue float_value(double v)
{
    if (std::isnan(v))
        return JsonValue(std::string("NaN"));
    if (std::isinf(v))
        return JsonValue(std::string(v < 0 ? "-Infinity" : "Infinity"));
    return JsonValue(v);
}

static int scalar_type_value(c10::ScalarType type)
{
    if (type == c10::ScalarType::Byte) return 1;
    if (type == c10::ScalarType::Char) return 2;
    if (type == c10::ScalarType::Short) return 3;
    if (type == c10::ScalarType::Int) return 4;
    if (type == c10::ScalarType::Long) return 5;
    if (type == c10::ScalarType::Half) return 6;
    if (type == c10::ScalarType::Float) return 7;
    if (type == c10::ScalarType::Double) return 8;
    if (type == c10::ScalarType::ComplexHalf) return 9;
    if (type == c10::ScalarType::ComplexFloat) return 10;
    if (type == c10::ScalarType::ComplexDouble) return 11;
    if (type == c10::ScalarType::Bool) return 12;
    if (type == c10::ScalarType::BFloat16) return 13;
    return 0;
}

static bool default_argument(const c10::IValue& value, const c10::TypePtr& type, JsonValue& arg, int depth)
{
    if (depth > 16)
        return false;
    if (value.isNone())
    {
        arg = argument_value("as_none", JsonValue(true));
        return true;
    }
    if (type->kind() == c10::TypeKind::OptionalType)
        return default_argument(value, type->cast<c10::OptionalType>()->getElementType(), arg, depth + 1);

    const c10::TypeKind kind = type->kind();
    if (value.isBool())
        arg = argument_value("as_bool", JsonValue(value.toBool()));
    else if (value.isInt())
    {
        const char* tag = "as_int";
        int64_t v = value.toInt();
        if (kind == c10::TypeKind::ScalarTypeType)
        {
            tag = "as_scalar_type";
            v = scalar_type_value(value.toScalarType());
        }
        if (kind == c10::TypeKind::LayoutType)
        {
            tag = "as_layout";
            const c10::Layout layout = value.toLayout();
            switch (layout)
            {
            case c10::Layout::Sparse:
                v = 1;
                break;
            case c10::Layout::SparseCsr:
                v = 2;
                break;
            case c10::Layout::SparseCsc:
                v = 3;
                break;
            case c10::Layout::SparseBsr:
                v = 4;
                break;
            case c10::Layout::SparseBsc:
                v = 5;
                break;
            case c10::Layout::Mkldnn:
                v = 6;
                break;
            case c10::Layout::Strided:
                v = 7;
                break;
            default:
                v = 0;
                break;
            }
        }
        if (kind == c10::TypeKind::MemoryFormatType)
        {
            tag = "as_memory_format";
            const c10::MemoryFormat format = value.toMemoryFormat();
            switch (format)
            {
            case c10::MemoryFormat::Contiguous:
                v = 1;
                break;
            case c10::MemoryFormat::ChannelsLast:
                v = 2;
                break;
            case c10::MemoryFormat::ChannelsLast3d:
                v = 3;
                break;
            case c10::MemoryFormat::Preserve:
                v = 4;
                break;
            default:
                v = 0;
                break;
            }
        }
        arg = argument_value(tag, JsonValue(v));
    }
    else if (value.isDouble())
        arg = argument_value("as_float", float_value(value.toDouble()));
    else if (value.isComplexDouble())
    {
        const c10::complex<double> v = value.toComplexDouble();
        JsonValue complex;
        complex.type = JsonValue::Object;
        complex.o["real"] = float_value(v.real());
        complex.o["imag"] = float_value(v.imag());
        arg = argument_value("as_complex", complex);
    }
    else if (value.isString())
        arg = argument_value("as_string", JsonValue(value.toStringRef()));
    else if (value.isDevice())
    {
        const c10::Device d = value.toDevice();
        JsonValue device;
        device.type = JsonValue::Object;
        device.o["type"] = JsonValue(c10::DeviceTypeName(d.type(), true));
        device.o["index"] = d.has_index() ? JsonValue((int64_t)d.index()) : JsonValue();
        arg = argument_value("as_device", device);
    }
    else if (value.isList() && kind == c10::TypeKind::ListType)
    {
        const c10::TypePtr element = type->cast<c10::ListType>()->getElementType();
        const c10::TypeKind element_kind = element->kind();
        const char* tag = 0;
        if (element_kind == c10::TypeKind::IntType || element_kind == c10::TypeKind::SymIntType)
            tag = "as_ints";
        if (element_kind == c10::TypeKind::FloatType)
            tag = "as_floats";
        if (element_kind == c10::TypeKind::BoolType)
            tag = "as_bools";
        if (element_kind == c10::TypeKind::StringType)
            tag = "as_strings";
        if (element_kind == c10::TypeKind::TensorType)
            tag = "as_tensors";
        if (element_kind == c10::TypeKind::OptionalType && element->cast<c10::OptionalType>()->getElementType()->kind() == c10::TypeKind::TensorType)
            tag = "as_optional_tensors";
        if (!tag)
            return false;
        JsonValue list;
        list.type = JsonValue::Array;
        for (const c10::IValue& v : value.toListRef())
        {
            JsonValue item;
            if (!default_argument(v, element, item, depth + 1))
                return false;
            list.a.push_back(element_kind == c10::TypeKind::OptionalType ? item : item.o.begin()->second);
        }
        arg = argument_value(tag, list);
    }
    else
        return false;

    return matches_type(arg, type, false, false, 0);
}

static int bind_operator(const JsonValue& node, const c10::FunctionSchema& schema, Pt2Operator& op)
{
    const JsonValue* inputs = node.get("inputs");
    const JsonValue* outputs = node.get("outputs");
    if (!inputs || inputs->type != JsonValue::Array || !outputs || outputs->type != JsonValue::Array)
    {
        fprintf(stderr, "pnnx: pt2 operator: inputs and outputs must be arrays\n");
        return -1;
    }
    if (schema.is_vararg() || schema.is_varret())
    {
        fprintf(stderr, "pnnx: pt2 operator: variadic schema is unsupported\n");
        return -1;
    }

    const std::vector<c10::Argument>& arguments = schema.arguments();
    const std::vector<c10::Argument>& returns = schema.returns();
    std::vector<const JsonValue*> bound(arguments.size(), 0);
    size_t positional_index = 0;
    bool saw_keyword = false;
    for (const JsonValue& input : inputs->a)
    {
        const JsonValue* name = input.get("name");
        const JsonValue* arg = input.get("arg");
        const JsonValue* kind = input.get("kind");
        if (!name || name->type != JsonValue::String || !arg || !kind || kind->type != JsonValue::Int || (kind->i != 1 && kind->i != 2))
        {
            fprintf(stderr, "pnnx: pt2 operator: input must have name, arg and positional/keyword kind\n");
            return -1;
        }
        size_t index = 0;
        while (index < arguments.size() && arguments[index].name() != name->s)
        {
            index++;
        }
        if (index == arguments.size() || bound[index])
        {
            fprintf(stderr, "pnnx: pt2 operator: unknown or duplicate argument %s\n", name->s.c_str());
            return -1;
        }
        if (kind->i == 1)
        {
            if (saw_keyword || arguments[index].kwarg_only() || index != positional_index)
            {
                fprintf(stderr, "pnnx: pt2 operator: invalid positional argument %s\n", name->s.c_str());
                return -1;
            }
            positional_index++;
        }
        else
            saw_keyword = true;
        bound[index] = arg;
    }

    const bool allow_number = allows_numbers_as_tensors(op.type);
    for (size_t i = 0; i < arguments.size(); i++)
    {
        Pt2Argument input;
        input.name = arguments[i].name();
        if (bound[i])
            input.value = *bound[i];
        else
        {
            if (!arguments[i].default_value())
            {
                fprintf(stderr, "pnnx: pt2 operator: missing required argument %s\n", input.name.c_str());
                return -1;
            }
            if (!default_argument(*arguments[i].default_value(), arguments[i].real_type(), input.value, 0))
            {
                fprintf(stderr, "pnnx: pt2 operator: unsupported default for argument %s\n", input.name.c_str());
                return -1;
            }
        }
        if (!matches_type(input.value, arguments[i].real_type(), allow_number, false, 0))
        {
            fprintf(stderr, "pnnx: pt2 operator: argument %s is incompatible with %s\n", input.name.c_str(), arguments[i].real_type()->str().c_str());
            return -1;
        }
        op.inputs.push_back(input);
        if (arguments[i].real_type()->kind() == c10::TypeKind::TensorType)
            op.tensor_inputs.push_back(i);
    }

    if (outputs->a.size() != returns.size())
    {
        fprintf(stderr, "pnnx: pt2 operator: return count does not match schema\n");
        return -1;
    }
    for (size_t i = 0; i < returns.size(); i++)
    {
        if (!matches_type(outputs->a[i], returns[i].real_type(), false, true, 0))
        {
            fprintf(stderr, "pnnx: pt2 operator: return %zu is incompatible with %s\n", i, returns[i].real_type()->str().c_str());
            return -1;
        }
    }
    op.outputs = outputs->a;
    op.aliases.resize(returns.size());
    for (size_t i = 0; i < arguments.size(); i++)
    {
        const c10::SchemaArgument input(c10::SchemaArgType::input, i);
        if (schema.is_mutable(input))
            op.writes.push_back(i);
        for (size_t j = 0; j < returns.size(); j++)
        {
            if (schema.may_contain_alias(c10::SchemaArgument(c10::SchemaArgType::output, j), input, true))
                op.aliases[j].push_back(i);
        }
    }
    return 0;
}

int bind_pt2_operator(const JsonValue& node, Pt2Operator& op)
{
    op = Pt2Operator();
    const JsonValue* target = node.get("target");
    if (!target || target->type != JsonValue::String)
    {
        fprintf(stderr, "pnnx: pt2 operator: pt2 node target must be a string\n");
        return -1;
    }
    const std::string prefix = "torch.ops.aten.";
    const size_t dot = target->s.rfind('.');
    if (target->s.compare(0, prefix.size(), prefix) != 0 || dot < prefix.size()
            || !is_identifier(target->s.substr(prefix.size(), dot - prefix.size())) || !is_identifier(target->s.substr(dot + 1)))
    {
        fprintf(stderr, "pnnx: pt2 operator: unsupported pt2 target %s\n", target->s.c_str());
        return -1;
    }

    Pt2Operator result;
    result.type = "aten::" + target->s.substr(prefix.size(), dot - prefix.size());
    std::string overload = target->s.substr(dot + 1);
    if (result.type == "aten::view" && overload == "dtype")
    {
        // The shared aten::view rewrite expects a shape, not a dtype.
        fprintf(stderr, "pnnx: pt2 operator: %s: dtype reinterpretation is unsupported\n", target->s.c_str());
        return -1;
    }
    if (overload == "default")
        overload.clear();
    try
    {
        const c10::OperatorHandle handle = c10::Dispatcher::singleton().findSchemaOrThrow(result.type.c_str(), overload.c_str());
        if (bind_operator(node, handle.schema(), result) == 0)
        {
            std::swap(op, result);
            return 0;
        }
    }
    catch (const c10::Error& e)
    {
        fprintf(stderr, "pnnx: pt2 operator: %s\n", e.what_without_backtrace());
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "pnnx: pt2 operator: %s\n", e.what());
    }
    return -1;
}

} // namespace pnnx
