// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program_operator.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <c10/core/DeviceType.h>
#include <torch/csrc/api/include/torch/version.h>
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 9)
#include <torch/csrc/jit/operator_upgraders/utils.h>
#endif

#include <exception>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace pnnx {

static bool is_identifier(const std::string& value)
{
    if (value.empty())
        return false;

    const unsigned char first = (unsigned char)value[0];
    if (!((first >= 'a' && first <= 'z') || (first >= 'A' && first <= 'Z') || first == '_'))
        return false;

    for (size_t i = 1; i < value.size(); i++)
    {
        const unsigned char ch = (unsigned char)value[i];
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_'))
            return false;
    }

    return true;
}

int parse_exported_operator_target(const std::string& target, ExportedOperatorTarget& result, std::string& error)
{
    result = ExportedOperatorTarget();
    error.clear();

    const std::string prefix = "torch.ops.";
    if (target.compare(0, prefix.size(), prefix) != 0)
    {
        error = "exported operator target must start with torch.ops.";
        return -1;
    }

    const std::string qualified_name = target.substr(prefix.size());
    const size_t namespace_separator = qualified_name.find('.');
    const size_t overload_separator = qualified_name.rfind('.');
    if (namespace_separator == std::string::npos || namespace_separator == overload_separator)
    {
        error = "exported operator target must contain namespace, operator and overload";
        return -1;
    }

    const std::string namespace_part = qualified_name.substr(0, namespace_separator);
    const std::string operator_part = qualified_name.substr(namespace_separator + 1, overload_separator - namespace_separator - 1);
    const std::string overload_part = qualified_name.substr(overload_separator + 1);
    if (!is_identifier(namespace_part))
    {
        error = "invalid namespace in exported target";
        return -1;
    }
    if (!is_identifier(operator_part))
    {
        error = "invalid operator name in exported target";
        return -1;
    }
    if (!is_identifier(overload_part))
    {
        error = "invalid overload name in exported target";
        return -1;
    }

    ExportedOperatorTarget parsed_target;
    parsed_target.namespace_name = namespace_part;
    parsed_target.operator_name = namespace_part + "::" + operator_part;
    if (overload_part != "default")
        parsed_target.overload_name = overload_part;

    result = parsed_target;
    return 0;
}

static std::string operator_context(const ExportedProgramHeader& header, const ExportedNode& node)
{
    std::ostringstream context;
    context << "torch " << (header.torch_version.empty() ? "unknown" : header.torch_version) << " aten opset ";

    const std::map<std::string, int64_t>::const_iterator aten_opset = header.opset_version.find("aten");
    if (aten_opset == header.opset_version.end())
        context << "missing";
    else
        context << aten_opset->second;

    context << " target " << node.target << ": ";
    return context.str();
}

int validate_exported_program_opset(const ExportedProgramHeader& header, std::string& error)
{
    error.clear();

    const std::map<std::string, int64_t>::const_iterator aten_opset = header.opset_version.find("aten");
    if (aten_opset == header.opset_version.end())
    {
        std::ostringstream message;
        message << "torch " << (header.torch_version.empty() ? "unknown" : header.torch_version) << " aten opset missing: missing aten opset";
        error = message.str();
        return -1;
    }

#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 9)
    std::ostringstream message;
    message << "torch " << (header.torch_version.empty() ? "unknown" : header.torch_version) << " aten opset " << aten_opset->second << ": exported program operator schemas require libtorch 2.9 or newer";
    error = message.str();
    return -1;
#else
    const uint64_t linked_opset = torch::jit::getMaxOperatorVersion();
    if (aten_opset->second < 0 || (uint64_t)aten_opset->second != linked_opset)
    {
        std::ostringstream message;
        message << "torch " << (header.torch_version.empty() ? "unknown" : header.torch_version) << " aten opset " << aten_opset->second
                << ": archive aten opset " << aten_opset->second << " does not match linked libtorch opset " << linked_opset;
        error = message.str();
        return -1;
    }

    return 0;
#endif
}

static int operator_error(const ExportedProgramHeader& header, const ExportedNode& node, const std::string& message, std::string& error)
{
    error = operator_context(header, node) + message;
    return -1;
}

struct ExportedSchemaArgumentBinding
{
    std::string name;
    bool keyword_only;
};

static int bind_exported_arguments(const ExportedNode& node,
                                   const ExportedProgramHeader& header,
                                   const std::vector<ExportedSchemaArgumentBinding>& schema_arguments,
                                   std::vector<const ExportedNamedArgument*>& bound_arguments,
                                   std::string& error)
{
    std::map<std::string, size_t> schema_indices;
    for (size_t i = 0; i < schema_arguments.size(); i++)
        schema_indices[schema_arguments[i].name] = i;

    bool has_missing_kind = false;
    bool has_explicit_kind = false;
    for (size_t i = 0; i < node.inputs.size(); i++)
    {
        if (node.inputs[i].kind == EXPORTED_ARGUMENT_KIND_UNKNOWN)
            return operator_error(header, node, "unknown argument kind for " + node.inputs[i].name, error);
        if (node.inputs[i].kind == EXPORTED_ARGUMENT_KIND_MISSING)
            has_missing_kind = true;
        else
            has_explicit_kind = true;
    }
    if (has_missing_kind && has_explicit_kind)
        return operator_error(header, node, "node mixes legacy and explicit argument kinds", error);

    bound_arguments.assign(schema_arguments.size(), 0);
    bool saw_keyword = false;
    size_t next_positional = 0;
    for (size_t i = 0; i < node.inputs.size(); i++)
    {
        const ExportedNamedArgument& input = node.inputs[i];
        const std::map<std::string, size_t>::const_iterator schema_index_it = schema_indices.find(input.name);
        if (schema_index_it == schema_indices.end())
            return operator_error(header, node, "unknown argument " + input.name, error);

        const size_t schema_index = schema_index_it->second;
        if (bound_arguments[schema_index])
            return operator_error(header, node, "duplicate argument " + input.name, error);

        if (!has_missing_kind && input.kind == EXPORTED_ARGUMENT_KIND_POSITIONAL)
        {
            if (saw_keyword)
                return operator_error(header, node, "positional argument " + input.name + " follows a keyword argument", error);
            if (schema_arguments[schema_index].keyword_only)
                return operator_error(header, node, "keyword-only argument " + input.name + " was serialized as positional", error);
            if (schema_index != next_positional)
            {
                const std::string expected_name = next_positional < schema_arguments.size() ? schema_arguments[next_positional].name : std::string("<none>");
                return operator_error(header, node, "expected positional argument " + expected_name + " but found " + input.name, error);
            }
            next_positional++;
        }
        else if (!has_missing_kind && input.kind == EXPORTED_ARGUMENT_KIND_KEYWORD)
        {
            saw_keyword = true;
        }

        if (input.arg.type == EXPORTED_ARGUMENT_UNSUPPORTED)
            return operator_error(header, node, "unsupported serialized argument " + input.arg.unsupported_tag + " for " + input.name, error);

        bound_arguments[schema_index] = &input;
    }

    return 0;
}

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 9)

static c10::TypePtr unwrap_optional_type(const c10::TypePtr& type)
{
    const c10::OptionalTypePtr optional_type = type->cast<c10::OptionalType>();
    if (optional_type)
        return optional_type->getElementType();
    return type;
}

static const char* exported_argument_type_name(ExportedArgumentType type)
{
    if (type == EXPORTED_ARGUMENT_NONE)
        return "none";
    if (type == EXPORTED_ARGUMENT_TENSOR)
        return "tensor";
    if (type == EXPORTED_ARGUMENT_TENSOR_LIST)
        return "tensor list";
    if (type == EXPORTED_ARGUMENT_INT)
        return "int";
    if (type == EXPORTED_ARGUMENT_INT_LIST)
        return "int list";
    if (type == EXPORTED_ARGUMENT_FLOAT)
        return "float";
    if (type == EXPORTED_ARGUMENT_FLOAT_LIST)
        return "float list";
    if (type == EXPORTED_ARGUMENT_COMPLEX)
        return "complex";
    if (type == EXPORTED_ARGUMENT_BOOL)
        return "bool";
    if (type == EXPORTED_ARGUMENT_BOOL_LIST)
        return "bool list";
    if (type == EXPORTED_ARGUMENT_STRING)
        return "string";
    if (type == EXPORTED_ARGUMENT_STRING_LIST)
        return "string list";
    if (type == EXPORTED_ARGUMENT_SCALAR_TYPE)
        return "scalar type";
    if (type == EXPORTED_ARGUMENT_MEMORY_FORMAT)
        return "memory format";
    if (type == EXPORTED_ARGUMENT_LAYOUT)
        return "layout";
    if (type == EXPORTED_ARGUMENT_DEVICE)
        return "device";
    if (type == EXPORTED_ARGUMENT_GRAPH)
        return "graph";
    return "unsupported";
}

static bool exported_list_element_type(ExportedArgumentType list_type, ExportedArgumentType& element_type)
{
    if (list_type == EXPORTED_ARGUMENT_TENSOR_LIST)
        element_type = EXPORTED_ARGUMENT_TENSOR;
    else if (list_type == EXPORTED_ARGUMENT_INT_LIST)
        element_type = EXPORTED_ARGUMENT_INT;
    else if (list_type == EXPORTED_ARGUMENT_FLOAT_LIST)
        element_type = EXPORTED_ARGUMENT_FLOAT;
    else if (list_type == EXPORTED_ARGUMENT_BOOL_LIST)
        element_type = EXPORTED_ARGUMENT_BOOL;
    else if (list_type == EXPORTED_ARGUMENT_STRING_LIST)
        element_type = EXPORTED_ARGUMENT_STRING;
    else
        return false;

    return true;
}

static bool exported_argument_type_is_number(ExportedArgumentType argument_type)
{
    return argument_type == EXPORTED_ARGUMENT_INT || argument_type == EXPORTED_ARGUMENT_FLOAT || argument_type == EXPORTED_ARGUMENT_COMPLEX || argument_type == EXPORTED_ARGUMENT_BOOL;
}

static bool operator_allows_numbers_as_tensors(const std::string& operator_name)
{
    // Mirrors torch::should_allow_numbers_as_tensors in libtorch_python, which
    // pnnx does not link.  The decision is per operator base, not per overload.
    static const char* const allowed_operators[] = {
        "aten::_conj",
        "aten::_to_copy",
        "aten::add",
        "aten::add_",
        "aten::copy",
        "aten::copy_",
        "aten::div",
        "aten::div_",
        "aten::divide",
        "aten::divide_",
        "aten::floor_divide",
        "aten::floor_divide_",
        "aten::mul",
        "aten::mul_",
        "aten::multiply",
        "aten::multiply_",
        "aten::sub",
        "aten::sub_",
        "aten::subtract",
        "aten::subtract_",
        "aten::to",
        "aten::true_divide",
        "aten::true_divide_",
    };

    for (size_t i = 0; i < sizeof(allowed_operators) / sizeof(allowed_operators[0]); i++)
    {
        if (operator_name == allowed_operators[i])
            return true;
    }

    return false;
}

static bool exported_argument_type_matches_schema(ExportedArgumentType argument_type, const c10::TypePtr& schema_type, bool allow_numbers_as_tensors)
{
    if (schema_type->kind() == c10::TypeKind::OptionalType)
    {
        if (argument_type == EXPORTED_ARGUMENT_NONE)
            return true;

        const c10::OptionalTypePtr optional_type = schema_type->cast<c10::OptionalType>();
        return exported_argument_type_matches_schema(argument_type, optional_type->getElementType(), allow_numbers_as_tensors);
    }

    if (schema_type->kind() == c10::TypeKind::UnionType)
    {
        const c10::UnionTypePtr union_type = schema_type->cast<c10::UnionType>();
        const at::ArrayRef<c10::TypePtr> contained_types = union_type->containedTypes();
        for (size_t i = 0; i < contained_types.size(); i++)
        {
            if (exported_argument_type_matches_schema(argument_type, contained_types[i], false))
                return true;
        }
        return false;
    }

    if (schema_type->kind() == c10::TypeKind::ListType)
    {
        ExportedArgumentType element_type;
        if (!exported_list_element_type(argument_type, element_type))
            return false;

        const c10::ListTypePtr list_type = schema_type->cast<c10::ListType>();
        return exported_argument_type_matches_schema(element_type, list_type->getElementType(), false);
    }

    if (schema_type->kind() == c10::TypeKind::NoneType)
        return argument_type == EXPORTED_ARGUMENT_NONE;
    if (schema_type->kind() == c10::TypeKind::TensorType)
        return argument_type == EXPORTED_ARGUMENT_TENSOR || (allow_numbers_as_tensors && exported_argument_type_is_number(argument_type));
    if (schema_type->kind() == c10::TypeKind::NumberType)
        return exported_argument_type_is_number(argument_type);
    if (schema_type->kind() == c10::TypeKind::IntType || schema_type->kind() == c10::TypeKind::SymIntType)
        return argument_type == EXPORTED_ARGUMENT_INT;
    if (schema_type->kind() == c10::TypeKind::FloatType || schema_type->kind() == c10::TypeKind::SymFloatType)
        return argument_type == EXPORTED_ARGUMENT_FLOAT;
    if (schema_type->kind() == c10::TypeKind::ComplexType)
        return argument_type == EXPORTED_ARGUMENT_COMPLEX;
    if (schema_type->kind() == c10::TypeKind::BoolType || schema_type->kind() == c10::TypeKind::SymBoolType)
        return argument_type == EXPORTED_ARGUMENT_BOOL;
    if (schema_type->kind() == c10::TypeKind::StringType)
        return argument_type == EXPORTED_ARGUMENT_STRING;
    if (schema_type->kind() == c10::TypeKind::ScalarTypeType)
        return argument_type == EXPORTED_ARGUMENT_SCALAR_TYPE;
    if (schema_type->kind() == c10::TypeKind::LayoutType)
        return argument_type == EXPORTED_ARGUMENT_LAYOUT;
    if (schema_type->kind() == c10::TypeKind::MemoryFormatType)
        return argument_type == EXPORTED_ARGUMENT_MEMORY_FORMAT;
    if (schema_type->kind() == c10::TypeKind::DeviceObjType)
        return argument_type == EXPORTED_ARGUMENT_DEVICE;

    return false;
}

static int serialize_scalar_type(at::ScalarType value, int64_t& serialized)
{
    if (value == at::ScalarType::Byte)
        serialized = 1;
    else if (value == at::ScalarType::Char)
        serialized = 2;
    else if (value == at::ScalarType::Short)
        serialized = 3;
    else if (value == at::ScalarType::Int)
        serialized = 4;
    else if (value == at::ScalarType::Long)
        serialized = 5;
    else if (value == at::ScalarType::Half)
        serialized = 6;
    else if (value == at::ScalarType::Float)
        serialized = 7;
    else if (value == at::ScalarType::Double)
        serialized = 8;
    else if (value == at::ScalarType::ComplexHalf)
        serialized = 9;
    else if (value == at::ScalarType::ComplexFloat)
        serialized = 10;
    else if (value == at::ScalarType::ComplexDouble)
        serialized = 11;
    else if (value == at::ScalarType::Bool)
        serialized = 12;
    else if (value == at::ScalarType::BFloat16)
        serialized = 13;
    else
        return -1;

    return 0;
}

static int serialize_layout(at::Layout value, int64_t& serialized)
{
    if (value == at::kSparse)
        serialized = 1;
    else if (value == at::kSparseCsr)
        serialized = 2;
    else if (value == at::kSparseCsc)
        serialized = 3;
    else if (value == at::kSparseBsr)
        serialized = 4;
    else if (value == at::kSparseBsc)
        serialized = 5;
    else if (value == at::kMkldnn)
        serialized = 6;
    else if (value == at::kStrided)
        serialized = 7;
    else
        return -1;

    return 0;
}

static int serialize_memory_format(at::MemoryFormat value, int64_t& serialized)
{
    if (value == at::MemoryFormat::Contiguous)
        serialized = 1;
    else if (value == at::MemoryFormat::ChannelsLast)
        serialized = 2;
    else if (value == at::MemoryFormat::ChannelsLast3d)
        serialized = 3;
    else if (value == at::MemoryFormat::Preserve)
        serialized = 4;
    else
        return -1;

    return 0;
}

static int convert_default_value(const c10::Argument& schema_argument, ExportedArgument& argument, std::string& error)
{
    argument = ExportedArgument();
    error.clear();

    if (!schema_argument.default_value())
    {
        error = "argument has no default value";
        return -1;
    }

    const c10::IValue& value = *schema_argument.default_value();
    const c10::TypePtr argument_type = unwrap_optional_type(schema_argument.real_type());

    if (value.isNone())
    {
        argument.type = EXPORTED_ARGUMENT_NONE;
        return 0;
    }
    if (value.isBool())
    {
        argument.type = EXPORTED_ARGUMENT_BOOL;
        argument.bool_value = value.toBool();
        return 0;
    }
    if (value.isInt())
    {
        if (argument_type->kind() == c10::TypeKind::ScalarTypeType)
        {
            argument.type = EXPORTED_ARGUMENT_SCALAR_TYPE;
            if (serialize_scalar_type(value.toScalarType(), argument.enum_value) != 0)
            {
                error = "unsupported ScalarType default";
                return -1;
            }
            return 0;
        }
        if (argument_type->kind() == c10::TypeKind::LayoutType)
        {
            argument.type = EXPORTED_ARGUMENT_LAYOUT;
            if (serialize_layout(value.toLayout(), argument.enum_value) != 0)
            {
                error = "unsupported Layout default";
                return -1;
            }
            return 0;
        }
        if (argument_type->kind() == c10::TypeKind::MemoryFormatType)
        {
            argument.type = EXPORTED_ARGUMENT_MEMORY_FORMAT;
            if (serialize_memory_format(value.toMemoryFormat(), argument.enum_value) != 0)
            {
                error = "unsupported MemoryFormat default";
                return -1;
            }
            return 0;
        }

        argument.type = EXPORTED_ARGUMENT_INT;
        argument.int_value = value.toInt();
        return 0;
    }
    if (value.isDouble())
    {
        argument.type = EXPORTED_ARGUMENT_FLOAT;
        argument.float_value = value.toDouble();
        return 0;
    }
    if (value.isString())
    {
        argument.type = EXPORTED_ARGUMENT_STRING;
        argument.string_value = value.toStringRef();
        return 0;
    }
    if (value.isIntList())
    {
        argument.type = EXPORTED_ARGUMENT_INT_LIST;
        argument.int_values = value.toIntVector();
        return 0;
    }
    if (value.isDoubleList())
    {
        argument.type = EXPORTED_ARGUMENT_FLOAT_LIST;
        argument.float_values = value.toDoubleVector();
        return 0;
    }
    if (value.isBoolList())
    {
        argument.type = EXPORTED_ARGUMENT_BOOL_LIST;
        const c10::List<bool> values = value.toBoolList();
        argument.bool_values.reserve(values.size());
        for (size_t i = 0; i < values.size(); i++)
            argument.bool_values.push_back(values.get(i));
        return 0;
    }
    if (value.isTensorList())
    {
        const c10::List<at::Tensor> values = value.toTensorList();
        if (!values.empty())
        {
            error = "non-empty tensor-list default cannot be represented";
            return -1;
        }
        argument.type = EXPORTED_ARGUMENT_TENSOR_LIST;
        return 0;
    }
    if (value.isList())
    {
        const c10::List<c10::IValue> values = value.toList();
        if (values.elementType()->kind() != c10::TypeKind::StringType)
        {
            error = "generic list default is not a string list";
            return -1;
        }

        bool all_strings = true;
        for (size_t i = 0; i < values.size(); i++)
            all_strings = all_strings && values.get(i).isString();

        if (!all_strings)
        {
            error = "generic list default is not a string list";
            return -1;
        }

        argument.type = EXPORTED_ARGUMENT_STRING_LIST;
        argument.string_values.reserve(values.size());
        for (size_t i = 0; i < values.size(); i++)
            argument.string_values.push_back(values.get(i).toStringRef());
        return 0;
    }
    if (value.isDevice())
    {
        const c10::Device device = value.toDevice();
        argument.type = EXPORTED_ARGUMENT_DEVICE;
        argument.device_value.type = c10::DeviceTypeName(device.type(), true);
        argument.device_value.has_index = device.has_index();
        if (device.has_index())
            argument.device_value.index = static_cast<int64_t>(static_cast<unsigned char>(device.index()));
        return 0;
    }

    error = "unsupported default IValue kind " + value.tagKind();
    return -1;
}

static int canonicalize_with_schema(const ExportedNode& node,
                                    const ExportedProgramHeader& header,
                                    const c10::FunctionSchema& schema,
                                    bool allow_numbers_as_tensors,
                                    std::vector<CanonicalExportedArgument>& result,
                                    std::string& error)
{
    if (schema.is_vararg())
        return operator_error(header, node, "variadic dispatcher schemas are unsupported", error);

    const std::vector<c10::Argument>& schema_arguments = schema.arguments();
    std::vector<ExportedSchemaArgumentBinding> argument_bindings(schema_arguments.size());
    for (size_t i = 0; i < schema_arguments.size(); i++)
    {
        argument_bindings[i].name = schema_arguments[i].name();
        argument_bindings[i].keyword_only = schema_arguments[i].kwarg_only();
    }

    std::vector<const ExportedNamedArgument*> bound_arguments;
    if (bind_exported_arguments(node, header, argument_bindings, bound_arguments, error) != 0)
        return -1;

    std::vector<CanonicalExportedArgument> canonical_arguments;
    canonical_arguments.reserve(schema_arguments.size());
    for (size_t i = 0; i < schema_arguments.size(); i++)
    {
        CanonicalExportedArgument argument;
        argument.name = schema_arguments[i].name();
        if (bound_arguments[i])
        {
            const ExportedArgumentType argument_type = bound_arguments[i]->arg.type;
            const c10::TypePtr& schema_type = schema_arguments[i].real_type();
            if (!exported_argument_type_matches_schema(argument_type, schema_type, allow_numbers_as_tensors))
            {
                return operator_error(header, node, "argument " + argument.name + " has serialized type " + exported_argument_type_name(argument_type) + " incompatible with dispatcher schema " + schema_type->str(), error);
            }
            argument.value = bound_arguments[i]->arg;
        }
        else
        {
            if (!schema_arguments[i].default_value())
                return operator_error(header, node, "missing required argument " + argument.name, error);

            std::string default_error;
            if (convert_default_value(schema_arguments[i], argument.value, default_error) != 0)
                return operator_error(header, node, "cannot materialize default for argument " + argument.name + ": " + default_error, error);
        }
        canonical_arguments.push_back(argument);
    }

    result.swap(canonical_arguments);
    return 0;
}

#endif

enum ExportedCustomArgumentType
{
    EXPORTED_CUSTOM_TENSOR,
    EXPORTED_CUSTOM_INT,
    EXPORTED_CUSTOM_SYM_INT,
    EXPORTED_CUSTOM_FLOAT,
    EXPORTED_CUSTOM_BOOL
};

struct ExportedCustomArgument
{
    const char* name;
    ExportedCustomArgumentType type;
};

static const ExportedCustomArgument torchvision_deform_conv2d_arguments[] = {
    {"input", EXPORTED_CUSTOM_TENSOR},
    {"weight", EXPORTED_CUSTOM_TENSOR},
    {"offset", EXPORTED_CUSTOM_TENSOR},
    {"mask", EXPORTED_CUSTOM_TENSOR},
    {"bias", EXPORTED_CUSTOM_TENSOR},
    {"stride_h", EXPORTED_CUSTOM_SYM_INT},
    {"stride_w", EXPORTED_CUSTOM_SYM_INT},
    {"pad_h", EXPORTED_CUSTOM_SYM_INT},
    {"pad_w", EXPORTED_CUSTOM_SYM_INT},
    {"dilation_h", EXPORTED_CUSTOM_SYM_INT},
    {"dilation_w", EXPORTED_CUSTOM_SYM_INT},
    {"groups", EXPORTED_CUSTOM_SYM_INT},
    {"offset_groups", EXPORTED_CUSTOM_SYM_INT},
    {"use_mask", EXPORTED_CUSTOM_BOOL},
};

static const ExportedCustomArgument torchvision_roi_align_arguments[] = {
    {"input", EXPORTED_CUSTOM_TENSOR},
    {"rois", EXPORTED_CUSTOM_TENSOR},
    {"spatial_scale", EXPORTED_CUSTOM_FLOAT},
    {"pooled_height", EXPORTED_CUSTOM_SYM_INT},
    {"pooled_width", EXPORTED_CUSTOM_SYM_INT},
    {"sampling_ratio", EXPORTED_CUSTOM_INT},
    {"aligned", EXPORTED_CUSTOM_BOOL},
};

static const char* custom_argument_type_name(ExportedCustomArgumentType type)
{
    if (type == EXPORTED_CUSTOM_TENSOR)
        return "Tensor";
    if (type == EXPORTED_CUSTOM_INT)
        return "int";
    if (type == EXPORTED_CUSTOM_SYM_INT)
        return "SymInt";
    if (type == EXPORTED_CUSTOM_FLOAT)
        return "float";
    return "bool";
}

static bool custom_argument_type_matches(ExportedCustomArgumentType expected, ExportedArgumentType actual)
{
    if (expected == EXPORTED_CUSTOM_TENSOR)
        return actual == EXPORTED_ARGUMENT_TENSOR;
    if (expected == EXPORTED_CUSTOM_INT)
        return actual == EXPORTED_ARGUMENT_INT;
    if (expected == EXPORTED_CUSTOM_SYM_INT)
        return actual == EXPORTED_ARGUMENT_INT;
    if (expected == EXPORTED_CUSTOM_FLOAT)
        return actual == EXPORTED_ARGUMENT_FLOAT;
    return actual == EXPORTED_ARGUMENT_BOOL;
}

static int canonicalize_with_custom_schema(const ExportedNode& node,
        const ExportedProgramHeader& header,
        const ExportedCustomArgument* schema_arguments,
        size_t schema_argument_count,
        std::vector<CanonicalExportedArgument>& result,
        std::string& error)
{
    std::vector<ExportedSchemaArgumentBinding> argument_bindings(schema_argument_count);
    for (size_t i = 0; i < schema_argument_count; i++)
    {
        argument_bindings[i].name = schema_arguments[i].name;
        argument_bindings[i].keyword_only = false;
    }

    std::vector<const ExportedNamedArgument*> bound_arguments;
    if (bind_exported_arguments(node, header, argument_bindings, bound_arguments, error) != 0)
        return -1;

    for (size_t i = 0; i < schema_argument_count; i++)
    {
        if (bound_arguments[i] && !custom_argument_type_matches(schema_arguments[i].type, bound_arguments[i]->arg.type))
            return operator_error(header, node, "argument " + std::string(schema_arguments[i].name) + " must be " + custom_argument_type_name(schema_arguments[i].type), error);
    }

    std::vector<CanonicalExportedArgument> canonical_arguments;
    canonical_arguments.reserve(schema_argument_count);
    for (size_t i = 0; i < schema_argument_count; i++)
    {
        if (!bound_arguments[i])
            return operator_error(header, node, "missing required argument " + std::string(schema_arguments[i].name), error);

        CanonicalExportedArgument argument;
        argument.name = schema_arguments[i].name;
        argument.value = bound_arguments[i]->arg;
        canonical_arguments.push_back(argument);
    }

    result.swap(canonical_arguments);
    return 0;
}

int canonicalize_exported_arguments(const ExportedNode& node,
                                    const ExportedProgramHeader& header,
                                    ExportedOperatorTarget& target,
                                    std::vector<CanonicalExportedArgument>& result,
                                    std::string& error)
{
    target = ExportedOperatorTarget();
    result.clear();
    error.clear();

    std::string target_error;
    if (parse_exported_operator_target(node.target, target, target_error) != 0)
        return operator_error(header, node, target_error, error);

    if (target.namespace_name == "torchvision" && target.operator_name == "torchvision::deform_conv2d" && target.overload_name.empty())
        return canonicalize_with_custom_schema(node, header, torchvision_deform_conv2d_arguments, sizeof(torchvision_deform_conv2d_arguments) / sizeof(torchvision_deform_conv2d_arguments[0]), result, error);

    if (target.namespace_name == "torchvision" && target.operator_name == "torchvision::roi_align" && target.overload_name.empty())
        return canonicalize_with_custom_schema(node, header, torchvision_roi_align_arguments, sizeof(torchvision_roi_align_arguments) / sizeof(torchvision_roi_align_arguments[0]), result, error);

    if (target.namespace_name != "aten")
        return operator_error(header, node, "unsupported exported operator " + node.target, error);

#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 9)
    return operator_error(header, node, "exported program operator schemas require libtorch 2.9 or newer", error);
#else
    try
    {
        const c10::OperatorHandle operator_handle = c10::Dispatcher::singleton().findSchemaOrThrow(target.operator_name.c_str(), target.overload_name.c_str());
        const c10::FunctionSchema& schema = operator_handle.schema();
        return canonicalize_with_schema(node, header, schema, operator_allows_numbers_as_tensors(target.operator_name), result, error);
    }
    catch (const c10::Error& e)
    {
        return operator_error(header, node, "cannot resolve dispatcher schema: " + std::string(e.what_without_backtrace()), error);
    }
    catch (const std::exception& e)
    {
        return operator_error(header, node, "dispatcher schema failure: " + std::string(e.what()), error);
    }
#endif
}

} // namespace pnnx
