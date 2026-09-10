// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exported_program.h"

#include "exported_program_graph.h"
#include "exported_program_operator.h"
#include "exported_program_schema.h"
#include "exported_program_tensor.h"
#include "pt2_archive.h"

#include <limits.h>
#include <stdio.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <exception>
#include <iomanip>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pnnx {

static int set_exported_tensor_shape(const ExportedTensorMeta& meta, const std::string& name, Operand* operand, std::string& error)
{
    std::vector<int> shape;
    shape.reserve(meta.sizes.size());
    for (size_t i = 0; i < meta.sizes.size(); i++)
    {
        if (meta.sizes[i] < 0 && (i >= meta.size_symbols.size() || meta.size_symbols[i].empty()))
        {
            std::ostringstream message;
            message << "tensor " << name << " has a negative or symbolic size at dimension " << i;
            error = message.str();
            return -1;
        }
        if (meta.sizes[i] > INT_MAX)
        {
            error = "tensor shape does not fit pnnx for " + name;
            return -1;
        }
        shape.push_back((int)meta.sizes[i]);
    }

    operand->shape.swap(shape);
    return 0;
}

static int set_tensor_metadata(const ExportedGraph& graph, const std::string& name, Operand* operand, std::string& error)
{
    const std::map<std::string, ExportedTensorMeta>::const_iterator meta_it = graph.tensor_values.find(name);
    if (meta_it == graph.tensor_values.end())
    {
        error = "missing tensor metadata for " + name;
        return -1;
    }

    const ExportedTensorMeta& meta = meta_it->second;
    if (meta.layout != 7)
    {
        std::ostringstream message;
        message << "unsupported tensor layout " << meta.layout << " for " << name;
        error = message.str();
        return -1;
    }
    const int pnnx_type = exported_tensor_dtype_to_pnnx_type(meta.dtype);
    if (pnnx_type == 0)
    {
        std::ostringstream message;
        message << "unsupported exported tensor dtype " << meta.dtype << " for " << name;
        error = message.str();
        return -1;
    }

    if (set_exported_tensor_shape(meta, name, operand, error) != 0)
        return -1;

    operand->type = pnnx_type;
    return 0;
}

static const char* exported_state_kind_name(ExportedInputKind kind)
{
    if (kind == EXPORTED_PARAMETER)
        return "parameter";
    if (kind == EXPORTED_BUFFER)
        return "buffer";
    return "tensor constant";
}

static const char* exported_output_kind_name(ExportedOutputKind kind)
{
    if (kind == EXPORTED_LOSS_OUTPUT)
        return "loss output";
    if (kind == EXPORTED_BUFFER_MUTATION)
        return "buffer mutation";
    if (kind == EXPORTED_PARAMETER_MUTATION)
        return "parameter mutation";
    if (kind == EXPORTED_GRADIENT_TO_PARAMETER)
        return "gradient to parameter";
    if (kind == EXPORTED_GRADIENT_TO_USER_INPUT)
        return "gradient to user input";
    if (kind == EXPORTED_USER_INPUT_MUTATION)
        return "user input mutation";
    if (kind == EXPORTED_OUTPUT_TOKEN)
        return "output token";

    return "unknown output kind";
}

static int validate_signature_kinds(const ExportedProgram& program, std::string& error)
{
    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const ExportedInputSpec& spec = program.input_specs[i];
        if (spec.kind == EXPORTED_USER_INPUT)
        {
            if (spec.arg.type != EXPORTED_ARGUMENT_TENSOR)
            {
                error = "user input " + spec.arg.name + " must be a tensor";
                return -1;
            }
            continue;
        }
        if (spec.kind == EXPORTED_PARAMETER || spec.kind == EXPORTED_TENSOR_CONSTANT)
        {
            if (spec.arg.type != EXPORTED_ARGUMENT_TENSOR)
            {
                error = "state input " + spec.target + " must be a tensor";
                return -1;
            }
            continue;
        }
        if (spec.kind == EXPORTED_BUFFER)
        {
            if (spec.arg.type != EXPORTED_ARGUMENT_TENSOR)
            {
                error = "buffer " + spec.target + " must be a tensor";
                return -1;
            }
            continue;
        }
        if (spec.kind == EXPORTED_CUSTOM_OBJ)
        {
            error = "custom object input " + spec.arg.name + " is unsupported";
            return -1;
        }
        if (spec.kind == EXPORTED_TOKEN)
        {
            error = "token input " + spec.arg.name + " is unsupported";
            return -1;
        }

        error = "constant input " + spec.arg.name + " is unsupported";
        return -1;
    }

    for (size_t i = 0; i < program.output_specs.size(); i++)
    {
        const ExportedOutputSpec& spec = program.output_specs[i];
        if (spec.kind != EXPORTED_USER_OUTPUT)
        {
            error = std::string("unsupported exported program ") + exported_output_kind_name(spec.kind);
            return -1;
        }
        if (spec.arg.type != EXPORTED_ARGUMENT_TENSOR)
        {
            error = "user output " + spec.arg.name + " must be a tensor";
            return -1;
        }
    }

    return 0;
}

static int validate_signature_arguments(const ExportedProgram& source_program, const ExportedGraph& normalized_graph, std::string& error)
{
    for (size_t i = 0; i < source_program.input_specs.size(); i++)
    {
        const ExportedArgument& signature = source_program.input_specs[i].arg;
        const ExportedArgument& graph = normalized_graph.inputs[i];
        if (signature.type != graph.type || signature.name != graph.name)
        {
            std::ostringstream message;
            message << "input spec " << i << " tensor " << signature.name << " does not match graph input " << graph.name;
            error = message.str();
            return -1;
        }
    }

    for (size_t i = 0; i < source_program.output_specs.size(); i++)
    {
        const ExportedArgument& signature = source_program.output_specs[i].arg;
        const ExportedArgument& graph = normalized_graph.outputs[i];
        if (signature.type != graph.type || signature.name != graph.name)
        {
            std::ostringstream message;
            message << "output spec " << i << " tensor " << signature.name << " does not match graph output " << graph.name;
            error = message.str();
            return -1;
        }
    }

    return 0;
}

static int validate_shape_symbols(const ExportedProgram& program, const ExportedGraph& graph, std::map<std::string, int>& symbols, std::string& error)
{
    for (const ExportedInputSpec& spec : program.input_specs)
    {
        const ExportedTensorMeta& meta = graph.tensor_values.at(spec.arg.name);
        for (const std::string& symbol : meta.size_symbols)
        {
            if (symbol.empty())
                continue;
            if (spec.kind != EXPORTED_USER_INPUT)
            {
                error = "state tensor dimensions must be static: " + spec.arg.name;
                return -1;
            }
            if (program.range_constraints.find(symbol) == program.range_constraints.end())
            {
                error = "missing range constraint for input dimension " + symbol;
                return -1;
            }
            if (symbols.find(symbol) == symbols.end())
                symbols[symbol] = (int)symbols.size();
        }
    }
    for (const auto& range : program.range_constraints)
    {
        if (symbols.find(range.first) == symbols.end())
        {
            error = "range constraint is not bound to an input dimension: " + range.first;
            return -1;
        }
    }
    for (const auto& tensor : graph.tensor_values)
    {
        for (const auto* dimensions : {
                    &tensor.second.size_symbols, &tensor.second.stride_symbols
                })
        {
            for (const std::string& symbol : *dimensions)
            {
                if (!symbol.empty() && symbols.find(symbol) == symbols.end())
                {
                    error = "unbound symbolic dimension " + symbol + " for tensor " + tensor.first;
                    return -1;
                }
            }
        }
    }
    for (const auto& value : graph.sym_int_values)
    {
        if (symbols.find(value.second) == symbols.end())
        {
            error = "unbound symbolic value " + value.first;
            return -1;
        }
    }
    return 0;
}

static std::string unique_name(const std::string& requested, std::set<std::string>& names)
{
    std::string sanitized;
    sanitized.reserve(requested.size() + 5);
    for (size_t i = 0; i < requested.size(); i++)
    {
        const char ch = requested[i];
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_')
            sanitized += ch;
        else
            sanitized += '_';
    }

    if (sanitized.empty() || (sanitized[0] >= '0' && sanitized[0] <= '9'))
        sanitized.insert(0, "pnnx_");

    if (names.insert(sanitized).second)
        return sanitized;

    for (size_t suffix = 1;; suffix++)
    {
        std::ostringstream candidate;
        candidate << sanitized << '_' << suffix;
        if (names.insert(candidate.str()).second)
            return candidate.str();
    }
}

static int construct_output_tree(const ExportedTreeSpec& tree_spec,
                                 const std::vector<Operand*>& flat_outputs,
                                 size_t& flat_index,
                                 Graph& graph,
                                 std::set<std::string>& operand_names,
                                 std::set<std::string>& operator_names,
                                 int& unknown_index,
                                 Operand*& output,
                                 std::string& error)
{
    if (tree_spec.type == EXPORTED_TREE_SPEC_LEAF)
    {
        if (flat_index >= flat_outputs.size())
        {
            error = "output treespec consumes too many graph outputs";
            return -1;
        }

        output = flat_outputs[flat_index++];
        return 0;
    }

    std::vector<Operand*> children;
    children.reserve(tree_spec.children.size());
    for (size_t i = 0; i < tree_spec.children.size(); i++)
    {
        Operand* child = 0;
        if (construct_output_tree(tree_spec.children[i], flat_outputs, flat_index, graph, operand_names, operator_names, unknown_index, child, error) != 0)
            return -1;
        children.push_back(child);
    }

    std::ostringstream generated_name;
    generated_name << "pnnx_" << unknown_index++;
    const char* operator_type = tree_spec.type == EXPORTED_TREE_SPEC_TUPLE ? "prim::TupleConstruct" : "prim::ListConstruct";
    Operator* construct = graph.new_operator(operator_type, unique_name(generated_name.str(), operator_names));
    output = graph.new_operand(unique_name(generated_name.str(), operand_names));
    output->producer = construct;
    construct->outputs.push_back(output);
    for (size_t i = 0; i < children.size(); i++)
    {
        children[i]->consumers.push_back(construct);
        construct->inputs.push_back(children[i]);
    }

    return 0;
}

static int exported_int_to_pnnx(int64_t value, int& converted)
{
    if (value < (int64_t)INT_MIN || value > (int64_t)INT_MAX)
        return -1;

    converted = (int)value;

    return 0;
}

static int exported_slice_sentinel_to_pnnx(const ExportedOperatorTarget& target, const std::string& argument_name, int64_t value, int& converted)
{
    const bool is_slice = target.operator_name == "aten::slice" || target.operator_name == "aten::slice_scatter";
    const bool is_slice_endpoint = argument_name == "start" || argument_name == "end";
    if (is_slice && is_slice_endpoint)
    {
        if (value == std::numeric_limits<int64_t>::max())
        {
            converted = INT_MAX;
            return 0;
        }
        if (value == std::numeric_limits<int64_t>::max() - 1)
        {
            converted = INT_MAX - 1;
            return 0;
        }
        if (value == std::numeric_limits<int64_t>::min())
        {
            converted = INT_MIN;
            return 0;
        }
        if (value == std::numeric_limits<int64_t>::min() + 1)
        {
            converted = INT_MIN + 1;
            return 0;
        }
    }

    return exported_int_to_pnnx(value, converted);
}

static int exported_float_to_pnnx(double value, float& converted)
{
    converted = (float)value;
    if (!std::isfinite(value) || !std::isfinite(converted) || (value != 0.0 && converted == 0.0f))
        return -1;

    return 0;
}

static bool exported_tensor_uses_high_precision_float(const std::string& name, const ExportedGraph& graph)
{
    const std::map<std::string, ExportedTensorMeta>::const_iterator meta_it = graph.tensor_values.find(name);
    if (meta_it == graph.tensor_values.end())
        return false;

    return meta_it->second.dtype == 8 || meta_it->second.dtype == 11; // Double or ComplexDouble
}

static bool exported_argument_uses_high_precision_float(const ExportedArgument& argument, const ExportedGraph& graph)
{
    if (argument.type == EXPORTED_ARGUMENT_TENSOR)
        return exported_tensor_uses_high_precision_float(argument.name, graph);

    if (argument.type == EXPORTED_ARGUMENT_TENSOR_LIST)
    {
        for (size_t i = 0; i < argument.tensor_names.size(); i++)
        {
            if (exported_tensor_uses_high_precision_float(argument.tensor_names[i], graph))
                return true;
        }
    }

    return false;
}

static void report_exported_float_narrowing(const ExportedNode& node, const std::string& argument_name, double value, std::set<std::string>& warnings)
{
    float converted = 0.f;
    if (exported_float_to_pnnx(value, converted) != 0 || (double)converted == value)
        return;

    std::ostringstream message;
    message << "warning: lossy scalar narrowing for " << node.target
            << " argument " << argument_name << ": "
            << std::setprecision(std::numeric_limits<double>::max_digits10) << value
            << " -> " << std::setprecision(std::numeric_limits<float>::max_digits10) << converted
            << " in f64/c128 tensor context";
    if (warnings.insert(message.str()).second)
        fprintf(stderr, "%s\n", message.str().c_str());
}

static void report_exported_scalar_narrowing(const ExportedNode& node,
        const std::vector<CanonicalExportedArgument>& arguments,
        const ExportedGraph& graph,
        std::set<std::string>& warnings)
{
    bool has_high_precision_tensor = false;
    for (size_t i = 0; i < arguments.size() && !has_high_precision_tensor; i++)
        has_high_precision_tensor = exported_argument_uses_high_precision_float(arguments[i].value, graph);
    for (size_t i = 0; i < node.outputs.size() && !has_high_precision_tensor; i++)
        has_high_precision_tensor = exported_argument_uses_high_precision_float(node.outputs[i], graph);
    if (!has_high_precision_tensor)
        return;

    for (size_t i = 0; i < arguments.size(); i++)
    {
        const CanonicalExportedArgument& argument = arguments[i];
        if (argument.value.type == EXPORTED_ARGUMENT_FLOAT)
        {
            report_exported_float_narrowing(node, argument.name, argument.value.float_value, warnings);
        }
        else if (argument.value.type == EXPORTED_ARGUMENT_FLOAT_LIST)
        {
            for (size_t j = 0; j < argument.value.float_values.size(); j++)
            {
                std::ostringstream item_name;
                item_name << argument.name << '[' << j << ']';
                report_exported_float_narrowing(node, item_name.str(), argument.value.float_values[j], warnings);
            }
        }
        else if (argument.value.type == EXPORTED_ARGUMENT_COMPLEX)
        {
            report_exported_float_narrowing(node, argument.name + ".real", argument.value.complex_real_value, warnings);
            report_exported_float_narrowing(node, argument.name + ".imag", argument.value.complex_imag_value, warnings);
        }
    }
}

static int exported_memory_format_to_pnnx(int64_t value, int& converted)
{
    if (value == 1)
        converted = 0; // contiguous
    else if (value == 2)
        converted = 2; // channels last
    else if (value == 3)
        converted = 3; // channels last 3d
    else if (value == 4)
        converted = 1; // preserve
    else
        return -1;

    return 0;
}

static int exported_scalar_type_to_pnnx(int64_t value, int& converted)
{
    if (value >= 1 && value <= 12)
        converted = (int)value - 1;
    else if (value == 13)
        converted = 15; // bfloat16
    else
        return -1;

    return 0;
}

static int exported_layout_to_pnnx(int64_t value, int& converted)
{
    if (value != 7) // strided
        return -1;

    converted = 0;
    return 0;
}

static bool exported_string_is_safe_pnnx_parameter(const std::string& value)
{
    if (value.empty())
        return false;

    // These spellings are decoded as non-string Parameters.
    if (value == "None" || value == "()" || value == "[]" || value == "[]f" || value == "True" || value == "False")
        return false;

    // The Python writers intentionally interpret these string Parameters as
    // expressions or floating-point sentinels rather than string literals.
    if (value.compare(0, 6, "torch.") == 0 || value == "inf" || value == "-inf")
        return false;

    const unsigned char first = (unsigned char)value[0];
    if (first == '(' || first == '[' || std::isdigit(first))
        return false;
    if (first == '-' && (value.size() == 1 || std::isdigit((unsigned char)value[1])))
        return false;

    for (size_t i = 0; i < value.size(); i++)
    {
        const unsigned char c = (unsigned char)value[i];
        if (std::isspace(c) || std::iscntrl(c) || c == '\'' || c == '"' || c == '\\')
            return false;
    }

    return true;
}

static int exported_argument_to_parameter(const ExportedArgument& argument, const ExportedOperatorTarget& target, const std::string& argument_name, Parameter& parameter, std::string& detail)
{
    detail.clear();

    if (argument.type == EXPORTED_ARGUMENT_NONE)
    {
        parameter.type = 0;
    }
    else if (argument.type == EXPORTED_ARGUMENT_INT)
    {
        parameter.type = 2;
        if (exported_slice_sentinel_to_pnnx(target, argument_name, argument.int_value, parameter.i) != 0)
        {
            std::ostringstream message;
            message << "integer value " << argument.int_value << " does not fit pnnx integer parameter";
            detail = message.str();
            return -1;
        }
    }
    else if (argument.type == EXPORTED_ARGUMENT_INT_LIST)
    {
        parameter.type = 5;
        parameter.ai.reserve(argument.int_values.size());
        for (size_t i = 0; i < argument.int_values.size(); i++)
        {
            int converted = 0;
            if (exported_int_to_pnnx(argument.int_values[i], converted) != 0)
            {
                std::ostringstream message;
                message << "integer list item " << i << " value " << argument.int_values[i] << " does not fit pnnx integer parameter";
                detail = message.str();
                return -1;
            }
            parameter.ai.push_back(converted);
        }
    }
    else if (argument.type == EXPORTED_ARGUMENT_FLOAT)
    {
        if (std::isinf(argument.float_value))
        {
            parameter.type = 4;
            parameter.s = std::signbit(argument.float_value) ? "-inf" : "inf";
        }
        else
        {
            parameter.type = 3;
            if (exported_float_to_pnnx(argument.float_value, parameter.f) != 0)
            {
                std::ostringstream message;
                message << "floating-point value " << argument.float_value << " does not fit pnnx float parameter";
                detail = message.str();
                return -1;
            }
        }
    }
    else if (argument.type == EXPORTED_ARGUMENT_FLOAT_LIST)
    {
        parameter.type = 6;
        parameter.af.reserve(argument.float_values.size());
        for (size_t i = 0; i < argument.float_values.size(); i++)
        {
            float converted = 0.f;
            if (exported_float_to_pnnx(argument.float_values[i], converted) != 0)
            {
                std::ostringstream message;
                message << "floating-point list item " << i << " value " << argument.float_values[i] << " does not fit pnnx float parameter";
                detail = message.str();
                return -1;
            }
            parameter.af.push_back(converted);
        }
    }
    else if (argument.type == EXPORTED_ARGUMENT_COMPLEX)
    {
        parameter.type = 10;
        float real = 0.f;
        float imag = 0.f;
        if (exported_float_to_pnnx(argument.complex_real_value, real) != 0 || exported_float_to_pnnx(argument.complex_imag_value, imag) != 0)
        {
            detail = "complex value does not fit pnnx float parameter";
            return -1;
        }
        parameter.c = std::complex<float>(real, imag);
    }
    else if (argument.type == EXPORTED_ARGUMENT_BOOL)
    {
        parameter.type = 1;
        parameter.b = argument.bool_value;
    }
    else if (argument.type == EXPORTED_ARGUMENT_STRING)
    {
        if (!exported_string_is_safe_pnnx_parameter(argument.string_value))
        {
            detail = "string is not safely representable by pnnx Parameter";
            return -1;
        }

        parameter.type = 4;
        parameter.s = argument.string_value;
    }
    else if (argument.type == EXPORTED_ARGUMENT_STRING_LIST)
    {
        detail = "string list arguments are unsupported by pnnx lowering";
        return -1;
    }
    else if (argument.type == EXPORTED_ARGUMENT_BOOL_LIST)
    {
        detail = "bool list is not representable by pnnx Parameter";
        return -1;
    }
    else if (argument.type == EXPORTED_ARGUMENT_MEMORY_FORMAT)
    {
        parameter.type = 2;
        if (exported_memory_format_to_pnnx(argument.enum_value, parameter.i) != 0)
        {
            detail = "memory format is not representable by pnnx Parameter";
            return -1;
        }
    }
    else if (argument.type == EXPORTED_ARGUMENT_SCALAR_TYPE)
    {
        parameter.type = 2;
        if (exported_scalar_type_to_pnnx(argument.enum_value, parameter.i) != 0)
        {
            detail = "scalar type is not representable by pnnx Parameter";
            return -1;
        }
    }
    else if (argument.type == EXPORTED_ARGUMENT_DEVICE)
    {
        if (!exported_string_is_safe_pnnx_parameter(argument.device_value.type))
        {
            detail = "device is not representable by pnnx Parameter";
            return -1;
        }
        if (argument.device_value.has_index && (argument.device_value.index < 0 || argument.device_value.index > 127))
        {
            detail = "device is not representable by pnnx Parameter";
            return -1;
        }

        parameter.type = 4;
        parameter.s = argument.device_value.type;
        if (argument.device_value.has_index)
        {
            std::ostringstream device;
            device << argument.device_value.type << ':' << argument.device_value.index;
            parameter.s = device.str();
        }
    }
    else if (argument.type == EXPORTED_ARGUMENT_LAYOUT)
    {
        parameter.type = 2;
        if (exported_layout_to_pnnx(argument.enum_value, parameter.i) != 0)
        {
            detail = "layout is not representable by pnnx Parameter";
            return -1;
        }
    }
    else
    {
        detail = "argument type is not representable by pnnx Parameter";
        return -1;
    }

    return 0;
}

static const ExportedArgument* find_canonical_argument(const std::vector<CanonicalExportedArgument>& arguments, const std::string& name)
{
    for (size_t i = 0; i < arguments.size(); i++)
    {
        if (arguments[i].name == name)
            return &arguments[i].value;
    }

    return 0;
}

static int validate_tensor_metadata_assertion(const ExportedNode& node,
        const std::vector<CanonicalExportedArgument>& arguments,
        const ExportedGraph& graph,
        const std::map<std::string, Operand*>& values,
        std::string& error)
{
    if (!node.outputs.empty())
    {
        error = "tensor metadata assertion must not produce outputs";
        return -1;
    }

    const ExportedArgument* tensor = find_canonical_argument(arguments, "a");
    const ExportedArgument* size = find_canonical_argument(arguments, "size");
    const ExportedArgument* stride = find_canonical_argument(arguments, "stride");
    const ExportedArgument* dtype = find_canonical_argument(arguments, "dtype");
    const ExportedArgument* device = find_canonical_argument(arguments, "device");
    const ExportedArgument* layout = find_canonical_argument(arguments, "layout");
    if (!tensor || !size || !stride || !dtype || !device || !layout)
    {
        error = "tensor metadata assertion dispatcher schema is incomplete";
        return -1;
    }
    if (tensor->type != EXPORTED_ARGUMENT_TENSOR)
    {
        error = "tensor metadata assertion input must be a tensor";
        return -1;
    }
    if (values.find(tensor->name) == values.end())
    {
        error = "tensor metadata assertion references unavailable tensor " + tensor->name;
        return -1;
    }

    const std::map<std::string, ExportedTensorMeta>::const_iterator meta_it = graph.tensor_values.find(tensor->name);
    if (meta_it == graph.tensor_values.end())
    {
        error = "tensor metadata assertion input " + tensor->name + " is missing metadata";
        return -1;
    }
    const ExportedTensorMeta& meta = meta_it->second;

    if (size->type != EXPORTED_ARGUMENT_NONE && (size->type != EXPORTED_ARGUMENT_INT_LIST || size->int_values != meta.sizes))
    {
        error = "tensor metadata assertion size does not match " + tensor->name;
        return -1;
    }
    if (stride->type != EXPORTED_ARGUMENT_NONE && (stride->type != EXPORTED_ARGUMENT_INT_LIST || stride->int_values != meta.strides))
    {
        error = "tensor metadata assertion stride does not match " + tensor->name;
        return -1;
    }
    if (dtype->type != EXPORTED_ARGUMENT_NONE && (dtype->type != EXPORTED_ARGUMENT_SCALAR_TYPE || dtype->enum_value != meta.dtype))
    {
        error = "tensor metadata assertion dtype does not match " + tensor->name;
        return -1;
    }
    if (device->type != EXPORTED_ARGUMENT_NONE
            && (device->type != EXPORTED_ARGUMENT_DEVICE
                || device->device_value.type != meta.device_type
                || device->device_value.has_index != meta.has_device_index
                || (device->device_value.has_index && device->device_value.index != meta.device_index)))
    {
        error = "tensor metadata assertion device does not match " + tensor->name;
        return -1;
    }
    if (layout->type != EXPORTED_ARGUMENT_NONE && (layout->type != EXPORTED_ARGUMENT_LAYOUT || layout->enum_value != meta.layout))
    {
        error = "tensor metadata assertion layout does not match " + tensor->name;
        return -1;
    }

    return 0;
}

struct ExportedTensorSources
{
    std::set<std::string> definite;
    std::set<std::string> possible;
};

static std::vector<std::string> exported_tensor_names(const ExportedArgument& argument)
{
    if (argument.type == EXPORTED_ARGUMENT_TENSOR)
        return std::vector<std::string>(1, argument.name);
    if (argument.type == EXPORTED_ARGUMENT_TENSOR_LIST)
        return argument.tensor_names;
    return std::vector<std::string>();
}

static bool exported_to_copies(const ExportedOperatorTarget& target,
                               const std::vector<CanonicalExportedArgument>& arguments,
                               const ExportedGraph& graph)
{
    if (target.operator_name != "aten::to")
        return false;
    const ExportedArgument* copy = find_canonical_argument(arguments, "copy");
    if (copy && copy->type == EXPORTED_ARGUMENT_BOOL && copy->bool_value)
        return true;

    const ExportedArgument* self = find_canonical_argument(arguments, "self");
    if (!self || self->type != EXPORTED_ARGUMENT_TENSOR)
        return false;
    const std::map<std::string, ExportedTensorMeta>::const_iterator meta = graph.tensor_values.find(self->name);
    if (meta == graph.tensor_values.end())
        return false;
    const ExportedArgument* dtype = find_canonical_argument(arguments, "dtype");
    if (dtype && dtype->type == EXPORTED_ARGUMENT_SCALAR_TYPE)
        return dtype->enum_value != meta->second.dtype;
    return false;
}

static bool exported_definite_view(const std::string& name)
{
    // Conditional copies (reshape/flatten/contiguous/to) deliberately stay May:
    // serialized example strides are not a runtime input-layout contract.
    return name == "aten::alias" || name == "aten::detach" || name == "aten::view"
           || name == "aten::_unsafe_view" || name == "aten::_reshape_alias"
           || name == "aten::slice" || name == "aten::select" || name == "aten::transpose"
           || name == "aten::t" || name == "aten::permute" || name == "aten::squeeze"
           || name == "aten::unsqueeze" || name == "aten::expand" || name == "aten::as_strided"
           || name == "aten::diagonal" || name == "aten::unfold";
}

static int propagate_exported_effects(const ExportedNode& node,
                                      const ExportedOperatorTarget& target,
                                      const std::vector<CanonicalExportedArgument>& arguments,
                                      const ExportedOperatorEffects& effects,
                                      const ExportedProgram& program,
                                      const ExportedGraph& graph,
                                      std::map<std::string, ExportedTensorSources>& sources,
                                      std::string& error)
{
    // Resolve every reference before checking effects, including individual list
    // members. Missing values must never be mistaken for a fresh local tensor.
    for (size_t j = 0; j < arguments.size(); j++)
    {
        const std::vector<std::string> names = exported_tensor_names(arguments[j].value);
        for (size_t k = 0; k < names.size(); k++)
        {
            if (sources.find(names[k]) == sources.end())
            {
                error = "unknown tensor value " + names[k] + " for argument " + arguments[j].name + " of " + node.target;
                return -1;
            }
        }
    }
    for (size_t i = 0; i < effects.mutable_input_indices.size(); i++)
    {
        const CanonicalExportedArgument& argument = arguments[effects.mutable_input_indices[i]];
        const std::vector<std::string> names = exported_tensor_names(argument.value);
        for (size_t k = 0; k < names.size(); k++)
        {
            const ExportedTensorSources& source = sources.find(names[k])->second;
            if (!source.definite.empty())
            {
                const std::string& origin = *source.definite.begin();
                for (size_t j = 0; j < program.input_specs.size(); j++)
                {
                    const ExportedInputSpec& spec = program.input_specs[j];
                    if (spec.arg.name != origin)
                        continue;
                    const std::string kind = spec.kind == EXPORTED_USER_INPUT ? "user-input" : exported_state_kind_name(spec.kind);
                    error = "unsupported " + kind + " mutation through argument " + argument.name + " of " + node.target + ": " + origin;
                    return -1;
                }
            }
            if (!source.possible.empty())
            {
                error = "cannot prove mutation is local through " + names[k] + " of " + *source.possible.begin()
                        + " through argument " + argument.name + " of " + node.target;
                return -1;
            }
        }
    }

    const bool copies = exported_to_copies(target, arguments, graph);
    for (size_t i = 0; i < node.outputs.size(); i++)
    {
        ExportedTensorSources output;
        const std::vector<size_t>& aliases = effects.output_alias_input_indices[i];
        for (size_t j = 0; !copies && j < aliases.size(); j++)
        {
            const ExportedArgument& input = arguments[aliases[j]].value;
            const std::vector<std::string> names = exported_tensor_names(input);
            // A schema container/wildcard edge gives no member correspondence.
            // Preserve each input member, merging only into possible origins.
            const bool definite = input.type == EXPORTED_ARGUMENT_TENSOR && node.outputs[i].type == EXPORTED_ARGUMENT_TENSOR
                                  && exported_definite_view(target.operator_name);
            for (size_t k = 0; k < names.size(); k++)
            {
                const ExportedTensorSources& source = sources.find(names[k])->second;
                std::set<std::string>& destination = definite ? output.definite : output.possible;
                destination.insert(source.definite.begin(), source.definite.end());
                output.possible.insert(source.possible.begin(), source.possible.end());
            }
        }
        const std::vector<std::string> names = exported_tensor_names(node.outputs[i]);
        for (size_t j = 0; j < names.size(); j++)
            sources.insert(std::make_pair(names[j], output));
    }
    return 0;
}

static int lower_exported_program(const ExportedProgram& source_program,
                                  std::map<std::string, MaterializedExportedTensor>& state,
                                  Graph& graph,
                                  std::string& error)
{
    // source_program has passed parse_exported_program, including output treespec validation.
    error.clear();
    if (!graph.ops.empty() || !graph.operands.empty())
    {
        error = "destination graph must be empty";
        return -1;
    }

    ExportedGraph normalized_graph;
    if (normalize_exported_program_graph(source_program.graph, normalized_graph, error) != 0)
        return -1;

    if (source_program.input_specs.size() != normalized_graph.inputs.size())
    {
        error = "input spec count does not match graph inputs";
        return -1;
    }
    if (source_program.output_specs.size() != normalized_graph.outputs.size())
    {
        error = "output spec count does not match graph outputs";
        return -1;
    }
    if (validate_signature_kinds(source_program, error) != 0)
        return -1;
    if (validate_signature_arguments(source_program, normalized_graph, error) != 0)
        return -1;
    if (validate_exported_program_opset(source_program.header, error) != 0)
        return -1;
    std::map<std::string, int> shape_symbols;
    if (validate_shape_symbols(source_program, normalized_graph, shape_symbols, error) != 0)
        return -1;
    Graph candidate;
    std::map<std::string, Operand*> values;
    std::map<std::string, ExportedTensorSources> sources;
    std::set<std::string> scalar_narrowing_warnings;
    std::set<std::string> operand_names;
    std::set<std::string> operator_names;
    std::map<std::string, size_t> state_uses;
    int input_index = 0;
    int unknown_index = 0;

    for (size_t i = 0; i < source_program.input_specs.size(); i++)
    {
        const ExportedInputSpec& spec = source_program.input_specs[i];
        if (spec.kind == EXPORTED_PARAMETER || spec.kind == EXPORTED_BUFFER || spec.kind == EXPORTED_TENSOR_CONSTANT)
            state_uses[spec.target]++;
    }

    for (size_t i = 0; i < source_program.input_specs.size(); i++)
    {
        const ExportedInputSpec& spec = source_program.input_specs[i];
        const std::string& name = spec.arg.name;
        sources[name].definite.insert(name);
        if (values.find(name) != values.end())
        {
            error = "tensor value " + name + " is defined more than once";
            return -1;
        }

        if (spec.kind == EXPORTED_PARAMETER || spec.kind == EXPORTED_BUFFER || spec.kind == EXPORTED_TENSOR_CONSTANT)
        {
            std::map<std::string, MaterializedExportedTensor>::iterator state_it = state.find(spec.target);
            if (state_it == state.end())
            {
                const char* state_kind = exported_state_kind_name(spec.kind);
                error = std::string(state_kind) + " " + spec.target + " is missing materialized state";
                return -1;
            }

            std::ostringstream state_name;
            state_name << "pnnx_state_" << i;
            Operator* op = candidate.new_operator("pnnx.Attribute", unique_name(state_name.str(), operator_names));
            Operand* operand = candidate.new_operand(unique_name(name, operand_names));
            operand->producer = op;
            op->outputs.push_back(operand);
            if (set_tensor_metadata(normalized_graph, name, operand, error) != 0)
                return -1;
            if (state_it->second.pnnx_type != operand->type)
            {
                error = std::string(exported_state_kind_name(spec.kind)) + " " + spec.target + " type does not match tensor metadata";
                return -1;
            }
            if (state_it->second.shape != operand->shape)
            {
                error = std::string(exported_state_kind_name(spec.kind)) + " " + spec.target + " shape does not match tensor metadata";
                return -1;
            }

            Attribute& attribute = op->attrs["data"];
            attribute.type = state_it->second.pnnx_type;
            attribute.shape = state_it->second.shape;
            std::map<std::string, size_t>::iterator uses_it = state_uses.find(spec.target);
            if (uses_it == state_uses.end() || uses_it->second == 0)
            {
                error = std::string(exported_state_kind_name(spec.kind)) + " " + spec.target + " has inconsistent use count";
                return -1;
            }
            uses_it->second--;
            if (uses_it->second == 0)
                attribute.data.swap(state_it->second.data);
            else
                attribute.data = state_it->second.data;
            values[name] = operand;
            continue;
        }

        std::ostringstream input_name;
        input_name << "pnnx_input_" << input_index++;
        Operator* op = candidate.new_operator("pnnx.Input", unique_name(input_name.str(), operator_names));
        Operand* operand = candidate.new_operand(unique_name(name, operand_names));
        operand->producer = op;
        op->outputs.push_back(operand);
        if (set_tensor_metadata(normalized_graph, name, operand, error) != 0)
            return -1;
        const ExportedTensorMeta& meta = normalized_graph.tensor_values.at(name);
        for (size_t j = 0; j < meta.size_symbols.size(); j++)
        {
            const std::string& symbol = meta.size_symbols[j];
            if (symbol.empty())
                continue;
            const std::pair<int, int>& bounds = source_program.range_constraints.at(symbol);
            op->params["__pt2_dim_" + std::to_string(j)] = std::vector<int> {shape_symbols.at(symbol), bounds.first, bounds.second};
        }
        values[name] = operand;
    }

    for (size_t i = 0; i < normalized_graph.nodes.size(); i++)
    {
        const ExportedNode& node = normalized_graph.nodes[i];
        std::vector<CanonicalExportedArgument> arguments;
        ExportedOperatorTarget target;
        ExportedOperatorEffects effects;
        if (canonicalize_exported_arguments(node, source_program.header, target, arguments, effects, error) != 0)
            return -1;
        report_exported_scalar_narrowing(node, arguments, normalized_graph, scalar_narrowing_warnings);
        if (propagate_exported_effects(node, target, arguments, effects, source_program, normalized_graph, sources, error) != 0)
            return -1;

        if (normalize_exported_operator_arguments(node, target, normalized_graph, arguments, error) != 0)
            return -1;

        if (target.operator_name == "aten::_assert_tensor_metadata" && target.overload_name.empty())
        {
            if (validate_tensor_metadata_assertion(node, arguments, normalized_graph, values, error) != 0)
                return -1;
            continue;
        }

        std::ostringstream generated_name;
        if (!node.has_name)
            generated_name << "pnnx_" << unknown_index++;
        const std::string requested_name = node.has_name ? node.name : generated_name.str();
        Operator* op = candidate.new_operator(target.operator_name, unique_name(requested_name, operator_names));

        if (target.operator_name == "aten::sym_size" && target.overload_name == "int")
        {
            const ExportedArgument* self = find_canonical_argument(arguments, "self");
            const ExportedArgument* dim = find_canonical_argument(arguments, "dim");
            if (!self || self->type != EXPORTED_ARGUMENT_TENSOR || !dim || dim->type != EXPORTED_ARGUMENT_INT
                    || node.outputs.size() != 1 || node.outputs[0].type != EXPORTED_ARGUMENT_SYM_INT)
            {
                error = "unsupported symbolic size query for " + node.target;
                return -1;
            }
            const ExportedTensorMeta& meta = normalized_graph.tensor_values.at(self->name);
            const int64_t axis = dim->int_value < 0 ? dim->int_value + (int64_t)meta.sizes.size() : dim->int_value;
            const auto symbol = normalized_graph.sym_int_values.find(node.outputs[0].name);
            if (axis < 0 || axis >= (int64_t)meta.size_symbols.size() || symbol == normalized_graph.sym_int_values.end()
                    || meta.size_symbols[axis] != symbol->second)
            {
                error = "symbolic size metadata does not match queried dimension";
                return -1;
            }
            op->type = "aten::size";
        }

        for (size_t j = 0; j < arguments.size(); j++)
        {
            Operand* operand = 0;
            if (arguments[j].value.type == EXPORTED_ARGUMENT_TENSOR || arguments[j].value.type == EXPORTED_ARGUMENT_SYM_INT)
            {
                if (arguments[j].value.type == EXPORTED_ARGUMENT_SYM_INT && normalized_graph.sym_int_values.find(arguments[j].value.name) == normalized_graph.sym_int_values.end())
                {
                    error = "unknown symbolic value " + arguments[j].value.name;
                    return -1;
                }
                const std::map<std::string, Operand*>::const_iterator value_it = values.find(arguments[j].value.name);
                if (value_it == values.end())
                {
                    error = "unknown tensor value " + arguments[j].value.name + " for " + node.target;
                    return -1;
                }

                operand = value_it->second;
            }
            else if (arguments[j].value.type == EXPORTED_ARGUMENT_TENSOR_LIST || arguments[j].value.type == EXPORTED_ARGUMENT_SYM_INT_LIST)
            {
                std::ostringstream list_name;
                list_name << "pnnx_" << unknown_index++;
                Operator* list = candidate.new_operator_before("prim::ListConstruct", unique_name(list_name.str(), operator_names), op);
                operand = candidate.new_operand(unique_name(list_name.str(), operand_names));
                operand->producer = list;
                list->outputs.push_back(operand);

                const ExportedArgument& argument = arguments[j].value;
                const bool symbolic = argument.type == EXPORTED_ARGUMENT_SYM_INT_LIST;
                const std::vector<std::string>& names = symbolic ? argument.int_names : argument.tensor_names;
                for (size_t k = 0; k < names.size(); k++)
                {
                    const std::string& tensor_name = names[k];
                    if (symbolic && tensor_name.empty())
                    {
                        int value = 0;
                        if (exported_int_to_pnnx(argument.int_values[k], value) != 0)
                        {
                            error = "symbolic shape list constant does not fit pnnx int parameter";
                            return -1;
                        }
                        const std::string name = "pnnx_" + std::to_string(unknown_index++);
                        Operator* constant = candidate.new_operator_before("prim::Constant", unique_name(name, operator_names), list);
                        Operand* item = candidate.new_operand(unique_name(name, operand_names));
                        constant->params["value"] = value;
                        constant->outputs.push_back(item);
                        item->producer = constant;
                        item->consumers.push_back(list);
                        list->inputs.push_back(item);
                        continue;
                    }
                    if (symbolic && normalized_graph.sym_int_values.find(tensor_name) == normalized_graph.sym_int_values.end())
                    {
                        error = "unknown symbolic value " + tensor_name;
                        return -1;
                    }
                    const std::map<std::string, Operand*>::const_iterator value_it = values.find(tensor_name);
                    if (value_it == values.end())
                    {
                        error = "unknown tensor value " + tensor_name + " for tensor-list argument " + arguments[j].name + " of " + node.target;
                        return -1;
                    }

                    value_it->second->consumers.push_back(list);
                    list->inputs.push_back(value_it->second);
                }
            }
            else
            {
                std::ostringstream constant_name;
                constant_name << "pnnx_" << unknown_index++;
                const std::string constant_operator_name = unique_name(constant_name.str(), operator_names);
                Operator* constant = candidate.new_operator_before("prim::Constant", constant_operator_name, op);
                const std::string constant_operand_name = unique_name(constant_name.str(), operand_names);
                operand = candidate.new_operand(constant_operand_name);
                operand->producer = constant;
                constant->outputs.push_back(operand);
                std::string detail;
                if (exported_argument_to_parameter(arguments[j].value, target, arguments[j].name, constant->params["value"], detail) != 0)
                {
                    error = "cannot lower non-tensor argument " + arguments[j].name + " for " + node.target + ": " + detail;
                    return -1;
                }
            }

            operand->consumers.push_back(op);
            op->inputs.push_back(operand);
            op->inputnames.push_back(arguments[j].name);
        }

        for (size_t j = 0; j < node.outputs.size(); j++)
        {
            if (node.outputs[j].type == EXPORTED_ARGUMENT_SYM_INT && op->type != "aten::size")
            {
                error = "unsupported symbolic computation for " + node.target;
                return -1;
            }
            if (node.outputs[j].type == EXPORTED_ARGUMENT_TENSOR || node.outputs[j].type == EXPORTED_ARGUMENT_SYM_INT)
            {
                const std::string& name = node.outputs[j].name;
                if (values.find(name) != values.end())
                {
                    error = "tensor value " + name + " is defined more than once";
                    return -1;
                }
                Operand* operand = candidate.new_operand(unique_name(name, operand_names));
                operand->producer = op;
                op->outputs.push_back(operand);
                if (node.outputs[j].type == EXPORTED_ARGUMENT_TENSOR && set_tensor_metadata(normalized_graph, name, operand, error) != 0)
                    return -1;
                values[name] = operand;
                continue;
            }

            if (node.outputs[j].type == EXPORTED_ARGUMENT_TENSOR_LIST)
            {
                std::ostringstream list_name;
                list_name << "pnnx_" << unknown_index++;
                Operator* unpack = candidate.new_operator_after("prim::ListUnpack", unique_name(list_name.str(), operator_names), op);
                Operand* list_operand = candidate.new_operand(unique_name(list_name.str(), operand_names));
                list_operand->producer = op;
                list_operand->consumers.push_back(unpack);
                op->outputs.push_back(list_operand);
                unpack->inputs.push_back(list_operand);

                for (size_t k = 0; k < node.outputs[j].tensor_names.size(); k++)
                {
                    const std::string& name = node.outputs[j].tensor_names[k];
                    if (values.find(name) != values.end())
                    {
                        error = "tensor value " + name + " is defined more than once";
                        return -1;
                    }
                    Operand* operand = candidate.new_operand(unique_name(name, operand_names));
                    operand->producer = unpack;
                    unpack->outputs.push_back(operand);
                    if (set_tensor_metadata(normalized_graph, name, operand, error) != 0)
                        return -1;
                    values[name] = operand;
                }
                continue;
            }

            if (node.outputs[j].type == EXPORTED_ARGUMENT_NONE)
            {
                // Keep unused return slots so later outputs retain their schema indices.
                std::ostringstream unused_name;
                unused_name << op->name << "_unused_" << j;
                Operand* operand = candidate.new_operand(unique_name(unused_name.str(), operand_names));
                operand->producer = op;
                op->outputs.push_back(operand);
                continue;
            }

            error = "unsupported non-tensor output for " + node.target;
            return -1;
        }
    }

    std::vector<Operand*> flat_outputs;
    flat_outputs.reserve(source_program.output_specs.size());
    for (size_t i = 0; i < source_program.output_specs.size(); i++)
    {
        const ExportedOutputSpec& spec = source_program.output_specs[i];
        const std::map<std::string, Operand*>::const_iterator value_it = values.find(spec.arg.name);
        if (value_it == values.end())
        {
            error = "unknown graph output " + spec.arg.name;
            return -1;
        }

        flat_outputs.push_back(value_it->second);
    }

    if (source_program.output_tree_spec.type != EXPORTED_TREE_SPEC_NONE)
    {
        size_t flat_index = 0;
        Operand* tree_output = 0;
        if (construct_output_tree(source_program.output_tree_spec, flat_outputs, flat_index, candidate, operand_names, operator_names, unknown_index, tree_output, error) != 0)
            return -1;
        if (flat_index != flat_outputs.size())
        {
            error = "output treespec did not consume all graph outputs";
            return -1;
        }

        Operator* op = candidate.new_operator("pnnx.Output", unique_name("pnnx_output_0", operator_names));
        tree_output->consumers.push_back(op);
        op->inputs.push_back(tree_output);
    }
    else
    {
        for (size_t i = 0; i < flat_outputs.size(); i++)
        {
            std::ostringstream output_name;
            output_name << "pnnx_output_" << i;
            Operator* op = candidate.new_operator("pnnx.Output", unique_name(output_name.str(), operator_names));
            Operand* operand = flat_outputs[i];
            operand->consumers.push_back(op);
            op->inputs.push_back(operand);
        }
    }

    graph.ops.swap(candidate.ops);
    graph.operands.swap(candidate.operands);
    return 0;
}

static int parse_program_json(const JsonValue& value, const std::string& entry, ExportedProgram& program, std::string& error)
{
    ExportedSchemaError schema_error;
    if (parse_exported_program(value, program, schema_error) != 0)
    {
        error = "invalid exported program " + entry + " at " + schema_error.path + ": " + schema_error.message;
        return -1;
    }

    return 0;
}

static int parse_payload_json(const JsonValue& value, const std::string& entry, ExportedPayloadConfig& config, std::string& error)
{
    ExportedSchemaError schema_error;
    if (parse_exported_payload_config(value, config, schema_error) != 0)
    {
        error = "invalid exported payload config " + entry + " at " + schema_error.path + ": " + schema_error.message;
        return -1;
    }

    return 0;
}

static int materialize_program_state(Pt2ArchiveReader& reader,
                                     const ExportedProgram& program,
                                     const ExportedPayloadConfig& weights,
                                     const ExportedPayloadConfig& constants,
                                     std::map<std::string, MaterializedExportedTensor>& state,
                                     std::string& error)
{
    std::map<std::string, std::vector<char> > storage_cache;
    std::map<std::string, size_t> storage_uses;

    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const ExportedInputSpec& spec = program.input_specs[i];
        if (spec.kind != EXPORTED_PARAMETER && spec.kind != EXPORTED_BUFFER && spec.kind != EXPORTED_TENSOR_CONSTANT)
            continue;

        const bool is_constant = spec.kind == EXPORTED_TENSOR_CONSTANT || (spec.kind == EXPORTED_BUFFER && !spec.persistent);
        const ExportedPayloadConfig& config = is_constant ? constants : weights;
        const std::map<std::string, ExportedPayloadEntry>::const_iterator entry_it = config.entries.find(spec.target);
        if (entry_it == config.entries.end())
            continue;

        const std::string directory = is_constant ? "/data/constants/" : "/data/weights/";
        storage_uses[reader.layout().root + directory + entry_it->second.path_name]++;
    }

    for (size_t i = 0; i < program.input_specs.size(); i++)
    {
        const ExportedInputSpec& spec = program.input_specs[i];
        if (spec.kind != EXPORTED_PARAMETER && spec.kind != EXPORTED_BUFFER && spec.kind != EXPORTED_TENSOR_CONSTANT)
            continue;

        const bool is_constant = spec.kind == EXPORTED_TENSOR_CONSTANT || (spec.kind == EXPORTED_BUFFER && !spec.persistent);
        const ExportedPayloadConfig& config = is_constant ? constants : weights;
        const ExportedPayloadConfig& wrong_config = is_constant ? weights : constants;
        const char* config_name = is_constant ? "constants" : "weights";
        const char* wrong_config_name = is_constant ? "weights" : "constants";
        const char* state_kind = exported_state_kind_name(spec.kind);

        if (wrong_config.entries.find(spec.target) != wrong_config.entries.end())
        {
            error = std::string(state_kind) + " " + spec.target + " is present in " + wrong_config_name + " config";
            return -1;
        }

        const std::map<std::string, ExportedPayloadEntry>::const_iterator entry_it = config.entries.find(spec.target);
        if (entry_it == config.entries.end())
        {
            error = std::string(state_kind) + " " + spec.target + " is missing from " + config_name + " config";
            return -1;
        }

        const ExportedPayloadEntry& entry = entry_it->second;
        const std::string directory = is_constant ? "/data/constants/" : "/data/weights/";
        const std::string payload_path = reader.layout().root + directory + entry.path_name;

        const bool expected_is_param = spec.kind == EXPORTED_PARAMETER;
        if (entry.is_param != expected_is_param)
        {
            error = std::string(state_kind) + " " + spec.target + " in " + config_name + " config has is_param=" + (entry.is_param ? "true" : "false");
            return -1;
        }

        if (entry.use_pickle || !entry.has_tensor_meta)
        {
            error = std::string("pickled payload is unsupported for ") + state_kind + " " + spec.target + " at " + payload_path;
            return -1;
        }

        std::map<std::string, std::vector<char> >::iterator storage_it = storage_cache.find(payload_path);
        if (storage_it == storage_cache.end())
        {
            storage_it = storage_cache.insert(std::make_pair(payload_path, std::vector<char>())).first;
            if (reader.read_blob(payload_path, storage_it->second, error) != 0)
                return -1;
        }

        MaterializedExportedTensor tensor;
        if (materialize_exported_tensor(entry.tensor_meta, storage_it->second, reader.layout().byte_order, tensor, error) != 0)
        {
            error = spec.target + " from " + payload_path + ": " + error;
            return -1;
        }

        state[spec.target] = std::move(tensor);

        std::map<std::string, size_t>::iterator uses_it = storage_uses.find(payload_path);
        if (uses_it != storage_uses.end() && uses_it->second > 0)
        {
            uses_it->second--;
            if (uses_it->second == 0)
                storage_cache.erase(payload_path);
        }
    }

    return 0;
}

int load_exported_program(const std::string& pt2path, const ModelFormatInfo& format_info, Graph& graph, std::string& error)
try
{
    error.clear();

    Pt2ArchiveReader reader;
    if (reader.open(pt2path, format_info, error) != 0)
        return -1;

    ExportedProgram program;
    ExportedPayloadConfig weights;
    ExportedPayloadConfig constants;
    {
        JsonValue model_json;
        if (reader.read_json(reader.layout().model_json_path, model_json, error) != 0)
            return -1;
        if (parse_program_json(model_json, reader.layout().model_json_path, program, error) != 0)
            return -1;
    }
    {
        JsonValue weights_json;
        if (reader.read_json(reader.layout().weights_config_path, weights_json, error) != 0)
            return -1;
        if (parse_payload_json(weights_json, reader.layout().weights_config_path, weights, error) != 0)
            return -1;
    }
    {
        JsonValue constants_json;
        if (reader.read_json(reader.layout().constants_config_path, constants_json, error) != 0)
            return -1;
        if (parse_payload_json(constants_json, reader.layout().constants_config_path, constants, error) != 0)
            return -1;
    }

    std::map<std::string, MaterializedExportedTensor> state;
    if (materialize_program_state(reader, program, weights, constants, state, error) != 0)
        return -1;

    return lower_exported_program(program, state, graph, error);
}
catch (const std::length_error&)
{
    error = "exported program allocation failed";
    return -1;
}
catch (const std::bad_alloc&)
{
    error = "exported program allocation failed";
    return -1;
}

} // namespace pnnx
