// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exported_program.h"

#include "exported_program_graph.h"
#include "exported_program_operator.h"
#include "exported_program_schema.h"
#include "exported_program_tensor.h"
#include "pt2_archive.h"

#include <torch/csrc/api/include/torch/version.h>
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 9)
#include <ATen/ops/empty.h>
#include <ATen/ops/einsum.h>
#include <c10/util/Exception.h>
#endif

#include <limits.h>

#include <cctype>
#include <complex>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pnnx {

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

    std::vector<int> shape;
    shape.reserve(meta.sizes.size());
    for (size_t i = 0; i < meta.sizes.size(); i++)
    {
        if (meta.sizes[i] < 0)
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

    operand->type = pnnx_type;
    operand->shape.swap(shape);
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

static std::string unique_name(const std::string& requested, std::set<std::string>& names)
{
    if (names.insert(requested).second)
        return requested;

    for (size_t suffix = 1;; suffix++)
    {
        std::ostringstream candidate;
        candidate << requested << '_' << suffix;
        if (names.insert(candidate.str()).second)
            return candidate.str();
    }
}

static int validate_output_tree(const ExportedTreeSpec& tree_spec, size_t& leaf_count, std::string& error)
{
    if (tree_spec.type == EXPORTED_TREE_SPEC_LEAF)
    {
        if (!tree_spec.children.empty())
        {
            error = "output treespec leaf must not have children";
            return -1;
        }

        leaf_count++;
        return 0;
    }

    if (tree_spec.type != EXPORTED_TREE_SPEC_TUPLE && tree_spec.type != EXPORTED_TREE_SPEC_LIST)
    {
        error = "invalid output treespec type";
        return -1;
    }

    for (size_t i = 0; i < tree_spec.children.size(); i++)
    {
        if (validate_output_tree(tree_spec.children[i], leaf_count, error) != 0)
            return -1;
    }

    return 0;
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
    if (value == std::numeric_limits<int64_t>::max())
        converted = INT_MAX;
    else if (value == std::numeric_limits<int64_t>::max() - 1)
        converted = INT_MAX - 1;
    else if (value == std::numeric_limits<int64_t>::min())
        converted = INT_MIN;
    else if (value == std::numeric_limits<int64_t>::min() + 1)
        converted = INT_MIN + 1;
    else if (value < (int64_t)INT_MIN || value > (int64_t)INT_MAX)
        return -1;
    else
        converted = (int)value;

    return 0;
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

static bool validate_and_normalize_exported_einsum_equation(const std::string& value, const std::vector<std::vector<int64_t> >& operand_shapes, const std::vector<int64_t>& output_shape, std::string& normalized, std::string& detail)
{
    detail.clear();

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 9)
    try
    {
        const at::TensorOptions options = at::TensorOptions().device(at::kMeta).dtype(at::kFloat);
        std::vector<at::Tensor> operands;
        operands.reserve(operand_shapes.size());
        for (size_t i = 0; i < operand_shapes.size(); i++)
        {
            if (operand_shapes[i].empty())
            {
                detail = "scalar einsum operands are unsupported";
                return false;
            }
            operands.push_back(at::empty(operand_shapes[i], options));
        }

        const at::Tensor output = at::einsum(value, operands);
        if (output.sizes().vec() != output_shape)
        {
            detail = "equation result shape does not match tensor metadata";
            return false;
        }
    }
    catch (const c10::Error& e)
    {
        detail = "invalid equation or tensor shapes: ";
        detail += e.what_without_backtrace();
        return false;
    }
    catch (const std::exception& e)
    {
        detail = "cannot validate equation: ";
        detail += e.what();
        return false;
    }
#else
    (void)value;
    (void)operand_shapes;
    (void)output_shape;
    normalized.clear();
    detail = "einsum equation validation requires libtorch 2.9 or newer";
    return false;
#endif

    normalized.clear();
    normalized.reserve(value.size());
    for (size_t i = 0; i < value.size(); i++)
    {
        if (value[i] != ' ')
            normalized.push_back(value[i]);
    }

    return true;
}

static int exported_argument_to_parameter(const ExportedArgument& argument, Parameter& parameter, std::string& detail)
{
    detail.clear();

    if (argument.type == EXPORTED_ARGUMENT_NONE)
    {
        parameter.type = 0;
    }
    else if (argument.type == EXPORTED_ARGUMENT_INT)
    {
        parameter.type = 2;
        if (exported_int_to_pnnx(argument.int_value, parameter.i) != 0)
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
        parameter.type = 3;
        parameter.f = (float)argument.float_value;
    }
    else if (argument.type == EXPORTED_ARGUMENT_FLOAT_LIST)
    {
        parameter.type = 6;
        parameter.af.reserve(argument.float_values.size());
        for (size_t i = 0; i < argument.float_values.size(); i++)
            parameter.af.push_back((float)argument.float_values[i]);
    }
    else if (argument.type == EXPORTED_ARGUMENT_COMPLEX)
    {
        parameter.type = 10;
        parameter.c = std::complex<float>((float)argument.complex_real_value, (float)argument.complex_imag_value);
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
        if (argument.device_value.type.empty())
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

static int lower_exported_program(const ExportedProgram& source_program,
                                  std::map<std::string, MaterializedExportedTensor>& state,
                                  Graph& graph,
                                  std::string& error)
{
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
    if (source_program.output_tree_spec.type != EXPORTED_TREE_SPEC_NONE)
    {
        size_t leaf_count = 0;
        if (validate_output_tree(source_program.output_tree_spec, leaf_count, error) != 0)
            return -1;
        if (leaf_count != normalized_graph.outputs.size())
        {
            error = "output treespec leaf count does not match graph outputs";
            return -1;
        }
    }

    Graph candidate;
    std::map<std::string, Operand*> values;
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

            Operator* op = candidate.new_operator("pnnx.Attribute", unique_name(spec.target, operator_names));
            Operand* operand = candidate.new_operand(name);
            operand_names.insert(name);
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
        Operand* operand = candidate.new_operand(name);
        operand_names.insert(name);
        operand->producer = op;
        op->outputs.push_back(operand);
        if (set_tensor_metadata(normalized_graph, name, operand, error) != 0)
            return -1;
        values[name] = operand;
    }

    for (size_t i = 0; i < normalized_graph.nodes.size(); i++)
    {
        const ExportedNode& node = normalized_graph.nodes[i];
        std::vector<CanonicalExportedArgument> arguments;
        ExportedOperatorTarget target;
        if (canonicalize_exported_arguments(node, source_program.header, target, arguments, error) != 0)
            return -1;

        if (target.operator_name == "aten::einsum" && target.overload_name.empty())
        {
            CanonicalExportedArgument* equation_argument = 0;
            const CanonicalExportedArgument* tensors_argument = 0;
            for (size_t j = 0; j < arguments.size(); j++)
            {
                if (arguments[j].name == "equation" && arguments[j].value.type == EXPORTED_ARGUMENT_STRING)
                    equation_argument = &arguments[j];
                if (arguments[j].name == "tensors" && arguments[j].value.type == EXPORTED_ARGUMENT_TENSOR_LIST)
                    tensors_argument = &arguments[j];
            }
            if (!equation_argument || !tensors_argument)
            {
                error = "cannot lower " + node.target + ": invalid canonical einsum arguments";
                return -1;
            }

            std::vector<std::vector<int64_t> > operand_shapes;
            operand_shapes.reserve(tensors_argument->value.tensor_names.size());
            for (size_t j = 0; j < tensors_argument->value.tensor_names.size(); j++)
            {
                const std::string& tensor_name = tensors_argument->value.tensor_names[j];
                const std::map<std::string, ExportedTensorMeta>::const_iterator meta_it = normalized_graph.tensor_values.find(tensor_name);
                if (meta_it == normalized_graph.tensor_values.end())
                {
                    error = "missing tensor metadata for einsum operand " + tensor_name;
                    return -1;
                }
                operand_shapes.push_back(meta_it->second.sizes);
            }

            if (node.outputs.size() != 1 || node.outputs[0].type != EXPORTED_ARGUMENT_TENSOR)
            {
                error = "cannot lower " + node.target + ": invalid einsum output";
                return -1;
            }
            const std::map<std::string, ExportedTensorMeta>::const_iterator output_meta_it = normalized_graph.tensor_values.find(node.outputs[0].name);
            if (output_meta_it == normalized_graph.tensor_values.end())
            {
                error = "missing tensor metadata for einsum output " + node.outputs[0].name;
                return -1;
            }

            std::string normalized_equation;
            std::string detail;
            if (!validate_and_normalize_exported_einsum_equation(equation_argument->value.string_value, operand_shapes, output_meta_it->second.sizes, normalized_equation, detail))
            {
                error = "cannot lower non-tensor argument equation for " + node.target + ": " + detail;
                return -1;
            }
            equation_argument->value.string_value.swap(normalized_equation);
        }

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

        for (size_t j = 0; j < arguments.size(); j++)
        {
            Operand* operand = 0;
            if (arguments[j].value.type == EXPORTED_ARGUMENT_TENSOR)
            {
                const std::map<std::string, Operand*>::const_iterator value_it = values.find(arguments[j].value.name);
                if (value_it == values.end())
                {
                    error = "unknown tensor value " + arguments[j].value.name + " for " + node.target;
                    return -1;
                }

                operand = value_it->second;
            }
            else if (arguments[j].value.type == EXPORTED_ARGUMENT_TENSOR_LIST)
            {
                std::ostringstream list_name;
                list_name << "pnnx_" << unknown_index++;
                Operator* list = candidate.new_operator_before("prim::ListConstruct", unique_name(list_name.str(), operator_names), op);
                operand = candidate.new_operand(unique_name(list_name.str(), operand_names));
                operand->producer = list;
                list->outputs.push_back(operand);

                for (size_t k = 0; k < arguments[j].value.tensor_names.size(); k++)
                {
                    const std::string& tensor_name = arguments[j].value.tensor_names[k];
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
                if (exported_argument_to_parameter(arguments[j].value, constant->params["value"], detail) != 0)
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
            if (node.outputs[j].type == EXPORTED_ARGUMENT_TENSOR)
            {
                const std::string& name = node.outputs[j].name;
                if (values.find(name) != values.end())
                {
                    error = "tensor value " + name + " is defined more than once";
                    return -1;
                }
                Operand* operand = candidate.new_operand(name);
                operand_names.insert(name);
                operand->producer = op;
                op->outputs.push_back(operand);
                if (set_tensor_metadata(normalized_graph, name, operand, error) != 0)
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
                    Operand* operand = candidate.new_operand(name);
                    operand_names.insert(name);
                    operand->producer = unpack;
                    unpack->outputs.push_back(operand);
                    if (set_tensor_metadata(normalized_graph, name, operand, error) != 0)
                        return -1;
                    values[name] = operand;
                }
                continue;
            }

            if (node.outputs[j].type == EXPORTED_ARGUMENT_NONE)
                continue;

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

} // namespace pnnx
