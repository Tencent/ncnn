// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exported_program.h"

#include <stdio.h>
#include <string.h>

#include <limits>

namespace pnnx {

static int to_pnnx_type(int scalar_type)
{
    if (scalar_type == 1) return 8;
    if (scalar_type == 2) return 7;
    if (scalar_type == 3) return 6;
    if (scalar_type == 4) return 4;
    if (scalar_type == 5) return 5;
    if (scalar_type == 6) return 3;
    if (scalar_type == 7) return 1;
    if (scalar_type == 8) return 2;
    if (scalar_type == 9) return 12;
    if (scalar_type == 10) return 10;
    if (scalar_type == 11) return 11;
    if (scalar_type == 12) return 9;
    if (scalar_type == 13) return 13;
    return 0;
}

static std::string symbolic_shape_key(const std::string& expression)
{
    std::string key;
    for (size_t i = 0; i < expression.size(); i++)
    {
        const char ch = expression[i];
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_')
            key.push_back(ch);
        else if (key.empty() || key[key.size() - 1] != '_')
            key.push_back('_');
    }
    if (key.empty())
        key = "symbol";
    return key;
}

static bool to_pnnx_shape(const std::vector<pt2::SymInt>& dimensions, std::vector<int>& shape, std::map<std::string, Parameter>* params, std::string& error)
{
    for (size_t i = 0; i < dimensions.size(); i++)
    {
        const pt2::SymInt& dimension = dimensions[i];
        if (dimension.type == pt2::SymInt::Expression)
        {
            shape.push_back(-233);
            if (params)
            {
                const std::string suffix = std::to_string(i);
                (*params)["__shape__" + suffix] = symbolic_shape_key(dimension.expression);
                (*params)["__shape_expr__" + suffix] = dimension.expression;
                if (dimension.has_hint)
                    (*params)["__shape_hint__" + suffix] = dimension.hint;
            }
            continue;
        }

        int64_t value = dimension.integer;

        if (value < -1 || value > INT_MAX)
        {
            error = "tensor dimension " + std::to_string(i) + " is out of range";
            return false;
        }
        shape.push_back((int)value);
    }
    return true;
}

static bool checked_multiply_size(size_t lhs, size_t rhs, size_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        return false;
    result = lhs * rhs;
    return true;
}

static bool materialize_attribute(const pt2::PayloadMeta& payload, const std::vector<char>& storage, Attribute& attribute, std::string& error)
{
    attribute.type = to_pnnx_type(payload.tensor_meta.scalar_type);
    if (attribute.type == 0)
    {
        error = "unsupported tensor scalar type " + std::to_string(payload.tensor_meta.scalar_type);
        return false;
    }
    if (!to_pnnx_shape(payload.tensor_meta.sizes, attribute.shape, &attribute.params, error))
        return false;

    size_t element_count = 1;
    for (size_t i = 0; i < attribute.shape.size(); i++)
    {
        if (attribute.shape[i] < 0)
        {
            error = "attribute shape must be static";
            return false;
        }
        if (!checked_multiply_size(element_count, (size_t)attribute.shape[i], element_count))
        {
            error = "attribute element count overflows size_t";
            return false;
        }
    }

    size_t byte_count = 0;
    if (!checked_multiply_size(element_count, attribute.elemsize(), byte_count))
    {
        error = "attribute byte size overflows size_t";
        return false;
    }
    attribute.data.resize(byte_count);
    if (element_count == 0)
        return true;

    size_t source_element = (size_t)payload.tensor_meta.storage_offset.integer;
    for (size_t output_element = 0; output_element < element_count; output_element++)
    {
        size_t remaining = output_element;
        source_element = (size_t)payload.tensor_meta.storage_offset.integer;
        for (size_t axis = attribute.shape.size(); axis > 0; axis--)
        {
            const size_t dimension = (size_t)attribute.shape[axis - 1];
            const size_t index = remaining % dimension;
            remaining /= dimension;
            source_element += index * (size_t)payload.tensor_meta.strides[axis - 1].integer;
        }
        memcpy(attribute.data.data() + output_element * attribute.elemsize(), storage.data() + source_element * attribute.elemsize(), attribute.elemsize());
    }
    return true;
}

static const pt2::PayloadMeta* find_payload(const pt2::ExportedProgramArchive& archive, const pt2::InputSpec& spec, const std::map<std::string, std::vector<char> >*& storages)
{
    if (spec.type == pt2::InputSpec::Parameter)
    {
        storages = &archive.state_dict_storages;
        std::map<std::string, pt2::PayloadMeta>::const_iterator it = archive.state_dict.find(spec.target);
        return it == archive.state_dict.end() ? 0 : &it->second;
    }

    std::map<std::string, pt2::PayloadMeta>::const_iterator state = archive.state_dict.find(spec.target);
    if (state != archive.state_dict.end())
    {
        storages = &archive.state_dict_storages;
        return &state->second;
    }

    storages = &archive.constant_storages;
    std::map<std::string, pt2::PayloadMeta>::const_iterator constant = archive.constants.find(spec.target);
    return constant == archive.constants.end() ? 0 : &constant->second;
}

int import_exported_program_inputs(const pt2::ExportedProgramArchive& archive, Graph& graph, std::string& error)
{
    error.clear();
    if (archive.program.graph.inputs.size() != archive.program.signature.inputs.size())
    {
        error = "graph input count does not match graph signature";
        return -1;
    }

    int user_input_index = 0;
    for (size_t i = 0; i < archive.program.signature.inputs.size(); i++)
    {
        const pt2::InputSpec& spec = archive.program.signature.inputs[i];
        if (spec.type == pt2::InputSpec::UserInput && spec.argument.type == pt2::Argument::Tensor)
        {
            std::map<std::string, pt2::TensorMeta>::const_iterator meta = archive.program.graph.tensor_values.find(spec.argument.name);
            if (meta == archive.program.graph.tensor_values.end())
            {
                error = spec.argument.name + ": tensor metadata is missing";
                return -1;
            }

            Operator* op = graph.new_operator("pnnx.Input", "pnnx_input_" + std::to_string(user_input_index++));
            Operand* operand = graph.new_operand(spec.argument.name);
            operand->producer = op;
            operand->type = to_pnnx_type(meta->second.scalar_type);
            if (operand->type == 0 || !to_pnnx_shape(meta->second.sizes, operand->shape, &operand->params, error))
            {
                if (error.empty()) error = spec.argument.name + ": unsupported input tensor type";
                return -1;
            }
            op->outputs.push_back(operand);
            continue;
        }

        if (spec.type == pt2::InputSpec::Parameter || spec.type == pt2::InputSpec::Buffer || spec.type == pt2::InputSpec::TensorConstant)
        {
            const std::map<std::string, std::vector<char> >* storages = 0;
            const pt2::PayloadMeta* payload = find_payload(archive, spec, storages);
            if (!payload)
            {
                error = spec.target + ": tensor payload is missing";
                return -1;
            }

            const std::string storage_path = (storages == &archive.state_dict_storages ? "data/weights/" : "data/constants/") + payload->path;
            std::map<std::string, std::vector<char> >::const_iterator storage = storages->find(storage_path);
            if (storage == storages->end())
            {
                error = storage_path + ": tensor storage is missing";
                return -1;
            }

            Operator* op = graph.new_operator("pnnx.Attribute", spec.target);
            if (!materialize_attribute(*payload, storage->second, op->attrs["data"], error))
            {
                error = spec.target + ": " + error;
                return -1;
            }
            Operand* operand = graph.new_operand(spec.argument.name);
            operand->producer = op;
            operand->type = op->attrs["data"].type;
            operand->shape = op->attrs["data"].shape;
            op->outputs.push_back(operand);
            continue;
        }

        error = "unsupported graph input at index " + std::to_string(i);
        return -1;
    }
    return 0;
}

static std::string normalize_target(const std::string& target)
{
    const std::string prefix = "torch.ops.";
    if (target.compare(0, prefix.size(), prefix) != 0)
        return target;

    const size_t namespace_end = target.find('.', prefix.size());
    if (namespace_end == std::string::npos)
        return target;
    const size_t operator_end = target.find('.', namespace_end + 1);
    const std::string name_space = target.substr(prefix.size(), namespace_end - prefix.size());
    const std::string operator_name = target.substr(namespace_end + 1, operator_end == std::string::npos ? std::string::npos : operator_end - namespace_end - 1);
    return name_space + "::" + operator_name;
}

static bool to_parameter(const pt2::Argument& argument, Parameter& parameter, std::string& error)
{
    if (argument.type == pt2::Argument::None)
    {
        parameter = Parameter();
        return true;
    }
    if (argument.type == pt2::Argument::Boolean)
    {
        parameter = Parameter(argument.boolean);
        return true;
    }
    if (argument.type == pt2::Argument::Integer)
    {
        if (argument.integer < INT_MIN || argument.integer > INT_MAX)
        {
            error = "integer argument is out of pnnx range";
            return false;
        }
        parameter = Parameter((int)argument.integer);
        return true;
    }
    if (argument.type == pt2::Argument::FloatingPoint)
    {
        parameter = Parameter(argument.floating_point);
        return true;
    }
    if (argument.type == pt2::Argument::String)
    {
        parameter = Parameter(argument.string);
        return true;
    }
    if (argument.type == pt2::Argument::Integers)
    {
        std::vector<int> values;
        for (size_t i = 0; i < argument.values.size(); i++)
        {
            if (argument.values[i].integer < INT_MIN || argument.values[i].integer > INT_MAX)
            {
                error = "integer list argument is out of pnnx range";
                return false;
            }
            values.push_back((int)argument.values[i].integer);
        }
        parameter = Parameter(values);
        return true;
    }
    if (argument.type == pt2::Argument::FloatingPoints)
    {
        std::vector<double> values;
        for (size_t i = 0; i < argument.values.size(); i++)
            values.push_back(argument.values[i].floating_point);
        parameter = Parameter(values);
        return true;
    }
    if (argument.type == pt2::Argument::Strings)
    {
        std::vector<std::string> values;
        for (size_t i = 0; i < argument.values.size(); i++)
            values.push_back(argument.values[i].string);
        parameter = Parameter(values);
        return true;
    }
    if (argument.type == pt2::Argument::ScalarType)
    {
        if (argument.integer <= 0 || argument.integer > INT_MAX)
        {
            error = "scalar type is out of range";
            return false;
        }
        parameter = Parameter((int)argument.integer - 1);
        return true;
    }
    if (argument.type == pt2::Argument::MemoryFormat || argument.type == pt2::Argument::Layout)
    {
        if (argument.integer < 0 || argument.integer > INT_MAX)
        {
            error = "enum argument is out of range";
            return false;
        }
        parameter = Parameter((int)argument.integer);
        return true;
    }
    if (argument.type == pt2::Argument::DeviceValue)
    {
        std::string device = argument.device.type;
        if (argument.device.has_index)
            device += ":" + std::to_string(argument.device.index);
        parameter = Parameter(device);
        return true;
    }

    error = "unsupported constant argument type " + std::to_string((int)argument.type);
    return false;
}

static Operand* make_constant(const pt2::Argument& argument, const std::string& name, Graph& graph, std::string& error)
{
    Parameter parameter;
    if (!to_parameter(argument, parameter, error))
        return 0;

    Operator* constant = graph.new_operator("prim::Constant", name);
    constant->params["value"] = parameter;
    Operand* output = graph.new_operand(name);
    output->producer = constant;
    constant->outputs.push_back(output);
    return output;
}

static bool is_list_argument(const pt2::Argument& argument)
{
    return argument.type == pt2::Argument::Tensors
           || argument.type == pt2::Argument::OptionalTensors
           || argument.type == pt2::Argument::Integers
           || argument.type == pt2::Argument::FloatingPoints
           || argument.type == pt2::Argument::Booleans
           || argument.type == pt2::Argument::Strings;
}

static Operand* resolve_argument(const pt2::Argument& argument, const std::string& name, Graph& graph, std::string& error)
{
    if (argument.type == pt2::Argument::Tensor)
    {
        Operand* input = graph.get_operand(argument.name);
        if (!input)
            error = "tensor input " + argument.name + " is not defined";
        return input;
    }

    if (argument.type == pt2::Argument::OptionalTensor)
    {
        if (argument.values.size() != 1)
        {
            error = "optional tensor must contain one variant";
            return 0;
        }
        return resolve_argument(argument.values[0], name, graph, error);
    }

    if (is_list_argument(argument))
    {
        std::vector<Operand*> items;
        for (size_t i = 0; i < argument.values.size(); i++)
        {
            Operand* item = resolve_argument(argument.values[i], name + "_item_" + std::to_string(i), graph, error);
            if (!item)
                return 0;
            items.push_back(item);
        }

        Operator* list = graph.new_operator("prim::ListConstruct", name);
        for (size_t i = 0; i < items.size(); i++)
        {
            Operand* item = items[i];
            item->consumers.push_back(list);
            list->inputs.push_back(item);
        }
        Operand* output = graph.new_operand(name);
        output->producer = list;
        list->outputs.push_back(output);
        return output;
    }

    return make_constant(argument, name, graph, error);
}

static bool collect_tensor_outputs(const pt2::Argument& argument, std::vector<std::string>& names, std::string& error)
{
    if (argument.type == pt2::Argument::Tensor)
    {
        names.push_back(argument.name);
        return true;
    }
    if (argument.type == pt2::Argument::Tensors)
    {
        for (size_t i = 0; i < argument.values.size(); i++)
        {
            if (argument.values[i].type != pt2::Argument::Tensor)
            {
                error = "tensor-list output contains a non-tensor value";
                return false;
            }
            names.push_back(argument.values[i].name);
        }
        return true;
    }
    error = "only tensor and tensor-list node outputs are supported";
    return false;
}

int import_exported_program_nodes(const pt2::ExportedProgram& program, Graph& graph, std::string& error)
{
    error.clear();
    int unnamed_node_index = 0;
    for (size_t i = 0; i < program.graph.nodes.size(); i++)
    {
        const pt2::Node& node = program.graph.nodes[i];
        const std::string name = node.name.empty() ? "pnnx_" + std::to_string(unnamed_node_index++) : node.name;
        const std::string target = normalize_target(node.target);
        if (target.find("::") == std::string::npos)
        {
            error = name + ": unsupported exported operator " + node.target;
            return -1;
        }

        std::vector<Operand*> inputs;
        std::vector<std::string> input_names;
        for (size_t j = 0; j < node.inputs.size(); j++)
        {
            const pt2::NamedArgument& named_argument = node.inputs[j];
            Operand* input = resolve_argument(named_argument.argument, name + "_arg_" + std::to_string(j), graph, error);
            if (!input)
            {
                error = name + "." + named_argument.name + ": " + error;
                return -1;
            }

            inputs.push_back(input);
            input_names.push_back(named_argument.name);
        }

        std::vector<std::string> output_names;
        for (size_t j = 0; j < node.outputs.size(); j++)
        {
            if (!collect_tensor_outputs(node.outputs[j], output_names, error))
            {
                error = name + ": " + error;
                return -1;
            }
        }

        std::vector<Operand*> outputs;
        for (size_t j = 0; j < output_names.size(); j++)
        {
            const std::string& output_name = output_names[j];
            if (graph.get_operand(output_name))
            {
                error = name + ": tensor output " + output_name + " is already defined";
                return -1;
            }

            Operand* output = graph.new_operand(output_name);
            std::map<std::string, pt2::TensorMeta>::const_iterator meta = program.graph.tensor_values.find(output_name);
            if (meta != program.graph.tensor_values.end())
            {
                output->type = to_pnnx_type(meta->second.scalar_type);
                if (output->type == 0 || !to_pnnx_shape(meta->second.sizes, output->shape, &output->params, error))
                {
                    if (error.empty()) error = name + ": unsupported output tensor type";
                    return -1;
                }
            }
            outputs.push_back(output);
        }

        Operator* op = graph.new_operator(target, name);
        op->inputs = inputs;
        op->inputnames = input_names;
        op->outputs = outputs;
        for (size_t j = 0; j < inputs.size(); j++)
            inputs[j]->consumers.push_back(op);
        for (size_t j = 0; j < outputs.size(); j++)
            outputs[j]->producer = op;
    }
    return 0;
}

int import_exported_program_outputs(const pt2::ExportedProgram& program, Graph& graph, std::string& error)
{
    error.clear();
    if (program.graph.outputs.size() != program.signature.outputs.size())
    {
        error = "graph output count does not match graph signature";
        return -1;
    }

    int output_index = 0;
    for (size_t i = 0; i < program.graph.outputs.size(); i++)
    {
        const pt2::Argument& output = program.graph.outputs[i];
        const pt2::OutputSpec& spec = program.signature.outputs[i];
        if (spec.type != pt2::OutputSpec::UserOutput || output.type != spec.argument.type)
        {
            error = "unsupported graph output at index " + std::to_string(i);
            return -1;
        }

        if (output.type == pt2::Argument::Tensor || output.type == pt2::Argument::Tensors)
        {
            std::vector<std::string> output_names;
            std::vector<std::string> signature_names;
            if (!collect_tensor_outputs(output, output_names, error) || !collect_tensor_outputs(spec.argument, signature_names, error) || output_names != signature_names)
            {
                if (error.empty()) error = "graph output does not match graph signature";
                return -1;
            }
            for (size_t j = 0; j < output_names.size(); j++)
            {
                Operand* operand = graph.get_operand(output_names[j]);
                if (!operand)
                {
                    error = "graph output " + output_names[j] + " is not defined";
                    return -1;
                }
                Operator* op = graph.new_operator("pnnx.Output", "pnnx_output_" + std::to_string(output_index++));
                operand->consumers.push_back(op);
                op->inputs.push_back(operand);
            }
            continue;
        }

        Operand* operand = resolve_argument(output, "pnnx_output_value_" + std::to_string(i), graph, error);
        if (!operand)
        {
            error = "graph output " + std::to_string(i) + ": " + error;
            return -1;
        }
        Operator* op = graph.new_operator("pnnx.Output", "pnnx_output_" + std::to_string(output_index++));
        operand->consumers.push_back(op);
        op->inputs.push_back(operand);
    }
    return 0;
}

static std::string extract_symbol_name(const std::string& expression)
{
    const std::string prefix = "Symbol('";
    if (expression.compare(0, prefix.size(), prefix) != 0)
        return std::string();
    const size_t end = expression.find('\'', prefix.size());
    if (end == std::string::npos)
        return std::string();
    return expression.substr(prefix.size(), end - prefix.size());
}

bool validate_exported_program_input_shapes(const pt2::ExportedProgram& program, const std::vector<std::vector<int64_t> >& input_shapes, std::string& error)
{
    error.clear();
    if (input_shapes.empty())
        return true;

    std::vector<std::pair<std::string, const pt2::TensorMeta*> > inputs;
    for (size_t i = 0; i < program.signature.inputs.size(); i++)
    {
        const pt2::InputSpec& spec = program.signature.inputs[i];
        if (spec.type != pt2::InputSpec::UserInput || spec.argument.type != pt2::Argument::Tensor)
            continue;
        std::map<std::string, pt2::TensorMeta>::const_iterator meta = program.graph.tensor_values.find(spec.argument.name);
        if (meta == program.graph.tensor_values.end())
        {
            error = spec.argument.name + ": tensor metadata is missing";
            return false;
        }
        inputs.push_back(std::make_pair(spec.argument.name, &meta->second));
    }

    if (input_shapes.size() != inputs.size())
    {
        error = "inputshape count mismatch: expected " + std::to_string(inputs.size()) + " but got " + std::to_string(input_shapes.size());
        return false;
    }

    std::map<std::string, int64_t> expression_values;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        const std::string& input_name = inputs[i].first;
        const pt2::TensorMeta& meta = *inputs[i].second;
        if (input_shapes[i].size() != meta.sizes.size())
        {
            error = "input " + input_name + " rank mismatch: expected " + std::to_string(meta.sizes.size()) + " but got " + std::to_string(input_shapes[i].size());
            return false;
        }

        for (size_t dimension = 0; dimension < meta.sizes.size(); dimension++)
        {
            const int64_t actual = input_shapes[i][dimension];
            const pt2::SymInt& expected = meta.sizes[dimension];
            const std::string location = "input " + input_name + " dimension " + std::to_string(dimension);
            if (actual < 0)
            {
                error = location + " has invalid value " + std::to_string(actual);
                return false;
            }
            if (actual > INT_MAX)
            {
                error = location + " is " + std::to_string(actual) + ", exceeds pnnx dimension limit " + std::to_string(INT_MAX);
                return false;
            }
            if (expected.type == pt2::SymInt::Integer)
            {
                if (actual != expected.integer)
                {
                    error = location + " is " + std::to_string(actual) + ", expected " + std::to_string(expected.integer);
                    return false;
                }
                continue;
            }

            std::map<std::string, int64_t>::const_iterator bound = expression_values.find(expected.expression);
            if (bound != expression_values.end() && bound->second != actual)
            {
                error = location + " is " + std::to_string(actual) + ", shared symbol requires " + std::to_string(bound->second);
                return false;
            }
            expression_values[expected.expression] = actual;

            const std::string symbol = extract_symbol_name(expected.expression);
            std::map<std::string, pt2::RangeConstraint>::const_iterator range = program.range_constraints.find(symbol);
            if (range != program.range_constraints.end())
            {
                if ((range->second.has_min && actual < range->second.min) || (range->second.has_max && actual > range->second.max))
                {
                    const std::string minimum = range->second.has_min ? std::to_string(range->second.min) : "-inf";
                    const std::string maximum = range->second.has_max ? std::to_string(range->second.max) : "inf";
                    error = location + " is " + std::to_string(actual) + ", allowed range is [" + minimum + ", " + maximum + "]";
                    return false;
                }
            }
        }
    }
    return true;
}

static void apply_input_shapes(const pt2::ExportedProgram& program, const std::vector<std::vector<int64_t> >& input_shapes, Graph& graph)
{
    if (input_shapes.empty())
        return;
    size_t input_index = 0;
    for (size_t i = 0; i < program.signature.inputs.size(); i++)
    {
        const pt2::InputSpec& spec = program.signature.inputs[i];
        if (spec.type != pt2::InputSpec::UserInput || spec.argument.type != pt2::Argument::Tensor)
            continue;
        Operand* operand = graph.get_operand(spec.argument.name);
        operand->shape.clear();
        for (size_t dimension = 0; dimension < input_shapes[input_index].size(); dimension++)
            operand->shape.push_back((int)input_shapes[input_index][dimension]);
        operand->params.clear();
        input_index++;
    }
}

int load_exported_program(const std::string& path, Graph& graph,
                          const std::vector<std::vector<int64_t> >& input_shapes,
                          const std::vector<std::vector<int64_t> >& input_shapes2)
{
    pt2::ExportedProgramArchive archive;
    std::string error;
    if (!pt2::load_exported_program_archive(path, archive, error))
    {
        fprintf(stderr, "load exported program failed: %s\n", error.c_str());
        return -1;
    }

    if (!validate_exported_program_input_shapes(archive.program, input_shapes, error) || !validate_exported_program_input_shapes(archive.program, input_shapes2, error))
    {
        fprintf(stderr, "load exported program failed: %s\n", error.c_str());
        return -1;
    }

    if (import_exported_program_inputs(archive, graph, error) != 0)
    {
        fprintf(stderr, "load exported program failed: %s\n", error.c_str());
        return -1;
    }
    apply_input_shapes(archive.program, input_shapes, graph);

    if (import_exported_program_nodes(archive.program, graph, error) != 0)
    {
        fprintf(stderr, "load exported program failed: %s\n", error.c_str());
        return -1;
    }

    if (import_exported_program_outputs(archive.program, graph, error) != 0)
    {
        fprintf(stderr, "load exported program failed: %s\n", error.c_str());
        return -1;
    }

    return 0;
}

} // namespace pnnx