// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "load_exported_program.h"

#include <stdio.h>
#include <string.h>

#include <cmath>
#include <limits>
#include <set>

#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>

#include "exported_program_defaults.h"

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

static bool is_symbol_identifier(const std::string& text);
static std::string extract_symbol_name(const std::string& expression, bool& positive);

static bool is_supported_symbolic_expression(const std::string& expression)
{
    if (is_symbol_identifier(expression))
        return true;
    const char* supported_identifiers[] = {"Symbol", "Integer", "Add", "Mul", "Pow"};
    for (size_t offset = 0; offset < expression.size();)
    {
        const char ch = expression[offset];
        if (ch == '\'' || ch == '"')
        {
            const char quote = ch;
            offset++;
            while (offset < expression.size() && expression[offset] != quote)
                offset++;
            if (offset < expression.size())
                offset++;
            continue;
        }
        if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '_')
        {
            size_t end = offset + 1;
            while (end < expression.size())
            {
                const char next = expression[end];
                if (!((next >= 'A' && next <= 'Z') || (next >= 'a' && next <= 'z') || (next >= '0' && next <= '9') || next == '_'))
                    break;
                end++;
            }
            const std::string identifier = expression.substr(offset, end - offset);
            bool supported = identifier == "True" || identifier == "False" || identifier == "integer" || identifier == "positive" || identifier == "negative" || identifier == "nonnegative" || identifier == "nonpositive";
            for (size_t i = 0; i < sizeof(supported_identifiers) / sizeof(supported_identifiers[0]); i++)
                supported = supported || identifier == supported_identifiers[i];
            if (!supported)
                return false;
            offset = end;
            continue;
        }
        offset++;
    }
    return true;
}

static bool to_pnnx_shape(const std::vector<pt2::SymInt>& dimensions, std::vector<int>& shape, std::map<std::string, Parameter>* params, std::string& error)
{
    for (size_t i = 0; i < dimensions.size(); i++)
    {
        const pt2::SymInt& dimension = dimensions[i];
        if (dimension.type == pt2::SymInt::Expression)
        {
            if (!dimension.has_hint && !is_supported_symbolic_expression(dimension.expression))
            {
                error = "unsupported symbolic expression without hint at dimension " + std::to_string(i) + ": " + dimension.expression;
                return false;
            }
            shape.push_back(-233);
            if (params)
            {
                const std::string suffix = std::to_string(i);
                (*params)["__shape__" + suffix] = symbolic_shape_key(dimension.expression);
                (*params)["__shape_expr__" + suffix] = dimension.expression;
                if (dimension.has_hint)
                {
                    if (dimension.hint < 0 || dimension.hint > INT_MAX)
                    {
                        error = "symbolic shape hint is out of pnnx range";
                        return false;
                    }
                    (*params)["__shape_hint__" + suffix] = dimension.hint;
                }
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
    // This entry is also reached without the archive reader. Never rely on a
    // previous validation: the shared check bounds every address and allocation.
    if (!pt2::validate_tensor_storage(payload, storage.size(), error))
        return false;
    attribute = Attribute();
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
        if (attribute.shape[i] == 0) element_count = 0;
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

static bool equal_float(double lhs, double rhs)
{
    // Preserve signed zero; serialized NaNs have no payload to compare.
    return (std::isnan(lhs) && std::isnan(rhs)) || (lhs == rhs && (lhs != 0.0 || std::signbit(lhs) == std::signbit(rhs)));
}

static bool arguments_agree(const pt2::Argument& lhs, const pt2::Argument& rhs)
{
    if (lhs.type != rhs.type || lhs.name != rhs.name || lhs.values.size() != rhs.values.size())
        return false;
    if (!lhs.name.empty() && (lhs.type == pt2::Argument::Tensor || lhs.type == pt2::Argument::SymInteger || lhs.type == pt2::Argument::SymBoolean || lhs.type == pt2::Argument::SymFloat))
        return true;
    switch (lhs.type)
    {
    case pt2::Argument::Unknown:
        return false;
    case pt2::Argument::Integer:
    case pt2::Argument::SymInteger:
    case pt2::Argument::ScalarType:
    case pt2::Argument::MemoryFormat:
    case pt2::Argument::Layout:
        return lhs.integer == rhs.integer;
    case pt2::Argument::Boolean:
    case pt2::Argument::SymBoolean:
        return lhs.boolean == rhs.boolean;
    case pt2::Argument::FloatingPoint:
    case pt2::Argument::SymFloat:
        return equal_float(lhs.floating_point, rhs.floating_point);
    case pt2::Argument::Complex:
        return equal_float(lhs.complex_real, rhs.complex_real) && equal_float(lhs.complex_imag, rhs.complex_imag);
    case pt2::Argument::String:
        return lhs.string == rhs.string;
    case pt2::Argument::DeviceValue:
        return lhs.device.type == rhs.device.type && lhs.device.has_index == rhs.device.has_index && (!lhs.device.has_index || lhs.device.index == rhs.device.index);
    default:
        break;
    }
    for (size_t i = 0; i < lhs.values.size(); i++)
        if (!arguments_agree(lhs.values[i], rhs.values[i])) return false;
    return true;
}

static bool preserve_input_contract(const pt2::ExportedProgram& program, const pt2::TensorMeta& meta, Operator* op, std::string& error)
{
    const std::vector<int>& shape = op->outputs[0]->shape;
    std::vector<std::string> symbols(shape.size()); // Empty means a static axis.
    std::vector<int> minimum(shape.size());
    std::vector<int> maximum(shape.size());
    for (size_t axis = 0; axis < shape.size(); axis++)
    {
        if (meta.sizes[axis].type == pt2::SymInt::Integer)
        {
            if (shape[axis] < 0)
            {
                error = "static input dimension must be nonnegative";
                return false;
            }
            minimum[axis] = maximum[axis] = shape[axis];
            continue;
        }

        bool positive = false;
        symbols[axis] = extract_symbol_name(meta.sizes[axis].expression, positive);
        if (symbols[axis].empty())
        {
            error = "unsupported input contract expression; only bare symbols are supported";
            return false;
        }
        // Intersect, never narrow int64 bounds by a wrapping cast. An empty
        // intersection must be refused, not clamped to an admissible endpoint.
        int64_t lower = positive ? 1 : 0;
        int64_t upper = INT_MAX;
        std::map<std::string, pt2::RangeConstraint>::const_iterator range = program.range_constraints.find(symbols[axis]);
        if (range != program.range_constraints.end())
        {
            if (range->second.has_min && range->second.min > lower) lower = range->second.min;
            if (range->second.has_max && range->second.max < upper) upper = range->second.max;
        }
        if (lower > upper)
        {
            error = "input symbol " + symbols[axis] + ": allowed range has no supported pnnx dimension";
            return false;
        }
        minimum[axis] = (int)lower;
        maximum[axis] = (int)upper;
    }

    // Keep the serialized contract on the operator: apply_input_shapes and
    // optimization passes may specialize/clear operand shape metadata. Hints
    // and conversion samples are not replacement runtime constraints.
    op->params["__pt2_input_type"] = op->outputs[0]->type;
    op->params["__pt2_input_shape"] = shape;
    op->params["__pt2_input_symbols"] = symbols;
    op->params["__pt2_input_min"] = minimum;
    op->params["__pt2_input_max"] = maximum;
    // These are PNNX IR/Python metadata, not native ncnn Input parameters.
    // save_ncnn skips reserved __ keys; raw ncnn .param enforces none of them.
    return true;
}

static bool validate_identifier(const std::string& name, const std::string& location, std::string& error)
{
    if (is_symbol_identifier(name))
        return true;
    // Do not echo untrusted bytes (including embedded NULs) into diagnostics.
    error = location + ": expected nonempty ASCII identifier [A-Za-z_][A-Za-z0-9_]*";
    return false;
}

static bool validate_argument_names(const pt2::Argument& argument, bool node_output, const std::string& location, std::string& error)
{
    const bool reference = argument.type == pt2::Argument::Tensor
                           || (node_output && (argument.type == pt2::Argument::SymInteger || argument.type == pt2::Argument::SymBoolean || argument.type == pt2::Argument::SymFloat));
    if ((reference || !argument.name.empty()) && !validate_identifier(argument.name, location + ".name", error))
        return false;
    for (size_t i = 0; i < argument.values.size(); i++)
        if (!validate_argument_names(argument.values[i], node_output, location + ".values[" + std::to_string(i) + "]", error)) return false;
    return true;
}

static bool is_attribute_fqn(const std::string& name)
{
    // ModuleList/Sequential indices are numeric, including at the root.
    for (size_t begin = 0;;)
    {
        const size_t end = name.find('.', begin);
        const std::string segment = name.substr(begin, end == std::string::npos ? end : end - begin);
        const bool numeric = !segment.empty() && segment.find_first_not_of("0123456789") == std::string::npos;
        if (!is_symbol_identifier(segment) && !numeric)
            return false;
        if (end == std::string::npos)
            return true;
        begin = end + 1;
    }
}

static std::string attribute_operator_name(const std::string& target)
{
    // A valid Sequential FQN such as 0.weight cannot begin self.0_weight_data.
    // Prefix only a validated numeric root; payload lookup still uses the FQN.
    // This emitted name participates in the same collision checks as all ops.
    return !target.empty() && target[0] >= '0' && target[0] <= '9' ? "_" + target : target;
}

static bool is_exported_target_name(const std::string& target)
{
    if (target.compare(0, 10, "_operator.") == 0)
        return is_symbol_identifier(target.substr(10));
    if (target.compare(0, 9, "operator.") == 0)
        return is_symbol_identifier(target.substr(9));
    if (target.compare(0, 10, "torch.ops.") != 0)
        return false;
    size_t begin = 10;
    for (int part = 0; part < 3; part++)
    {
        const size_t end = target.find('.', begin);
        if ((part == 2) != (end == std::string::npos)
                || !is_symbol_identifier(target.substr(begin, end == std::string::npos ? end : end - begin)))
            return false;
        if (part != 2) begin = end + 1;
    }
    // Syntax is necessary, not authority: append_default_arguments still checks
    // the actual dispatcher schema or exact supported pure Python function.
    return true;
}

static bool validate_exported_program_names(const pt2::ExportedProgram& program, std::string& error)
{
    size_t index = 0;
    for (std::map<std::string, pt2::TensorMeta>::const_iterator it = program.graph.tensor_values.begin(); it != program.graph.tensor_values.end(); ++it, ++index)
        if (!validate_identifier(it->first, "graph.tensor_values[" + std::to_string(index) + "].name", error)) return false;
    index = 0;
    for (std::map<std::string, pt2::SymInt>::const_iterator it = program.graph.sym_int_values.begin(); it != program.graph.sym_int_values.end(); ++it, ++index)
        if (!validate_identifier(it->first, "graph.sym_int_values[" + std::to_string(index) + "].name", error)) return false;
    for (size_t i = 0; i < program.graph.inputs.size(); i++)
        if (!validate_argument_names(program.graph.inputs[i], false, "graph.inputs[" + std::to_string(i) + "]", error)) return false;
    for (size_t i = 0; i < program.signature.inputs.size(); i++)
    {
        const pt2::InputSpec& spec = program.signature.inputs[i];
        const std::string location = "signature.inputs[" + std::to_string(i) + "]";
        if (!validate_argument_names(spec.argument, false, location + ".argument", error)) return false;
        if ((spec.type == pt2::InputSpec::Parameter || spec.type == pt2::InputSpec::Buffer || spec.type == pt2::InputSpec::TensorConstant) && !is_attribute_fqn(spec.target))
        {
            error = location + ".target: expected ASCII FQN with nonempty identifier or numeric segments";
            return false;
        }
    }
    for (size_t i = 0; i < program.graph.nodes.size(); i++)
    {
        const pt2::Node& node = program.graph.nodes[i];
        const std::string location = "graph.nodes[" + std::to_string(i) + "]";
        if (!node.name.empty() && !validate_identifier(node.name, location + ".name", error)) return false;
        if (!is_exported_target_name(node.target))
        {
            error = location + ".target: expected torch.ops.<namespace>.<operator>.<overload> or operator.<function> / _operator.<function> with ASCII identifiers";
            return false;
        }
        for (size_t j = 0; j < node.inputs.size(); j++)
        {
            const std::string input_location = location + ".inputs[" + std::to_string(j) + "]";
            if (!validate_identifier(node.inputs[j].name, input_location + ".name", error)
                    || !validate_argument_names(node.inputs[j].argument, false, input_location + ".argument", error)) return false;
        }
        for (size_t j = 0; j < node.outputs.size(); j++)
            if (!validate_argument_names(node.outputs[j], true, location + ".outputs[" + std::to_string(j) + "]", error)) return false;
    }
    for (size_t i = 0; i < program.graph.outputs.size(); i++)
        if (!validate_argument_names(program.graph.outputs[i], false, "graph.outputs[" + std::to_string(i) + "]", error)) return false;
    for (size_t i = 0; i < program.signature.outputs.size(); i++)
        if (!validate_argument_names(program.signature.outputs[i].argument, false, "signature.outputs[" + std::to_string(i) + "].argument", error)) return false;
    return true;
}

struct ImportNames
{
    // Keep operator/self names separate from v_<operand> variables. A node and
    // its result normally share a name; that is not a collision.
    std::map<std::string, std::string> operators;
    std::set<std::string> values;

    void collect(const pt2::Argument& argument)
    {
        if (!argument.name.empty()) values.insert(argument.name);
        for (size_t i = 0; i < argument.values.size(); i++) collect(argument.values[i]);
    }

    bool reserve_operator(const std::string& name, std::string& error)
    {
        std::string identifier = name;
        // Match IR identifier sanitization, including names of preexisting ops.
        for (size_t i = 0; i < identifier.size(); i++)
            if (identifier[i] == '.' || identifier[i] == ':' || identifier[i] == '/') identifier[i] = '_';
        const std::pair<std::map<std::string, std::string>::iterator, bool> inserted = operators.insert(std::make_pair(identifier, name));
        if (inserted.second)
            return true;
        error = "operator name collision: '" + name + "' conflicts with '" + inserted.first->second + "' after identifier sanitization to '" + identifier + "'";
        return false;
    }

    bool initialize(const pt2::ExportedProgram& program, const Graph& graph, std::string& error)
    {
        for (size_t i = 0; i < graph.ops.size(); i++)
            if (!reserve_operator(graph.ops[i]->name, error)) return false;
        for (size_t i = 0; i < graph.operands.size(); i++) values.insert(graph.operands[i]->name);
        for (std::map<std::string, pt2::TensorMeta>::const_iterator it = program.graph.tensor_values.begin(); it != program.graph.tensor_values.end(); ++it) values.insert(it->first);
        for (std::map<std::string, pt2::SymInt>::const_iterator it = program.graph.sym_int_values.begin(); it != program.graph.sym_int_values.end(); ++it) values.insert(it->first);
        for (size_t i = 0; i < program.graph.inputs.size(); i++) collect(program.graph.inputs[i]);
        for (size_t i = 0; i < program.signature.inputs.size(); i++) collect(program.signature.inputs[i].argument);
        for (size_t i = 0; i < program.graph.nodes.size(); i++)
        {
            const pt2::Node& node = program.graph.nodes[i];
            for (size_t j = 0; j < node.inputs.size(); j++) collect(node.inputs[j].argument);
            for (size_t j = 0; j < node.outputs.size(); j++) collect(node.outputs[j]);
        }
        for (size_t i = 0; i < program.graph.outputs.size(); i++) collect(program.graph.outputs[i]);
        for (size_t i = 0; i < program.signature.outputs.size(); i++) collect(program.signature.outputs[i].argument);
        return true;
    }

    bool reserve_generated_value(const std::string& name, std::string& error)
    {
        // Include future definitions and references, not just current IR values:
        // a generated constant must never accidentally satisfy a model reference.
        if (!values.insert(name).second)
        {
            error = "generated value name collision: '" + name + "' is already defined or referenced";
            return false;
        }
        return reserve_operator(name, error);
    }
};

int import_exported_program_inputs(const pt2::ExportedProgramArchive& archive, Graph& graph, std::string& error)
{
    error.clear();
    if (!pt2::validate_exported_program_version(archive.program, error) || !validate_exported_program_names(archive.program, error))
        return -1;
    if (archive.program.graph.inputs.size() != archive.program.signature.inputs.size())
    {
        error = "graph input count does not match graph signature";
        return -1;
    }

    std::set<std::string> input_names;
    for (size_t i = 0; i < archive.program.signature.inputs.size(); i++)
    {
        const pt2::Argument& argument = archive.program.signature.inputs[i].argument;
        if (!arguments_agree(archive.program.graph.inputs[i], argument))
        {
            error = "graph input " + std::to_string(i) + " does not match graph signature";
            return -1;
        }
        if (argument.name.empty() || !input_names.insert(argument.name).second || graph.get_operand(argument.name))
        {
            error = "graph input " + std::to_string(i) + ": empty or duplicate input name " + argument.name;
            return -1;
        }
    }
    if (!validate_exported_program_input_shapes(archive.program, std::vector<std::vector<int64_t> >(), error))
        return -1;

    ImportNames names;
    if (!names.initialize(archive.program, graph, error))
        return -1;
    int input_index = 0;
    for (size_t i = 0; i < archive.program.signature.inputs.size(); i++)
    {
        const pt2::InputSpec& spec = archive.program.signature.inputs[i];
        if (spec.type == pt2::InputSpec::UserInput && spec.argument.type == pt2::Argument::Tensor)
        {
            if (!names.reserve_operator("pnnx_input_" + std::to_string(input_index++), error)) return -1;
        }
        else if (spec.type == pt2::InputSpec::Parameter || spec.type == pt2::InputSpec::Buffer || spec.type == pt2::InputSpec::TensorConstant)
        {
            if (!names.reserve_operator(attribute_operator_name(spec.target), error)) return -1;
        }
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
            if (!preserve_input_contract(archive.program, meta->second, op, error))
            {
                error = spec.argument.name + ": " + error;
                return -1;
            }
            continue;
        }

        if (spec.type == pt2::InputSpec::Parameter || spec.type == pt2::InputSpec::Buffer || spec.type == pt2::InputSpec::TensorConstant)
        {
            if (spec.argument.type != pt2::Argument::Tensor)
            {
                error = "graph input " + std::to_string(i) + ": lifted input must be a tensor";
                return -1;
            }
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

            Operator* op = graph.new_operator("pnnx.Attribute", attribute_operator_name(spec.target));
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

static bool restore_factory_dtypes(pt2::ExportedProgram& program, std::string& error)
{
    for (size_t i = 0; i < program.graph.nodes.size(); i++)
    {
        pt2::Node& node = program.graph.nodes[i];
        // Only these exact non-out factories are covered. Serde may encode an
        // integer fill as as_float, and the producer's default dtype may differ
        // from the runtime's. Never recover dtype by coercing the fill scalar.
        if (node.target != "torch.ops.aten.full.default" && node.target != "torch.ops.aten.full_like.default")
            continue;

        size_t dtype_index = node.inputs.size();
        for (size_t j = 0; j < node.inputs.size(); j++)
        {
            if (node.inputs[j].name == "dtype")
            {
                dtype_index = j;
                break;
            }
        }
        // Leave explicit values (and duplicate/malformed arguments) to the
        // existing schema and constant validation; metadata is not an override.
        if (dtype_index != node.inputs.size() && node.inputs[dtype_index].argument.type != pt2::Argument::None)
            continue;
        if (node.outputs.size() != 1 || node.outputs[0].type != pt2::Argument::Tensor)
            continue;

        const std::map<std::string, pt2::TensorMeta>::const_iterator meta = program.graph.tensor_values.find(node.outputs[0].name);
        // Metadata is optional for in-memory schema-only graphs. With no output
        // metadata, retain the original default rather than guessing from self,
        // fill_value or another tensor. full_like also uses its result metadata.
        if (meta == program.graph.tensor_values.end())
            continue;
        const int scalar_type = meta->second.scalar_type;
        if (to_pnnx_type(scalar_type) == 0)
        {
            error = (node.name.empty() ? "unnamed node" : node.name) + " (" + node.target + ").dtype: unsupported serde scalar type " + std::to_string(scalar_type) + " in output tensor metadata for " + node.outputs[0].name;
            return false;
        }

        if (dtype_index == node.inputs.size())
        {
            pt2::NamedArgument dtype;
            dtype.name = "dtype";
            dtype.kind = pt2::NamedArgument::Keyword;
            node.inputs.push_back(dtype);
        }
        // Keep the serde enum until to_parameter maps it to c10 (e.g. bf16).
        // Mutate only the variant/value so malformed names or nested references
        // on a supplied None cannot be erased before schema validation.
        node.inputs[dtype_index].argument.type = pt2::Argument::ScalarType;
        node.inputs[dtype_index].argument.integer = scalar_type;
    }
    return true;
}

static std::string normalize_target(const std::string& target)
{
    if (target.compare(0, 10, "_operator.") == 0)
        return "operator." + target.substr(10);

    const std::string prefix = "torch.ops.";
    if (target.compare(0, prefix.size(), prefix) != 0)
        return target;

    const size_t namespace_end = target.find('.', prefix.size());
    if (namespace_end == std::string::npos)
        return target;
    const size_t operator_end = target.find('.', namespace_end + 1);
    const std::string name_space = target.substr(prefix.size(), namespace_end - prefix.size());
    const std::string operator_name = target.substr(namespace_end + 1, operator_end == std::string::npos ? std::string::npos : operator_end - namespace_end - 1);
    if (name_space == "aten" && operator_name == "lift_fresh_copy")
        return "Tensor.clone";
    if (name_space == "aten" && operator_name == "alias")
        return target;
    if (name_space == "aten" && operator_name == "tril")
        return "Tensor.tril";
    if (name_space == "aten" && operator_name == "item")
        return "Tensor.item";
    if (name_space == "aten" && (operator_name == "rnn_tanh" || operator_name == "rnn_relu" || operator_name == "gru" || operator_name == "lstm"))
        return "torch._VF." + operator_name;
    if (name_space == "aten" && operator_name == "chunk")
        return "torch.chunk";
    if (name_space == "aten" && (operator_name == "split" || operator_name == "split_with_sizes"))
        return "torch.split";
    if (name_space == "aten" && operator_name == "tensor_split")
        return "torch.tensor_split";
    if (name_space == "aten" && operator_name == "unbind")
        return "torch.unbind";
    if (name_space == "aten" && (operator_name == "full" || operator_name == "hann_window" || operator_name == "hamming_window" || operator_name == "sym_size" || operator_name == "_assert_scalar"))
        return target;
    return name_space + "::" + operator_name;
}

static bool to_parameter(const pt2::Argument& argument, Parameter& parameter, std::string& error)
{
    if (argument.type == pt2::Argument::None)
    {
        parameter = Parameter();
        return true;
    }
    if (argument.type == pt2::Argument::Boolean || (argument.type == pt2::Argument::SymBoolean && argument.name.empty()))
    {
        parameter = Parameter(argument.boolean);
        return true;
    }
    if (argument.type == pt2::Argument::Integer)
    {
        int64_t value = argument.integer;
        if (value == std::numeric_limits<int64_t>::max()) value = INT_MAX;
        if (value == std::numeric_limits<int64_t>::max() - 1) value = INT_MAX - 1;
        if (value == std::numeric_limits<int64_t>::min()) value = INT_MIN;
        if (value == std::numeric_limits<int64_t>::min() + 1) value = INT_MIN + 1;
        if (value < INT_MIN || value > INT_MAX)
        {
            error = "integer argument is out of pnnx range";
            return false;
        }
        parameter = Parameter((int)value);
        return true;
    }
    if (argument.type == pt2::Argument::SymInteger && argument.name.empty())
    {
        // Unlike ATen slice sentinels, a concrete SymInt is an actual value.
        if (argument.integer < INT_MIN || argument.integer > INT_MAX)
        {
            error = "symbolic integer argument is out of pnnx range";
            return false;
        }
        parameter = Parameter((int)argument.integer);
        return true;
    }
    if (argument.type == pt2::Argument::FloatingPoint || (argument.type == pt2::Argument::SymFloat && argument.name.empty()))
    {
        parameter = Parameter(argument.floating_point);
        return true;
    }
    if (argument.type == pt2::Argument::Complex)
    {
        parameter = Parameter(std::complex<float>((float)argument.complex_real, (float)argument.complex_imag));
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
        // Serde is not the c10 enum: e.g. serde BFLOAT16=13, c10 BFloat16=15.
        c10::ScalarType dtype;
        switch (argument.integer)
        {
        case 1:
            dtype = c10::ScalarType::Byte;
            break;
        case 2:
            dtype = c10::ScalarType::Char;
            break;
        case 3:
            dtype = c10::ScalarType::Short;
            break;
        case 4:
            dtype = c10::ScalarType::Int;
            break;
        case 5:
            dtype = c10::ScalarType::Long;
            break;
        case 6:
            dtype = c10::ScalarType::Half;
            break;
        case 7:
            dtype = c10::ScalarType::Float;
            break;
        case 8:
            dtype = c10::ScalarType::Double;
            break;
        case 9:
            dtype = c10::ScalarType::ComplexHalf;
            break;
        case 10:
            dtype = c10::ScalarType::ComplexFloat;
            break;
        case 11:
            dtype = c10::ScalarType::ComplexDouble;
            break;
        case 12:
            dtype = c10::ScalarType::Bool;
            break;
        case 13:
            dtype = c10::ScalarType::BFloat16;
            break;
        default:
            error = "unsupported serde scalar type " + std::to_string(argument.integer);
            return false;
        }
        parameter = Parameter((int)dtype);
        return true;
    }
    if (argument.type == pt2::Argument::MemoryFormat)
    {
        c10::MemoryFormat format;
        switch (argument.integer)
        {
        case 1:
            format = c10::MemoryFormat::Contiguous;
            break;
        case 2:
            format = c10::MemoryFormat::ChannelsLast;
            break;
        case 3:
            format = c10::MemoryFormat::ChannelsLast3d;
            break;
        case 4:
            format = c10::MemoryFormat::Preserve;
            break;
        default:
            error = "unsupported serde memory format " + std::to_string(argument.integer);
            return false;
        }
        parameter = Parameter((int)format);
        return true;
    }
    if (argument.type == pt2::Argument::Layout)
    {
        if (argument.integer != 7)
        {
            error = "unsupported serde layout " + std::to_string(argument.integer) + "; only Strided (7) is supported";
            return false;
        }
        parameter = Parameter((int)c10::Layout::Strided);
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

static Operand* make_constant(const pt2::Argument& argument, const std::string& name, Graph& graph, ImportNames& names, std::string& error)
{
    Parameter parameter;
    if (!to_parameter(argument, parameter, error) || !names.reserve_generated_value(name, error))
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
           || argument.type == pt2::Argument::SymIntegers
           || argument.type == pt2::Argument::Strings;
}

static Operand* resolve_argument(const pt2::Argument& argument, const std::string& name, Graph& graph, ImportNames& names, std::string& error)
{
    if (argument.type == pt2::Argument::Tensor
            || ((argument.type == pt2::Argument::SymInteger || argument.type == pt2::Argument::SymBoolean || argument.type == pt2::Argument::SymFloat) && !argument.name.empty()))
    {
        Operand* input = graph.get_operand(argument.name);
        if (!input)
            error = "input " + argument.name + " is not defined";
        return input;
    }

    if (argument.type == pt2::Argument::OptionalTensor)
    {
        if (argument.values.size() != 1)
        {
            error = "optional tensor must contain one variant";
            return 0;
        }
        return resolve_argument(argument.values[0], name, graph, names, error);
    }

    if (is_list_argument(argument))
    {
        if (argument.values.empty() && argument.type != pt2::Argument::Tensors && argument.type != pt2::Argument::OptionalTensors)
            return make_constant(argument, name, graph, names, error);

        if (!names.reserve_generated_value(name, error))
            return 0;

        std::vector<Operand*> items;
        for (size_t i = 0; i < argument.values.size(); i++)
        {
            Operand* item = resolve_argument(argument.values[i], name + "_item_" + std::to_string(i), graph, names, error);
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

    return make_constant(argument, name, graph, names, error);
}

static bool collect_tensor_outputs(const pt2::Argument& argument, std::vector<std::string>& names, std::string& error)
{
    if (argument.type == pt2::Argument::Tensor || argument.type == pt2::Argument::SymInteger || argument.type == pt2::Argument::SymBoolean || argument.type == pt2::Argument::SymFloat)
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
    error = "only tensor, tensor-list and symbolic scalar node outputs are supported";
    return false;
}

int import_exported_program_nodes(const pt2::ExportedProgram& program, Graph& graph, std::string& error)
{
    error.clear();
    if (!pt2::validate_exported_program_version(program, error) || !validate_exported_program_names(program, error))
        return -1;
    // Public in-memory import must perform the same whole-program metadata,
    // ownership and liveness checks as archive loading. Never trust node tags
    // or an already existing IR operand as proof that a write target is local.
    pt2::ExportedProgram normalized = program;
    // Restore on the private copy before schema ordering/default binding. All
    // resulting constants still go through ImportNames and resolve_argument.
    if (!restore_factory_dtypes(normalized, error) || !pt2::append_default_arguments(normalized, error))
        return -1;
    ImportNames names;
    if (!names.initialize(normalized, graph, error))
        return -1;
    // Reserve even later node names before creating any argument constants.
    // These two guards are not emitted and consume no unnamed-node index.
    int reserved_node_index = 0;
    for (size_t i = 0; i < normalized.graph.nodes.size(); i++)
    {
        const pt2::Node& node = normalized.graph.nodes[i];
        if (node.target == "torch.ops.aten._assert_tensor_metadata.default" || node.target == "torch.ops.aten._assert_scalar.default")
            continue;
        const std::string name = node.name.empty() ? "pnnx_" + std::to_string(reserved_node_index++) : node.name;
        if (!names.reserve_operator(name, error)) return -1;
    }
    int unnamed_node_index = 0;
    for (size_t i = 0; i < normalized.graph.nodes.size(); i++)
    {
        const pt2::Node& node = normalized.graph.nodes[i];
        // Only exact, evaluated guards may be discharged. Even a metadata entry
        // and a graph input declaration do not establish an imported IR operand.
        if (node.target == "torch.ops.aten._assert_tensor_metadata.default")
        {
            const Operand* input = graph.get_operand(node.inputs[0].argument.name);
            if (!input)
            {
                error = node.name + " (" + node.target + "): input " + node.inputs[0].argument.name + " is not defined";
                return -1;
            }
            const pt2::TensorMeta& meta = program.graph.tensor_values.at(node.inputs[0].argument.name);
            std::vector<int> shape;
            if (!to_pnnx_shape(meta.sizes, shape, 0, error) || input->type != to_pnnx_type(meta.scalar_type) || input->shape != shape)
            {
                error = node.name + " (" + node.target + "): imported tensor dtype/shape disagrees with guard metadata for " + input->name;
                return -1;
            }
            // append_default_arguments validates and retains this node. Before
            // dropping it, transfer an explicit stride check on a direct user
            // input to its runtime contract. Ordinary strided metadata, and
            // an omitted/None stride argument, do not imply exact strides.
            Operator* producer = input->producer;
            if (producer && producer->type == "pnnx.Input")
            {
                if (!producer->params.count("__pt2_input_type"))
                {
                    error = node.name + ": guarded input is missing its PT2 input contract";
                    return -1;
                }
                for (size_t j = 1; j < node.inputs.size(); j++)
                {
                    const pt2::NamedArgument& argument = node.inputs[j];
                    if (argument.name != "stride" || argument.argument.type == pt2::Argument::None)
                        continue;
                    std::vector<int> stride;
                    for (size_t axis = 0; axis < argument.argument.values.size(); axis++)
                    {
                        const int64_t value = argument.argument.values[axis].integer;
                        if (value < 0 || value > INT_MAX)
                        {
                            error = node.name + ": guarded input stride is out of pnnx range";
                            return -1;
                        }
                        stride.push_back((int)value);
                    }
                    producer->params["__pt2_input_stride"] = stride;
                }
            }
            continue;
        }
        if (node.target == "torch.ops.aten._assert_scalar.default")
            continue;

        const std::string name = node.name.empty() ? "pnnx_" + std::to_string(unnamed_node_index++) : node.name;
        std::string target = normalize_target(node.target);
        bool scalar_full = false;
        if (node.target == "torch.ops.aten.full.default" && !node.inputs.empty())
        {
            const pt2::NamedArgument& size = node.inputs[0];
            scalar_full = size.name == "size"
                          && (size.argument.type == pt2::Argument::Integers || size.argument.type == pt2::Argument::SymIntegers)
                          && size.argument.values.empty();
            if (scalar_full)
                target = "aten::scalar_tensor";
        }
        if (target.find("::") == std::string::npos && target.compare(0, 6, "torch.") != 0 && target.compare(0, 7, "Tensor.") != 0 && target.compare(0, 9, "operator.") != 0)
        {
            error = name + ": unsupported exported operator " + node.target;
            return -1;
        }

        std::vector<Operand*> inputs;
        std::vector<std::string> input_names;
        for (size_t j = 0; j < node.inputs.size(); j++)
        {
            const pt2::NamedArgument& named_argument = node.inputs[j];
            if (scalar_full && named_argument.name == "size")
                continue;
            Operand* input = resolve_argument(named_argument.argument, name + "_arg_" + std::to_string(j), graph, names, error);
            if (!input)
            {
                error = name + "." + named_argument.name + ": " + error;
                return -1;
            }

            inputs.push_back(input);
            std::string input_name = named_argument.name;
            if (target == "torch.chunk" || target == "torch.tensor_split" || target == "torch.unbind")
            {
                if (input_name == "self") input_name = "input";
            }
            if (target == "torch.split")
            {
                if (input_name == "self") input_name = "tensor";
                if (input_name == "split_size" || input_name == "split_sizes") input_name = "split_size_or_sections";
            }
            input_names.push_back(input_name);
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
            if (output_name.empty() || graph.get_operand(output_name))
            {
                error = name + ": tensor output " + output_name + " is empty or already defined";
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
    if (!pt2::validate_exported_program_version(program, error) || !validate_exported_program_names(program, error))
        return -1;
    if (program.graph.outputs.size() != program.signature.outputs.size())
    {
        error = "graph output count does not match graph signature";
        return -1;
    }

    ImportNames names;
    if (!names.initialize(program, graph, error))
        return -1;
    int reserved_output_index = 0;
    for (size_t i = 0; i < program.graph.outputs.size(); i++)
    {
        const pt2::Argument& output = program.graph.outputs[i];
        const size_t count = output.type == pt2::Argument::Tensors ? output.values.size() : 1;
        for (size_t j = 0; j < count; j++)
            if (!names.reserve_operator("pnnx_output_" + std::to_string(reserved_output_index++), error)) return -1;
    }

    int output_index = 0;
    for (size_t i = 0; i < program.graph.outputs.size(); i++)
    {
        const pt2::Argument& output = program.graph.outputs[i];
        const pt2::OutputSpec& spec = program.signature.outputs[i];
        if (spec.type != pt2::OutputSpec::UserOutput)
        {
            error = "graph output " + std::to_string(i) + ": unsupported signature output kind " + std::to_string((int)spec.type) + " for " + spec.target + "; mutation, gradient and token outputs are not supported";
            return -1;
        }
        if (output.type != spec.argument.type)
        {
            error = "graph output " + std::to_string(i) + " type does not match graph signature";
            return -1;
        }
        if (!arguments_agree(output, spec.argument))
        {
            error = "graph output " + std::to_string(i) + " does not match graph signature (name or constant value)";
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

        Operand* operand = resolve_argument(output, "pnnx_output_value_" + std::to_string(i), graph, names, error);
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

static bool is_symbol_identifier(const std::string& text)
{
    if (text.empty()) return false;
    for (size_t i = 0; i < text.size(); i++)
    {
        const char ch = text[i];
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_' || (i != 0 && ch >= '0' && ch <= '9')))
            return false;
    }
    return true;
}

static std::string extract_symbol_name(const std::string& expression, bool& positive)
{
    positive = false;
    if (is_symbol_identifier(expression))
        return expression;
    // Accept only an entire Symbol, not a prefix of Symbol(...) + ... . These
    // are the srepr assumptions emitted for ordinary nonnegative input sizes.
    std::string compact;
    bool quoted = false;
    for (size_t i = 0; i < expression.size(); i++)
    {
        if (expression[i] == '\'') quoted = !quoted;
        if (quoted || (expression[i] != ' ' && expression[i] != '\t')) compact.push_back(expression[i]);
    }
    const std::string prefix = "Symbol('";
    if (compact.compare(0, prefix.size(), prefix) != 0)
        return std::string();
    const size_t end = compact.find('\'', prefix.size());
    if (end == std::string::npos || !is_symbol_identifier(compact.substr(prefix.size(), end - prefix.size())))
        return std::string();
    size_t offset = end + 1;
    while (offset < compact.size() && compact[offset] == ',')
    {
        const size_t next = compact.find_first_of(",)", offset + 1);
        if (next == std::string::npos) return std::string();
        const std::string assumption = compact.substr(offset + 1, next - offset - 1);
        if (assumption == "positive=True")
            positive = true;
        else if (assumption != "integer=True" && assumption != "nonnegative=True")
            return std::string();
        offset = next;
    }
    if (offset + 1 != compact.size() || compact[offset] != ')')
        return std::string();
    return compact.substr(prefix.size(), end - prefix.size());
}

bool validate_exported_program_input_shapes(const pt2::ExportedProgram& program, const std::vector<std::vector<int64_t> >& input_shapes, std::string& error)
{
    error.clear();
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
        for (size_t dimension = 0; dimension < meta->second.sizes.size(); dimension++)
        {
            const pt2::SymInt& expected = meta->second.sizes[dimension];
            bool positive = false;
            if (expected.type == pt2::SymInt::Expression && extract_symbol_name(expected.expression, positive).empty())
            {
                error = "input " + spec.argument.name + " dimension " + std::to_string(dimension) + ": unvalidated derived input expression " + expected.expression + "; only bare symbols are supported, hints are not constraints";
                return false;
            }
        }
        inputs.push_back(std::make_pair(spec.argument.name, &meta->second));
    }

    for (std::map<std::string, pt2::RangeConstraint>::const_iterator it = program.range_constraints.begin(); it != program.range_constraints.end(); ++it)
    {
        if (!is_symbol_identifier(it->first))
        {
            error = "range_constraints." + it->first + ": unvalidated derived range constraint; only bare symbols are supported";
            return false;
        }
        if (it->second.has_min && it->second.has_max && it->second.min > it->second.max)
        {
            error = "range_constraints." + it->first + ": minimum exceeds maximum";
            return false;
        }
    }
    // Validate conversion-time samples here. The independent runtime contract
    // is preserved on pnnx.Input and emitted by Graph::python, not specialized
    // to these samples. Native ncnn callers must enforce it themselves.
    if (input_shapes.empty())
        return true;

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

            bool positive = false;
            const std::string symbol = extract_symbol_name(expected.expression, positive);
            if (positive && actual == 0)
            {
                error = location + " violates positive symbol assumption";
                return false;
            }
            std::map<std::string, int64_t>::const_iterator bound = expression_values.find(symbol);
            if (bound != expression_values.end() && bound->second != actual)
            {
                error = location + " is " + std::to_string(actual) + ", shared symbol requires " + std::to_string(bound->second);
                return false;
            }
            expression_values[symbol] = actual;
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

    if (!validate_exported_program_names(archive.program, error)
            || !restore_factory_dtypes(archive.program, error)
            || !pt2::append_default_arguments(archive.program, error))
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