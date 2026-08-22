// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pt2_graph_lowering.h"

#include <limits.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#include <torch/csrc/api/include/torch/version.h>

#include "ir.h"
#include "pt2_program.h"
#include "pt2_weights.h"
#include "utils.h"

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>

namespace pnnx {

static int scalar_type_to_pnnx(int type)
{
    static const int types[] = {0, 8, 7, 6, 4, 5, 3, 1, 2, 12, 10, 11, 9, 13};
    return type >= 1 && type <= 13 ? types[type] : 0;
}

static int scalar_type_to_c10(int type)
{
    static const int types[] = {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15};
    return type >= 1 && type <= 13 ? types[type] : -1;
}

static bool integer_can_be_parameter(int64_t value)
{
    return (value >= INT_MIN && value <= INT_MAX) || value == INT64_MIN || value == INT64_MIN + 1 || value == INT64_MAX - 1 || value == INT64_MAX;
}

static int parameter_from_argument(const Pt2Argument& arg, Parameter& value, std::string& error)
{
    if (arg.type == Pt2Argument::None)
        value = Parameter();
    else if (arg.type == Pt2Argument::Bool)
        value = Parameter(arg.b);
    else if (arg.type == Pt2Argument::Int)
    {
        if (!integer_can_be_parameter(arg.i))
        {
            error = "integer argument is outside the pnnx parameter range";
            return -1;
        }
        value = Parameter((long long)arg.i);
    }
    else if (arg.type == Pt2Argument::Ints)
    {
        for (size_t i = 0; i < arg.ai.size(); i++)
        {
            if (!integer_can_be_parameter(arg.ai[i]))
            {
                error = "integer list argument is outside the pnnx parameter range";
                return -1;
            }
        }
        value = Parameter(arg.ai);
    }
    else if (arg.type == Pt2Argument::Float)
        value = std::isinf(arg.f) ? Parameter(arg.f < 0 ? "-inf" : "inf") : Parameter(arg.f);
    else if (arg.type == Pt2Argument::Floats)
        value = Parameter(arg.af);
    else if (arg.type == Pt2Argument::Complex)
        value = Parameter(std::complex<double>(arg.af[0], arg.af[1]));
    else if (arg.type == Pt2Argument::String)
        value = Parameter(arg.s);
    else if (arg.type == Pt2Argument::Device)
    {
        if (arg.s != "cpu" || arg.i != -1)
        {
            error = "only unindexed CPU device arguments are supported";
            return -1;
        }
        value = Parameter(arg.s);
    }
    else if (arg.type == Pt2Argument::Strings)
        value = Parameter(arg.as);
    else if (arg.type == Pt2Argument::ScalarType)
    {
        const int type = scalar_type_to_c10((int)arg.i);
        if (type < 0)
        {
            error = "unsupported scalar type argument " + std::to_string(arg.i);
            return -1;
        }
        value = Parameter(type);
    }
    else if (arg.type == Pt2Argument::MemoryFormat)
    {
        if (arg.i < 1 || arg.i > 4)
        {
            error = "unsupported memory format argument " + std::to_string(arg.i);
            return -1;
        }
        value = Parameter((int)arg.i - 1);
    }
    else if (arg.type == Pt2Argument::Layout)
    {
        if (arg.i != 7)
        {
            error = "only strided layout arguments are supported";
            return -1;
        }
        value = Parameter(0);
    }
    else if (arg.type == Pt2Argument::SymInt && !arg.b)
    {
        if (!integer_can_be_parameter(arg.i))
        {
            error = "symbolic integer argument is outside the pnnx parameter range";
            return -1;
        }
        value = Parameter((long long)arg.i);
    }
    else
    {
        error = "argument cannot be represented as a constant";
        return -1;
    }
    return 0;
}

static int argument_from_ivalue(const c10::IValue& value, Pt2Argument& arg, std::string& error)
{
    if (value.isNone())
        arg.type = Pt2Argument::None;
    else if (value.isBool())
    {
        arg.type = Pt2Argument::Bool;
        arg.b = value.toBool();
    }
    else if (value.isInt())
    {
        arg.type = Pt2Argument::Int;
        arg.i = value.toInt();
    }
    else if (value.isDouble())
    {
        arg.type = Pt2Argument::Float;
        arg.f = value.toDouble();
    }
    else if (value.isString())
    {
        arg.type = Pt2Argument::String;
        arg.s = value.toStringRef();
    }
    else if (value.isDevice())
    {
        arg.type = Pt2Argument::Device;
        const c10::Device device = value.toDevice();
        arg.s = device.str();
        arg.i = device.has_index() ? device.index() : -1;
    }
    else if (value.isIntList())
    {
        arg.type = Pt2Argument::Ints;
        arg.ai = value.toIntVector();
    }
    else if (value.isDoubleList())
    {
        arg.type = Pt2Argument::Floats;
        arg.af = value.toDoubleVector();
    }
    else if (value.isBoolList())
    {
        arg.type = Pt2Argument::Bools;
        const c10::List<bool> values = value.toBoolList();
        for (size_t i = 0; i < values.size(); i++)
            arg.ab.push_back(values.get(i));
    }
    else
    {
        error = "unsupported dispatcher default value";
        return -1;
    }
    return 0;
}

static bool output_matches_schema(const Pt2Argument& output, const c10::TypePtr& type)
{
    if (output.type == Pt2Argument::Tensor)
        return type->kind() == c10::TypeKind::TensorType;
    if (output.type == Pt2Argument::Tensors)
        return type->kind() == c10::TypeKind::ListType && type->cast<c10::ListType>()->getElementType()->kind() == c10::TypeKind::TensorType;
    if (output.type == Pt2Argument::SymInt)
        return type->kind() == c10::TypeKind::SymIntType || type->kind() == c10::TypeKind::IntType || type->kind() == c10::TypeKind::NumberType;
    return false;
}

class Pt2GraphLowering
{
public:
    Pt2GraphLowering(const Pt2Program& _program, Pt2Weights& _weights, Graph& _graph, std::string& _error)
        : program(_program), weights(_weights), graph(_graph), error(_error)
    {
        generated_index = 0;
        input_index = 0;
        output_index = 0;

        for (std::map<std::string, Pt2Tensor>::const_iterator it = program.tensors.begin(); it != program.tensors.end(); ++it)
            reserved_names.insert(it->first);
        for (std::map<std::string, Pt2SymInt>::const_iterator it = program.sym_ints.begin(); it != program.sym_ints.end(); ++it)
            reserved_names.insert(it->first);
        for (std::set<std::string>::const_iterator it = program.sym_bools.begin(); it != program.sym_bools.end(); ++it)
            reserved_names.insert(*it);
        for (size_t i = 0; i < program.nodes.size(); i++)
        {
            std::string name = program.nodes[i].name;
            if (name.empty() && !program.nodes[i].outputs.empty())
                name = program.nodes[i].outputs[0].s;
            if (!name.empty())
            {
                program_operator_names.insert(name);
                reserved_names.insert(name);
            }
        }
        for (size_t i = 0; i < program.input_specs.size(); i++)
        {
            if (program.input_specs[i].kind == Pt2InputSpec::Parameter || program.input_specs[i].kind == Pt2InputSpec::Buffer || program.input_specs[i].kind == Pt2InputSpec::TensorConstant)
                reserved_names.insert(program.input_specs[i].target);
        }
    }

    int lower()
    {
        if (!graph.ops.empty() || !graph.operands.empty())
            return fail("destination graph is not empty");
        std::map<std::string, int>::const_iterator aten_opset = program.opset_versions.find("aten");
        if (aten_opset == program.opset_versions.end() || aten_opset->second != 10)
            return fail("unsupported aten opset " + (aten_opset == program.opset_versions.end() ? std::string("missing") : std::to_string(aten_opset->second)));

        for (size_t i = 0; i < program.input_specs.size(); i++)
        {
            if (lower_input(program.input_specs[i]) != 0)
                return -1;
        }
        for (size_t i = 0; i < program.nodes.size(); i++)
        {
            if (lower_node(i, program.nodes[i]) != 0)
                return -1;
        }
        for (size_t i = 0; i < program.output_specs.size(); i++)
        {
            if (lower_output(program.output_specs[i]) != 0)
                return -1;
        }
        return verify();
    }

private:
    int fail(const std::string& message)
    {
        error = message;
        return -1;
    }

    int fail_node(size_t index, const Pt2Node& node, const std::string& message, const std::string& schema = std::string())
    {
        const std::string name = !node.name.empty() ? node.name : !node.outputs.empty() ? node.outputs[0].s : std::string("unnamed");
        error = "node " + std::to_string(index) + " (" + name + ", target " + node.target + "): " + message;
        if (!schema.empty())
            error += "; schema " + schema;
        return -1;
    }

    std::string generated_name()
    {
        for (;;)
        {
            const std::string name = "pnnx_" + std::to_string(generated_index++);
            if (reserved_names.find(name) == reserved_names.end() && operand_names.find(name) == operand_names.end() && operator_names.find(name) == operator_names.end())
                return name;
        }
    }

    int add_operator_name(const std::string& name)
    {
        if (!operator_names.insert(name).second)
            return fail("duplicate operator name " + name);
        return 0;
    }

    Operand* new_operand(const std::string& name)
    {
        if (!operand_names.insert(name).second)
        {
            fail("duplicate operand name " + name);
            return 0;
        }
        return graph.new_operand(name);
    }

    int set_tensor_meta(const std::string& name, Operand* operand)
    {
        std::map<std::string, Pt2Tensor>::const_iterator it = program.tensors.find(name);
        if (it == program.tensors.end())
            return fail("missing tensor metadata for " + name);
        operand->type = scalar_type_to_pnnx(it->second.dtype);
        if (!operand->type)
            return fail("unsupported tensor dtype for " + name);
        operand->shape.reserve(it->second.sizes.size());
        for (size_t i = 0; i < it->second.sizes.size(); i++)
        {
            const Pt2SymInt& size = it->second.sizes[i];
            if (size.symbolic)
                return fail("dynamic tensor shape is unsupported for " + name);
            if (size.value < -1 || size.value > INT_MAX)
                return fail("invalid tensor dimension for " + name);
            operand->shape.push_back((int)size.value);
        }
        return 0;
    }

    int lower_input(const Pt2InputSpec& spec)
    {
        if (spec.kind == Pt2InputSpec::ConstantInput)
            return 0;
        if (spec.arg.type != Pt2Argument::Tensor)
            return fail("only tensor graph inputs are supported");

        if (spec.kind == Pt2InputSpec::UserInput)
        {
            std::string name = "pnnx_input_" + std::to_string(input_index++);
            if (program_operator_names.find(name) != program_operator_names.end() || operator_names.find(name) != operator_names.end())
                name = generated_name();
            if (add_operator_name(name) != 0)
                return -1;
            Operator* op = graph.new_operator("pnnx.Input", name);
            Operand* operand = new_operand(spec.arg.s);
            if (!operand || set_tensor_meta(spec.arg.s, operand) != 0)
                return -1;
            operand->producer = op;
            op->outputs.push_back(operand);
            return 0;
        }

        std::map<std::string, Pt2Weight>::iterator it = weights.values.find(spec.target);
        if (it == weights.values.end())
            return fail("missing weight " + spec.target);
        if (it->second.kind != spec.kind)
            return fail("weight kind mismatch for " + spec.target);
        std::string name = spec.target;
        if (program_operator_names.find(name) != program_operator_names.end() || operator_names.find(name) != operator_names.end())
            name = generated_name();
        if (add_operator_name(name) != 0)
            return -1;
        Operator* op = graph.new_operator("pnnx.Attribute", name);
        op->attrs["data"] = std::move(it->second.attribute);
        Operand* operand = new_operand(spec.arg.s);
        if (!operand || set_tensor_meta(spec.arg.s, operand) != 0)
            return -1;
        operand->producer = op;
        op->outputs.push_back(operand);
        return 0;
    }

    int add_constant(const Pt2Argument& arg, Operand*& operand)
    {
        Parameter value;
        std::string message;
        if (parameter_from_argument(arg, value, message) != 0)
            return fail(message);
        const std::string name = generated_name();
        operator_names.insert(name);
        Operator* op = graph.new_operator("prim::Constant", name);
        op->params["value"] = value;
        operand = new_operand(name);
        if (!operand)
            return -1;
        operand->producer = op;
        op->outputs.push_back(operand);
        return 0;
    }

    int add_list(const Pt2Argument& arg, Operand*& operand)
    {
        const std::string name = generated_name();
        operator_names.insert(name);
        std::vector<Operand*> items;
        const size_t count = arg.type == Pt2Argument::OptionalTensors ? arg.as.size() : arg.type == Pt2Argument::Bools ? arg.ab.size() : arg.args.size();
        for (size_t i = 0; i < count; i++)
        {
            Operand* item = 0;
            if (arg.type == Pt2Argument::OptionalTensors && !arg.as[i].empty())
                item = graph.get_operand(arg.as[i]);
            else if (arg.type == Pt2Argument::SymInts)
            {
                if (materialize_argument(arg.args[i], item) != 0)
                    return -1;
            }
            else
            {
                Pt2Argument value;
                value.type = arg.type == Pt2Argument::Bools ? Pt2Argument::Bool : Pt2Argument::None;
                if (arg.type == Pt2Argument::Bools)
                    value.b = arg.ab[i];
                if (add_constant(value, item) != 0)
                    return -1;
            }
            if (!item)
                return fail("list references an unknown tensor");
            items.push_back(item);
        }
        Operator* op = graph.new_operator("prim::ListConstruct", name);
        op->inputs = items;
        for (size_t i = 0; i < items.size(); i++)
            items[i]->consumers.push_back(op);
        operand = new_operand(name);
        if (!operand)
            return -1;
        operand->producer = op;
        op->outputs.push_back(operand);
        return 0;
    }

    int materialize_argument(const Pt2Argument& arg, Operand*& operand)
    {
        if (arg.type == Pt2Argument::Tensor)
        {
            operand = graph.get_operand(arg.s);
            return operand ? 0 : fail("unknown tensor value " + arg.s);
        }
        if (arg.type == Pt2Argument::OptionalTensor && arg.b)
        {
            operand = graph.get_operand(arg.s);
            return operand ? 0 : fail("unknown tensor value " + arg.s);
        }
        if (arg.type == Pt2Argument::SymInt && arg.b)
        {
            operand = graph.get_operand(arg.s);
            return operand ? 0 : fail("unknown symbolic integer " + arg.s);
        }
        if (arg.type == Pt2Argument::Tensors)
        {
            const std::string name = generated_name();
            operator_names.insert(name);
            Operator* op = graph.new_operator("prim::ListConstruct", name);
            for (size_t i = 0; i < arg.as.size(); i++)
            {
                Operand* item = graph.get_operand(arg.as[i]);
                if (!item)
                    return fail("list references an unknown tensor " + arg.as[i]);
                item->consumers.push_back(op);
                op->inputs.push_back(item);
            }
            operand = new_operand(name);
            if (!operand)
                return -1;
            operand->producer = op;
            op->outputs.push_back(operand);
            return 0;
        }
        if (arg.type == Pt2Argument::OptionalTensors || arg.type == Pt2Argument::Bools || arg.type == Pt2Argument::SymInts)
            return add_list(arg, operand);
        if (arg.type == Pt2Argument::OptionalTensor)
        {
            Pt2Argument none;
            return add_constant(none, operand);
        }
        return add_constant(arg, operand);
    }

    int parse_target(const std::string& target, std::string& name, std::string& overload)
    {
        const std::string operator_prefix = "_operator.";
        if (target.compare(0, operator_prefix.size(), operator_prefix) == 0)
        {
            name = target == "_operator.floordiv" ? "aten::floor_divide" : "aten::" + target.substr(operator_prefix.size());
            overload = "int";
            return 0;
        }

        const std::string prefix = "torch.ops.";
        if (target.compare(0, prefix.size(), prefix) != 0)
            return -1;
        const size_t namespace_end = target.find('.', prefix.size());
        const size_t operator_end = namespace_end == std::string::npos ? std::string::npos : target.find('.', namespace_end + 1);
        if (namespace_end == std::string::npos || operator_end == std::string::npos || operator_end + 1 == target.size())
            return -1;
        name = target.substr(prefix.size(), namespace_end - prefix.size()) + "::" + target.substr(namespace_end + 1, operator_end - namespace_end - 1);
        overload = target.substr(operator_end + 1);
        if (overload == "default")
            overload.clear();
        return 0;
    }

    int lower_integer_operator(size_t index, const Pt2Node& node, const std::string& name)
    {
        if (name != "aten::add" && name != "aten::sub" && name != "aten::mul" && name != "aten::floor_divide")
            return fail_node(index, node, "unsupported symbolic integer operator");
        if (node.inputs.size() != 2 || node.outputs.size() != 1 || node.outputs[0].type != Pt2Argument::SymInt || !node.outputs[0].b)
            return fail_node(index, node, "invalid symbolic integer operator");

        const std::string operator_name = !node.name.empty() ? node.name : node.outputs[0].s;
        if (add_operator_name(operator_name) != 0)
            return fail_node(index, node, error);
        std::vector<Operand*> inputs;
        for (size_t i = 0; i < node.inputs.size(); i++)
        {
            Operand* operand = 0;
            if (materialize_argument(node.inputs[i].arg, operand) != 0)
                return fail_node(index, node, error + " for argument " + node.inputs[i].name);
            inputs.push_back(operand);
        }
        Operator* op = graph.new_operator(name, operator_name);
        op->inputs = inputs;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            inputs[i]->consumers.push_back(op);
            op->inputnames.push_back(node.inputs[i].name);
        }
        Operand* output = new_operand(node.outputs[0].s);
        if (!output)
            return fail_node(index, node, error);
        output->producer = op;
        op->outputs.push_back(output);
        return 0;
    }

    int lower_assert_tensor_metadata(size_t index, const Pt2Node& node)
    {
        if (!node.outputs.empty())
            return fail_node(index, node, "invalid tensor metadata assertion output");

        const Pt2Tensor* tensor = 0;
        std::set<std::string> names;
        for (size_t i = 0; i < node.inputs.size(); i++)
        {
            const Pt2NamedArgument& input = node.inputs[i];
            if (!names.insert(input.name).second)
                return fail_node(index, node, "duplicate argument " + input.name);
            if (input.name == "a")
            {
                if (tensor || input.arg.type != Pt2Argument::Tensor)
                    return fail_node(index, node, "invalid tensor metadata assertion input");
                std::map<std::string, Pt2Tensor>::const_iterator it = program.tensors.find(input.arg.s);
                if (it == program.tensors.end())
                    return fail_node(index, node, "unknown tensor metadata assertion input");
                tensor = &it->second;
            }
        }
        if (!tensor)
            return fail_node(index, node, "tensor metadata assertion is missing input a");

        for (size_t i = 0; i < node.inputs.size(); i++)
        {
            const Pt2NamedArgument& input = node.inputs[i];
            if (input.name == "a")
                continue;
            if ((input.name == "size" || input.name == "stride" || input.name == "dtype" || input.name == "device" || input.name == "layout") && input.arg.type == Pt2Argument::None)
                continue;
            if (input.name == "dtype")
            {
                if (input.arg.type != Pt2Argument::ScalarType || input.arg.i != tensor->dtype)
                    return fail_node(index, node, "tensor metadata dtype assertion mismatch");
            }
            else if (input.name == "device")
            {
                if (input.arg.type != Pt2Argument::Device || input.arg.s != tensor->device || input.arg.i != tensor->device_index)
                    return fail_node(index, node, "tensor metadata device assertion mismatch");
            }
            else if (input.name == "layout")
            {
                if (input.arg.type != Pt2Argument::Layout || input.arg.i != tensor->layout)
                    return fail_node(index, node, "tensor metadata layout assertion mismatch");
            }
            else
            {
                return fail_node(index, node, "unsupported tensor metadata assertion argument " + input.name);
            }
        }
        return 0;
    }

    int lower_node(size_t index, const Pt2Node& node)
    {
        bool has_output = false;
        for (size_t i = 0; i < node.outputs.size(); i++)
            has_output |= node.outputs[i].type == Pt2Argument::Tensor || node.outputs[i].type == Pt2Argument::Tensors || node.outputs[i].type == Pt2Argument::SymInt;
        if (!has_output)
        {
            if (node.target == "torch.ops.aten._assert_tensor_metadata.default")
                return lower_assert_tensor_metadata(index, node);
            return fail_node(index, node, "operator without a tensor or symbolic integer output is unsupported");
        }

        std::string name;
        std::string overload;
        if (parse_target(node.target, name, overload) != 0)
            return fail_node(index, node, "invalid operator target");
        if (node.target.compare(0, 10, "_operator.") == 0)
            return lower_integer_operator(index, node, name);

        const auto handle = c10::Dispatcher::singleton().findSchema(c10::OperatorName(name, overload));
        if (!handle)
            return fail_node(index, node, "dispatcher schema was not found");
        const c10::FunctionSchema& schema = handle->schema();
        std::ostringstream schema_stream;
        schema_stream << schema;

        std::map<std::string, const Pt2Argument*> inputs;
        for (size_t i = 0; i < node.inputs.size(); i++)
        {
            if (!inputs.insert(std::make_pair(node.inputs[i].name, &node.inputs[i].arg)).second)
                return fail_node(index, node, "duplicate argument " + node.inputs[i].name, schema_stream.str());
        }

        const std::string operator_name = !node.name.empty() ? node.name : !node.outputs.empty() && !node.outputs[0].s.empty() ? node.outputs[0].s : generated_name();
        if (add_operator_name(operator_name) != 0)
            return fail_node(index, node, error, schema_stream.str());
        std::vector<Operand*> operator_inputs;
        std::vector<std::string> input_names;
        const std::vector<c10::Argument>& schema_arguments = schema.arguments();
        for (size_t i = 0; i < schema_arguments.size(); i++)
        {
            const c10::Argument& schema_argument = schema_arguments[i];
            Pt2Argument default_argument;
            const Pt2Argument* argument = 0;
            std::map<std::string, const Pt2Argument*>::iterator it = inputs.find(schema_argument.name());
            if (it != inputs.end())
            {
                argument = it->second;
                inputs.erase(it);
            }
            else if (schema_argument.default_value())
            {
                std::string message;
                if (argument_from_ivalue(*schema_argument.default_value(), default_argument, message) != 0)
                    return fail_node(index, node, message + " for argument " + schema_argument.name(), schema_stream.str());
                argument = &default_argument;
            }
            else
                return fail_node(index, node, "missing required argument " + schema_argument.name(), schema_stream.str());

            Pt2Argument empty_stride;
            if (schema_argument.name() == "stride" && argument->type == Pt2Argument::Ints && argument->ai.empty()
                && (name == "aten::avg_pool1d" || name == "aten::avg_pool2d" || name == "aten::avg_pool3d"
                    || name == "aten::max_pool1d" || name == "aten::max_pool2d" || name == "aten::max_pool3d"
                    || name == "aten::max_pool1d_with_indices" || name == "aten::max_pool2d_with_indices" || name == "aten::max_pool3d_with_indices"))
                argument = &empty_stride;

            Operand* operand = 0;
            if (materialize_argument(*argument, operand) != 0)
                return fail_node(index, node, error + " for argument " + schema_argument.name(), schema_stream.str());
            operator_inputs.push_back(operand);
            input_names.push_back(schema_argument.name());
        }
        if (!inputs.empty())
            return fail_node(index, node, "unknown argument " + inputs.begin()->first, schema_stream.str());
        if (node.outputs.size() != schema.returns().size())
            return fail_node(index, node, "output count does not match dispatcher schema", schema_stream.str());
        for (size_t i = 0; i < node.outputs.size(); i++)
        {
            if (!output_matches_schema(node.outputs[i], schema.returns()[i].type()))
                return fail_node(index, node, "output type does not match dispatcher schema", schema_stream.str());
        }

        Operator* op = graph.new_operator(name, operator_name);
        op->inputs = operator_inputs;
        op->inputnames = input_names;
        for (size_t i = 0; i < operator_inputs.size(); i++)
            operator_inputs[i]->consumers.push_back(op);
        for (size_t i = 0; i < node.outputs.size(); i++)
        {
            if (node.outputs[i].type == Pt2Argument::Tensor)
            {
                Operand* operand = new_operand(node.outputs[i].s);
                if (!operand || set_tensor_meta(node.outputs[i].s, operand) != 0)
                    return fail_node(index, node, error, schema_stream.str());
                operand->producer = op;
                op->outputs.push_back(operand);
            }
            else if (node.outputs[i].type == Pt2Argument::Tensors)
            {
                const std::string list_name = generated_name();
                Operand* list = new_operand(list_name);
                if (!list)
                    return fail_node(index, node, error, schema_stream.str());
                list->producer = op;
                op->outputs.push_back(list);

                const std::string unpack_name = generated_name();
                if (add_operator_name(unpack_name) != 0)
                    return fail_node(index, node, error, schema_stream.str());
                Operator* unpack = graph.new_operator("prim::ListUnpack", unpack_name);
                unpack->inputs.push_back(list);
                list->consumers.push_back(unpack);
                for (size_t j = 0; j < node.outputs[i].as.size(); j++)
                {
                    Operand* operand = new_operand(node.outputs[i].as[j]);
                    if (!operand || set_tensor_meta(node.outputs[i].as[j], operand) != 0)
                        return fail_node(index, node, error, schema_stream.str());
                    operand->producer = unpack;
                    unpack->outputs.push_back(operand);
                }
            }
            else
            {
                Operand* operand = new_operand(node.outputs[i].s);
                if (!operand)
                    return fail_node(index, node, error, schema_stream.str());
                operand->producer = op;
                op->outputs.push_back(operand);
            }
        }
        if (name == "aten::_weight_norm" && op->inputs.size() == 3
            && op->inputs[0]->producer->type == "pnnx.Attribute" && op->inputs[1]->producer->type == "pnnx.Attribute"
            && op->inputs[2]->producer->type == "prim::Constant" && op->inputs[2]->producer->params.at("value").i == 0)
        {
            Attribute weight = op->inputs[0]->producer->attrs.at("data");
            std::vector<float> weight_data = weight.get_float32_data();
            const std::vector<float> weight_g = op->inputs[1]->producer->attrs.at("data").get_float32_data();
            if (!weight.shape.empty() && weight.shape[0] > 0 && weight_g.size() >= (size_t)weight.shape[0]
                && weight_data.size() % weight.shape[0] == 0)
            {
                const int dim0 = weight.shape[0];
                apply_weight_norm(weight_data, weight_g, dim0, weight_data.size() / dim0);
                weight.set_float32_data(weight_data);
                for (size_t i = 0; i < op->inputs.size(); i++)
                    op->inputs[i]->remove_consumer(op);
                op->inputs.clear();
                op->inputnames.clear();
                op->type = "pnnx.Attribute";
                op->attrs["data"] = weight;
            }
        }
        if (name == "aten::alias" || name == "aten::lift_fresh_copy")
        {
            op->type = "torch.clone";
            op->inputnames[0] = "input";
        }
        if (name == "aten::new_full")
        {
            op->inputs[0]->remove_consumer(op);
            op->inputs.erase(op->inputs.begin());
            op->inputnames.erase(op->inputnames.begin());
            op->type = "aten::full";
        }
        if (name == "aten::hann_window" || name == "aten::hamming_window")
            op->type = name == "aten::hann_window" ? "torch.hann_window" : "torch.hamming_window";
        if (name == "aten::_upsample_nearest_exact2d" || name == "aten::_upsample_nearest_exact3d")
        {
            op->inputnames[1] = "size";
            op->inputnames[2] = "scale_factor";
        }
        if (name == "aten::gru" || name == "aten::lstm" || name == "aten::rnn_tanh" || name == "aten::rnn_relu")
            op->type = "torch._VF." + name.substr(6);
        if (name == "aten::tril")
        {
            op->type = "torch.tril";
            op->inputnames[0] = "input";
        }
        if (name == "aten::sym_size")
            op->type = "aten::size";
        return 0;
    }

    int lower_output(const Pt2OutputSpec& spec)
    {
        if (spec.arg.type != Pt2Argument::Tensor)
            return fail("only tensor user outputs are supported");
        Operand* operand = graph.get_operand(spec.arg.s);
        if (!operand)
            return fail("unknown graph output " + spec.arg.s);
        std::string name = "pnnx_output_" + std::to_string(output_index++);
        if (program_operator_names.find(name) != program_operator_names.end() || operator_names.find(name) != operator_names.end())
            name = generated_name();
        if (add_operator_name(name) != 0)
            return -1;
        Operator* op = graph.new_operator("pnnx.Output", name);
        operand->consumers.push_back(op);
        op->inputs.push_back(operand);
        return 0;
    }

    int verify_attribute(const Operator* op)
    {
        if (op->type != "pnnx.Attribute")
            return 0;
        std::map<std::string, Attribute>::const_iterator it = op->attrs.find("data");
        if (it == op->attrs.end() || it->second.elemsize() == 0)
            return fail("invalid attribute " + op->name);
        size_t count = 1;
        for (size_t i = 0; i < it->second.shape.size(); i++)
        {
            if (it->second.shape[i] < 0 || (it->second.shape[i] != 0 && count > SIZE_MAX / (size_t)it->second.shape[i]))
                return fail("invalid attribute shape " + op->name);
            count *= (size_t)it->second.shape[i];
        }
        if (count > SIZE_MAX / it->second.elemsize() || count * it->second.elemsize() != it->second.data.size())
            return fail("attribute byte size mismatch " + op->name);
        return 0;
    }

    int verify()
    {
        std::set<const Operator*> operators(graph.ops.begin(), graph.ops.end());
        std::set<const Operand*> operands(graph.operands.begin(), graph.operands.end());
        if (operators.size() != graph.ops.size() || operands.size() != graph.operands.size())
            return fail("graph contains duplicate object pointers");
        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            const Operator* op = graph.ops[i];
            if (verify_attribute(op) != 0)
                return -1;
            if (!op->inputnames.empty() && op->inputnames.size() != op->inputs.size())
                return fail("operator input name count mismatch " + op->name);
            for (size_t j = 0; j < op->inputs.size(); j++)
            {
                const Operand* operand = op->inputs[j];
                if (operands.find(operand) == operands.end() || std::count(operand->consumers.begin(), operand->consumers.end(), op) != std::count(op->inputs.begin(), op->inputs.end(), operand))
                    return fail("broken input edge at operator " + op->name);
            }
            for (size_t j = 0; j < op->outputs.size(); j++)
            {
                const Operand* operand = op->outputs[j];
                if (operands.find(operand) == operands.end() || operand->producer != op)
                    return fail("broken output edge at operator " + op->name);
            }
        }
        for (size_t i = 0; i < graph.operands.size(); i++)
        {
            const Operand* operand = graph.operands[i];
            if (!operand->producer || operators.find(operand->producer) == operators.end() || std::count(operand->producer->outputs.begin(), operand->producer->outputs.end(), operand) != 1)
                return fail("operand has no valid producer " + operand->name);
            for (size_t j = 0; j < operand->consumers.size(); j++)
            {
                if (operators.find(operand->consumers[j]) == operators.end())
                    return fail("operand has an invalid consumer " + operand->name);
            }
        }
        return 0;
    }

    const Pt2Program& program;
    Pt2Weights& weights;
    Graph& graph;
    std::string& error;
    size_t generated_index;
    size_t input_index;
    size_t output_index;
    std::set<std::string> operator_names;
    std::set<std::string> program_operator_names;
    std::set<std::string> operand_names;
    std::set<std::string> reserved_names;
};

int lower_pt2_graph(const Pt2Program& program, Pt2Weights& weights, Graph& graph, std::string& error)
{
    error.clear();
    try
    {
        Pt2GraphLowering lowering(program, weights, graph, error);
        return lowering.lower();
    }
    catch (const c10::Error& e)
    {
        if (error.empty())
            error = e.msg();
        return -1;
    }
}

} // namespace pnnx

#else

namespace pnnx {

int lower_pt2_graph(const Pt2Program&, Pt2Weights&, Graph&, std::string& error)
{
    error = "PT2 graph lowering requires PyTorch 2.6 or newer";
    return -1;
}

} // namespace pnnx

#endif
