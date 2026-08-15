// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pt2_graph_lowering.h"

#include <limits.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#include <torch/csrc/api/include/torch/version.h>

#include "ir.h"
#include "pt2_program.h"
#include "pt2_weights.h"

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

static int parameter_from_argument(const Pt2Argument& arg, Parameter& value, std::string& error)
{
    if (arg.type == Pt2Argument::None)
        value = Parameter();
    else if (arg.type == Pt2Argument::Bool)
        value = Parameter(arg.b);
    else if (arg.type == Pt2Argument::Int)
        value = Parameter((long long)arg.i);
    else if (arg.type == Pt2Argument::Ints)
        value = Parameter(arg.ai);
    else if (arg.type == Pt2Argument::Float)
        value = Parameter(arg.f);
    else if (arg.type == Pt2Argument::Floats)
        value = Parameter(arg.af);
    else if (arg.type == Pt2Argument::String || arg.type == Pt2Argument::Device)
        value = Parameter(arg.s);
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
        value = Parameter((long long)arg.i);
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
        arg.s = value.toDevice().str();
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

class Pt2GraphLowering
{
public:
    Pt2GraphLowering(const Pt2Program& _program, Pt2Weights& _weights, Graph& _graph, std::string& _error)
        : program(_program), weights(_weights), graph(_graph), error(_error)
    {
        generated_index = 0;
        input_index = 0;
        output_index = 0;
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
            if (operand_names.find(name) == operand_names.end() && operator_names.find(name) == operator_names.end())
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
            if (size.value < 0 || size.value > INT_MAX)
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
            const std::string name = "pnnx_input_" + std::to_string(input_index++);
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
        if (add_operator_name(spec.target) != 0)
            return -1;
        Operator* op = graph.new_operator("pnnx.Attribute", spec.target);
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
        const size_t count = arg.type == Pt2Argument::OptionalTensors ? arg.as.size() : arg.type == Pt2Argument::Bools ? arg.ab.size() : 0;
        for (size_t i = 0; i < count; i++)
        {
            Operand* item = 0;
            if (arg.type == Pt2Argument::OptionalTensors && !arg.as[i].empty())
                item = graph.get_operand(arg.as[i]);
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
        if (arg.type == Pt2Argument::OptionalTensors || arg.type == Pt2Argument::Bools)
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

    int lower_node(size_t index, const Pt2Node& node)
    {
        std::string name;
        std::string overload;
        if (parse_target(node.target, name, overload) != 0)
            return fail_node(index, node, "invalid operator target");

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

        const std::string operator_name = !node.name.empty() ? node.name : !node.outputs.empty() ? node.outputs[0].s : generated_name();
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

        Operator* op = graph.new_operator(name, operator_name);
        op->inputs = operator_inputs;
        op->inputnames = input_names;
        for (size_t i = 0; i < operator_inputs.size(); i++)
            operator_inputs[i]->consumers.push_back(op);
        for (size_t i = 0; i < node.outputs.size(); i++)
        {
            if (node.outputs[i].type != Pt2Argument::Tensor)
                return fail_node(index, node, "only tensor outputs are supported", schema_stream.str());
            Operand* operand = new_operand(node.outputs[i].s);
            if (!operand || set_tensor_meta(node.outputs[i].s, operand) != 0)
                return fail_node(index, node, error, schema_stream.str());
            operand->producer = op;
            op->outputs.push_back(operand);
        }
        return 0;
    }

    int lower_output(const Pt2OutputSpec& spec)
    {
        if (spec.arg.type != Pt2Argument::Tensor)
            return fail("only tensor user outputs are supported");
        Operand* operand = graph.get_operand(spec.arg.s);
        if (!operand)
            return fail("unknown graph output " + spec.arg.s);
        const std::string name = "pnnx_output_" + std::to_string(output_index++);
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
    std::set<std::string> operand_names;
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
