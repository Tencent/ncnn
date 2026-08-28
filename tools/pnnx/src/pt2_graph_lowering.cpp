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

static int scalar_type_to_c10(int type)
{
    static const int types[] = {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15};
    return type >= 1 && type <= 13 ? types[type] : -1;
}

static bool integer_can_be_parameter(int64_t value)
{
    return (value >= INT_MIN && value <= INT_MAX) || value == INT64_MIN || value == INT64_MIN + 1 || value == INT64_MAX - 1 || value == INT64_MAX;
}

static int sym_kind(const Pt2Argument& arg)
{
    if (arg.type == Pt2Argument::Bool || arg.type == Pt2Argument::SymBool)
        return 1;
    if (arg.type == Pt2Argument::Int || arg.type == Pt2Argument::SymInt)
        return 2;
    if (arg.type == Pt2Argument::Float || arg.type == Pt2Argument::SymFloat)
        return 3;
    return 0;
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
    else if (arg.type == Pt2Argument::SymInt && arg.s.empty())
    {
        if (!integer_can_be_parameter(arg.i))
        {
            error = "symbolic integer argument is outside the pnnx parameter range";
            return -1;
        }
        value = Parameter((long long)arg.i);
    }
    else if (arg.type == Pt2Argument::SymFloat && arg.s.empty())
        value = std::isinf(arg.f) ? Parameter(arg.f < 0 ? "-inf" : "inf") : Parameter(arg.f);
    else if (arg.type == Pt2Argument::SymBool && arg.s.empty())
        value = Parameter(arg.b);
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

static bool match_argument(const Pt2Argument& arg, const c10::TypePtr& type)
{
    if (type->kind() == c10::TypeKind::OptionalType)
    {
        if (arg.type == Pt2Argument::None || (arg.type == Pt2Argument::OptionalTensor && !arg.has_tensor))
            return true;
        return match_argument(arg, type->cast<c10::OptionalType>()->getElementType());
    }

    if (type->kind() == c10::TypeKind::ListType)
    {
        const c10::TypePtr element = type->cast<c10::ListType>()->getElementType();
        const c10::TypeKind kind = element->kind();
        if (arg.type == Pt2Argument::Tensors)
            return kind == c10::TypeKind::TensorType || (kind == c10::TypeKind::OptionalType && element->cast<c10::OptionalType>()->getElementType()->kind() == c10::TypeKind::TensorType);
        if (arg.type == Pt2Argument::OptionalTensors)
            return kind == c10::TypeKind::OptionalType && element->cast<c10::OptionalType>()->getElementType()->kind() == c10::TypeKind::TensorType;
        if (arg.type == Pt2Argument::Ints)
            return kind == c10::TypeKind::IntType || kind == c10::TypeKind::SymIntType;
        if (arg.type == Pt2Argument::Floats)
            return kind == c10::TypeKind::FloatType;
        if (arg.type == Pt2Argument::Strings)
            return kind == c10::TypeKind::StringType;
        if (arg.type == Pt2Argument::Bools)
            return kind == c10::TypeKind::BoolType;
        if (arg.type == Pt2Argument::SymInts)
            return kind == c10::TypeKind::IntType || kind == c10::TypeKind::SymIntType || kind == c10::TypeKind::NumberType;
        if (arg.type == Pt2Argument::SymFloats)
            return kind == c10::TypeKind::FloatType || kind == c10::TypeKind::SymFloatType || kind == c10::TypeKind::NumberType;
        if (arg.type == Pt2Argument::SymBools)
            return kind == c10::TypeKind::BoolType || kind == c10::TypeKind::SymBoolType;
        return false;
    }

    const c10::TypeKind kind = type->kind();
    if (arg.type == Pt2Argument::None)
        return kind == c10::TypeKind::NoneType;
    if (arg.type == Pt2Argument::Tensor || (arg.type == Pt2Argument::OptionalTensor && arg.has_tensor))
        return kind == c10::TypeKind::TensorType;
    if (arg.type == Pt2Argument::Int)
        return kind == c10::TypeKind::IntType || kind == c10::TypeKind::NumberType;
    if (arg.type == Pt2Argument::Float)
        return kind == c10::TypeKind::FloatType || kind == c10::TypeKind::NumberType;
    if (arg.type == Pt2Argument::Complex)
        return kind == c10::TypeKind::ComplexType || kind == c10::TypeKind::NumberType;
    if (arg.type == Pt2Argument::String)
        return kind == c10::TypeKind::StringType;
    if (arg.type == Pt2Argument::Bool)
        return kind == c10::TypeKind::BoolType;
    if (arg.type == Pt2Argument::ScalarType || arg.type == Pt2Argument::MemoryFormat || arg.type == Pt2Argument::Layout)
        return kind == c10::TypeKind::IntType;
    if (arg.type == Pt2Argument::Device)
        return kind == c10::TypeKind::DeviceObjType;
    if (arg.type == Pt2Argument::SymInt)
        return kind == c10::TypeKind::SymIntType || kind == c10::TypeKind::IntType || kind == c10::TypeKind::NumberType;
    if (arg.type == Pt2Argument::SymBool)
        return kind == c10::TypeKind::SymBoolType || kind == c10::TypeKind::BoolType || kind == c10::TypeKind::NumberType;
    if (arg.type == Pt2Argument::SymFloat)
        return kind == c10::TypeKind::SymFloatType || kind == c10::TypeKind::FloatType || kind == c10::TypeKind::NumberType;
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
        for (std::map<std::string, std::string>::const_iterator it = program.sym_floats.begin(); it != program.sym_floats.end(); ++it)
            reserved_names.insert(it->first);
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
        return 0;
    }

private:
    int fail(const std::string& message)
    {
        error = message;
        return -1;
    }

    int fail_node(size_t index, const Pt2Node& node, const std::string& message, const c10::FunctionSchema* schema = 0)
    {
        const std::string name = !node.name.empty() ? node.name : !node.outputs.empty() ? node.outputs[0].s : std::string("unnamed");
        error = "node " + std::to_string(index) + " (" + name + ", target " + node.target + "): " + message;
        if (schema)
        {
            std::ostringstream ss;
            ss << *schema;
            error += "; schema " + ss.str();
        }
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

        std::map<std::string, Attribute>::iterator it = weights.values.find(spec.target);
        if (it == weights.values.end())
            return fail("missing weight " + spec.target);
        std::string name = spec.target;
        if (program_operator_names.find(name) != program_operator_names.end() || operator_names.find(name) != operator_names.end())
            name = generated_name();
        if (add_operator_name(name) != 0)
            return -1;
        Operator* op = graph.new_operator("pnnx.Attribute", name);
        op->attrs["data"] = std::move(it->second);
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
        const size_t count = arg.type == Pt2Argument::Tensors || arg.type == Pt2Argument::OptionalTensors ? arg.as.size() : arg.type == Pt2Argument::Bools ? arg.ab.size() : arg.args.size();
        for (size_t i = 0; i < count; i++)
        {
            Operand* item = 0;
            if (arg.type == Pt2Argument::Tensors || (arg.type == Pt2Argument::OptionalTensors && !arg.as[i].empty()))
                item = graph.get_operand(arg.as[i]);
            else if (arg.type == Pt2Argument::SymInts || arg.type == Pt2Argument::SymBools || arg.type == Pt2Argument::SymFloats)
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
        if (arg.type == Pt2Argument::OptionalTensor && arg.has_tensor)
        {
            operand = graph.get_operand(arg.s);
            return operand ? 0 : fail("unknown tensor value " + arg.s);
        }
        if ((arg.type == Pt2Argument::SymInt || arg.type == Pt2Argument::SymBool || arg.type == Pt2Argument::SymFloat) && !arg.s.empty())
        {
            operand = graph.get_operand(arg.s);
            return operand ? 0 : fail("unknown symbolic value " + arg.s);
        }
        if (arg.type == Pt2Argument::Tensors || arg.type == Pt2Argument::OptionalTensors || arg.type == Pt2Argument::Bools ||
            arg.type == Pt2Argument::SymInts || arg.type == Pt2Argument::SymBools || arg.type == Pt2Argument::SymFloats)
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
        if (target.compare(0, 10, "torch.ops.") != 0)
            return -1;
        const size_t namespace_end = target.find('.', 10);
        const size_t operator_end = namespace_end == std::string::npos ? std::string::npos : target.find('.', namespace_end + 1);
        if (namespace_end == std::string::npos || operator_end == std::string::npos || operator_end + 1 == target.size())
            return -1;
        name = target.substr(10, namespace_end - 10) + "::" + target.substr(namespace_end + 1, operator_end - namespace_end - 1);
        overload = target.substr(operator_end + 1);
        if (overload == "default")
            overload.clear();
        return 0;
    }

    int lower_symbolic_operator(size_t index, const Pt2Node& node)
    {
        const char* expr = 0;
        const char* aten = 0;
        int input_count = 2;
        bool compare = false;

        if (node.target == "_operator.add") { expr = "add"; aten = "aten::add"; }
        else if (node.target == "_operator.sub") { expr = "sub"; aten = "aten::sub"; }
        else if (node.target == "_operator.mul") { expr = "mul"; aten = "aten::mul"; }
        else if (node.target == "_operator.truediv") { expr = "div"; aten = "aten::div"; }
        else if (node.target == "_operator.floordiv") { expr = "floor_divide"; aten = "aten::floor_divide"; }
        else if (node.target == "_operator.mod") expr = "remainder";
        else if (node.target == "_operator.pow") { expr = "pow"; aten = "aten::pow"; }
        else if (node.target == "_operator.and_") { expr = "and"; aten = "aten::__and__"; }
        else if (node.target == "_operator.or_") { expr = "or"; aten = "aten::__or__"; }
        else if (node.target == "_operator.lshift") { expr = "lshift"; aten = "aten::__lshift__"; }
        else if (node.target == "_operator.rshift") { expr = "rshift"; aten = "aten::__rshift__"; }
        else if (node.target == "_operator.eq") { expr = "eq"; compare = true; }
        else if (node.target == "_operator.ne") { expr = "ne"; compare = true; }
        else if (node.target == "_operator.lt") { expr = "lt"; compare = true; }
        else if (node.target == "_operator.le") { expr = "le"; compare = true; }
        else if (node.target == "_operator.gt") { expr = "gt"; compare = true; }
        else if (node.target == "_operator.ge") { expr = "ge"; compare = true; }
        else if (node.target == "_operator.neg") { expr = "neg"; aten = "aten::neg"; input_count = 1; }
        else if (node.target == "_operator.pos") { expr = "pos"; input_count = 1; }
        else if (node.target == "math.trunc") { expr = "sym_trunc"; input_count = 1; }
        else if (node.target == "torch.sym_not") { expr = "sym_not"; input_count = 1; }
        else if (node.target == "torch.sym_int") { expr = "sym_int"; input_count = 1; }
        else if (node.target == "torch.sym_float") { expr = "sym_float"; input_count = 1; }
        else if (node.target == "torch.sym_ite") { expr = "sym_ite"; input_count = 3; }
        else if (node.target == "torch.sym_max") expr = "sym_max";
        else if (node.target == "torch.sym_min") expr = "sym_min";
        else if (node.target == "torch._sym_sqrt" || node.target == "torch.sym_sqrt") { expr = "sym_sqrt"; input_count = 1; }
        else
            return fail_node(index, node, "unsupported symbolic operator");

        if (node.inputs.size() != (size_t)input_count || node.outputs.size() != 1)
            return fail_node(index, node, "invalid symbolic operator");
        int kinds[3];
        for (size_t i = 0; i < node.inputs.size(); i++)
        {
            const Pt2Argument& arg = node.inputs[i].arg;
            kinds[i] = sym_kind(arg);
            if (!kinds[i])
                return fail_node(index, node, "invalid symbolic operator input");
        }

        int result_kind = 0;
        if (node.target == "torch.sym_not")
            result_kind = kinds[0] == 1 ? 1 : 0;
        else if (node.target == "math.trunc" || node.target == "torch.sym_int")
            result_kind = kinds[0] >= 2 ? 2 : 0;
        else if (node.target == "torch.sym_float" || node.target == "torch._sym_sqrt" || node.target == "torch.sym_sqrt")
            result_kind = kinds[0] >= 2 ? 3 : 0;
        else if (node.target == "torch.sym_ite")
            result_kind = kinds[0] == 1 && (kinds[1] == kinds[2] || (kinds[1] >= 2 && kinds[2] >= 2)) ? std::max(kinds[1], kinds[2]) : 0;
        else if (node.target == "_operator.and_" || node.target == "_operator.or_")
            result_kind = kinds[0] == kinds[1] && kinds[0] <= 2 ? kinds[0] : 0;
        else if (node.target == "_operator.lshift" || node.target == "_operator.rshift")
            result_kind = kinds[0] == 2 && kinds[1] == 2 ? 2 : 0;
        else if (compare)
            result_kind = (((node.target == "_operator.eq" || node.target == "_operator.ne") && kinds[0] == kinds[1]) ||
                           (kinds[0] >= 2 && kinds[1] >= 2)) ? 1 : 0;
        else if (input_count == 1)
            result_kind = kinds[0] >= 2 ? kinds[0] : 0;
        else
            result_kind = kinds[0] >= 2 && kinds[1] >= 2 ? (node.target == "_operator.truediv" ? 3 : std::max(kinds[0], kinds[1])) : 0;

        const Pt2Argument& outarg = node.outputs[0];
        const Pt2Argument::Type expected_type = result_kind == 1 ? Pt2Argument::SymBool : result_kind == 2 ? Pt2Argument::SymInt : Pt2Argument::SymFloat;
        if (!result_kind || outarg.s.empty() || outarg.type != expected_type)
            return fail_node(index, node, "invalid symbolic operator output");

        const std::string operator_name = !node.name.empty() ? node.name : node.outputs[0].s;
        if (add_operator_name(operator_name) != 0)
            return fail_node(index, node, error);
        std::vector<Operand*> inputs;
        std::string expression;
        if (aten)
        {
            for (size_t i = 0; i < node.inputs.size(); i++)
            {
                Operand* operand = 0;
                if (materialize_argument(node.inputs[i].arg, operand) != 0)
                    return fail_node(index, node, error + " for argument " + node.inputs[i].name);
                inputs.push_back(operand);
            }
        }
        else
        {
            std::string args[3];
            for (size_t i = 0; i < node.inputs.size(); i++)
            {
                const Pt2Argument& arg = node.inputs[i].arg;
                if (!arg.s.empty())
                {
                    Operand* operand = 0;
                    if (materialize_argument(arg, operand) != 0)
                        return fail_node(index, node, error + " for argument " + node.inputs[i].name);
                    args[i] = "@" + std::to_string(inputs.size());
                    inputs.push_back(operand);
                }
                else
                {
                    Parameter value;
                    std::string message;
                    if (parameter_from_argument(arg, value, message) != 0)
                        return fail_node(index, node, message + " for argument " + node.inputs[i].name);
                    args[i] = Parameter::encode_to_string(value);
                }
            }
            if (node.target == "_operator.mod")
                expression = "sub(" + args[0] + ",mul(floor_divide(" + args[0] + "," + args[1] + ")," + args[1] + "))";
            else
            {
                expression = std::string(expr) + "(";
                for (size_t i = 0; i < node.inputs.size(); i++)
                    expression += args[i] + (i + 1 == node.inputs.size() ? ")" : ",");
            }
        }
        Operator* op = graph.new_operator(aten ? aten : "pnnx.Expression", operator_name);
        if (aten)
        {
            for (size_t i = 0; i < node.inputs.size(); i++)
                op->inputnames.push_back(node.inputs[i].name);
        }
        else
            op->params["expr"] = expression;
        op->inputs = inputs;
        for (size_t i = 0; i < inputs.size(); i++)
            inputs[i]->consumers.push_back(op);
        Operand* output = new_operand(node.outputs[0].s);
        if (!output)
            return fail_node(index, node, error);
        output->producer = op;
        op->outputs.push_back(output);
        return 0;
    }

    int lower_assert_scalar(size_t index, const Pt2Node& node)
    {
        if (!node.outputs.empty() || node.inputs.size() != 2 ||
            node.inputs[0].name != "self" || (node.inputs[0].arg.type != Pt2Argument::SymBool && node.inputs[0].arg.type != Pt2Argument::Bool) ||
            node.inputs[1].name != "assert_msg" || node.inputs[1].arg.type != Pt2Argument::String)
            return fail_node(index, node, "invalid scalar assertion");
        Operand* condition = 0;
        if (materialize_argument(node.inputs[0].arg, condition) != 0)
            return fail_node(index, node, error);
        const std::string name = !node.name.empty() ? node.name : generated_name();
        if (add_operator_name(name) != 0)
            return fail_node(index, node, error);
        Operator* op = graph.new_operator("pnnx.Assert", name);
        op->inputs.push_back(condition);
        condition->consumers.push_back(op);
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
            if (input.name == "size" || input.name == "stride")
            {
                const std::vector<Pt2SymInt>& values = input.name == "size" ? tensor->sizes : tensor->strides;
                const size_t count = input.arg.type == Pt2Argument::Ints ? input.arg.ai.size() : input.arg.type == Pt2Argument::SymInts ? input.arg.args.size() : 0;
                bool match = (input.arg.type == Pt2Argument::Ints || input.arg.type == Pt2Argument::SymInts) && count == values.size();
                for (size_t j = 0; match && j < count; j++)
                {
                    if (input.arg.type == Pt2Argument::Ints || input.arg.args[j].s.empty())
                    {
                        const int64_t value = input.arg.type == Pt2Argument::Ints ? input.arg.ai[j] : input.arg.args[j].i;
                        match = !values[j].symbolic && values[j].value == value;
                        continue;
                    }

                    std::map<std::string, Pt2SymInt>::const_iterator it = program.sym_ints.find(input.arg.args[j].s);
                    if (it == program.sym_ints.end())
                    {
                        match = false;
                        continue;
                    }
                    const Pt2SymInt& value = it->second;
                    if (values[j].symbolic || value.symbolic || !values[j].expression.empty() || !value.expression.empty())
                        match = !values[j].expression.empty() && values[j].expression == value.expression;
                    else
                        match = values[j].value == value.value;
                }
                if (!match)
                    return fail_node(index, node, "tensor metadata " + input.name + " assertion mismatch");
            }
            else if (input.name == "dtype")
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
            has_output |= node.outputs[i].type == Pt2Argument::Tensor || node.outputs[i].type == Pt2Argument::Tensors || node.outputs[i].type == Pt2Argument::SymInt || node.outputs[i].type == Pt2Argument::SymBool || node.outputs[i].type == Pt2Argument::SymFloat;
        if (!has_output)
        {
            if (node.target == "torch.ops.aten._assert_tensor_metadata.default")
                return lower_assert_tensor_metadata(index, node);
            if (node.target == "torch.ops.aten._assert_scalar.default")
                return lower_assert_scalar(index, node);
            return fail_node(index, node, "operator without a supported output is unsupported");
        }

        if (node.target.compare(0, 10, "torch.ops.") != 0)
            return lower_symbolic_operator(index, node);
        std::string name;
        std::string overload;
        if (parse_target(node.target, name, overload) != 0)
            return fail_node(index, node, "invalid operator target");

        const auto handle = c10::Dispatcher::singleton().findSchema(c10::OperatorName(name, overload));
        if (!handle)
            return fail_node(index, node, "dispatcher schema was not found");
        const c10::FunctionSchema& schema = handle->schema();

        std::map<std::string, const Pt2Argument*> args;
        for (size_t i = 0; i < node.inputs.size(); i++)
        {
            if (!args.insert(std::make_pair(node.inputs[i].name, &node.inputs[i].arg)).second)
                return fail_node(index, node, "duplicate argument " + node.inputs[i].name, &schema);
        }

        const std::string opname = !node.name.empty() ? node.name : !node.outputs.empty() && !node.outputs[0].s.empty() ? node.outputs[0].s : generated_name();
        if (add_operator_name(opname) != 0)
            return fail_node(index, node, error, &schema);
        std::vector<Operand*> op_inputs;
        std::vector<std::string> input_names;
        const std::vector<c10::Argument>& schema_args = schema.arguments();
        for (size_t i = 0; i < schema_args.size(); i++)
        {
            const c10::Argument& schema_arg = schema_args[i];
            Pt2Argument default_arg;
            const Pt2Argument* arg = 0;
            std::map<std::string, const Pt2Argument*>::iterator it = args.find(schema_arg.name());
            if (it != args.end())
            {
                arg = it->second;
                args.erase(it);
            }
            else if (schema_arg.default_value())
            {
                std::string message;
                if (argument_from_ivalue(*schema_arg.default_value(), default_arg, message) != 0)
                    return fail_node(index, node, message + " for argument " + schema_arg.name(), &schema);
                arg = &default_arg;
            }
            else
                return fail_node(index, node, "missing required argument " + schema_arg.name(), &schema);

            bool matches = match_argument(*arg, schema_arg.type());
            if (!matches && schema_arg.name() == "other" && schema_arg.type()->kind() == c10::TypeKind::TensorType &&
                (name == "aten::add" || name == "aten::add_" || name == "aten::sub" || name == "aten::sub_" ||
                 name == "aten::mul" || name == "aten::mul_" || name == "aten::div" || name == "aten::div_") &&
                (arg->type == Pt2Argument::Int || arg->type == Pt2Argument::Float || arg->type == Pt2Argument::Complex ||
                 arg->type == Pt2Argument::SymInt || arg->type == Pt2Argument::SymFloat))
                matches = true;
            if (!matches)
                return fail_node(index, node, "argument type does not match dispatcher schema for " + schema_arg.name(), &schema);

            Pt2Argument empty_stride;
            if (schema_arg.name() == "stride" && arg->type == Pt2Argument::Ints && arg->ai.empty()
                && (name == "aten::avg_pool1d" || name == "aten::avg_pool2d" || name == "aten::avg_pool3d"
                    || name == "aten::max_pool1d" || name == "aten::max_pool2d" || name == "aten::max_pool3d"
                    || name == "aten::max_pool1d_with_indices" || name == "aten::max_pool2d_with_indices" || name == "aten::max_pool3d_with_indices"))
                arg = &empty_stride;

            Operand* operand = 0;
            if (materialize_argument(*arg, operand) != 0)
                return fail_node(index, node, error + " for argument " + schema_arg.name(), &schema);
            op_inputs.push_back(operand);
            input_names.push_back(schema_arg.name());
        }
        if (!args.empty())
            return fail_node(index, node, "unknown argument " + args.begin()->first, &schema);
        if (node.outputs.size() != schema.returns().size())
            return fail_node(index, node, "output count does not match dispatcher schema", &schema);
        for (size_t i = 0; i < node.outputs.size(); i++)
        {
            const Pt2Argument& output = node.outputs[i];
            if ((output.type != Pt2Argument::Tensor && output.type != Pt2Argument::Tensors && output.type != Pt2Argument::SymInt && output.type != Pt2Argument::SymBool && output.type != Pt2Argument::SymFloat) ||
                !match_argument(output, schema.returns()[i].type()))
                return fail_node(index, node, "output type does not match dispatcher schema", &schema);
        }

        Operator* op = graph.new_operator(name, opname);
        op->inputs = op_inputs;
        op->inputnames = input_names;
        for (size_t i = 0; i < op_inputs.size(); i++)
            op_inputs[i]->consumers.push_back(op);
        for (size_t i = 0; i < node.outputs.size(); i++)
        {
            if (node.outputs[i].type == Pt2Argument::Tensor)
            {
                Operand* operand = new_operand(node.outputs[i].s);
                if (!operand || set_tensor_meta(node.outputs[i].s, operand) != 0)
                    return fail_node(index, node, error, &schema);
                operand->producer = op;
                op->outputs.push_back(operand);
            }
            else if (node.outputs[i].type == Pt2Argument::Tensors)
            {
                const std::string list_name = generated_name();
                Operand* list = new_operand(list_name);
                if (!list)
                    return fail_node(index, node, error, &schema);
                list->producer = op;
                op->outputs.push_back(list);

                const std::string unpack_name = generated_name();
                if (add_operator_name(unpack_name) != 0)
                    return fail_node(index, node, error, &schema);
                Operator* unpack = graph.new_operator("prim::ListUnpack", unpack_name);
                unpack->inputs.push_back(list);
                list->consumers.push_back(unpack);
                for (size_t j = 0; j < node.outputs[i].as.size(); j++)
                {
                    Operand* operand = new_operand(node.outputs[i].as[j]);
                    if (!operand || set_tensor_meta(node.outputs[i].as[j], operand) != 0)
                        return fail_node(index, node, error, &schema);
                    operand->producer = unpack;
                    unpack->outputs.push_back(operand);
                }
            }
            else
            {
                Operand* operand = new_operand(node.outputs[i].s);
                if (!operand)
                    return fail_node(index, node, error, &schema);
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

    int lower_output(const Pt2Argument& arg)
    {
        if (arg.type != Pt2Argument::Tensor && arg.type != Pt2Argument::SymInt && arg.type != Pt2Argument::SymBool && arg.type != Pt2Argument::SymFloat)
            return fail("unsupported user output");
        Operand* operand = graph.get_operand(arg.s);
        if (!operand)
            return fail("unknown graph output " + arg.s);
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
