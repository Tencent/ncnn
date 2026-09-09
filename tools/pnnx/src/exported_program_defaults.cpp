// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program_defaults.h"

#include <ATen/core/dispatch/Dispatcher.h>

namespace pnnx {
namespace pt2 {

static bool from_ivalue(const c10::IValue& value, Argument& argument)
{
    if (value.isNone())
    {
        argument.type = Argument::None;
        return true;
    }
    if (value.isBool())
    {
        argument.type = Argument::Boolean;
        argument.boolean = value.toBool();
        return true;
    }
    if (value.isInt())
    {
        argument.type = Argument::Integer;
        argument.integer = value.toInt();
        return true;
    }
    if (value.isDouble())
    {
        argument.type = Argument::FloatingPoint;
        argument.floating_point = value.toDouble();
        return true;
    }
    if (value.isString())
    {
        argument.type = Argument::String;
        argument.string = value.toStringRef();
        return true;
    }
    if (value.isIntList())
    {
        argument.type = Argument::Integers;
        const c10::List<int64_t> values = value.toIntList();
        for (size_t i = 0; i < values.size(); i++)
        {
            Argument item;
            item.type = Argument::Integer;
            item.integer = values.get(i);
            argument.values.push_back(item);
        }
        return true;
    }
    if (value.isDoubleList())
    {
        argument.type = Argument::FloatingPoints;
        const c10::List<double> values = value.toDoubleList();
        for (size_t i = 0; i < values.size(); i++)
        {
            Argument item;
            item.type = Argument::FloatingPoint;
            item.floating_point = values.get(i);
            argument.values.push_back(item);
        }
        return true;
    }
    if (value.isBoolList())
    {
        argument.type = Argument::Booleans;
        const c10::List<bool> values = value.toBoolList();
        for (size_t i = 0; i < values.size(); i++)
        {
            Argument item;
            item.type = Argument::Boolean;
            item.boolean = values.get(i);
            argument.values.push_back(item);
        }
        return true;
    }
    return false;
}

static bool parse_target(const std::string& target, std::string& name, std::string& overload)
{
    const std::string prefix = "torch.ops.";
    if (target.compare(0, prefix.size(), prefix) != 0)
        return false;
    const size_t namespace_end = target.find('.', prefix.size());
    const size_t operator_end = namespace_end == std::string::npos ? std::string::npos : target.find('.', namespace_end + 1);
    if (namespace_end == std::string::npos || operator_end == std::string::npos)
        return false;
    name = target.substr(prefix.size(), namespace_end - prefix.size()) + "::" + target.substr(namespace_end + 1, operator_end - namespace_end - 1);
    overload = target.substr(operator_end + 1);
    if (overload == "default")
        overload.clear();
    return true;
}

static bool alias_writes(const c10::AliasInfo* info)
{
    if (!info)
        return false;
    if (info->isWrite())
        return true;
    for (size_t i = 0; i < info->containedTypes().size(); i++)
        if (alias_writes(&info->containedTypes()[i])) return true;
    return false;
}

static bool is_scalar_argument(const Argument& argument)
{
    return argument.type == Argument::Integer || argument.type == Argument::FloatingPoint || argument.type == Argument::Boolean || argument.type == Argument::SymInteger || argument.type == Argument::SymFloat || argument.type == Argument::SymBoolean || argument.type == Argument::Complex;
}

static bool validate_argument_references(const Argument& argument, bool output, const std::string& location, std::string& error)
{
    if (argument.type == Argument::Unknown)
    {
        error = location + ": unknown serialized argument type";
        return false;
    }
    const bool symbolic = argument.type == Argument::SymInteger || argument.type == Argument::SymFloat || argument.type == Argument::SymBoolean;
    if (!argument.name.empty() && argument.type != Argument::Tensor && !symbolic)
    {
        error = location + ": reference name " + argument.name + " is not valid on a constant or container argument";
        return false;
    }
    if ((argument.type == Argument::Tensor || (output && symbolic)) && argument.name.empty())
    {
        error = location + ": reference name is empty for tensor or symbolic output";
        return false;
    }
    const bool container = argument.type == Argument::Tensors || argument.type == Argument::OptionalTensor || argument.type == Argument::OptionalTensors
                           || argument.type == Argument::Integers || argument.type == Argument::SymIntegers || argument.type == Argument::FloatingPoints || argument.type == Argument::SymFloats
                           || argument.type == Argument::Booleans || argument.type == Argument::SymBooleans || argument.type == Argument::Strings;
    if (!container && !argument.values.empty())
    {
        error = location + ": non-container argument contains nested values/references";
        return false;
    }
    // A Node alone cannot resolve references. The importer must still look up
    // named tensors/symbolic scalars, never use their default-valued data fields.
    for (size_t i = 0; i < argument.values.size(); i++)
    {
        if (!validate_argument_references(argument.values[i], output, location + "[" + std::to_string(i) + "]", error))
            return false;
    }
    return true;
}

static bool argument_matches_type(const Argument& argument, const c10::TypePtr& type)
{
    if (argument.type == Argument::OptionalTensor)
        return argument.values.size() == 1 && argument_matches_type(argument.values[0], type);
    if (type->kind() == c10::TypeKind::OptionalType)
        return argument.type == Argument::None || argument_matches_type(argument, type->cast<c10::OptionalType>()->getElementType());
    if (type->kind() == c10::TypeKind::ListType)
    {
        Argument element;
        switch (argument.type)
        {
        case Argument::Tensors:
            element.type = Argument::Tensor;
            break;
        case Argument::OptionalTensors:
            element.type = Argument::None;
            break;
        case Argument::Integers:
            element.type = Argument::Integer;
            break;
        case Argument::SymIntegers:
            element.type = Argument::SymInteger;
            break;
        case Argument::FloatingPoints:
            element.type = Argument::FloatingPoint;
            break;
        case Argument::SymFloats:
            element.type = Argument::SymFloat;
            break;
        case Argument::Booleans:
            element.type = Argument::Boolean;
            break;
        case Argument::SymBooleans:
            element.type = Argument::SymBoolean;
            break;
        case Argument::Strings:
            element.type = Argument::String;
            break;
        default:
            return false;
        }
        const c10::TypePtr& element_type = type->cast<c10::ListType>()->getElementType();
        if (argument.values.empty())
            return argument_matches_type(element, element_type);
        for (size_t i = 0; i < argument.values.size(); i++)
            if (!argument_matches_type(argument.values[i], element_type)) return false;
        return true;
    }
    const bool integer = argument.type == Argument::Integer || argument.type == Argument::SymInteger;
    const bool floating = argument.type == Argument::FloatingPoint || argument.type == Argument::SymFloat;
    const bool boolean = argument.type == Argument::Boolean || argument.type == Argument::SymBoolean;
    switch (type->kind())
    {
    case c10::TypeKind::TensorType:
        return argument.type == Argument::Tensor;
    case c10::TypeKind::IntType:
        // Older schemas erase dtype/layout/memory_format to int.
        return integer || argument.type == Argument::ScalarType || argument.type == Argument::Layout || argument.type == Argument::MemoryFormat;
    case c10::TypeKind::SymIntType:
        return integer;
    case c10::TypeKind::FloatType:
    case c10::TypeKind::SymFloatType:
        return floating || integer;
    case c10::TypeKind::BoolType:
    case c10::TypeKind::SymBoolType:
        return boolean;
    case c10::TypeKind::NumberType:
        return integer || floating || boolean || argument.type == Argument::Complex;
    case c10::TypeKind::ComplexType:
        return argument.type == Argument::Complex || integer || floating;
    case c10::TypeKind::StringType:
        return argument.type == Argument::String;
    case c10::TypeKind::DeviceObjType:
        return argument.type == Argument::DeviceValue;
    // Dispatcher defaults for these enums are already c10-encoded integers;
    // keep them as Integer, rather than reinterpreting them as serde enums.
    case c10::TypeKind::ScalarTypeType:
        return argument.type == Argument::ScalarType || (argument.type == Argument::Integer && ((argument.integer >= 0 && argument.integer <= 11) || argument.integer == 15));
    case c10::TypeKind::LayoutType:
        return argument.type == Argument::Layout || (argument.type == Argument::Integer && argument.integer == 0);
    case c10::TypeKind::MemoryFormatType:
        return argument.type == Argument::MemoryFormat || (argument.type == Argument::Integer && argument.integer >= 0 && argument.integer <= 3);
    case c10::TypeKind::NoneType:
        return argument.type == Argument::None;
    case c10::TypeKind::AnyType:
        return argument.type != Argument::Unknown;
    default:
        return false;
    }
}

static bool output_matches_type(const Argument& argument, const c10::TypePtr& type)
{
    if (type->kind() == c10::TypeKind::OptionalType)
        return argument.type == Argument::None || output_matches_type(argument, type->cast<c10::OptionalType>()->getElementType());
    if (type->kind() == c10::TypeKind::ListType)
    {
        const c10::TypePtr& element_type = type->cast<c10::ListType>()->getElementType();
        // Serde represents Tensor[] as ONE as_tensors argument, even when empty.
        if (argument.type != Argument::Tensors || element_type->kind() != c10::TypeKind::TensorType)
            return false;
        for (size_t i = 0; i < argument.values.size(); i++)
            if (!output_matches_type(argument.values[i], element_type)) return false;
        return true;
    }
    // Serde names scalar results through as_sym_* even for concrete metadata.
    // Unlike inputs, return values must not use numeric argument coercions.
    switch (type->kind())
    {
    case c10::TypeKind::TensorType:
        return argument.type == Argument::Tensor;
    case c10::TypeKind::IntType:
    case c10::TypeKind::SymIntType:
        return argument.type == Argument::SymInteger;
    case c10::TypeKind::FloatType:
    case c10::TypeKind::SymFloatType:
        return argument.type == Argument::SymFloat;
    case c10::TypeKind::BoolType:
    case c10::TypeKind::SymBoolType:
        return argument.type == Argument::SymBoolean;
    case c10::TypeKind::NumberType:
        return argument.type == Argument::SymInteger || argument.type == Argument::SymFloat || argument.type == Argument::SymBoolean;
    case c10::TypeKind::NoneType:
        return argument.type == Argument::None;
    default:
        return false;
    }
}

bool normalize_exported_program_node(Node& node, std::string& error)
{
    error.clear();
    const std::string location = (node.name.empty() ? "unnamed node" : node.name) + " (" + node.target + ")";
    for (size_t i = 0; i < node.inputs.size(); i++)
        if (!validate_argument_references(node.inputs[i].argument, false, location + "." + node.inputs[i].name, error)) return false;
    for (size_t i = 0; i < node.outputs.size(); i++)
        if (!validate_argument_references(node.outputs[i], true, location + ".outputs[" + std::to_string(i) + "]", error)) return false;
    std::string name;
    std::string overload;
    if (!parse_target(node.target, name, overload))
    {
        // Python operator functions have no dispatcher schema. Restrict this
        // escape hatch to known non-writing functions, never operator.setitem.
        const char* pure_operators[] = {"add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "neg", "pos", "abs", "eq", "ne", "lt", "le", "gt", "ge", "and_", "or_", "xor", "not_", "getitem"};
        for (size_t i = 0; i < sizeof(pure_operators) / sizeof(pure_operators[0]); i++)
        {
            if (node.target == std::string("_operator.") + pure_operators[i] || node.target == std::string("operator.") + pure_operators[i])
            {
                if (node.outputs.empty() || (node.outputs.size() == 1 && node.outputs[0].type == Argument::None))
                    break;
                return true;
            }
        }
        error = location + ": unsupported operator without a validated pure schema";
        return false;
    }

    c10::optional<c10::OperatorHandle> handle = c10::Dispatcher::singleton().findSchema({name, overload});
    if (handle.has_value() && overload == "Tensor")
    {
        for (size_t i = 0; i < node.inputs.size(); i++)
        {
            if (node.inputs[i].name != "other" || !is_scalar_argument(node.inputs[i].argument))
                continue;

            c10::optional<c10::OperatorHandle> scalar_handle = c10::Dispatcher::singleton().findSchema({name, "Scalar"});
            if (scalar_handle.has_value())
            {
                handle = scalar_handle;
                node.target.replace(node.target.size() - overload.size(), overload.size(), "Scalar");
                overload = "Scalar";
            }
            break;
        }
    }
    if (!handle.has_value())
    {
        error = location + ": operator schema was not found";
        return false;
    }
    const c10::FunctionSchema& schema = handle->schema();
    const std::vector<c10::Argument>& schema_arguments = schema.arguments();
    for (size_t i = 0; i < schema_arguments.size(); i++)
    {
        if (alias_writes(schema_arguments[i].alias_info()))
        {
            error = location + ": unsupported alias write/mutation of argument " + schema_arguments[i].name() + "; all schema writes are conservatively rejected, including local temporaries (alias/liveness analysis is not implemented)";
            return false;
        }
    }
    for (size_t i = 0; i < schema.returns().size(); i++)
    {
        if (alias_writes(schema.returns()[i].alias_info()))
        {
            error = location + ": unsupported alias write on return value; all schema writes are conservatively rejected, including local temporaries (alias/liveness analysis is not implemented)";
            return false;
        }
    }
    // Only ATen has a version contract here. A custom schema alone cannot prove
    // that external effects (files, state, tokens, Python callbacks) are absent.
    if (name.compare(0, 6, "aten::") != 0)
    {
        error = location + ": unsupported operator namespace; external side effects are not validated";
        return false;
    }
    const bool scalar_guard = node.target == "torch.ops.aten._assert_scalar.default";
    if (!scalar_guard && (name.find("_assert") != std::string::npos || name.find("sym_constrain") != std::string::npos || name == "aten::_test_check_tensor"))
    {
        error = location + ": unsupported guard; runtime guard evaluation is not implemented";
        return false;
    }
    // These factories advance RNG state without a tensor alias write. Do not
    // generalize to every nondeterministic tag: e.g. attention with dropout=0
    // does not necessarily consume RNG state.
    if (name == "aten::rand_like" || name == "aten::randn_like")
    {
        error = location + ": unsupported RNG state effect; random state preservation is not implemented";
        return false;
    }
    if (!scalar_guard && schema.aliasAnalysis() != c10::AliasAnalysisKind::FROM_SCHEMA && schema.aliasAnalysis() != c10::AliasAnalysisKind::PURE_FUNCTION)
    {
        error = location + ": unsupported side effects: schema alias analysis is " + c10::toString(schema.aliasAnalysis());
        return false;
    }
    const std::vector<c10::Argument>& schema_returns = schema.returns();
    // Multiple schema returns are separate serialized arguments, not a single
    // tensor list. A void scalar guard also permits the legacy [as_none] form.
    const bool void_guard_none = scalar_guard && schema_returns.empty() && node.outputs.size() == 1 && node.outputs[0].type == Argument::None;
    if (!void_guard_none && node.outputs.size() != schema_returns.size())
    {
        error = location + ": output count " + std::to_string(node.outputs.size()) + " does not match schema return count " + std::to_string(schema_returns.size());
        return false;
    }
    for (size_t i = 0; i < schema_returns.size(); i++)
    {
        if (!output_matches_type(node.outputs[i], schema_returns[i].real_type()))
        {
            error = location + ".outputs[" + std::to_string(i) + "]: output type " + std::to_string((int)node.outputs[i].type) + " does not match schema return type " + schema_returns[i].real_type()->str();
            return false;
        }
    }
    bool no_values = true;
    for (size_t i = 0; i < node.outputs.size(); i++)
        no_values = no_values && node.outputs[i].type == Argument::None;
    if (!scalar_guard && (schema.returns().empty() || no_values))
    {
        error = location + ": unsupported no-value/None-only node; side effects cannot be discarded";
        return false;
    }

    std::map<std::string, size_t> present;
    for (size_t i = 0; i < node.inputs.size(); i++)
    {
        if (!present.insert(std::make_pair(node.inputs[i].name, i)).second)
        {
            error = location + ": duplicate argument " + node.inputs[i].name;
            return false;
        }
    }
    std::vector<NamedArgument> ordered_inputs;
    ordered_inputs.reserve(schema_arguments.size());
    size_t matched_input_count = 0;
    for (size_t i = 0; i < schema_arguments.size(); i++)
    {
        const c10::Argument& schema_argument = schema_arguments[i];
        NamedArgument argument;
        std::map<std::string, size_t>::const_iterator it = present.find(schema_argument.name());
        if (it != present.end())
        {
            argument = node.inputs[it->second];
            matched_input_count++;
        }
        else
        {
            if (!schema_argument.default_value().has_value())
            {
                error = location + ": required argument " + schema_argument.name() + " is missing";
                return false;
            }
            argument.name = schema_argument.name();
            argument.kind = schema_argument.kwarg_only() ? NamedArgument::Keyword : NamedArgument::Positional;
            if (!from_ivalue(*schema_argument.default_value(), argument.argument))
            {
                error = location + ": unsupported default value for argument " + schema_argument.name();
                return false;
            }
        }
        if (!argument_matches_type(argument.argument, schema_argument.real_type()))
        {
            error = location + "." + schema_argument.name() + ": argument type " + std::to_string((int)argument.argument.type) + " does not match schema type " + schema_argument.real_type()->str();
            return false;
        }
        ordered_inputs.push_back(argument);
    }
    if (matched_input_count != node.inputs.size())
    {
        error = location + ": argument was not found in operator schema";
        return false;
    }
    if (scalar_guard)
    {
        if (!no_values || ordered_inputs.size() != 2)
        {
            error = location + ": malformed scalar guard";
            return false;
        }
        const Argument& predicate = ordered_inputs[0].argument;
        if ((predicate.type != Argument::Boolean && predicate.type != Argument::SymBoolean) || !predicate.name.empty())
        {
            error = location + ": unsupported guard; only a concrete boolean can be evaluated, runtime guards are not implemented";
            return false;
        }
        if (!predicate.boolean)
        {
            error = location + ": scalar guard is false: " + ordered_inputs[1].argument.string;
            return false;
        }
    }
    node.inputs.swap(ordered_inputs);
    return true;
}

bool append_default_arguments(ExportedProgram& program, std::string& error)
{
    error.clear();
    for (size_t i = 0; i < program.graph.nodes.size(); i++)
    {
        if (!normalize_exported_program_node(program.graph.nodes[i], error))
            return false;
    }
    return true;
}

} // namespace pt2
} // namespace pnnx