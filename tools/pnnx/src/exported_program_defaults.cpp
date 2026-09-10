// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "exported_program_defaults.h"

#include <set>

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/ScalarType.h>

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

static bool has_reference(const Argument& argument)
{
    if (!argument.name.empty())
        return true;
    for (size_t i = 0; i < argument.values.size(); i++)
        if (has_reference(argument.values[i])) return true;
    return false;
}

static int metadata_dtype(int serde)
{
    // Explicit serde mapping, not a cast: BFLOAT16=13 is c10 BFloat16=15.
    const c10::ScalarType types[] = {c10::ScalarType::Byte, c10::ScalarType::Char, c10::ScalarType::Short,
                                   c10::ScalarType::Int, c10::ScalarType::Long, c10::ScalarType::Half,
                                   c10::ScalarType::Float, c10::ScalarType::Double, c10::ScalarType::ComplexHalf,
                                   c10::ScalarType::ComplexFloat, c10::ScalarType::ComplexDouble, c10::ScalarType::Bool,
                                   c10::ScalarType::BFloat16};
    return serde >= 1 && serde <= 13 ? (int)types[serde - 1] : -1;
}

static bool static_dimensions(const std::vector<SymInt>& dimensions)
{
    for (size_t i = 0; i < dimensions.size(); i++)
        if (dimensions[i].type != SymInt::Integer || dimensions[i].integer < 0) return false;
    return true;
}

static bool validate_metadata_guard(const std::vector<NamedArgument>& inputs, const Graph* context, const std::string& location, std::string& error)
{
    if (!context)
    {
        error = location + ": metadata guard requires graph metadata context";
        return false;
    }
    if (inputs.empty() || inputs[0].name != "a" || inputs[0].argument.type != Argument::Tensor)
    {
        error = location + ": unsupported metadata guard schema";
        return false;
    }
    const std::string& reference = inputs[0].argument.name;
    std::map<std::string, TensorMeta>::const_iterator it = context->tensor_values.find(reference);
    if (it == context->tensor_values.end())
    {
        error = location + ": tensor metadata is missing for " + reference;
        return false;
    }
    const TensorMeta& meta = it->second;
    if (metadata_dtype(meta.scalar_type) < 0 || meta.device.type != "cpu" || meta.device.has_index || meta.layout != 7
            || meta.sizes.size() != meta.strides.size() || !static_dimensions(meta.sizes) || !static_dimensions(meta.strides))
    {
        error = location + ": metadata guard requires known dtype, CPU/Strided and static sizes/strides for " + reference + "; hints are not proof";
        return false;
    }
    for (size_t i = 1; i < inputs.size(); i++)
    {
        const std::string& field = inputs[i].name;
        const Argument& expected = inputs[i].argument;
        if (field != "size" && field != "stride" && field != "dtype" && field != "device" && field != "layout")
        {
            error = location + ": unsupported metadata guard field " + field;
            return false;
        }
        if (has_reference(expected))
        {
            error = location + "." + field + ": runtime-dependent metadata guard argument is not supported";
            return false;
        }
        // None means no check, including dispatcher defaults. It is not a
        // default dtype/device/layout, and must not be decoded as serde zero.
        if (expected.type == Argument::None)
            continue;
        bool matches = false;
        if (field == "size" || field == "stride")
        {
            const std::vector<SymInt>& actual = field == "size" ? meta.sizes : meta.strides;
            matches = (expected.type == Argument::Integers || expected.type == Argument::SymIntegers) && expected.values.size() == actual.size();
            for (size_t j = 0; matches && j < actual.size(); j++)
                matches = (expected.values[j].type == Argument::Integer || expected.values[j].type == Argument::SymInteger) && expected.values[j].integer == actual[j].integer;
        }
        else if (field == "dtype")
        {
            // Explicit as_scalar_type is serde; schema defaults/as_int are c10.
            const int64_t dtype = expected.type == Argument::ScalarType
                                      ? (expected.integer >= 1 && expected.integer <= 13 ? metadata_dtype((int)expected.integer) : -1)
                                      : expected.type == Argument::Integer ? expected.integer : -1;
            matches = dtype >= 0 && dtype == metadata_dtype(meta.scalar_type);
        }
        else if (field == "device")
        {
            matches = expected.type == Argument::DeviceValue && expected.device.type == meta.device.type
                      && expected.device.has_index == meta.device.has_index && (!expected.device.has_index || expected.device.index == meta.device.index);
        }
        else if (field == "layout")
        {
            matches = (expected.type == Argument::Layout && expected.integer == 7) || (expected.type == Argument::Integer && expected.integer == 0);
        }
        if (!matches)
        {
            error = location + ": metadata guard is false or unsupported for " + reference + "." + field;
            return false;
        }
    }
    return true;
}

static const char* local_inplace_counterpart(const std::string& name, const std::string& overload)
{
    // Exact dispatcher pairs, not a generic trailing-underscore rewrite.
    static const struct
    {
        const char* inplace;
        const char* overload;
        const char* functional;
    } pairs[] = {
        {"aten::relu_", "", "aten::relu"},
        {"aten::relu6_", "", "aten::relu6"},
        {"aten::hardtanh_", "", "aten::hardtanh"},
        {"aten::hardsigmoid_", "", "aten::hardsigmoid"},
        {"aten::hardswish_", "", "aten::hardswish"},
        {"aten::silu_", "", "aten::silu"},
        {"aten::sigmoid_", "", "aten::sigmoid"},
        {"aten::tanh_", "", "aten::tanh"},
        {"aten::clamp_", "", "aten::clamp"},
        {"aten::clamp_", "Tensor", "aten::clamp"},
        {"aten::leaky_relu_", "", "aten::leaky_relu"},
        {"aten::elu_", "", "aten::elu"},
        {"aten::celu_", "", "aten::celu"},
        {"aten::fill_", "Scalar", "aten::fill"},
        {"aten::fill_", "Tensor", "aten::fill"},
        {"aten::add_", "Scalar", "aten::add"},
        {"aten::add_", "Tensor", "aten::add"},
        {"aten::sub_", "Scalar", "aten::sub"},
        {"aten::sub_", "Tensor", "aten::sub"},
        {"aten::mul_", "Scalar", "aten::mul"},
        {"aten::mul_", "Tensor", "aten::mul"},
        {"aten::div_", "Scalar", "aten::div"},
        {"aten::div_", "Tensor", "aten::div"}
    };
    for (size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); i++)
        if (name == pairs[i].inplace && overload == pairs[i].overload) return pairs[i].functional;
    return 0;
}

static bool known_allocator(const std::string& name, const std::string& overload)
{
    // Only these non-out overloads prove a new whole tensor. In particular,
    // _unsafe_view has no alias annotation, and contiguous may return self.
    // No RNG factories, arbitrary pure operators or containers establish roots.
    if (name == "aten::add" || name == "aten::mul" || name == "aten::sub" || name == "aten::div")
        return overload == "Tensor" || overload == "Scalar";
    // Functional pointwise/fill counterparts allocate too. Keep this explicit
    // so a normalized chain can prove ownership again on subsequent imports.
    if (name == "aten::fill")
        return overload == "Tensor" || overload == "Scalar";
    if (name == "aten::clamp")
        return overload.empty() || overload == "Tensor";
    if (name == "aten::empty")
        return overload == "memory_format";
    return overload.empty() && (name == "aten::clone" || name == "aten::zeros" || name == "aten::ones"
                               || name == "aten::new_empty" || name == "aten::new_zeros" || name == "aten::new_ones"
                               || name == "aten::relu" || name == "aten::relu6" || name == "aten::hardtanh"
                               || name == "aten::hardsigmoid" || name == "aten::hardswish" || name == "aten::silu"
                               || name == "aten::sigmoid" || name == "aten::tanh" || name == "aten::leaky_relu"
                               || name == "aten::elu" || name == "aten::celu");
}

static bool normalize_node(Node& node, std::string& error, const Graph* metadata_context, bool check_local_inplace)
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
    // Internal deferral only: append_default_arguments proves ownership and
    // liveness before committing this node. Public single-node calls never defer.
    const char* counterpart = check_local_inplace ? local_inplace_counterpart(name, overload) : 0;
    c10::optional<c10::OperatorHandle> functional_handle;
    if (counterpart) functional_handle = c10::Dispatcher::singleton().findSchema({counterpart, overload});
    const bool local_inplace = functional_handle.has_value() && functional_handle->schema().returns().size() == 1
                               && !functional_handle->schema().returns()[0].alias_info()
                               && functional_handle->schema().returns()[0].real_type()->kind() == c10::TypeKind::TensorType;
    // detach_ is NOT a value-writing pointwise allocator. Defer only its exact
    // schema to the whole-program no-grad lifted-constant/identity-alias proof.
    const bool constant_detach = check_local_inplace && name == "aten::detach_" && overload.empty();
    for (size_t i = 0; i < schema_arguments.size(); i++)
    {
        if (alias_writes(schema_arguments[i].alias_info()) && !((local_inplace || constant_detach) && i == 0 && schema_arguments[i].name() == "self"))
        {
            error = location + ": unsupported alias write/mutation of argument " + schema_arguments[i].name() + "; only whole-program proven single-use unaliased local whitelisted pointwise/fill_ mutations are supported";
            return false;
        }
    }
    for (size_t i = 0; i < schema.returns().size(); i++)
    {
        if (alias_writes(schema.returns()[i].alias_info()) && !((local_inplace || constant_detach) && i == 0 && schema.returns().size() == 1))
        {
            error = location + ": unsupported alias write on return value";
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
    const bool metadata_guard = node.target == "torch.ops.aten._assert_tensor_metadata.default";
    const bool evaluated_guard = scalar_guard || metadata_guard;
    if (!evaluated_guard && (name.find("_assert") != std::string::npos || name.find("sym_constrain") != std::string::npos || name == "aten::_test_check_tensor"))
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
    if (!evaluated_guard && schema.aliasAnalysis() != c10::AliasAnalysisKind::FROM_SCHEMA && schema.aliasAnalysis() != c10::AliasAnalysisKind::PURE_FUNCTION)
    {
        error = location + ": unsupported side effects: schema alias analysis is " + c10::toString(schema.aliasAnalysis());
        return false;
    }
    const std::vector<c10::Argument>& schema_returns = schema.returns();
    // Multiple schema returns are separate serialized arguments, not a single
    // tensor list. Evaluated void guards also permit serde's [as_none] form.
    const bool void_guard_none = evaluated_guard && schema_returns.empty() && node.outputs.size() == 1 && node.outputs[0].type == Argument::None;
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
    if (!evaluated_guard && (schema.returns().empty() || no_values))
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
    if (metadata_guard && (!no_values || !schema_returns.empty() || !validate_metadata_guard(ordered_inputs, metadata_context, location, error)))
    {
        if (error.empty()) error = location + ": malformed metadata guard return";
        return false;
    }
    node.inputs.swap(ordered_inputs);
    return true;
}

bool normalize_exported_program_node(Node& node, std::string& error, const Graph* metadata_context)
{
    return normalize_node(node, error, metadata_context, false);
}

static void tensor_references(const Argument& argument, std::vector<std::string>& names)
{
    if (argument.type == Argument::Tensor)
        names.push_back(argument.name);
    for (size_t i = 0; i < argument.values.size(); i++)
        tensor_references(argument.values[i], names);
}

typedef std::set<std::string> TensorRoots;
typedef std::map<std::string, TensorRoots> AliasProvenance;

static TensorRoots argument_roots(const Argument& argument, const AliasProvenance& provenance)
{
    std::vector<std::string> names;
    tensor_references(argument, names);
    TensorRoots roots;
    for (size_t i = 0; i < names.size(); i++)
    {
        AliasProvenance::const_iterator it = provenance.find(names[i]);
        if (it == provenance.end())
            roots.insert(""); // Unresolved reference is unknown, never fresh.
        else
            roots.insert(it->second.begin(), it->second.end());
    }
    return roots;
}

static void alias_sets(const c10::AliasInfo* info, std::set<c10::Symbol>& sets)
{
    if (!info)
        return;
    sets.insert(info->beforeSets().begin(), info->beforeSets().end());
    sets.insert(info->afterSets().begin(), info->afterSets().end());
    for (size_t i = 0; i < info->containedTypes().size(); i++)
        alias_sets(&info->containedTypes()[i], sets);
}

static TensorRoots unknown_roots(const AliasProvenance& provenance)
{
    TensorRoots roots;
    roots.insert("");
    for (AliasProvenance::const_iterator it = provenance.begin(); it != provenance.end(); ++it)
        roots.insert(it->second.begin(), it->second.end());
    return roots;
}

static bool record_provenance(const Node& node, AliasProvenance& provenance, bool constant_lift, std::string& error)
{
    std::string name;
    std::string overload;
    const bool parsed = parse_target(node.target, name, overload);
    c10::optional<c10::OperatorHandle> handle;
    if (parsed) handle = c10::Dispatcher::singleton().findSchema({name, overload});
    // Unknown/Python aliases cannot establish fresh ownership. Wildcard and
    // unmatched schema aliases conservatively include all preceding roots.
    std::vector<std::pair<std::string, TensorRoots> > outputs;
    for (size_t i = 0; i < node.outputs.size(); i++)
    {
        std::vector<std::string> names;
        tensor_references(node.outputs[i], names);
        if (names.empty()) continue;
        TensorRoots roots;
        bool fresh = false;
        if (handle.has_value() && i < handle->schema().returns().size())
        {
            const c10::FunctionSchema& schema = handle->schema();
            const c10::AliasInfo* output_alias = schema.returns()[i].alias_info();
            // Absence of alias annotations is necessary, never sufficient:
            // only known allocators or a proven constant lift establish roots.
            fresh = (known_allocator(name, overload) || constant_lift) && !output_alias && schema.returns().size() == 1
                    && node.outputs.size() == 1 && node.outputs[i].type == Argument::Tensor;
            std::set<c10::Symbol> output_sets;
            alias_sets(output_alias, output_sets);
            if (output_sets.count(c10::AliasInfo::wildcardSet()))
                roots = unknown_roots(provenance);
            else if (!fresh)
            {
                for (size_t j = 0; j < schema.arguments().size(); j++)
                {
                    std::set<c10::Symbol> input_sets;
                    alias_sets(schema.arguments()[j].alias_info(), input_sets);
                    bool matches = input_sets.count(c10::AliasInfo::wildcardSet()) != 0;
                    for (std::set<c10::Symbol>::const_iterator s = output_sets.begin(); s != output_sets.end(); ++s)
                        matches = matches || input_sets.count(*s) != 0;
                    if (matches)
                    {
                        const TensorRoots input_roots = argument_roots(node.inputs[j].argument, provenance);
                        roots.insert(input_roots.begin(), input_roots.end());
                    }
                }
            }
        }
        if (!fresh && roots.empty()) roots = unknown_roots(provenance);
        for (size_t j = 0; j < names.size(); j++)
        {
            TensorRoots result = roots;
            if (fresh) result.insert(names[j]);
            outputs.push_back(std::make_pair(names[j], result));
        }
    }
    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (outputs[i].first.empty() || !provenance.insert(outputs[i]).second)
        {
            error = node.name + " (" + node.target + "): tensor output " + outputs[i].first + " is empty or already defined";
            return false;
        }
    }
    return true;
}

static bool same_dimension(const SymInt& a, const SymInt& b)
{
    // Structural metadata comparison only; equal concrete hints do not prove
    // equal runtime expressions. No symbolic evaluation/runtime is introduced.
    return a.type == b.type && ((a.type == SymInt::Integer && a.integer == b.integer)
                               || (a.type == SymInt::Expression && a.expression == b.expression));
}

static bool validate_local_inplace_metadata(const Node& node, const Graph& graph, std::string& error)
{
    const std::map<std::string, TensorMeta>::const_iterator self = graph.tensor_values.find(node.inputs[0].argument.name);
    const std::map<std::string, TensorMeta>::const_iterator result = graph.tensor_values.find(node.outputs[0].name);
    // Keep the existing optional metadata contract, but never silently rewrite
    // a recorded in-place return whose metadata disagrees with its target.
    if (self == graph.tensor_values.end() || result == graph.tensor_values.end())
        return true;
    const TensorMeta& a = self->second;
    const TensorMeta& b = result->second;
    bool matches = a.scalar_type == b.scalar_type && a.layout == b.layout && a.device.type == b.device.type
                   && a.device.has_index == b.device.has_index && (!a.device.has_index || a.device.index == b.device.index)
                   && a.sizes.size() == b.sizes.size() && a.strides.size() == b.strides.size()
                   && same_dimension(a.storage_offset, b.storage_offset);
    for (size_t i = 0; matches && i < a.sizes.size(); i++)
        matches = same_dimension(a.sizes[i], b.sizes[i]);
    for (size_t i = 0; matches && i < a.strides.size(); i++)
        matches = same_dimension(a.strides[i], b.strides[i]);
    if (!matches)
        error = node.name + " (" + node.target + "): unsupported alias write/mutation of argument self; in-place return metadata disagrees with target";
    return matches;
}

static bool static_tensor_metadata(const TensorMeta& meta)
{
    return metadata_dtype(meta.scalar_type) >= 0 && meta.device.type == "cpu" && !meta.device.has_index && meta.layout == 7
           && meta.sizes.size() == meta.strides.size() && static_dimensions(meta.sizes) && static_dimensions(meta.strides)
           && meta.storage_offset.type == SymInt::Integer && meta.storage_offset.integer >= 0;
}

static bool concrete_real_scalar(const Argument& argument)
{
    return !has_reference(argument) && (argument.type == Argument::Integer || argument.type == Argument::FloatingPoint
                                       || argument.type == Argument::Boolean || argument.type == Argument::SymInteger
                                       || argument.type == Argument::SymFloat || argument.type == Argument::SymBoolean);
}

static bool validate_arithmetic_inplace(const Node& node, const Graph& graph, std::string& error)
{
    const std::string location = node.name + " (" + node.target + ")";
    const std::map<std::string, TensorMeta>::const_iterator self = graph.tensor_values.find(node.inputs[0].argument.name);
    const std::map<std::string, TensorMeta>::const_iterator result = graph.tensor_values.find(node.outputs[0].name);
    if (self == graph.tensor_values.end() || result == graph.tensor_values.end()
            || !static_tensor_metadata(self->second) || !static_tensor_metadata(result->second))
    {
        error = location + ": unsupported arithmetic alias write; dtype promotion proof requires static input/output tensor metadata";
        return false;
    }
    const TensorMeta& a = self->second;
    // In-place kernels cast back to self; functional kernels use result_type.
    // Only real floating self plus wrapped real scalars or same-dtype tensors
    // proves these agree. Integer division and mixed tensor dtypes are NOT safe
    // merely because the recorded in-place output has the same dtype as self.
    if ((a.scalar_type != 6 && a.scalar_type != 7 && a.scalar_type != 8 && a.scalar_type != 13)
            || a.scalar_type != result->second.scalar_type)
    {
        error = location + ": unsupported arithmetic alias write; dtype promotion requires floating self and unchanged result dtype";
        return false;
    }
    for (size_t i = 1; i < node.inputs.size(); i++)
    {
        const Argument& argument = node.inputs[i].argument;
        if (node.inputs[i].name == "other" && argument.type == Argument::Tensor)
        {
            const std::map<std::string, TensorMeta>::const_iterator other = graph.tensor_values.find(argument.name);
            if (other == graph.tensor_values.end() || !static_tensor_metadata(other->second) || other->second.scalar_type != a.scalar_type)
            {
                error = location + ": unsupported arithmetic alias write; dtype promotion requires static same-dtype other tensor";
                return false;
            }
            const std::vector<SymInt>& sizes = other->second.sizes;
            bool broadcasts = sizes.size() <= a.sizes.size();
            for (size_t j = 0; broadcasts && j < sizes.size(); j++)
                broadcasts = sizes[sizes.size() - 1 - j].integer == 1 || sizes[sizes.size() - 1 - j].integer == a.sizes[a.sizes.size() - 1 - j].integer;
            if (!broadcasts)
            {
                error = location + ": unsupported arithmetic alias write; other tensor must broadcast without changing self shape";
                return false;
            }
        }
        else if (!concrete_real_scalar(argument))
        {
            error = location + ": unsupported arithmetic alias write; dtype promotion requires concrete real scalar other/alpha";
            return false;
        }
    }
    return true;
}

static bool is_constant_lift_fresh_copy(const Node& node, const Graph& graph, const AliasProvenance& provenance,
                                       const TensorRoots& constants, const TensorRoots& constant_aliases)
{
    // Only this exact copy preserves constant-source permission. A clone of a
    // user tensor, another fresh operator, or a copy of a view proves nothing.
    if (node.target != "torch.ops.aten.lift_fresh_copy.default" || node.inputs.size() != 1
            || node.inputs[0].name != "self" || node.inputs[0].argument.type != Argument::Tensor
            || node.outputs.size() != 1 || node.outputs[0].type != Argument::Tensor)
        return false;
    const std::string& self = node.inputs[0].argument.name;
    const AliasProvenance::const_iterator roots = provenance.find(self);
    if (!constant_aliases.count(self) || roots == provenance.end() || roots->second.size() != 1 || !constants.count(*roots->second.begin()))
        return false;
    const std::map<std::string, TensorMeta>::const_iterator a = graph.tensor_values.find(self);
    const std::map<std::string, TensorMeta>::const_iterator b = graph.tensor_values.find(node.outputs[0].name);
    const std::map<std::string, TensorMeta>::const_iterator root = graph.tensor_values.find(*roots->second.begin());
    if (a == graph.tensor_values.end() || b == graph.tensor_values.end() || root == graph.tensor_values.end())
        return false;
    const TensorMeta* metas[] = {&root->second, &a->second, &b->second};
    for (size_t i = 0; i < sizeof(metas) / sizeof(metas[0]); i++)
    {
        const TensorMeta& meta = *metas[i];
        // Static metadata requires CPU (no device index) and strided layout.
        // A copy may change strides/offset, but not sizes, dtype or no-grad.
        if (!static_tensor_metadata(meta) || meta.requires_grad || meta.scalar_type != a->second.scalar_type
                || meta.sizes.size() != a->second.sizes.size())
            return false;
        for (size_t j = 0; j < meta.sizes.size(); j++)
            if (!same_dimension(meta.sizes[j], a->second.sizes[j])) return false;
    }
    return true;
}

static bool validate_constant_detach(const Node& node, const Graph& graph, const AliasProvenance& provenance,
                                     const TensorRoots& constants, const TensorRoots& constant_aliases, std::string& error)
{
    const std::string& self = node.inputs[0].argument.name;
    const AliasProvenance::const_iterator roots = provenance.find(self);
    if (!constant_aliases.count(self) || roots == provenance.end() || roots->second.size() != 1 || !constants.count(*roots->second.begin()))
    {
        error = node.name + " (" + node.target + "): unsupported detach_ alias write; requires a non-view alias rooted only in a lifted TensorConstant, never user input, parameter or buffer";
        return false;
    }
    const std::map<std::string, TensorMeta>::const_iterator a = graph.tensor_values.find(self);
    const std::map<std::string, TensorMeta>::const_iterator b = graph.tensor_values.find(node.outputs[0].name);
    const std::map<std::string, TensorMeta>::const_iterator root = graph.tensor_values.find(*roots->second.begin());
    if (a == graph.tensor_values.end() || b == graph.tensor_values.end() || root == graph.tensor_values.end()
            || a->second.requires_grad || b->second.requires_grad || root->second.requires_grad)
    {
        error = node.name + " (" + node.target + "): unsupported detach_ alias write; tensor, result and lifted constant metadata must have requires_grad=false";
        return false;
    }
    return validate_local_inplace_metadata(node, graph, error);
}

bool append_default_arguments(ExportedProgram& program, std::string& error)
{
    error.clear();
    AliasProvenance provenance;
    std::vector<std::string> external_names;
    for (size_t i = 0; i < program.graph.inputs.size(); i++)
        tensor_references(program.graph.inputs[i], external_names);
    const TensorRoots external(external_names.begin(), external_names.end());
    for (TensorRoots::const_iterator it = external.begin(); it != external.end(); ++it)
        provenance[*it].insert(*it);

    TensorRoots user_inputs;
    TensorRoots constants;
    TensorRoots nonconstants;
    for (size_t i = 0; i < program.signature.inputs.size(); i++)
    {
        const InputSpec& spec = program.signature.inputs[i];
        std::vector<std::string> names;
        tensor_references(spec.argument, names);
        if (spec.type == InputSpec::UserInput)
            user_inputs.insert(names.begin(), names.end());
        if (spec.type == InputSpec::TensorConstant && spec.argument.type == Argument::Tensor && !spec.target.empty())
            constants.insert(names.begin(), names.end());
        else
            nonconstants.insert(names.begin(), names.end());
    }
    for (TensorRoots::iterator it = constants.begin(); it != constants.end();)
    {
        if (!external.count(*it) || nonconstants.count(*it))
            it = constants.erase(it);
        else
            ++it;
    }
    // Only direct constants, proven constant lifts and detach results are
    // known non-view aliases. Constant lifts establish their own constant root.
    // Schema alias roots alone cannot distinguish a view on which detach_ is
    // invalid. Do not propagate this permission through alias/view/containers.
    TensorRoots constant_aliases = constants;

    // Include nested references, graph returns and signature returns. An old
    // target escaping through any of them cannot be silently left unmodified.
    std::vector<std::string> uses;
    for (size_t i = 0; i < program.graph.nodes.size(); i++)
        for (size_t j = 0; j < program.graph.nodes[i].inputs.size(); j++)
            tensor_references(program.graph.nodes[i].inputs[j].argument, uses);
    for (size_t i = 0; i < program.graph.outputs.size(); i++)
        tensor_references(program.graph.outputs[i], uses);
    for (size_t i = 0; i < program.signature.outputs.size(); i++)
        tensor_references(program.signature.outputs[i].argument, uses);
    std::map<std::string, size_t> use_counts;
    for (size_t i = 0; i < uses.size(); i++) use_counts[uses[i]]++;

    std::vector<Node> normalized = program.graph.nodes;
    for (size_t i = 0; i < normalized.size(); i++)
    {
        Node& node = normalized[i];
        if (!normalize_node(node, error, &program.graph, true))
            return false;
        const std::string location = node.name + " (" + node.target + ")";
        if (node.target == "torch.ops.aten._assert_tensor_metadata.default")
        {
            const std::string& reference = node.inputs[0].argument.name;
            if (!provenance.count(reference))
            {
                error = location + ": input " + reference + " is not defined before metadata guard";
                return false;
            }
            if (user_inputs.count(reference) && program.graph.tensor_values.at(reference).scalar_type != 7)
            {
                error = location + ": guarded graph inputs require static float32; native ncnn is not a generic dtype runtime";
                return false;
            }
        }
        std::string name;
        std::string overload;
        const char* counterpart = parse_target(node.target, name, overload) ? local_inplace_counterpart(name, overload) : 0;
        if (counterpart)
        {
            const std::string& self = node.inputs[0].argument.name;
            AliasProvenance::const_iterator it = provenance.find(self);
            // Constant-derived roots authorize detach_, never new value writes.
            // Otherwise only known_allocator introduces a local root named self.
            // Views of that root are not whole targets, even with a single use.
            bool owned = it != provenance.end() && it->second.size() == 1 && it->second.count(self) && !external.count(self) && !constants.count(self);
            for (AliasProvenance::const_iterator other = provenance.begin(); owned && other != provenance.end(); ++other)
                if (other->first != self && other->second.count(self)) owned = false;
            if (!owned || use_counts[self] != 1)
            {
                error = location + ": unsupported alias write/mutation of argument self; requires a single-use unaliased local target allocated by a known allocator, with no external roots, existing aliases, other uses or escaping original outputs";
                return false;
            }
            if (!validate_local_inplace_metadata(node, program.graph, error))
                return false;
            if ((name == "aten::add_" || name == "aten::sub_" || name == "aten::mul_" || name == "aten::div_")
                    && !validate_arithmetic_inplace(node, program.graph, error))
                return false;
            // Only the mutation result may be used afterwards. No alias engine
            // or rewiring is needed. The original schema/return contract and
            // argument order were checked before inspecting self; validate the
            // actual functional counterpart again, including explicit defaults.
            node.target = "torch.ops.aten." + std::string(counterpart).substr(6) + "." + (overload.empty() ? "default" : overload);
            if (!normalize_exported_program_node(node, error, &program.graph))
                return false;
        }
        else if (node.target == "torch.ops.aten.detach_.default")
        {
            if (!validate_constant_detach(node, program.graph, provenance, constants, constant_aliases, error))
                return false;
            node.target = "torch.ops.aten.detach.default";
            if (!normalize_exported_program_node(node, error, &program.graph))
                return false;
        }
        const bool constant_lift = is_constant_lift_fresh_copy(node, program.graph, provenance, constants, constant_aliases);
        if (!record_provenance(node, provenance, constant_lift, error))
            return false;
        if (constant_lift)
        {
            const std::string& result = node.outputs[0].name;
            const TensorRoots& roots = provenance.at(result);
            // Also require the schema to have proved a fresh, non-alias return.
            if (roots.size() == 1 && roots.count(result))
            {
                constants.insert(result);
                constant_aliases.insert(result);
            }
        }
        if (node.target == "torch.ops.aten.detach.default" && constant_aliases.count(node.inputs[0].argument.name))
            constant_aliases.insert(node.outputs[0].name);
    }
    program.graph.nodes.swap(normalized);
    return true;
}

} // namespace pt2
} // namespace pnnx