// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fold_constants.h"
#include "load_pt2/pt2_tensor.h"

#include <ATen/ATen.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/grad_mode.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdio.h>
#include <string.h>

namespace pnnx {

static at::ScalarType scalar_type(int type)
{
    const int dtype = pt2_scalar_type(type);
    return dtype < 0 ? at::ScalarType::Undefined : (at::ScalarType)dtype;
}

static bool constant_value(const Operand* operand, const c10::TypePtr& type, const std::map<const Operand*, double>& floating_constants, c10::IValue& value)
{
    const Operator* op = operand->producer;
    if (type->kind() == c10::TypeKind::OptionalType)
    {
        if (op->type == "prim::Constant" && op->params.at("value").type == 0)
        {
            value = c10::IValue();
            return true;
        }
        return constant_value(operand, type->cast<c10::OptionalType>()->getElementType(), floating_constants, value);
    }
    if (op->type == "pnnx.Attribute")
    {
        const Attribute& attr = op->attrs.at("data");
        if (type->kind() != c10::TypeKind::TensorType || scalar_type(attr.type) == at::ScalarType::Undefined)
            return false;
        std::vector<int64_t> shape(attr.shape.begin(), attr.shape.end());
        if (attr.data.empty())
            value = at::empty(shape, at::TensorOptions().dtype(scalar_type(attr.type)));
        else
            value = at::from_blob((void*)attr.data.data(), shape, at::TensorOptions().dtype(scalar_type(attr.type)));
        return true;
    }
    if (type->kind() == c10::TypeKind::ListType && op->type == "prim::ListConstruct")
    {
        const c10::TypePtr element = type->cast<c10::ListType>()->getElementType();
        c10::impl::GenericList list(element);
        for (const Operand* input : op->inputs)
        {
            c10::IValue item;
            if (!constant_value(input, element, floating_constants, item))
                return false;
            list.push_back(item);
        }
        value = list;
        return true;
    }
    if (op->type != "prim::Constant")
        return false;
    const Parameter& p = op->params.at("value");
    if (type->kind() == c10::TypeKind::DeviceObjType && p.type == 4 && p.s == "cpu")
        value = c10::Device(c10::kCPU);
    else if (p.type == 0)
        value = c10::IValue();
    else if (p.type == 1)
        value = p.b;
    else if (p.type == 2)
        value = (int64_t)p.i;
    else if (p.type == 3)
        value = floating_constants.count(operand) ? floating_constants.at(operand) : (double)p.f;
    else if (p.type == 4)
        value = p.s;
    else if (p.type == 5)
        value = std::vector<int64_t>(p.ai.begin(), p.ai.end());
    else if (p.type == 6)
        value = std::vector<double>(p.af.begin(), p.af.end());
    else if (p.type == 7)
        value = p.as;
    else if (p.type == 10)
        value = c10::complex<double>(p.c.real(), p.c.imag());
    else
        return false;
    // Export permits wrapped Python numbers in some Tensor argument slots.
    if (type->kind() == c10::TypeKind::TensorType && value.isScalar())
        value = at::native::wrapped_scalar_tensor(value.toScalar());
    return true;
}

static void replace_constant(Operator* op, const Parameter& value)
{
    for (Operand* input : op->inputs)
        input->remove_consumer(op);
    op->inputs.clear();
    op->inputnames.clear();
    op->params.clear();
    op->type = "prim::Constant";
    op->params["value"] = value;
}

static bool fold_size(Operator* op)
{
    if (op->type != "aten::size" || op->inputs.size() != 2 || op->outputs.size() != 1)
        return false;
    const Operand* source = op->inputs[0];
    const Operator* dim = op->inputs[1]->producer;
    if (dim->type != "prim::Constant" || dim->params.at("value").type != 2)
        return false;
    int axis = dim->params.at("value").i;
    if (axis < 0)
        axis += (int)source->shape.size();
    if (axis < 0 || axis >= (int)source->shape.size() || source->shape[axis] < 0)
        return false;
    replace_constant(op, source->shape[axis]);
    return true;
}

static bool fold_shape_expression(Operator* op)
{
    if (op->outputs.size() != 1 || op->outputs[0]->type != 0 || op->inputs.size() != 2)
        return false;
    if (op->type != "aten::add" && op->type != "aten::sub" && op->type != "aten::mul" && op->type != "aten::floor_divide")
        return false;
    const Operator* a = op->inputs[0]->producer;
    const Operator* b = op->inputs[1]->producer;
    if (a->type != "prim::Constant" || b->type != "prim::Constant" || a->params.at("value").type != 2 || b->params.at("value").type != 2)
        return false;
    const int64_t x = a->params.at("value").i;
    const int64_t y = b->params.at("value").i;
    int64_t result;
    if (op->type == "aten::add")
        result = x + y;
    else if (op->type == "aten::sub")
        result = x - y;
    else if (op->type == "aten::mul")
        result = x * y;
    else
    {
        if (y <= 0)
            return false;
        result = x / y - (x % y < 0 ? 1 : 0);
    }
    if (result < INT_MIN || result > INT_MAX)
        return false;
    replace_constant(op, (int)result);
    return true;
}

int fold_pt2_constants(Graph& graph, const std::map<Operator*, std::string>& targets, const std::map<const Operand*, double>& floating_constants)
{
    // Only deterministic value operations are evaluated. In particular, empty
    // allocations, random generators, layout queries and mutations are excluded.
    static const std::set<std::string> foldable = {
        "aten::arange", "aten::linspace", "aten::logspace", "aten::eye",
        "aten::ones", "aten::zeros", "aten::full", "aten::full_like", "aten::scalar_tensor", "aten::new_ones", "aten::new_zeros", "aten::new_full",
        "aten::hann_window", "aten::hamming_window", "aten::blackman_window", "aten::bartlett_window", "aten::kaiser_window",
        "aten::add", "aten::sub", "aten::rsub", "aten::mul", "aten::div", "aten::floor_divide", "aten::remainder", "aten::fmod", "aten::pow",
        "aten::neg", "aten::abs", "aten::sqrt", "aten::exp", "aten::log", "aten::sin", "aten::cos",
        "aten::clamp", "aten::clamp_min", "aten::clamp_max", "aten::where", "aten::masked_fill",
        "aten::eq", "aten::ne", "aten::lt", "aten::le", "aten::gt", "aten::ge",
        "aten::logical_not", "aten::logical_and", "aten::logical_or", "aten::bitwise_not", "aten::bitwise_and", "aten::bitwise_or",
        "aten::tril", "aten::triu", "aten::clone", "aten::_to_copy", "aten::to",
        "aten::reshape", "aten::view", "aten::unsqueeze", "aten::squeeze", "aten::transpose", "aten::permute", "aten::t",
        "aten::slice", "aten::select", "aten::expand", "aten::repeat", "aten::cat", "aten::stack",
        "aten::matmul", "aten::mm", "aten::bmm", "aten::sum", "aten::prod", "aten::mean", "aten::cumsum", "aten::cumprod"
    };

    // Keep alias-producing ancestors of mutations intact for functionize.
    static const std::set<std::string> aliases = {
        "aten::slice", "aten::select", "aten::view", "aten::reshape", "aten::flatten",
        "aten::transpose", "aten::t", "aten::permute", "aten::squeeze", "aten::unsqueeze",
        "aten::alias", "aten::detach", "aten::expand", "aten::as_strided"
    };
    std::set<const Operand*> mutable_inputs;
    for (auto it = graph.ops.rbegin(); it != graph.ops.rend(); ++it)
    {
        const Operator* op = *it;
        bool mutable_value = op->type.size() > 2 && op->type.back() == '_' && op->type[op->type.size() - 2] != '_';
        if (aliases.count(op->type))
            for (const Operand* output : op->outputs)
                mutable_value |= mutable_inputs.count(output) != 0;
        if (mutable_value && !op->inputs.empty())
            mutable_inputs.insert(op->inputs[0]);
    }

    at::NoGradGuard guard;
    for (Operator* op : graph.ops)
    {
        if (op->outputs.size() != 1 || mutable_inputs.count(op->outputs[0]))
            continue;
        if (fold_size(op) || fold_shape_expression(op))
            continue;
        if (!foldable.count(op->type) || !targets.count(op) || op->outputs[0]->type == 0)
            continue;
        std::string target = targets.at(op);
        if (target == "torch.ops.aten.lift_fresh_copy.default")
            target = "torch.ops.aten.clone.default";
        const size_t dot = target.rfind('.');
        if (target.compare(0, 15, "torch.ops.aten.") != 0 || dot == std::string::npos)
            continue;
        const std::string overload = target.substr(dot + 1) == "default" ? "" : target.substr(dot + 1);
        try
        {
            const c10::OperatorHandle handle = c10::Dispatcher::singleton().findSchemaOrThrow(op->type.c_str(), overload.c_str());
            const auto& arguments = handle.schema().arguments();
            if (handle.schema().is_mutable() || arguments.size() != op->inputs.size() || handle.schema().returns().size() != 1)
                continue;
            c10::Stack stack;
            for (size_t i = 0; i < arguments.size(); i++)
            {
                c10::IValue value;
                if (i == 0 && (op->type == "aten::new_full" || op->type == "aten::new_ones" || op->type == "aten::new_zeros")
                        && scalar_type(op->inputs[0]->type) != at::ScalarType::Undefined)
                {
                    // These factories use only the source dtype and device.
                    value = at::empty({0}, at::TensorOptions().dtype(scalar_type(op->inputs[0]->type)));
                }
                else if (!constant_value(op->inputs[i], arguments[i].type(), floating_constants, value))
                    break;
                stack.push_back(value);
            }
            if (stack.size() != arguments.size())
                continue;
            handle.callBoxed(stack);
            if (stack.size() != 1 || !stack[0].isTensor())
                continue;
            at::Tensor tensor = stack[0].toTensor();
            Operand* output = op->outputs[0];
            if (!tensor.defined() || !tensor.device().is_cpu() || tensor.layout() != at::kStrided || tensor.scalar_type() != scalar_type(output->type)
                    || tensor.dim() != (int64_t)output->shape.size())
                continue;
            bool compatible = true;
            for (size_t i = 0; i < output->shape.size(); i++)
                if (tensor.size(i) > INT_MAX || (output->shape[i] >= 0 && tensor.size(i) != output->shape[i]))
                    compatible = false;
            if (!compatible)
                continue;
            tensor = tensor.contiguous();
            Attribute attr;
            attr.type = output->type;
            attr.shape.assign(tensor.sizes().begin(), tensor.sizes().end());
            attr.data.resize(tensor.nbytes());
            if (!attr.data.empty())
                memcpy(attr.data.data(), tensor.const_data_ptr(), attr.data.size());
            for (Operand* input : op->inputs)
                input->remove_consumer(op);
            op->inputs.clear();
            op->inputnames.clear();
            op->params.clear();
            op->type = "pnnx.Attribute";
            output->shape = attr.shape;
            op->attrs["data"] = std::move(attr);
        }
        catch (const c10::Error&)
        {
            // A missing CPU kernel or an unsupported constant argument leaves
            // the original operator available to the regular conversion passes.
        }
    }

    // A float32 Parameter must not change an integer factory's fill value.
    // Static factories used the serialized scalar above; preserve the integer
    // conversion for factories whose sizes still depend on runtime inputs.
    for (Operator* op : graph.ops)
    {
        if (op->outputs.size() != 1 || op->outputs[0]->type < 4 || op->outputs[0]->type > 9)
            continue;
        int fill = -1;
        if (op->type == "aten::scalar_tensor")
            fill = 0;
        if (op->type == "aten::full" || op->type == "aten::full_like" || op->type == "aten::fill" || op->type == "aten::fill_")
            fill = 1;
        if (op->type == "aten::new_full" || op->type == "aten::masked_fill" || op->type == "aten::masked_fill_")
            fill = 2;
        if (fill < 0 && op->type != "aten::arange")
            continue;
        for (size_t i = 0; i < op->inputs.size(); i++)
        {
            if (fill >= 0 && i != (size_t)fill)
                continue;
            const auto original = floating_constants.find(op->inputs[i]);
            if (original == floating_constants.end())
                continue;
            const double value = original->second;
            Parameter& p = op->inputs[i]->producer->params.at("value");
            if (p.type != 3 || value == (double)p.f || !std::isfinite(value))
                continue;
            if (fill >= 0)
            {
                const double integer = op->outputs[0]->type == 9 ? (value != 0 ? 1 : 0) : std::trunc(value);
                if (integer >= INT_MIN && integer <= INT_MAX)
                {
                    p = (int)integer;
                    continue;
                }
                if (integer == std::trunc((double)p.f))
                    continue;
            }
            fprintf(stderr, "pt2 constants: integer factory scalar cannot be represented at %s (%s)\n", op->type.c_str(), op->name.c_str());
            return -1;
        }
    }
    return 0;
}

} // namespace pnnx
