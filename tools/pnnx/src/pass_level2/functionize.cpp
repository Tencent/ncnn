// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "functionize.h"

#include <algorithm>
#include <functional>
#include <stdio.h>
#include <map>
#include <vector>

namespace pnnx {

static bool is_alias_op(const Operator* op)
{
    return op->type == "aten::slice" || op->type == "aten::select"
           || op->type == "aten::view"
           || op->type == "aten::transpose" || op->type == "aten::t" || op->type == "aten::permute"
           || op->type == "aten::squeeze" || op->type == "aten::unsqueeze"
           || op->type == "aten::detach" || op->type == "aten::alias";
}

static bool is_inplace_op(const Operator* op)
{
    const std::string& type = op->type;
    return type.size() > 2 && type[type.size() - 2] != '_' && type[type.size() - 1] == '_';
}

static Operand* tensor_like(Graph& graph, Operator* op, const Operand* meta)
{
    Operand* out = graph.new_operand(op->name + "_out");
    out->producer = op;
    out->type = meta->type;
    out->shape = meta->shape;
    out->params = meta->params;
    op->outputs.push_back(out);
    for (Operand* input : op->inputs)
        input->consumers.push_back(op);
    return out;
}

struct Alias
{
    Operand* parent;
    Operator* op;
    // The non-tensor arguments are values at view creation time. They must not
    // be recomputed from a later tensor version when the alias is replayed.
    std::vector<Operand*> args;
};

int functionize(Graph& graph)
{
    // Keep storage identity separate from SSA values. Each write creates a new
    // root value; every later observation replays its own view chain from that
    // version. In particular, every input (including tuple/list outputs) needs
    // replay, not just inputs[0]. Earlier reads and snapshots remain unchanged.
    const std::vector<Operator*> ops = graph.ops;
    std::map<Operand*, Alias> aliases;
    std::map<Operand*, Alias> containers;
    std::map<Operand*, Operand*> versions;
    std::map<Operand*, Operand*> replayed;
    int index = 0;

    std::function<Operand*(Operand*, Operator*)> resolve = [&](Operand* value, Operator* before) -> Operand* {
        if (replayed.count(value))
            return replayed.at(value);
        if (containers.count(value))
        {
            const Alias& container = containers.at(value);
            std::vector<Operand*> args;
            for (Operand* input : container.args)
                args.push_back(resolve(input, before));
            Operator* op = graph.new_operator_before(container.op->type, container.op->name + "_replay_" + std::to_string(index++), before);
            op->inputs = args;
            return replayed[value] = tensor_like(graph, op, value);
        }
        if (!aliases.count(value))
            return versions.count(value) ? versions.at(value) : value;
        const Alias& alias = aliases.at(value);
        Operand* parent = resolve(alias.parent, before);
        if (!alias.op)
            return replayed[value] = parent;
        Operator* op = graph.new_operator_before(alias.op->type, alias.op->name + "_replay_" + std::to_string(index++), before);
        op->params = alias.op->params;
        op->attrs = alias.op->attrs;
        op->inputnames = alias.op->inputnames;
        op->inputs.push_back(parent);
        op->inputs.insert(op->inputs.end(), alias.args.begin(), alias.args.end());
        return replayed[value] = tensor_like(graph, op, value);
    };

    for (Operator* op : ops)
    {
        const std::vector<Operand*> inputs = op->inputs;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            Operand* value = resolve(inputs[i], op);
            if (value == inputs[i])
                continue;
            inputs[i]->remove_consumer(op);
            op->inputs[i] = value;
            value->consumers.push_back(op);
        }

        // Containers hold tensor aliases, not snapshots of their values. A
        // list/tuple constructed before a write must observe the updated values
        // when consumed later (including nested output containers).
        if (op->type == "prim::ListConstruct" || op->type == "prim::TupleConstruct")
        {
            Alias container;
            container.parent = 0;
            container.op = op;
            container.args = inputs;
            containers[op->outputs[0]] = container;
            replayed[op->outputs[0]] = op->outputs[0];
            continue;
        }

        if ((op->type == "prim::ListUnpack" || op->type == "prim::TupleUnpack")
                && inputs.size() == 1 && containers.count(inputs[0]))
        {
            const std::vector<Operand*>& items = containers.at(inputs[0]).args;
            if (items.size() == op->outputs.size())
            {
                for (size_t i = 0; i < items.size(); i++)
                {
                    Alias alias;
                    alias.parent = items[i];
                    alias.op = 0;
                    aliases[op->outputs[i]] = alias;
                    replayed[op->outputs[i]] = op->outputs[i];
                }
                continue;
            }
        }

        if (is_alias_op(op))
        {
            Alias alias;
            alias.parent = inputs[0];
            alias.op = op;
            alias.args.assign(op->inputs.begin() + 1, op->inputs.end());
            aliases[op->outputs[0]] = alias;
            replayed[op->outputs[0]] = op->outputs[0];
            continue;
        }

        if (!is_inplace_op(op) || op->inputs.empty() || op->outputs.size() != 1)
            continue;

        Operand* target = inputs[0];
        Operand* value = op->outputs[0];
        Operator* after = op;
        if (op->type == "aten::copy_")
        {
            // copy_ returns a value with the destination's dtype and shape,
            // regardless of source broadcasting or promotion. Establish this
            // before reversing any views, including for TorchScript inputs.
            Operand* destination = op->inputs[0];
            Operand* source = op->inputs[1];
            if (source->type == 0 || source->type != destination->type)
            {
                Operator* cast = graph.new_operator_before("aten::type_as", op->name + "_cast", op);
                cast->inputs = {source, destination};
                Operand* converted = tensor_like(graph, cast, source);
                converted->type = destination->type;
                source = converted;
            }
            if (source->shape != destination->shape
                    || std::find(destination->shape.begin(), destination->shape.end(), -1) != destination->shape.end())
            {
                Operator* expand = graph.new_operator_before("aten::expand_as", op->name + "_broadcast", op);
                expand->inputs = {source, destination};
                source = tensor_like(graph, expand, destination);
            }
            for (Operand* input : op->inputs)
                input->remove_consumer(op);
            op->type = "aten::clone";
            op->inputs = {source};
            source->consumers.push_back(op);
            op->inputnames.clear();
            op->params.clear();
            op->params["memory_format"] = Parameter();
        }
        else
        {
            op->type.resize(op->type.size() - 1);
            // Functional arithmetic may promote mixed Tensor operands, whereas
            // an in-place operation stores the result in the destination dtype.
            bool mixed_dtype = false;
            const Operand* destination = op->inputs[0];
            for (size_t i = 1; i < op->inputs.size(); i++)
            {
                const Operand* input = op->inputs[i];
                if (input->type == 0 || input->type == destination->type)
                    continue;
                const bool floating = destination->type == 1 || destination->type == 2 || destination->type == 3 || destination->type == 13;
                const bool real = input->type <= 9 || input->type == 13;
                // Number constants in traced graphs can be zero-dimensional
                // tensors. They use scalar promotion, not Tensor promotion.
                if (floating && real && input->shape.empty() && !destination->shape.empty())
                    continue;
                if (floating && input->type >= 4 && input->type <= 9)
                    continue;
                if (destination->type == 2 && real)
                    continue;
                if (destination->type == 1 && (input->type == 3 || input->type == 13))
                    continue;
                if (destination->type >= 4 && destination->type <= 8 && input->type >= 4 && input->type <= 9
                        && input->shape.empty() && !destination->shape.empty())
                    continue;
                mixed_dtype = true;
            }
            if (mixed_dtype)
            {
                Operator* cast = graph.new_operator_after("aten::type_as", op->name + "_cast", op);
                cast->inputs = {value, op->inputs[0]};
                Operand* converted = tensor_like(graph, cast, value);
                // The original metadata describes the in-place result, not
                // the promoted functional intermediate.
                value->type = 0;
                value = converted;
                after = cast;
            }
        }

        auto append = [&](const std::string& type, const std::vector<Operand*>& args, const Operand* meta) {
            Operator* inverse = graph.new_operator_after(type, op->name + "_writeback_" + std::to_string(index++), after);
            inverse->inputs = args;
            after = inverse;
            return tensor_like(graph, inverse, meta);
        };

        Operand* root = target;
        while (aliases.count(root))
        {
            const Alias& alias = aliases.at(root);
            if (!alias.op)
            {
                root = alias.parent;
                continue;
            }
            const std::string& type = alias.op->type;
            if (type == "aten::slice" || type == "aten::select")
            {
                // Adjacent slice/select views are reversed together by
                // fuse_slice_copy, including nested slices of the same axis.
                Operand* view = resolve(root, op);
                do
                {
                    root = aliases.at(root).parent;
                } while (aliases.count(root) && aliases.at(root).op
                         && (aliases.at(root).op->type == "aten::slice" || aliases.at(root).op->type == "aten::select"));
                value = append("aten::copy", {view, value}, root);
                continue;
            }
            Operand* parent = resolve(alias.parent, op);
            if (type == "aten::view" || type == "aten::squeeze" || type == "aten::unsqueeze")
            {
                value = append("aten::reshape_as", {value, parent}, alias.parent);
            }
            else if (type == "aten::transpose" || type == "aten::t")
            {
                std::vector<Operand*> args = {value};
                args.insert(args.end(), alias.args.begin(), alias.args.end());
                value = append(type, args, alias.parent);
                after->params = alias.op->params;
                after->inputnames = alias.op->inputnames;
            }
            else if (type == "aten::permute")
            {
                // Inverse permutations need a literal, valid dimension order.
                // Scripted models may compute it dynamically; do not guess or
                // let malformed dimensions escape as an uncaught exception.
                std::vector<int> dims;
                bool known = alias.args.size() == 1;
                if (known)
                {
                    const Operator* order = alias.args[0]->producer;
                    if (order->type == "prim::Constant" && order->has_param("value") && order->params.at("value").type == 5)
                        dims = order->params.at("value").ai;
                    else if (order->type == "prim::ListConstruct")
                    {
                        for (Operand* dim : order->inputs)
                        {
                            const Operator* producer = dim->producer;
                            if (producer->type != "prim::Constant" || !producer->has_param("value") || producer->params.at("value").type != 2)
                            {
                                known = false;
                                break;
                            }
                            dims.push_back(producer->params.at("value").i);
                        }
                    }
                    else
                        known = false;
                }
                const int rank = (int)dims.size();
                std::vector<int> inverse(rank, -1);
                for (int i = 0; known && i < rank; i++)
                {
                    const int dim = dims[i] < 0 ? dims[i] + rank : dims[i];
                    if (dim < 0 || dim >= rank || inverse[dim] != -1)
                        known = false;
                    else
                        inverse[dim] = i;
                }
                if (!known || (!parent->shape.empty() && parent->shape.size() != dims.size()))
                {
                    fprintf(stderr, "functionize: %s requires a constant valid permutation for mutation writeback\n", alias.op->name.c_str());
                    return -1;
                }
                Operator* constant = graph.new_operator_after("prim::Constant", op->name + "_inverse_dims_" + std::to_string(index++), after);
                constant->params["value"] = inverse;
                Operand* dims_value = graph.new_operand(constant->name);
                dims_value->producer = constant;
                constant->outputs.push_back(dims_value);
                after = constant;
                value = append(type, {value, dims_value}, alias.parent);
            }
            // detach/alias have the identity inverse.
            root = alias.parent;
        }

        versions[root] = value;
        replayed.clear();
        Alias result_alias;
        result_alias.parent = target;
        result_alias.op = 0;
        aliases[op->outputs[0]] = result_alias;
    }
    return 0;
}

} // namespace pnnx
