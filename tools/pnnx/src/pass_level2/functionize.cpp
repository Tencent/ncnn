// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "functionize.h"

#include <algorithm>
#include <functional>
#include <stdio.h>
#include <map>
#include <set>
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

static bool is_untracked_alias_op(const Operator* op)
{
    return op->type == "aten::reshape" || op->type == "aten::reshape_as" || op->type == "aten::flatten"
           || op->type == "aten::expand" || op->type == "aten::expand_as" || op->type == "aten::as_strided"
           || op->type == "aten::diagonal" || op->type == "aten::narrow" || op->type == "aten::unfold"
           || op->type == "aten::split" || op->type == "aten::split_with_sizes" || op->type == "aten::chunk"
           || op->type == "aten::unbind" || op->type == "aten::contiguous";
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

static void normalize_copy(Graph& graph, Operator* op, bool slice)
{
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
    const bool static_shape = std::find(destination->shape.begin(), destination->shape.end(), -1) == destination->shape.end();
    if (static_shape && source->shape != destination->shape)
    {
        Operator* size = graph.new_operator_before("prim::Constant", op->name + "_size", op);
        size->params["value"] = destination->shape;
        Operand* sizes = graph.new_operand(size->name);
        sizes->producer = size;
        size->outputs.push_back(sizes);
        Operator* expand = graph.new_operator_before("aten::expand", op->name + "_broadcast", op);
        expand->inputs = {source, sizes};
        source = tensor_like(graph, expand, destination);
    }
    else if (!static_shape && source != destination && (!slice || source->shape != destination->shape))
    {
        // Slice writeback performs broadcasting itself. Other inverse
        // views need the full destination shape before they run.
        Operator* expand = graph.new_operator_before("aten::expand_as", op->name + "_broadcast", op);
        expand->inputs = {source, destination};
        source = tensor_like(graph, expand, destination);
    }
    for (Operand* input : op->inputs)
        input->remove_consumer(op);
    op->type = "aten::clone";
    op->inputs = {source};
    source->consumers.push_back(op);
    op->outputs[0]->type = source->type;
    op->outputs[0]->shape = source->shape;
    op->inputnames.clear();
    op->params.clear();
    op->params["memory_format"] = Parameter();
}

struct Alias
{
    Operand* parent;
    Operator* op;
    // View arguments are snapshots taken at creation time.
    std::vector<Operand*> args;
};

int functionize(Graph& graph)
{
    if (std::none_of(graph.ops.begin(), graph.ops.end(), [](const Operator* op) {
        return is_inplace_op(op) || op->type == "aten::copy";
    }))
        return 0;

    // Writes replace the root value. Later reads replay aliases on that version.
    const std::vector<Operator*> ops = graph.ops;
    std::map<Operand*, Alias> aliases;
    std::map<Operand*, Alias> containers;
    std::map<Operand*, Operand*> versions;
    std::map<Operand*, Operand*> replayed;
    std::set<Operand*> resolved;
    std::map<Operand*, int> remaining;
    std::map<Operand*, Operand*> roots;
    std::map<Operand*, Operand*> storage;
    std::set<Operand*> uncertain_storage;
    std::map<Operand*, std::vector<std::pair<Operand*, Operand*> > > observations;
    int index = 0;

    for (Operator* op : ops)
        for (Operand* input : op->inputs)
            remaining[input]++;

    auto root_of = [&](Operand* value) {
        return roots.count(value) ? roots.at(value) : value;
    };
    auto storage_of = [&](Operand* value) {
        return storage.count(value) ? storage.at(value) : value;
    };
    auto track = [&](Operand* value, Operand* parent, bool invertible) {
        storage[value] = storage_of(parent);
        roots[value] = invertible ? root_of(parent) : value;
        if (!invertible)
            uncertain_storage.insert(storage[value]);
    };
    std::function<void(Operand*, Operand*)> observe = [&](Operand* owner, Operand* value) {
        while (aliases.count(value) && !aliases.at(value).op)
            value = aliases.at(value).parent;
        if (containers.count(value))
        {
            for (Operand* item : containers.at(value).args)
                observe(owner, item);
        }
        else
            observations[storage_of(value)].push_back({owner, root_of(value)});
    };

    std::function<Operand*(Operand*, Operator*)> resolve = [&](Operand* value, Operator* before) -> Operand* {
        if (resolved.count(value))
            return replayed.at(value);
        if (containers.count(value))
        {
            resolved.insert(value);
            const Alias& container = containers.at(value);
            std::vector<Operand*> args;
            for (Operand* input : container.args)
                args.push_back(resolve(input, before));
            Operand* previous = replayed.at(value);
            if (args == previous->producer->inputs)
                return previous;
            Operator* op = graph.new_operator_before(container.op->type, container.op->name + "_replay_" + std::to_string(index++), before);
            op->inputs = args;
            return replayed[value] = tensor_like(graph, op, value);
        }
        if (!aliases.count(value))
            return versions.count(value) ? versions.at(value) : value;
        resolved.insert(value);
        const Alias& alias = aliases.at(value);
        Operand* parent = resolve(alias.parent, before);
        if (!alias.op)
            return replayed[value] = parent;
        Operand* previous = replayed.at(value);
        if (parent == previous->producer->inputs[0])
            return previous;
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
        for (Operand* input : inputs)
            remaining[input]--;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            Operand* value = resolve(inputs[i], op);
            if (value == inputs[i])
                continue;
            inputs[i]->remove_consumer(op);
            op->inputs[i] = value;
            value->consumers.push_back(op);
        }

        // Containers retain aliases across writes, including nested containers.
        if (op->type == "prim::ListConstruct" || op->type == "prim::TupleConstruct")
        {
            Alias container;
            container.parent = 0;
            container.op = op;
            container.args = inputs;
            containers[op->outputs[0]] = container;
            replayed[op->outputs[0]] = op->outputs[0];
            resolved.insert(op->outputs[0]);
            observe(op->outputs[0], op->outputs[0]);
            continue;
        }

        Operand* container = inputs.size() == 1 ? inputs[0] : 0;
        while (aliases.count(container) && !aliases.at(container).op)
            container = aliases.at(container).parent;
        if ((op->type == "prim::ListUnpack" || op->type == "prim::TupleUnpack")
                && containers.count(container))
        {
            const std::vector<Operand*>& items = containers.at(container).args;
            if (items.size() == op->outputs.size())
            {
                for (size_t i = 0; i < items.size(); i++)
                {
                    Alias alias;
                    alias.parent = items[i];
                    alias.op = 0;
                    aliases[op->outputs[i]] = alias;
                    replayed[op->outputs[i]] = op->outputs[i];
                    resolved.insert(op->outputs[i]);
                    track(op->outputs[i], items[i], true);
                    observe(op->outputs[i], op->outputs[i]);
                }
                continue;
            }
        }

        if ((op->type == "aten::__getitem__" || op->type == "prim::TupleIndex") && inputs.size() == 2)
        {
            Operand* list = inputs[0];
            while (aliases.count(list) && !aliases.at(list).op)
                list = aliases.at(list).parent;
            const Operator* index_op = inputs[1]->producer;
            if (containers.count(list) && index_op->type == "prim::Constant" && index_op->has_param("value") && index_op->params.at("value").type == 2)
            {
                const std::vector<Operand*>& items = containers.at(list).args;
                int item = index_op->params.at("value").i;
                if (item < 0)
                    item += (int)items.size();
                if (item >= 0 && item < (int)items.size())
                {
                    Alias alias;
                    alias.parent = items[item];
                    alias.op = 0;
                    aliases[op->outputs[0]] = alias;
                    track(op->outputs[0], items[item], true);
                    observe(op->outputs[0], op->outputs[0]);
                    continue;
                }
            }
        }

        if (op->type == "aten::detach_")
            op->type = "aten::detach";

        if (is_alias_op(op))
        {
            Alias alias;
            alias.parent = inputs[0];
            alias.op = op;
            alias.args.assign(op->inputs.begin() + 1, op->inputs.end());
            aliases[op->outputs[0]] = alias;
            replayed[op->outputs[0]] = op->outputs[0];
            resolved.insert(op->outputs[0]);
            track(op->outputs[0], inputs[0], true);
            observe(op->outputs[0], op->outputs[0]);
            continue;
        }

        if (op->type == "aten::copy")
        {
            // Existing functional copies do not write back through aliases.
            // Only copies emitted below represent inverse view updates.
            normalize_copy(graph, op, false);
            observe(op->outputs[0], op->outputs[0]);
            continue;
        }

        const bool untracked = is_untracked_alias_op(op)
                               || ((op->type == "prim::ListUnpack" || op->type == "prim::TupleUnpack")
                                   && inputs.size() == 1 && storage.count(inputs[0]));
        for (Operand* output : op->outputs)
        {
            if (!inputs.empty() && (untracked || is_inplace_op(op)))
                track(output, inputs[0], !untracked);
            observe(output, output);
        }

        if (!is_inplace_op(op) || op->inputs.empty() || op->outputs.size() != 1)
            continue;

        if (op->type == "aten::transpose_" || op->type == "aten::t_"
                || op->type == "aten::squeeze_" || op->type == "aten::unsqueeze_"
                || op->type == "aten::resize_" || op->type == "aten::resize_as_"
                || op->type == "aten::set_" || op->type == "aten::as_strided_")
        {
            fprintf(stderr, "functionize: %s changes tensor metadata or storage\n", op->type.c_str());
            return -1;
        }

        Operand* target = inputs[0];
        if (uncertain_storage.count(storage_of(target)))
        {
            // A view that may allocate cannot be replayed as an unconditional
            // alias. It is safe as a local root only while no other view is live.
            for (const auto& observation : observations.at(storage_of(target)))
            {
                if (remaining[observation.first] > 0 && observation.second != root_of(target))
                {
                    fprintf(stderr, "functionize: %s has an unsupported observable alias\n", op->name.c_str());
                    return -1;
                }
            }
        }
        Operand* value = op->outputs[0];
        Operator* after = op;
        if (op->type == "aten::copy_")
        {
            Operand* view = target;
            while (aliases.count(view) && !aliases.at(view).op)
                view = aliases.at(view).parent;
            const bool slice = aliases.count(view) && (aliases.at(view).op->type == "aten::slice" || aliases.at(view).op->type == "aten::select");
            normalize_copy(graph, op, slice);
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
                if (input->producer->type == "prim::Constant" && input->producer->has_param("value"))
                {
                    const int type = input->producer->params.at("value").type;
                    if ((type >= 0 && type <= 7) || type == 10 || type == 11)
                        continue;
                }
                if (input->type == 0 || input->type == destination->type)
                    continue;
                const bool floating = destination->type == 1 || destination->type == 2 || destination->type == 3 || destination->type == 13;
                const bool complex = destination->type >= 10 && destination->type <= 12;
                const bool real = input->type <= 9 || input->type == 13;
                // Number constants in traced graphs can be zero-dimensional
                // tensors. They use scalar promotion, not Tensor promotion.
                if (((floating && real) || complex) && input->shape.empty() && !destination->shape.empty())
                    continue;
                if ((floating || complex) && input->type >= 4 && input->type <= 9)
                    continue;
                if (destination->type == 2 && real)
                    continue;
                if (destination->type == 1 && (input->type == 3 || input->type == 13))
                    continue;
                // Real operands only promote a complex tensor when their
                // component precision is higher. Avoid an unnecessary cast
                // and an extra dependency on the old destination otherwise.
                if (destination->type == 11)
                    continue;
                if (destination->type == 10 && input->type != 2 && input->type != 11)
                    continue;
                if (destination->type == 12 && input->type == 3)
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
                const Operator* dim = alias.args.empty() ? 0 : alias.args[0]->producer;
                if (!dim || dim->type != "prim::Constant" || !dim->has_param("value") || dim->params.at("value").type != 2)
                {
                    fprintf(stderr, "functionize: %s requires a constant dimension for mutation writeback\n", alias.op->name.c_str());
                    return -1;
                }
                std::vector<Operand*> args = {resolve(alias.parent, op), value};
                args.insert(args.end(), alias.args.begin(), alias.args.end());
                root = alias.parent;
                // Keep the writeback region independent of subsequent view rewrites.
                value = append("pnnx.SliceCopy", args, root);
                if (type == "aten::slice")
                {
                    after->inputnames = {"self", "src", "dim", "start", "end", "step"};
                }
                else
                {
                    after->inputnames = {"self", "src", "dim", "select"};
                    after->params["start"] = 0;
                    after->params["end"] = 0;
                    after->params["step"] = 0;
                }
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
        resolved.clear();
        Alias result_alias;
        result_alias.parent = target;
        result_alias.op = 0;
        aliases[op->outputs[0]] = result_alias;
    }
    return 0;
}

} // namespace pnnx
