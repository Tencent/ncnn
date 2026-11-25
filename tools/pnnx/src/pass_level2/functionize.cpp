// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "functionize.h"
#include "torch_alias.h"

#include <algorithm>
#include <stdio.h>
#include <map>
#include <vector>

namespace pnnx {

static Operand* tensor_like(Graph& graph, Operator* op, const Operand* meta)
{
    Operand* out = graph.new_operand(op->name + "_out");
    out->producer = op;
    out->type = meta->type;
    out->shape = meta->shape;
    out->params = meta->params;
    op->outputs.push_back(out);
    for (Operand* input : op->inputs)
    {
        input->consumers.push_back(op);
    }
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
        // skip dynamic slice expansion only for matching recorded shapes
        Operator* expand = graph.new_operator_before("aten::expand_as", op->name + "_broadcast", op);
        expand->inputs = {source, destination};
        source = tensor_like(graph, expand, destination);
    }
    for (Operand* input : op->inputs)
    {
        input->remove_consumer(op);
    }
    op->type = "aten::clone";
    op->inputs = {source};
    source->consumers.push_back(op);
    op->outputs[0]->type = source->type;
    op->outputs[0]->shape = source->shape;
    op->inputnames.clear();
    op->params.clear();
    op->params["memory_format"] = Parameter();
}

static Operator* normalize_inplace(Graph& graph, Operator* op)
{
    Operand* value = op->outputs[0];
    op->type.resize(op->type.size() - 1);
    // cast promoted arithmetic back to the in-place destination dtype
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
        // frontends normalize floating tensors to FP32
        const bool floating = destination->type == 1;
        if (destination->type == 10)
            continue;
        const bool real = input->type > 0 && input->type <= 9;
        // zero-dimensional traced constants follow scalar promotion
        if (floating && real && input->shape.empty() && !destination->shape.empty())
            continue;
        if (floating && input->type >= 4 && input->type <= 9)
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
        tensor_like(graph, cast, value);
        // the promoted intermediate has unknown dtype until the final cast
        value->type = -1;
        return cast;
    }
    return op;
}

struct View
{
    Operand* parent;
    Operator* op;
    // keep parent geometry and view arguments fixed at creation
    std::vector<Operand*> args;
};

struct Observation
{
    Operand* owner;
    Operand* value;
};

struct TensorState
{
    TensorState()
    {
        identity = 0;
        view = 0;
        root = 0;
        storage = 0;
        value = 0;
        replayed = 0;
        generation = -1;
        remaining = 0;
    }

    Operand* identity;
    Operand* view;
    Operand* root;
    Operand* storage;
    Operand* value;
    Operand* replayed;
    int generation;
    int remaining;
};

struct StorageState
{
    StorageState()
    {
        uncertain = false;
    }

    bool uncertain;
    std::vector<Observation> observations;
};

struct Container
{
    Operator* op;
    std::vector<Operand*> args;
};

struct FunctionizeState
{
    FunctionizeState(Graph& g)
        : graph(g), generation(0), index(0)
    {
    }

    Operand* identity_of(Operand* value)
    {
        while (tensors[value].identity)
        {
            value = tensors[value].identity;
        }
        return value;
    }

    Operand* view_of(Operand* value)
    {
        value = identity_of(value);
        return tensors[value].view ? tensors[value].view : value;
    }

    Operand* root_of(Operand* value)
    {
        return tensors[value].root ? tensors[value].root : value;
    }

    Operand* storage_of(Operand* value)
    {
        return tensors[value].storage ? tensors[value].storage : value;
    }

    void track(Operand* value, Operand* parent, bool invertible)
    {
        TensorState& state = tensors[value];
        state.storage = storage_of(parent);
        state.root = invertible ? root_of(parent) : value;
        if (!invertible)
            storages[state.storage].uncertain = true;
    }

    void observe(Operand* owner, Operand* value)
    {
        value = identity_of(value);
        if (containers.count(value))
        {
            for (Operand* item : containers.at(value).args)
            {
                observe(owner, item);
            }
        }
        else
            storages[storage_of(value)].observations.push_back({owner, value});
    }

    void remember(Operand* value)
    {
        tensors[value].replayed = value;
        tensors[value].generation = generation;
    }

    Operand* resolve_view(Operand* value, Operator* before)
    {
        TensorState& state = tensors[value];
        if (!views.count(value))
            return state.value ? state.value : value;
        if (state.generation == generation)
            return state.replayed;
        state.generation = generation;
        const View& view = views.at(value);
        Operand* parent = resolve_view(view.parent, before);
        Operand* previous = state.replayed;
        if (parent == previous->producer->inputs[0])
            return previous;
        Operator* op = graph.new_operator_before(view.op->type, view.op->name + "_replay_" + std::to_string(index++), before);
        op->params = view.op->params;
        op->attrs = view.op->attrs;
        op->inputnames = view.op->inputnames;
        op->inputs.push_back(parent);
        op->inputs.insert(op->inputs.end(), view.args.begin(), view.args.end());
        return state.replayed = tensor_like(graph, op, value);
    }

    Operand* resolve(Operand* value, Operator* before)
    {
        value = identity_of(value);
        if (!containers.count(value))
            return resolve_view(view_of(value), before);
        TensorState& state = tensors[value];
        if (state.generation == generation)
            return state.replayed;
        state.generation = generation;
        const Container& container = containers.at(value);
        std::vector<Operand*> args;
        for (Operand* input : container.args)
        {
            args.push_back(resolve(input, before));
        }
        Operand* previous = state.replayed;
        if (args == previous->producer->inputs)
            return previous;
        Operator* op = graph.new_operator_before(container.op->type, container.op->name + "_replay_" + std::to_string(index++), before);
        op->inputs = args;
        return state.replayed = tensor_like(graph, op, value);
    }

    Graph& graph;
    std::map<Operand*, TensorState> tensors;
    std::map<Operand*, StorageState> storages;
    std::map<Operand*, View> views;
    std::map<Operand*, Container> containers;
    int generation;
    int index;
};

static Operand* append_writeback(FunctionizeState& state, const Operator* op, Operator*& after, const std::string& type, const std::vector<Operand*>& args, const Operand* meta)
{
    Operator* inverse = state.graph.new_operator_after(type, op->name + "_writeback_" + std::to_string(state.index++), after);
    inverse->inputs = args;
    after = inverse;
    return tensor_like(state.graph, inverse, meta);
}

static int writeback_views(FunctionizeState& state, Operator* op, Operator*& after, Operand*& root, Operand*& value)
{
    Graph& graph = state.graph;
    while (state.views.count(root))
    {
        const View& alias = state.views.at(root);
        const std::string& type = alias.op->type;
        if (type == "aten::slice" || type == "aten::select")
        {
            const Operator* dim = alias.args.empty() ? 0 : alias.args[0]->producer;
            if (!dim || dim->type != "prim::Constant" || !dim->has_param("value") || dim->params.at("value").type != 2)
            {
                fprintf(stderr, "pnnx: functionize: %s requires a constant dimension for mutation writeback\n", alias.op->name.c_str());
                return -1;
            }
            std::vector<Operand*> args = {state.resolve_view(alias.parent, op), value};
            args.insert(args.end(), alias.args.begin(), alias.args.end());
            root = alias.parent;
            // keep writeback geometry independent of later view rewrites
            value = append_writeback(state, op, after, "pnnx.SliceCopy", args, root);
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
        Operand* parent = state.resolve_view(alias.parent, op);
        if (type == "aten::view" || type == "aten::squeeze" || type == "aten::unsqueeze")
        {
            value = append_writeback(state, op, after, "aten::reshape_as", {value, parent}, alias.parent);
        }
        else if (type == "aten::transpose" || type == "aten::t")
        {
            std::vector<Operand*> args = {value};
            args.insert(args.end(), alias.args.begin(), alias.args.end());
            value = append_writeback(state, op, after, type, args, alias.parent);
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
                fprintf(stderr, "pnnx: functionize: %s requires a constant valid permutation for mutation writeback\n", alias.op->name.c_str());
                return -1;
            }
            Operator* constant = graph.new_operator_after("prim::Constant", op->name + "_inverse_dims_" + std::to_string(state.index++), after);
            constant->params["value"] = inverse;
            Operand* dims_value = graph.new_operand(constant->name);
            dims_value->producer = constant;
            constant->outputs.push_back(dims_value);
            after = constant;
            value = append_writeback(state, op, after, type, {value, dims_value}, alias.parent);
        }
        // other views have an identity inverse
        root = alias.parent;
    }

    return 0;
}

int functionize(Graph& graph)
{
    bool has_mutation = false;
    for (const Operator* op : graph.ops)
    {
        if (torch_mutation_kind(op->type) != TorchMutationNone || op->type == "aten::copy")
        {
            has_mutation = true;
            break;
        }
    }
    if (!has_mutation)
        return 0;

    // Writes replace the root value. Later reads replay aliases on that version.
    const std::vector<Operator*> ops = graph.ops;
    FunctionizeState state(graph);
    for (Operator* op : ops)
    {
        for (Operand* input : op->inputs)
        {
            state.tensors[input].remaining++;
        }
    }

    for (Operator* op : ops)
    {
        const std::vector<Operand*> inputs = op->inputs;
        for (Operand* input : inputs)
        {
            state.tensors[input].remaining--;
        }
        for (size_t i = 0; i < inputs.size(); i++)
        {
            Operand* value = state.resolve(inputs[i], op);
            if (value == inputs[i])
                continue;
            inputs[i]->remove_consumer(op);
            op->inputs[i] = value;
            value->consumers.push_back(op);
        }

        // retain nested container aliases across writes
        if (op->type == "prim::ListConstruct" || op->type == "prim::TupleConstruct")
        {
            Container container;
            container.op = op;
            container.args = inputs;
            state.containers[op->outputs[0]] = container;
            state.remember(op->outputs[0]);
            state.observe(op->outputs[0], op->outputs[0]);
            continue;
        }

        Operand* container = inputs.size() == 1 ? inputs[0] : 0;
        container = state.identity_of(container);
        if ((op->type == "prim::ListUnpack" || op->type == "prim::TupleUnpack")
                && state.containers.count(container))
        {
            const std::vector<Operand*>& items = state.containers.at(container).args;
            if (items.size() == op->outputs.size())
            {
                for (size_t i = 0; i < items.size(); i++)
                {
                    state.tensors[op->outputs[i]].identity = state.identity_of(items[i]);
                    state.track(op->outputs[i], items[i], true);
                    state.observe(op->outputs[i], op->outputs[i]);
                }
                continue;
            }
        }

        if ((op->type == "aten::__getitem__" || op->type == "prim::TupleIndex") && inputs.size() == 2)
        {
            Operand* list = state.identity_of(inputs[0]);
            const Operator* index_op = inputs[1]->producer;
            if (state.containers.count(list) && index_op->type == "prim::Constant" && index_op->has_param("value") && index_op->params.at("value").type == 2)
            {
                const std::vector<Operand*>& items = state.containers.at(list).args;
                int item = index_op->params.at("value").i;
                if (item < 0)
                    item += (int)items.size();
                if (item >= 0 && item < (int)items.size())
                {
                    state.tensors[op->outputs[0]].identity = state.identity_of(items[item]);
                    state.track(op->outputs[0], items[item], true);
                    state.observe(op->outputs[0], op->outputs[0]);
                    continue;
                }
            }
        }

        const TorchAliasKind alias_kind = torch_alias_kind(op);
        const TorchMutationKind mutation_kind = torch_mutation_kind(op->type);
        if (alias_kind == TorchAliasIdentity)
        {
            if (mutation_kind != TorchMutationNone)
                op->type.resize(op->type.size() - 1);
            state.tensors[op->outputs[0]].identity = state.identity_of(inputs[0]);
            state.track(op->outputs[0], inputs[0], true);
            state.observe(op->outputs[0], op->outputs[0]);
            continue;
        }

        if (alias_kind == TorchAliasView)
        {
            View alias;
            alias.parent = state.view_of(inputs[0]);
            alias.op = op;
            alias.args.assign(op->inputs.begin() + 1, op->inputs.end());
            state.views[op->outputs[0]] = alias;
            state.remember(op->outputs[0]);
            state.track(op->outputs[0], inputs[0], true);
            state.observe(op->outputs[0], op->outputs[0]);
            continue;
        }

        if (op->type == "aten::copy")
        {
            // Existing functional copies do not write back through aliases.
            // Only copies emitted below represent inverse view updates.
            normalize_copy(graph, op, false);
            state.observe(op->outputs[0], op->outputs[0]);
            continue;
        }

        const bool untracked = alias_kind == TorchAliasUntracked
                               || ((op->type == "prim::ListUnpack" || op->type == "prim::TupleUnpack")
                                   && inputs.size() == 1 && state.tensors[inputs[0]].storage);
        for (Operand* output : op->outputs)
        {
            if (!inputs.empty() && (untracked || mutation_kind != TorchMutationNone))
                state.track(output, inputs[0], !untracked);
            state.observe(output, output);
        }

        if (mutation_kind == TorchMutationNone || op->inputs.empty() || op->outputs.size() != 1)
            continue;

        if (mutation_kind == TorchMutationStorage)
        {
            fprintf(stderr, "pnnx: functionize: %s changes tensor metadata or storage\n", op->type.c_str());
            return -1;
        }

        Operand* target = state.view_of(inputs[0]);
        if (state.storages[state.storage_of(target)].uncertain)
        {
            // A view that may allocate cannot be replayed as an unconditional
            // alias. It is safe as a local root only while no other view is live.
            for (const auto& observation : state.storages[state.storage_of(target)].observations)
            {
                if (state.tensors[observation.owner].remaining > 0 && state.root_of(observation.value) != state.root_of(target))
                {
                    fprintf(stderr, "pnnx: functionize: %s has an unsupported observable alias\n", op->name.c_str());
                    return -1;
                }
            }
        }
        Operand* value = op->outputs[0];
        Operator* after = op;
        if (op->type == "aten::copy_")
        {
            const bool slice = state.views.count(target) && (state.views.at(target).op->type == "aten::slice" || state.views.at(target).op->type == "aten::select");
            normalize_copy(graph, op, slice);
        }
        else if (mutation_kind == TorchMutationMetadata)
        {
            // update this object and its identities; existing views keep their geometry
            op->type.resize(op->type.size() - 1);
            View alias;
            alias.parent = target;
            alias.op = op;
            alias.args.assign(op->inputs.begin() + 1, op->inputs.end());
            state.views[value] = alias;
            state.remember(value);
            state.tensors[state.identity_of(inputs[0])].view = value;
            state.tensors[value].identity = state.identity_of(inputs[0]);
            state.generation++;
            continue;
        }
        else
        {
            after = normalize_inplace(graph, op);
            value = after->outputs[0];
        }

        Operand* root = target;
        if (writeback_views(state, op, after, root, value) != 0)
            return -1;

        state.tensors[root].value = value;
        state.generation++;
        state.tensors[op->outputs[0]].identity = state.identity_of(inputs[0]);
    }
    return 0;
}

} // namespace pnnx
