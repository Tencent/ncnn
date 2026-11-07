// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_mutation.h"

#include <algorithm>
#include <stdio.h>

namespace pnnx {

static bool is_mutable(const Operator* op)
{
    const std::string& type = op->type;
    return type.size() > 1 && type[type.size() - 1] == '_' && type[type.size() - 2] != '_';
}

static const Operand* alias_root(const Operand* r)
{
    while (r->producer->type == "aten::slice" || r->producer->type == "aten::select" || r->producer->type == "aten::view")
        r = r->producer->inputs[0];
    return r;
}

static const Operand* mutation_root(const Operand* r)
{
    r = alias_root(r);
    while (is_mutable(r->producer))
        r = alias_root(r->producer->inputs[0]);
    return r;
}

static bool check_alias_uses(const Graph& graph)
{
    std::map<const Operator*, size_t> order;
    for (size_t i = 0; i < graph.ops.size(); i++)
        order[graph.ops[i]] = i;
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        const Operator* mutation = graph.ops[i];
        if (!is_mutable(mutation))
            continue;
        const Operand* root = alias_root(mutation->inputs[0]);
        for (size_t j = i + 1; j < graph.ops.size(); j++)
        {
            const Operator* consumer = graph.ops[j];
            for (size_t k = 1; k < consumer->inputs.size(); k++)
            {
                const Operand* input = consumer->inputs[k];
                if (input == root || input == consumer->inputs[0] || order.at(input->producer) >= i)
                    continue;
                if (alias_root(input) == root)
                {
                    // Shared functionize replays the first input's alias chain.
                    // Other old views need independent replay; even an ancestor
                    // of the first view may be split by its per-consumer shadows.
                    fprintf(stderr, "pt2 mutation: mutation has unsupported multi-input alias use\n");
                    return false;
                }
            }
        }
    }
    return true;
}

static Operand* constant_before(Graph& graph, Operator* before, const char* suffix, const Parameter& value)
{
    Operator* op = graph.new_operator_before("prim::Constant", before->name + suffix, before);
    op->params["value"] = value;
    Operand* r = graph.new_operand(op->name);
    r->producer = op;
    op->outputs.push_back(r);
    return r;
}

static Operand* tensor_before(Graph& graph, Operator* before, const char* type, const char* suffix, const std::vector<Operand*>& inputs, const Operand* meta)
{
    Operator* op = graph.new_operator_before(type, before->name + suffix, before);
    op->inputs = inputs;
    for (Operand* input : inputs)
        input->consumers.push_back(op);
    Operand* r = graph.new_operand(op->name);
    r->producer = op;
    r->type = meta->type;
    r->shape = meta->shape;
    op->outputs.push_back(r);
    return r;
}

int normalize_pt2_mutation(Graph& graph)
{
    std::map<const Operator*, size_t> order;
    for (size_t i = 0; i < graph.ops.size(); i++)
        order[graph.ops[i]] = i;

    // copy_ uses its output for root writeback, not its ATen destination-view
    // result. Other mutable operations already produce the correct value; only
    // observations after a later write need to follow the destination alias.
    for (Operator* op : graph.ops)
    {
        if (!is_mutable(op))
            continue;
        if (op->inputs.empty() || op->outputs.size() != 1)
        {
            fprintf(stderr, "pt2 mutation: unsupported mutable signature\n");
            return -1;
        }
        Operand* target = op->inputs[0];
        Operand* result = op->outputs[0];
        for (Operator* consumer : std::vector<Operator*>(result->consumers))
        {
            bool alias_result = op->type == "aten::copy_" || op->type == "aten::fill_";
            // A derived view or another write through this result must retain
            // storage identity, even if its first use precedes the next write.
            if (consumer->type == "aten::slice" || consumer->type == "aten::select" || consumer->type == "aten::view"
                    || (is_mutable(consumer) && consumer->inputs[0] == result))
                alias_result = true;
            for (size_t i = order.at(op) + 1; !alias_result && i < order.at(consumer); i++)
            {
                const Operator* later = graph.ops[i];
                if (is_mutable(later) && mutation_root(later->inputs[0]) == mutation_root(target))
                    alias_result = true;
            }
            if (!alias_result)
                continue;
            for (Operand*& input : consumer->inputs)
            {
                if (input == result)
                    input = target;
            }
            target->consumers.push_back(consumer);
            result->remove_consumer(consumer);
        }
    }

    if (!check_alias_uses(graph))
        return -1;

    for (Operator* op : std::vector<Operator*>(graph.ops))
    {
        if (op->type == "aten::fill" || op->type == "aten::fill_")
        {
            // Allocation and writeback are distinct: functional fill must not
            // modify self, while fill_ still participates in alias propagation.
            const bool inplace = op->type == "aten::fill_";
            std::vector<Operand*> inputs = op->inputs;
            const char* suffixes[] = {"_dtype", "_layout", "_device", "_pin_memory", "_memory_format"};
            for (const char* suffix : suffixes)
                inputs.push_back(constant_before(graph, op, suffix, Parameter()));
            if (inplace)
            {
                Operand* filled = tensor_before(graph, op, "aten::full_like", "_filled", inputs, op->inputs[0]);
                op->inputs[1]->remove_consumer(op);
                op->inputs[1] = filled;
                filled->consumers.push_back(op);
                op->type = "aten::copy_";
            }
            else
            {
                for (size_t i = op->inputs.size(); i < inputs.size(); i++)
                    inputs[i]->consumers.push_back(op);
                op->inputs = inputs;
                op->type = "aten::full_like";
            }
            op->inputnames.clear();
        }
        if (op->type != "aten::copy" && op->type != "aten::copy_")
            continue;
        if (op->inputs.size() < 2 || op->outputs.size() != 1)
        {
            fprintf(stderr, "pt2 copy: unsupported copy signature\n");
            return -1;
        }
        Operand* target = op->inputs[0];
        Operand* source = op->inputs[1];
        static const int types[] = {-1, 6, 7, 5, 3, 4, 2, 1, 0, 11, 9, 10, 8, 15};
        if (target->type < 1 || target->type > 13)
        {
            fprintf(stderr, "pt2 copy: unknown destination dtype\n");
            return -1;
        }
        // Shared copy elimination assumes identical shape and dtype. Establish
        // that contract before functionize adds any slice/select writeback.
        if (source->type != target->type)
        {
            Operand* dtype = constant_before(graph, op, "_dtype", Parameter(types[target->type]));
            Operand* blocking = constant_before(graph, op, "_blocking", Parameter(false));
            Operand* copy = constant_before(graph, op, "_copy", Parameter(false));
            Operand* format = constant_before(graph, op, "_format", Parameter());
            source = tensor_before(graph, op, "aten::to", "_cast", {source, dtype, blocking, copy, format}, source);
            source->type = target->type;
        }
        // Prefer a literal size when known so shared/ncnn expand lowering can
        // eliminate the shape-only dependency on the destination tensor.
        const bool static_shape = std::find_if(target->shape.begin(), target->shape.end(), [](int d) {
            return d < 0;
        })
        == target->shape.end();
        if (static_shape && source->shape != target->shape)
        {
            Operand* size = constant_before(graph, op, "_size", Parameter(target->shape));
            Operand* implicit = constant_before(graph, op, "_implicit", Parameter(false));
            source = tensor_before(graph, op, "aten::expand", "_broadcast", {source, size, implicit}, target);
        }
        else if (!static_shape)
        {
            // Equal symbolic metadata does not prove equal runtime dimensions.
            source = tensor_before(graph, op, "aten::expand_as", "_broadcast", {source, target}, target);
        }
        if (op->type == "aten::copy_")
        {
            op->inputs[1]->remove_consumer(op);
            op->inputs[1] = source;
            source->consumers.push_back(op);
        }
        else
        {
            // Functional copy never writes back through a destination view.
            for (Operand* input : op->inputs)
                input->remove_consumer(op);
            Operand* format = constant_before(graph, op, "_clone_format", Parameter());
            op->type = "aten::clone";
            op->inputs = {source, format};
            for (Operand* input : op->inputs)
                input->consumers.push_back(op);
            op->inputnames.clear();
        }
    }
    return 0;
}

} // namespace pnnx
