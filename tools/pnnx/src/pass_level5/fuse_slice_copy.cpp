// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_slice_copy.h"

#include <algorithm>
#include <vector>

namespace pnnx {

struct SliceIndex
{
    int dim;
    int start;
    int end;
    int step;
    int select;

    bool operator==(const SliceIndex& rhs) const
    {
        return dim == rhs.dim && start == rhs.start && end == rhs.end && step == rhs.step && select == rhs.select;
    }
};

static bool get_slice_indices(const Operator* op, std::vector<SliceIndex>& indices)
{
    std::vector<SliceIndex> slices;
    if (op->type == "Tensor.select")
    {
        if (op->has_param("dims") || !op->has_param("dim") || !op->has_param("index"))
            return false;
        const Parameter& dim = op->params.at("dim");
        const Parameter& index = op->params.at("index");
        if (dim.type != 2 || index.type != 2)
            return false;
        slices.push_back({dim.i, 0, 0, 0, index.i});
    }
    else if (op->has_param("dims"))
    {
        if (!op->has_param("starts") || !op->has_param("ends") || !op->has_param("steps"))
            return false;
        const Parameter& dims = op->params.at("dims");
        const Parameter& starts = op->params.at("starts");
        const Parameter& ends = op->params.at("ends");
        const Parameter& steps = op->params.at("steps");
        const Parameter* selects = op->has_param("selects") ? &op->params.at("selects") : 0;
        if (dims.type != 5 || starts.type != 5 || ends.type != 5 || steps.type != 5)
            return false;
        if (starts.ai.size() != dims.ai.size() || ends.ai.size() != dims.ai.size() || steps.ai.size() != dims.ai.size())
            return false;
        if (selects ? (selects->type != 5 || selects->ai.size() != dims.ai.size()) : op->has_input("selects"))
            return false;
        for (size_t i = 0; i < dims.ai.size(); i++)
            slices.push_back({dims.ai[i], starts.ai[i], ends.ai[i], steps.ai[i], selects ? selects->ai[i] : INT_MAX});
    }
    else
    {
        if (!op->has_param("dim") || !op->has_param("start") || !op->has_param("end") || !op->has_param("step"))
            return false;
        const Parameter& dim = op->params.at("dim");
        const Parameter& start = op->params.at("start");
        const Parameter& end = op->params.at("end");
        const Parameter& step = op->params.at("step");
        const Parameter* select = op->has_param("select") ? &op->params.at("select") : 0;
        if (dim.type != 2 || (start.type != 0 && start.type != 2) || (end.type != 0 && end.type != 2) || (step.type != 0 && step.type != 2))
            return false;
        if (select ? select->type != 2 : op->has_input("select"))
            return false;
        slices.push_back({dim.i, start.type == 0 ? 0 : start.i, end.type == 0 ? INT_MAX : end.i, step.type == 0 ? 1 : step.i, select ? select->i : INT_MAX});
    }

    const int rank = op->inputs[0]->shape.size();
    for (SliceIndex s : slices)
    {
        if (s.dim < 0)
            s.dim += rank;
        if (s.dim < 0 || s.dim >= rank || (!indices.empty() && s.dim <= indices.back().dim))
            return false;
        if (s.step < 0 || (s.step == 0 && s.select == INT_MAX))
            return false;
        indices.push_back(s);
    }
    return !indices.empty();
}

static bool append_slice_indices(std::vector<SliceIndex>& outer, const std::vector<SliceIndex>& inner)
{
    const std::vector<SliceIndex> parent = outer;
    for (SliceIndex s : inner)
    {
        // Inner axes refer to the view after the outer selects removed axes.
        for (const SliceIndex& p : parent)
            if (p.step == 0 && s.dim >= p.dim)
                s.dim++;

        // Repeated axes need offset/stride composition, not multi-axis indexing.
        if (!outer.empty() && s.dim <= outer.back().dim)
            return false;
        outer.push_back(s);
    }
    return true;
}

static bool is_slice_region(const Operand* view, const Operand* parent, const std::vector<SliceIndex>& region)
{
    std::vector<std::vector<SliceIndex> > chain;
    while (view != parent)
    {
        const Operator* op = view->producer;
        if (!op || (op->type != "Tensor.slice" && op->type != "Tensor.select") || op->inputs.size() != 1 || op->outputs.size() != 1)
            return false;
        std::vector<SliceIndex> indices;
        if (!get_slice_indices(op, indices))
            return false;
        chain.push_back(indices);
        view = op->inputs[0];
    }

    std::vector<SliceIndex> indices;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it)
        if (!append_slice_indices(indices, *it))
            return false;
    return indices == region;
}

static bool is_full_slice(const Operator* op)
{
    if (op->inputs.size() != 2 || !op->has_param("dim") || op->params.at("dim").type != 2 || !op->has_param("start") || !op->has_param("end") || !op->has_param("step"))
        return false;
    const Parameter& start = op->params.at("start");
    const Parameter& end = op->params.at("end");
    const Parameter& step = op->params.at("step");
    if ((start.type != 0 && (start.type != 2 || start.i != 0))
            || (step.type != 0 && (step.type != 2 || step.i != 1)))
        return false;
    if (end.type == 0 || (end.type == 2 && end.i == INT_MAX))
        return true;

    const std::vector<int>& shape = op->inputs[0]->shape;
    int dim = op->params.at("dim").i;
    if (dim < 0)
        dim += (int)shape.size();
    return dim >= 0 && dim < (int)shape.size() && shape[dim] >= 0 && end.type == 2 && end.i >= shape[dim];
}

void fuse_slice_copy(Graph& graph)
{
    const std::vector<Operator*> ops = graph.ops;
    for (auto it = ops.rbegin(); it != ops.rend(); ++it)
    {
        Operator* op = *it;
        if (op->type != "pnnx.SliceCopy" || !is_full_slice(op))
            continue;

        Operand* parent = op->inputs[0];
        Operand* value = op->inputs[1];
        // Equal unknown dimensions do not prove that broadcasting is unnecessary.
        const bool same_shape = !parent->shape.empty() && parent->shape == value->shape
                                && std::find(parent->shape.begin(), parent->shape.end(), -1) == parent->shape.end();
        const bool same_parent = value->producer->type == "pnnx.SliceCopy" && value->producer->inputs[0] == parent;
        if (!same_shape && !same_parent && value != parent)
            continue;

        Operand* out = op->outputs[0];
        for (Operator* consumer : out->consumers)
        {
            for (Operand*& input : consumer->inputs)
                if (input == out)
                    input = value;
            value->consumers.push_back(consumer);
        }
        for (Operand* input : op->inputs)
            input->remove_consumer(op);
        value->name = out->name;
        graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), out));
        delete out;
        graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));
        delete op;
    }

    while (1)
    {
        bool matched = false;
        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];
            if (op->type != "pnnx.SliceCopy" || op->inputs.size() != 2)
                continue;
            Operand* value = op->inputs[1];
            Operator* inner = value->producer;
            if (!inner || inner->type != "pnnx.SliceCopy" || inner->inputs.size() != 2 || inner->outputs.size() != 1 || value->consumers.size() != 1)
                continue;

            std::vector<SliceIndex> outer_indices;
            std::vector<SliceIndex> inner_indices;
            if (!get_slice_indices(op, outer_indices) || !get_slice_indices(inner, inner_indices))
                continue;
            if (!is_slice_region(inner->inputs[0], op->inputs[0], outer_indices))
                continue;
            if (!append_slice_indices(outer_indices, inner_indices))
                continue;

            std::vector<int> dims, starts, ends, steps, selects;
            for (const SliceIndex& s : outer_indices)
            {
                dims.push_back(s.dim);
                starts.push_back(s.start);
                ends.push_back(s.end);
                steps.push_back(s.step);
                selects.push_back(s.select);
            }
            op->params.clear();
            op->params["dims"] = dims;
            op->params["starts"] = starts;
            op->params["ends"] = ends;
            op->params["steps"] = steps;
            op->params["selects"] = selects;
            op->inputs[1] = inner->inputs[1];
            op->inputs[1]->consumers.push_back(op);
            for (Operand* input : inner->inputs)
                input->remove_consumer(inner);
            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), value));
            delete value;
            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), inner));
            delete inner;
            // Dead views are removed by the existing level5 dead-code pass.
            matched = true;
            break;
        }
        if (!matched)
            break;
    }

    const std::vector<Operator*> copies = graph.ops;
    for (Operator* op : copies)
    {
        if (op->type != "pnnx.SliceCopy")
            continue;

        Operand* parent = op->inputs[0];
        Operator* clone = graph.new_operator_before("Tensor.clone", op->name + "_clone", op);
        Operand* cloned = graph.new_operand(clone->name + "_out");
        cloned->type = parent->type;
        cloned->shape = parent->shape;
        cloned->params = parent->params;
        cloned->producer = clone;
        clone->inputs.push_back(parent);
        parent->consumers.push_back(clone);
        clone->outputs.push_back(cloned);

        op->type = "Tensor.slice_copy";
        if (op->has_param("start") && op->params.at("start").type == 0)
            op->params["start"] = 0;
        if (op->has_param("end") && op->params.at("end").type == 0)
            op->params["end"] = INT_MAX;
        op->inputs[0] = cloned;
        parent->remove_consumer(op);
        cloned->consumers.push_back(op);
    }
}

} // namespace pnnx
