// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_slice_copy.h"

#include <algorithm>
#include <vector>

namespace pnnx {

static bool is_full_slice(const Operator* op)
{
    if (!op->has_param("dim") || !op->has_param("start") || !op->has_param("end") || !op->has_param("step"))
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
        op->inputs[0] = cloned;
        parent->remove_consumer(op);
        cloned->consumers.push_back(op);
    }
}

} // namespace pnnx
