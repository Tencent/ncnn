// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_slice.h"

#include "parameter.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_slice(Graph& graph, Operator* op)
{
    if (op->inputs.size() != 1 || op->outputs.empty() || op->inputs[0]->shape.empty())
    {
        fprintf(stderr, "caffe layer %s (%s): Slice requires one known-shape input and nonempty outputs; inputs=%zu outputs=%zu input_rank=%zu (provide inputshape if unknown)\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size(), op->inputs.empty() ? 0u : op->inputs[0]->shape.size());
        return -1;
    }
    const std::vector<int> shape = op->inputs[0]->shape;
    int axis = get_int(op, "axis", get_int(op, "slice_dim", 1));
    if (axis < 0)
        axis += shape.size();
    if (axis <= 0 || axis >= (int)shape.size() || shape[axis] <= 0)
    {
        fprintf(stderr, "caffe layer %s (%s): Slice requires positive non-batch dimension; normalized axis=%d rank=%zu dimension=%d\n", op->name.c_str(), op->type.c_str(), axis, shape.size(), axis >= 0 && axis < (int)shape.size() ? shape[axis] : -1);
        return -1;
    }
    const std::vector<int> points = get_parameter(op, "slice_point", std::vector<int>()).ai;
    std::vector<int> sections;
    if (points.empty())
    {
        if (shape[axis] % op->outputs.size())
        {
            fprintf(stderr, "caffe layer %s (%s): slice axis size %d is not divisible by output count %zu\n", op->name.c_str(), op->type.c_str(), shape[axis], op->outputs.size());
            return -1;
        }
        sections.resize(op->outputs.size(), shape[axis] / op->outputs.size());
    }
    else
    {
        if (points.size() + 1 != op->outputs.size())
        {
            fprintf(stderr, "caffe layer %s (%s): slice_point count %zu requires %zu outputs, got %zu\n", op->name.c_str(), op->type.c_str(), points.size(), points.size() + 1, op->outputs.size());
            return -1;
        }
        int previous = 0;
        for (int point : points)
        {
            if (point <= previous || point >= shape[axis])
            {
                fprintf(stderr, "caffe layer %s (%s): slice_point %d must be greater than previous %d and below axis size %d\n", op->name.c_str(), op->type.c_str(), point, previous, shape[axis]);
                return -1;
            }
            sections.push_back(point - previous);
            previous = point;
        }
        sections.push_back(shape[axis] - previous);
    }
    op->params.clear();
    op->params["dim"] = axis;
    op->params["split_size_or_sections"] = sections;
    for (size_t i = 0; i < sections.size(); i++)
    {
        op->outputs[i]->shape = shape;
        op->outputs[i]->shape[axis] = sections[i];
    }
    Operator* unpack = graph.new_operator_after("prim::ListUnpack", op->name + "_unpack", op);
    unpack->outputs = op->outputs;
    for (Operand* output : unpack->outputs)
        output->producer = unpack;
    Operand* list = graph.new_operand(op->name + "_list");
    list->producer = op;
    list->consumers.push_back(unpack);
    unpack->inputs.push_back(list);
    op->outputs = {list};
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
