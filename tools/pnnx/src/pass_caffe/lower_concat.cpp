// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_concat.h"

#include "parameter.h"

#include <limits.h>
#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

static int normalize_axis(int axis, int rank)
{
    return axis < 0 ? axis + rank : axis;
}

int lower_concat(Graph& graph, Operator* op)
{
    if (op->inputs.empty() || op->outputs.size() != 1 || op->inputs[0]->shape.empty())
    {
        fprintf(stderr, "caffe layer %s (%s): Concat requires nonempty inputs, one output and known input shape; inputs=%zu outputs=%zu input_rank=%zu (provide inputshape if unknown)\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size(), op->inputs.empty() ? 0u : op->inputs[0]->shape.size());
        return -1;
    }
    const std::vector<int> shape = op->inputs[0]->shape;
    op->outputs[0]->shape = shape;
    const int axis = normalize_axis(get_int(op, "axis", get_int(op, "concat_dim", 1)), shape.size());
    if (axis <= 0 || axis >= (int)shape.size())
    {
        fprintf(stderr, "caffe layer %s (%s): axis must select a non-batch input dimension; normalized axis=%d rank=%zu\n", op->name.c_str(), op->type.c_str(), axis, shape.size());
        return -1;
    }
    std::vector<int> out = shape;
    out[axis] = 0;
    for (const Operand* input : op->inputs)
    {
        if (input->shape.size() != shape.size())
        {
            fprintf(stderr, "caffe layer %s (%s): input %s rank %zu differs from first input rank %zu\n", op->name.c_str(), op->type.c_str(), input->name.c_str(), input->shape.size(), shape.size());
            return -1;
        }
        for (size_t i = 0; i < shape.size(); i++)
            if ((int)i != axis && input->shape[i] != shape[i])
            {
                fprintf(stderr, "caffe layer %s (%s): input %s dimension %zu is %d, expected %d outside concat axis %d\n", op->name.c_str(), op->type.c_str(), input->name.c_str(), i, input->shape[i], shape[i], axis);
                return -1;
            }
        if (input->shape[axis] <= 0 || out[axis] > INT_MAX - input->shape[axis])
        {
            fprintf(stderr, "caffe layer %s (%s): concat dimension must be positive and total fit INT_MAX; input %s dimension=%d accumulated=%d\n", op->name.c_str(), op->type.c_str(), input->name.c_str(), input->shape[axis], out[axis]);
            return -1;
        }
        out[axis] += input->shape[axis];
    }
    op->type = "aten::cat";
    op->params.clear();
    op->params["dim"] = axis;
    op->outputs[0]->shape = out;
    if (op->inputs.size() == 1)
    {
        op->type = "F.dropout";
        op->params.clear();
        op->params["p"] = 0.f;
        op->params["training"] = false;
        return 0;
    }

    // match the TNN frontend's list input convention for level2 cat
    Operator* list = graph.new_operator_before("prim::ListConstruct", op->name + "_lc", op);
    Operand* r = graph.new_operand(op->name + "_lc");
    r->producer = list;
    r->consumers.push_back(op);
    list->outputs.push_back(r);
    for (Operand* input : op->inputs)
    {
        input->remove_consumer(op);
        input->consumers.push_back(list);
        list->inputs.push_back(input);
    }
    op->inputs.clear();
    op->inputs.push_back(r);
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
