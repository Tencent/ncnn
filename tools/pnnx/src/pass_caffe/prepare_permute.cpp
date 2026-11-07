// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_permute.h"

#include "parameter.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_permute(Operator* op)
{
    if (op->inputs.size() != 1 || op->outputs.size() != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): expected one input and one output, got %zu and %zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size());
        return -1;
    }
    if (op->inputs[0]->shape.empty())
    {
        fprintf(stderr, "caffe layer %s (%s): input shape is unknown; provide inputshape\n", op->name.c_str(), op->type.c_str());
        return -1;
    }
    const std::vector<int> shape = op->inputs[0]->shape;
    op->outputs[0]->shape = shape;
    std::vector<int> order = get_parameter(op, "order", std::vector<int>()).ai;
    std::vector<bool> used(shape.size(), false);
    for (int dim : order)
    {
        if (dim < 0 || dim >= (int)shape.size() || used[dim])
        {
            fprintf(stderr, "caffe layer %s (%s): permutation dimension %d is out of range or repeated for rank %zu\n", op->name.c_str(), op->type.c_str(), dim, shape.size());
            return -1;
        }
        used[dim] = true;
    }
    for (size_t i = 0; i < shape.size(); i++)
    {
        if (!used[i])
            order.push_back(i);
    }
    if (order[0] != 0)
    {
        fprintf(stderr, "caffe layer %s (%s): permutation must keep batch first; first dimension=%d\n", op->name.c_str(), op->type.c_str(), order[0]);
        return -1;
    }
    std::vector<int> out;
    for (int dim : order)
        out.push_back(shape[dim]);
    op->outputs[0]->shape = out;

    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
