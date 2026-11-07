// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_flatten.h"

#include "parameter.h"

#include <limits.h>
#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

static int normalize_axis(int axis, int rank)
{
    return axis < 0 ? axis + rank : axis;
}

int prepare_flatten(Operator* op)
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
    const int axis = normalize_axis(get_int(op, "axis", 1), shape.size());
    const int end = normalize_axis(get_int(op, "end_axis", -1), shape.size());
    if (axis <= 0 || end < axis || end >= (int)shape.size())
    {
        fprintf(stderr, "caffe layer %s (%s): Flatten requires a non-batch inclusive axis range; normalized axis=%d end_axis=%d rank=%zu\n", op->name.c_str(), op->type.c_str(), axis, end, shape.size());
        return -1;
    }
    std::vector<int> out(shape.begin(), shape.begin() + axis);
    int count = 1;
    for (int i = axis; i <= end; i++)
    {
        if (shape[i] <= 0 || count > INT_MAX / shape[i])
        {
            fprintf(stderr, "caffe layer %s (%s): flattened dimensions must be positive and product fit INT_MAX; dimension %d=%d accumulated=%d\n", op->name.c_str(), op->type.c_str(), i, shape[i], count);
            return -1;
        }
        count *= shape[i];
    }
    out.push_back(count);
    out.insert(out.end(), shape.begin() + end + 1, shape.end());
    op->outputs[0]->shape = out;

    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
