// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_reshape.h"

#include "parameter.h"

#include <limits.h>
#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_reshape(Operator* op)
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
    int axis = get_int(op, "axis", 0);
    if (axis < 0)
        axis += shape.size() + 1;
    int num_axes = get_int(op, "num_axes", -1);
    if (num_axes == -1)
        num_axes = shape.size() - axis;
    if (axis < 0 || axis > (int)shape.size() || num_axes < 0 || num_axes > (int)shape.size() - axis)
    {
        fprintf(stderr, "caffe layer %s (%s): reshape replacement range is invalid; normalized axis=%d num_axes=%d rank=%zu\n", op->name.c_str(), op->type.c_str(), axis, num_axes, shape.size());
        return -1;
    }
    std::vector<int> replacement = get_parameter(op, "shape.dim", std::vector<int>()).ai;
    int inferred = -1;
    long long count = 1;
    for (size_t i = 0; i < replacement.size(); i++)
    {
        if (replacement[i] == 0)
        {
            if (axis + i >= shape.size())
            {
                fprintf(stderr, "caffe layer %s (%s): reshape zero at index %zu cannot copy input dimension %zu from rank %zu\n", op->name.c_str(), op->type.c_str(), i, axis + i, shape.size());
                return -1;
            }
            replacement[i] = shape[axis + i];
        }
        if (replacement[i] == -1)
        {
            if (inferred != -1)
            {
                fprintf(stderr, "caffe layer %s (%s): reshape has multiple inferred dimensions at indices %d and %zu\n", op->name.c_str(), op->type.c_str(), inferred, i);
                return -1;
            }
            inferred = i;
        }
        else if (replacement[i] <= 0)
        {
            fprintf(stderr, "caffe layer %s (%s): reshape dimension %zu must be positive, zero-copy, or a single -1; resolved value=%d\n", op->name.c_str(), op->type.c_str(), i, replacement[i]);
            return -1;
        }
        else
        {
            if (count > LLONG_MAX / replacement[i])
            {
                fprintf(stderr, "caffe layer %s (%s): reshape replacement element count exceeds LLONG_MAX at index %zu (%lld * %d)\n", op->name.c_str(), op->type.c_str(), i, count, replacement[i]);
                return -1;
            }
            count *= replacement[i];
        }
    }
    long long original = 1;
    for (int i = axis; i < axis + num_axes; i++)
    {
        if (shape[i] <= 0 || original > LLONG_MAX / shape[i])
        {
            fprintf(stderr, "caffe layer %s (%s): reshape input dimensions must be positive and product fit LLONG_MAX; dimension %d=%d accumulated=%lld\n", op->name.c_str(), op->type.c_str(), i, shape[i], original);
            return -1;
        }
        original *= shape[i];
    }
    if (inferred >= 0)
    {
        if (original % count || original / count > INT_MAX)
        {
            fprintf(stderr, "caffe layer %s (%s): reshape inferred dimension must be integral and fit INT_MAX; input elements=%lld known replacement elements=%lld\n", op->name.c_str(), op->type.c_str(), original, count);
            return -1;
        }
        replacement[inferred] = (int)(original / count);
    }
    else if (original != count)
    {
        fprintf(stderr, "caffe layer %s (%s): reshape changes element count from %lld to %lld\n", op->name.c_str(), op->type.c_str(), original, count);
        return -1;
    }
    std::vector<int> out(shape.begin(), shape.begin() + axis);
    out.insert(out.end(), replacement.begin(), replacement.end());
    out.insert(out.end(), shape.begin() + axis + num_axes, shape.end());
    if (out.empty() || out[0] != shape[0])
    {
        fprintf(stderr, "caffe layer %s (%s): reshape must retain batch dimension %d; output rank=%zu first dimension=%d\n", op->name.c_str(), op->type.c_str(), shape[0], out.size(), out.empty() ? -1 : out[0]);
        return -1;
    }
    op->outputs[0]->shape = out;

    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
