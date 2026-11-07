// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_reorg.h"

#include "parameter.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_reorg(Operator* op)
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
    const int stride = get_int(op, "stride", 1);
    if (shape.size() != 4 || stride <= 0 || shape[2] <= 0 || shape[3] <= 0 || shape[2] % stride || shape[3] % stride)
    {
        fprintf(stderr, "caffe layer %s (%s): Reorg requires rank 4 and positive spatial dimensions divisible by positive stride; rank=%zu height=%d width=%d stride=%d\n", op->name.c_str(), op->type.c_str(), shape.size(), shape.size() > 2 ? shape[2] : -1, shape.size() > 3 ? shape[3] : -1, stride);
        return -1;
    }
    op->params.clear();
    op->params["downscale_factor"] = stride;
    op->outputs[0]->shape = {shape[0], shape[1] * stride * stride, shape[2] / stride, shape[3] / stride};

    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
