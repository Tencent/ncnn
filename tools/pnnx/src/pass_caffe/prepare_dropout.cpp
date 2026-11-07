// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_dropout.h"

#include "parameter.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_dropout(Operator* op)
{
    if (op->inputs.size() != 1 || op->outputs.size() != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): expected one input and one output, got %zu and %zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size());
        return -1;
    }
    if (!has_known_shape(op->inputs[0]))
    {
        fprintf(stderr, "caffe layer %s (%s): input shape is unknown; provide inputshape\n", op->name.c_str(), op->type.c_str());
        return -1;
    }
    const std::vector<int> shape = op->inputs[0]->shape;
    op->outputs[0]->shape = shape;
    const float ratio = get_float(op, "dropout_ratio", .5f);
    if (ratio < 0.f || ratio >= 1.f)
    {
        fprintf(stderr, "caffe layer %s (%s): dropout_ratio must be in [0,1), got %g\n", op->name.c_str(), op->type.c_str(), ratio);
        return -1;
    }
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
