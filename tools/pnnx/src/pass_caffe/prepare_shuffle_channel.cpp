// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_shuffle_channel.h"

#include "parameter.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_shuffle_channel(Operator* op)
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
    const int groups = get_int(op, "group", 1);
    if (shape.size() != 4 || groups <= 0 || shape[1] <= 0 || shape[1] % groups)
    {
        fprintf(stderr, "caffe layer %s (%s): ShuffleChannel requires rank 4 and positive channels divisible by positive group; rank=%zu channels=%d group=%d\n", op->name.c_str(), op->type.c_str(), shape.size(), shape.size() > 1 ? shape[1] : -1, groups);
        return -1;
    }
    op->params.clear();
    op->params["groups"] = groups;

    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
