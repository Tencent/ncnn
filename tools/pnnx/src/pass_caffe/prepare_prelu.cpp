// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_prelu.h"

#include "parameter.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_prelu(Operator* op)
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
    const int channels = get_bool(op, "channel_shared", false) ? 1 : shape.size() > 1 ? shape[1] : 0;
    if (channels <= 0 || op->attrs.size() != 1 || !op->attrs.count("0") || op->attrs.at("0").elemcount() != channels)
    {
        fprintf(stderr, "caffe layer %s (%s): PReLU requires positive channel count and one matching slope blob 0; channels=%d blobs=%zu slope elements=%lld (-1 means missing)\n", op->name.c_str(), op->type.c_str(), channels, op->attrs.size(), op->attrs.count("0") ? (long long)op->attrs.at("0").elemcount() : -1LL);
        return -1;
    }
    op->attrs.at("0").shape = {channels};

    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
