// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_tile.h"

#include "parameter.h"

#include <limits.h>
#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_tile(Operator* op)
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
    std::vector<int> shape = op->inputs[0]->shape;
    int axis = get_int(op, "axis", 1);
    const int tiles = get_int(op, "tiles", 0);
    if (axis < 0)
        axis += (int)shape.size();
    if (axis < 0 || axis >= (int)shape.size() || tiles <= 0 || (axis == 0 && tiles != 1))
    {
        fprintf(stderr, "caffe layer %s (%s): Tile requires valid axis, positive tiles and unchanged batch; normalized axis=%d rank=%zu tiles=%d\n", op->name.c_str(), op->type.c_str(), axis, shape.size(), tiles);
        return -1;
    }
    if (shape[axis] <= 0 || shape[axis] > INT_MAX / tiles)
    {
        fprintf(stderr, "caffe layer %s (%s): tiled dimension must be positive and product fit INT_MAX; dimension=%d tiles=%d\n", op->name.c_str(), op->type.c_str(), shape[axis], tiles);
        return -1;
    }
    std::vector<int> sizes(shape.size(), 1);
    sizes[axis] = tiles;
    shape[axis] *= tiles;
    op->params.clear();
    op->params["sizes"] = sizes;
    op->outputs[0]->shape = shape;
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
