// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_crop.h"

#include "parameter.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int lower_crop(Operator* op)
{
    if (op->inputs.empty() || op->outputs.size() != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): expected nonempty inputs and one output, got %zu inputs and %zu outputs\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size());
        return -1;
    }
    const std::vector<int> shape = op->inputs[0]->shape;
    if (shape.size() < 2 || shape.size() > 4 || op->inputs.size() != 2 || op->inputs[1]->shape.size() != shape.size())
    {
        fprintf(stderr, "caffe layer %s (%s): Crop requires two inputs of equal rank 2 through 4; inputs=%zu source_rank=%zu reference_rank=%zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), shape.size(), op->inputs.size() < 2 ? 0u : op->inputs[1]->shape.size());
        return -1;
    }
    int axis = get_int(op, "axis", 2);
    if (axis < 0)
        axis += shape.size();
    const std::vector<int> offsets = get_parameter(op, "offset", std::vector<int>()).ai;
    if (axis <= 0 || axis >= (int)shape.size() || (!offsets.empty() && offsets.size() != 1 && offsets.size() != shape.size() - axis))
    {
        fprintf(stderr, "caffe layer %s (%s): Crop requires non-batch axis and 0, 1 or rank-axis offsets; normalized axis=%d rank=%zu offset_count=%zu\n", op->name.c_str(), op->type.c_str(), axis, shape.size(), offsets.size());
        return -1;
    }
    std::vector<int> dims;
    std::vector<int> starts;
    std::vector<int> ends;
    std::vector<int> out = shape;
    for (int i = axis; i < (int)shape.size(); i++)
    {
        const int offset = offsets.empty() ? 0 : offsets.size() == 1 ? offsets[0] : offsets[i - axis];
        const int size = op->inputs[1]->shape[i];
        if (offset < 0 || size <= 0 || size > shape[i] || offset > shape[i] - size)
        {
            fprintf(stderr, "caffe layer %s (%s): Crop dimension %d requires nonnegative offset and positive size within input; offset=%d size=%d input_size=%d\n", op->name.c_str(), op->type.c_str(), i, offset, size, shape[i]);
            return -1;
        }
        dims.push_back(i);
        starts.push_back(offset);
        ends.push_back(offset + size);
        out[i] = size;
    }
    op->inputs[1]->remove_consumer(op);
    op->inputs.resize(1);
    op->type = "Tensor.slice";
    op->params.clear();
    op->params["dims"] = dims;
    op->params["starts"] = starts;
    op->params["ends"] = ends;
    op->params["steps"] = std::vector<int>(dims.size(), 1);
    op->outputs[0]->shape = out;
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
