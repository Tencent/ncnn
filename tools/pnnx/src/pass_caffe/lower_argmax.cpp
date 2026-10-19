// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_argmax.h"

#include "parameter.h"

#include <algorithm>
#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int lower_argmax(Operator* op)
{
    if (op->inputs.size() != 1 || op->outputs.size() != 1 || !op->attrs.empty())
    {
        fprintf(stderr, "caffe layer %s (%s): expected one input/output and no blobs; got inputs=%zu outputs=%zu blobs=%zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size(), op->attrs.size());
        return -1;
    }
    const std::vector<int> shape = op->inputs[0]->shape;
    if (shape.empty() || shape.size() > 5 || shape[0] != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): requires rank1..5 and singleton leading dimension; rank=%zu leading=%d\n", op->name.c_str(), op->type.c_str(), shape.size(), shape.empty() ? 0 : shape[0]);
        return -1;
    }
    const int top_k = get_int(op, "top_k", 1);
    const bool values = get_bool(op, "out_max_val", false);
    const bool has_axis = op->has_param("axis");
    int axis = get_int(op, "axis", 0);
    if (axis < 0)
        axis += shape.size();
    if (has_axis && (axis < 0 || axis >= (int)shape.size()))
    {
        fprintf(stderr, "caffe layer %s (%s): normalized axis=%d is outside rank=%zu\n", op->name.c_str(), op->type.c_str(), axis, shape.size());
        return -1;
    }
    int64_t count = 1;
    for (int dim : shape)
    {
        if (dim <= 0 || count > INT_MAX / dim)
        {
            fprintf(stderr, "caffe layer %s (%s): input dimension=%d is nonpositive or element count overflows (preceding count=%lld)\n", op->name.c_str(), op->type.c_str(), dim, (long long)count);
            return -1;
        }
        count *= dim;
    }
    const int length = has_axis ? shape[axis] : (int)count;
    if (top_k <= 0 || top_k > length)
    {
        fprintf(stderr, "caffe layer %s (%s): top_k=%d must be in [1,%d]\n", op->name.c_str(), op->type.c_str(), top_k, length);
        return -1;
    }
    std::vector<int> output_shape;
    if (has_axis)
    {
        output_shape = shape;
        output_shape[axis] = top_k;
    }
    else
    {
        output_shape.resize(std::max((size_t)3, shape.size()), 1);
        output_shape[1] = values ? 2 : 1;
        output_shape[2] = top_k;
    }
    if (has_axis && values && top_k == 1)
    {
        op->type = "torch.amax";
        op->params.clear();
        op->params["dim"] = std::vector<int>{axis};
        op->params["keepdim"] = true;
        if (axis == 0)
        {
            op->type = "F.dropout";
            op->params.clear();
            op->params["p"] = 0.f;
            op->params["training"] = false;
        }
    }
    else
    {
        // the native ArgMax sorts one contiguous flattened instance. An axis
        // view is equivalent only when all other dimensions are singleton
        if (has_axis && count != length)
        {
            fprintf(stderr, "Caffe ArgMax %s: per-axis indices or top-k values with multiple slices are not supported\n", op->name.c_str());
            return -1;
        }
        if (count % 4 != 0)
        {
            fprintf(stderr, "caffe layer %s (%s): existing ncnn ArgMax includes allocation padding in its input length; %lld float32 elements are not aligned to four elements and cannot be converted safely\n", op->name.c_str(), op->type.c_str(), (long long)count);
            return -1;
        }
        op->params.clear();
        op->params["top_k"] = top_k;
        // 0: indices, 1: values, 2: indices followed by values
        op->params["mode"] = values ? (has_axis ? 1 : 2) : 0;
        op->params["shape"] = output_shape;
    }
    op->outputs[0]->shape = output_shape;
    op->outputs[0]->type = 1;
    op->outputs[0]->params["__batch_index"] = output_shape.size() == 1 ? 233 : 0;
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
