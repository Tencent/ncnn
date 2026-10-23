// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_pooling.h"

#include "parameter.h"

#include <algorithm>
#include <limits.h>
#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_pooling(Operator* op)
{
    if (op->inputs.size() != 1 || op->outputs.size() != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): expected one input and one output, got %zu and %zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size());
        return -1;
    }
    if (op->inputs[0]->shape.size() != 4)
    {
        fprintf(stderr, "caffe layer %s (%s): expected rank4 input, got rank%zu\n", op->name.c_str(), op->type.c_str(), op->inputs[0]->shape.size());
        return -1;
    }
    const int method = get_int(op, "pool", 0);
    const bool global = get_bool(op, "global_pooling", false);
    if (method != 0 && method != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): pool=%d is unsupported, expected MAX(0) or AVE(1)\n", op->name.c_str(), op->type.c_str(), method);
        return -1;
    }
    std::vector<int> shape = op->inputs[0]->shape;
    const int k = get_int(op, "kernel_size", 0);
    const int s = get_int(op, "stride", 1);
    const int p = get_int(op, "pad", 0);
    std::vector<int> kernel = {get_int(op, "kernel_h", 0), get_int(op, "kernel_w", 0)};
    std::vector<int> stride = {get_int(op, "stride_h", 0), get_int(op, "stride_w", 0)};
    std::vector<int> pad = {get_int(op, "pad_h", 0), get_int(op, "pad_w", 0)};
    bool clipped_window = false;
    for (int i = 0; i < 2; i++)
    {
        if (!kernel[i])
            kernel[i] = k;
        if (!stride[i])
            stride[i] = s;
        if (!pad[i])
            pad[i] = p;
        if (global)
        {
            if (stride[i] != 1 || pad[i] != 0)
            {
                fprintf(stderr, "caffe layer %s (%s): global pooling spatial dimension %d requires stride=1 pad=0, got stride=%d pad=%d\n", op->name.c_str(), op->type.c_str(), i, stride[i], pad[i]);
                return -1;
            }
            kernel[i] = shape[i + 2];
        }
        if (shape[i + 2] <= 0 || kernel[i] <= 0 || stride[i] <= 0 || pad[i] < 0 || pad[i] >= kernel[i])
        {
            fprintf(stderr, "caffe layer %s (%s): invalid spatial dimension %d input=%d kernel=%d stride=%d pad=%d; require positive input/kernel/stride and 0 <= pad < kernel\n", op->name.c_str(), op->type.c_str(), i, shape[i + 2], kernel[i], stride[i], pad[i]);
            return -1;
        }
    }
    for (int i = 0; i < 2; i++)
    {
        // caffe removes the last window if it starts entirely in padding
        const int64_t padded_size = (int64_t)shape[i + 2] + 2 * (int64_t)pad[i];
        const int64_t numerator = padded_size - kernel[i];
        int64_t out = (numerator >= 0 ? (numerator + stride[i] - 1) / stride[i] : numerator / stride[i]) + 1;
        if ((pad[0] || pad[1]) && (out - 1) * stride[i] >= (int64_t)shape[i + 2] + pad[i])
        {
            out--;
            clipped_window = true;
        }
        if (out <= 0 || out > INT_MAX || padded_size > INT_MAX || (out - 1) * stride[i] + kernel[i] > INT_MAX)
        {
            fprintf(stderr, "caffe layer %s (%s): spatial dimension %d output=%lld padded_input=%lld final_window_end=%lld exceeds supported geometry\n", op->name.c_str(), op->type.c_str(), i, (long long)out, (long long)padded_size, (long long)((out - 1) * stride[i] + kernel[i]));
            return -1;
        }
        shape[i + 2] = (int)out;
    }
    if ((int64_t)kernel[0] * kernel[1] > INT_MAX || (int64_t)shape[2] * shape[3] > INT_MAX)
    {
        fprintf(stderr, "caffe layer %s (%s): kernel area %lld or output area %lld exceeds INT_MAX\n", op->name.c_str(), op->type.c_str(), (long long)kernel[0] * kernel[1], (long long)shape[2] * shape[3]);
        return -1;
    }
    const std::vector<int>& input_shape = op->inputs[0]->shape;
    const int tail_h = std::max(0, (shape[2] - 1) * stride[0] + kernel[0] - input_shape[2] - 2 * pad[0]);
    const int tail_w = std::max(0, (shape[3] - 1) * stride[1] + kernel[1] - input_shape[3] - 2 * pad[1]);
    // PyTorch drops fully empty ceil windows even without configured padding;
    // Caffe MAX pooling keeps them and returns -FLT_MAX.
    const bool empty_window = (shape[2] - 1) * stride[0] >= input_shape[2] + pad[0] || (shape[3] - 1) * stride[1] >= input_shape[3] + pad[1];
    const bool explicit_padding = !global && (clipped_window || empty_window || pad[0] > kernel[0] / 2 || pad[1] > kernel[1] / 2 || (method == 1 && (tail_h || tail_w)));

    // prepare Caffe's output geometry once; level2 constructs the pool graph
    op->params.clear();
    op->params["pool"] = method;
    op->params["global_pooling"] = global;
    op->params["kernel_size"] = kernel;
    op->params["stride"] = stride;
    op->params["pad"] = pad;
    op->params["explicit_padding"] = explicit_padding;
    op->attrs.clear();
    if (explicit_padding)
    {
        op->params["padding"] = std::vector<int> {pad[1], pad[1] + tail_w, pad[0], pad[0] + tail_h};
        if (method == 1)
        {
            // include configured padding, but exclude the extra ceiled tail
            std::vector<float> correction(shape[2] * shape[3]);
            for (int y = 0; y < shape[2]; y++)
            {
                const int start_h = y * stride[0] - pad[0];
                const int divisor_h = std::min(start_h + kernel[0], input_shape[2] + pad[0]) - start_h;
                for (int x = 0; x < shape[3]; x++)
                {
                    const int start_w = x * stride[1] - pad[1];
                    const int divisor_w = std::min(start_w + kernel[1], input_shape[3] + pad[1]) - start_w;
                    correction[y * shape[3] + x] = (float)(kernel[0] * kernel[1]) / (divisor_h * divisor_w);
                }
            }
            op->attrs["correction"] = Attribute({1, 1, shape[2], shape[3]}, correction);
        }
    }
    op->outputs[0]->shape = shape;
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
