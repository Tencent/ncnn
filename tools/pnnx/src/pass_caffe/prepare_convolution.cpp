// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_convolution.h"

#include "parameter.h"

#include <limits.h>
#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

static std::vector<int> spatial_parameter(const Operator* op, const char* key, const char* hkey, const char* wkey, int value)
{
    std::vector<int> v = get_parameter(op, key, std::vector<int>()).ai;
    if (v.empty())
        v = {value, value};
    else if (v.size() == 1)
        v.push_back(v[0]);
    if (hkey && get_int(op, hkey, 0) != 0)
        v[0] = get_int(op, hkey, 0);
    if (wkey && get_int(op, wkey, 0) != 0)
        v[1] = get_int(op, wkey, 0);
    return v;
}

int prepare_convolution(Operator* op)
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

    const bool transpose = op->type == "caffe.Deconvolution";
    const bool depthwise = op->type == "caffe.ConvolutionDepthwise" || op->type == "caffe.DepthwiseConvolution";
    const int channels = op->inputs[0]->shape[1];
    const int outputs = get_int(op, "num_output", 0);
    const int groups = depthwise ? channels : get_int(op, "group", 1);
    const bool bias = get_bool(op, "bias_term", true);
    const std::vector<int> kernel = spatial_parameter(op, "kernel_size", "kernel_h", "kernel_w", 0);
    const std::vector<int> stride = spatial_parameter(op, "stride", "stride_h", "stride_w", 1);
    const std::vector<int> pad = spatial_parameter(op, "pad", "pad_h", "pad_w", 0);
    const std::vector<int> dilation = spatial_parameter(op, "dilation", 0, 0, 1);
    if (get_int(op, "axis", 1) != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): axis=%d is unsupported, expected 1\n", op->name.c_str(), op->type.c_str(), get_int(op, "axis", 1));
        return -1;
    }
    if (channels <= 0 || outputs <= 0 || groups <= 0 || channels % groups || outputs % groups)
    {
        fprintf(stderr, "caffe layer %s (%s): channels=%d num_output=%d group=%d must be positive with both channel counts divisible by group\n", op->name.c_str(), op->type.c_str(), channels, outputs, groups);
        return -1;
    }
    if (kernel.size() != 2 || stride.size() != 2 || pad.size() != 2 || dilation.size() != 2)
    {
        fprintf(stderr, "caffe layer %s (%s): expected two spatial values, got kernel=%zu stride=%zu pad=%zu dilation=%zu\n", op->name.c_str(), op->type.c_str(), kernel.size(), stride.size(), pad.size(), dilation.size());
        return -1;
    }
    for (int i = 0; i < 2; i++)
        if (kernel[i] <= 0 || stride[i] <= 0 || dilation[i] <= 0 || pad[i] < 0)
        {
            fprintf(stderr, "caffe layer %s (%s): invalid spatial dimension %d kernel=%d stride=%d dilation=%d pad=%d\n", op->name.c_str(), op->type.c_str(), i, kernel[i], stride[i], dilation[i], pad[i]);
            return -1;
        }
    if (op->attrs.size() != (bias ? 2u : 1u) || op->attrs.find("0") == op->attrs.end())
    {
        fprintf(stderr, "caffe layer %s (%s): bias_term=%d requires %d blobs including weight blob 0, got %zu blobs\n", op->name.c_str(), op->type.c_str(), bias, bias ? 2 : 1, op->attrs.size());
        return -1;
    }
    Attribute weight = op->attrs.at("0");
    long long weight_count = outputs;
    for (int dim : std::vector<int> {channels / groups, kernel[0], kernel[1]})
    {
        if (weight_count > LLONG_MAX / dim)
        {
            fprintf(stderr, "caffe layer %s (%s): weight element count overflows for num_output=%d channels=%d group=%d kernel=(%d,%d)\n", op->name.c_str(), op->type.c_str(), outputs, channels, groups, kernel[0], kernel[1]);
            return -1;
        }
        weight_count *= dim;
    }
    if (weight.elemcount() != weight_count)
    {
        fprintf(stderr, "caffe layer %s (%s): weight blob 0 has %lld elements, expected %lld\n", op->name.c_str(), op->type.c_str(), (long long)weight.elemcount(), weight_count);
        return -1;
    }
    weight.shape = transpose ? std::vector<int> {channels, outputs / groups, kernel[0], kernel[1]} :
                   std::vector<int> {outputs, channels / groups, kernel[0], kernel[1]};
    Attribute bias_data;
    if (bias)
    {
        if (op->attrs.find("1") == op->attrs.end() || op->attrs.at("1").elemcount() != outputs)
        {
            fprintf(stderr, "caffe layer %s (%s): bias blob 1 has %lld elements (-1 means missing), expected %d\n", op->name.c_str(), op->type.c_str(), op->attrs.count("1") ? (long long)op->attrs.at("1").elemcount() : -1LL, outputs);
            return -1;
        }
        bias_data = op->attrs.at("1");
        bias_data.shape = {outputs};
    }
    std::vector<int> shape = op->inputs[0]->shape;
    shape[1] = outputs;
    for (int i = 0; i < 2; i++)
    {
        if (shape[i + 2] <= 0)
        {
            fprintf(stderr, "caffe layer %s (%s): input spatial dimension %d must be positive, got %d\n", op->name.c_str(), op->type.c_str(), i, shape[i + 2]);
            return -1;
        }
        const int64_t extent = (int64_t)dilation[i] * (kernel[i] - 1) + 1;
        const int64_t padded = (int64_t)shape[i + 2] + (int64_t)pad[i] * 2;
        if (!transpose && padded < extent)
        {
            fprintf(stderr, "caffe layer %s (%s): spatial dimension %d padded input %lld is smaller than effective kernel %lld\n", op->name.c_str(), op->type.c_str(), i, (long long)padded, (long long)extent);
            return -1;
        }
        const int64_t output_size = transpose ? (int64_t)(shape[i + 2] - 1) * stride[i] - (int64_t)pad[i] * 2 + extent : (padded - extent) / stride[i] + 1;
        if (output_size <= 0 || output_size > INT_MAX)
        {
            fprintf(stderr, "caffe layer %s (%s): spatial dimension %d computed output %lld is outside [1, INT_MAX]\n", op->name.c_str(), op->type.c_str(), i, (long long)output_size);
            return -1;
        }
        shape[i + 2] = (int)output_size;
    }
    // normalize source parameters and weight shapes before level2 rewriting
    // depthwise aliases differ only in their implicit group count
    if (depthwise)
        op->type = "caffe.Convolution";
    op->params.clear();
    op->params["num_output"] = outputs;
    op->params["kernel_size"] = kernel;
    op->params["stride"] = stride;
    op->params["pad"] = pad;
    op->params["dilation"] = dilation;
    op->params["group"] = groups;
    op->params["bias_term"] = bias;
    op->attrs.clear();
    op->attrs["0"] = weight;
    if (bias)
        op->attrs["1"] = bias_data;
    op->outputs[0]->shape = shape;
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
