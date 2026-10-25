// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_scale.h"

#include "parameter.h"

#include <limits.h>
#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

static int prepare_channel_scale(Operator* op)
{
    const std::vector<int>& shape = op->inputs[0]->shape;
    const bool bias = get_bool(op, "bias_term", false);
    if (shape.size() != 4)
    {
        fprintf(stderr, "caffe layer %s (%s): channel affine map requires rank 4, got %zu\n", op->name.c_str(), op->type.c_str(), shape.size());
        return -1;
    }
    if (shape[1] <= 0)
    {
        fprintf(stderr, "caffe layer %s (%s): channel dimension must be positive, got %d\n", op->name.c_str(), op->type.c_str(), shape[1]);
        return -1;
    }
    if (get_int(op, "axis", 1) != 1 || get_int(op, "num_axes", 1) != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): channel affine map requires axis=1 and num_axes=1, got %d and %d\n", op->name.c_str(), op->type.c_str(), get_int(op, "axis", 1), get_int(op, "num_axes", 1));
        return -1;
    }
    const int channels = shape[1];
    if (op->attrs.size() != (bias ? 2u : 1u))
    {
        fprintf(stderr, "caffe layer %s (%s): bias_term=%d requires %d affine blobs, got %zu\n", op->name.c_str(), op->type.c_str(), bias, bias ? 2 : 1, op->attrs.size());
        return -1;
    }
    if (!op->attrs.count("0"))
    {
        fprintf(stderr, "caffe layer %s (%s): missing scale blob 0\n", op->name.c_str(), op->type.c_str());
        return -1;
    }
    if (op->attrs.at("0").elemcount() != channels)
    {
        fprintf(stderr, "caffe layer %s (%s): scale blob 0 has %lld elements, expected %d channels\n", op->name.c_str(), op->type.c_str(), (long long)op->attrs.at("0").elemcount(), channels);
        return -1;
    }
    Attribute weight = op->attrs.at("0");
    weight.shape = {channels};
    Attribute bias_data({channels}, std::vector<float>(channels, 0.f));
    if (bias)
    {
        if (!op->attrs.count("1"))
        {
            fprintf(stderr, "caffe layer %s (%s): bias_term=true requires bias blob 1\n", op->name.c_str(), op->type.c_str());
            return -1;
        }
        if (op->attrs.at("1").elemcount() != channels)
        {
            fprintf(stderr, "caffe layer %s (%s): bias blob 1 has %lld elements, expected %d channels\n", op->name.c_str(), op->type.c_str(), (long long)op->attrs.at("1").elemcount(), channels);
            return -1;
        }
        bias_data = op->attrs.at("1");
        bias_data.shape = {channels};
    }
    // keep the validated channel affine map for the level2 rewrite
    op->params.clear();
    op->params["num_features"] = channels;
    op->attrs.clear();
    op->attrs["0"] = weight;
    op->attrs["1"] = bias_data;
    return 0;
}

static int lower_broadcast_scale(Graph& graph, Operator* op)
{
    const std::vector<int>& shape = op->inputs[0]->shape;
    int axis = get_int(op, "axis", 1);
    if (axis < 0)
        axis += (int)shape.size();
    int num_axes = get_int(op, "num_axes", 1);
    const bool dynamic = op->inputs.size() == 2;
    const bool bias = get_bool(op, "bias_term", false);
    if (dynamic)
        num_axes = op->inputs[1]->shape.size();
    if (num_axes == -1)
        num_axes = shape.size() - axis;
    if (axis < 0 || axis > (int)shape.size())
    {
        fprintf(stderr, "caffe layer %s (%s): axis %d normalizes to %d outside input rank %zu\n", op->name.c_str(), op->type.c_str(), get_int(op, "axis", 1), axis, shape.size());
        return -1;
    }
    if (axis == 0 && num_axes != 0)
    {
        fprintf(stderr, "caffe layer %s (%s): scaling the batch axis is unsupported (axis=0, num_axes=%d)\n", op->name.c_str(), op->type.c_str(), num_axes);
        return -1;
    }
    if (num_axes < 0 || num_axes > (int)shape.size() - axis)
    {
        fprintf(stderr, "caffe layer %s (%s): effective num_axes=%d exceeds available axes %zu from axis %d\n", op->name.c_str(), op->type.c_str(), num_axes, shape.size() - axis, axis);
        return -1;
    }
    std::vector<int> broadcast(shape.size(), 1);
    int count = 1;
    for (int i = 0; i < num_axes; i++)
    {
        if (shape[axis + i] <= 0)
        {
            fprintf(stderr, "caffe layer %s (%s): input dimension %d must be positive, got %d\n", op->name.c_str(), op->type.c_str(), axis + i, shape[axis + i]);
            return -1;
        }
        if (count > INT_MAX / shape[axis + i])
        {
            fprintf(stderr, "caffe layer %s (%s): scale element count exceeds INT_MAX (%d * %d)\n", op->name.c_str(), op->type.c_str(), count, shape[axis + i]);
            return -1;
        }
        if (dynamic && op->inputs[1]->shape[i] != shape[axis + i])
        {
            fprintf(stderr, "caffe layer %s (%s): scale input dimension %d is %d, expected data dimension %d size %d\n", op->name.c_str(), op->type.c_str(), i, op->inputs[1]->shape[i], axis + i, shape[axis + i]);
            return -1;
        }
        broadcast[axis + i] = shape[axis + i];
        count *= broadcast[axis + i];
    }
    if (op->attrs.size() != (dynamic ? (bias ? 1u : 0u) : (bias ? 2u : 1u)))
    {
        fprintf(stderr, "caffe layer %s (%s): dynamic_scale=%d bias_term=%d requires %d blobs, got %zu\n", op->name.c_str(), op->type.c_str(), dynamic, bias, dynamic ? (bias ? 1 : 0) : (bias ? 2 : 1), op->attrs.size());
        return -1;
    }
    for (size_t i = 0; i < op->attrs.size(); i++)
    {
        if (!op->attrs.count(std::to_string(i)))
        {
            fprintf(stderr, "caffe layer %s (%s): missing affine blob %zu\n", op->name.c_str(), op->type.c_str(), i);
            return -1;
        }
        if (op->attrs.at(std::to_string(i)).elemcount() != count)
        {
            fprintf(stderr, "caffe layer %s (%s): affine blob %zu has %lld elements, expected %d\n", op->name.c_str(), op->type.c_str(), i, (long long)op->attrs.at(std::to_string(i)).elemcount(), count);
            return -1;
        }
    }
    Operand* input = op->inputs[0];
    Operand* weight;
    if (dynamic)
    {
        // a scale tensor has parameter axes, not an image batch axis
        Operand* scale = op->inputs[1];
        scale->params["__batch_index"] = 233;
        Operator* reshape = graph.new_operator_before("Tensor.reshape", op->name + "_reshape", op);
        reshape->params["shape"] = std::vector<int>(broadcast.begin() + 1, broadcast.end());
        reshape->inputs = {scale};
        scale->consumers.push_back(reshape);
        weight = graph.new_operand(op->name + "_weight");
        reshape->outputs = {weight};
        weight->producer = reshape;
        weight->shape = reshape->params.at("shape").ai;
        weight->type = scale->type;
        weight->params["__batch_index"] = 233;
    }
    else
    {
        Operator* attr = graph.new_operator_before("pnnx.Attribute", op->name + "_weight", op);
        attr->attrs["data"] = op->attrs.at("0");
        attr->attrs["data"].shape = broadcast;
        weight = graph.new_operand(op->name + "_weight");
        attr->outputs = {weight};
        weight->producer = attr;
        weight->shape = broadcast;
        weight->type = attr->attrs.at("data").type;
        weight->params["__batch_index"] = 0;
    }
    Operand* product = input;
    Operand* bias_data = 0;
    if (bias)
    {
        Operator* mul = graph.new_operator_before("aten::mul", op->name + "_mul", op);
        mul->inputs = {input, weight};
        input->consumers.push_back(mul);
        weight->consumers.push_back(mul);
        product = graph.new_operand(op->name + "_scaled");
        mul->outputs = {product};
        product->producer = mul;
        product->shape = shape;
        product->type = input->type;
        product->params = input->params;

        Operator* attr = graph.new_operator_before("pnnx.Attribute", op->name + "_bias", op);
        attr->attrs["data"] = op->attrs.at(dynamic ? "0" : "1");
        attr->attrs["data"].shape = broadcast;
        bias_data = graph.new_operand(op->name + "_bias");
        attr->outputs = {bias_data};
        bias_data->producer = attr;
        bias_data->shape = broadcast;
        bias_data->type = attr->attrs.at("data").type;
        bias_data->params["__batch_index"] = 0;
    }
    for (Operand* x : op->inputs)
        x->remove_consumer(op);
    op->inputs = bias ? std::vector<Operand*> {product, bias_data} :
                 std::vector<Operand*> {input, weight};
    for (Operand* x : op->inputs)
        x->consumers.push_back(op);
    op->type = bias ? "aten::add" : "aten::mul";
    op->params.clear();
    op->attrs.clear();
    return 0;
}

int lower_scale(Graph& graph, Operator* op)
{
    if (op->inputs.empty() || op->outputs.size() != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): expected at least one input and one output, got %zu and %zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size());
        return -1;
    }
    if (op->inputs[0]->shape.empty())
    {
        fprintf(stderr, "caffe layer %s (%s): input shape is unknown; provide inputshape\n", op->name.c_str(), op->type.c_str());
        return -1;
    }
    const std::vector<int>& shape = op->inputs[0]->shape;
    op->outputs[0]->shape = shape;
    if (op->type == "caffe.BN")
    {
        if (op->inputs.size() != 1)
        {
            fprintf(stderr, "caffe layer %s (%s): BN requires one input, got %zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size());
            return -1;
        }
        op->params["axis"] = 1;
        op->params["num_axes"] = 1;
        op->params["bias_term"] = true;
        if (shape.size() == 2 || shape.size() == 3)
            return lower_broadcast_scale(graph, op);
        return prepare_channel_scale(op);
    }
    if (op->inputs.size() == 1 && shape.size() == 4 && get_int(op, "axis", 1) == 1 && get_int(op, "num_axes", 1) == 1)
        return prepare_channel_scale(op);
    if (op->inputs.size() <= 2)
        return lower_broadcast_scale(graph, op);
    fprintf(stderr, "caffe layer %s (%s): Scale supports one or two inputs, got %zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size());
    return -1;
}

} // namespace caffe2pnnx

} // namespace pnnx
