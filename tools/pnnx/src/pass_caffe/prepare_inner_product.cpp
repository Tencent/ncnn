// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_inner_product.h"

#include "parameter.h"

#include <limits.h>
#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_inner_product(Graph& graph, Operator* op)
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
    int axis = get_int(op, "axis", 1);
    if (axis < 0)
        axis += (int)shape.size();
    const int outputs = get_int(op, "num_output", 0);
    const bool bias = get_bool(op, "bias_term", true);
    const bool transpose = get_bool(op, "transpose", false);
    if (axis <= 0 || axis >= (int)shape.size())
    {
        fprintf(stderr, "caffe layer %s (%s): axis %d normalizes to %d; expected a non-batch axis below rank %zu\n", op->name.c_str(), op->type.c_str(), get_int(op, "axis", 1), axis, shape.size());
        return -1;
    }
    if (outputs <= 0)
    {
        fprintf(stderr, "caffe layer %s (%s): num_output must be positive, got %d\n", op->name.c_str(), op->type.c_str(), outputs);
        return -1;
    }
    if (op->attrs.size() != (bias ? 2u : 1u))
    {
        fprintf(stderr, "caffe layer %s (%s): bias_term=%d requires %d weight blobs, got %zu\n", op->name.c_str(), op->type.c_str(), bias, bias ? 2 : 1, op->attrs.size());
        return -1;
    }
    if (op->attrs.find("0") == op->attrs.end())
    {
        fprintf(stderr, "caffe layer %s (%s): missing weight blob 0\n", op->name.c_str(), op->type.c_str());
        return -1;
    }
    int features = 1;
    for (size_t i = axis; i < shape.size(); i++)
    {
        if (shape[i] <= 0)
        {
            fprintf(stderr, "caffe layer %s (%s): input dimension %zu must be positive, got %d\n", op->name.c_str(), op->type.c_str(), i, shape[i]);
            return -1;
        }
        if (features > INT_MAX / shape[i])
        {
            fprintf(stderr, "caffe layer %s (%s): flattened feature count exceeds INT_MAX at dimension %zu (%d * %d)\n", op->name.c_str(), op->type.c_str(), i, features, shape[i]);
            return -1;
        }
        features *= shape[i];
    }
    Attribute weight = op->attrs.at("0");
    if (weight.elemcount() != (long long)outputs * features)
    {
        fprintf(stderr, "caffe layer %s (%s): weight blob 0 has %lld elements, expected %lld (%d outputs * %d features)\n", op->name.c_str(), op->type.c_str(), (long long)weight.elemcount(), (long long)outputs * features, outputs, features);
        return -1;
    }
    if (transpose)
    {
        std::vector<float> data = weight.get_float32_data();
        std::vector<float> transposed(data.size());
        for (int i = 0; i < outputs; i++)
            for (int j = 0; j < features; j++)
                transposed[i * features + j] = data[j * outputs + i];
        weight = Attribute({outputs, features}, transposed);
    }
    weight.shape = {outputs, features};
    Attribute bias_data;
    if (bias)
    {
        if (op->attrs.find("1") == op->attrs.end())
        {
            fprintf(stderr, "caffe layer %s (%s): bias_term=true requires bias blob 1\n", op->name.c_str(), op->type.c_str());
            return -1;
        }
        if (op->attrs.at("1").elemcount() != outputs)
        {
            fprintf(stderr, "caffe layer %s (%s): bias blob 1 has %lld elements, expected %d\n", op->name.c_str(), op->type.c_str(), (long long)op->attrs.at("1").elemcount(), outputs);
            return -1;
        }
        bias_data = op->attrs.at("1");
        bias_data.shape = {outputs};
    }
    if ((int)shape.size() != axis + 1)
    {
        Operator* flatten = graph.new_operator_before("torch.flatten", op->name + "_flatten", op);
        flatten->params["start_dim"] = axis;
        flatten->params["end_dim"] = -1;
        Operand* input = op->inputs[0];
        Operand* r = graph.new_operand(op->name + "_flatten");
        input->remove_consumer(op);
        input->consumers.push_back(flatten);
        flatten->inputs.push_back(input);
        flatten->outputs.push_back(r);
        r->producer = flatten;
        r->consumers.push_back(op);
        r->type = 1;
        r->shape.assign(shape.begin(), shape.begin() + axis);
        r->shape.push_back(features);
        if (input->params.find("__batch_index") != input->params.end())
            r->params["__batch_index"] = input->params.at("__batch_index");
        op->inputs[0] = r;
    }
    // keep the source node until level2 constructs the functional linear graph
    op->params.clear();
    op->params["num_output"] = outputs;
    op->params["bias_term"] = bias;
    op->attrs.clear();
    op->attrs["0"] = weight;
    if (bias)
        op->attrs["1"] = bias_data;
    op->outputs[0]->shape.assign(shape.begin(), shape.begin() + axis);
    op->outputs[0]->shape.push_back(outputs);
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
