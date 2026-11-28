// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_batch_norm.h"

#include "parameter.h"

#include <cmath>
#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_batch_norm(Graph& graph, Operator* op)
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
    const std::vector<int>& shape = op->inputs[0]->shape;
    op->outputs[0]->shape = shape;
    op->outputs[0]->params = op->inputs[0]->params;
    if (shape.size() < 2 || shape.size() > 5)
    {
        fprintf(stderr, "caffe layer %s (%s): expected input rank 2 through 5, got %zu\n", op->name.c_str(), op->type.c_str(), shape.size());
        return -1;
    }
    if (shape[1] <= 0)
    {
        fprintf(stderr, "caffe layer %s (%s): channel dimension must be positive, got %d\n", op->name.c_str(), op->type.c_str(), shape[1]);
        return -1;
    }
    if (op->attrs.size() != 2 && op->attrs.size() != 3)
    {
        fprintf(stderr, "caffe layer %s (%s): expected 2 or 3 statistic blobs, got %zu\n", op->name.c_str(), op->type.c_str(), op->attrs.size());
        return -1;
    }
    if (!op->attrs.count("0") || !op->attrs.count("1"))
    {
        fprintf(stderr, "caffe layer %s (%s): missing mean blob 0 or variance blob 1 (present=%d,%d)\n", op->name.c_str(), op->type.c_str(), (int)op->attrs.count("0"), (int)op->attrs.count("1"));
        return -1;
    }
    if (!get_bool(op, "use_global_stats", true))
    {
        fprintf(stderr, "caffe layer %s (%s): use_global_stats=false is not supported for inference\n", op->name.c_str(), op->type.c_str());
        return -1;
    }
    const int channels = shape[1];
    if (op->attrs.at("0").elemcount() != channels || op->attrs.at("1").elemcount() != channels)
    {
        fprintf(stderr, "caffe layer %s (%s): mean/variance blobs have %lld/%lld elements, expected %d each\n", op->name.c_str(), op->type.c_str(), (long long)op->attrs.at("0").elemcount(), (long long)op->attrs.at("1").elemcount(), channels);
        return -1;
    }
    if (op->attrs.size() == 3 && !op->attrs.count("2"))
    {
        fprintf(stderr, "caffe layer %s (%s): missing statistic scale-factor blob 2\n", op->name.c_str(), op->type.c_str());
        return -1;
    }
    if (op->attrs.size() == 3 && op->attrs.at("2").elemcount() != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): scale-factor blob 2 has %lld elements, expected 1\n", op->name.c_str(), op->type.c_str(), (long long)op->attrs.at("2").elemcount());
        return -1;
    }
    std::vector<float> mean = op->attrs.at("0").get_float32_data();
    std::vector<float> var = op->attrs.at("1").get_float32_data();
    const float factor = op->attrs.size() == 3 ? op->attrs.at("2").get_float32_data()[0] : 1.f;
    const float scale = factor == 0.f ? 0.f : 1.f / factor;
    for (int i = 0; i < channels; i++)
    {
        mean[i] *= scale;
        var[i] *= scale;
    }
    const float eps = get_float(op, "eps", 1e-5f);
    op->params.clear();
    op->params["eps"] = eps;
    op->attrs.clear();
    op->attrs["0"] = Attribute({channels}, mean);
    op->attrs["1"] = Attribute({channels}, var);
    Operand* input = op->inputs[0];
    if (input->params.count("__batch_index") && input->params.at("__batch_index").i != 0)
    {
        // use broadcast affine values when the leading dimension is not a batch
        std::vector<float> weight(channels);
        std::vector<float> bias(channels);
        for (int i = 0; i < channels; i++)
        {
            weight[i] = 1.f / sqrtf(var[i] + eps);
            bias[i] = -mean[i] * weight[i];
        }
        std::vector<int> broadcast(shape.size(), 1);
        broadcast[1] = channels;
        Operand* values[2];
        for (int i = 0; i < 2; i++)
        {
            Operator* attr = graph.new_operator_before("pnnx.Attribute", op->name + (i == 0 ? "_weight" : "_bias"), op);
            attr->attrs["data"] = Attribute({channels}, i == 0 ? weight : bias);
            attr->attrs["data"].shape = broadcast;
            Operand* r = graph.new_operand(attr->name);
            attr->outputs.push_back(r);
            r->producer = attr;
            r->shape = broadcast;
            r->type = 1;
            r->params = input->params;
            values[i] = r;
        }
        Operator* mul = graph.new_operator_before("aten::mul", op->name + "_scaled", op);
        mul->inputs = {input, values[0]};
        input->consumers.push_back(mul);
        values[0]->consumers.push_back(mul);
        Operand* scaled = graph.new_operand(mul->name);
        mul->outputs.push_back(scaled);
        scaled->producer = mul;
        scaled->shape = shape;
        scaled->type = 1;
        scaled->params = input->params;

        input->remove_consumer(op);
        op->inputs = {scaled, values[1]};
        scaled->consumers.push_back(op);
        values[1]->consumers.push_back(op);
        op->type = "aten::add";
        op->params.clear();
        op->attrs.clear();
    }
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
