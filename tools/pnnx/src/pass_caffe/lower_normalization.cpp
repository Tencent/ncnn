// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_normalization.h"

#include "parameter.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

static Operand* intermediate(Graph& graph, Operator* before, const char* type, const std::string& suffix, const std::vector<Operand*>& inputs, const std::vector<int>& shape)
{
    Operator* op = graph.new_operator_before(type, before->name + suffix, before);
    Operand* r = graph.new_operand(before->name + suffix);
    op->inputs = inputs;
    for (Operand* input : inputs)
        input->consumers.push_back(op);
    op->outputs.push_back(r);
    r->producer = op;
    r->shape = shape;
    r->type = 1;
    r->params["__batch_index"] = 0;
    return r;
}

int lower_normalization(Graph& graph, Operator* op)
{
    if (op->inputs.empty() || op->outputs.size() != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): expected nonempty inputs and one output, got %zu inputs and %zu outputs\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size());
        return -1;
    }
    const std::vector<int> shape = op->inputs[0]->shape;
    const bool mvn = op->type == "caffe.MVN";
    const bool variance = get_bool(op, "normalize_variance", true);
    const bool across = mvn ? get_bool(op, "across_channels", false) : get_bool(op, "across_spatial", true);
    const bool lower_rank_mvn = mvn && across && (shape.size() == 2 || shape.size() == 3);
    const bool lower_rank_normalize = !mvn && across && get_bool(op, "channel_shared", true) && (shape.size() == 2 || shape.size() == 3);
    if (op->inputs.size() != 1 || (shape.size() != 4 && !lower_rank_mvn && !lower_rank_normalize))
    {
        fprintf(stderr, "caffe layer %s (%s): normalization requires one rank-4 input; rank2/3 requires MVN across_channels=true or Normalize across_spatial=true/channel_shared=true; inputs=%zu rank=%zu across=%d channel_shared=%d\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), shape.size(), across, get_bool(op, "channel_shared", true));
        return -1;
    }
    for (size_t i = 1; i < shape.size(); i++)
    {
        if (shape[i] <= 0)
        {
            fprintf(stderr, "caffe layer %s (%s): normalization dimension %zu must be positive, got %d\n", op->name.c_str(), op->type.c_str(), i, shape[i]);
            return -1;
        }
    }
    const float eps = mvn ? get_float(op, "eps", 1e-9f) : get_float(op, "eps", 1e-10f);
    std::vector<int> dims;
    if (across)
    {
        for (size_t i = 1; i < shape.size(); i++)
            dims.push_back(i);
    }
    else if (mvn)
    {
        dims = {2, 3};
    }
    else
    {
        dims = {1};
    }
    std::vector<int> reduced = shape;
    for (int dim : dims)
        reduced[dim] = 1;
    Attribute scale;
    if (!mvn)
    {
        const bool shared = get_bool(op, "channel_shared", true);
        const int channels = shared ? 1 : shape[1];
        if (op->attrs.size() != 1 || !op->attrs.count("0") || op->attrs.at("0").elemcount() != channels)
        {
            fprintf(stderr, "caffe layer %s (%s): Normalize requires one scale blob 0 with %d elements (channel_shared=%d); blobs=%zu scale_elements=%lld (-1 means missing)\n", op->name.c_str(), op->type.c_str(), channels, shared, op->attrs.size(), op->attrs.count("0") ? (long long)op->attrs.at("0").elemcount() : -1LL);
            return -1;
        }
        scale = op->attrs.at("0");
        scale.shape = {1, channels, 1, 1};
        if (lower_rank_normalize)
            scale.shape = std::vector<int>(shape.size(), 1);
    }
    Operand* input = op->inputs[0];
    Operand* centered = input;
    if (mvn)
    {
        Operand* mean = intermediate(graph, op, "torch.mean", "_mean", {input}, reduced);
        mean->producer->params["dim"] = dims;
        mean->producer->params["keepdim"] = true;
        if (!variance)
        {
            op->type = "aten::sub";
            op->params.clear();
            op->attrs.clear();
            op->inputs.push_back(mean);
            mean->consumers.push_back(op);
            op->outputs[0]->shape = shape;
            return 0;
        }
        centered = intermediate(graph, op, "aten::sub", "_center", {input, mean}, shape);
    }
    std::vector<Operand*> inputs = {centered};
    if (!mvn || variance)
    {
        Operand* square = intermediate(graph, op, "aten::mul", "_square", {centered, centered}, shape);
        Operand* sum = intermediate(graph, op, mvn ? "torch.mean" : "torch.sum", "_reduce", {square}, reduced);
        sum->producer->params["dim"] = dims;
        sum->producer->params["keepdim"] = true;
        Operand* epsilon = intermediate(graph, op, "prim::Constant", "_eps", {}, {});
        epsilon->producer->params["value"] = eps;
        epsilon->params["__batch_index"] = 233;
        Operand* denominator;
        if (mvn)
        {
            Operand* root = intermediate(graph, op, "aten::sqrt", "_sqrt", {sum}, reduced);
            denominator = intermediate(graph, op, "aten::add", "_denominator", {root, epsilon}, reduced);
        }
        else
        {
            Operand* shifted = intermediate(graph, op, "aten::add", "_shift", {sum, epsilon}, reduced);
            denominator = intermediate(graph, op, "aten::sqrt", "_denominator", {shifted}, reduced);
        }
        inputs.push_back(denominator);
    }
    if (!mvn)
    {
        Operand* normalized = intermediate(graph, op, "aten::div", "_normalized", inputs, shape);
        Operand* weight = intermediate(graph, op, "pnnx.Attribute", "_scale", {}, scale.shape);
        weight->producer->attrs["data"] = scale;
        inputs = {normalized, weight};
    }
    input->remove_consumer(op);
    op->inputs = inputs;
    for (Operand* x : inputs)
        x->consumers.push_back(op);
    op->type = mvn ? "aten::div" : "aten::mul";
    op->params.clear();
    op->attrs.clear();
    op->outputs[0]->shape = shape;
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
