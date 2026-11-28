// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_prelu.h"

#include "parameter.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_prelu(Graph& graph, Operator* op)
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
    op->outputs[0]->params = op->inputs[0]->params;
    const int channels = get_bool(op, "channel_shared", false) ? 1 : shape.size() > 1 ? shape[1] : 0;
    if (channels <= 0 || op->attrs.size() != 1 || !op->attrs.count("0") || op->attrs.at("0").elemcount() != channels)
    {
        fprintf(stderr, "caffe layer %s (%s): PReLU requires positive channel count and one matching slope blob 0; channels=%d blobs=%zu slope elements=%lld (-1 means missing)\n", op->name.c_str(), op->type.c_str(), channels, op->attrs.size(), op->attrs.count("0") ? (long long)op->attrs.at("0").elemcount() : -1LL);
        return -1;
    }
    op->attrs.at("0").shape = {channels};

    Operand* input = op->inputs[0];
    if (input->params.count("__batch_index") && input->params.at("__batch_index").i != 0)
    {
        // express channel-wise PReLU without assuming a leading batch axis
        Operator* slope = graph.new_operator_before("pnnx.Attribute", op->name + "_slope", op);
        slope->attrs["data"] = op->attrs.at("0");
        slope->attrs["data"].shape = std::vector<int>(shape.size(), 1);
        if (shape.size() > 1)
            slope->attrs["data"].shape[1] = channels;
        Operand* weight = graph.new_operand(slope->name);
        slope->outputs.push_back(weight);
        weight->producer = slope;
        weight->shape = slope->attrs.at("data").shape;
        weight->type = 1;
        weight->params = input->params;

        Operator* constant = graph.new_operator_before("prim::Constant", op->name + "_zero", op);
        constant->params["value"] = 0.f;
        Operand* zero = graph.new_operand(constant->name);
        constant->outputs.push_back(zero);
        zero->producer = constant;
        zero->type = 1;
        zero->params["__batch_index"] = 233;

        Operand* parts[2];
        for (int i = 0; i < 2; i++)
        {
            Operator* clamp = graph.new_operator_before(i == 0 ? "aten::max" : "aten::min", op->name + (i == 0 ? "_positive" : "_negative"), op);
            clamp->inputs = {input, zero};
            input->consumers.push_back(clamp);
            zero->consumers.push_back(clamp);
            Operand* r = graph.new_operand(clamp->name);
            clamp->outputs.push_back(r);
            r->producer = clamp;
            r->shape = shape;
            r->type = 1;
            r->params = input->params;
            parts[i] = r;
        }
        Operator* mul = graph.new_operator_before("aten::mul", op->name + "_scaled", op);
        mul->inputs = {parts[1], weight};
        parts[1]->consumers.push_back(mul);
        weight->consumers.push_back(mul);
        Operand* scaled = graph.new_operand(mul->name);
        mul->outputs.push_back(scaled);
        scaled->producer = mul;
        scaled->shape = shape;
        scaled->type = 1;
        scaled->params = input->params;

        input->remove_consumer(op);
        op->inputs = {parts[0], scaled};
        parts[0]->consumers.push_back(op);
        scaled->consumers.push_back(op);
        op->type = "aten::add";
        op->params.clear();
        op->attrs.clear();
    }

    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
