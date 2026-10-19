// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_embed.h"

#include "parameter.h"

#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_embed_indices(Graph& graph)
{
    std::set<Operand*> roots;
    for (Operator* op : graph.ops)
    {
        if (op->type != "caffe.Embed")
            continue;
        if (op->inputs.size() != 1)
        {
            fprintf(stderr, "caffe layer %s (%s): expected one index input, got %zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size());
            return -1;
        }
        Operand* input = op->inputs[0];
        while (input->producer->type == "caffe.Split" && input->producer->inputs.size() == 1)
            input = input->producer->inputs[0];
        if (input->producer->type != "pnnx.Input")
        {
            fprintf(stderr, "Caffe Embed %s: indices must come from an Input through only Split layers, not %s (%s)\n", op->name.c_str(), input->producer->name.c_str(), input->producer->type.c_str());
            return -1;
        }
        roots.insert(input);
    }

    // validate every branch before changing any input storage type
    std::set<Operand*> indices;
    std::vector<Operand*> pending(roots.begin(), roots.end());
    for (size_t i = 0; i < pending.size(); i++)
    {
        Operand* input = pending[i];
        if (!indices.insert(input).second)
            continue;
        for (Operator* op : input->consumers)
        {
            if (op->type == "caffe.Embed" && op->inputs.size() == 1)
                continue;
            if (op->type == "pnnx.Output")
                continue;
            if (op->type == "caffe.Split" && op->inputs.size() == 1 && !op->outputs.empty())
            {
                pending.insert(pending.end(), op->outputs.begin(), op->outputs.end());
                continue;
            }
            fprintf(stderr, "Caffe Embed indices %s are also consumed by %s (%s); only Split and Embed layers preserve integer index storage\n", input->name.c_str(), op->name.c_str(), op->type.c_str());
            return -1;
        }
    }
    for (Operand* input : indices)
        input->type = 4;
    return 0;
}

int prepare_embed(Operator* op)
{
    if (op->inputs.size() != 1 || op->outputs.size() != 1 || op->inputs[0]->shape.empty())
    {
        fprintf(stderr, "caffe layer %s (%s): expected one input/output with known nonempty shape; inputs=%zu outputs=%zu rank=%zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size(), op->inputs.empty() ? 0u : op->inputs[0]->shape.size());
        return -1;
    }

    Operand* input = op->inputs[0];
    const int outputs = get_int(op, "num_output", 0);
    const int vocabulary = get_int(op, "input_dim", 0);
    const bool bias = get_bool(op, "bias_term", true);
    if (outputs <= 0 || vocabulary <= 0 || input->type != 4 || input->shape.size() > 2)
    {
        fprintf(stderr, "caffe layer %s (%s): require positive input_dim/num_output and int32 rank1/2 indices; input_dim=%d num_output=%d dtype=%d rank=%zu\n", op->name.c_str(), op->type.c_str(), vocabulary, outputs, input->type, input->shape.size());
        return -1;
    }
    if (op->attrs.size() != (bias ? 2u : 1u) || !op->attrs.count("0") || op->attrs.at("0").elemcount() != (int64_t)vocabulary * outputs)
    {
        fprintf(stderr, "caffe layer %s (%s): bias_term=%d requires %d blobs and weight elements=%lld; got blobs=%zu weight elements=%lld (-1 means missing)\n", op->name.c_str(), op->type.c_str(), bias, bias ? 2 : 1, (long long)vocabulary * outputs, op->attrs.size(), op->attrs.count("0") ? (long long)op->attrs.at("0").elemcount() : -1LL);
        return -1;
    }
    Attribute weight = op->attrs.at("0");
    weight.shape = {vocabulary, outputs};
    Attribute bias_data;
    if (bias)
    {
        if (!op->attrs.count("1") || op->attrs.at("1").elemcount() != outputs)
        {
            fprintf(stderr, "caffe layer %s (%s): bias blob 1 elements=%lld (-1 means missing), expected %d\n", op->name.c_str(), op->type.c_str(), op->attrs.count("1") ? (long long)op->attrs.at("1").elemcount() : -1LL, outputs);
            return -1;
        }
        bias_data = op->attrs.at("1");
        bias_data.shape = {outputs};
    }
    // embed indices use integer storage at the model boundary in ncnn
    input->type = 4;
    const int batch_index = input->shape.size() == 2 ? 0 : 233;
    input->params["__batch_index"] = batch_index;
    std::vector<int> shape = input->shape;
    shape.push_back(outputs);
    op->params.clear();
    op->params["bias_term"] = bias;
    op->attrs.clear();
    op->attrs["0"] = weight;
    if (bias)
        op->attrs["1"] = bias_data;
    op->outputs[0]->shape = shape;
    op->outputs[0]->params["__batch_index"] = batch_index;
    return 0;
}


} // namespace caffe2pnnx

} // namespace pnnx
