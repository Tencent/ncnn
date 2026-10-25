// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_lstm.h"

#include "parameter.h"

#include <algorithm>
#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

static Attribute reorder_lstm_gates(const Attribute& attr, const std::vector<int>& shape)
{
    const std::vector<float> data = attr.get_float32_data();
    const size_t gate_size = data.size() / 4;
    std::vector<float> reordered(data.size());
    // caffe stores IFOG, while nn.LSTM expects IFGO
    const int order[4] = {0, 1, 3, 2};
    for (int i = 0; i < 4; i++)
        std::copy(data.begin() + order[i] * gate_size, data.begin() + (order[i] + 1) * gate_size, reordered.begin() + i * gate_size);
    Attribute result({(int)reordered.size()}, reordered);
    result.shape = shape;
    return result;
}

int lower_lstm(Graph& graph, Operator* op)
{
    if (op->inputs.size() == 2)
    {
        fprintf(stderr, "caffe layer %s (%s): continuation input is unsupported by the existing ncnn LSTM; conversion refused\n", op->name.c_str(), op->type.c_str());
        return -1;
    }
    if (op->inputs.size() != 1 || op->outputs.size() != 1 || get_bool(op, "expose_hidden", false))
    {
        fprintf(stderr, "Caffe LSTM %s: expected one data input, one output, and no exposed state\n", op->name.c_str());
        return -1;
    }
    const std::vector<int> shape = op->inputs[0]->shape;
    if (shape.size() != 2 && shape.size() != 3)
    {
        fprintf(stderr, "caffe layer %s (%s): expected rank2 or rank3 sequence input, got rank%zu\n", op->name.c_str(), op->type.c_str(), shape.size());
        return -1;
    }
    if (shape.size() == 3 && shape[1] != 1)
    {
        fprintf(stderr, "Caffe LSTM %s: ncnn export requires sequence batch size 1\n", op->name.c_str());
        return -1;
    }
    const int input_size = shape.back();
    const int hidden_size = get_int(op, "num_output", 0);
    if (shape[0] <= 0 || input_size <= 0 || hidden_size <= 0)
    {
        fprintf(stderr, "caffe layer %s (%s): sequence length=%d input_size=%d num_output=%d must be positive\n", op->name.c_str(), op->type.c_str(), shape[0], input_size, hidden_size);
        return -1;
    }
    if (op->attrs.size() != 3 || !op->attrs.count("0") || !op->attrs.count("1") || !op->attrs.count("2"))
    {
        fprintf(stderr, "caffe layer %s (%s): expected input weight, bias and recurrent weight blobs 0/1/2, got %zu blobs\n", op->name.c_str(), op->type.c_str(), op->attrs.size());
        return -1;
    }
    if (op->attrs.at("0").elemcount() != 4LL * hidden_size * input_size || op->attrs.at("1").elemcount() != 4LL * hidden_size || op->attrs.at("2").elemcount() != 4LL * hidden_size * hidden_size)
    {
        fprintf(stderr, "caffe layer %s (%s): weight blob sizes are %lld/%lld/%lld, expected %lld/%lld/%lld\n", op->name.c_str(), op->type.c_str(), (long long)op->attrs.at("0").elemcount(), (long long)op->attrs.at("1").elemcount(), (long long)op->attrs.at("2").elemcount(), 4LL * hidden_size * input_size, 4LL * hidden_size, 4LL * hidden_size * hidden_size);
        return -1;
    }

    const Attribute weight_ih = reorder_lstm_gates(op->attrs.at("0"), {4 * hidden_size, input_size});
    const Attribute bias = reorder_lstm_gates(op->attrs.at("1"), {4 * hidden_size});
    const Attribute weight_hh = reorder_lstm_gates(op->attrs.at("2"), {4 * hidden_size, hidden_size});
    op->type = "nn.LSTM";
    op->params.clear();
    op->params["input_size"] = input_size;
    op->params["hidden_size"] = hidden_size;
    op->params["num_layers"] = 1;
    op->params["bias"] = true;
    op->params["batch_first"] = false;
    op->params["bidirectional"] = false;
    op->params["proj_size"] = 0;
    op->attrs.clear();
    op->attrs["weight_ih_l0"] = weight_ih;
    op->attrs["weight_hh_l0"] = weight_hh;
    op->attrs["bias_ih_l0"] = bias;
    op->attrs["bias_hh_l0"] = Attribute({4 * hidden_size}, std::vector<float>(4 * hidden_size, 0.f));

    const int batch_index = shape.size() == 2 ? 233 : 1;
    op->inputs[0]->params["__batch_index"] = batch_index;
    op->outputs[0]->params["__batch_index"] = batch_index;
    op->outputs[0]->shape = shape;
    op->outputs[0]->shape.back() = hidden_size;
    if (shape.size() == 2)
    {
        // Older PyTorch versions require an explicit sequence batch dimension.
        Operand* input = op->inputs[0];
        Operand* output = op->outputs[0];
        Operator* batch = graph.new_operator_before("Tensor.reshape", op->name + "_batch", op);
        Operand* batched_input = graph.new_operand(op->name + "_batch");
        batched_input->type = input->type;
        batched_input->shape = {shape[0], 1, input_size};
        batched_input->params["__batch_index"] = 1;
        batched_input->producer = batch;
        batched_input->consumers.push_back(op);
        batch->params["shape"] = batched_input->shape;
        batch->inputs.push_back(input);
        batch->outputs.push_back(batched_input);
        input->remove_consumer(op);
        input->consumers.push_back(batch);
        op->inputs[0] = batched_input;

        Operator* unbatch = graph.new_operator_after("Tensor.reshape", op->name + "_unbatch", op);
        Operand* batched_output = graph.new_operand(op->name + "_batched_output");
        batched_output->type = 1;
        batched_output->shape = {shape[0], 1, hidden_size};
        batched_output->params["__batch_index"] = 1;
        batched_output->producer = op;
        batched_output->consumers.push_back(unbatch);
        unbatch->params["shape"] = output->shape;
        unbatch->inputs.push_back(batched_output);
        unbatch->outputs.push_back(output);
        output->producer = unbatch;
        output->type = 1;
        op->outputs[0] = batched_output;
    }
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
