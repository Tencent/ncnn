// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_recurrent.h"

#include <limits.h>

#include <map>
#include <string>
#include <vector>

namespace pnnx {

enum RecurrentType
{
    RECURRENT_GRU,
    RECURRENT_LSTM,
    RECURRENT_RNN_TANH,
    RECURRENT_RNN_RELU
};

static bool get_recurrent_type(const Operator* op, RecurrentType& recurrent_type)
{
    if (op->type == "aten::gru")
        recurrent_type = RECURRENT_GRU;
    else if (op->type == "aten::lstm")
        recurrent_type = RECURRENT_LSTM;
    else if (op->type == "aten::rnn_tanh")
        recurrent_type = RECURRENT_RNN_TANH;
    else if (op->type == "aten::rnn_relu")
        recurrent_type = RECURRENT_RNN_RELU;
    else
        return false;

    return true;
}

static bool get_static_argument(const Operator* op, const char* name, Parameter& value)
{
    if (!op->has_input(name))
        return false;

    const Operand* input = op->named_input(name);
    if (!input || !input->producer || input->producer->type != "prim::Constant" || !input->producer->has_param("value"))
        return false;

    value = input->producer->params.at("value");
    return true;
}

static bool get_bool_argument(const Operator* op, const char* name, bool& value)
{
    Parameter param;
    if (!get_static_argument(op, name, param) || param.type != 1)
        return false;

    value = param.b;
    return true;
}

static bool get_int_argument(const Operator* op, const char* name, int& value)
{
    Parameter param;
    if (!get_static_argument(op, name, param) || param.type != 2)
        return false;

    value = param.i;
    return true;
}

static bool get_float_argument(const Operator* op, const char* name, float& value)
{
    Parameter param;
    if (!get_static_argument(op, name, param))
        return false;

    if (param.type == 2)
    {
        value = param.i;
        return true;
    }
    if (param.type == 3)
    {
        value = param.f;
        return true;
    }

    return false;
}

static bool get_attribute(const Operand* operand, Attribute& attr)
{
    if (!operand || !operand->producer || operand->producer->type != "pnnx.Attribute" || !operand->producer->has_attr("data"))
        return false;

    attr = operand->producer->attrs.at("data");
    return true;
}

static std::string recurrent_attribute_name(const char* name, int layer, int direction)
{
    std::string key = std::string(name) + "_l" + std::to_string(layer);
    if (direction == 1)
        key += "_reverse";
    return key;
}

static bool fuse_recurrent(Operator* op)
{
    RecurrentType recurrent_type;
    if (!get_recurrent_type(op, recurrent_type))
        return false;

    const bool is_lstm = recurrent_type == RECURRENT_LSTM;
    const int gate_count = recurrent_type == RECURRENT_GRU ? 3 : (is_lstm ? 4 : 1);
    const size_t output_count = is_lstm ? 3 : 2;
    if (op->outputs.size() != output_count)
        return false;

    if (!op->has_input("input") || !op->has_input("hx") || !op->has_input("params"))
        return false;

    bool bias = false;
    bool train = false;
    bool bidirectional = false;
    bool batch_first = false;
    int num_layers = 0;
    float dropout = 0.f;
    if (!get_bool_argument(op, "has_biases", bias)
            || !get_int_argument(op, "num_layers", num_layers)
            || !get_float_argument(op, "dropout", dropout)
            || !get_bool_argument(op, "train", train)
            || !get_bool_argument(op, "bidirectional", bidirectional)
            || !get_bool_argument(op, "batch_first", batch_first))
        return false;

    if (num_layers <= 0 || dropout < 0.f || dropout > 1.f || train)
        return false;

    const Operand* params_operand = op->named_input("params");
    const Operator* params_op = params_operand ? params_operand->producer : 0;
    if (!params_op || params_op->type != "prim::ListConstruct")
        return false;

    const std::vector<Operand*>& weights = params_op->inputs;
    if (weights.size() < 2)
        return false;

    Attribute first_weight_ih;
    Attribute first_weight_hh;
    if (!get_attribute(weights[0], first_weight_ih) || !get_attribute(weights[1], first_weight_hh))
        return false;

    if (first_weight_ih.shape.size() != 2 || first_weight_hh.shape.size() != 2
            || first_weight_ih.shape[0] <= 0 || first_weight_ih.shape[0] % gate_count != 0
            || first_weight_ih.shape[1] <= 0)
        return false;

    const int input_size = first_weight_ih.shape[1];
    const int hidden_size = first_weight_ih.shape[0] / gate_count;
    if (first_weight_hh.shape[0] != gate_count * hidden_size || first_weight_hh.shape[1] <= 0)
        return false;

    const int proj_size = is_lstm && first_weight_hh.shape[1] != hidden_size ? first_weight_hh.shape[1] : 0;
    if (proj_size < 0 || proj_size >= hidden_size)
        return false;

    const int recurrent_size = proj_size > 0 ? proj_size : hidden_size;
    const int direction_count = bidirectional ? 2 : 1;
    const int weights_per_direction = 2 + (bias ? 2 : 0) + (proj_size > 0 ? 1 : 0);
    const size_t weights_per_layer = (size_t)direction_count * weights_per_direction;
    if (weights.size() % weights_per_layer != 0 || weights.size() / weights_per_layer != (size_t)num_layers)
        return false;

    if (recurrent_size > INT_MAX / direction_count)
        return false;
    const int stacked_recurrent_size = recurrent_size * direction_count;

    std::map<std::string, Attribute> attrs;
    size_t weight_index = 0;
    for (int layer = 0; layer < num_layers; layer++)
    {
        const int layer_input_size = layer == 0 ? input_size : stacked_recurrent_size;
        for (int direction = 0; direction < direction_count; direction++)
        {
            Attribute weight_ih;
            Attribute weight_hh;
            if (!get_attribute(weights[weight_index++], weight_ih) || !get_attribute(weights[weight_index++], weight_hh))
                return false;

            if (weight_ih.shape != std::vector<int>({gate_count * hidden_size, layer_input_size})
                    || weight_hh.shape != std::vector<int>({gate_count * hidden_size, recurrent_size}))
                return false;

            attrs[recurrent_attribute_name("weight_ih", layer, direction)] = weight_ih;
            attrs[recurrent_attribute_name("weight_hh", layer, direction)] = weight_hh;

            if (bias)
            {
                Attribute bias_ih;
                Attribute bias_hh;
                if (!get_attribute(weights[weight_index++], bias_ih) || !get_attribute(weights[weight_index++], bias_hh))
                    return false;

                if (bias_ih.shape != std::vector<int>({gate_count * hidden_size})
                        || bias_hh.shape != std::vector<int>({gate_count * hidden_size}))
                    return false;

                attrs[recurrent_attribute_name("bias_ih", layer, direction)] = bias_ih;
                attrs[recurrent_attribute_name("bias_hh", layer, direction)] = bias_hh;
            }

            if (proj_size > 0)
            {
                Attribute weight_hr;
                if (!get_attribute(weights[weight_index++], weight_hr))
                    return false;

                if (weight_hr.shape != std::vector<int>({proj_size, hidden_size}))
                    return false;

                attrs[recurrent_attribute_name("weight_hr", layer, direction)] = weight_hr;
            }
        }
    }

    std::vector<Operand*> inputs;
    inputs.push_back(op->named_input("input"));
    if (is_lstm)
    {
        const Operand* hx_operand = op->named_input("hx");
        const Operator* hx_op = hx_operand ? hx_operand->producer : 0;
        if (!hx_op || hx_op->type != "prim::ListConstruct" || hx_op->inputs.size() != 2)
            return false;

        inputs.push_back(hx_op->inputs[0]);
        inputs.push_back(hx_op->inputs[1]);
    }
    else
    {
        inputs.push_back(op->named_input("hx"));
    }

    for (size_t i = 0; i < op->inputs.size(); i++)
        op->inputs[i]->remove_consumer(op);
    for (size_t i = 0; i < inputs.size(); i++)
        inputs[i]->consumers.push_back(op);

    op->inputs = inputs;
    op->inputnames.clear();
    op->params.clear();
    op->attrs.swap(attrs);

    op->params["input_size"] = input_size;
    op->params["hidden_size"] = hidden_size;
    op->params["num_layers"] = num_layers;
    op->params["bias"] = bias;
    op->params["batch_first"] = batch_first;
    op->params["bidirectional"] = bidirectional;

    if (recurrent_type == RECURRENT_GRU)
        op->type = "nn.GRU";
    else if (is_lstm)
    {
        op->type = "nn.LSTM";
        op->params["proj_size"] = proj_size;
    }
    else
    {
        op->type = "nn.RNN";
        op->params["nonlinearity"] = recurrent_type == RECURRENT_RNN_RELU ? "relu" : "tanh";
    }

    return true;
}

void fuse_recurrent(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
        fuse_recurrent(graph.ops[i]);
}

} // namespace pnnx
