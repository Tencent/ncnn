// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_exported_rnn.h"

#include <string>
#include <vector>

namespace pnnx {

static const Parameter* get_constant(const Operand* operand)
{
    if (!operand || !operand->producer || operand->producer->type != "prim::Constant")
        return 0;

    std::map<std::string, Parameter>::const_iterator it = operand->producer->params.find("value");
    if (it == operand->producer->params.end())
        return 0;

    return &it->second;
}

static bool get_bool_constant(const Operand* operand, bool& value)
{
    const Parameter* p = get_constant(operand);
    if (!p || p->type != 1)
        return false;
    value = p->b;
    return true;
}

static bool get_int_constant(const Operand* operand, int& value)
{
    const Parameter* p = get_constant(operand);
    if (!p || p->type != 2)
        return false;
    value = p->i;
    return true;
}

static bool get_float_constant(const Operand* operand, float& value)
{
    const Parameter* p = get_constant(operand);
    if (!p || p->type != 3)
        return false;
    value = p->f;
    return true;
}

static bool get_attribute(const Operand* operand, Attribute& value)
{
    if (!operand || !operand->producer || operand->producer->type != "pnnx.Attribute")
        return false;

    std::map<std::string, Attribute>::const_iterator it = operand->producer->attrs.find("data");
    if (it == operand->producer->attrs.end())
        return false;

    value = it->second;
    return true;
}

static void replace_inputs(Operator* op, const std::vector<Operand*>& inputs, const std::vector<std::string>& inputnames)
{
    for (size_t i = 0; i < op->inputs.size(); i++)
        op->inputs[i]->remove_consumer(op);

    op->inputs = inputs;
    op->inputnames = inputnames;
    for (size_t i = 0; i < op->inputs.size(); i++)
        op->inputs[i]->consumers.push_back(op);
}

void fuse_exported_rnn(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];
        const bool is_gru = op->type == "aten::gru";
        const bool is_lstm = op->type == "aten::lstm";
        const bool is_relu_rnn = op->type == "aten::rnn_relu";
        const bool is_rnn = op->type == "aten::rnn_tanh" || is_relu_rnn;
        if ((!is_gru && !is_lstm && !is_rnn) || op->inputs.size() != 9)
            continue;

        Operator* weight_list = op->inputs[2]->producer;
        if (!weight_list || weight_list->type != "prim::ListConstruct" || weight_list->inputs.empty())
            continue;

        bool has_biases;
        bool bidirectional;
        bool batch_first;
        int num_layers;
        float dropout;
        if (!get_bool_constant(op->inputs[3], has_biases)
                || !get_int_constant(op->inputs[4], num_layers)
                || !get_float_constant(op->inputs[5], dropout)
                || !get_bool_constant(op->inputs[7], bidirectional)
                || !get_bool_constant(op->inputs[8], batch_first)
                || num_layers <= 0)
            continue;

        Operator* hidden_list = 0;
        if (is_lstm)
        {
            hidden_list = op->inputs[1]->producer;
            if (!hidden_list || hidden_list->type != "prim::ListConstruct" || hidden_list->inputs.size() != 2)
                continue;
        }

        const int num_directions = bidirectional ? 2 : 1;
        const int weight_group_count = num_layers * num_directions;
        const int base_group_size = has_biases ? 4 : 2;
        const int flat_weight_count = (int)weight_list->inputs.size();
        bool has_projection = false;
        if (flat_weight_count == weight_group_count * base_group_size)
        {
            has_projection = false;
        }
        else if (is_lstm && flat_weight_count == weight_group_count * (base_group_size + 1))
        {
            has_projection = true;
        }
        else
        {
            continue;
        }

        Attribute first_weight;
        if (!get_attribute(weight_list->inputs[0], first_weight) || first_weight.shape.size() != 2)
            continue;

        const int gate_count = is_lstm ? 4 : (is_gru ? 3 : 1);
        if (first_weight.shape[0] % gate_count != 0)
            continue;

        std::map<std::string, Attribute> attributes;
        size_t weight_index = 0;
        int projection_size = 0;
        bool valid = true;
        for (int layer = 0; layer < num_layers && valid; layer++)
        {
            for (int direction = 0; direction < num_directions && valid; direction++)
            {
                const std::string suffix = std::string("_l") + std::to_string(layer) + (direction ? "_reverse" : "");
                const char* names[] = {"weight_ih", "weight_hh", "bias_ih", "bias_hh", "weight_hr"};
                const int group_size = base_group_size + (has_projection ? 1 : 0);
                for (int j = 0; j < group_size; j++)
                {
                    Attribute attribute;
                    if (!get_attribute(weight_list->inputs[weight_index++], attribute))
                    {
                        valid = false;
                        break;
                    }
                    attributes[std::string(names[j]) + suffix] = attribute;
                    if (has_projection && j == group_size - 1 && attribute.shape.size() == 2)
                        projection_size = attribute.shape[0];
                }
            }
        }
        if (!valid)
            continue;

        op->type = is_lstm ? "nn.LSTM" : (is_gru ? "nn.GRU" : "nn.RNN");
        op->params["input_size"] = first_weight.shape[1];
        op->params["hidden_size"] = first_weight.shape[0] / gate_count;
        op->params["num_layers"] = num_layers;
        op->params["bias"] = has_biases;
        op->params["batch_first"] = batch_first;
        op->params["dropout"] = dropout;
        op->params["bidirectional"] = bidirectional;
        op->attrs.swap(attributes);
        if (is_rnn)
            op->params["nonlinearity"] = is_relu_rnn ? "relu" : "tanh";
        if (is_lstm)
            op->params["proj_size"] = projection_size;

        if (is_lstm)
            replace_inputs(op, {op->inputs[0], hidden_list->inputs[0], hidden_list->inputs[1]}, {"input", "h_0", "c_0"});
        else
            replace_inputs(op, {op->inputs[0], op->inputs[1]}, {"input", "h_0"});
    }
}

} // namespace pnnx
