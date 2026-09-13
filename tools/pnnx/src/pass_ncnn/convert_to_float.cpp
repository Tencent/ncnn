// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_to_float.h"

#include <stdint.h>
#include <string.h>

namespace pnnx {

namespace ncnn {

static Attribute float_attribute(const Attribute& attr)
{
    Attribute converted;
    converted.type = 1;
    converted.shape = attr.shape;
    converted.data.resize((size_t)attr.elemcount() * 4);

    if (attr.type >= 4 && attr.type <= 8)
    {
        for (size_t i = 0; i < converted.data.size() / 4; i++)
        {
            float value;
            if (attr.type == 4)
            {
                int32_t v;
                memcpy(&v, attr.data.data() + i * 4, 4);
                value = (float)v;
            }
            else if (attr.type == 5)
            {
                int64_t v;
                memcpy(&v, attr.data.data() + i * 8, 8);
                value = (float)v;
            }
            else if (attr.type == 6)
            {
                int16_t v;
                memcpy(&v, attr.data.data() + i * 2, 2);
                value = v;
            }
            else if (attr.type == 7)
            {
                value = ((const int8_t*)attr.data.data())[i];
            }
            else
            {
                value = ((const uint8_t*)attr.data.data())[i];
            }
            memcpy(converted.data.data() + i * 4, &value, 4);
        }
    }
    else
    {
        const std::vector<float> values = attr.get_float32_data();
        memcpy(converted.data.data(), values.data(), converted.data.size());
    }
    return converted;
}

static bool is_layout_operator(const Operator* op)
{
    return op->type == "torch.clone" || op->type == "Tensor.clone" || op->type == "torch.t" || op->type == "torch.transpose"
           || op->type == "Tensor.permute" || op->type == "Tensor.reshape" || op->type == "Tensor.reshape_as"
           || op->type == "torch.flatten" || op->type == "torch.squeeze" || op->type == "torch.unsqueeze"
           || op->type == "Tensor.slice" || op->type == "Tensor.select" || op->type == "Tensor.expand" || op->type == "Tensor.repeat";
}

static Operand* float_constant_input(Graph& graph, Operand* input, Operator* consumer, std::map<Operand*, Operand*>& converted)
{
    // Only clone the numeric branch. A shared transpose/reshape may also feed
    // Embedding, which must retain the original integer representation.
    std::vector<Operand*> chain;
    Operand* current = input;
    while (converted.find(current) == converted.end())
    {
        Operator* op = current->producer;
        if ((current->type != 4 && current->type != 5) || op->outputs.size() != 1)
            return 0;
        chain.push_back(current);
        if (op->type == "pnnx.Attribute")
            break;
        if (!is_layout_operator(op) || op->inputs.empty())
            return 0;
        current = op->inputs[0];
    }

    for (auto it = chain.rbegin(); it != chain.rend(); ++it)
    {
        Operand* original = *it;
        Operator* op = original->producer;
        Operator* shadow = graph.new_operator_before(op->type, op->name + "_float", consumer);
        shadow->params = op->params;
        shadow->attrs = op->attrs;
        shadow->inputnames = op->inputnames;
        shadow->inputs = op->inputs;
        if (op->type == "pnnx.Attribute")
            shadow->attrs["data"] = float_attribute(op->attrs.at("data"));
        else
            shadow->inputs[0] = converted.at(op->inputs[0]);
        for (Operand* in : shadow->inputs)
            in->consumers.push_back(shadow);

        Operand* output = graph.new_operand(original->name + "_float");
        output->producer = shadow;
        output->type = 1;
        output->shape = original->shape;
        output->params = original->params;
        shadow->outputs.push_back(output);
        converted[original] = output;
    }
    return converted.at(input);
}

void convert_to_float(Graph& graph)
{
    for (Operator* op : graph.ops)
    {
        for (auto& x : op->attrs)
        {
            const int type = x.second.type;
            if (type == 2 || type == 3 || type == 6 || type == 7 || type == 8 || type == 13)
                x.second = float_attribute(x.second);
        }
    }

    std::map<Operand*, Operand*> converted;
    const std::vector<Operator*> ops = graph.ops;
    for (Operator* op : ops)
    {
        if (op->type != "BinaryOp" && op->type != "UnaryOp")
            continue;

        for (Operand*& input : op->inputs)
        {
            if (input->type != 4 && input->type != 5)
                continue;
            Operand* replacement = float_constant_input(graph, input, op, converted);
            if (!replacement)
                continue;
            input->remove_consumer(op);
            replacement->consumers.push_back(op);
            input = replacement;
        }
    }
}

} // namespace ncnn

} // namespace pnnx
