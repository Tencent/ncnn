// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_batch_reshape.h"

namespace pnnx {

namespace ncnn {

static bool fold_batch_after_permute(Operator* op)
{
    if (op->type != "Tensor.permute" && op->type != "torch.transpose")
        return false;

    Operand* in = op->inputs[0];
    Operand* out = op->outputs[0];

    const int input_batch_index = in->params["__batch_index"].i;
    const int output_batch_index = out->params["__batch_index"].i;

    if (input_batch_index != 0 || output_batch_index == 0 || output_batch_index == 233)
        return false;

    if (out->consumers.size() != 1)
        return false;

    Operator* op2 = out->consumers[0];
    if (op2->type == "Tensor.permute" || op2->type == "torch.transpose")
    {
        Operand* out2 = op2->outputs[0];
        if (out2->consumers.size() != 1)
            return false;

        op2 = out2->consumers[0];
    }

    if (op2->type != "Tensor.reshape" && op2->type != "torch.flatten")
        return false;

    if (op2->outputs[0]->params.find("__batch_index") != op2->outputs[0]->params.end() && op2->outputs[0]->params["__batch_index"].i != 233)
        return false;

    return true;
}

static bool extract_batch_after_permute(Operator* op)
{
    if (op->type != "Tensor.permute" && op->type != "torch.transpose")
        return false;

    Operand* in = op->inputs[0];
    Operand* out = op->outputs[0];

    const int input_batch_index = in->params["__batch_index"].i;
    const int output_batch_index = out->params["__batch_index"].i;

    if (input_batch_index == 233 || output_batch_index != 0)
        return false;

    if (in->consumers.size() != 1)
        return false;

    Operator* op0 = in->producer;
    if (op0->type != "Tensor.reshape")
        return false;

    if (op0->inputs[0]->params["__batch_index"].i != 233)
        return false;

    return true;
}

void convert_batch_reshape(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (fold_batch_after_permute(op))
            {
                Operand* in = op->inputs[0];
                Operand* out = op->outputs[0];

                Operator* reshape = graph.new_operator_before("Tensor.reshape", op->name + "_ncnnbatch2dim", op);
                Operand* reshape_out = graph.new_operand(op->name + "_ncnnbatch2dim_out");

                reshape->inputs.push_back(in);
                reshape->outputs.push_back(reshape_out);
                reshape->params["shape"] = in->shape;

                in->remove_consumer(op);
                in->consumers.push_back(reshape);

                reshape_out->producer = reshape;
                reshape_out->consumers.push_back(op);
                reshape_out->type = in->type;
                reshape_out->shape = in->shape;
                reshape_out->params["__batch_index"] = 233;

                op->inputs[0] = reshape_out;
                out->params["__batch_index"] = 233;

                Operator* op2 = out->consumers[0];
                if (op2->type == "Tensor.permute" || op2->type == "torch.transpose")
                {
                    op2->outputs[0]->params["__batch_index"] = 233;
                    op2 = op2->outputs[0]->consumers[0];
                }

                op2->outputs[0]->params["__batch_index"] = 233;

                matched = true;
                break;
            }
            else if (extract_batch_after_permute(op))
            {
                Operand* in = op->inputs[0];
                Operand* out = op->outputs[0];

                Operator* reshape = graph.new_operator_after("Tensor.reshape", op->name + "_ncnndim2batch", op);
                Operand* reshape_in = graph.new_operand(op->name + "_ncnndim2batch_in");

                reshape_in->producer = op;
                reshape_in->consumers.push_back(reshape);
                reshape_in->type = out->type;
                reshape_in->shape = out->shape;
                reshape_in->params["__batch_index"] = 233;

                reshape->inputs.push_back(reshape_in);
                reshape->outputs.push_back(out);
                reshape->params["shape"] = out->shape;

                op->outputs[0] = reshape_in;

                in->params["__batch_index"] = 233;
                out->producer = reshape;
                out->params["__batch_index"] = 0;

                matched = true;
                break;
            }
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
