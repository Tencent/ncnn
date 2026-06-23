// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_batch_reshape.h"

namespace pnnx {

namespace ncnn {

static bool is_layout_op(const Operator* op)
{
    return op->type == "Tensor.permute" || op->type == "torch.transpose";
}

static bool is_identity_op(const Operator* op)
{
    return op->type == "Noop" || op->type == "Tensor.clone" || op->type == "torch.clone";
}

static bool is_reshape_op(const Operator* op)
{
    return op->type == "Tensor.reshape" || op->type == "torch.flatten";
}

static bool is_batch_move_only(const Operator* op)
{
    if (op->type != "Tensor.permute")
        return false;

    Operand* in = op->inputs[0];
    Operand* out = op->outputs[0];

    const int input_batch_index = in->params["__batch_index"].i;
    const int output_batch_index = out->params["__batch_index"].i;

    if (input_batch_index < 0 || input_batch_index == 233 || output_batch_index < 0 || output_batch_index == 233)
        return false;

    const int input_rank = (int)in->shape.size();
    const std::vector<int>& dims = op->params.at("dims").ai;

    std::vector<int> new_dims;
    for (int i = 0; i < (int)dims.size(); i++)
    {
        int dim = dims[i];
        if (dim < 0)
            dim += input_rank;

        if (dim == input_batch_index)
            continue;

        new_dims.push_back(dim > input_batch_index ? dim - 1 : dim);
    }

    for (int i = 0; i < (int)new_dims.size(); i++)
    {
        if (new_dims[i] != i)
            return false;
    }

    return true;
}

static bool fold_batch_to_output_after_permute(Operator* op)
{
    if (!is_layout_op(op))
        return false;

    Operand* out = op->outputs[0];

    if (out->consumers.size() != 1 || out->consumers[0]->type != "pnnx.Output")
        return false;

    return is_batch_move_only(op);
}

static bool fold_batch_after_permute(Operator* op, std::vector<Operator*>& chain)
{
    if (!is_layout_op(op))
        return false;

    Operand* in = op->inputs[0];
    Operand* out = op->outputs[0];

    const int input_batch_index = in->params["__batch_index"].i;
    const int output_batch_index = out->params["__batch_index"].i;

    if (input_batch_index != 0 || output_batch_index == 0 || output_batch_index == 233)
        return false;

    chain.clear();
    chain.push_back(op);

    while (out->consumers.size() == 1)
    {
        Operator* op2 = out->consumers[0];
        if (op2->type == "pnnx.Output")
            return false;

        if (is_reshape_op(op2))
        {
            if (op2->outputs[0]->params.find("__batch_index") != op2->outputs[0]->params.end() && op2->outputs[0]->params["__batch_index"].i != 233)
                return false;

            chain.push_back(op2);
            return true;
        }

        if (!is_layout_op(op2) && !is_identity_op(op2))
            return false;

        chain.push_back(op2);
        out = op2->outputs[0];
    }

    return false;
}

static bool extract_batch_after_permute(Operator* op, std::vector<Operator*>& chain)
{
    if (!is_layout_op(op))
        return false;

    Operand* in = op->inputs[0];
    Operand* out = op->outputs[0];

    const int input_batch_index = in->params["__batch_index"].i;
    const int output_batch_index = out->params["__batch_index"].i;

    if (input_batch_index != 0 || output_batch_index != 0)
        return false;

    chain.clear();
    chain.push_back(op);

    while (in->consumers.size() == 1)
    {
        Operator* op0 = in->producer;
        if (op0->type == "Tensor.reshape")
        {
            if (op0->inputs[0]->params["__batch_index"].i != 233)
                return false;

            chain.push_back(op0);
            return true;
        }

        if (!is_layout_op(op0) && !is_identity_op(op0))
            return false;

        chain.push_back(op0);
        in = op0->inputs[0];
    }

    return false;
}

void convert_batch_reshape(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            std::vector<Operator*> chain;

            if (fold_batch_to_output_after_permute(op))
            {
                Operand* out = op->outputs[0];
                const int batch_axis = out->params["__batch_index"].i;

                op->type = "Tensor.reshape";
                op->params.clear();
                op->params["shape"] = out->shape;

                out->params["__batch_axis"] = batch_axis;
                out->params["__batch_index"] = 233;

                matched = true;
                break;
            }

            if (fold_batch_after_permute(op, chain))
            {
                Operand* in = op->inputs[0];

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

                for (Operator* x : chain)
                {
                    x->outputs[0]->params["__batch_index"] = 233;
                }

                matched = true;
                break;
            }

            if (extract_batch_after_permute(op, chain))
            {
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

                for (Operator* x : chain)
                {
                    x->outputs[0]->params["__batch_index"] = 233;
                }

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
