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

static bool is_elementwise_op(const Operator* op)
{
    static const char* elementwise_ops[] = {
        "F.celu",
        "F.elu",
        "F.gelu",
        "F.hardshrink",
        "F.hardsigmoid",
        "F.hardswish",
        "F.hardtanh",
        "F.leaky_relu",
        "F.logsigmoid",
        "F.mish",
        "F.relu",
        "F.relu6",
        "F.selu",
        "F.sigmoid",
        "F.silu",
        "F.softplus",
        "F.softshrink",
        "F.softsign",
        "F.tanh",
        "F.tanhshrink",
        "F.threshold",

        "nn.CELU",
        "nn.ELU",
        "nn.GELU",
        "nn.Hardshrink",
        "nn.Hardsigmoid",
        "nn.Hardswish",
        "nn.Hardtanh",
        "nn.LeakyReLU",
        "nn.LogSigmoid",
        "nn.Mish",
        "nn.ReLU",
        "nn.ReLU6",
        "nn.SELU",
        "nn.Sigmoid",
        "nn.SiLU",
        "nn.Softplus",
        "nn.Softshrink",
        "nn.Softsign",
        "nn.Tanh",
        "nn.Tanhshrink",
        "nn.Threshold",

        "torch.abs",
        "torch.acos",
        "torch.acosh",
        "torch.asin",
        "torch.asinh",
        "torch.atan",
        "torch.atanh",
        "torch.ceil",
        "torch.clamp",
        "torch.cos",
        "torch.cosh",
        "torch.exp",
        "torch.floor",
        "torch.imag",
        "torch.log",
        "torch.log10",
        "torch.neg",
        "torch.real",
        "torch.reciprocal",
        "torch.rsqrt",
        "torch.sign",
        "torch.sin",
        "torch.sinh",
        "torch.sqrt",
        "torch.square",
        "torch.tan",
        "torch.tanh",
        "torch.trunc",
    };

    if (op->inputs.size() != 1 || op->outputs.size() != 1 || op->inputs[0]->shape != op->outputs[0]->shape)
        return false;

    const size_t elementwise_ops_count = sizeof(elementwise_ops) / sizeof(const char*);
    for (size_t i = 0; i < elementwise_ops_count; i++)
    {
        if (op->type == elementwise_ops[i])
            return true;
    }

    return false;
}

static bool is_reshape_op(const Operator* op)
{
    return op->type == "Tensor.reshape" || op->type == "torch.flatten";
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

        if (!is_layout_op(op2) && !is_identity_op(op2) && !is_elementwise_op(op2))
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

        if (!is_layout_op(op0) && !is_identity_op(op0) && !is_elementwise_op(op0))
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
