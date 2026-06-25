// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_batch_layout.h"

namespace pnnx {

namespace ncnn {

static int get_batch_index(const std::map<Operand*, int>& batch_indices, Operand* r)
{
    std::map<Operand*, int>::const_iterator it = batch_indices.find(r);
    if (it != batch_indices.end())
        return it->second;

    return r->params.at("__batch_index").i;
}

static int default_ncnn_batch_in_shape(int batch_index)
{
    return batch_index == 0 ? 0 : 1;
}

static int get_ncnn_batch_in_shape(const Operand* r)
{
    if (r->params.find("__ncnn_batch_in_shape") == r->params.end())
        return default_ncnn_batch_in_shape(r->params.at("__batch_index").i);

    return r->params.at("__ncnn_batch_in_shape").i;
}

static void set_ncnn_batch_in_shape(Operand* r, int batch_in_shape)
{
    r->params["__ncnn_batch_in_shape"] = batch_in_shape;
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

static bool is_layout_op(const Operator* op)
{
    return op->type == "Tensor.permute" || op->type == "torch.transpose" || op->type == "torch.t";
}

static bool is_axis_op(const Operator* op)
{
    return op->type == "torch.cat" || op->type == "torch.chunk" || op->type == "torch.split" || op->type == "torch.tensor_split"
           || op->type == "Tensor.slice" || op->type == "Tensor.select";
}

static bool is_reshape_op(const Operator* op)
{
    return op->type == "Tensor.reshape" || op->type == "torch.flatten" || op->type == "Tensor.unflatten";
}

static bool is_recurrent_op(const Operator* op)
{
    return op->type == "nn.RNN" || op->type == "nn.LSTM" || op->type == "nn.GRU";
}

static bool is_binary_eltwise_op(const Operator* op)
{
    return op->type == "BinaryOp" || op->type == "Eltwise";
}

static bool is_attention_op(const Operator* op)
{
    return op->type == "F.scaled_dot_product_attention" || op->type == "nn.MultiheadAttention";
}

static bool is_layout_transparent_op(const Operator* op)
{
    return is_identity_op(op) || is_elementwise_op(op) || is_layout_op(op);
}

static int get_consumer_ncnn_batch_in_shape(const Operator* op, int input_index, const Operand* r)
{
    if (op->type == "pnnx.Output")
        return get_ncnn_batch_in_shape(r);

    if (op->type == "pnnx.Expression" || op->type == "pnnx.SliceIndexes")
        return get_ncnn_batch_in_shape(r);

    if (op->type == "Tensor.reshape" && input_index != 0)
        return get_ncnn_batch_in_shape(r);

    if ((op->type == "Tensor.slice" || op->type == "Tensor.select") && input_index != 0)
        return get_ncnn_batch_in_shape(r);

    if (op->type == "F.scaled_dot_product_attention")
        return r->params.at("__batch_index").i == 233 ? 1 : 0;

    if (op->type == "nn.MultiheadAttention")
    {
        if (op->inputnames.size() == op->inputs.size() && op->inputnames[input_index] == "attn_mask")
            return 1;

        return r->params.at("__batch_index").i == 233 ? 1 : 0;
    }

    if (is_binary_eltwise_op(op))
    {
        if (r->params.at("__batch_index").i == 233)
            return 1;

        if (!op->outputs.empty())
            return default_ncnn_batch_in_shape(op->outputs[0]->params.at("__batch_index").i);
    }

    if (is_recurrent_op(op))
        return get_ncnn_batch_in_shape(r);

    if (is_layout_op(op) && r->params.at("__batch_index").i != 233)
        return 0;

    if (is_axis_op(op))
        return get_ncnn_batch_in_shape(r);

    if (is_reshape_op(op) && input_index == 0 && get_ncnn_batch_in_shape(r) == 0)
    {
        const int batch_index = r->params.at("__batch_index").i;
        if (batch_index > 0 && batch_index != 233 && r->shape.size() <= 4 && !op->outputs.empty() && !op->outputs[0]->shape.empty() && op->outputs[0]->params.at("__batch_index").i == 233)
            return 1;
    }

    if (is_layout_transparent_op(op) || is_reshape_op(op))
        return get_ncnn_batch_in_shape(r);

    return default_ncnn_batch_in_shape(r->params.at("__batch_index").i);
}

static void replace_consumer_input(Operator* op, int input_index, Operand* old_operand, Operand* new_operand)
{
    old_operand->remove_consumer(op);
    new_operand->consumers.push_back(op);
    op->inputs[input_index] = new_operand;
}

static Operator* insert_batch_to_dim(Graph& graph, Operator* op, int input_index, Operand* in, const std::map<Operand*, int>& batch_indices)
{
    const std::string name = op->name + "_ncnnbatch2dim_" + std::to_string(input_index);

    Operator* reshape = graph.new_operator_before("Tensor.reshape", name, op);
    Operand* reshape_out = graph.new_operand(name + "_out");

    reshape->inputs.push_back(in);
    reshape->outputs.push_back(reshape_out);

    std::vector<int> shape = in->shape;
    for (int& s : shape)
    {
        if (s == -1)
            s = 0;
    }
    reshape->params["shape"] = shape;

    reshape_out->producer = reshape;
    reshape_out->type = in->type;
    reshape_out->shape = in->shape;
    reshape_out->params["__batch_index"] = in->params["__batch_index"];
    reshape_out->params["__batch_axis"] = get_batch_index(batch_indices, in);

    set_ncnn_batch_in_shape(reshape_out, 1);

    replace_consumer_input(op, input_index, in, reshape_out);

    return reshape;
}

static Operator* insert_dim_to_batch(Graph& graph, Operator* op, int input_index, Operand* in, const std::map<Operand*, int>& batch_indices)
{
    const std::string name = op->name + "_ncnndim2batch_" + std::to_string(input_index);

    Operator* reshape = graph.new_operator_before("Tensor.reshape", name, op);
    Operand* reshape_out = graph.new_operand(name + "_out");

    reshape->inputs.push_back(in);
    reshape->outputs.push_back(reshape_out);

    std::vector<int> shape = in->shape;
    for (int& s : shape)
    {
        if (s == -1)
            s = 0;
    }
    reshape->params["shape"] = shape;

    reshape_out->producer = reshape;
    reshape_out->type = in->type;
    reshape_out->shape = in->shape;
    reshape_out->params["__batch_index"] = in->params["__batch_index"];
    reshape_out->params["__batch_axis"] = get_batch_index(batch_indices, in);

    set_ncnn_batch_in_shape(reshape_out, 0);

    replace_consumer_input(op, input_index, in, reshape_out);

    return reshape;
}

static void set_reshape_batch_axis(Operator* op, const std::map<Operand*, int>& batch_indices)
{
    if (op->inputs.empty() || op->outputs.empty())
        return;

    Operand* in = op->inputs[0];
    Operand* out = op->outputs[0];

    const int in_batch_in_shape = get_ncnn_batch_in_shape(in);
    const int out_batch_in_shape = get_ncnn_batch_in_shape(out);
    if (in_batch_in_shape == out_batch_in_shape)
        return;

    if (in_batch_in_shape == 0 && out_batch_in_shape == 1)
    {
        int batch_index = get_batch_index(batch_indices, out);
        if (batch_index != 233)
            out->params["__batch_axis"] = batch_index;
    }
    if (in_batch_in_shape == 1 && out_batch_in_shape == 0)
    {
        const int batch_index = get_batch_index(batch_indices, in);
        out->params["__batch_axis"] = batch_index == 233 ? get_batch_index(batch_indices, out) : batch_index;
    }
}

void convert_batch_layout(Graph& graph)
{
    std::map<Operand*, int> batch_indices;
    for (Operand* r : graph.operands)
    {
        batch_indices[r] = r->params.at("__batch_index").i;
        set_ncnn_batch_in_shape(r, default_ncnn_batch_in_shape(batch_indices[r]));

        if (r->producer && r->producer->type == "pnnx.Input" && r->producer->params.find("__torch_batch_index") != r->producer->params.end() && r->producer->params["__torch_batch_index"].i != 233)
            set_ncnn_batch_in_shape(r, 0);
    }

    std::vector<Operator*> ops = graph.ops;
    for (Operator* op : ops)
    {
        if (op->type == "pnnx.Input")
            continue;

        for (int i = 0; i < (int)op->inputs.size(); i++)
        {
            Operand* in = op->inputs[i];

            const int batch_in_shape = get_ncnn_batch_in_shape(in);
            const int required_batch_in_shape = get_consumer_ncnn_batch_in_shape(op, i, in);
            if (batch_in_shape == required_batch_in_shape)
                continue;

            Operator* reshape = 0;
            if (batch_in_shape == 0 && required_batch_in_shape == 1)
                reshape = insert_batch_to_dim(graph, op, i, in, batch_indices);
            if (batch_in_shape == 1 && required_batch_in_shape == 0)
                reshape = insert_dim_to_batch(graph, op, i, in, batch_indices);

            if (!reshape)
                continue;

            batch_indices[reshape->outputs[0]] = batch_indices[in];
            set_reshape_batch_axis(reshape, batch_indices);
        }

        if (op->outputs.empty())
            continue;

        if (is_layout_transparent_op(op))
        {
            const int batch_in_shape = op->inputs.empty() ? default_ncnn_batch_in_shape(get_batch_index(batch_indices, op->outputs[0])) : get_ncnn_batch_in_shape(op->inputs[0]);
            for (Operand* r : op->outputs)
                set_ncnn_batch_in_shape(r, batch_in_shape);
        }
        else if (is_recurrent_op(op))
        {
            const int batch_in_shape = op->inputs.empty() ? default_ncnn_batch_in_shape(get_batch_index(batch_indices, op->outputs[0])) : get_ncnn_batch_in_shape(op->inputs[0]);
            for (Operand* r : op->outputs)
                set_ncnn_batch_in_shape(r, batch_in_shape);
        }
        else if (is_reshape_op(op))
        {
            for (Operand* r : op->outputs)
            {
                const int batch_index = get_batch_index(batch_indices, r);
                if (r->shape.size() > 4 && batch_index != 233)
                    set_ncnn_batch_in_shape(r, 0);
                else
                    set_ncnn_batch_in_shape(r, default_ncnn_batch_in_shape(batch_index));
            }

            set_reshape_batch_axis(op, batch_indices);
        }
        else if (is_binary_eltwise_op(op))
        {
            for (Operand* r : op->outputs)
                set_ncnn_batch_in_shape(r, default_ncnn_batch_in_shape(get_batch_index(batch_indices, r)));
        }
        else if (is_attention_op(op))
        {
            for (Operand* r : op->outputs)
                set_ncnn_batch_in_shape(r, get_batch_index(batch_indices, r) == 233 ? 1 : 0);
        }
        else
        {
            const int batch_in_shape = op->inputs.empty() ? default_ncnn_batch_in_shape(get_batch_index(batch_indices, op->outputs[0])) : get_ncnn_batch_in_shape(op->inputs[0]);
            for (Operand* r : op->outputs)
                set_ncnn_batch_in_shape(r, batch_in_shape);
        }
    }

    for (Operand* r : graph.operands)
    {
        const int batch_index = get_batch_index(batch_indices, r);

        r->params["__batch_index"] = batch_index;
    }
}

} // namespace ncnn

} // namespace pnnx
