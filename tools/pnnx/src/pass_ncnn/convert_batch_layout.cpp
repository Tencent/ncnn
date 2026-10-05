// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_batch_layout.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

static int get_batch_index(const std::map<Operand*, int>& batch_indices, Operand* r)
{
    std::map<Operand*, int>::const_iterator it = batch_indices.find(r);
    if (it != batch_indices.end())
        return it->second;

    return r->params.at("__batch_index").i;
}

static int default_ncnn_batch_axis(int batch_index)
{
    return batch_index == 0 ? 0 : 233;
}

static int get_ncnn_batch_axis(const Operand* r)
{
    if (r->params.find("__ncnn_batch_axis") != r->params.end())
        return r->params.at("__ncnn_batch_axis").i;

    return default_ncnn_batch_axis(r->params.at("__batch_index").i);
}

static void set_ncnn_batch_axis(Operand* r, int batch_axis)
{
    r->params["__ncnn_batch_axis"] = batch_axis;
}

static int normalize_axis(int axis, int rank)
{
    if (axis < 0 && rank > 0)
        axis += rank;

    return axis;
}

static int solve_select_batch_axis(const Operator* op, int batch_axis, int input_rank)
{
    if (batch_axis == 233)
        return 233;

    int dim = normalize_axis(op->params.at("dim").i, input_rank);
    if (dim < 0)
        return batch_axis;

    if (dim == batch_axis)
        return 233;

    if (dim < batch_axis)
        return batch_axis - 1;

    return batch_axis;
}

static std::vector<int> get_slice_selected_axes(const Operator* op, int input_rank)
{
    std::vector<int> axes;
    if (op->has_param("dims"))
        axes = op->params.at("dims").ai;
    else if (op->has_param("dim"))
        axes = std::vector<int> {op->params.at("dim").i};
    else
        return std::vector<int>();

    std::vector<int> steps;
    if (op->has_param("steps"))
        steps = op->params.at("steps").ai;
    else if (op->has_param("step"))
        steps = std::vector<int> {op->params.at("step").i};
    else
        return std::vector<int>();

    if (axes.size() != steps.size())
        return std::vector<int>();

    std::vector<int> selected_axes;
    for (int i = 0; i < (int)axes.size(); i++)
    {
        if (steps[i] != 0)
            continue;

        int axis = normalize_axis(axes[i], input_rank);
        if (axis < 0)
            continue;

        selected_axes.push_back(axis);
    }

    std::sort(selected_axes.begin(), selected_axes.end());
    selected_axes.erase(std::unique(selected_axes.begin(), selected_axes.end()), selected_axes.end());

    return selected_axes;
}

static int solve_slice_batch_axis(const Operator* op, int batch_axis, int input_rank)
{
    if (batch_axis == 233)
        return 233;

    std::vector<int> selected_axes = get_slice_selected_axes(op, input_rank);

    int select_before_batch = 0;
    for (int axis : selected_axes)
    {
        if (axis == batch_axis)
            return 233;

        if (axis < batch_axis)
            select_before_batch += 1;
    }

    return batch_axis - select_before_batch;
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
    return op->type == "Tensor.reshape" || op->type == "Tensor.reshape_as" || op->type == "torch.flatten" || op->type == "Tensor.unflatten";
}

static bool is_recurrent_op(const Operator* op)
{
    return op->type == "nn.RNN" || op->type == "nn.LSTM" || op->type == "nn.GRU";
}

static bool is_binary_eltwise_op(const Operator* op)
{
    return op->type == "BinaryOp" || op->type == "Eltwise";
}

static bool is_layout_transparent_op(const Operator* op)
{
    return is_identity_op(op) || is_elementwise_op(op) || is_layout_op(op);
}

static int get_consumer_ncnn_batch_axis(const Operator* op, int input_index, const Operand* r)
{
    if (op->type == "pnnx.Output")
        return get_ncnn_batch_axis(r);

    if (op->type == "pnnx.Expression" || op->type == "pnnx.SliceIndexes")
        return get_ncnn_batch_axis(r);

    if ((op->type == "Tensor.reshape" || op->type == "Tensor.reshape_as") && input_index != 0)
        return get_ncnn_batch_axis(r);

    if ((op->type == "Tensor.slice" || op->type == "Tensor.select") && input_index != 0)
        return get_ncnn_batch_axis(r);

    if (op->type == "F.scaled_dot_product_attention")
        return r->params.at("__batch_index").i == 233 ? 233 : 0;

    if (op->type == "nn.MultiheadAttention")
    {
        if (op->inputnames.size() == op->inputs.size() && op->inputnames[input_index] == "attn_mask")
            return 233;

        if (r->params.at("__batch_index").i == 233)
            return 233;

        bool batch_first = false;
        if (op->params.find("batch_first") != op->params.end())
            batch_first = op->params.at("batch_first").b;

        return batch_first ? 0 : 1;
    }

    if (is_binary_eltwise_op(op))
    {
        if (r->params.at("__batch_index").i == 233)
            return 233;

        if (!op->outputs.empty())
            return default_ncnn_batch_axis(op->outputs[0]->params.at("__batch_index").i);
    }

    if (is_recurrent_op(op))
        return get_ncnn_batch_axis(r);

    if (is_layout_op(op) && r->params.at("__batch_index").i != 233)
        return r->params.at("__batch_index").i;

    if (is_axis_op(op))
        return get_ncnn_batch_axis(r);

    if (is_reshape_op(op) && input_index == 0 && get_ncnn_batch_axis(r) != 233)
    {
        const int batch_index = r->params.at("__batch_index").i;
        if (batch_index > 0 && batch_index != 233 && r->shape.size() <= 4 && !op->outputs.empty() && !op->outputs[0]->shape.empty() && op->outputs[0]->params.at("__batch_index").i == 233)
            return 233;
    }

    if (is_layout_transparent_op(op) || is_reshape_op(op))
        return get_ncnn_batch_axis(r);

    return default_ncnn_batch_axis(r->params.at("__batch_index").i);
}

static void replace_consumer_input(Operator* op, int input_index, Operand* old_operand, Operand* new_operand)
{
    bool still_consume_old_operand = false;
    for (int i = 0; i < (int)op->inputs.size(); i++)
    {
        if (i != input_index && op->inputs[i] == old_operand)
        {
            still_consume_old_operand = true;
            break;
        }
    }

    if (!still_consume_old_operand)
        old_operand->remove_consumer(op);

    new_operand->consumers.push_back(op);
    op->inputs[input_index] = new_operand;
}

static Operator* insert_batch_to_dim(Graph& graph, Operator* op, int input_index, Operand* in)
{
    if (in->shape.empty())
    {
        fprintf(stderr, "skip batch-to-dim layout rewrite for unknown-rank tensor\n");
        return 0;
    }

    const std::string name = op->name + "_ncnnbatch2dim_" + std::to_string(input_index);

    Operator* reshape = graph.new_operator_before("Tensor.reshape", name, op);
    Operand* reshape_out = graph.new_operand(name + "_out");

    reshape->inputs.push_back(in);
    reshape->outputs.push_back(reshape_out);
    in->consumers.push_back(reshape);

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
    reshape_out->params["__batch_index"] = in->params.at("__batch_index");

    set_ncnn_batch_axis(reshape_out, 233);

    replace_consumer_input(op, input_index, in, reshape_out);

    return reshape;
}

static Operator* insert_dim_to_batch(Graph& graph, Operator* op, int input_index, Operand* in, const std::map<Operand*, int>& batch_indices)
{
    if (in->shape.empty())
    {
        fprintf(stderr, "skip dim-to-batch layout rewrite for unknown-rank tensor\n");
        return 0;
    }

    const std::string name = op->name + "_ncnndim2batch_" + std::to_string(input_index);

    Operator* reshape = graph.new_operator_before("Tensor.reshape", name, op);
    Operand* reshape_out = graph.new_operand(name + "_out");

    reshape->inputs.push_back(in);
    reshape->outputs.push_back(reshape_out);
    in->consumers.push_back(reshape);

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
    reshape_out->params["__batch_index"] = in->params.at("__batch_index");

    set_ncnn_batch_axis(reshape_out, get_batch_index(batch_indices, in));

    replace_consumer_input(op, input_index, in, reshape_out);

    return reshape;
}

static void set_reshape_batch_axis(Operator* op, const std::map<Operand*, int>& batch_indices)
{
    if (op->inputs.empty() || op->outputs.empty())
        return;

    Operand* in = op->inputs[0];
    Operand* out = op->outputs[0];

    const int in_batch_axis = get_ncnn_batch_axis(in);
    const int out_batch_axis = get_ncnn_batch_axis(out);
    if (in_batch_axis == out_batch_axis)
        return;

    if (in_batch_axis == 233 && out_batch_axis != 233)
    {
        const int batch_index = get_batch_index(batch_indices, in);
        const int batch_axis = batch_index == 233 ? get_batch_index(batch_indices, out) : batch_index;
        set_ncnn_batch_axis(out, batch_axis);
    }
}

void convert_batch_layout(Graph& graph)
{
    std::map<Operand*, int> batch_indices;
    for (Operand* r : graph.operands)
    {
        batch_indices[r] = r->params.at("__batch_index").i;
        set_ncnn_batch_axis(r, default_ncnn_batch_axis(batch_indices[r]));

        if (r->producer && r->producer->type == "pnnx.Input" && batch_indices[r] != 233)
            set_ncnn_batch_axis(r, batch_indices[r]);
    }

    std::vector<Operator*> ops = graph.ops;
    for (Operator* op : ops)
    {
        if (op->type == "pnnx.Input")
            continue;

        for (int i = 0; i < (int)op->inputs.size(); i++)
        {
            Operand* in = op->inputs[i];

            const int batch_axis = get_ncnn_batch_axis(in);
            const int required_batch_axis = get_consumer_ncnn_batch_axis(op, i, in);
            if (batch_axis == required_batch_axis)
                continue;

            Operator* reshape = 0;
            if (batch_axis != 233 && required_batch_axis == 233)
                reshape = insert_batch_to_dim(graph, op, i, in);
            if (batch_axis == 233 && required_batch_axis != 233)
                reshape = insert_dim_to_batch(graph, op, i, in, batch_indices);

            if (!reshape)
                continue;

            batch_indices[reshape->outputs[0]] = batch_indices[in];
            set_reshape_batch_axis(reshape, batch_indices);
        }

        if (op->outputs.empty())
            continue;

        if (is_layout_op(op))
        {
            const int batch_axis = op->inputs.empty() ? default_ncnn_batch_axis(get_batch_index(batch_indices, op->outputs[0])) : get_ncnn_batch_axis(op->inputs[0]);
            for (Operand* r : op->outputs)
                set_ncnn_batch_axis(r, batch_axis == 233 ? 233 : get_batch_index(batch_indices, r));
        }
        else if (is_layout_transparent_op(op))
        {
            const int batch_axis = op->inputs.empty() ? default_ncnn_batch_axis(get_batch_index(batch_indices, op->outputs[0])) : get_ncnn_batch_axis(op->inputs[0]);
            for (Operand* r : op->outputs)
                set_ncnn_batch_axis(r, batch_axis);
        }
        else if (is_recurrent_op(op))
        {
            const int batch_axis = op->inputs.empty() ? default_ncnn_batch_axis(get_batch_index(batch_indices, op->outputs[0])) : get_ncnn_batch_axis(op->inputs[0]);
            for (size_t i = 0; i < op->outputs.size(); i++)
            {
                Operand* r = op->outputs[i];
                set_ncnn_batch_axis(r, i == 0 ? batch_axis : default_ncnn_batch_axis(get_batch_index(batch_indices, r)));
            }
        }
        else if (is_reshape_op(op))
        {
            for (Operand* r : op->outputs)
            {
                const int batch_index = get_batch_index(batch_indices, r);
                if (batch_index != 233)
                    set_ncnn_batch_axis(r, batch_index);
                else
                    set_ncnn_batch_axis(r, default_ncnn_batch_axis(batch_index));
            }

            set_reshape_batch_axis(op, batch_indices);
        }
        else if (op->type == "Tensor.select")
        {
            int batch_axis = op->inputs.empty() ? default_ncnn_batch_axis(get_batch_index(batch_indices, op->outputs[0])) : get_ncnn_batch_axis(op->inputs[0]);
            int input_rank = op->inputs.empty() ? 0 : (int)op->inputs[0]->shape.size();
            int output_rank = op->outputs.empty() ? 0 : (int)op->outputs[0]->shape.size();
            if (input_rank == 0 && output_rank > 0)
                input_rank = output_rank + 1;

            batch_axis = solve_select_batch_axis(op, batch_axis, input_rank);
            for (Operand* r : op->outputs)
                set_ncnn_batch_axis(r, batch_axis);
        }
        else if (op->type == "Tensor.slice")
        {
            int batch_axis = op->inputs.empty() ? default_ncnn_batch_axis(get_batch_index(batch_indices, op->outputs[0])) : get_ncnn_batch_axis(op->inputs[0]);
            int input_rank = op->inputs.empty() ? 0 : (int)op->inputs[0]->shape.size();
            batch_axis = solve_slice_batch_axis(op, batch_axis, input_rank);
            for (Operand* r : op->outputs)
                set_ncnn_batch_axis(r, batch_axis);
        }
        else if (op->type == "torch.squeeze")
        {
            int batch_axis = op->inputs.empty() ? default_ncnn_batch_axis(get_batch_index(batch_indices, op->outputs[0])) : get_ncnn_batch_axis(op->inputs[0]);
            if (batch_axis != 233)
            {
                int input_rank = op->inputs[0]->shape.size();
                const std::vector<int>& input_shape = op->inputs[0]->shape;
                if (op->has_param("dim"))
                {
                    int squeeze_before_batch = 0;
                    if (op->params.at("dim").type == 2)
                    {
                        int dim = op->params.at("dim").i;
                        if (dim < 0 && input_rank > 0)
                            dim += input_rank;

                        const bool squeezed = dim >= 0 && dim < input_rank && input_shape[dim] == 1;
                        if (squeezed && dim == batch_axis)
                            batch_axis = 233;
                        else if (squeezed && dim >= 0 && dim < batch_axis)
                            squeeze_before_batch = 1;
                    }
                    else
                    {
                        const std::vector<int>& dims = op->params.at("dim").ai;
                        for (int dim : dims)
                        {
                            if (dim < 0 && input_rank > 0)
                                dim += input_rank;

                            const bool squeezed = dim >= 0 && dim < input_rank && input_shape[dim] == 1;
                            if (squeezed && dim == batch_axis)
                            {
                                batch_axis = 233;
                                break;
                            }
                            if (squeezed && dim >= 0 && dim < batch_axis)
                                squeeze_before_batch += 1;
                        }
                    }
                    if (batch_axis != 233)
                        batch_axis -= squeeze_before_batch;
                }
                else
                {
                    if (input_shape.empty())
                    {
                        fprintf(stderr, "skip squeeze batch axis rewrite for unknown-rank tensor\n");
                    }
                    else if (batch_axis >= 0 && batch_axis < input_rank && input_shape[batch_axis] == 1)
                    {
                        batch_axis = 233;
                    }
                    else
                    {
                        int squeeze_before_batch = 0;
                        for (int i = 0; i < batch_axis; i++)
                        {
                            if (input_shape[i] == 1)
                                squeeze_before_batch += 1;
                        }
                        batch_axis -= squeeze_before_batch;
                    }
                }
            }
            for (Operand* r : op->outputs)
                set_ncnn_batch_axis(r, batch_axis);
        }
        else if (op->type == "torch.unsqueeze")
        {
            int batch_axis = op->inputs.empty() ? default_ncnn_batch_axis(get_batch_index(batch_indices, op->outputs[0])) : get_ncnn_batch_axis(op->inputs[0]);
            if (batch_axis != 233)
            {
                int input_rank = op->inputs[0]->shape.size();
                int output_rank = op->outputs[0]->shape.size();
                int dim = op->params.at("dim").i;
                if (dim < 0 && input_rank > 0)
                    dim += input_rank + 1;
                if (dim < 0 && input_rank == 0 && output_rank > 0)
                    dim += output_rank;

                if (dim >= 0 && dim <= batch_axis)
                    batch_axis += 1;
            }
            for (Operand* r : op->outputs)
                set_ncnn_batch_axis(r, batch_axis);
        }
        else if (is_binary_eltwise_op(op))
        {
            for (Operand* r : op->outputs)
                set_ncnn_batch_axis(r, default_ncnn_batch_axis(get_batch_index(batch_indices, r)));
        }
        else if (op->type == "F.scaled_dot_product_attention")
        {
            for (Operand* r : op->outputs)
                set_ncnn_batch_axis(r, get_batch_index(batch_indices, r) == 233 ? 233 : 0);
        }
        else if (op->type == "nn.MultiheadAttention")
        {
            bool batch_first = false;
            if (op->params.find("batch_first") != op->params.end())
                batch_first = op->params.at("batch_first").b;

            const int batch_axis = batch_first ? 0 : 1;
            for (Operand* r : op->outputs)
                set_ncnn_batch_axis(r, get_batch_index(batch_indices, r) == 233 ? 233 : batch_axis);
        }
        else
        {
            const int batch_axis = op->inputs.empty() ? default_ncnn_batch_axis(get_batch_index(batch_indices, op->outputs[0])) : get_ncnn_batch_axis(op->inputs[0]);
            for (Operand* r : op->outputs)
                set_ncnn_batch_axis(r, batch_axis);
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
