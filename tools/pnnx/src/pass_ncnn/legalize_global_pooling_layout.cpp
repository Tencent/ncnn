// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "legalize_global_pooling_layout.h"

namespace pnnx {

namespace ncnn {

static int get_ncnn_batch_axis(const Operand* r)
{
    if (r->params.find("__ncnn_batch_axis") != r->params.end())
        return r->params.at("__ncnn_batch_axis").i;

    return 233;
}

static bool is_global_pooling(const Operator* op)
{
    static const char* global_pooling_ops[] = {
        "F.adaptive_avg_pool2d",
        "F.adaptive_avg_pool3d",
        "F.adaptive_max_pool2d",
        "F.adaptive_max_pool3d",
        "nn.AdaptiveAvgPool2d",
        "nn.AdaptiveAvgPool3d",
        "nn.AdaptiveMaxPool2d",
        "nn.AdaptiveMaxPool3d",
    };

    const size_t global_pooling_ops_count = sizeof(global_pooling_ops) / sizeof(const char*);
    for (size_t i = 0; i < global_pooling_ops_count; i++)
    {
        if (op->type != global_pooling_ops[i])
            continue;

        if (!op->has_param("output_size"))
            return false;

        const std::vector<int>& output_size = op->params.at("output_size").ai;
        return output_size == std::vector<int> {1, 1} || output_size == std::vector<int> {1, 1, 1};
    }

    return false;
}

static bool get_compact_shape(const Operand* r, std::vector<int>& compact_shape)
{
    const std::vector<int>& shape = r->shape;
    if (shape.size() == 4 && get_ncnn_batch_axis(r) == 0 && shape[2] == 1 && shape[3] == 1)
    {
        compact_shape = {shape[0], shape[1]};
        return true;
    }
    if (shape.size() == 5 && get_ncnn_batch_axis(r) == 0 && shape[2] == 1 && shape[3] == 1 && shape[4] == 1)
    {
        compact_shape = {shape[0], shape[1]};
        return true;
    }
    if (shape.size() == 3 && get_ncnn_batch_axis(r) == 233 && shape[1] == 1 && shape[2] == 1)
    {
        compact_shape = {shape[0]};
        return true;
    }
    if (shape.size() == 4 && get_ncnn_batch_axis(r) == 233 && shape[1] == 1 && shape[2] == 1 && shape[3] == 1)
    {
        compact_shape = {shape[0]};
        return true;
    }

    return false;
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

    if (op->inputs.size() != 1 || op->outputs.size() != 1)
        return false;

    const size_t elementwise_ops_count = sizeof(elementwise_ops) / sizeof(const char*);
    for (size_t i = 0; i < elementwise_ops_count; i++)
    {
        if (op->type == elementwise_ops[i])
            return true;
    }

    return false;
}

static bool is_zero_padding(const Parameter& p)
{
    if (p.type == 2)
        return p.i == 0;

    if (p.type != 5)
        return false;

    for (int x : p.ai)
    {
        if (x != 0)
            return false;
    }

    return true;
}

static bool is_all_ones(const Parameter& p)
{
    if (p.type == 2)
        return p.i == 1;

    if (p.type != 5)
        return false;

    for (int x : p.ai)
    {
        if (x != 1)
            return false;
    }

    return true;
}

static bool is_1x1_conv(const Operator* op)
{
    if (op->type != "nn.Conv2d")
        return false;

    if (!op->has_param("kernel_size") || !op->has_param("stride") || !op->has_param("padding") || !op->has_param("dilation"))
        return false;

    if (!is_all_ones(op->params.at("kernel_size")))
        return false;
    if (!is_all_ones(op->params.at("stride")))
        return false;
    if (!is_zero_padding(op->params.at("padding")))
        return false;
    if (!is_all_ones(op->params.at("dilation")))
        return false;
    if (op->has_param("groups") && op->params.at("groups").i != 1)
        return false;
    if (op->has_param("padding_mode") && op->params.at("padding_mode").s != "zeros")
        return false;
    if (!op->has_attr("weight"))
        return false;

    const Attribute& weight = op->attrs.at("weight");
    if (weight.shape.size() != 4 || weight.shape[2] != 1 || weight.shape[3] != 1)
        return false;

    return true;
}

static bool replace_consumer_input(Operator* op, int input_index, Operand* old_operand, Operand* new_operand)
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

    return still_consume_old_operand;
}

static bool insert_restore_reshape(Graph& graph, Operator* op, int input_index, Operand* in, const std::vector<int>& shape)
{
    const std::string name = op->name + "_ncnnglobalpoolreshape_" + std::to_string(input_index);

    std::vector<int> reshape_shape = shape;
    const int ncnn_batch_axis = get_ncnn_batch_axis(in);
    if (ncnn_batch_axis == 0 && shape.size() == 4 && shape[2] == 1 && shape[3] == 1)
    {
        reshape_shape = {0, -1, 1, 1};
    }
    if (ncnn_batch_axis == 0 && shape.size() == 5 && shape[2] == 1 && shape[3] == 1 && shape[4] == 1)
    {
        reshape_shape = {0, -1, 1, 1, 1};
    }
    if (ncnn_batch_axis == 233 && shape.size() == 3 && shape[1] == 1 && shape[2] == 1)
    {
        reshape_shape = {-1, 1, 1};
    }
    if (ncnn_batch_axis == 233 && shape.size() == 4 && shape[1] == 1 && shape[2] == 1 && shape[3] == 1)
    {
        reshape_shape = {-1, 1, 1, 1};
    }

    Operator* reshape = graph.new_operator_before("Tensor.reshape", name, op);
    Operand* reshape_out = graph.new_operand(name + "_out");

    reshape->inputs.push_back(in);
    reshape->outputs.push_back(reshape_out);
    reshape->params["shape"] = reshape_shape;
    in->consumers.push_back(reshape);

    reshape_out->producer = reshape;
    reshape_out->type = in->type;
    reshape_out->shape = shape;
    reshape_out->params["__batch_index"] = in->params.at("__batch_index");
    reshape_out->params["__ncnn_batch_axis"] = in->params.at("__ncnn_batch_axis");

    return replace_consumer_input(op, input_index, in, reshape_out);
}

static bool is_spatial_broadcast(const Operand* a, const Operand* b)
{
    const int a_batch_axis = get_ncnn_batch_axis(a);
    const int b_batch_axis = get_ncnn_batch_axis(b);

    if (a_batch_axis == 0 && b_batch_axis == 0)
    {
        if (a->shape.size() != 2 || b->shape.size() < 4)
            return false;

        return a->shape[0] == b->shape[0] && a->shape[1] == b->shape[1];
    }

    if (a_batch_axis == 233 && b_batch_axis == 233)
    {
        if (a->shape.size() != 1 || b->shape.size() < 3)
            return false;

        return a->shape[0] == b->shape[0];
    }

    return false;
}

static bool is_vector_reshape(const Operator* op, const std::vector<int>& compact_shape)
{
    if (op->outputs.empty())
        return false;

    if (op->type == "torch.flatten")
        return op->outputs[0]->shape == compact_shape;

    if (op->type == "Tensor.reshape")
        return op->outputs[0]->shape == compact_shape;

    return false;
}

void legalize_global_pooling_layout(Graph& graph)
{
    std::vector<Operand*> compact_operands;
    std::vector<std::vector<int> > restore_shapes;

    for (Operator* op : graph.ops)
    {
        if (!is_global_pooling(op))
            continue;
        if (op->outputs.empty())
            continue;

        Operand* out = op->outputs[0];
        std::vector<int> compact_shape;
        if (!get_compact_shape(out, compact_shape))
            continue;

        compact_operands.push_back(out);
        restore_shapes.push_back(out->shape);
        out->shape = compact_shape;
    }

    for (size_t i = 0; i < compact_operands.size(); i++)
    {
        Operand* r = compact_operands[i];
        const std::vector<int> restore_shape = restore_shapes[i];
        const std::vector<Operator*> consumers = r->consumers;

        for (size_t j = 0; j < consumers.size(); j++)
        {
            Operator* op = consumers[j];

            int input_index = -1;
            for (int k = 0; k < (int)op->inputs.size(); k++)
            {
                if (op->inputs[k] == r)
                {
                    input_index = k;
                    break;
                }
            }
            if (input_index == -1)
                continue;

            if (op->type == "pnnx.Output")
            {
                if (restore_shape == r->shape)
                    continue;

                insert_restore_reshape(graph, op, input_index, r, restore_shape);
                continue;
            }

            if (is_elementwise_op(op))
            {
                std::vector<int> out_restore_shape = op->outputs[0]->shape;
                if (out_restore_shape.empty())
                    out_restore_shape = restore_shape;

                op->outputs[0]->shape = r->shape;
                compact_operands.push_back(op->outputs[0]);
                restore_shapes.push_back(out_restore_shape);
                continue;
            }

            if (is_vector_reshape(op, r->shape))
            {
                std::vector<int> out_restore_shape = op->outputs[0]->shape;
                if (out_restore_shape.empty())
                    out_restore_shape = r->shape;

                op->type = "Noop";
                op->outputs[0]->shape = r->shape;
                compact_operands.push_back(op->outputs[0]);
                restore_shapes.push_back(out_restore_shape);
                continue;
            }

            if (is_1x1_conv(op) && op->outputs.size() == 1)
            {
                const Attribute& weight = op->attrs.at("weight");
                std::vector<int> out_restore_shape = op->outputs[0]->shape;

                op->type = "nn.Linear";
                op->params.clear();
                op->params["in_features"] = weight.shape[1];
                op->params["out_features"] = weight.shape[0];
                op->params["bias"] = op->has_attr("bias");
                op->attrs["weight"].shape = {weight.shape[0], weight.shape[1]};

                std::vector<int> out_shape = r->shape;
                out_shape[out_shape.size() - 1] = weight.shape[0];
                op->outputs[0]->shape = out_shape;

                if (out_restore_shape.empty())
                {
                    out_restore_shape = restore_shape;
                    out_restore_shape[get_ncnn_batch_axis(r) == 0 ? 1 : 0] = weight.shape[0];
                }

                compact_operands.push_back(op->outputs[0]);
                restore_shapes.push_back(out_restore_shape);
                continue;
            }

            if (op->type == "BinaryOp" && op->inputs.size() == 2 && op->outputs.size() == 1)
            {
                const int other_index = input_index == 0 ? 1 : 0;
                Operand* other = op->inputs[other_index];
                if (is_spatial_broadcast(r, other))
                {
                    op->outputs[0]->shape = other->shape;
                    op->outputs[0]->params["__batch_index"] = other->params.at("__batch_index");
                    op->outputs[0]->params["__ncnn_batch_axis"] = other->params.at("__ncnn_batch_axis");
                    continue;
                }
            }

            if (restore_shape == r->shape)
                continue;

            insert_restore_reshape(graph, op, input_index, r, restore_shape);
        }
    }
}

} // namespace ncnn

} // namespace pnnx
