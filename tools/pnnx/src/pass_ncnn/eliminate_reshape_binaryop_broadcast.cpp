// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_reshape_binaryop_broadcast.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

static int get_ncnn_batch_axis(const Operand* r)
{
    if (r->params.find("__ncnn_batch_axis") != r->params.end())
        return r->params.at("__ncnn_batch_axis").i;

    return 233;
}

static bool is_channel_broadcast_reshape(const Operand* reshape_in, const std::vector<int>& reshape_shape, const Operand* other)
{
    const std::vector<int>& in_shape = reshape_in->shape;
    const std::vector<int>& other_shape = other->shape;
    const int in_batch_axis = get_ncnn_batch_axis(reshape_in);
    const int other_batch_axis = get_ncnn_batch_axis(other);

    if (in_batch_axis == 233 && other_batch_axis == 233)
    {
        if (in_shape.size() != 1 || reshape_shape.size() < 3 || other_shape.size() != reshape_shape.size())
            return false;

        if (reshape_shape[0] != in_shape[0] || other_shape[0] != in_shape[0])
            return false;

        for (size_t i = 1; i < reshape_shape.size(); i++)
        {
            if (reshape_shape[i] != 1)
                return false;
        }

        return true;
    }

    return false;
}

static bool is_binaryop_channel_broadcast(Operator* op, Operand* binaryop_in, const Operand* reshape_in, const std::vector<int>& reshape_shape)
{
    if (op->type != "BinaryOp")
        return false;
    if (op->inputs.size() != 2)
        return false;

    const int input_index = op->inputs[0] == binaryop_in ? 0 : op->inputs[1] == binaryop_in ? 1 : -1;
    if (input_index == -1)
        return false;

    Operand* other = op->inputs[input_index == 0 ? 1 : 0];
    return is_channel_broadcast_reshape(reshape_in, reshape_shape, other);
}

void eliminate_reshape_binaryop_broadcast(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* reshape = graph.ops[i];

            if (reshape->type != "Reshape")
                continue;
            if (reshape->inputs.size() != 1 || reshape->outputs.size() != 1)
                continue;

            Operand* reshape_in = reshape->inputs[0];
            Operand* reshape_out = reshape->outputs[0];
            const std::vector<int>& reshape_shape = reshape_out->shape;

            if (reshape_out->consumers.empty())
                continue;

            bool can_eliminate = true;
            for (Operator* x : reshape_out->consumers)
            {
                if (!is_binaryop_channel_broadcast(x, reshape_out, reshape_in, reshape_shape))
                {
                    can_eliminate = false;
                    break;
                }
            }

            if (!can_eliminate)
                continue;

            matched = true;

            reshape_in->remove_consumer(reshape);
            for (Operator* x : reshape_out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == reshape_out)
                        x->inputs[j] = reshape_in;
                }

                reshape_in->consumers.push_back(x);
            }

            reshape_out->producer = 0;
            reshape_out->consumers.clear();

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), reshape_out));
            delete reshape_out;

            reshape->inputs.clear();
            reshape->outputs.clear();

            graph.ops.erase(graph.ops.begin() + i);
            delete reshape;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
