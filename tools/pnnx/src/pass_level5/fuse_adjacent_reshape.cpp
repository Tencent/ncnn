// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_adjacent_reshape.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

static bool is_reshape_like_op(const Operator* op)
{
    return op->type == "Tensor.reshape" || op->type == "torch.squeeze" || op->type == "torch.unsqueeze";
}

// compute the output shape of a single reshape-like op applied to `shape`
// returns false when the shape cannot be determined safely
static bool compose_reshape_like_op_shape(const Operator* op, std::vector<int>& shape)
{
    if (op->type == "Tensor.reshape")
    {
        if (op->inputs.size() != 1)
            return false;

        if (op->params.find("shape") == op->params.end())
            return false;

        const std::vector<int>& params = op->params.at("shape").ai;

        std::vector<int> outshape;
        outshape.reserve(params.size());

        int dynamic_dim_count = 0;
        for (size_t i = 0; i < params.size(); i++)
        {
            int dimsize = params[i];

            if (dimsize == 0)
            {
                // copy input dim i
                if (i >= shape.size() || shape[i] == -1)
                    return false;

                dimsize = shape[i];
            }

            if (dimsize == -1)
                dynamic_dim_count += 1;

            if (dynamic_dim_count > 1)
                return false;

            outshape.push_back(dimsize);
        }

        shape = outshape;

        return true;
    }
    else if (op->type == "torch.unsqueeze")
    {
        if (op->inputs.size() != 1)
            return false;

        if (op->params.find("dim") == op->params.end())
            return false;

        int dim = op->params.at("dim").i;
        if (dim < 0)
            dim += (int)shape.size() + 1;

        if (dim < 0 || dim > (int)shape.size())
            return false;

        shape.insert(shape.begin() + dim, 1);

        return true;
    }
    else if (op->type == "torch.squeeze")
    {
        if (op->inputs.size() != 1)
            return false;

        if (op->params.find("dim") == op->params.end())
        {
            // squeeze all size-1 dims
            std::vector<int> outshape;
            outshape.reserve(shape.size());

            for (size_t i = 0; i < shape.size(); i++)
            {
                if (shape[i] == -1)
                    return false;

                if (shape[i] != 1)
                    outshape.push_back(shape[i]);
            }

            shape = outshape;

            return true;
        }

        int dim = op->params.at("dim").i;
        if (dim < 0)
            dim += (int)shape.size();

        if (dim < 0 || dim >= (int)shape.size())
            return false;

        if (shape[dim] == -1)
            return false;

        if (shape[dim] == 1)
            shape.erase(shape.begin() + dim);

        return true;
    }

    return false;
}

void fuse_adjacent_reshape(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (int i = (int)graph.ops.size() - 1; i > 0; i--)
        {
            Operator* op = graph.ops[i];

            // look for Tensor.reshape / torch.squeeze / torch.unsqueeze chain
            if (!is_reshape_like_op(op))
                continue;

            std::vector<Operator*> reshapes_to_delete;
            const Operand* in0 = op->inputs[0];
            while (in0->producer && in0->consumers.size() == 1 && is_reshape_like_op(in0->producer))
            {
                reshapes_to_delete.push_back(in0->producer);
                in0 = in0->producer->inputs[0];
            }

            if (reshapes_to_delete.empty())
                continue;

            // compute the merged reshape output shape
            std::vector<int> merged_shape;
            if (!op->outputs[0]->shape.empty())
            {
                merged_shape = op->outputs[0]->shape;
            }
            else
            {
                // unknown final shape, compose from the last reshape-like op params
                std::vector<int> in_shape = op->inputs[0]->shape;
                if (in_shape.empty() && (op->type == "torch.squeeze" || op->type == "torch.unsqueeze"))
                    continue;

                if (!compose_reshape_like_op_shape(op, in_shape))
                    continue;

                merged_shape = in_shape;
            }

            // the merged reshape param must not contain more than one dynamic dim
            int dynamic_dim_count = 0;
            for (int v : merged_shape)
            {
                if (v == -1)
                    dynamic_dim_count += 1;
            }

            if (dynamic_dim_count > 1)
                continue;

            // keep the last reshape only
            matched = true;

            op->type = "Tensor.reshape";

            op->params.clear();
            op->params["shape"] = merged_shape;

            // drop the dynamic shape input if any
            if (op->inputs.size() == 2)
            {
                op->inputs[1]->remove_consumer(op);
                op->inputs.resize(1);
            }

            for (auto& op0 : reshapes_to_delete)
            {
                for (auto& x : op0->inputs)
                {
                    x->remove_consumer(op0);
                }

                Operand* op0_in = op0->inputs[0];
                Operand* op0_out = op0->outputs[0];

                for (auto& x : op0_out->consumers)
                {
                    for (size_t j = 0; j < x->inputs.size(); j++)
                    {
                        if (x->inputs[j] == op0_out)
                            x->inputs[j] = op0_in;
                    }

                    op0_in->consumers.push_back(x);
                }

                op0_in->name = op0_out->name;

                op0_out->producer = 0;
                op0_out->consumers.clear();

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), op0_out));
                delete op0_out;

                op0->inputs.clear();
                op0->outputs.clear();

                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op0));
                delete op0;
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
