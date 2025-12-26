// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_adjacent_permute.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fuse_adjacent_permute(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (int i = (int)graph.ops.size() - 1; i > 0; i--)
        {
            Operator* op = graph.ops[i];

            // look for Tensor.permute / torch.transpose chain
            if (op->type != "Tensor.permute" && op->type != "torch.transpose")
                continue;

            if (op->type == "torch.transpose" && op->outputs[0]->shape.empty())
                continue;

            std::vector<int> permute_dims;
            if (op->type == "Tensor.permute")
            {
                permute_dims = op->params.at("dims").ai;
            }
            if (op->type == "torch.transpose")
            {
                const int shape_rank = (int)op->outputs[0]->shape.size();
                if (shape_rank > 0)
                {
                    permute_dims.resize(shape_rank);
                    for (int j = 0; j < shape_rank; j++)
                    {
                        permute_dims[j] = j;
                    }

                    int dim0 = op->params.at("dim0").i;
                    int dim1 = op->params.at("dim1").i;
                    if (dim0 < 0)
                        dim0 += shape_rank;
                    if (dim1 < 0)
                        dim1 += shape_rank;

                    std::swap(permute_dims[dim0], permute_dims[dim1]);
                }
            }

            if (permute_dims.empty())
                continue;

            const int shape_rank = (int)permute_dims.size();

            std::vector<Operator*> permutes_to_delete;
            const Operand* in0 = op->inputs[0];
            while (in0->consumers.size() == 1 && (in0->producer->type == "Tensor.permute" || in0->producer->type == "torch.transpose"))
            {
                Operator* op0 = in0->producer;
                if (op0->type == "Tensor.permute")
                {
                    const std::vector<int>& dims = op0->params.at("dims").ai;
                    // assert dims.size() == shape_rank
                    for (int j = 0; j < shape_rank; j++)
                    {
                        permute_dims[j] = dims[permute_dims[j]];
                    }
                }
                if (op0->type == "torch.transpose")
                {
                    int dim0 = op0->params.at("dim0").i;
                    int dim1 = op0->params.at("dim1").i;
                    if (dim0 < 0)
                        dim0 += shape_rank;
                    if (dim1 < 0)
                        dim1 += shape_rank;

                    int dim0_j = -1;
                    int dim1_j = -1;
                    for (int j = 0; j < shape_rank; j++)
                    {
                        if (permute_dims[j] == dim0)
                        {
                            dim0_j = j;
                        }
                        if (permute_dims[j] == dim1)
                        {
                            dim1_j = j;
                        }
                    }
                    if (dim0_j == -1 || dim1_j == -1)
                    {
                        // should never reach here
                        continue;
                    }

                    std::swap(permute_dims[dim0_j], permute_dims[dim1_j]);
                }

                permutes_to_delete.push_back(op0);
                in0 = op0->inputs[0];
            }

            if (permutes_to_delete.empty())
                continue;

            // keep the last permute only
            matched = true;

            op->type = "Tensor.permute";

            op->params.clear();
            op->params["dims"] = permute_dims;

            for (auto& op0 : permutes_to_delete)
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
