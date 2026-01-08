// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_adjacent_reshape.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fuse_adjacent_reshape(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (int i = (int)graph.ops.size() - 1; i > 0; i--)
        {
            Operator* op = graph.ops[i];

            // look for Tensor.reshape / torch.squeeze / torch.unsqueeze chain
            if (op->type != "Tensor.reshape" && op->type != "torch.squeeze" && op->type != "torch.unsqueeze")
                continue;

            if ((op->type == "torch.squeeze" || op->type == "torch.unsqueeze") && op->outputs[0]->shape.empty())
                continue;

            std::vector<Operator*> reshapes_to_delete;
            const Operand* in0 = op->inputs[0];
            while (in0->consumers.size() == 1 && (in0->producer->type == "Tensor.reshape" || in0->producer->type == "torch.squeeze" || in0->producer->type == "torch.unsqueeze"))
            {
                reshapes_to_delete.push_back(in0->producer);
                in0 = in0->producer->inputs[0];
            }

            if (reshapes_to_delete.empty())
                continue;

            // keep the last reshape only
            matched = true;

            op->type = "Tensor.reshape";

            if (!op->outputs[0]->shape.empty())
            {
                op->params.clear();
                op->params["shape"] = op->outputs[0]->shape;
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
