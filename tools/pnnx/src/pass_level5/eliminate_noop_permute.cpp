// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_noop_permute.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_noop_permute(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.permute")
                continue;

            const std::vector<int>& permute_dims = op->params.at("dims").ai;
            const int shape_rank = (int)permute_dims.size();

            bool is_noop_permute = true;
            for (int i = 0; i < shape_rank; i++)
            {
                if (permute_dims[i] != i)
                {
                    is_noop_permute = false;
                    break;
                }
            }

            if (!is_noop_permute)
                continue;

            matched = true;

            for (auto& x : op->inputs)
            {
                x->remove_consumer(op);
            }

            Operand* op_out = op->outputs[0];

            for (auto& x : op_out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == op_out)
                        x->inputs[j] = op->inputs[0];
                }

                op->inputs[0]->consumers.push_back(x);
            }

            op->inputs[0]->name = op_out->name;

            op_out->producer = 0;
            op_out->consumers.clear();

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), op_out));
            delete op_out;

            op->inputs.clear();
            op->outputs.clear();

            graph.ops.erase(graph.ops.begin() + i);
            delete op;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
