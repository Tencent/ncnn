// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_squeeze_unsqueeze_pair.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_squeeze_unsqueeze_pair(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "torch.squeeze")
                continue;

            if (op->outputs[0]->consumers.size() != 1)
                continue;

            Operator* op2 = op->outputs[0]->consumers[0];
            if (op2->type != "torch.unsqueeze")
                continue;

            if (op->inputs.size() != op2->outputs.size())
                continue;

            if (!op->has_param("dim") || !op2->has_param("dim"))
                continue;

            if (op->params["dim"].i != op2->params["dim"].i)
                continue;

            matched = true;

            const size_t count = op->inputs.size();

            for (size_t j = 0; j < count; j++)
            {
                op->inputs[j]->remove_consumer(op);

                for (auto& x : op2->outputs[j]->consumers)
                {
                    op->inputs[j]->consumers.push_back(x);

                    for (size_t k = 0; k < x->inputs.size(); k++)
                    {
                        if (x->inputs[k] == op2->outputs[j])
                            x->inputs[k] = op->inputs[j];
                    }
                }

                op2->outputs[j]->producer = 0;
                op2->outputs[j]->consumers.clear();

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), op2->outputs[j]));
                delete op2->outputs[j];
            }

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), op->outputs[0]));
            delete op->outputs[0];

            op->inputs.clear();
            op->outputs.clear();

            op2->inputs.clear();
            op2->outputs.clear();

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));

            delete op;

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op2));

            delete op2;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
