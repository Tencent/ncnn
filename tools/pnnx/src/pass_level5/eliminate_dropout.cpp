// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_dropout.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_dropout(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "F.alpha_dropout" && op->type != "F.dropout" && op->type != "F.dropout2d" && op->type != "F.dropout3d" && op->type != "F.feature_alpha_dropout" && op->type != "nn.AlphaDropout" && op->type != "nn.Dropout" && op->type != "nn.Dropout2d" && op->type != "nn.Dropout3d")
                continue;

            // delete noop-like dropout
            matched = true;

            for (auto& x : op->inputs)
            {
                x->remove_consumer(op);
            }

            Operand* dropout_out = op->outputs[0];

            for (auto& x : dropout_out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == dropout_out)
                        x->inputs[j] = op->inputs[0];
                }

                op->inputs[0]->consumers.push_back(x);
            }

            op->inputs[0]->name = dropout_out->name;

            dropout_out->producer = 0;
            dropout_out->consumers.clear();

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), dropout_out));
            delete dropout_out;

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
