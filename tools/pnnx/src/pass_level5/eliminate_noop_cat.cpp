// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_noop_cat.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_noop_cat(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "torch.cat")
                continue;

            if (op->inputs.size() > 1)
                continue;

            // delete noop-like cat
            matched = true;

            op->inputs[0]->remove_consumer(op);

            Operand* cat_out = op->outputs[0];

            for (auto& x : cat_out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == cat_out)
                        x->inputs[j] = op->inputs[0];
                }

                op->inputs[0]->consumers.push_back(x);
            }

            op->inputs[0]->name = cat_out->name;

            cat_out->producer = 0;
            cat_out->consumers.clear();

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), cat_out));
            delete cat_out;

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
