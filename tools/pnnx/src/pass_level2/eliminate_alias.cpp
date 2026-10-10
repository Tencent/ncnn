// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_alias.h"

#include <algorithm>

namespace pnnx {

void eliminate_alias(Graph& graph)
{
    for (;;)
    {
        bool need_eliminate = false;

        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (op->type != "aten::alias")
                continue;

            need_eliminate = true;

            Operand* in = op->inputs[0];
            Operand* out = op->outputs[0];

            in->remove_consumer(op);

            for (auto& x : out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == out)
                        x->inputs[j] = in;
                }

                in->consumers.push_back(x);
            }

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), out));
            delete out;

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));
            delete op;

            break;
        }

        if (!need_eliminate)
            break;
    }
}

} // namespace pnnx
