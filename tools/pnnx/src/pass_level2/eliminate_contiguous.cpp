// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_contiguous.h"

#include <algorithm>
#include <vector>

namespace pnnx {

void eliminate_contiguous(Graph& graph)
{
    for (;;)
    {
        bool need_eliminate = false;

        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (op->type != "aten::contiguous" && op->type != "aten::alias")
                continue;
            if (op->inputs.empty() || op->outputs.empty())
                continue;

            // fprintf(stderr, "eliminate_contiguous %s %s\n", op->type.c_str(), op->name.c_str());

            need_eliminate = true;

            Operand* in0 = op->inputs[0];
            Operand* out = op->outputs[0];

            in0->remove_consumer(op);

            std::vector<Operand*> extras;
            for (size_t k = 1; k < op->inputs.size(); k++)
            {
                Operand* extra = op->inputs[k];
                extra->remove_consumer(op);
                extras.push_back(extra);
            }

            for (auto& x : out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == out)
                        x->inputs[j] = in0;
                }

                in0->consumers.push_back(x);
            }

            for (size_t k = 0; k < extras.size(); k++)
            {
                Operand* extra = extras[k];
                if (!extra->consumers.empty())
                    continue;

                Operator* prod = extra->producer;
                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), extra));
                delete extra;
                if (prod && prod->inputs.empty() && prod->outputs.size() <= 1)
                {
                    graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), prod));
                    delete prod;
                }
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
