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

            if (op->type != "aten::contiguous")
                continue;

            // fprintf(stderr, "eliminate_contiguous %s %s\n", op->type.c_str(), op->name.c_str());

            need_eliminate = true;

            Operand* in0 = op->inputs[0];
            Operand* in1 = op->inputs[1];
            Operand* out = op->outputs[0];

            in0->remove_consumer(op);
            in1->remove_consumer(op);

            for (auto& x : out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == out)
                        x->inputs[j] = in0;
                }

                in0->consumers.push_back(x);
            }

            if (in1->consumers.empty())
            {
                Operator* op_memory_format = in1->producer;

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), in1));
                delete in1;

                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op_memory_format));
                delete op_memory_format;
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
