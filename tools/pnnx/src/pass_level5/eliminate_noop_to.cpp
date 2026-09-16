// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_noop_to.h"

#include <algorithm>

namespace pnnx {

void eliminate_noop_to(Graph& graph)
{
    // remove Tensor.to where input and output dtypes are the same (a no-op
    // cast). On the pt2 path, once constant integer arange folds to an f32
    // Attribute, the downstream Tensor.to(dtype=float) has f32 on both sides
    // and is a safe no-op to remove.
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.to")
                continue;

            if (op->inputs.size() != 1 || op->outputs.size() != 1)
                continue;

            if (op->inputs[0]->type == 0 || op->outputs[0]->type == 0)
                continue;

            if (op->inputs[0]->type != op->outputs[0]->type)
                continue;

            // delete noop-like to
            matched = true;

            for (auto& x : op->inputs)
            {
                x->remove_consumer(op);
            }

            Operand* to_out = op->outputs[0];

            for (auto& x : to_out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == to_out)
                        x->inputs[j] = op->inputs[0];
                }

                op->inputs[0]->consumers.push_back(x);
            }

            op->inputs[0]->name = to_out->name;

            to_out->producer = 0;
            to_out->consumers.clear();

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), to_out));
            delete to_out;

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
