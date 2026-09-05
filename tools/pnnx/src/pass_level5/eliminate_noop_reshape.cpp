// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_noop_reshape.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_noop_reshape(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.reshape")
                continue;

            const std::vector<int>& input_shape = op->inputs[0]->shape;
            const std::vector<int>& output_shape = op->outputs[0]->shape;
            if (input_shape != output_shape)
                continue;

            if (input_shape.empty())
                continue;

            // with -1 (unresolved symbolic/dynamic dims) we cannot prove the
            // reshape is a no-op (the real size may be hidden behind -1, e.g.
            // funnel relative_shift 32->16 compression), skip deletion
            bool has_dynamic = false;
            for (size_t j = 0; j < input_shape.size(); j++)
            {
                if (input_shape[j] == -1)
                {
                    has_dynamic = true;
                    break;
                }
            }

            if (has_dynamic)
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
