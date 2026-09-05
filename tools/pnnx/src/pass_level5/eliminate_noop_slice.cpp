// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_noop_slice.h"

#include <limits.h>
#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_noop_slice(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.slice")
                continue;

            if (op->inputs.size() != 1)
                continue;

            if (!op->inputs[0]->shape.empty() && op->inputs[0]->shape == op->outputs[0]->shape)
            {
                // with -1 (unresolved symbolic/dynamic dims) we cannot prove the
                // slice is a no-op (the real size may be hidden behind -1, e.g.
                // funnel relative_shift 32->31->16), skip deletion
                bool has_dynamic = false;
                for (int s : op->inputs[0]->shape)
                {
                    if (s == -1)
                    {
                        has_dynamic = true;
                        break;
                    }
                }
                if (!has_dynamic)
                    matched = true;
            }

            if (op->has_param("start") && op->has_param("end") && op->has_param("step"))
            {
                int start = op->params.at("start").i;
                int end = op->params.at("end").i;
                int step = op->params.at("step").i;

                if (start == 0 && end == INT_MAX && step == 1)
                {
                    // delete noop-like slice
                    matched = true;
                }
            }

            if (matched)
            {
                for (auto& x : op->inputs)
                {
                    x->remove_consumer(op);
                }

                Operand* slice_out = op->outputs[0];

                for (auto& x : slice_out->consumers)
                {
                    for (size_t j = 0; j < x->inputs.size(); j++)
                    {
                        if (x->inputs[j] == slice_out)
                            x->inputs[j] = op->inputs[0];
                    }

                    op->inputs[0]->consumers.push_back(x);
                }

                op->inputs[0]->name = slice_out->name;

                slice_out->producer = 0;
                slice_out->consumers.clear();

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), slice_out));
                delete slice_out;

                op->inputs.clear();
                op->outputs.clear();

                graph.ops.erase(graph.ops.begin() + i);
                delete op;

                break;
            }
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
