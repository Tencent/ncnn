// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_noop_copy.h"

#include <algorithm>
#include <vector>

namespace pnnx {

void eliminate_noop_copy(Graph& graph)
{
    for (;;)
    {
        bool need_eliminate = false;

        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (op->type != "aten::lift_fresh_copy" && op->type != "aten::detach_" && op->type != "aten::detach" && op->type != "aten::alias" && op->type != "aten::_to_copy")
                continue;

            // these are single-input single-output identity operations
            if (op->inputs.size() != 1 || op->outputs.size() != 1)
                continue;

            Operand* in0 = op->inputs[0];
            Operand* out = op->outputs[0];

            // _to_copy may change dtype / device / memory_format; only drop it
            // when the dtype is provably unchanged. note: when the guard fails
            // we must NOT have set need_eliminate, or the outer loop would spin
            // forever over this op.
            if (op->type == "aten::_to_copy")
            {
                if (in0->type == 0 || in0->type != out->type)
                    continue;
            }

            need_eliminate = true;

            in0->remove_consumer(op);

            for (auto& x : out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == out)
                        x->inputs[j] = in0;
                }

                in0->consumers.push_back(x);
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
