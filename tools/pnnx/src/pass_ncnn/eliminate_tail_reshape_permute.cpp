// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_tail_reshape_permute.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

void eliminate_tail_reshape_permute(Graph& graph)
{
    for (;;)
    {
        bool need_eliminate = false;

        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Reshape" && op->type != "Permute")
                continue;

            Operand* op_out = op->outputs[0];

            if (op_out->consumers.size() != 1)
                continue;

            Operator* op2 = op_out->consumers[0];

            if (op2->type != "pnnx.Output")
                continue;

            need_eliminate = true;

            op->inputs[0]->remove_consumer(op);

            op->inputs[0]->params = op_out->params;

            for (size_t j = 0; j < op2->inputs.size(); j++)
            {
                if (op2->inputs[j] == op_out)
                    op2->inputs[j] = op->inputs[0];
            }

            op->inputs[0]->consumers.push_back(op2);

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

        if (!need_eliminate)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
