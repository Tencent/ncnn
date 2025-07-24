// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_noop.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

void eliminate_noop(Graph& graph)
{
    for (;;)
    {
        bool need_eliminate = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Noop" && op->type != "Tensor.clone")
                continue;

            need_eliminate = true;

            Operand* op_in = op->inputs[0];
            Operand* op_out = op->outputs[0];

            op_in->remove_consumer(op);

            op_in->params = op_out->params;

            for (auto& x : op_out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == op_out)
                        x->inputs[j] = op_in;
                }

                op_in->consumers.push_back(x);
            }

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
