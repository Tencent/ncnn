// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_output.h"

namespace pnnx {

namespace ncnn {

void eliminate_output(Graph& graph)
{
    int output_index = 0;

    for (;;)
    {
        bool need_eliminate = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "pnnx.Output")
                continue;

            need_eliminate = true;

            // canonicalize output name
            for (int j = 0; j < (int)op->inputs.size(); j++)
            {
                op->inputs[j]->name = std::string("out") + std::to_string(output_index);
                output_index++;
            }

            for (Operand* r : op->inputs)
            {
                r->remove_consumer(op);
            }

            op->inputs.clear();

            for (Operand* r : op->outputs)
            {
                r->producer = 0;
            }

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
