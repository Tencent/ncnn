// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dead_code_elimination.h"

namespace pnnx {

void dead_code_elimination(Graph& graph)
{
    // dead op elimination
    for (;;)
    {
        bool need_eliminate = false;

        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (op->type == "pnnx.Output")
                continue;

            int consumers = 0;
            for (const Operand* operand : op->outputs)
            {
                consumers += (int)operand->consumers.size();
            }

            if (consumers == 0)
            {
                need_eliminate = true;

                //                 fprintf(stderr, "delete %s %s\n", op->type.c_str(), op->name.c_str());

                for (Operand* operand : op->inputs)
                {
                    operand->remove_consumer(op);
                }

                op->inputs.clear();

                for (Operand* operand : op->outputs)
                {
                    operand->producer = 0;
                }

                op->outputs.clear();

                graph.ops.erase(graph.ops.begin() + i);
                delete op;

                break;
            }
        }

        if (!need_eliminate)
            break;
    }

    // dead operand elimination
    for (;;)
    {
        bool need_eliminate = false;

        for (int i = (int)graph.operands.size() - 1; i >= 0; i--)
        {
            Operand* operand = graph.operands[i];

            int consumers = (int)operand->consumers.size();

            if (operand->producer == 0 && consumers == 0)
            {
                need_eliminate = true;

                //                 fprintf(stderr, "delete operand %s\n", operand->name.c_str());

                graph.operands.erase(graph.operands.begin() + i);
                delete operand;

                break;
            }
        }

        if (!need_eliminate)
            break;
    }
}

} // namespace pnnx
