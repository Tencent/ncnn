// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_einsum_operands.h"
#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fuse_einsum_operands(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "torch.einsum")
                continue;

            if (op->inputs.size() < 1)
                continue;

            Operand* operands = op->inputs.size() == 1 ? op->inputs[0] : op->inputs[1];

            if (operands->consumers.size() != 1)
                continue;

            Operator* op2 = operands->producer;
            if (op2->type != "prim::ListConstruct")
                continue;

            matched = true;

            operands->producer = 0;
            operands->remove_consumer(op);

            std::vector<Operand*> new_inputs;
            std::vector<std::string> new_inputnames;

            if (op->inputs.size() > 1)
            {
                new_inputs.push_back(op->inputs[0]);
                new_inputnames.push_back(op->inputnames[0]);
            }

            for (auto& x : op2->inputs)
            {
                x->remove_consumer(op2);
                x->consumers.push_back(op);
                new_inputs.push_back(x);
                new_inputnames.push_back("");
            }

            for (size_t j = 2; j < op->inputs.size(); j++)
            {
                new_inputs.push_back(op->inputs[j]);
                new_inputnames.push_back(op->inputnames[j]);
            }

            op->inputs = new_inputs;
            op->inputnames = new_inputnames;

            op2->inputs.clear();
            op2->outputs.clear();

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op2));

            delete op2;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
