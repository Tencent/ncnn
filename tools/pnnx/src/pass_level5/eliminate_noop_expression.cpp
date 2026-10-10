// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_noop_expression.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_noop_expression(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "pnnx.Expression")
                continue;

            if (op->inputs.size() != 1 || op->outputs.size() != 1)
                continue;

            const std::string& expr = op->params.at("expr").s;
            if (expr != "@0")
                continue;

            // delete noop-like expr
            matched = true;

            for (auto& x : op->inputs)
            {
                x->remove_consumer(op);
            }

            Operand* expr_out = op->outputs[0];

            for (auto& x : expr_out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == expr_out)
                        x->inputs[j] = op->inputs[0];
                }

                op->inputs[0]->consumers.push_back(x);
            }

            op->inputs[0]->name = expr_out->name;

            expr_out->producer = 0;
            expr_out->consumers.clear();

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), expr_out));
            delete expr_out;

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
