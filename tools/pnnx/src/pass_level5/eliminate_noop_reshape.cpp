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

            // matching input/output metadata does not prove identity
            // an external shape reference or infer may change on another input
            if (op->inputs.size() != 1 || !op->has_param("shape"))
                continue;

            const Operand* input = op->inputs[0];
            const std::vector<int>& target_shape = op->params.at("shape").ai;
            if (target_shape.empty())
                continue;

            bool identity = input->shape.size() == 1 && target_shape.size() == 1 && target_shape[0] == -1;
            if (!identity && (input->producer->type == "pnnx.Input" || input->producer->type == "pnnx.Attribute") && target_shape == input->shape)
            {
                identity = true;
                for (int s : target_shape)
                {
                    if (s <= 0)
                    {
                        identity = false;
                        break;
                    }
                }
            }

            if (!identity)
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
