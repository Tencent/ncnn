// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "expand_quantization_modules.h"
#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void expand_quantization_modules(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type == "nn.intrinsic.quantized.ConvReLU2d")
            {
                op->type = "nn.quantized.Conv2d";
            }
            else if (op->type == "nn.intrinsic.quantized.LinearReLU")
            {
                op->type = "nn.quantized.Linear";
            }
            else
            {
                continue;
            }

            // expand to nn.quantized.Conv2d / nn.quantized.Linear + nn.ReLU

            matched = true;

            // insert new operator before all output consumers
            const Operator* cur = 0;
            {
                int cur_index = graph.ops.size() - 1;
                for (auto& c : op->outputs[0]->consumers)
                {
                    int c_index = std::find(graph.ops.begin(), graph.ops.end(), c) - graph.ops.begin();
                    cur_index = std::min(cur_index, c_index);
                }

                cur = graph.ops[cur_index];
            }

            Operator* op_relu = graph.new_operator_before("nn.ReLU", op->name + "_relu", cur);

            Operand* r0 = graph.new_operand(op->name + "_norelu");

            r0->producer = op;
            r0->consumers.push_back(op_relu);

            op_relu->inputs.push_back(r0);
            op_relu->outputs.push_back(op->outputs[0]);
            op_relu->outputs[0]->producer = op_relu;

            op->outputs[0] = r0;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
