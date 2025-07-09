// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_split.h"
#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

void insert_split(Graph& graph)
{
    int opindex = 0;
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];
            for (auto& x : op->outputs)
            {
                if (x->consumers.size() <= 1)
                    continue;

                matched = true;

                // insert split
                Operator* split = graph.new_operator_after("Split", std::string("splitncnn_") + std::to_string(opindex++), op);

                split->inputs.push_back(x);

                for (size_t j = 0; j < x->consumers.size(); j++)
                {
                    Operator* op2 = x->consumers[j];

                    Operand* operand = graph.new_operand(x->name + "_" + std::to_string(j));
                    operand->producer = split;
                    operand->consumers.push_back(op2);

                    operand->params["__batch_index"] = x->params["__batch_index"];

                    split->outputs.push_back(operand);

                    for (size_t k = 0; k < op2->inputs.size(); k++)
                    {
                        if (op2->inputs[k] == x)
                        {
                            op2->inputs[k] = operand;
                            break;
                        }
                    }
                }

                x->consumers.clear();
                x->consumers.push_back(split);

                break;
            }

            if (matched)
                break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
