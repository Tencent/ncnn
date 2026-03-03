// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eliminate_size_numtotensor_int.h"

#include <algorithm>
#include <vector>

namespace pnnx {

void eliminate_size_numtotensor_int(Graph& graph)
{
    // from aten::size - prim::NumToTensor - aten::Int
    //   to aten::size

    for (;;)
    {
        bool need_eliminate = false;

        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];
            if (op->type != "aten::size")
                continue;

            std::vector<Operator*> ops_NumToTensor_to_remove;
            std::vector<Operator*> ops_Int_to_remove;
            std::vector<Operand*> ops_NumToTensor_outputs_to_remove;
            std::vector<Operand*> ops_Int_outputs_to_remove;

            for (auto x : op->outputs[0]->consumers)
            {
                if (x->type != "prim::NumToTensor")
                    continue;

                bool x_is_dead = true;

                for (auto y : x->outputs[0]->consumers)
                {
                    if (y->type != "aten::Int")
                    {
                        x_is_dead = false;
                        continue;
                    }

                    // drop y and y->outputs[0]
                    ops_Int_to_remove.push_back(y);
                    ops_Int_outputs_to_remove.push_back(y->outputs[0]);
                }

                if (x_is_dead)
                {
                    // drop x and x->outputs[0]
                    ops_NumToTensor_to_remove.push_back(x);
                    ops_NumToTensor_outputs_to_remove.push_back(x->outputs[0]);
                }
            }

            if (ops_NumToTensor_to_remove.empty())
                continue;

            if (ops_NumToTensor_to_remove.size() != op->outputs[0]->consumers.size())
                continue;

            need_eliminate = true;

            for (auto x : ops_NumToTensor_to_remove)
            {
                op->outputs[0]->remove_consumer(x);
                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), x));
            }

            for (auto x : ops_Int_to_remove)
            {
                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), x));
            }

            for (auto x : ops_NumToTensor_outputs_to_remove)
            {
                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), x));
            }

            for (auto x : ops_Int_outputs_to_remove)
            {
                // op - x - y is the chain
                for (auto z : x->consumers)
                {
                    for (size_t j = 0; j < z->inputs.size(); j++)
                    {
                        if (z->inputs[j] == x)
                        {
                            z->inputs[j] = op->outputs[0];
                            op->outputs[0]->consumers.push_back(z);
                        }
                    }
                }

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), x));
            }
        }

        if (!need_eliminate)
            break;
    }
}

} // namespace pnnx
