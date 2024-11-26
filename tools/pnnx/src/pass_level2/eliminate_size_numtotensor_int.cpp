// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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
