// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "eliminate_tail_reshape_permute.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

void eliminate_tail_reshape_permute(Graph& graph)
{
    for (;;)
    {
        bool need_eliminate = false;

        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Reshape" && op->type != "Permute")
                continue;

            Operand* op_out = op->outputs[0];

            if (op_out->consumers.size() != 1)
                continue;

            Operator* op2 = op_out->consumers[0];

            if (op2->type != "pnnx.Output")
                continue;

            need_eliminate = true;

            op->inputs[0]->remove_consumer(op);

            op->inputs[0]->params = op_out->params;

            for (size_t j = 0; j < op2->inputs.size(); j++)
            {
                if (op2->inputs[j] == op_out)
                    op2->inputs[j] = op->inputs[0];
            }

            op->inputs[0]->consumers.push_back(op2);

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
