// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "fix_inplace_copy_output.h"
#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fix_inplace_copy_output(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.copy")
                continue;

            if (op->outputs[0]->consumers.size() != 0)
                continue;

            matched = true;

            // Tensor.slice   5 1 in0 .... a
            // Tensor.slice   5 1 a .... b
            // Tensor.copy    2 1 b in1 out

            // find in0 from slice chain
            Operand* in0 = op->inputs[0];
            while (in0->producer->type == "Tensor.slice")
            {
                in0 = in0->producer->inputs[0];
            }

            // replace all the following uses of in0 with out
            Operand* out0 = op->outputs[0];
            for (size_t j = i; j < graph.ops.size(); j++)
            {
                Operator* op2 = graph.ops[j];

                bool use_in0 = false;
                for (size_t k = 0; k < op2->inputs.size(); k++)
                {
                    if (op2->inputs[k] == in0)
                    {
                        op2->inputs[k] = out0;
                        use_in0 = true;
                    }
                }

                if (use_in0)
                {
                    in0->remove_consumer(op2);
                    out0->consumers.push_back(op2);
                }
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
