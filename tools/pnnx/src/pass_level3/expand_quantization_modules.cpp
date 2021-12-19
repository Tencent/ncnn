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
