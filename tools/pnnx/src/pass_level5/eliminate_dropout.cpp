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

#include "eliminate_dropout.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_dropout(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "F.alpha_dropout" && op->type != "F.dropout" && op->type != "F.dropout2d" && op->type != "F.dropout3d" && op->type != "F.feature_alpha_dropout" && op->type != "nn.AlphaDropout" && op->type != "nn.Dropout" && op->type != "nn.Dropout2d" && op->type != "nn.Dropout3d")
                continue;

            // delete noop-like dropout
            matched = true;

            for (auto& x : op->inputs)
            {
                x->remove_consumer(op);
            }

            Operand* dropout_out = op->outputs[0];

            for (auto& x : dropout_out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == dropout_out)
                        x->inputs[j] = op->inputs[0];
                }

                op->inputs[0]->consumers.push_back(x);
            }

            dropout_out->producer = 0;
            dropout_out->consumers.clear();

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), dropout_out));
            delete dropout_out;

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
