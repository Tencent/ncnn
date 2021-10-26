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

#include "eliminate_view_reshape.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_view_reshape(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.view" && op->type != "Tensor.reshape")
                continue;

            const std::vector<int>& input_shape = op->inputs[0]->shape;
            const std::vector<int>& output_shape = op->outputs[0]->shape;
            if (input_shape != output_shape)
                continue;

            if (input_shape.empty())
                continue;

            matched = true;

            for (auto& x : op->inputs)
            {
                x->remove_consumer(op);
            }

            for (auto& x : op->outputs[0]->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == op->outputs[0])
                        x->inputs[j] = op->inputs[0];
                }

                op->inputs[0]->consumers.push_back(x);
            }

            op->outputs[0]->producer = 0;
            op->outputs[0]->consumers.clear();

            op->inputs.clear();
            op->outputs.clear();

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));

            delete op;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
