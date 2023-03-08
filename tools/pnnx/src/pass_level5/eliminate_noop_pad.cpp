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

#include "eliminate_noop_pad.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_noop_pad(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "F.pad")
                continue;

            if (op->params.find("pad") == op->params.end())
                continue;

            const std::vector<int>& pad = op->params.at("pad").ai;

            bool noop_pad = true;
            for (auto p : pad)
            {
                if (p != 0)
                {
                    noop_pad = false;
                    break;
                }
            }

            if (!noop_pad)
                continue;

            // delete noop-like pad
            matched = true;

            for (auto& x : op->inputs)
            {
                x->remove_consumer(op);
            }

            Operand* pad_out = op->outputs[0];

            for (auto& x : pad_out->consumers)
            {
                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == pad_out)
                        x->inputs[j] = op->inputs[0];
                }

                op->inputs[0]->consumers.push_back(x);
            }

            op->inputs[0]->name = pad_out->name;

            pad_out->producer = 0;
            pad_out->consumers.clear();

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), pad_out));
            delete pad_out;

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
