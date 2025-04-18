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

#include "eliminate_identity_operator.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_identity_operator(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op0 = graph.ops[i];

            if (op0->type == "pnnx.Input" || op0->type == "pnnx.Output" || op0->type == "pnnx.Attribute" || op0->type == "torch.clone")
                continue;

            Operator* op1 = 0;

            for (size_t j = i + 1; j < graph.ops.size(); j++)
            {
                op1 = graph.ops[j];

                if (op1->type == "pnnx.Input" || op1->type == "pnnx.Output" || op0->type == "pnnx.Attribute" || op1->type == "torch.clone")
                    continue;

                if (op0->type != op1->type)
                    continue;

                if (op0->inputs != op1->inputs)
                    continue;

                if (op0->outputs.size() != op1->outputs.size())
                    continue;

                if (op0->params != op1->params)
                    continue;

                if (op0->attrs != op1->attrs)
                    continue;

                // we find same operator with same inputs
                matched = true;
                break;
            }

            if (!matched)
                continue;

            // fprintf(stderr, "eliminate_identity_operator %s   %s %s\n", op0->type.c_str(), op0->name.c_str(), op1->name.c_str());

            int input_count = (int)op0->inputs.size();
            for (int j = 0; j < input_count; j++)
            {
                Operand* in0 = op0->inputs[j];

                in0->remove_consumer(op1);
            }

            int output_count = (int)op0->outputs.size();
            for (int j = 0; j < output_count; j++)
            {
                Operand* out0 = op0->outputs[j];
                Operand* out1 = op1->outputs[j];

                for (auto x : out1->consumers)
                {
                    for (size_t k = 0; k < x->inputs.size(); k++)
                    {
                        if (x->inputs[k] == out1)
                            x->inputs[k] = out0;
                    }

                    out0->consumers.push_back(x);
                }

                out1->consumers.clear();
            }

            // delete op1 and its output operands
            for (int j = 0; j < output_count; j++)
            {
                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), op1->outputs[j]));
                delete op1->outputs[j];
            }

            op1->inputs.clear();
            op1->outputs.clear();

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op1));
            delete op1;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
