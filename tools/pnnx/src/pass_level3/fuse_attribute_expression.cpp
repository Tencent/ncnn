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

#include "fuse_attribute_expression.h"
#include <math.h>
#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fuse_attribute_expression(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "pnnx.Attribute")
                continue;

            if (op->outputs.size() != 1)
                continue;

            if (op->outputs[0]->consumers.size() != 1)
                continue;

            Operator* op2 = op->outputs[0]->consumers[0];
            Operator* op3 = 0;
            Operator* op4 = 0;

            float y = 0.f;
            float z = 0.f;

            if (op2->type == "aten::add" || op2->type == "aten::sub")
            {
                if (op2->inputs[0] != op->outputs[0])
                    continue;

                op3 = op2->inputs[1]->producer;
                if (op3->type != "prim::Constant")
                    continue;

                if (op3->params["value"].type == 2)
                {
                    y = op3->params["value"].i;
                }
                else if (op3->params["value"].type == 3)
                {
                    y = op3->params["value"].f;
                }
                else
                {
                    // not a scalar
                    continue;
                }

                op4 = op2->inputs[2]->producer;
                if (op4->type != "prim::Constant")
                    continue;

                if (op4->params["value"].type == 2)
                {
                    z = op4->params["value"].i;
                }
                else if (op4->params["value"].type == 3)
                {
                    z = op4->params["value"].f;
                }
                else
                {
                    // not a scalar
                    continue;
                }
            }
            else if (op2->type == "aten::mul" || op2->type == "aten::div" || op2->type == "aten::pow")
            {
                if (op2->inputs[0] != op->outputs[0])
                    continue;

                op3 = op2->inputs[1]->producer;
                if (op3->type != "prim::Constant")
                    continue;

                if (op3->params["value"].type == 2)
                {
                    y = op3->params["value"].i;
                }
                else if (op3->params["value"].type == 3)
                {
                    y = op3->params["value"].f;
                }
                else
                {
                    // not a scalar
                    continue;
                }
            }
            else
            {
                // todo more operator type
                continue;
            }

            matched = true;

            // apply mul
            {
                auto it = op->attrs.begin();
                std::string attr_key = it->first;
                const Attribute& attr = it->second;

                float* weight = (float*)attr.data.data();
                const int weight_size = attr.data.size() / sizeof(float);

                if (op2->type == "aten::add")
                {
                    for (int i = 0; i < weight_size; i++)
                        weight[i] += y * z;
                }
                else if (op2->type == "aten::sub")
                {
                    for (int i = 0; i < weight_size; i++)
                        weight[i] -= y * z;
                }
                else if (op2->type == "aten::mul")
                {
                    for (int i = 0; i < weight_size; i++)
                        weight[i] *= y;
                }
                else if (op2->type == "aten::div")
                {
                    for (int i = 0; i < weight_size; i++)
                        weight[i] /= y;
                }
                else if (op2->type == "aten::pow")
                {
                    for (int i = 0; i < weight_size; i++)
                        weight[i] = (float)pow(weight[i], y);
                }

                op->attrs[attr_key] = attr;
            }

            op2->outputs[0]->producer = op;

            for (auto& x : op2->inputs)
            {
                x->producer = 0;
                x->remove_consumer(op2);
            }

            op->outputs = op2->outputs;

            op2->inputs.clear();
            op2->outputs.clear();

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op2));

            delete op2;

            if (op3 && op3->outputs[0]->consumers.empty())
            {
                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op3));

                delete op3;
            }

            if (op4 && op4->outputs[0]->consumers.empty())
            {
                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op4));

                delete op4;
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
