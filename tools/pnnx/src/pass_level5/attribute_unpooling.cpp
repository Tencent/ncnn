// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "attribute_unpooling.h"

#include <algorithm>

namespace pnnx {

void attribute_unpooling(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "pnnx.Attribute")
                continue;

            Operand* attr = op->outputs[0];

            if (attr->consumers.size() < 2)
                continue;

            // multiple modules share same weight
            matched = true;

            for (int i = 1; i < (int)attr->consumers.size(); i++)
            {
                Operator* x = attr->consumers[i];

                Operator* op2 = graph.new_operator_after("pnnx.Attribute", op->name + "_" + std::to_string(i), op);

                op2->inputnames = op->inputnames;
                op2->params = op->params;
                op2->attrs = op->attrs;

                Operand* attr2 = graph.new_operand(attr->name + "_" + std::to_string(i));

                attr2->type = attr->type;
                attr2->shape = attr->shape;
                attr2->params = attr->params;

                op2->outputs.push_back(attr2);

                attr2->producer = op2;
                attr2->consumers.push_back(x);

                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == attr)
                        x->inputs[j] = attr2;
                }
            }

            attr->consumers.resize(1);

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
