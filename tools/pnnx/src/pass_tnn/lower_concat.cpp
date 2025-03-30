// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "lower_concat.h"

#include "pass_level2.h"

namespace pnnx {

namespace tnn2pnnx {

void lower_concat(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];

        if (op->type != "tnn.Concat")
            continue;

        const int dim = op->params["arg0"].i;

        op->type = "aten::cat";
        op->params.clear();
        op->params["dim"] = dim;

        // insert listconstruct for inputs
        Operator* op0 = graph.new_operator_before("prim::ListConstruct", op->name + "_lc", op);
        Operand* r = graph.new_operand(op->name + "_lc");

        r->producer = op0;
        r->consumers.push_back(op);

        op0->outputs.push_back(r);

        for (size_t j = 0; j < op->inputs.size(); j++)
        {
            Operand* x = op->inputs[j];

            x->remove_consumer(op);
            x->consumers.push_back(op0);
            op0->inputs.push_back(x);
        }

        op->inputs.clear();
        op->inputs.push_back(r);
    }
}

} // namespace tnn2pnnx

} // namespace pnnx
