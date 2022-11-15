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

#include "fuse_adjacent_reshape.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fuse_adjacent_reshape(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (int i = (int)graph.ops.size() - 1; i > 0; i--)
        {
            Operator* op = graph.ops[i];

            // look for Tensor.view / Tensor.reshape / torch.squeeze / torch.unsqueeze chain
            if (op->type != "Tensor.view" && op->type != "Tensor.reshape" && op->type != "torch.squeeze" && op->type != "torch.unsqueeze")
                continue;

            if ((op->type == "torch.squeeze" || op->type == "torch.unsqueeze") && op->outputs[0]->shape.empty())
                continue;

            std::vector<Operator*> reshapes_to_delete;
            const Operand* in0 = op->inputs[0];
            while (in0->consumers.size() == 1 && (in0->producer->type == "Tensor.view" || in0->producer->type == "Tensor.reshape" || in0->producer->type == "torch.squeeze" || in0->producer->type == "torch.unsqueeze"))
            {
                reshapes_to_delete.push_back(in0->producer);
                in0 = in0->producer->inputs[0];
            }

            if (reshapes_to_delete.empty())
                continue;

            // keep the last reshape only
            matched = true;

            op->type = "Tensor.reshape";

            if (!op->outputs[0]->shape.empty())
            {
                op->params.clear();
                op->params["shape"] = op->outputs[0]->shape;
            }

            for (auto& op0 : reshapes_to_delete)
            {
                for (auto& x : op0->inputs)
                {
                    x->remove_consumer(op0);
                }

                Operand* op0_in = op0->inputs[0];
                Operand* op0_out = op0->outputs[0];

                for (auto& x : op0_out->consumers)
                {
                    for (size_t j = 0; j < x->inputs.size(); j++)
                    {
                        if (x->inputs[j] == op0_out)
                            x->inputs[j] = op0_in;
                    }

                    op0_in->consumers.push_back(x);
                }

                op0_in->name = op0_out->name;

                op0_out->producer = 0;
                op0_out->consumers.clear();

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), op0_out));
                delete op0_out;

                op0->inputs.clear();
                op0->outputs.clear();

                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op0));
                delete op0;
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
