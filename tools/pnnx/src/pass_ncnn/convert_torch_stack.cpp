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

#include "convert_torch_stack.h"

namespace pnnx {

namespace ncnn {

void convert_torch_stack(Graph& graph)
{
    int op_index = 0;

    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (op->type != "torch.stack")
                continue;

            matched = true;

            op->type = "Concat";
            op->name = std::string("stack_") + std::to_string(op_index++);

            const int batch_index = op->inputs[0]->params["__batch_index"].i;

            int axis = op->params.at("dim").i;
            if (axis == batch_index)
            {
                fprintf(stderr, "stack along batch axis %d is not supported\n", batch_index);
                continue;
            }

            if (axis < 0)
            {
                int input_rank = op->inputs[0]->shape.size();
                axis = input_rank + axis;
            }

            if (axis > batch_index)
                axis -= 1;

            op->params["0"] = axis;

            op->params.erase("dim");

            // reshape for output, expand the stack dim
            {
                Operand* out = op->outputs[0];

                Operator* reshape = graph.new_operator_after("Tensor.reshape", op->name + "_ncnnreshape", op);

                Operand* reshape_in = graph.new_operand(op->name + "_ncnnreshape_in");

                reshape->inputs.push_back(reshape_in);
                reshape->outputs.push_back(out);

                op->outputs[0] = reshape_in;

                out->producer = reshape;
                reshape_in->producer = op;
                reshape_in->consumers.push_back(reshape);

                reshape->params["shape"] = out->shape;
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
