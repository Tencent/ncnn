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

#include "insert_reshape_pooling.h"
#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

void insert_reshape_pooling(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "nn.MaxPool1d" && op->type != "nn.MaxPool2d" && op->type != "nn.MaxPool3d")
                continue;

            int input_rank = op->inputs[0]->shape.size();
            if (input_rank == 0)
                continue;

            fprintf(stderr, "insert_reshape_pooling %d\n", input_rank);

            // nn.MaxPool1d    2d-3d-2d
            // nn.MaxPool2d    3d-4d-3d
            // nn.MaxPool3d    4d-5d-4d
            bool insert_reshape = false;
            if ((op->type == "nn.MaxPool1d" && input_rank == 2)
                    || (op->type == "nn.MaxPool2d" && input_rank == 3)
                    || (op->type == "nn.MaxPool3d" && input_rank == 4))
            {
                insert_reshape = true;
            }

            if (!insert_reshape)
                continue;

            matched = true;

            Operand* pooling_in = op->inputs[0];
            Operand* pooling_out = op->outputs[0];

            Operator* reshape0 = graph.new_operator_before("Tensor.reshape", op->name + "_ncnnreshape0", op);
            Operator* reshape1 = graph.new_operator_after("Tensor.reshape", op->name + "_ncnnreshape1", op);

            Operand* reshape0_out = graph.new_operand(op->name + "_ncnnreshape0_out");
            Operand* reshape1_in = graph.new_operand(op->name + "_ncnnreshape1_in");

            reshape0->inputs.push_back(pooling_in);
            reshape0->outputs.push_back(reshape0_out);
            reshape1->inputs.push_back(reshape1_in);
            reshape1->outputs.push_back(pooling_out);

            for (size_t j = 0; j < pooling_in->consumers.size(); j++)
            {
                if (pooling_in->consumers[j] == op)
                {
                    pooling_in->consumers[j] = reshape0;
                    break;
                }
            }
            pooling_out->producer = reshape1;

            op->inputs[0] = reshape0_out;
            op->outputs[0] = reshape1_in;

            reshape0_out->producer = reshape0;
            reshape0_out->consumers.push_back(op);
            reshape1_in->producer = op;
            reshape1_in->consumers.push_back(reshape1);

            std::vector<int> reshape0_shape = pooling_in->shape;
            reshape0_shape.insert(reshape0_shape.begin(), 1);
            std::vector<int> reshape1_shape = pooling_out->shape;

            reshape0->params["shape"] = reshape0_shape;
            reshape1->params["shape"] = reshape1_shape;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
