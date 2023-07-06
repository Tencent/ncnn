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

#include "insert_reshape_numpy_binaryop_broadcast.h"
#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

void insert_reshape_numpy_binaryop_broadcast(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "BinaryOp")
                continue;

            if (op->inputs.size() != 2)
                continue;

            if (op->inputs[0]->shape.empty() || op->inputs[1]->shape.empty())
                continue;

            int batch_index0 = op->inputs[0]->params["__batch_index"].i;
            int batch_index1 = op->inputs[1]->params["__batch_index"].i;
            if (batch_index0 != batch_index1)
            {
                fprintf(stderr, "binaryop broadcast across batch axis %d and %d is not supported\n", batch_index0, batch_index1);
                continue;
            }

            if (op->inputs[0]->shape.size() == 5 && batch_index0 == 233)
            {
                if (op->inputs[0]->shape[0] == 1)
                {
                    fprintf(stderr, "assume reshape 5-rank tensor has batch_index 0\n");
                    batch_index0 = 0;
                }
            }
            if (op->inputs[1]->shape.size() == 5 && batch_index1 == 233)
            {
                if (op->inputs[1]->shape[0] == 1)
                {
                    fprintf(stderr, "assume reshape 5-rank tensor has batch_index 0\n");
                    batch_index1 = 0;
                }
            }

            // drop shape batch index
            std::vector<int> new_shape0;
            std::vector<int> new_shape1;
            for (int j = 0; j < (int)op->inputs[0]->shape.size(); j++)
            {
                if (j == batch_index0 && (op->inputs[0]->shape[j] == 1 || op->inputs[0]->shape[j] == op->inputs[1]->shape[j]))
                    continue;

                new_shape0.push_back(op->inputs[0]->shape[j]);
            }
            for (int j = 0; j < (int)op->inputs[1]->shape.size(); j++)
            {
                if (j == batch_index1 && (op->inputs[1]->shape[j] == 1 || op->inputs[1]->shape[j] == op->inputs[0]->shape[j]))
                    continue;

                new_shape1.push_back(op->inputs[1]->shape[j]);
            }

            const int input_rank0 = (int)new_shape0.size();
            const int input_rank1 = (int)new_shape1.size();

            if (input_rank0 >= 5)
            {
                fprintf(stderr, "binaryop tensor0 with rank %d is not supported yet!\n", (int)op->inputs[0]->shape.size());
            }

            if (input_rank1 >= 5)
            {
                fprintf(stderr, "binaryop tensor1 with rank %d is not supported yet!\n", (int)op->inputs[1]->shape.size());
            }

            if (input_rank0 == input_rank1)
            {
                // no broadcast after ignoring batch index
                continue;
            }

            // fprintf(stderr, "insert_reshape_numpy_binaryop_broadcast %d %d\n", input_rank0, input_rank1);

            matched = true;

            const int binaryop_lower_rank_in_index = input_rank0 < input_rank1 ? 0 : 1;

            Operand* binaryop_lower_rank_in = op->inputs[binaryop_lower_rank_in_index];

            Operator* reshape0 = graph.new_operator_before("Tensor.reshape", op->name + "_ncnnreshape0", op);

            Operand* reshape0_out = graph.new_operand(op->name + "_ncnnreshape0_out");

            reshape0->inputs.push_back(binaryop_lower_rank_in);
            reshape0->outputs.push_back(reshape0_out);

            for (size_t j = 0; j < binaryop_lower_rank_in->consumers.size(); j++)
            {
                if (binaryop_lower_rank_in->consumers[j] == op)
                {
                    binaryop_lower_rank_in->consumers[j] = reshape0;
                    break;
                }
            }

            op->inputs[binaryop_lower_rank_in_index] = reshape0_out;

            reshape0_out->producer = reshape0;
            reshape0_out->consumers.push_back(op);

            reshape0_out->params["__batch_index"] = input_rank0 < input_rank1 ? batch_index0 : batch_index1;

            // insert explicit broadcast index for missing ranks
            std::vector<int> reshape0_shape = input_rank0 < input_rank1 ? new_shape0 : new_shape1;
            for (int j = 0; j < std::abs(input_rank0 - input_rank1); j++)
            {
                reshape0_shape.insert(reshape0_shape.begin(), 1);
            }

            if (batch_index0 != 233)
            {
                reshape0_shape.insert(reshape0_shape.begin() + batch_index0, 1);
            }

            reshape0->params["shape"] = reshape0_shape;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
