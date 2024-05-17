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

#include "fuse_slice_to_tensor_split.h"

#include <limits.h>
#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fuse_slice_to_tensor_split(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.slice")
                continue;

            Operand* op_in = op->inputs[0];

            if (op->params.find("dims") == op->params.end()
                    || op->params.find("starts") == op->params.end()
                    || op->params.find("ends") == op->params.end()
                    || op->params.find("steps") == op->params.end())
                continue;

            if (op->params.at("dims").ai.size() != 1)
                continue;

            int dim = op->params.at("dims").ai[0];
            int start = op->params.at("starts").ai[0];
            int end = op->params.at("ends").ai[0];
            int step = op->params.at("steps").ai[0];
            if (start != 0 || step != 1)
                continue;

            // slice 0 i j k ... n
            std::vector<int> tensor_split_indices;
            std::vector<Operator*> slice_n_ops;

            tensor_split_indices.push_back(end);
            slice_n_ops.push_back(op);

            Operator* cur = op;

            bool full_dimsize_slice = false;
            while (1)
            {
                // find slice with starts == end
                Operator* op2 = 0;

                for (auto x : op_in->consumers)
                {
                    if (x->type != "Tensor.slice")
                        continue;

                    if (x->inputs[0] != op_in)
                        continue;

                    if (x->params.find("dims") == x->params.end()
                            || x->params.find("starts") == x->params.end()
                            || x->params.find("ends") == x->params.end()
                            || x->params.find("steps") == x->params.end())
                        continue;

                    if (x->params.at("dims").ai.size() != 1)
                        continue;

                    int dim2 = x->params.at("dims").ai[0];
                    int start2 = x->params.at("starts").ai[0];
                    int step2 = x->params.at("steps").ai[0];
                    if (step2 != 1)
                        continue;

                    if (dim == dim2 && start2 == end)
                    {
                        op2 = x;
                        break;
                    }
                }

                if (!op2)
                    break;

                if (std::find(graph.ops.begin(), graph.ops.end(), op2) < std::find(graph.ops.begin(), graph.ops.end(), cur))
                    cur = op2;

                int end2 = op2->params.at("ends").ai[0];
                if (end2 == INT_MAX)
                {
                    slice_n_ops.push_back(op2);
                    full_dimsize_slice = true;
                    break;
                }
                if (!op_in->shape.empty() && end2 == op_in->shape[dim])
                {
                    slice_n_ops.push_back(op2);
                    full_dimsize_slice = true;
                    break;
                }

                tensor_split_indices.push_back(end2);
                slice_n_ops.push_back(op2);

                end = end2;
            }

            if (!full_dimsize_slice)
                continue;

            matched = true;

            // delete all slice ops and replace with tensor_split
            Operator* op_tensor_split = graph.new_operator_before("torch.tensor_split", op->name, cur);
            op_tensor_split->params["dim"] = dim;
            op_tensor_split->params["indices"] = tensor_split_indices;

            op_tensor_split->inputs.push_back(op_in);
            for (size_t j = 0; j < slice_n_ops.size(); j++)
            {
                op_in->consumers.erase(std::find(op_in->consumers.begin(), op_in->consumers.end(), slice_n_ops[j]));
            }
            op_in->consumers.push_back(op_tensor_split);

            op_tensor_split->outputs.resize(slice_n_ops.size());
            for (size_t j = 0; j < slice_n_ops.size(); j++)
            {
                op_tensor_split->outputs[j] = slice_n_ops[j]->outputs[0];
                slice_n_ops[j]->outputs[0]->producer = op_tensor_split;
            }

            for (size_t j = 0; j < slice_n_ops.size(); j++)
            {
                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), slice_n_ops[j]));
                delete slice_n_ops[j];
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
