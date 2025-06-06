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

#include "convert_torch_tensor_split.h"

namespace pnnx {

namespace ncnn {

void convert_torch_tensor_split(Graph& graph)
{
    int op_index = 0;

    for (Operator* op : graph.ops)
    {
        if (op->type != "torch.tensor_split")
            continue;

        op->type = "Slice";
        op->name = std::string("tensor_split_") + std::to_string(op_index++);

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        int axis = op->params.at("dim").i;
        if (axis == batch_index)
        {
            fprintf(stderr, "tensor_split along batch axis %d is not supported\n", batch_index);
            continue;
        }

        if (axis < 0)
        {
            int input_rank = op->inputs[0]->shape.size();
            axis = input_rank + axis;
        }

        if (op->params.find("sections") != op->params.end())
        {
            int sections = op->params.at("sections").i;

            if (!op->inputs[0]->shape.empty())
            {
                int size = op->inputs[0]->shape[axis];
                if (size % sections != 0)
                {
                    fprintf(stderr, "tensor_split with non-perfect divided size %d / %d is not supported\n", size, sections);
                }
            }

            op->params["0"].type = 5;
            op->params["0"].ai.resize(sections, -233);

            op->params.erase("sections");
        }
        else
        {
            const std::vector<int>& indices = op->params.at("indices").ai;

            bool has_negative_indice = false;
            for (auto x : indices)
            {
                if (x < 0)
                {
                    // negative indice
                    has_negative_indice = true;
                    break;
                }
            }

            if (has_negative_indice)
            {
                op->params["2"] = indices;
            }
            else
            {
                op->params["0"].type = 5;
                op->params["0"].ai.resize(indices.size() + 1);

                for (size_t i = 0; i < indices.size() + 1; i++)
                {
                    if (i == 0)
                    {
                        op->params["0"].ai[i] = indices[i];
                    }
                    else if (i == indices.size())
                    {
                        op->params["0"].ai[i] = -233;
                    }
                    else
                    {
                        op->params["0"].ai[i] = indices[i] - indices[i - 1];
                    }
                }
            }

            op->params.erase("indices");
        }

        if (axis > batch_index)
            axis -= 1;

        op->params["1"] = axis;
        op->params.erase("dim");
    }
}

} // namespace ncnn

} // namespace pnnx
