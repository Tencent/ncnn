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

#include "convert_torch_split.h"

namespace pnnx {

namespace ncnn {

void convert_torch_split(Graph& graph)
{
    int op_index = 0;

    for (Operator* op : graph.ops)
    {
        if (op->type != "torch.split")
            continue;

        op->type = "Slice";
        op->name = std::string("split_") + std::to_string(op_index++);

        const Parameter& split_size_or_sections = op->params.at("split_size_or_sections");
        if (split_size_or_sections.type != 2 && split_size_or_sections.type != 5)
        {
            fprintf(stderr, "malformed split split_size_or_sections type %d\n", split_size_or_sections.type);
            continue;
        }

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        int axis = op->params.at("dim").i;
        if (axis == batch_index)
        {
            fprintf(stderr, "split along batch axis %d is not supported\n", batch_index);
            continue;
        }

        if (axis < 0)
        {
            int input_rank = op->inputs[0]->shape.size();
            axis = input_rank + axis;
        }

        if (axis > batch_index)
            axis -= 1;

        if (split_size_or_sections.type == 2)
        {
            const size_t output_size = op->outputs.size();
            op->params["0"].type = 5;
            op->params["0"].ai.resize(output_size, split_size_or_sections.i);
            op->params["0"].ai[output_size - 1] = -233;
        }
        else
        {
            op->params["0"] = split_size_or_sections;
        }

        op->params["1"] = axis;

        op->params.erase("split_size_or_sections");
        op->params.erase("dim");
    }
}

} // namespace ncnn

} // namespace pnnx
