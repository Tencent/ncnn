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

#include "convert_torch_cat.h"

namespace pnnx {

namespace ncnn {

void convert_torch_cat(Graph& graph)
{
    int op_index = 0;

    for (Operator* op : graph.ops)
    {
        if (op->type != "torch.cat")
            continue;

        op->type = "Concat";
        op->name = std::string("cat_") + std::to_string(op_index++);

        int axis = op->params.at("dim").i;
        if (axis == 0)
        {
            fprintf(stderr, "cat along batch axis is not supported\n");
            continue;
        }

        if (axis < 0)
        {
            int input_rank = op->inputs[0]->shape.size();
            axis = input_rank + axis;
        }

        op->params["0"] = axis - 1;

        op->params.erase("dim");
    }
}

} // namespace ncnn

} // namespace pnnx
