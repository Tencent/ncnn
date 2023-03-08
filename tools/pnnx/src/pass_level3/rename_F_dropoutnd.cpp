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

#include "rename_F_dropoutnd.h"
#include <algorithm>

namespace pnnx {

void rename_F_dropoutnd(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];

        if (op->type != "F.dropoutnd")
            continue;

        Operand* r = op->inputs[0];

        int input_rank = r->shape.size();
        if (input_rank == 4)
        {
            op->type = "F.dropout2d";
        }
        else if (input_rank == 5)
        {
            op->type = "F.dropout3d";
        }
        else
        {
            fprintf(stderr, "F.dropoutnd fallback to F.dropout2d for unknown input rank\n");
            op->type = "F.dropout2d";
        }
    }
}

} // namespace pnnx
