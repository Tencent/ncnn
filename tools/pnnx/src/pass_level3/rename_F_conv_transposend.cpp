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

#include "rename_F_conv_transposend.h"
#include <algorithm>

namespace pnnx {

void rename_F_conv_transposend(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];

        if (op->type != "F.conv_transposend")
            continue;

        Operator* stride = op->inputs[3]->producer;
        if (stride->type != "prim::ListConstruct")
            continue;

        size_t n = stride->inputs.size();
        if (n == 1)
        {
            op->type = "F.conv_transpose1d";
        }
        if (n == 2)
        {
            op->type = "F.conv_transpose2d";
        }
        if (n == 3)
        {
            op->type = "F.conv_transpose3d";
        }
    }
}

} // namespace pnnx
