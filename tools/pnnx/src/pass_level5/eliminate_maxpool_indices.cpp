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

#include "eliminate_maxpool_indices.h"

#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void eliminate_maxpool_indices(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "F.adaptive_max_pool1d" && op->type != "F.adaptive_max_pool2d" && op->type != "F.adaptive_max_pool3d"
                    && op->type != "F.max_pool1d" && op->type != "F.max_pool2d" && op->type != "F.max_pool3d"
                    && op->type != "nn.AdaptiveMaxPool1d" && op->type != "nn.AdaptiveMaxPool2d" && op->type != "nn.AdaptiveMaxPool3d"
                    && op->type != "nn.MaxPool1d" && op->type != "nn.MaxPool2d" && op->type != "nn.MaxPool3d")
                continue;

            if (op->outputs.size() != 2)
                continue;

            if (op->params.find("return_indices") == op->params.end())
                continue;

            if (op->params.at("return_indices").b == false)
                continue;

            Operand* op_indices = op->outputs[1];

            if (!op_indices->consumers.empty())
                continue;

            matched = true;

            op->params["return_indices"] = false;
            op->outputs.resize(1);

            op_indices->producer = 0;

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), op_indices));
            delete op_indices;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
