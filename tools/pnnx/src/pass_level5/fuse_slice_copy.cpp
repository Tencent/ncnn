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

#include "fuse_slice_copy.h"

#include <limits.h>
#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fuse_slice_copy(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.copy")
                continue;

            Operator* op_slice = op->inputs[0]->producer;

            if (op_slice->type != "Tensor.slice")
                continue;

            if (op_slice->params.find("dims") == op_slice->params.end()
                    || op_slice->params.find("starts") == op_slice->params.end()
                    || op_slice->params.find("ends") == op_slice->params.end()
                    || op_slice->params.find("steps") == op_slice->params.end())
                continue;

            matched = true;

            op->type = "Tensor.slice_copy";

            op->inputs[0]->remove_consumer(op);
            op->inputs[0] = op_slice->inputs[0];
            op_slice->inputs[0]->consumers.push_back(op);

            op->params["dims"] = op_slice->params["dims"];
            op->params["starts"] = op_slice->params["starts"];
            op->params["ends"] = op_slice->params["ends"];
            op->params["steps"] = op_slice->params["steps"];

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
