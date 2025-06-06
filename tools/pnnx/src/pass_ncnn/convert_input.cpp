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

#include "convert_input.h"

namespace pnnx {

namespace ncnn {

void convert_input(Graph& graph)
{
    int index = 0;

    for (Operator* op : graph.ops)
    {
        if (op->type != "pnnx.Input")
            continue;

        op->type = "Input";
        op->name = std::string("in") + std::to_string(index);

        // canonicalize output name
        op->outputs[0]->name = std::string("in") + std::to_string(index);
        index++;
    }
}

} // namespace ncnn

} // namespace pnnx
