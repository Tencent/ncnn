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

#include "flatten_input.h"

namespace pnnx {

void flatten_input(std::shared_ptr<torch::jit::Graph>& graph)
{
    while (1)
    {
        bool matched = false;

        for (torch::jit::Node* n : graph->nodes())
        {
            if (n->kind() != c10::prim::TupleUnpack && n->kind() != c10::prim::ListUnpack)
                continue;

            for (size_t i = 1; i < graph->inputs().size(); i++)
            {
                if (n->input(0) == graph->inputs()[i])
                {
                    matched = true;

                    for (size_t j = 0; j < n->outputs().size(); j++)
                    {
                        torch::jit::Value* v2 = graph->insertInput(i + 1 + j);
                        n->output(j)->replaceAllUsesWith(v2);
                    }
                    n->destroy();
                    graph->eraseInput(i);
                    break;
                }
            }

            if (matched)
                break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
