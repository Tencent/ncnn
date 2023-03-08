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

#include "constant_unpooling.h"

#include <unordered_map>
#include <unordered_set>

namespace pnnx {

void ConstantUnpooling(std::shared_ptr<torch::jit::Graph>& graph, torch::jit::Block* block, std::unordered_set<torch::jit::Node*>& constants)
{
    for (auto it = block->nodes().begin(); it != block->nodes().end();)
    {
        auto node = *it;
        // node may be moved to a different block so advance iterator now
        ++it;

        if (!node->blocks().empty())
        {
            // Traverse sub-blocks.
            for (auto block : node->blocks())
            {
                ConstantUnpooling(graph, block, constants);
            }

            continue;
        }

        for (int i = 0; i < (int)node->inputs().size(); i++)
        {
            const auto& in = node->input(i);

            if (in->node()->kind() != c10::prim::Constant)
                continue;

            // input constant node
            if (constants.find(in->node()) == constants.end())
            {
                constants.insert(in->node());
                continue;
            }

            torch::jit::WithInsertPoint guard(node);

            std::unordered_map<torch::jit::Value*, torch::jit::Value*> value_map;
            auto value_map_func = [&](torch::jit::Value* v) {
                return value_map.at(v);
            };

            //             graph->setInsertPoint(node);

            auto* new_constant_node = graph->insertNode(graph->createClone(in->node(), value_map_func, false));

            //             fprintf(stderr, "new_constant_node %s\n", new_constant_node->outputs()[0]->debugName().c_str());

            // create new constant node
            node->replaceInput(i, new_constant_node->outputs()[0]);
        }
    }
}

void constant_unpooling(std::shared_ptr<torch::jit::Graph>& graph)
{
    std::unordered_set<torch::jit::Node*> constants;
    ConstantUnpooling(graph, graph->block(), constants);
}

} // namespace pnnx
