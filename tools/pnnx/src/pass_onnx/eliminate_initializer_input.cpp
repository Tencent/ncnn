// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "eliminate_initializer_input.h"

#include <sstream>
#include <string>
#include <unordered_set>

namespace pnnx {

namespace onnx2pnnx {

void eliminate_initializer_input(onnx::ModelProto& model)
{
    // collect initializers
    std::unordered_set<std::string> initializers;
    {
        const onnx::GraphProto& graph = model.graph();
        for (int i = 0; i < graph.initializer_size(); i++)
        {
            initializers.insert(graph.initializer(i).name());
        }
    }

    // collect initializer graph input
    std::vector<int> initializer_input_indexes;
    {
        const onnx::GraphProto& graph = model.graph();
        for (int i = 0; i < graph.input_size(); i++)
        {
            const std::string& input_name = graph.input(i).name();
            if (initializers.find(input_name) == initializers.end())
                continue;

            initializer_input_indexes.push_back(i);
        }
    }

    // eliminate initializer graph input
    {
        onnx::GraphProto* graph = model.mutable_graph();

        for (size_t i = 0; i < initializer_input_indexes.size(); i++)
        {
            const int initializer_input_index = initializer_input_indexes[i];

            //  ..... iii .......
            const int graph_input_size = graph->input_size();
            for (int j = initializer_input_index; j < graph_input_size - 1; j++)
            {
                graph->mutable_input()->SwapElements(j, j + 1);
            }

            //  ..... ....... iii
            graph->mutable_input()->RemoveLast();
        }
    }
}

} // namespace onnx2pnnx

} // namespace pnnx
