// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
