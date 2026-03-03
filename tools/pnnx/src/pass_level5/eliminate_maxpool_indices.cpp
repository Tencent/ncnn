// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
