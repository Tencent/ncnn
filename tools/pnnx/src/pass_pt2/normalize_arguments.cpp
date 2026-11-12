// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_arguments.h"

namespace pnnx {

void normalize_pt2_arguments(Graph& graph)
{
    for (Operator* op : graph.ops)
    {
        int ndim = 0;
        if (op->type == "aten::conv2d" || op->type == "aten::conv_transpose2d")
            ndim = 2;
        if (op->type == "aten::conv3d" || op->type == "aten::conv_transpose3d")
            ndim = 3;
        if (ndim != 0)
        {
            // ATen accepts singleton lists for every spatial convolution argument.
            // Expand them before the shared rules identify the convolution dimension.
            for (size_t i = 3; i < op->inputs.size(); i++)
            {
                if (i == 6) // groups
                    continue;
                Operator* constant = op->inputs[i]->producer;
                if (constant->type != "prim::Constant")
                    continue;
                Parameter& value = constant->params.at("value");
                if (value.type == 5 && value.ai.size() == 1)
                    value = std::vector<int>(ndim, value.ai[0]);
            }
        }

        if (op->type != "aten::slice" || op->inputs.size() != 5)
            continue;
        // Tensor slice permits None bounds; the existing pnnx slice rules use
        // integer bounds. ATen Tensor slicing requires a positive step.
        for (size_t i = 2; i <= 3; i++)
        {
            Operator* constant = op->inputs[i]->producer;
            if (constant->type == "prim::Constant" && constant->params.at("value").type == 0)
                constant->params["value"] = i == 2 ? 0 : INT_MAX;
        }
    }
}

} // namespace pnnx
