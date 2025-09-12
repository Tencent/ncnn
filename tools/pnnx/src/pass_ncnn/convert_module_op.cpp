// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_module_op.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

void convert_module_op(Graph& graph, const std::vector<std::string>& module_operators)
{
    for (Operator* op : graph.ops)
    {
        if (std::find(module_operators.begin(), module_operators.end(), op->type) == module_operators.end())
            continue;

        // collect moduleop attribute shape info
        int index = 10;
        for (const auto& it : op->attrs)
        {
            op->params[std::to_string(index)] = it.second.shape;
            index++;
        }
    }
}

} // namespace ncnn

} // namespace pnnx
