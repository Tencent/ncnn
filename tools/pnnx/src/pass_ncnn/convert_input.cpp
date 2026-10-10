// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
