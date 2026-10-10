// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "canonicalize.h"

namespace pnnx {

void canonicalize(Graph& graph)
{
    int i = 0;

    for (Operator* op : graph.ops)
    {
        for (Operand* operand : op->outputs)
        {
            operand->name = std::to_string(i);

            i++;
        }
    }
}

} // namespace pnnx
