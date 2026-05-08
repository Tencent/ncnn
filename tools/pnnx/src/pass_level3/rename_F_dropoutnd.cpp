// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rename_F_dropoutnd.h"
#include <algorithm>

namespace pnnx {

void rename_F_dropoutnd(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];

        if (op->type != "F.dropoutnd")
            continue;

        Operand* r = op->inputs[0];

        size_t input_rank = r->shape.size();
        if (input_rank == 4)
        {
            op->type = "F.dropout2d";
        }
        else if (input_rank == 5)
        {
            op->type = "F.dropout3d";
        }
        else
        {
            fprintf(stderr, "F.dropoutnd fallback to F.dropout2d for unknown input rank\n");
            op->type = "F.dropout2d";
        }
    }
}

} // namespace pnnx
