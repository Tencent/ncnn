// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_scalar.h"

namespace pnnx {

void normalize_pt2_scalar(Graph& graph)
{
    for (Operator* op : graph.ops)
    {
        // Tensor returns have metadata types. Only exported scalar results
        // lack a tensor dtype, including comparisons of Python scalar values.
        if (op->outputs.size() != 1 || op->outputs[0]->type != 0)
            continue;

        if (op->type == "aten::item" && op->inputs.size() == 1)
        {
            op->type = "Tensor.item";
            continue;
        }

        if (op->inputs.size() != 2)
            continue;

        if (op->type == "aten::eq" || op->type == "aten::ne" || op->type == "aten::lt"
            || op->type == "aten::le" || op->type == "aten::gt" || op->type == "aten::ge")
        {
            op->params["expr"] = op->type.substr(6) + "(@0,@1)";
            op->type = "pnnx.Expression";
        }
        else if (op->type == "aten::__or__")
        {
            op->params["expr"] = "or(@0,@1)";
            op->type = "pnnx.Expression";
        }
    }
}

} // namespace pnnx
