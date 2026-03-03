// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_torch_cat.h"

namespace pnnx {

namespace ncnn {

void convert_torch_cat(Graph& graph)
{
    int op_index = 0;

    for (Operator* op : graph.ops)
    {
        if (op->type != "torch.cat")
            continue;

        op->type = "Concat";
        op->name = std::string("cat_") + std::to_string(op_index++);

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        int axis = op->params.at("dim").i;
        if (axis == batch_index)
        {
            fprintf(stderr, "cat along batch axis %d is not supported\n", batch_index);
            continue;
        }

        if (axis < 0)
        {
            int input_rank = op->inputs[0]->shape.size();
            axis = input_rank + axis;
        }

        if (axis > batch_index)
            axis -= 1;

        op->params["0"] = axis;

        op->params.erase("dim");
    }
}

} // namespace ncnn

} // namespace pnnx
