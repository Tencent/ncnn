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

        const int ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;

        int axis = op->params.at("dim").i;
        if (axis < 0)
        {
            int input_rank = op->inputs[0]->shape.size();
            if (input_rank == 0 && !op->outputs.empty())
                input_rank = op->outputs[0]->shape.size();
            if (input_rank > 0)
                axis = input_rank + axis;
            else if (ncnn_batch_axis != 233)
                fprintf(stderr, "cat axis around batch axis %d is unknown\n", ncnn_batch_axis);
        }

        bool axis_is_batch = false;
        if (ncnn_batch_axis != 233 && axis == ncnn_batch_axis)
        {
            fprintf(stderr, "cat along batch axis %d is not supported\n", ncnn_batch_axis);
            axis_is_batch = true;
        }

        if (axis_is_batch)
        {
            // keep Concat op for future across-batch support
            op->params["0"] = -233;

            op->params.erase("dim");
            continue;
        }

        if (ncnn_batch_axis != 233 && axis > ncnn_batch_axis)
            axis -= 1;

        op->params["0"] = axis;

        op->params.erase("dim");
    }
}

} // namespace ncnn

} // namespace pnnx
