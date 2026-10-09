// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_torch_chunk.h"

namespace pnnx {

namespace ncnn {

void convert_torch_chunk(Graph& graph)
{
    int op_index = 0;

    for (Operator* op : graph.ops)
    {
        if (op->type != "torch.chunk")
            continue;

        op->type = "Slice";
        op->name = std::string("chunk_") + std::to_string(op_index++);

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
                fprintf(stderr, "chunk axis around batch axis %d is unknown\n", ncnn_batch_axis);
        }

        bool axis_is_batch = false;
        if (ncnn_batch_axis != 233 && axis == ncnn_batch_axis)
        {
            fprintf(stderr, "chunk along batch axis %d is not supported\n", ncnn_batch_axis);
            axis_is_batch = true;
        }

        int chunks = op->params.at("chunks").i;

        if (axis_is_batch)
        {
            // keep Slice op for future across-batch support
            op->params["0"].type = 5;
            op->params["0"].ai.resize(chunks, -233);

            op->params["1"] = -233;

            op->params.erase("chunks");
            op->params.erase("dim");
            continue;
        }

        if (!op->inputs[0]->shape.empty() && axis >= 0 && axis < (int)op->inputs[0]->shape.size())
        {
            int size = op->inputs[0]->shape[axis];
            if (size % chunks != 0)
            {
                fprintf(stderr, "chunk with non-perfect divided size %d / %d is not supported\n", size, chunks);
            }
        }

        if (ncnn_batch_axis != 233 && axis > ncnn_batch_axis)
            axis -= 1;

        op->params["0"].type = 5;
        op->params["0"].ai.resize(chunks, -233);

        op->params["1"] = axis;

        op->params.erase("chunks");
        op->params.erase("dim");
    }
}

} // namespace ncnn

} // namespace pnnx
