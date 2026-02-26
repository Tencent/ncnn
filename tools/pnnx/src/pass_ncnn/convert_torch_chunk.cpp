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

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        int axis = op->params.at("dim").i;
        if (axis == batch_index)
        {
            fprintf(stderr, "chunk along batch axis %d is not supported\n", batch_index);
            continue;
        }

        if (axis < 0)
        {
            int input_rank = op->inputs[0]->shape.size();
            axis = input_rank + axis;
        }

        int chunks = op->params.at("chunks").i;

        if (!op->inputs[0]->shape.empty())
        {
            int size = op->inputs[0]->shape[axis];
            if (size % chunks != 0)
            {
                fprintf(stderr, "chunk with non-perfect divided size %d / %d is not supported\n", size, chunks);
            }
        }

        if (axis > batch_index)
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
