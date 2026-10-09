// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_torch_chunk.h"
#include "reshape_shape.h"
#include "split_shape.h"

namespace pnnx {

namespace ncnn {

void convert_torch_chunk(Graph& graph)
{
    int op_index = 0;

    const std::vector<Operator*> ops = graph.ops;
    for (Operator* op : ops)
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

        const int size = axis >= 0 && axis < (int)op->inputs[0]->shape.size() ? op->inputs[0]->shape[axis] : -1;
        const std::string extent = get_logical_dim_expr(op->inputs[0], 0, axis);
        if (size < 0 && !extent.empty())
        {
            // Chunk uses ceil(size / chunks), and can return fewer than chunks outputs.
            const std::string step = "+(1,//(-(" + extent + ",1)," + std::to_string(chunks) + "))";
            std::vector<std::string> boundaries;
            for (size_t i = 0; i <= op->outputs.size(); i++)
                boundaries.push_back("min(" + extent + ",*(" + std::to_string(i) + "," + step + "))");
            const int physical_axis = ncnn_batch_axis != 233 && axis > ncnn_batch_axis ? axis - 1 : axis;
            split_with_dynamic_crops(graph, op, physical_axis, boundaries);
            continue;
        }

        if (ncnn_batch_axis != 233 && axis > ncnn_batch_axis)
            axis -= 1;

        op->params["0"].type = 5;
        op->params["0"].ai.resize(op->outputs.size(), -233);
        if (size > 0 && size % chunks != 0)
        {
            const int step = 1 + (size - 1) / chunks;
            for (size_t i = 0; i + 1 < op->outputs.size(); i++)
                op->params["0"].ai[i] = step;
        }

        op->params["1"] = axis;

        op->params.erase("chunks");
        op->params.erase("dim");
    }
}

} // namespace ncnn

} // namespace pnnx
