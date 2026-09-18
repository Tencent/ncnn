// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_torch_split.h"

namespace pnnx {

namespace ncnn {

void convert_torch_split(Graph& graph)
{
    int op_index = 0;

    for (Operator* op : graph.ops)
    {
        if (op->type != "torch.split")
            continue;

        op->type = "Slice";
        op->name = std::string("split_") + std::to_string(op_index++);

        const Parameter& split_size_or_sections = op->params.at("split_size_or_sections");
        if (split_size_or_sections.type != 2 && split_size_or_sections.type != 5)
        {
            fprintf(stderr, "malformed split split_size_or_sections type %d\n", split_size_or_sections.type);
            continue;
        }

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
                fprintf(stderr, "split axis around batch axis %d is unknown\n", ncnn_batch_axis);
        }

        bool axis_is_batch = false;
        if (ncnn_batch_axis != 233 && axis == ncnn_batch_axis)
        {
            fprintf(stderr, "split along batch axis %d is not supported\n", ncnn_batch_axis);
            axis_is_batch = true;
        }

        if (axis_is_batch)
        {
            // keep Slice op for future across-batch support
            if (split_size_or_sections.type == 2)
            {
                const size_t output_size = op->outputs.size();
                op->params["0"].type = 5;
                op->params["0"].ai.resize(output_size, split_size_or_sections.i);
                op->params["0"].ai[output_size - 1] = -233;
            }
            else
            {
                op->params["0"] = split_size_or_sections;
            }

            op->params["1"] = -233;

            op->params.erase("split_size_or_sections");
            op->params.erase("dim");
            continue;
        }

        if (ncnn_batch_axis != 233 && axis > ncnn_batch_axis)
            axis -= 1;

        if (split_size_or_sections.type == 2)
        {
            const size_t output_size = op->outputs.size();
            op->params["0"].type = 5;
            op->params["0"].ai.resize(output_size, split_size_or_sections.i);
            op->params["0"].ai[output_size - 1] = -233;
        }
        else
        {
            op->params["0"] = split_size_or_sections;
        }

        op->params["1"] = axis;

        op->params.erase("split_size_or_sections");
        op->params.erase("dim");
    }
}

} // namespace ncnn

} // namespace pnnx
