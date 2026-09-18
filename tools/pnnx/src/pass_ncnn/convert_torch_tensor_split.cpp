// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_torch_tensor_split.h"

namespace pnnx {

namespace ncnn {

void convert_torch_tensor_split(Graph& graph)
{
    int op_index = 0;

    for (Operator* op : graph.ops)
    {
        if (op->type != "torch.tensor_split")
            continue;

        op->type = "Slice";
        op->name = std::string("tensor_split_") + std::to_string(op_index++);

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
                fprintf(stderr, "tensor_split axis around batch axis %d is unknown\n", ncnn_batch_axis);
        }

        bool axis_is_batch = false;
        if (ncnn_batch_axis != 233 && axis == ncnn_batch_axis)
        {
            fprintf(stderr, "tensor_split along batch axis %d is not supported\n", ncnn_batch_axis);
            axis_is_batch = true;
        }

        if (op->params.find("sections") != op->params.end())
        {
            int sections = op->params.at("sections").i;

            if (axis_is_batch)
            {
                // keep Slice op for future across-batch support
                op->params["0"].type = 5;
                op->params["0"].ai.resize(sections, -233);

                op->params["1"] = -233;

                op->params.erase("sections");
                op->params.erase("dim");
                continue;
            }

            if (!op->inputs[0]->shape.empty() && axis >= 0 && axis < (int)op->inputs[0]->shape.size())
            {
                int size = op->inputs[0]->shape[axis];
                if (size % sections != 0)
                {
                    fprintf(stderr, "tensor_split with non-perfect divided size %d / %d is not supported\n", size, sections);
                }
            }

            op->params["0"].type = 5;
            op->params["0"].ai.resize(sections, -233);

            op->params.erase("sections");
        }
        else
        {
            const std::vector<int>& indices = op->params.at("indices").ai;

            if (axis_is_batch)
            {
                // keep Slice op for future across-batch support
                op->params["2"] = indices;
                op->params["1"] = -233;

                op->params.erase("indices");
                op->params.erase("dim");
                continue;
            }

            bool has_negative_indice = false;
            for (auto x : indices)
            {
                if (x < 0)
                {
                    // negative indice
                    has_negative_indice = true;
                    break;
                }
            }

            if (has_negative_indice)
            {
                op->params["2"] = indices;
            }
            else
            {
                op->params["0"].type = 5;
                op->params["0"].ai.resize(indices.size() + 1);

                for (size_t i = 0; i < indices.size() + 1; i++)
                {
                    if (i == 0)
                    {
                        op->params["0"].ai[i] = indices[i];
                    }
                    else if (i == indices.size())
                    {
                        op->params["0"].ai[i] = -233;
                    }
                    else
                    {
                        op->params["0"].ai[i] = indices[i] - indices[i - 1];
                    }
                }
            }

            op->params.erase("indices");
        }

        if (ncnn_batch_axis != 233 && axis > ncnn_batch_axis)
            axis -= 1;

        op->params["1"] = axis;
        op->params.erase("dim");
    }
}

} // namespace ncnn

} // namespace pnnx
