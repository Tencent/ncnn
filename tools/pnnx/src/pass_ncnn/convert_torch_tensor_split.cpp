// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_torch_tensor_split.h"
#include "reshape_shape.h"
#include "split_shape.h"

namespace pnnx {

namespace ncnn {

void convert_torch_tensor_split(Graph& graph)
{
    int op_index = 0;

    const std::vector<Operator*> ops = graph.ops;
    for (Operator* op : ops)
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

            const int size = axis >= 0 && axis < (int)op->inputs[0]->shape.size() ? op->inputs[0]->shape[axis] : -1;
            const std::string extent = get_logical_dim_expr(op->inputs[0], 0, axis);
            if (size < 0 && !extent.empty())
            {
                // Tensor split gives the first size % sections outputs one extra element.
                const std::string quotient = "//(" + extent + "," + std::to_string(sections) + ")";
                const std::string remainder = "-(" + extent + ",*(" + quotient + "," + std::to_string(sections) + "))";
                std::vector<std::string> boundaries;
                for (int i = 0; i <= sections; i++)
                    boundaries.push_back("+(*(" + std::to_string(i) + "," + quotient + "),min(" + std::to_string(i) + "," + remainder + "))");
                const int physical_axis = ncnn_batch_axis != 233 && axis > ncnn_batch_axis ? axis - 1 : axis;
                split_with_dynamic_crops(graph, op, physical_axis, boundaries);
                continue;
            }

            op->params["0"].type = 5;
            op->params["0"].ai.resize(sections, -233);
            if (size > 0 && size % sections != 0)
            {
                for (int i = 0; i + 1 < sections; i++)
                    op->params["0"].ai[i] = size / sections + (i < size % sections ? 1 : 0);
            }

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
