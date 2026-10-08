// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_Tensor_slice.h"
#include "reshape_shape.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

void convert_Tensor_slice(Graph& graph)
{
    int op_index = 0;

    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (op->type != "Tensor.slice")
                continue;

            std::vector<int> axes;
            std::vector<int> starts;
            std::vector<int> ends;
            std::vector<int> steps;
            std::vector<int> selects;

            if (op->has_param("dims"))
            {
                axes = op->params.at("dims").ai;
            }
            else if (op->has_param("dim"))
            {
                axes = std::vector<int> {op->params.at("dim").i};
            }
            else
            {
                fprintf(stderr, "slice with dynamic dim is not supported\n");
                continue;
            }

            if (op->has_param("starts"))
            {
                starts = op->params.at("starts").ai;
            }
            else if (op->has_param("start"))
            {
                starts = std::vector<int> {op->params.at("start").i};
            }
            else
            {
                fprintf(stderr, "slice with dynamic start is not supported\n");
                continue;
            }

            if (op->has_param("ends"))
            {
                ends = op->params.at("ends").ai;
            }
            else if (op->has_param("end"))
            {
                ends = std::vector<int> {op->params.at("end").i};
            }
            else
            {
                fprintf(stderr, "slice with dynamic end is not supported\n");
                continue;
            }

            if (op->has_param("steps"))
            {
                steps = op->params.at("steps").ai;
            }
            else if (op->has_param("step"))
            {
                steps = std::vector<int> {op->params.at("step").i};
            }
            else
            {
                fprintf(stderr, "slice with dynamic step is not supported\n");
                continue;
            }

            if (op->has_param("selects"))
            {
                selects = op->params.at("selects").ai;
            }
            else if (op->has_param("select"))
            {
                selects = std::vector<int> {op->params.at("select").i};
            }
            else if (op->has_input("selects") || op->has_input("select"))
            {
                fprintf(stderr, "slice with dynamic select is not supported\n");
                continue;
            }
            else
            {
                // without select index
            }

            const int axes_rank = axes.size();

            int select_count = 0;
            bool unsupported = false;
            std::vector<int> select_axis_indices;
            for (int i = 0; i < axes_rank; i++)
            {
                if (steps[i] == 0)
                {
                    // simulate select as slice
                    starts[i] = selects[i];
                    ends[i] = selects[i] + 1;
                    steps[i] = 1;
                    select_axis_indices.push_back(i);
                }
                else if (steps[i] != 1)
                {
                    fprintf(stderr, "slice with step %d is not supported\n", steps[i]);
                    unsupported = true;
                    break;
                }
            }
            if (unsupported)
                continue;

            const int batch_index = op->inputs[0]->params["__batch_index"].i;
            const int ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;

            {
                int input_rank = op->inputs[0]->shape.size();
                if (input_rank == 0 && !op->outputs.empty())
                    input_rank = op->outputs[0]->shape.size() + select_axis_indices.size();

                if (ncnn_batch_axis >= 0 && ncnn_batch_axis < input_rank)
                    input_rank -= 1;

                if (input_rank > 4)
                {
                    fprintf(stderr, "slice %d-rank tensor with %d-rank axes is not possible!\n", input_rank, axes_rank);
                }
            }

            int input_rank0 = op->inputs[0]->shape.size();
            if (input_rank0 == 0 && !op->outputs.empty())
                input_rank0 = op->outputs[0]->shape.size() + select_axis_indices.size();
            std::vector<int> select_logical_axes;
            for (int i = 0; i < axes_rank; i++)
            {
                if (axes[i] < 0 && input_rank0 > 0)
                {
                    axes[i] = input_rank0 + axes[i];
                }

                if (axes[i] == ncnn_batch_axis)
                {
                    if (starts[i] != 0 || ends[i] != INT_MAX)
                        fprintf(stderr, "slice along batch axis is not supported\n");
                    axes[i] = -233;
                    continue;
                }

                if (std::find(select_axis_indices.begin(), select_axis_indices.end(), i) != select_axis_indices.end())
                {
                    select_count += 1;
                    select_logical_axes.push_back(axes[i]);
                }

                if (ncnn_batch_axis != 233 && axes[i] > ncnn_batch_axis)
                    axes[i] -= 1;

                if (ends[i] == INT_MAX)
                    ends[i] = -233;
            }
            Operand reshape_shape = *op->outputs[0];
            std::map<std::string, Parameter> reshape_params;
            if (select_count > 0)
            {
                reshape_shape.producer = op;
                reshape_shape.params["__ncnn_batch_axis"] = ncnn_batch_axis;
                reshape_shape.type = op->outputs[0]->type;
                reshape_shape.shape = op->outputs[0]->shape;
                std::sort(select_logical_axes.begin(), select_logical_axes.end());
                if (reshape_shape.shape.empty())
                    reshape_shape.shape.resize(input_rank0 - select_count, -1);
                for (int axis : select_logical_axes)
                    reshape_shape.shape.insert(reshape_shape.shape.begin() + axis, 1);

                auto shape = logical_shape(&reshape_shape, 0);
                for (auto it = select_logical_axes.rbegin(); it != select_logical_axes.rend(); ++it)
                    shape.erase(shape.begin() + *it);
                if (!resolve_reshape_shape({&reshape_shape}, op->outputs[0], shape, reshape_params))
                    continue;
            }

            matched = true;

            op->type = "Crop";
            op->name = std::string("slice_") + std::to_string(op_index++);

            {
                std::vector<int> axes2;
                std::vector<int> starts2;
                std::vector<int> ends2;
                for (int i = 0; i < axes_rank; i++)
                {
                    if (axes[i] == -233)
                        continue;

                    axes2.push_back(axes[i]);
                    starts2.push_back(starts[i]);
                    ends2.push_back(ends[i]);
                }
                axes = axes2;
                starts = starts2;
                ends = ends2;
            }

            if (axes.empty())
            {
                axes = std::vector<int> {0};
                starts = std::vector<int> {0};
                ends = std::vector<int> {-233};
            }

            op->params["9"] = starts;
            op->params["10"] = ends;
            op->params["11"] = axes;

            op->params.erase("dim");
            op->params.erase("dims");
            op->params.erase("start");
            op->params.erase("starts");
            op->params.erase("end");
            op->params.erase("ends");
            op->params.erase("step");
            op->params.erase("steps");
            op->params.erase("select");
            op->params.erase("selects");

            // reshape for output, squeezing the slice dim
            if (select_count > 0)
            {
                Operand* out = op->outputs[0];

                Operator* reshape = graph.new_operator_after("Reshape", op->name + "_ncnnreshape", op);

                Operand* reshape_in = graph.new_operand(op->name + "_ncnnreshape_in");

                reshape_in->params["__batch_index"] = batch_index;
                reshape_in->params["__ncnn_batch_axis"] = ncnn_batch_axis;

                reshape->inputs.push_back(reshape_in);
                reshape->outputs.push_back(out);

                op->outputs[0] = reshape_in;

                out->producer = reshape;
                reshape_in->producer = op;
                reshape_in->consumers.push_back(reshape);

                reshape_in->type = reshape_shape.type;
                reshape_in->shape = reshape_shape.shape;
                write_reshape_shape(reshape, reshape_params);
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
