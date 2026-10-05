// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_Tensor_slice_copy.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

void convert_Tensor_slice_copy(Graph& graph)
{
    int op_index = 0;

    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (op->type != "Tensor.slice_copy")
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
                fprintf(stderr, "slice_copy with dynamic dim is not supported\n");
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
                fprintf(stderr, "slice_copy with dynamic start is not supported\n");
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
                fprintf(stderr, "slice_copy with dynamic end is not supported\n");
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
                fprintf(stderr, "slice_copy with dynamic step is not supported\n");
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
                fprintf(stderr, "slice_copy with dynamic select is not supported\n");
                continue;
            }
            else
            {
                // without select index
            }

            const int axes_rank = axes.size();

            bool unsupported = false;
            std::vector<int> select_axis_indices;
            for (int i = 0; i < axes_rank; i++)
            {
                if (steps[i] == 0)
                {
                    // simulate select as slice_copy
                    starts[i] = selects[i];
                    ends[i] = selects[i] + 1;
                    steps[i] = 1;
                    select_axis_indices.push_back(i);
                }
                else if (steps[i] != 1)
                {
                    fprintf(stderr, "slice_copy with step %d is not supported\n", steps[i]);
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
                    input_rank = op->outputs[0]->shape.size();

                if (ncnn_batch_axis >= 0 && ncnn_batch_axis < input_rank)
                    input_rank -= 1;

                if (input_rank > 4)
                {
                    fprintf(stderr, "slice_copy %d-rank tensor with %d-rank axes is not possible!\n", input_rank, axes_rank);
                }
            }

            int input_rank0 = op->inputs[0]->shape.size();
            if (input_rank0 == 0 && !op->outputs.empty())
                input_rank0 = op->outputs[0]->shape.size();
            std::vector<int> axes_in_shape = axes;
            bool has_select = false;
            std::vector<int> selected_axis_indices;
            for (int i = 0; i < axes_rank; i++)
            {
                if (axes[i] < 0 && input_rank0 > 0)
                {
                    axes[i] = input_rank0 + axes[i];
                }
                axes_in_shape[i] = axes[i];

                if (axes[i] == ncnn_batch_axis)
                {
                    if (starts[i] != 0 || ends[i] != INT_MAX)
                    {
                        fprintf(stderr, "slice_copy along batch axis is not supported\n");
                    }
                    axes[i] = -233;
                    continue;
                }

                if (std::find(select_axis_indices.begin(), select_axis_indices.end(), i) != select_axis_indices.end())
                {
                    has_select = true;
                    selected_axis_indices.push_back(i);
                }

                if (ncnn_batch_axis != 233 && axes[i] > ncnn_batch_axis)
                    axes[i] -= 1;

                if (ends[i] == INT_MAX)
                    ends[i] = -233;
            }
            {
                std::vector<int> axes2;
                std::vector<int> starts2;
                for (int i = 0; i < axes_rank; i++)
                {
                    if (axes[i] == -233)
                        continue;

                    axes2.push_back(axes[i]);
                    starts2.push_back(starts[i]);
                }
                axes = axes2;
                starts = starts2;
            }

            if (axes.empty())
            {
                axes = std::vector<int> {0};
                starts = std::vector<int> {0};
            }

            matched = true;

            op->type = "CopyTo";
            op->name = std::string("slice_copy_") + std::to_string(op_index++);

            // op->params["9"] = starts;
            // op->params["10"] = ends;
            // op->params["11"] = axes;

            op->params["9"] = starts;
            // op->params["10"] = ends; // ncnn always resolve ends from src blob
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

            // reshape for output, squeezing the slice_copy dim
            if (has_select)
            {
                Operand* in = op->inputs[1];

                Operator* reshape = graph.new_operator_before("Tensor.reshape", op->name + "_ncnnreshape", op);

                Operand* reshape_out = graph.new_operand(op->name + "_ncnnreshape_out");

                reshape_out->params["__batch_index"] = batch_index;
                reshape_out->params["__ncnn_batch_axis"] = ncnn_batch_axis;

                reshape->inputs.push_back(in);
                reshape->outputs.push_back(reshape_out);

                op->inputs[1] = reshape_out;

                reshape_out->producer = reshape;
                reshape_out->consumers.push_back(op);
                in->remove_consumer(op);
                in->consumers.push_back(reshape);

                std::vector<int> shape = in->shape;
                for (auto si : selected_axis_indices)
                {
                    // unsqueeze
                    int sa = axes_in_shape[si];
                    if (shape.empty())
                        continue;
                    if (sa < 0 || sa > (int)shape.size())
                        continue;

                    shape.insert(shape.begin() + sa, 1);
                }

                if (!shape.empty())
                    reshape->params["shape"] = shape;
                else
                    reshape->params["shape"] = std::vector<int> {-1};
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
