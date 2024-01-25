// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "convert_Tensor_slice_copy.h"

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

            matched = true;

            op->type = "CopyTo";
            op->name = std::string("slice_copy_") + std::to_string(op_index++);

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

            bool has_select = false;
            std::vector<int> selected_axes;
            for (int i = 0; i < axes_rank; i++)
            {
                axes[i] += selected_axes.size();

                if (steps[i] == 0)
                {
                    // simulate select as slice_copy
                    starts[i] = selects[i];
                    ends[i] = selects[i] + 1;
                    steps[i] = 1;
                    has_select = true;
                    selected_axes.push_back(axes[i]);
                }
                else if (steps[i] != 1)
                {
                    fprintf(stderr, "slice_copy with step %d is not supported\n", steps[i]);
                }
            }

            const int batch_index = op->inputs[0]->params["__batch_index"].i;

            {
                int input_rank = op->inputs[0]->shape.size();

                if (batch_index >= 0 && batch_index < input_rank)
                    input_rank -= 1;

                if (input_rank > 4)
                {
                    fprintf(stderr, "slice_copy %d-rank tensor with %d-rank axes is not possible!\n", input_rank, axes_rank);
                    continue;
                }
            }

            for (int i = 0; i < axes_rank; i++)
            {
                if (axes[i] == batch_index && (starts[i] != 0 || ends[i] != INT_MAX))
                {
                    fprintf(stderr, "slice_copy along batch axis is not supported\n");
                    continue;
                }

                if (axes[i] < 0)
                {
                    int input_rank = op->inputs[0]->shape.size();
                    axes[i] = input_rank + axes[i];
                }

                if (axes[i] > batch_index)
                    axes[i] -= 1;

                if (ends[i] == INT_MAX)
                    ends[i] = -233;
            }

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

                reshape->inputs.push_back(in);
                reshape->outputs.push_back(reshape_out);

                op->inputs[1] = reshape_out;

                reshape_out->producer = reshape;
                reshape_out->consumers.push_back(op);
                in->remove_consumer(op);
                in->consumers.push_back(reshape);

                std::vector<int> shape = in->shape;
                for (auto sa : selected_axes)
                {
                    // unsqueeze
                    shape.insert(shape.begin() + sa, 1);
                }

                reshape->params["shape"] = shape;
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
