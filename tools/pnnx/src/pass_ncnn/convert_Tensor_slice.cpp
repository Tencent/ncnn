// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_Tensor_slice.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

void convert_Tensor_slice(Graph& graph)
{
    // pre-process: expand single-axis slice with step>1 (strided sampling, e.g.
    // deepseek MLA x[..., ::2]) into reshape(-> K*step) + Crop(take the start-th)
    // + reshape(-> K). Only supports 4D input with the step axis being the last
    // dim (dim=rank-1) and the axis length divisible by step.
    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (op->type != "Tensor.slice")
                continue;
            if (op->inputs.size() != 1 || op->outputs.size() != 1)
                continue;
            if (!op->has_param("dim") || !op->has_param("start") || !op->has_param("end") || !op->has_param("step"))
                continue;

            const int dim = op->params.at("dim").i;
            const int start = op->params.at("start").i;
            const int step = op->params.at("step").i;
            if (step == 1)
                continue;

            Operand* in = op->inputs[0];
            Operand* out = op->outputs[0];
            const std::vector<int>& in_shape = in->shape;
            const int rank = (int)in_shape.size();

            // only support 4D with the step axis being the last dim
            if (rank != 4 || dim != rank - 1)
            {
                fprintf(stderr, "slice with step %d not supported (rank=%d dim=%d)\n", step, rank, dim);
                continue;
            }
            const int L = in_shape[dim];
            if (L <= 0 || L % step != 0 || start < 0 || start >= step)
                continue;
            // this lowering (reshape into step-wide groups + Crop of one column)
            // samples the full tail [..., start::step] with exactly L / step
            // elements. A finite end such as [..., 1:5:2] must not use it: the
            // K = L / step reshape would sample past the end and change both the
            // output shape and values. full-tail slices arrive with end = INT_MAX.
            {
                const int end = op->params.at("end").i;
                if (end != INT_MAX && end < L)
                    continue;
            }
            const int K = L / step;

            // copy the input batch axis (avoid Tensor_reshape reading unset garbage that triggers batch mode)
            int fl_batch_axis = 233;
            if (in->params.find("__ncnn_batch_axis") != in->params.end())
                fl_batch_axis = in->params.at("__ncnn_batch_axis").i;

            matched = true;

            // reshape0: (b,a,c,L) -> (b,a,c*K,step) (keep 4D and batch, step on the w dim)
            Operator* reshape0 = graph.new_operator_before("Tensor.reshape", op->name + "_ncnnreshape0", op);
            Operand* reshape0_out = graph.new_operand(op->name + "_ncnnreshape0_out");
            reshape0->inputs.push_back(in);
            reshape0->outputs.push_back(reshape0_out);
            reshape0_out->producer = reshape0;
            reshape0_out->consumers.push_back(op);
            std::vector<int> reshape0_shape = {in_shape[0], in_shape[1], in_shape[2] * K, step};
            reshape0->params["shape"] = reshape0_shape;
            reshape0_out->shape = reshape0_shape;
            reshape0_out->type = in->type;
            reshape0_out->params["__ncnn_batch_axis"] = fl_batch_axis;

            // original op -> Crop: take [start, start+1) on the w dim
            // ncnn Crop axis mapping: dims=3 -> 0=c 1=h 2=w; dims=4 -> 0=c 1=d 2=h 3=w.
            // reshape0 output (b, a, c*K, step): when the batch dim is stripped
            // (batch_axis!=233) the physical tensor is 3D (c,h,w) and w is axis 2;
            // otherwise it is 4D and w is axis 3.
            const int crop_axis = (fl_batch_axis == 233) ? 3 : 2;
            op->type = "Crop";
            op->inputs[0] = reshape0_out;
            // consumer bookkeeping: reshape0 consumes in now, op consumes reshape0_out
            {
                auto itc = std::find(in->consumers.begin(), in->consumers.end(), op);
                if (itc != in->consumers.end())
                    *itc = reshape0;
            }
            op->params["9"] = std::vector<int> {start};
            op->params["10"] = std::vector<int> {start + 1};
            op->params["11"] = std::vector<int> {crop_axis};
            op->params.erase("dim");
            op->params.erase("start");
            op->params.erase("end");
            op->params.erase("step");

            // reshape1: (b*a, c, K, 1) -> (b, a, c, K)
            Operand* crop_out = op->outputs[0];
            Operator* reshape1 = graph.new_operator_after("Tensor.reshape", op->name + "_ncnnreshape1", op);
            Operand* reshape1_in = graph.new_operand(op->name + "_ncnnreshape1_in");
            reshape1->inputs.push_back(reshape1_in);
            reshape1->outputs.push_back(crop_out);
            op->outputs[0] = reshape1_in;
            crop_out->producer = reshape1;
            reshape1_in->producer = op;
            reshape1_in->consumers.push_back(reshape1);

            std::vector<int> reshape1_shape = {in_shape[0], in_shape[1], in_shape[2], K};
            reshape1->params["shape"] = reshape1_shape;
            reshape1_in->shape = {in_shape[0], in_shape[1], in_shape[2] * K, 1};
            reshape1_in->type = in->type;
            reshape1_in->params["__ncnn_batch_axis"] = fl_batch_axis;
            crop_out->shape = reshape1_shape;

            break;
        }

        if (!matched)
            break;
    }

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
                    if (i >= (int)selects.size())
                    {
                        fprintf(stderr, "slice with step 0 but no select index is not supported\n");
                        unsupported = true;
                        break;
                    }
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
                    select_count += 1;

                if (ncnn_batch_axis != 233 && axes[i] > ncnn_batch_axis)
                    axes[i] -= 1;

                if (ends[i] == INT_MAX)
                    ends[i] = -233;
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

                Operator* reshape = graph.new_operator_after("Tensor.reshape", op->name + "_ncnnreshape", op);

                Operand* reshape_in = graph.new_operand(op->name + "_ncnnreshape_in");

                reshape_in->params["__batch_index"] = batch_index;
                reshape_in->params["__ncnn_batch_axis"] = ncnn_batch_axis;

                reshape->inputs.push_back(reshape_in);
                reshape->outputs.push_back(out);

                op->outputs[0] = reshape_in;

                out->producer = reshape;
                reshape_in->producer = op;
                reshape_in->consumers.push_back(reshape);

                if (!out->shape.empty())
                    reshape->params["shape"] = out->shape;
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
