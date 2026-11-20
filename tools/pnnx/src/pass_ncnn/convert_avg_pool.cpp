// Copyright 2026 ktz03
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_avg_pool.h"

namespace pnnx {

namespace ncnn {

void convert_avg_pool(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            int spatial_rank = 0;
            if (op->type == "F.avg_pool1d" || op->type == "nn.AvgPool1d")
                spatial_rank = 1;
            if (op->type == "F.avg_pool2d" || op->type == "nn.AvgPool2d")
                spatial_rank = 2;
            if (op->type == "F.avg_pool3d" || op->type == "nn.AvgPool3d")
                spatial_rank = 3;

            if (!spatial_rank || !op->params.at("ceil_mode").b || !op->params.at("count_include_pad").b)
                continue;
            if (op->params.find("divisor_override") != op->params.end() && op->params.at("divisor_override").type != 0)
                continue;

            const std::vector<int>& kernel = op->params.at("kernel_size").ai;
            const std::vector<int>& padding = op->params.at("padding").ai;
            const std::vector<int>& stride = op->params.at("stride").type == 0 ? kernel : op->params.at("stride").ai;
            Operand* in = op->inputs[0];

            // Retain the existing lowering if it produces an extra all-padding
            // output window. Excluding the tail would otherwise divide by zero.
            bool safe = true;
            for (int d = 0; d < spatial_rank; d++)
            {
                int length = (int)in->shape.size() >= spatial_rank ? in->shape[in->shape.size() - spatial_rank + d] : -1;
                if (length > 0)
                {
                    const int extent = length + padding[d] * 2 - kernel[d];
                    if (extent < 0 || ((extent + stride[d] - 1) / stride[d]) * stride[d] >= length + padding[d])
                        safe = false;
                }
                else if (stride[d] + padding[d] > kernel[d])
                {
                    safe = false;
                }
            }
            if (!safe)
                continue;

            matched = true;
            op->params["count_include_pad"] = false;

            bool has_padding = false;
            for (int p : padding)
                has_padding |= p != 0;
            if (!has_padding)
                break;

            // Make only the explicit padding part of the input. Pooling then
            // counts these zeros, while excluding any additional ceil tail.
            Operator* pad = graph.new_operator_before("Padding", op->name + "_pad", op);
            Operand* padded = graph.new_operand(op->name + "_padded");
            padded->type = in->type;
            padded->shape = in->shape;
            padded->params = in->params;
            for (int d = 0; d < spatial_rank; d++)
            {
                if (padded->shape.size() >= (size_t)spatial_rank && padded->shape[padded->shape.size() - spatial_rank + d] > 0)
                    padded->shape[padded->shape.size() - spatial_rank + d] += padding[d] * 2;
            }
            pad->params["0"] = spatial_rank >= 2 ? padding[spatial_rank - 2] : 0;
            pad->params["1"] = pad->params["0"];
            pad->params["2"] = padding[spatial_rank - 1];
            pad->params["3"] = pad->params["2"];
            pad->params["4"] = 0;
            pad->params["5"] = 0.f;
            pad->params["6"] = 0;
            if (spatial_rank == 3)
            {
                pad->params["7"] = padding[0];
                pad->params["8"] = padding[0];
            }
            pad->inputs.push_back(in);
            pad->outputs.push_back(padded);
            in->remove_consumer(op);
            in->consumers.push_back(pad);
            padded->producer = pad;
            padded->consumers.push_back(op);
            op->inputs[0] = padded;
            op->params["padding"] = std::vector<int>(spatial_rank, 0);
            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
