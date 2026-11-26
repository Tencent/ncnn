// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "crop_pooling_output.h"

namespace pnnx {

namespace ncnn {

void crop_pooling_output(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];
        if (op->type != "Pooling" || !op->has_param("0") || op->params.at("0").i != 0 || !op->has_param("5") || op->params.at("5").i != 0)
            continue;

        const int kernel_w = op->params.at("1").i;
        const int kernel_h = op->params.at("11").i;
        const int stride_w = op->params.at("2").i;
        const int stride_h = op->params.at("12").i;
        const int pad_left = op->params.at("3").i;
        const int pad_top = op->params.at("13").i;
        const int pad_right = op->has_param("14") ? op->params.at("14").i : pad_left;
        const int pad_bottom = op->has_param("15") ? op->params.at("15").i : pad_top;
        if (stride_w + pad_right <= kernel_w && stride_h + pad_bottom <= kernel_h)
            continue;

        Operand* input = op->inputs[0];
        Operand* output = op->outputs[0];
        Operator* crop = graph.new_operator_after("Crop", op->name + "_crop", op);
        Operand* pooled = graph.new_operand(op->name + "_uncropped");
        pooled->producer = op;
        pooled->consumers.push_back(crop);
        pooled->type = output->type;
        pooled->params = output->params;
        op->outputs[0] = pooled;
        input->consumers.push_back(crop);
        output->producer = crop;
        crop->inputs = {pooled, input};
        crop->outputs = {output};

        // Ceil-mode windows must start before the right/bottom input boundary.
        // Crop clamps these runtime limits to the actual pooled dimensions.
        crop->params["19"] = "0,0";
        crop->params["20"] = "//(+(1h," + std::to_string(pad_top + stride_h - 1) + ")," + std::to_string(stride_h) + "),//(+(1w," + std::to_string(pad_left + stride_w - 1) + ")," + std::to_string(stride_w) + ")";
        crop->params["21"] = "-2,-1";
    }
}

} // namespace ncnn

} // namespace pnnx
