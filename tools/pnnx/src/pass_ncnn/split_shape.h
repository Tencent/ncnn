// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_NCNN_SPLIT_SHAPE_H
#define PNNX_NCNN_SPLIT_SHAPE_H

#include "pass_ncnn.h"

namespace pnnx {
namespace ncnn {

static void split_with_dynamic_crops(Graph& graph, Operator* op, int axis, const std::vector<std::string>& boundaries)
{
    Operand* input = op->inputs[0];
    const std::vector<Operand*> outputs = op->outputs;
    op->outputs.resize(1);
    op->type = "Crop";
    op->params.clear();

    for (size_t i = 0; i < outputs.size(); i++)
    {
        Operator* crop = op;
        if (i != 0)
        {
            crop = graph.new_operator_before("Crop", op->name + "_" + std::to_string(i), op);
            crop->inputs.push_back(input);
            crop->outputs.push_back(outputs[i]);
            input->consumers.push_back(crop);
            outputs[i]->producer = crop;
        }
        crop->params["19"] = boundaries[i];
        crop->params["20"] = boundaries[i + 1];
        crop->params["21"] = std::to_string(axis);
    }
}

} // namespace ncnn
} // namespace pnnx

#endif // PNNX_NCNN_SPLIT_SHAPE_H
