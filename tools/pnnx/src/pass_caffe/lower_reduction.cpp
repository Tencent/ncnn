// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_reduction.h"

#include "parameter.h"

#include <cmath>
#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

static Operand* reduction_input(Graph& graph, Operator* before, const char* type, const std::string& suffix, Operand* input, const std::vector<int>& shape, int batch_index)
{
    Operator* op = graph.new_operator_before(type, before->name + suffix, before);
    Operand* output = graph.new_operand(before->name + suffix);
    op->inputs.push_back(input);
    input->consumers.push_back(op);
    op->outputs.push_back(output);
    output->producer = op;
    output->shape = shape;
    output->type = 1;
    output->params["__batch_index"] = batch_index;
    return output;
}

int lower_reduction(Graph& graph, Operator* op)
{
    if (op->inputs.size() != 1 || op->outputs.size() != 1 || !op->attrs.empty())
    {
        fprintf(stderr, "caffe layer %s (%s): expected one input/output and no blobs; got inputs=%zu outputs=%zu blobs=%zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size(), op->attrs.size());
        return -1;
    }
    Operand* input = op->inputs[0];
    const std::vector<int> shape = input->shape;
    const int rank = shape.size();
    int axis = get_int(op, "axis", 0);
    if (axis < 0)
        axis += rank;
    const int operation = get_int(op, "operation", 1);
    const float coeff = get_float(op, "coeff", 1.f);
    if (axis < 0 || axis >= rank || operation < 1 || operation > 4 || !std::isfinite(coeff))
    {
        fprintf(stderr, "caffe layer %s (%s): invalid normalized axis=%d rank=%d operation=%d coeff=%g; require valid axis, operation1..4 and finite coefficient\n", op->name.c_str(), op->type.c_str(), axis, rank, operation, coeff);
        return -1;
    }
    int64_t count = 1;
    for (int dim : shape)
    {
        if (dim <= 0 || count > INT_MAX / dim)
        {
            fprintf(stderr, "caffe layer %s (%s): input dimension=%d is nonpositive or element count overflows (preceding count=%lld)\n", op->name.c_str(), op->type.c_str(), dim, (long long)count);
            return -1;
        }
        count *= dim;
    }
    int batch_index = rank == 1 ? 233 : 0;
    if (input->params.count("__batch_index"))
        batch_index = input->params.at("__batch_index").i;
    if (rank == 1)
        batch_index = 233;
    if (batch_index != 233 && (batch_index < 0 || batch_index >= rank))
    {
        fprintf(stderr, "caffe layer %s (%s): batch index=%d is outside rank=%d\n", op->name.c_str(), op->type.c_str(), batch_index, rank);
        return -1;
    }
    if (rank - (batch_index != 233 ? 1 : 0) > 4)
    {
        fprintf(stderr, "caffe layer %s (%s): native rank exceeds 4; source rank=%d batch index=%d\n", op->name.c_str(), op->type.c_str(), rank, batch_index);
        return -1;
    }
    if (batch_index != 233 && shape[batch_index] != 1)
    {
        fprintf(stderr, "Caffe Reduction %s: ncnn export requires singleton batch or an unbatched input\n", op->name.c_str());
        return -1;
    }

    std::vector<int> reduced_shape(shape.begin(), shape.begin() + axis);
    const bool reduce_batch = batch_index != 233 && axis <= batch_index;
    const int output_batch_index = !reduce_batch && batch_index < axis && reduced_shape.size() > 1 ? batch_index : 233;
    std::vector<int> dims;
    for (int i = axis; i < rank; i++)
    {
        if (reduce_batch && i == batch_index)
            continue;
        dims.push_back(reduce_batch && i > batch_index ? i - 1 : i);
    }
    if (dims.empty())
    {
        fprintf(stderr, "caffe layer %s (%s): no reduction dimensions remain for axis=%d rank=%d batch index=%d\n", op->name.c_str(), op->type.c_str(), axis, rank, batch_index);
        return -1;
    }

    input->remove_consumer(op);
    std::vector<int> current_shape = shape;
    if (reduce_batch)
    {
        // removing a singleton batch is an identity in ncnn. This also avoids
        // asking the generic reduction exporter to reduce its omitted batch
        current_shape.erase(current_shape.begin() + batch_index);
        input = reduction_input(graph, op, "torch.squeeze", "_unbatch", input, current_shape, 233);
        input->producer->params["dim"] = batch_index;
        batch_index = 233;
    }
    if (operation == 2 || operation == 3)
    {
        Operand* value = input;
        input = reduction_input(graph, op, operation == 2 ? "aten::abs" : "aten::mul", "_values", value, current_shape, batch_index);
        if (operation == 3)
        {
            input->producer->inputs.push_back(value);
            value->consumers.push_back(input->producer);
        }
    }
    if (coeff != 1.f)
    {
        input = reduction_input(graph, op, operation == 4 ? "torch.mean" : "torch.sum", "_reduce", input, reduced_shape, output_batch_index);
        input->producer->params["dim"] = dims;
        input->producer->params["keepdim"] = false;
        Operator* constant = graph.new_operator_before("prim::Constant", op->name + "_coeff", op);
        constant->params["value"] = coeff;
        Operand* scale = graph.new_operand(constant->name);
        scale->producer = constant;
        constant->outputs.push_back(scale);
        scale->consumers.push_back(op);
        op->inputs.push_back(scale);
        op->type = "aten::mul";
        op->params.clear();
    }
    else
    {
        op->type = operation == 4 ? "torch.mean" : "torch.sum";
        op->params.clear();
        op->params["dim"] = dims;
        op->params["keepdim"] = false;
    }
    op->inputs[0] = input;
    input->consumers.push_back(op);
    op->outputs[0]->shape = reduced_shape;
    op->outputs[0]->params["__batch_index"] = output_batch_index;
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
