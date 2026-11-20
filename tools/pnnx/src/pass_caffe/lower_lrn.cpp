// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_lrn.h"

#include "parameter.h"

#include <stdio.h>

#include <cmath>

namespace pnnx {

namespace caffe2pnnx {

static int lower_lrn_within_channel(Graph& graph, Operator* op)
{
    if (op->inputs.size() != 1 || op->outputs.size() != 1 || !op->attrs.empty() || op->inputs[0]->shape.size() != 4)
    {
        fprintf(stderr, "caffe layer %s (%s): WITHIN_CHANNEL requires one rank4 input/output and no weights; inputs=%zu outputs=%zu blobs=%zu rank=%zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size(), op->attrs.size(), op->inputs.empty() ? 0u : op->inputs[0]->shape.size());
        return -1;
    }
    const std::vector<int> shape = op->inputs[0]->shape;
    const int size = get_int(op, "local_size", 5);
    const float alpha = get_float(op, "alpha", 1.f);
    const float beta = get_float(op, "beta", .75f);
    if (get_int(op, "norm_region", 0) != 1 || size <= 0 || size % 2 != 1 || (int64_t)size * size > INT_MAX || !std::isfinite(alpha) || !std::isfinite(beta))
    {
        fprintf(stderr, "caffe layer %s (%s): invalid WITHIN_CHANNEL parameters norm_region=%d local_size=%d alpha=%g beta=%g; require region1, positive odd size with area<=INT_MAX and finite alpha/beta\n", op->name.c_str(), op->type.c_str(), get_int(op, "norm_region", 0), size, alpha, beta);
        return -1;
    }
    int64_t count = 1;
    for (int dim : shape)
    {
        if (dim <= 0 || count > INT_MAX / dim)
        {
            fprintf(stderr, "caffe layer %s (%s): invalid input dimension=%d or element count overflow (preceding count=%lld)\n", op->name.c_str(), op->type.c_str(), dim, (long long)count);
            return -1;
        }
        count *= dim;
    }
    if ((int64_t)shape[2] + size - 1 > INT_MAX || (int64_t)shape[3] + size - 1 > INT_MAX)
    {
        fprintf(stderr, "caffe layer %s (%s): padded spatial size exceeds INT_MAX; height=%d width=%d local_size=%d\n", op->name.c_str(), op->type.c_str(), shape[2], shape[3], size);
        return -1;
    }

    Operand* input = op->inputs[0];
    Operator* square = graph.new_operator_before("aten::mul", op->name + "_square", op);
    square->inputs.push_back(input);
    input->consumers.push_back(square);
    square->inputs.push_back(input);
    input->consumers.push_back(square);
    Operand* squared = graph.new_operand(op->name + "_square");
    squared->producer = square;
    squared->shape = shape;
    squared->type = 1;
    squared->params["__batch_index"] = 0;
    square->outputs.push_back(squared);

    // caffe averages squared spatial windows including all declared padding
    // with an odd kernel and stride 1 there is no partial ceil window
    Operator* pool = graph.new_operator_before("F.avg_pool2d", op->name + "_average", op);
    pool->params["kernel_size"] = std::vector<int> {size, size};
    pool->params["stride"] = std::vector<int> {1, 1};
    pool->params["padding"] = std::vector<int> {size / 2, size / 2};
    pool->params["ceil_mode"] = false;
    pool->params["count_include_pad"] = true;
    pool->params["divisor_override"] = Parameter();
    pool->inputs.push_back(squared);
    squared->consumers.push_back(pool);
    Operand* average = graph.new_operand(op->name + "_average");
    average->producer = pool;
    average->shape = shape;
    average->type = 1;
    average->params["__batch_index"] = 0;
    pool->outputs.push_back(average);

    // bvlc caffe uses a fixed shift of 1 for WITHIN_CHANNEL; k is only used
    // by ACROSS_CHANNELS. Preserve this distinction for nondefault k too
    Operand* factor = average;
    const char* types[] = {"aten::mul", "aten::add", "aten::pow"};
    const char* suffixes[] = {"_scale", "_shift", "_power"};
    const float values[] = {alpha, 1.f, -beta};
    for (int i = 0; i < 3; i++)
    {
        Operator* constant = graph.new_operator_before("prim::Constant", op->name + suffixes[i] + "_value", op);
        constant->params["value"] = values[i];
        Operand* value = graph.new_operand(constant->name);
        value->producer = constant;
        value->type = 1;
        value->params["__batch_index"] = 233;
        constant->outputs.push_back(value);
        Operator* arithmetic = graph.new_operator_before(types[i], op->name + suffixes[i], op);
        arithmetic->inputs = {factor, value};
        factor->consumers.push_back(arithmetic);
        value->consumers.push_back(arithmetic);
        factor = graph.new_operand(arithmetic->name);
        factor->producer = arithmetic;
        factor->shape = shape;
        factor->type = 1;
        factor->params["__batch_index"] = 0;
        arithmetic->outputs.push_back(factor);
    }
    op->type = "aten::mul";
    op->params.clear();
    op->inputs.push_back(factor);
    factor->consumers.push_back(op);
    op->outputs[0]->shape = shape;
    op->outputs[0]->params["__batch_index"] = 0;
    return 0;
}

int lower_lrn(Graph& graph, Operator* op)
{
    if (op->inputs.size() != 1 || op->outputs.size() != 1 || op->inputs[0]->shape.empty())
    {
        fprintf(stderr, "caffe layer %s (%s): expected one input/output with known nonempty shape; inputs=%zu outputs=%zu rank=%zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size(), op->inputs.empty() ? 0u : op->inputs[0]->shape.size());
        return -1;
    }
    const std::vector<int> shape = op->inputs[0]->shape;
    op->outputs[0]->shape = shape;
    if (get_int(op, "norm_region", 0) == 1)
        return lower_lrn_within_channel(graph, op);
    const int size = get_int(op, "local_size", 5);
    const float alpha = get_float(op, "alpha", 1.f);
    const float beta = get_float(op, "beta", .75f);
    const float k = get_float(op, "k", 1.f);
    if (get_int(op, "norm_region", 0) != 0 || size <= 0 || size % 2 != 1 || shape.size() < 3)
    {
        fprintf(stderr, "caffe layer %s (%s): ACROSS_CHANNELS requires region0, positive odd local_size and rank>=3; norm_region=%d local_size=%d rank=%zu\n", op->name.c_str(), op->type.c_str(), get_int(op, "norm_region", 0), size, shape.size());
        return -1;
    }
    op->params.clear();
    op->params["size"] = size;
    op->params["alpha"] = alpha;
    op->params["beta"] = beta;
    op->params["k"] = k;

    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
