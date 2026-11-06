// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_as_strided.h"

#include <algorithm>
#include <limits.h>

namespace pnnx {

static Operand* constant_before(Graph& graph, Operator* before, const char* suffix, const Parameter& value)
{
    Operator* op = graph.new_operator_before("prim::Constant", before->name + suffix, before);
    op->params["value"] = value;
    Operand* r = graph.new_operand(op->name);
    r->producer = op;
    op->outputs.push_back(r);
    return r;
}

bool normalize_pt2_as_strided(Graph& graph, Operator* op, const Pt2TensorMeta& source, std::string& error)
{
    if (op->inputs.size() != 4 || op->outputs.size() != 1 || source.shape.size() != source.strides.size())
    {
        error = "invalid as_strided signature or source layout";
        return false;
    }
    std::vector<int> axes;
    for (size_t i = 0; i < source.shape.size(); i++)
        axes.push_back((int)i);
    std::stable_sort(axes.begin(), axes.end(), [&](int a, int b) { return source.strides[a] > source.strides[b]; });
    int64_t count = 1;
    for (auto i = axes.rbegin(); i != axes.rend(); ++i)
    {
        const int size = source.shape[*i];
        if (size <= 0 || (size > 1 && source.strides[*i] != count) || count > INT_MAX / size)
        {
            error = "as_strided requires a static non-overlapping dense source";
            return false;
        }
        count *= size;
    }
    Parameter args[3];
    for (size_t i = 1; i < 4; i++)
    {
        const Operator* constant = op->inputs[i]->producer;
        if (constant->type != "prim::Constant" || !constant->has_param("value"))
        {
            error = "as_strided requires static size, stride and offset";
            return false;
        }
        args[i - 1] = constant->params.at("value");
    }
    if (args[0].type != 5 || args[1].type != 5 || args[0].ai.size() != args[1].ai.size()
        || (args[2].type != 0 && args[2].type != 2))
    {
        error = "invalid as_strided size, stride or offset";
        return false;
    }
    if (args[2].type == 2 && (args[2].i < 0 || source.storage_offset < 0 || args[2].i < source.storage_offset))
    {
        error = "as_strided cannot prove an explicit offset inside its source tensor";
        return false;
    }
    const int64_t offset = args[2].type == 0 ? 0 : (int64_t)args[2].i - source.storage_offset;
    int64_t last = offset;
    if (offset < 0 || offset >= count)
    {
        error = "as_strided accesses storage outside its source tensor";
        return false;
    }
    for (size_t i = 0; i < args[0].ai.size(); i++)
    {
        const int size = args[0].ai[i];
        const int stride = args[1].ai[i];
        if (size <= 0 || stride < 0 || (size > 1 && stride > (count - 1 - last) / (size - 1)))
        {
            error = "as_strided accesses storage outside its source tensor";
            return false;
        }
        last += (int64_t)(size - 1) * stride;
    }

    // Public optimizations may change physical layouts while preserving values.
    // Reconstruct the original dense storage order from logical tensor axes,
    // then flatten it before applying the explicitly bounded strided view.
    Operand* input = op->inputs[0];
    Operand* dims = constant_before(graph, op, "_axes", Parameter(axes));
    Operator* permute = graph.new_operator_before("aten::permute", op->name + "_permute", op);
    permute->inputs = {input, dims};
    input->consumers.push_back(permute);
    dims->consumers.push_back(permute);
    Operand* physical = graph.new_operand(permute->name);
    physical->producer = permute;
    physical->type = input->type;
    for (int axis : axes)
        physical->shape.push_back(source.shape[axis]);
    permute->outputs.push_back(physical);

    Operand* size = constant_before(graph, op, "_flat_size", Parameter(std::vector<int>{(int)count}));
    Operator* reshape = graph.new_operator_before("aten::reshape", op->name + "_flatten", op);
    reshape->inputs = {physical, size};
    physical->consumers.push_back(reshape);
    size->consumers.push_back(reshape);
    Operand* flat = graph.new_operand(reshape->name);
    flat->producer = reshape;
    flat->type = input->type;
    flat->shape = {(int)count};
    reshape->outputs.push_back(flat);
    // A reshape may still share a nonzero storage offset. Materialize this
    // one-dimensional sequence so literal offsets address its own storage.
    Operand* format = constant_before(graph, op, "_format", Parameter(0));
    Operator* clone = graph.new_operator_before("aten::clone", op->name + "_storage", op);
    clone->inputs = {flat, format};
    flat->consumers.push_back(clone);
    format->consumers.push_back(clone);
    Operand* storage = graph.new_operand(clone->name);
    storage->producer = clone;
    storage->type = input->type;
    storage->shape = flat->shape;
    clone->outputs.push_back(storage);
    input->remove_consumer(op);
    op->inputs[0] = storage;
    storage->consumers.push_back(op);
    op->inputs[3]->remove_consumer(op);
    op->inputs[3] = constant_before(graph, op, "_offset", Parameter((int)offset));
    op->inputs[3]->consumers.push_back(op);
    return true;
}

} // namespace pnnx
