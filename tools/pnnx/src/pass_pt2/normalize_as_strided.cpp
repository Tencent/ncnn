// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_as_strided.h"

#include <stdio.h>

#include <algorithm>
#include <limits.h>

namespace pnnx {

int normalize_pt2_layout_query(Operator* op, const Pt2TensorMeta& source)
{
    const bool stride = op->type == "aten::stride" || op->type == "aten::sym_stride";
    if (op->outputs.size() != 1 || op->inputs.size() != (stride ? 2u : 1u))
    {
        fprintf(stderr, "pnnx: pt2 layout: unsupported layout query signature\n");
        return -1;
    }
    int64_t value = source.storage_offset;
    if (stride)
    {
        const Operator* dim = op->inputs[1]->producer;
        if (dim->type != "prim::Constant" || !dim->has_param("value") || dim->params.at("value").type != 2)
        {
            fprintf(stderr, "pnnx: pt2 layout: stride query requires a static dimension\n");
            return -1;
        }
        int64_t axis = dim->params.at("value").i;
        if (axis < 0)
            axis += (int64_t)source.strides.size();
        if (axis < 0 || axis >= (int64_t)source.strides.size())
        {
            fprintf(stderr, "pnnx: pt2 layout: stride query dimension out of range\n");
            return -1;
        }
        value = source.strides[axis];
    }
    if (value < 0 || value > INT_MAX)
    {
        fprintf(stderr, "pnnx: pt2 layout: layout query requires a static value in pnnx parameter range\n");
        return -1;
    }
    for (Operand* r : op->inputs)
    {
        r->remove_consumer(op);
    }
    op->inputs.clear();
    op->inputnames.clear();
    op->type = "prim::Constant";
    op->params["value"] = (int)value;
    return 0;
}

static Operand* constant_before(Graph& graph, Operator* before, const char* suffix, const Parameter& value)
{
    Operator* op = graph.new_operator_before("prim::Constant", before->name + suffix, before);
    op->params["value"] = value;
    Operand* r = graph.new_operand(op->name);
    r->producer = op;
    op->outputs.push_back(r);
    return r;
}

int normalize_pt2_as_strided(Graph& graph, Operator* op, const Pt2TensorMeta& source)
{
    if (op->inputs.size() != 4 || op->outputs.size() != 1 || source.shape.size() != source.strides.size())
    {
        fprintf(stderr, "pnnx: pt2 layout: invalid as_strided signature or source layout\n");
        return -1;
    }
    std::vector<std::pair<int64_t, int> > order;
    for (size_t i = 0; i < source.shape.size(); i++)
    {
        order.push_back(std::make_pair(-source.strides[i], (int)i));
    }
    std::sort(order.begin(), order.end());
    std::vector<int> axes;
    for (const auto& axis : order)
    {
        axes.push_back(axis.second);
    }
    int64_t count = 1;
    for (auto i = axes.rbegin(); i != axes.rend(); ++i)
    {
        const int size = source.shape[*i];
        if (size <= 0 || (size > 1 && source.strides[*i] != count) || count > INT_MAX / size)
        {
            fprintf(stderr, "pnnx: pt2 layout: as_strided requires a static non-overlapping dense source\n");
            return -1;
        }
        count *= size;
    }
    Parameter args[3];
    for (size_t i = 1; i < 4; i++)
    {
        const Operator* constant = op->inputs[i]->producer;
        // read constant stride lists without deleting shared elements
        if (i < 3 && constant->type == "prim::ListConstruct")
        {
            args[i - 1].type = 5;
            for (const Operand* r : constant->inputs)
            {
                const Operator* item = r->producer;
                if (item->type != "prim::Constant" || !item->has_param("value") || item->params.at("value").type != 2)
                {
                    fprintf(stderr, "pnnx: pt2 layout: as_strided requires static size, stride and offset\n");
                    return -1;
                }
                args[i - 1].ai.push_back(item->params.at("value").i);
            }
            continue;
        }
        if (constant->type != "prim::Constant" || !constant->has_param("value"))
        {
            fprintf(stderr, "pnnx: pt2 layout: as_strided requires static size, stride and offset\n");
            return -1;
        }
        args[i - 1] = constant->params.at("value");
    }
    if (args[0].type != 5 || args[1].type != 5 || args[0].ai.size() != args[1].ai.size()
            || (args[2].type != 0 && args[2].type != 2))
    {
        fprintf(stderr, "pnnx: pt2 layout: invalid as_strided size, stride or offset\n");
        return -1;
    }
    if (args[2].type == 2 && (args[2].i < 0 || source.storage_offset < 0 || args[2].i < source.storage_offset))
    {
        fprintf(stderr, "pnnx: pt2 layout: as_strided cannot prove an explicit offset inside its source tensor\n");
        return -1;
    }
    const int64_t offset = args[2].type == 0 ? 0 : (int64_t)args[2].i - source.storage_offset;
    int64_t last = offset;
    if (offset < 0 || offset >= count)
    {
        fprintf(stderr, "pnnx: pt2 layout: as_strided accesses storage outside its source tensor\n");
        return -1;
    }
    for (size_t i = 0; i < args[0].ai.size(); i++)
    {
        const int size = args[0].ai[i];
        const int stride = args[1].ai[i];
        if (size <= 0 || stride < 0 || (size > 1 && stride > (count - 1 - last) / (size - 1)))
        {
            fprintf(stderr, "pnnx: pt2 layout: as_strided accesses storage outside its source tensor\n");
            return -1;
        }
        last += (int64_t)(size - 1) * stride;
    }

    // reconstruct the original dense storage order before applying as_strided
    for (size_t i = 1; i < 3; i++)
    {
        if (op->inputs[i]->producer->type == "prim::Constant")
            continue;
        op->inputs[i]->remove_consumer(op);
        op->inputs[i] = constant_before(graph, op, i == 1 ? "_size" : "_stride", args[i - 1]);
        op->inputs[i]->consumers.push_back(op);
    }
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
    {
        physical->shape.push_back(source.shape[axis]);
    }
    permute->outputs.push_back(physical);

    Operand* size = constant_before(graph, op, "_flat_size", Parameter(std::vector<int> {(int)count}));
    Operator* reshape = graph.new_operator_before("aten::reshape", op->name + "_flatten", op);
    reshape->inputs = {physical, size};
    physical->consumers.push_back(reshape);
    size->consumers.push_back(reshape);
    Operand* flat = graph.new_operand(reshape->name);
    flat->producer = reshape;
    flat->type = input->type;
    flat->shape = {(int)count};
    reshape->outputs.push_back(flat);
    // materialize storage so literal offsets start at zero
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
    return 0;
}

} // namespace pnnx
