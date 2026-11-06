// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_dtype.h"

#include <algorithm>
#include <float.h>
#include <stdio.h>

namespace pnnx {

static int scalar_type(int type)
{
    static const int types[] = {-1, 6, 7, 5, 3, 4, 2, 1, 0, 11, 9, 10, 8, 15};
    return type >= 1 && type <= 13 ? types[type] : -1;
}

static int operand_type(int dtype)
{
    for (int type = 1; type <= 13; type++)
        if (scalar_type(type) == dtype)
            return type;
    return 0;
}

static Operand* new_constant(Graph& graph, const Operator* before, const Parameter& value, int& id)
{
    const std::string name = "pt2_dtype_" + std::to_string(id++);
    Operator* op = graph.new_operator_before("prim::Constant", name, before);
    op->params["value"] = value;
    Operand* r = graph.new_operand(name);
    r->producer = op;
    op->outputs.push_back(r);
    return r;
}

static Operand* cast_before(Graph& graph, const Operator* before, Operand* source, int type, int& id)
{
    Operand* dtype = new_constant(graph, before, Parameter(scalar_type(type)), id);
    Operand* non_blocking = new_constant(graph, before, Parameter(false), id);
    Operand* copy = new_constant(graph, before, Parameter(false), id);
    Operand* format = new_constant(graph, before, Parameter(), id);
    const std::string name = "pt2_dtype_" + std::to_string(id++);
    Operator* op = graph.new_operator_before("aten::to", name, before);
    op->inputs = {source, dtype, non_blocking, copy, format};
    for (Operand* input : op->inputs)
        input->consumers.push_back(op);
    Operand* r = graph.new_operand(name);
    r->producer = op;
    r->type = type;
    r->shape = source->shape;
    op->outputs.push_back(r);
    return r;
}

int normalize_pt2_dtype(Graph& graph)
{
    int id = 0;
    // Snapshot: casts inserted here already use the existing aten::to contract.
    const std::vector<Operator*> ops = graph.ops;
    for (Operator* op : ops)
    {
        if (op->type == "aten::new_full" && op->inputs.size() == 7)
        {
            Operand* source = op->inputs[0];
            Parameter& dtype = op->inputs[3]->producer->params.at("value");
            if (dtype.type == 0)
                dtype = scalar_type(source->type);
            source->remove_consumer(op);
            op->inputs.erase(op->inputs.begin());
            op->type = "aten::full";
        }
        if (op->type == "aten::rms_norm" && op->inputs.size() == 4)
        {
            Parameter& eps = op->inputs[3]->producer->params.at("value");
            if (eps.type == 0)
            {
                const int type = op->inputs[0]->type;
                if (type != 1 && type != 2 && type != 3 && type != 13)
                {
                    fprintf(stderr, "pt2 dtype: unsupported RMSNorm input type\n");
                    return -1;
                }
                // PyTorch promotes half and bfloat16 to float32 opmath.
                eps = type == 2 ? (float)DBL_EPSILON : FLT_EPSILON;
            }
        }
        const bool softmax = op->type == "aten::softmax" || op->type == "aten::log_softmax";
        const bool reduce = op->type == "aten::sum" || op->type == "aten::mean" || op->type == "aten::prod";
        if (!softmax && !reduce)
            continue;
        if ((softmax && op->inputs.size() != 3) || (reduce && op->inputs.size() != 2 && op->inputs.size() != 4))
        {
            fprintf(stderr, "pt2 dtype: unsupported %s signature\n", op->type.c_str());
            return -1;
        }
        Parameter& dtype = op->inputs.back()->producer->params.at("value");
        if (dtype.type == 0)
            continue;
        const int type = dtype.type == 2 ? operand_type(dtype.i) : 0;
        if (type == 0)
        {
            fprintf(stderr, "pt2 dtype: unsupported dtype for %s\n", op->type.c_str());
            return -1;
        }
        // These ATen operators define dtype as converting the input before computation.
        Operand* source = op->inputs[0];
        if (source->type != type)
        {
            Operand* converted = cast_before(graph, op, source, type, id);
            source->remove_consumer(op);
            op->inputs[0] = converted;
            converted->consumers.push_back(op);
        }
        dtype = Parameter();
        if ((op->type == "aten::sum" || op->type == "aten::prod") && type >= 4 && type <= 9 && type != 5)
        {
            // Without dtype, integer reductions promote to int64. Narrowing the final
            // integer sum/product preserves the requested modular integer arithmetic.
            Operand* result = op->outputs[0];
            const std::string name = "pt2_dtype_" + std::to_string(id++);
            Operand* promoted = graph.new_operand(name);
            promoted->type = 5;
            promoted->shape = result->shape;
            promoted->producer = op;
            op->outputs[0] = promoted;
            // Insert before the next original operator so the cast follows the reduction.
            const auto current = std::find(graph.ops.begin(), graph.ops.end(), op);
            Operator* next = current + 1 == graph.ops.end() ? 0 : *(current + 1);
            Operand* converted = cast_before(graph, next, promoted, type, id);
            Operator* cast = converted->producer;
            cast->outputs[0] = result;
            result->producer = cast;
            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), converted));
            delete converted;
        }
    }
    return 0;
}

} // namespace pnnx
