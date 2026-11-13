// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_dtype.h"
#include "pt2_tensor.h"

#include <algorithm>
#include <float.h>
#include <stdio.h>
#include <string.h>

namespace pnnx {

int normalize_pt2_scalar_type(int type)
{
    // c10 ScalarType: half=5, float=6, double=7, complex32=8, complex64=9, complex128=10, bfloat16=15
    if (type == 5 || type == 7 || type == 15)
        return 6;
    if (type == 8 || type == 10)
        return 9;
    return type;
}

void normalize_pt2_tensor_dtype(Graph& graph)
{
    // validate original storage and aliases before FP32 normalization
    for (Operator* op : graph.ops)
    {
        for (auto& item : op->attrs)
        {
            Attribute& attr = item.second;
            if (attr.type == 11)
            {
                // store FP32 real and imaginary components
                std::vector<float> data(attr.data.size() / sizeof(double));
                const double* p = (const double*)attr.data.data();
                for (size_t i = 0; i < data.size(); i++)
                {
                    data[i] = (float)p[i];
                }
                attr.type = 10;
                attr.data.resize(data.size() * sizeof(float));
                if (!data.empty())
                    memcpy(attr.data.data(), data.data(), attr.data.size());
                continue;
            }
            if (attr.type != 2 && attr.type != 3 && attr.type != 13)
                continue;
            const std::vector<float> data = attr.get_float32_data();
            attr.type = 1;
            attr.set_float32_data(data);
        }
    }
    for (Operand* r : graph.operands)
    {
        if (r->type == 2 || r->type == 3 || r->type == 13)
            r->type = 1;
        if (r->type == 11)
            r->type = 10;
    }
}

static int operand_type(int dtype)
{
    for (int type = 1; type <= 13; type++)
    {
        if (pt2_scalar_type(type) == dtype)
            return type;
    }
    return -1;
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
    Operand* dtype = new_constant(graph, before, Parameter(pt2_scalar_type(type)), id);
    Operand* non_blocking = new_constant(graph, before, Parameter(false), id);
    Operand* copy = new_constant(graph, before, Parameter(false), id);
    Operand* format = new_constant(graph, before, Parameter(), id);
    const std::string name = "pt2_dtype_" + std::to_string(id++);
    Operator* op = graph.new_operator_before("aten::to", name, before);
    op->inputs = {source, dtype, non_blocking, copy, format};
    for (Operand* input : op->inputs)
    {
        input->consumers.push_back(op);
    }
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
    // snapshot operators before inserting casts
    const std::vector<Operator*> ops = graph.ops;
    for (Operator* op : ops)
    {
        if (op->type == "aten::new_full" && op->inputs.size() == 7)
        {
            Operand* source = op->inputs[0];
            Parameter& dtype = op->inputs[3]->producer->params.at("value");
            if (dtype.type == 0)
                dtype = pt2_scalar_type(source->type);
            source->remove_consumer(op);
            op->inputs.erase(op->inputs.begin());
            op->type = "aten::full";
        }
        if (op->type == "aten::rms_norm" && op->inputs.size() == 4)
        {
            Operator* constant = op->inputs[3]->producer;
            if (constant->type != "prim::Constant" || !constant->has_param("value"))
            {
                fprintf(stderr, "pnnx: pt2 dtype: RMSNorm epsilon must be constant\n");
                return -1;
            }
            Parameter& eps = constant->params.at("value");
            if (eps.type == 0)
            {
                if (op->inputs[0]->type != 1)
                {
                    fprintf(stderr, "pnnx: pt2 dtype: unsupported RMSNorm input type\n");
                    return -1;
                }
                eps = FLT_EPSILON;
            }
        }
        const bool softmax = op->type == "aten::softmax" || op->type == "aten::log_softmax";
        const bool reduce = op->type == "aten::sum" || op->type == "aten::mean" || op->type == "aten::prod";
        const bool scan = op->type == "aten::cumsum" || op->type == "aten::cumprod";
        const bool norm = op->type == "aten::linalg_vector_norm";
        if (!softmax && !reduce && !scan && !norm)
            continue;
        if (((softmax || scan) && op->inputs.size() != 3)
                || (reduce && op->inputs.size() != 2 && op->inputs.size() != 4)
                || (norm && op->inputs.size() != 5))
        {
            fprintf(stderr, "pnnx: pt2 dtype: unsupported %s signature\n", op->type.c_str());
            return -1;
        }
        Parameter& dtype = op->inputs.back()->producer->params.at("value");
        if (dtype.type == 0)
            continue;
        const int type = dtype.type == 2 ? operand_type(dtype.i) : -1;
        if (type == -1)
        {
            fprintf(stderr, "pnnx: pt2 dtype: unsupported dtype for %s\n", op->type.c_str());
            return -1;
        }
        // convert inputs before dtype-sensitive computation
        Operand* source = op->inputs[0];
        if (source->type != type)
        {
            Operand* converted = cast_before(graph, op, source, type, id);
            source->remove_consumer(op);
            op->inputs[0] = converted;
            converted->consumers.push_back(op);
        }
        dtype = Parameter();
        if ((op->type == "aten::sum" || op->type == "aten::prod" || scan) && type >= 4 && type <= 9 && type != 5)
        {
            // integer reductions promote to int64; narrow the final modular result
            Operand* result = op->outputs[0];
            const std::string name = "pt2_dtype_" + std::to_string(id++);
            Operand* promoted = graph.new_operand(name);
            promoted->type = 5;
            promoted->shape = result->shape;
            promoted->producer = op;
            op->outputs[0] = promoted;
            // insert the cast immediately after the reduction
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
