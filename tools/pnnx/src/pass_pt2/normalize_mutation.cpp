// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_mutation.h"

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

static Operand* tensor_before(Graph& graph, Operator* before, const char* type, const char* suffix, const std::vector<Operand*>& inputs, const Operand* meta)
{
    Operator* op = graph.new_operator_before(type, before->name + suffix, before);
    op->inputs = inputs;
    for (Operand* input : inputs)
    {
        input->consumers.push_back(op);
    }
    Operand* r = graph.new_operand(op->name);
    r->producer = op;
    r->type = meta->type;
    r->shape = meta->shape;
    op->outputs.push_back(r);
    return r;
}

int normalize_pt2_mutation(Graph& graph)
{
    for (Operator* op : std::vector<Operator*>(graph.ops))
    {
        if (op->type == "aten::fill" || op->type == "aten::fill_")
        {
            // Allocation and writeback are distinct: functional fill must not
            // modify self, while fill_ still participates in alias propagation.
            const bool inplace = op->type == "aten::fill_";
            if (op->inputs[1]->type != 0)
            {
                // A zero-dimensional Tensor fill uses copy's dtype conversion
                // and broadcasting. full_like takes a Number fill value.
                op->type = inplace ? "aten::copy_" : "aten::copy";
                op->inputnames.clear();
                continue;
            }
            std::vector<Operand*> inputs = op->inputs;
            const char* suffixes[] = {"_dtype", "_layout", "_device", "_pin_memory", "_memory_format"};
            for (const char* suffix : suffixes)
            {
                inputs.push_back(constant_before(graph, op, suffix, Parameter()));
            }
            if (inplace)
            {
                Operand* filled = tensor_before(graph, op, "aten::full_like", "_filled", inputs, op->inputs[0]);
                op->inputs[1]->remove_consumer(op);
                op->inputs[1] = filled;
                filled->consumers.push_back(op);
                op->type = "aten::copy_";
            }
            else
            {
                for (size_t i = op->inputs.size(); i < inputs.size(); i++)
                {
                    inputs[i]->consumers.push_back(op);
                }
                op->inputs = inputs;
                op->type = "aten::full_like";
            }
            op->inputnames.clear();
        }
    }
    return 0;
}

} // namespace pnnx
