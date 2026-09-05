// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_aten_new_empty.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

// aten::new_empty produces an uninitialized tensor, usually used as a concat
// buffer (e.g. MLA kv concat: new_empty followed by slice_copy writing each
// segment). Convert it to a zeros Attribute (uninitialized values are
// nondeterministic; zeros is a deterministic choice, and if slice_copy fully
// overwrites it the result matches concat).
void convert_aten_new_empty(Graph& graph)
{
    for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
    {
        Operator* op = graph.ops[i];

        if (op->type != "aten::new_empty")
            continue;
        if (op->inputs.size() != 1 || op->outputs.size() != 1)
            continue;
        if (!op->has_param("size"))
            continue;

        const std::vector<int>& shape = op->params.at("size").ai;

        // reject dynamic / invalid (non-positive) dimensions before allocating
        size_t elem_count = 1;
        for (int s : shape)
        {
            if (s <= 0 || elem_count > (size_t)-1 / (size_t)s)
            {
                elem_count = 0;
                break;
            }
            elem_count *= (size_t)s;
        }
        if (elem_count == 0)
            continue; // cannot build a static zeros buffer

        Operand* self = op->inputs[0];
        Operand* out = op->outputs[0];

        // preserve the recorded dtype: pt2 aten::new_empty carries the dtype as
        // a param (a torch dtype enum) and the IR also records it on the output
        // operand. defaulting every buffer to f32 would change the dtype/bytes
        // seen by the following slice_copy / CopyTo and consumers.
        int dtype = out->type;
        if (op->has_param("dtype") && op->params.at("dtype").type == 2)
        {
            const int t = op->params.at("dtype").i;
            if (t == 0)
                dtype = 8; // uint8
            else if (t == 1)
                dtype = 7; // int8
            else if (t == 2)
                dtype = 6; // int16
            else if (t == 3)
                dtype = 4; // int32
            else if (t == 4)
                dtype = 5; // int64
            else if (t == 5)
                dtype = 3; // float16
            else if (t == 6)
                dtype = 1; // float32
            else if (t == 7)
                dtype = 2; // float64
            else if (t == 9)
                dtype = 10; // complex64
            else if (t == 10)
                dtype = 11; // complex128
            else if (t == 11)
                dtype = 9; // bool
            else if (t == 15)
                dtype = 13; // bfloat16
        }
        if (dtype == 0)
            dtype = 1; // unknown -> f32

        size_t es = 4;
        if (dtype == 2 || dtype == 5 || dtype == 10) es = 8; // f64/i64/c64
        if (dtype == 11) es = 16;                            // c128
        if (dtype == 3 || dtype == 6 || dtype == 13) es = 2; // f16/i16/bf16
        if (dtype == 7 || dtype == 8 || dtype == 9) es = 1;  // i8/u8/bool
        if (elem_count > (size_t)-1 / es)
            continue; // cannot allocate

        // build a zeros Attribute with the recorded dtype
        Operator* attr = graph.new_operator_before("pnnx.Attribute", op->name + "_zeros", op);
        Operand* attr_out = graph.new_operand(op->name + "_zeros_out");
        attr->outputs.push_back(attr_out);
        attr_out->producer = attr;
        attr_out->shape = shape;
        attr_out->type = dtype;
        attr_out->consumers = out->consumers;

        Attribute a;
        a.type = dtype;
        a.shape = shape;
        a.data = std::vector<char>(elem_count * es, 0); // zeros
        attr->attrs["data"] = a;

        // reconnect consumers of out to attr_out
        for (Operator* c : out->consumers)
        {
            for (size_t j = 0; j < c->inputs.size(); j++)
            {
                if (c->inputs[j] == out)
                    c->inputs[j] = attr_out;
            }
        }

        // remove op: first remove it from the consumers of self
        auto it = std::find(self->consumers.begin(), self->consumers.end(), op);
        if (it != self->consumers.end())
            self->consumers.erase(it);
        op->inputs.clear();
        op->outputs.clear();
        graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));
        delete op;

        // remove out (its consumers are already reconnected to attr_out)
        out->producer = 0;
        out->consumers.clear();
        graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), out));
        delete out;
    }
}

} // namespace ncnn

} // namespace pnnx
