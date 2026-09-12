// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "torch_baddbmm.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

void convert_aten_baddbmm(Graph& graph)
{
    for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
    {
        Operator* op = graph.ops[i];

        // pass_level2 normalizes the 5-input aten::baddbmm to torch.baddbmm
        // (constant folding leaves the three tensor inputs plus alpha/beta
        // params); the bare 3-input aten::baddbmm stays as-is. accept both.
        if (op->type != "aten::baddbmm" && op->type != "torch.baddbmm")
            continue;

        // 3-input variant: self, batch1, batch2 (beta=1, alpha=1) -> self + batch1 @ batch2
        if (op->inputs.size() != 3)
            continue;

        // torch.baddbmm carries folded alpha/beta params; only the unit form
        // maps to mm + add, so decline any explicit scaling instead of
        // silently dropping it
        if (op->type == "torch.baddbmm")
        {
            float alpha = 1.f;
            float beta = 1.f;
            if (op->params.count("alpha"))
            {
                const Parameter& p = op->params.at("alpha");
                alpha = p.type == 3 ? p.f : (p.type == 2 ? (float)p.i : 1.f);
            }
            if (op->params.count("beta"))
            {
                const Parameter& p = op->params.at("beta");
                beta = p.type == 3 ? p.f : (p.type == 2 ? (float)p.i : 1.f);
            }
            if (alpha != 1.f || beta != 1.f)
            {
                fprintf(stderr, "unsupported baddbmm alpha=%f beta=%f scaling\n", alpha, beta);
                continue;
            }
        }

        Operand* self = op->inputs[0];
        Operand* batch1 = op->inputs[1];
        Operand* batch2 = op->inputs[2];

        // MatMul(batch1, batch2)
        Operator* mm = graph.new_operator_before("MatMul", op->name + "_mm", op);
        mm->inputs.push_back(batch1);
        mm->inputs.push_back(batch2);
        batch1->consumers.push_back(mm);
        batch2->consumers.push_back(mm);

        Operand* mm_out = graph.new_operand(mm->name + "_out");
        mm_out->producer = mm;
        mm->outputs.push_back(mm_out);

        // BinaryOp add(mm_out, self) 0=0
        Operator* add = graph.new_operator_before("BinaryOp", op->name + "_add", op);
        add->params["0"] = 0; // add
        add->inputs.push_back(mm_out);
        add->inputs.push_back(self);
        mm_out->consumers.push_back(add);
        self->consumers.push_back(add);

        for (Operand* r : op->outputs)
        {
            r->producer = add;
            add->outputs.push_back(r);
        }

        // remove baddbmm
        for (Operand* in : op->inputs)
        {
            in->consumers.erase(std::find(in->consumers.begin(), in->consumers.end(), op));
        }
        op->inputs.clear();
        op->outputs.clear();

        graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));
        delete op;
    }
}

} // namespace ncnn

} // namespace pnnx
