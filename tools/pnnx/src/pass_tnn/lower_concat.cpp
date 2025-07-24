// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_concat.h"

#include "pass_level2.h"

namespace pnnx {

namespace tnn2pnnx {

void lower_concat(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];

        if (op->type != "tnn.Concat")
            continue;

        const int dim = op->params["arg0"].i;

        op->type = "aten::cat";
        op->params.clear();
        op->params["dim"] = dim;

        // insert listconstruct for inputs
        Operator* op0 = graph.new_operator_before("prim::ListConstruct", op->name + "_lc", op);
        Operand* r = graph.new_operand(op->name + "_lc");

        r->producer = op0;
        r->consumers.push_back(op);

        op0->outputs.push_back(r);

        for (size_t j = 0; j < op->inputs.size(); j++)
        {
            Operand* x = op->inputs[j];

            x->remove_consumer(op);
            x->consumers.push_back(op0);
            op0->inputs.push_back(x);
        }

        op->inputs.clear();
        op->inputs.push_back(r);
    }
}

} // namespace tnn2pnnx

} // namespace pnnx
