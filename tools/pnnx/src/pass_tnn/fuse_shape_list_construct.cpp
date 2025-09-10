// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_shape_list_construct.h"

#include <algorithm>

namespace pnnx {

namespace tnn2pnnx {

void fuse_shape_list_construct(Graph& graph)
{
    // TODO unpool tnn.Unsqueeze

    // a0 = pnnx.Attribute @data=(1)i32
    // a1 = tnn.Unsqueeze(..., arg0=1, arg1=0)
    // y = tnn.Concat(a0, a1, ..., arg0=0)
    // tnn.Reshape(x, y, args=...) / tnn.ConstantOfShape(y)

    // prim::ListConstruct (a0, a1, ...)
    // tnn.Reshape(x, y) / tnn.ConstantOfShape(y)

    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "tnn.Concat")
                continue;

            if (op->outputs[0]->consumers.size() != 1)
                continue;

            Operator* op2 = op->outputs[0]->consumers[0];
            if (op2->type == "tnn.Reshape")
            {
                if (op2->inputs.size() != 2)
                    continue;

                if (op2->inputs[1] != op->outputs[0])
                    continue;
            }
            else if (op2->type == "tnn.ConstantOfShape")
            {
                if (op2->inputs[0] != op->outputs[0])
                    continue;
            }
            else if (op2->type == "tnn.Expand")
            {
                if (op2->inputs[1] != op->outputs[0])
                    continue;
            }
            else
            {
                continue;
            }

            matched = true;

            fprintf(stderr, "match concat + reshape/constantofshape/expand\n");

            op->type = "prim::ListConstruct";

            // drop tnn.Unsqueeze between aten::size and prim::ListConstruct

            const size_t count = op->inputs.size();
            for (size_t j = 0; j < count; j++)
            {
                Operand* r = op->inputs[j];

                if (r->producer->type != "tnn.Unsqueeze")
                    continue;

                Operator* op_uqz = r->producer;

                Operand* r0 = op_uqz->inputs[0];

                if (r0->producer->type != "aten::size")
                    continue;

                // drop tnn.Unsqueeze

                r0->remove_consumer(op_uqz);
                r->remove_consumer(op);

                op->inputs[j] = r0;
                r0->consumers.push_back(op);

                if (r->consumers.empty())
                {
                    graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), r));
                    delete r;

                    graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op_uqz));
                    delete op_uqz;
                }
            }

            if (op2->type == "tnn.Reshape")
            {
                // drop tnn.Reshape args
                op2->params.clear();
            }
            if (op2->type == "tnn.Expand")
            {
                // drop tnn.Expand args
                op2->params.clear();
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace tnn2pnnx

} // namespace pnnx
