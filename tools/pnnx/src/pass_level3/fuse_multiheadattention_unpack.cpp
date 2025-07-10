// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_multiheadattention_unpack.h"
#include <algorithm>
#include "pass_level2.h"

namespace pnnx {

void fuse_multiheadattention_unpack(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "nn.MultiheadAttention")
                continue;

            if (op->outputs.size() != 1)
                continue;

            if (op->outputs[0]->consumers.size() != 1)
                continue;

            Operator* op2 = op->outputs[0]->consumers[0];
            if (op2->type != "prim::TupleUnpack")
                continue;

            matched = true;

            op->outputs[0]->producer = 0;
            op->outputs[0]->remove_consumer(op2);

            for (auto& x : op2->outputs)
            {
                x->producer = op;
            }

            op->outputs = op2->outputs;

            op2->inputs.clear();
            op2->outputs.clear();

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op2));

            delete op2;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
