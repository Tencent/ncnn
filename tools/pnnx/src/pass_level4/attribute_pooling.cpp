// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "attribute_pooling.h"

#include <algorithm>

namespace pnnx {

void attribute_pooling(Graph& graph)
{
    std::vector<Operator*> attribute_ops;
    for (int i = 0; i < (int)graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];

        if (op->type != "pnnx.Attribute")
            continue;

        if (!op->has_attr("data"))
            continue;

        const Attribute& attr = op->attrs.at("data");

        bool pooled = false;
        for (size_t j = 0; j < attribute_ops.size(); j++)
        {
            Operator* op_a = attribute_ops[j];
            const Attribute& attr_a = op_a->attrs.at("data");

            if (attr_a == attr)
            {
                // reset op_out uses to op_a_out
                Operand* op_out = op->outputs[0];
                Operand* op_a_out = op_a->outputs[0];

                for (auto& x : op_out->consumers)
                {
                    for (size_t k = 0; k < x->inputs.size(); k++)
                    {
                        if (x->inputs[k] == op_out)
                            x->inputs[k] = op_a_out;
                    }

                    op_a_out->consumers.push_back(x);
                }

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), op_out));
                delete op_out;

                graph.ops.erase(graph.ops.begin() + i);
                delete op;

                i--;

                pooled = true;
                break;
            }
        }

        if (!pooled)
        {
            attribute_ops.push_back(op);
        }
    }
}

} // namespace pnnx
