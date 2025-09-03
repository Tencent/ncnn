// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "attribute_unpooling.h"

#include <algorithm>

namespace pnnx {

void attribute_unpooling(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "pnnx.Attribute")
                continue;

            Operand* attr = op->outputs[0];

            if (attr->consumers.size() < 2)
                continue;

            // multiple modules share same weight
            matched = true;

            for (int i = 1; i < (int)attr->consumers.size(); i++)
            {
                Operator* x = attr->consumers[i];

                Operator* op2 = graph.new_operator_after("pnnx.Attribute", op->name + "_" + std::to_string(i), op);

                op2->inputnames = op->inputnames;
                op2->params = op->params;
                op2->attrs = op->attrs;

                Operand* attr2 = graph.new_operand(attr->name + "_" + std::to_string(i));

                attr2->type = attr->type;
                attr2->shape = attr->shape;
                attr2->params = attr->params;

                op2->outputs.push_back(attr2);

                attr2->producer = op2;
                attr2->consumers.push_back(x);

                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == attr)
                        x->inputs[j] = attr2;
                }
            }

            attr->consumers.resize(1);

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
