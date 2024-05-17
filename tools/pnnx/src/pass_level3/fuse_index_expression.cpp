// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "fuse_index_expression.h"

#include <algorithm>

namespace pnnx {

static void replaceAll(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos)
    {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

static std::string fuse_attribute_expression(Operator* op_expr)
{
    std::string expr = op_expr->params["expr"].s;

    for (int i = (int)op_expr->inputs.size() - 1; i >= 0; i--)
    {
        Operator* op_attr = op_expr->inputs[i]->producer;

        const Attribute& attr = op_attr->attrs.begin()->second;

        std::string attr_expr;

        int count = attr.shape[0];

        if (attr.type == 9)
        {
            // bool
            const char* pdata = (const char*)attr.data.data();
            attr_expr += "[";
            for (int j = 0; j < count; j++)
            {
                const char* ls = pdata[j] != 0 ? "True" : "False";
                attr_expr += ls;

                if (j != count - 1)
                    attr_expr += ",";
            }
            attr_expr += "]";
        }
        else if (attr.type == 5)
        {
            // i64
            const int64_t* pdata = (const int64_t*)attr.data.data();
            attr_expr += "[";
            for (int j = 0; j < count; j++)
            {
                int64_t n = pdata[j];
                attr_expr += std::to_string(n);

                if (j != count - 1)
                    attr_expr += ",";
            }
            attr_expr += "]";
        }
        else
        {
            fprintf(stderr, "unsupported index expression input %d attr type %d\n", i, attr.type);
        }

        // replace @i with attr_expr
        replaceAll(expr, "@" + std::to_string(i), attr_expr);
    }

    return expr;
}

void fuse_index_expression(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.index")
                continue;

            if (op->inputs.size() != 2)
                continue;

            if (op->inputs[1]->consumers.size() != 1)
                continue;

            Operator* op2 = op->inputs[1]->producer;
            if (op2->type != "pnnx.Expression")
                continue;

            bool all_inputs_fusable = true;
            for (Operand* a : op2->inputs)
            {
                if (a->producer->type != "pnnx.Attribute")
                {
                    all_inputs_fusable = false;
                    break;
                }
            }
            if (!all_inputs_fusable)
                continue;

            matched = true;

            std::string expr = fuse_attribute_expression(op2);

            op->params["expr"] = expr;

            op->inputs[1]->producer = 0;
            op->inputs[1]->remove_consumer(op);

            op->inputs.resize(1);

            for (auto& x : op2->inputs)
            {
                x->remove_consumer(op2);
            }

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
