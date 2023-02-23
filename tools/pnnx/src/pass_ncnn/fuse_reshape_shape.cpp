// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "fuse_reshape_shape.h"

#include <math.h>

#include <iostream>
#include <sstream>
#include <algorithm>
#include <stack>
#include <vector>
#include <string>

namespace pnnx {

namespace ncnn {

static bool token_is_interger_literal(const std::string& t)
{
    std::istringstream iss(t);
    int f;
    iss >> std::noskipws >> f;
    return iss.eof() && !iss.fail();
}

static std::vector<int> build_shape(const std::string& expr)
{
    std::string listexpr = expr.substr(1, expr.size() - 2);

    std::vector<int> shape;

    std::string t;
    int level = 0;
    for (size_t i = 0; i < listexpr.size(); i++)
    {
        char ch = listexpr[i];

        if (ch == '(' || ch == '[')
        {
            level += 1;
            t = "-1";
        }
        else if (ch == ')' || ch == ']')
        {
            level -= 1;
            t = "-1";
        }
        else if (level == 0 && ch == ',')
        {
            int dimsize = token_is_interger_literal(t) ? std::stoi(t) : -1;
            shape.push_back(dimsize);
            t.clear();
        }
        else
        {
            t += ch;
        }
    }

    if (level == 0 && !t.empty())
    {
        int dimsize = token_is_interger_literal(t) ? std::stoi(t) : -1;
        shape.push_back(dimsize);
    }

    return shape;
}

void fuse_reshape_shape(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.reshape" && op->type != "Tensor.view")
                continue;

            if (op->inputs.size() != 2)
                continue;

            Operator* op_expr = op->inputs[1]->producer;
            if (op_expr->type != "pnnx.Expression")
                continue;

            std::string expr = op_expr->params.at("expr").s;
            if (expr.empty() || expr[0] != '[')
                continue;

            // case1 only one dynamic dim-size
            //      [1,64,int(mul(1,size(@0,3)))]
            // case2 two dynamic dim-size with one for batch axis
            //      [int(size(@0,0)),64,int(mul(1,size(@0,3)))]

            std::vector<int> shape = build_shape(expr);

            const int batch_index_0 = op->inputs[0]->params["__batch_index"].i;
            const int batch_index_1 = op->outputs[0]->params["__batch_index"].i;

            const int batch_index = batch_index_1 == 233 ? batch_index_0 : batch_index_1;

            // count dynamic
            int dynamic_dimsize_count = 0;
            for (int j = 0; j < (int)shape.size(); j++)
            {
                if (shape[j] == -1 && j != batch_index)
                {
                    dynamic_dimsize_count++;
                }
            }

            if (dynamic_dimsize_count != 1)
                continue;

            matched = true;

            if (batch_index != 233)
                shape[batch_index] = 1;

            op->params["shape"] = shape;

            op->inputs.resize(1);
            op_expr->outputs[0]->remove_consumer(op);

            if (op_expr->outputs[0]->consumers.size() == 0)
            {
                // remove expression operator
                op_expr->inputs[0]->remove_consumer(op_expr);

                Operand* op_expr_out = op_expr->outputs[0];

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), op_expr_out));
                delete op_expr_out;

                op_expr->inputs.clear();
                op_expr->outputs.clear();

                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op_expr));
                delete op_expr;
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
