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

#include "eliminate_reshape_shape_expression.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <stack>
#include <vector>
#include <string>

namespace pnnx {

static bool token_is_interger_literal(const std::string& t)
{
    std::istringstream iss(t);
    int f;
    iss >> std::noskipws >> f;
    return iss.eof() && !iss.fail();
}

static void build_shape(const std::string& expr, std::vector<int>& shape, std::vector<std::string>& expr_tokens)
{
    std::string listexpr = expr.substr(1, expr.size() - 2);

    std::string t;
    std::string et;
    int level = 0;
    for (size_t i = 0; i < listexpr.size(); i++)
    {
        char ch = listexpr[i];

        if (ch == '(' || ch == '[')
        {
            level += 1;
            t = "-1";
            et += ch;
        }
        else if (ch == ')' || ch == ']')
        {
            level -= 1;
            t = "-1";
            et += ch;
        }
        else if (level == 0 && ch == ',')
        {
            int dimsize = token_is_interger_literal(t) ? std::stoi(t) : -1;
            shape.push_back(dimsize);
            expr_tokens.push_back(et);
            t.clear();
            et.clear();
        }
        else
        {
            t += ch;
            et += ch;
        }
    }

    if (level == 0 && !t.empty())
    {
        int dimsize = token_is_interger_literal(t) ? std::stoi(t) : -1;
        shape.push_back(dimsize);
    }

    if (level == 0 && !et.empty())
    {
        expr_tokens.push_back(et);
    }
}

static std::string build_expr(const std::vector<std::string>& expr_tokens)
{
    std::string expr;

    expr += '[';
    for (int i = 0; i < (int)expr_tokens.size(); i++)
    {
        expr += expr_tokens[i];
        if (i != (int)expr_tokens.size() - 1)
            expr += ',';
    }
    expr += ']';

    return expr;
}

void eliminate_reshape_shape_expression(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.view" && op->type != "Tensor.reshape")
                continue;

            if (op->inputs.size() != 2)
                continue;

            Operator* op_expr = op->inputs[1]->producer;
            if (op_expr->type != "pnnx.Expression")
                continue;

            std::string expr = op_expr->params.at("expr").s;
            if (expr.empty() || expr[0] != '[')
                continue;

            std::vector<int> outshape = op->outputs[0]->shape;
            if (outshape.empty())
                continue;

            std::vector<int> shape;
            std::vector<std::string> expr_tokens;
            build_shape(expr, shape, expr_tokens);

            // replace -1 with static dim-size
            for (size_t j = 0; j < outshape.size(); j++)
            {
                if (outshape[j] != -1)
                {
                    shape[j] = outshape[j];
                    expr_tokens[j] = std::to_string(outshape[j]);
                }
            }

            // if only one dynamic dim-size, drop expression
            int dynamic_dim_count = 0;
            for (size_t j = 0; j < shape.size(); j++)
            {
                if (shape[j] == -1)
                {
                    dynamic_dim_count += 1;
                }
            }

            if (dynamic_dim_count > 1)
            {
                op_expr->params["expr"] = build_expr(expr_tokens);
                continue;
            }

            matched = true;

            op->params["shape"] = shape;

            op->inputs.resize(1);
            op_expr->outputs[0]->remove_consumer(op);

            if (op_expr->outputs[0]->consumers.size() == 0)
            {
                // remove expression operator
                for (auto x : op_expr->inputs)
                {
                    x->remove_consumer(op_expr);
                }

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

    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];

        if (op->type != "Tensor.view" && op->type != "Tensor.reshape")
            continue;

        if (op->inputs.size() != 1)
            continue;

        std::vector<int> outshape = op->outputs[0]->shape;
        if (outshape.empty())
            continue;

        std::vector<int> shape = op->params.at("shape").ai;

        // replace -1 with static dim-size
        for (size_t j = 0; j < outshape.size(); j++)
        {
            if (outshape[j] != -1)
            {
                shape[j] = outshape[j];
            }
        }

        op->params["shape"] = shape;
    }
}

} // namespace pnnx
