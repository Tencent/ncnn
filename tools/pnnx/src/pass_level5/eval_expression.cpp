// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "eval_expression.h"

#include <math.h>

#include <iostream>
#include <sstream>
#include <algorithm>
#include <stack>
#include <vector>
#include <string>

namespace pnnx {

static bool token_is_argument(const std::string& t)
{
    if (t[0] != '@' || t.size() < 2)
        return false;

    for (size_t i = 1; i < t.size(); i++)
    {
        if (t[i] < '0' || t[i] > '9')
            return false;
    }

    return true;
}

static bool token_is_literal(const std::string& t)
{
    std::istringstream iss(t);
    float f;
    iss >> std::noskipws >> f;
    return iss.eof() && !iss.fail();

    //     for (size_t i = 0; i < t.size(); i++)
    //     {
    //         if (i == 0 && t[i] == '-')
    //             continue;
    //
    //         if (t[i] < '0' || t[i] > '9')
    //         {
    //             if (t[i] != '.' && t[i] != 'e')
    //                 return false;
    //         }
    //     }
    //
    //     return true;
}

static std::string eval_expression(const Operator* op)
{
    std::string expr = op->params.at("expr").s;

    //     fprintf(stderr, "eval_expression %s\n", expr.c_str());

    // split into tokens
    std::vector<std::string> tokens;
    {
        std::string t;
        for (size_t i = 0; i < expr.size(); i++)
        {
            char ch = expr[i];

            if (ch == '[') // list
            {
                t += ch;
                tokens.push_back(t);
                t.clear();
            }
            else if (ch == '(' || ch == ')' || ch == ',' || ch == ']')
            {
                if (!t.empty())
                {
                    tokens.push_back(t);
                    t.clear();
                }
            }
            else
            {
                t += ch;
            }
        }

        if (!t.empty())
        {
            tokens.push_back(t);
        }
    }

    // scan and stack
    std::stack<std::string> exprstack;
    for (int i = (int)tokens.size() - 1; i >= 0; i--)
    {
        const std::string& t = tokens[i];

        if (t == "size")
        {
            std::string a = exprstack.top();
            exprstack.pop();
            std::string b = exprstack.top();
            exprstack.pop();

            if (token_is_argument(a) && token_is_literal(b))
            {
                int input_index = std::stoi(a.substr(1));
                if (op->inputs[input_index]->shape.empty())
                {
                    std::string r = std::string("size(") + a + "," + b + ")";
                    exprstack.push(r);
                }
                else
                {
                    int bi = std::stoi(b);
                    int r = op->inputs[input_index]->shape[bi];
                    exprstack.push(std::to_string(r));
                }
            }
            else
            {
                std::string r = std::string("size(") + a + "," + b + ")";
                exprstack.push(r);
            }
        }
        else if (t == "int" || t == "sqrt" || t == "rsqrt" || t == "neg")
        {
            std::string a = exprstack.top();
            exprstack.pop();

            if (token_is_literal(a))
            {
                float af = std::stof(a);

                if (t == "int")
                {
                    int r = int(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "sqrt")
                {
                    float r = sqrt(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "rsqrt")
                {
                    float r = 1.f / sqrt(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "neg")
                {
                    float r = -af;
                    exprstack.push(std::to_string(r));
                }
            }
            else
            {
                std::string r = t + "(" + a + ")";
                exprstack.push(r);
            }
        }
        else if (t == "add" || t == "sub" || t == "mul" || t == "div" || t == "floor_divide" || t == "pow" || t == "remainder")
        {
            std::string a = exprstack.top();
            exprstack.pop();
            std::string b = exprstack.top();
            exprstack.pop();

            if (token_is_literal(a) && token_is_literal(b))
            {
                float af = std::stof(a);
                float bf = std::stof(b);

                if (t == "add")
                {
                    float r = af + bf;
                    exprstack.push(std::to_string(r));
                }
                if (t == "sub")
                {
                    float r = af - bf;
                    exprstack.push(std::to_string(r));
                }
                if (t == "mul")
                {
                    float r = af * bf;
                    exprstack.push(std::to_string(r));
                }
                if (t == "div")
                {
                    float r = af / bf;
                    exprstack.push(std::to_string(r));
                }
                if (t == "floor_divide")
                {
                    int r = (int)af / (int)bf;
                    exprstack.push(std::to_string(r));
                }
                if (t == "pow")
                {
                    float r = pow(af, bf);
                    exprstack.push(std::to_string(r));
                }
                if (t == "remainder")
                {
                    float r = fmod(af, bf);
                    if (af * bf < 0)
                        r += bf;
                    exprstack.push(std::to_string(r));
                }
            }
            else
            {
                std::string r = t + "(" + a + "," + b + ")";
                exprstack.push(r);
            }
        }
        else if (t == "[") // list
        {
            std::vector<std::string> elements;
            while (!exprstack.empty())
            {
                std::string a = exprstack.top();
                exprstack.pop();

                elements.push_back(a);
            }

            std::string r = "[";
            for (int j = 0; j < (int)elements.size() - 1; j++)
            {
                r += elements[j];
                if (j + 1 != (int)elements.size())
                    r += ",";
            }
            if (!elements.empty())
            {
                r += elements[elements.size() - 1];
            }
            r += "]";

            exprstack.push(r);
        }
        else if (t[0] == '@')
        {
            exprstack.push(t);
        }
        else
        {
            // literal
            exprstack.push(t);
        }
    }

    std::string r = exprstack.top();
    exprstack.pop();

    //     fprintf(stderr, "eval_expression return %s\n", r.c_str());

    return r;
}

static std::string canonicalize_arguments(const Operator* op, std::vector<Operand*>& inputs)
{
    std::string expr = op->params.at("expr").s;

    // split into tokens
    std::vector<std::string> tokens;
    {
        std::string t;
        for (size_t i = 0; i < expr.size(); i++)
        {
            char ch = expr[i];

            if (ch == '[') // list
            {
                t += ch;
                tokens.push_back(t);
                t.clear();
            }
            else if (ch == '(' || ch == ')' || ch == ',' || ch == ']')
            {
                if (!t.empty())
                {
                    tokens.push_back(t);
                    t.clear();
                }

                t += ch;
                tokens.push_back(t);
                t.clear();
            }
            else
            {
                t += ch;
            }
        }

        if (!t.empty())
        {
            tokens.push_back(t);
        }
    }

    std::string r;
    for (auto t : tokens)
    {
        if (t[0] == '@')
        {
            int input_index = std::stoi(t.substr(1));
            Operand* operand = op->inputs[input_index];

            int new_input_index;

            auto it = std::find(inputs.begin(), inputs.end(), operand);
            if (it == inputs.end())
            {
                new_input_index = inputs.size();
                inputs.push_back(operand);
            }
            else
            {
                new_input_index = it - inputs.begin();
            }
            r += std::string("@") + std::to_string(new_input_index);
        }
        else
        {
            r += t;
        }
    }

    //     fprintf(stderr, "canonicalize_arguments return %s\n", r.c_str());

    return r;
}

void eval_expression(Graph& graph)
{
    for (Operator* op : graph.ops)
    {
        if (op->type != "pnnx.Expression")
            continue;

        std::string expr_eval = eval_expression(op);

        op->params["expr"] = expr_eval;

        std::vector<Operand*> inputs;
        std::string expr_canonicalize = canonicalize_arguments(op, inputs);

        op->params["expr"] = expr_canonicalize;

        for (auto r : op->inputs)
        {
            r->remove_consumer(op);
        }

        for (auto r : inputs)
        {
            r->consumers.push_back(op);
        }

        op->inputs = inputs;
    }
}

} // namespace pnnx
