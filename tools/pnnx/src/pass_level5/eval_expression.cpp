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

#include <fenv.h>
#include <float.h>
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

static bool token_is_complex(const std::string& t)
{
    // 2.000000e+00+3.000000e+00j
    if (t[t.size() - 1] != 'j')
        return false;

    return true;
}

static bool token_is_literal(const std::string& t)
{
    if (token_is_complex(t))
        return true;

    std::istringstream iss(t);
    float f;
    iss >> std::noskipws >> f;
    return iss.eof() && !iss.fail();
}

static bool token_is_interger_literal(const std::string& t)
{
    std::istringstream iss(t);
    int f;
    iss >> std::noskipws >> f;
    return iss.eof() && !iss.fail();
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
                    if (bi < 0)
                        bi = op->inputs[input_index]->shape.size() + bi;
                    int r = op->inputs[input_index]->shape[bi];
                    if (r == -1)
                    {
                        // do not evaluate dynamic size info as -1
                        // just keep the size expression
                        std::string r = std::string("size(") + a + "," + b + ")";
                        exprstack.push(r);
                    }
                    else
                    {
                        exprstack.push(std::to_string(r));
                    }
                }
            }
            else
            {
                std::string r = std::string("size(") + a + "," + b + ")";
                exprstack.push(r);
            }
        }
        else if (t == "int"
                 || t == "abs"
                 || t == "acos"
                 || t == "acosh"
                 || t == "asin"
                 || t == "asinh"
                 || t == "atan"
                 || t == "atanh"
                 || t == "ceil"
                 || t == "cos"
                 || t == "cosh"
                 || t == "exp"
                 || t == "floor"
                 || t == "log"
                 || t == "log10"
                 || t == "neg"
                 || t == "reciprocal"
                 || t == "round"
                 || t == "rsqrt"
                 || t == "sign"
                 || t == "sin"
                 || t == "sinh"
                 || t == "sqrt"
                 || t == "square"
                 || t == "tan"
                 || t == "tanh"
                 || t == "trunc"
                 || t == "torch.bool"
                 || t == "torch.float"
                 || t == "torch.long")
        {
            std::string a = exprstack.top();
            exprstack.pop();

            if (token_is_literal(a))
            {
                float af = std::stof(a);

                if (t == "int")
                {
                    int r = int(af);
                    if (token_is_interger_literal(a))
                    {
                        r = std::stoi(a);
                    }

                    exprstack.push(std::to_string(r));
                }
                if (t == "abs")
                {
                    float r = abs(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "acos")
                {
                    float r = acos(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "acosh")
                {
                    float r = acosh(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "asin")
                {
                    float r = asin(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "asinh")
                {
                    float r = asinh(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "atan")
                {
                    float r = atan(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "atanh")
                {
                    float r = atanh(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "ceil")
                {
                    float r = ceil(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "cos")
                {
                    float r = cos(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "cosh")
                {
                    float r = cosh(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "exp")
                {
                    float r = exp(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "floor")
                {
                    float r = floor(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "log")
                {
                    float r = log(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "log10")
                {
                    float r = log10(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "neg")
                {
                    float r = -af;
                    exprstack.push(std::to_string(r));
                }
                if (t == "reciprocal")
                {
                    float r = 1.f / af;
                    exprstack.push(std::to_string(r));
                }
                if (t == "round")
                {
                    // round to nearest even
                    int old_rm = fegetround();
                    fesetround(FE_TONEAREST);
                    float r = nearbyintf(af);
                    fesetround(old_rm);
                    exprstack.push(std::to_string(r));
                }
                if (t == "rsqrt")
                {
                    float r = 1.f / sqrt(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "sign")
                {
                    float r = af > 0.f ? 1.f : (af == 0.f ? 0.f : -1.f);
                    exprstack.push(std::to_string(r));
                }
                if (t == "sin")
                {
                    float r = sin(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "sinh")
                {
                    float r = sinh(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "sqrt")
                {
                    float r = sqrt(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "square")
                {
                    float r = af * af;
                    exprstack.push(std::to_string(r));
                }
                if (t == "tan")
                {
                    float r = tan(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "tanh")
                {
                    float r = tanh(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "trunc")
                {
                    float r = trunc(af);
                    exprstack.push(std::to_string(r));
                }
                if (t == "torch.bool")
                {
                    int r = int(af);
                    if (token_is_interger_literal(a))
                    {
                        r = std::stoi(a);
                    }

                    exprstack.push(r == 0 ? "False" : "True");
                }
                if (t == "torch.float")
                {
                    float r = af;
                    exprstack.push(std::to_string(r));
                }
                if (t == "torch.long")
                {
                    long r = long(af);
                    if (token_is_interger_literal(a))
                    {
                        r = std::stol(a);
                    }

                    exprstack.push(std::to_string(r));
                }
            }
            else
            {
                std::string r = t + "(" + a + ")";
                exprstack.push(r);
            }
        }
        else if (t == "atan2"
                 || t == "add"
                 || t == "sub"
                 || t == "max"
                 || t == "maximum"
                 || t == "min"
                 || t == "minimum"
                 || t == "mul"
                 || t == "div"
                 || t == "floor_divide"
                 || t == "fmod"
                 || t == "pow"
                 || t == "remainder")
        {
            std::string a = exprstack.top();
            exprstack.pop();
            std::string b = exprstack.top();
            exprstack.pop();

            if (token_is_literal(a) && token_is_literal(b))
            {
                float af = std::stof(a);
                float bf = std::stof(b);

                if (t == "atan2")
                {
                    float r = atan2(af, bf);
                    exprstack.push(std::to_string(r));
                }
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
                if (t == "max" || t == "maximum")
                {
                    float r = std::max(af, bf);
                    exprstack.push(std::to_string(r));
                }
                if (t == "minimum")
                {
                    float r = std::min(af, bf);
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
                if (t == "fmod")
                {
                    float r = fmod(af, bf);
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
        else if (t == "and" || t == "or" || t == "xor" || t == "lshift" || t == "rshift")
        {
            std::string a = exprstack.top();
            exprstack.pop();
            std::string b = exprstack.top();
            exprstack.pop();

            if (token_is_interger_literal(a) && token_is_interger_literal(b))
            {
                int ai = std::stoi(a);
                int bi = std::stoi(b);

                if (t == "and")
                {
                    int r = ai & bi;
                    exprstack.push(std::to_string(r));
                }
                if (t == "or")
                {
                    int r = ai | bi;
                    exprstack.push(std::to_string(r));
                }
                if (t == "xor")
                {
                    int r = ai ^ bi;
                    exprstack.push(std::to_string(r));
                }
                if (t == "lshift")
                {
                    int r = ai << bi;
                    exprstack.push(std::to_string(r));
                }
                if (t == "rshift")
                {
                    int r = ai >> bi;
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
    while (!exprstack.empty())
    {
        r += std::string(",") + exprstack.top();
        exprstack.pop();
    }

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
