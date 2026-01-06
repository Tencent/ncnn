// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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

#include "utils.h"

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
    struct typed_expr
    {
        std::string expr;
        int type; // 0=i 1=f 2=cp 3=other
        int literal;
        int i;
        float f;

        typed_expr()
            : type(3), literal(0), i(0), f(0.f)
        {
        }

        typed_expr(int _i)
            : type(0), literal(1), i(_i), f(0.f)
        {
            // fprintf(stderr, "typed_expr i %d\n", i);
        }

        typed_expr(float _f)
            : type(1), literal(1), i(0), f(_f)
        {
            // fprintf(stderr, "typed_expr f %f\n", f);
        }

        typed_expr(const std::string& _expr)
            : expr(_expr), type(3), literal(0), i(0), f(0.f)
        {
            // fprintf(stderr, "typed_expr ? %s\n", expr.c_str());
        }

        typed_expr(const std::string& _expr, int _type)
            : expr(_expr), type(_type), literal(0), i(0), f(0.f)
        {
            // fprintf(stderr, "typed_expr %d %s\n", type, expr.c_str());
        }

        bool is_literal() const
        {
            return literal == 1;
        }

        bool is_interger_literal() const
        {
            return type == 0 && literal == 1;
        }

        std::string to_expr() const
        {
            if (literal == 1)
            {
                if (type == 0)
                    return std::to_string(i);
                if (type == 1)
                    return float_to_string(f);
            }

            return expr;
        }
    };

    std::stack<typed_expr> exprstack;
    for (int i = (int)tokens.size() - 1; i >= 0; i--)
    {
        const std::string& t = tokens[i];

        if (t == "size")
        {
            std::string a = exprstack.top().to_expr();
            exprstack.pop();

            if (exprstack.empty())
            {
                std::string r = std::string("size(") + a + ")";
                exprstack.push(r);
            }
            else
            {
                typed_expr b = exprstack.top();
                exprstack.pop();

                if (token_is_argument(a) && b.is_interger_literal())
                {
                    int bi = b.i;

                    int input_index = std::stoi(a.substr(1));
                    if (op->inputs[input_index]->shape.empty())
                    {
                        std::string r = std::string("size(") + a + "," + std::to_string(bi) + ")";
                        exprstack.push(typed_expr(r, 0));
                    }
                    else
                    {
                        if (bi < 0)
                            bi = op->inputs[input_index]->shape.size() + bi;
                        int r = op->inputs[input_index]->shape[bi];
                        if (r == -1)
                        {
                            // do not evaluate dynamic size info as -1
                            // just keep the size expression
                            std::string r = std::string("size(") + a + "," + std::to_string(bi) + ")";
                            exprstack.push(typed_expr(r, 0));
                        }
                        else
                        {
                            exprstack.push(r);
                        }
                    }
                }
                else
                {
                    std::string r = std::string("size(") + a + "," + b.to_expr() + ")";
                    exprstack.push(typed_expr(r, 0));
                }
            }
        }
        else if (t == "int"
                 || t == "ceil"
                 || t == "floor"
                 || t == "round"
                 || t == "trunc")
        {
            typed_expr a = exprstack.top();
            exprstack.pop();

            if (a.is_interger_literal())
            {
                // noop
                exprstack.push(a);
            }
            else if (a.is_literal())
            {
                const float af = a.f;

                int r = 0;
                if (t == "int")
                {
                    r = int(af);
                }
                if (t == "ceil")
                {
                    r = ceil(af);
                }
                if (t == "floor")
                {
                    r = floor(af);
                }
                if (t == "round")
                {
                    // round to nearest even
                    int old_rm = fegetround();
                    fesetround(FE_TONEAREST);
                    r = nearbyintf(af);
                    fesetround(old_rm);
                }
                if (t == "trunc")
                {
                    r = trunc(af);
                }
                exprstack.push(r);
            }
            else if (a.type == 0)
            {
                // noop
                exprstack.push(a);
            }
            else
            {
                std::string r = t + "(" + a.to_expr() + ")";
                if (a.type < 2)
                {
                    exprstack.push(typed_expr(r, 0));
                }
                else
                {
                    exprstack.push(r);
                }
            }
        }
        else if (t == "neg"
                 || t == "sign"
                 || t == "square")
        {
            typed_expr a = exprstack.top();
            exprstack.pop();

            if (a.is_interger_literal())
            {
                const int ai = a.i;

                int r = 0;
                if (t == "neg")
                {
                    r = -ai;
                }
                if (t == "sign")
                {
                    r = ai > 0 ? 1 : (ai == 0 ? 0 : -1);
                }
                if (t == "square")
                {
                    r = ai * ai;
                }

                exprstack.push(r);
            }
            else if (a.is_literal())
            {
                const float af = a.f;

                float r = 0;
                if (t == "neg")
                {
                    r = -af;
                }
                if (t == "sign")
                {
                    r = af > 0.f ? 1.f : (af == 0.f ? 0.f : -1.f);
                }
                if (t == "square")
                {
                    r = af * af;
                }
                exprstack.push(r);
            }
            else
            {
                std::string r = t + "(" + a.to_expr() + ")";
                exprstack.push(typed_expr(r, a.type));
            }
        }
        else if (t == "abs"
                 || t == "acos"
                 || t == "acosh"
                 || t == "asin"
                 || t == "asinh"
                 || t == "atan"
                 || t == "atanh"
                 || t == "cos"
                 || t == "cosh"
                 || t == "erf"
                 || t == "exp"
                 || t == "log"
                 || t == "log10"
                 || t == "reciprocal"
                 || t == "rsqrt"
                 || t == "sin"
                 || t == "sinh"
                 || t == "sqrt"
                 || t == "tan"
                 || t == "tanh"
                 || t == "torch.float")
        {
            typed_expr a = exprstack.top();
            exprstack.pop();

            if (a.is_literal())
            {
                float af = a.type == 0 ? a.i : a.f;

                float r = 0.f;
                if (t == "abs")
                {
                    r = abs(af);
                }
                if (t == "acos")
                {
                    r = acos(af);
                }
                if (t == "acosh")
                {
                    r = acosh(af);
                }
                if (t == "asin")
                {
                    r = asin(af);
                }
                if (t == "asinh")
                {
                    r = asinh(af);
                }
                if (t == "atan")
                {
                    r = atan(af);
                }
                if (t == "atanh")
                {
                    r = atanh(af);
                }
                if (t == "cos")
                {
                    r = cos(af);
                }
                if (t == "cosh")
                {
                    r = cosh(af);
                }
                if (t == "erf")
                {
                    r = erf(af);
                }
                if (t == "exp")
                {
                    r = exp(af);
                }
                if (t == "log")
                {
                    r = log(af);
                }
                if (t == "log10")
                {
                    r = log10(af);
                }
                if (t == "reciprocal")
                {
                    r = 1.f / af;
                }
                if (t == "rsqrt")
                {
                    r = 1.f / sqrt(af);
                }
                if (t == "sin")
                {
                    r = sin(af);
                }
                if (t == "sinh")
                {
                    r = sinh(af);
                }
                if (t == "sqrt")
                {
                    r = sqrt(af);
                }
                if (t == "tan")
                {
                    r = tan(af);
                }
                if (t == "tanh")
                {
                    r = tanh(af);
                }
                if (t == "torch.float")
                {
                    // noop
                    r = af;
                }
                exprstack.push(r);
            }
            else
            {
                std::string r = t + "(" + a.to_expr() + ")";
                exprstack.push(r);
            }
        }
        else if (t == "torch.bool"
                 || t == "torch.long")
        {
            typed_expr a = exprstack.top();
            exprstack.pop();

            if (a.is_literal())
            {
                if (t == "torch.bool")
                {
                    bool r = a.type == 0 ? (a.i != 0) : (a.f != 0.f);
                    std::string rs = r ? "True" : "False";
                    exprstack.push(rs);
                }
                if (t == "torch.long")
                {
                    long r = a.type == 0 ? long(a.i) : long(a.f);
                    exprstack.push(std::to_string(r));
                }
            }
            else
            {
                std::string r = t + "(" + a.to_expr() + ")";
                exprstack.push(r);
            }
        }
        else if (t == "add"
                 || t == "sub"
                 || t == "max"
                 || t == "maximum"
                 || t == "min"
                 || t == "minimum"
                 || t == "mul"
                 || t == "floor_divide"
                 || t == "remainder")
        {
            typed_expr a = exprstack.top();
            exprstack.pop();
            typed_expr b = exprstack.top();
            exprstack.pop();

            if (a.is_interger_literal() && b.is_interger_literal())
            {
                const int ai = a.i;
                const int bi = b.i;

                int r = 0;
                if (t == "add")
                {
                    r = ai + bi;
                }
                if (t == "sub")
                {
                    r = ai - bi;
                }
                if (t == "max" || t == "maximum")
                {
                    r = std::max(ai, bi);
                }
                if (t == "min" || t == "minimum")
                {
                    r = std::min(ai, bi);
                }
                if (t == "mul")
                {
                    r = ai * bi;
                }
                if (t == "floor_divide")
                {
                    r = ai / bi;
                }
                if (t == "remainder")
                {
                    r = ai % bi;
                }
                exprstack.push(r);
            }
            else if (a.is_literal() && b.is_literal())
            {
                const float af = a.type == 0 ? a.i : a.f;
                const float bf = b.type == 0 ? b.i : b.f;

                float r = 0.f;
                if (t == "add")
                {
                    r = af + bf;
                }
                if (t == "sub")
                {
                    r = af - bf;
                }
                if (t == "max" || t == "maximum")
                {
                    r = std::max(af, bf);
                }
                if (t == "min" || t == "minimum")
                {
                    r = std::min(af, bf);
                }
                if (t == "mul")
                {
                    r = af * bf;
                }
                if (t == "floor_divide")
                {
                    r = (int)af / (int)bf;
                }
                if (t == "remainder")
                {
                    r = fmod(af, bf);
                    if (af * bf < 0)
                        r += bf;
                }
                exprstack.push(r);
            }
            else
            {
                std::string r = t + "(" + a.to_expr() + "," + b.to_expr() + ")";
                if (a.type == 0 && b.type == 0)
                {
                    exprstack.push(typed_expr(r, 0));
                }
                else if (a.type == 1 || b.type == 1)
                {
                    exprstack.push(typed_expr(r, 1));
                }
                else
                {
                    exprstack.push(r);
                }
            }
        }
        else if (t == "atan2"
                 || t == "div"
                 || t == "fmod"
                 || t == "pow"
                 || t == "logaddexp")
        {
            typed_expr a = exprstack.top();
            exprstack.pop();
            typed_expr b = exprstack.top();
            exprstack.pop();

            if (a.is_literal() && b.is_literal())
            {
                const float af = a.type == 0 ? a.i : a.f;
                const float bf = b.type == 0 ? b.i : b.f;

                float r = 0.f;
                if (t == "atan2")
                {
                    r = atan2(af, bf);
                }
                if (t == "div")
                {
                    r = af / bf;
                }
                if (t == "fmod")
                {
                    r = fmod(af, bf);
                }
                if (t == "pow")
                {
                    r = pow(af, bf);
                }
                if (t == "logaddexp")
                {
                    r = log(exp(af) + exp(bf));
                }
                exprstack.push(r);
            }
            else
            {
                std::string r = t + "(" + a.to_expr() + "," + b.to_expr() + ")";
                if (a.type == 1 || b.type == 1)
                {
                    exprstack.push(typed_expr(r, 1));
                }
                else
                {
                    exprstack.push(r);
                }
            }
        }
        else if (t == "and" || t == "or" || t == "xor" || t == "lshift" || t == "rshift")
        {
            typed_expr a = exprstack.top();
            exprstack.pop();
            typed_expr b = exprstack.top();
            exprstack.pop();

            if (a.is_interger_literal() && b.is_interger_literal())
            {
                const int ai = a.i;
                const int bi = b.i;

                int r = 0;
                if (t == "and")
                {
                    r = ai & bi;
                }
                if (t == "or")
                {
                    r = ai | bi;
                }
                if (t == "xor")
                {
                    r = ai ^ bi;
                }
                if (t == "lshift")
                {
                    r = ai << bi;
                }
                if (t == "rshift")
                {
                    r = ai >> bi;
                }
                exprstack.push(r);
            }
            else
            {
                std::string r = t + "(" + a.to_expr() + "," + b.to_expr() + ")";
                exprstack.push(typed_expr(r, 0)); // bitwise always produce integer
            }
        }
        else if (t == "[") // list
        {
            std::vector<typed_expr> elements;
            while (!exprstack.empty())
            {
                typed_expr a = exprstack.top();
                exprstack.pop();

                elements.push_back(a);
            }

            std::string r = "[";
            for (int j = 0; j < (int)elements.size() - 1; j++)
            {
                r += elements[j].to_expr();
                if (j + 1 != (int)elements.size())
                    r += ",";
            }
            if (!elements.empty())
            {
                r += elements[elements.size() - 1].to_expr();
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
            if (token_is_complex(t))
            {
                exprstack.push(t);
            }
            else if (token_is_interger_literal(t))
            {
                exprstack.push(std::stoi(t));
            }
            else if (token_is_literal(t))
            {
                exprstack.push(std::stof(t));
            }
            else
            {
                exprstack.push(t);
            }
        }
    }

    std::string r = exprstack.top().to_expr();
    exprstack.pop();
    while (!exprstack.empty())
    {
        r += std::string(",") + exprstack.top().to_expr();
        exprstack.pop();
    }

    // fprintf(stderr, "eval_expression return %s\n", r.c_str());

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
