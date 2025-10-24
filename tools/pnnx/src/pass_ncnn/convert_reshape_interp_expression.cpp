// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_reshape_interp_expression.h"

#include <algorithm>
#include <stack>

namespace pnnx {

namespace ncnn {

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

static bool token_is_ncnn_argument(const std::string& t)
{
    char tt = t[t.size() - 1];
    if ((tt != 'w' && tt != 'h' && tt != 'd' && tt != 'c') || t.size() < 2)
        return false;

    for (size_t i = 0; i + 1 < t.size(); i++)
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
    if (token_is_ncnn_argument(t))
        return false;

    if (token_is_complex(t))
        return true;

    std::istringstream iss(t);
    float f;
    iss >> std::noskipws >> f;
    return iss.eof() && !iss.fail();
}

// static void print_tokens(const std::vector<std::string>& tokens)
// {
//     std::string r;
//     for (auto x : tokens)
//     {
//         r += x + " ";
//     }
//     fprintf(stderr, "tokens = %s\n", r.c_str());
// }

void convert_reshape_interp_expression(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (op->type != "Tensor.reshape"
                    && op->type != "F.upsample" && op->type != "F.upsample_nearest" && op->type != "F.upsample_bilinear" && op->type != "F.interpolate")
                continue;

            if (op->inputs.size() != 2)
                continue;

            if (op->inputs[1]->producer->type != "pnnx.Expression")
                continue;

            matched = true;

            Operator* op_expr = op->inputs[1]->producer;
            const std::string& expr = op_expr->params["expr"].s;

            // fuse expression into reshape expr

            // fprintf(stderr, "convert reshape expression begin %s\n", expr.c_str());

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

            // print_tokens(tokens);

            // filter unknown tokens
            for (std::string& t : tokens)
            {
                if (t == "add") t = "+";
                if (t == "sub") t = "-";
                if (t == "mul") t = "*";
                if (t == "div") t = "/";
                if (t == "floor_divide") t = "//";
                if (t == "maximum") t = "max";
                if (t == "minimum") t = "min";
                if (t == "int") t = "trunc";

                if (t == "torch.bool" || t == "torch.float" || t == "torch.long")
                    fprintf(stderr, "shape expression got unsupported op %s\n", t.c_str());
            }

            // collect inputs and references
            std::map<Operand*, int> references;

            // begin with input blob
            int reference_index = 0;
            {
                references[op->inputs[0]] = reference_index++;
            }

            for (size_t i = 0; i < tokens.size(); i++)
            {
                std::string& t = tokens[i];

                if (t[0] != '@')
                    continue;

                int input_index = std::stoi(t.substr(1));
                Operand* r = op_expr->inputs[input_index];

                if (references.find(r) == references.end())
                {
                    references[r] = reference_index++;
                }

                t = "@" + std::to_string(references[r]);
            }

            // print_tokens(tokens);

            std::vector<Operand*> ordered_references(references.size());
            for (auto x : references)
            {
                ordered_references[x.second] = x.first;
            }

            // change nchw annotation to w,h,c / w,h,d,c with batch index dropped

            struct typed_value
            {
                int type; // 0=i 1=f
                union
                {
                    int i;
                    float f;
                };

                typed_value()
                    : type(0), i(0)
                {
                }
                typed_value(int _i)
                    : type(0), i(_i)
                {
                }
                typed_value(float _f)
                    : type(1), f(_f)
                {
                }

                int to_int()
                {
                    if (type == 0)
                        return i;

                    // trunc by default
                    return (int)f;
                }
            };

            // scan and stack
            std::stack<std::string> exprstack;
            for (int i = (int)tokens.size() - 1; i >= 0; i--)
            {
                const std::string& t = tokens[i];

                if (t == "size")
                {
                    std::string a = exprstack.top();
                    exprstack.pop();

                    // fprintf(stderr, "size %s\n", a.c_str());

                    if (exprstack.empty())
                    {
                        std::string r = std::string("size(") + a + ")";
                        exprstack.push(r);
                    }
                    else
                    {
                        std::string b = exprstack.top();
                        exprstack.pop();

                        // fprintf(stderr, "size %s %s\n", a.c_str(), b.c_str());

                        if (token_is_argument(a) && token_is_literal(b))
                        {
                            int input_index = std::stoi(a.substr(1));
                            if (ordered_references[input_index]->shape.empty())
                            {
                                std::string r = std::string("size(") + a + "," + b + ")";
                                exprstack.push(r);
                            }
                            else
                            {
                                if (input_index > 9)
                                {
                                    // ncnn can only handle at most 10 reference blobs
                                    fprintf(stderr, "expression with large reference id %d is not supported yet\n", input_index);
                                }

                                int bi = std::stoi(b);

                                const int a_batch_index = ordered_references[input_index]->params["__batch_index"].i;

                                if (bi == a_batch_index)
                                {
                                    fprintf(stderr, "reshape expression refer to batch axis %d is not supported\n", a_batch_index);
                                    std::string r = std::string("size(") + a + "," + b + ")";
                                    exprstack.push(r);
                                }
                                else
                                {
                                    int a_rank = (int)ordered_references[input_index]->shape.size();

                                    if (bi < 0)
                                        bi = a_rank + bi;

                                    if (bi > a_batch_index)
                                    {
                                        a_rank -= 1;
                                        bi -= 1;
                                    }

                                    if (a_rank == 1 && bi == 0)
                                    {
                                        exprstack.push(std::to_string(input_index) + "w");
                                    }
                                    else if (a_rank == 2 && bi == 0)
                                    {
                                        exprstack.push(std::to_string(input_index) + "h");
                                    }
                                    else if (a_rank == 2 && bi == 1)
                                    {
                                        exprstack.push(std::to_string(input_index) + "w");
                                    }
                                    else if (a_rank == 3 && bi == 0)
                                    {
                                        exprstack.push(std::to_string(input_index) + "c");
                                    }
                                    else if (a_rank == 3 && bi == 1)
                                    {
                                        exprstack.push(std::to_string(input_index) + "h");
                                    }
                                    else if (a_rank == 3 && bi == 2)
                                    {
                                        exprstack.push(std::to_string(input_index) + "w");
                                    }
                                    else if (a_rank == 4 && bi == 0)
                                    {
                                        exprstack.push(std::to_string(input_index) + "c");
                                    }
                                    else if (a_rank == 4 && bi == 1)
                                    {
                                        exprstack.push(std::to_string(input_index) + "d");
                                    }
                                    else if (a_rank == 4 && bi == 2)
                                    {
                                        exprstack.push(std::to_string(input_index) + "h");
                                    }
                                    else if (a_rank == 4 && bi == 3)
                                    {
                                        exprstack.push(std::to_string(input_index) + "w");
                                    }
                                    else
                                    {
                                        fprintf(stderr, "reshape expression refer to %d-rank dim %d is not supported\n", a_rank, bi);
                                        std::string r = std::string("size(") + a + "," + b + ")";
                                        exprstack.push(r);
                                    }
                                }
                            }
                        }
                        else
                        {
                            std::string r = std::string("size(") + a + "," + b + ")";
                            exprstack.push(r);
                        }
                    }
                }
                else if (t == "ceil"
                         || t == "floor"
                         || t == "round"
                         || t == "trunc")
                {
                    std::string a = exprstack.top();
                    exprstack.pop();

                    std::string r = t + "(" + a + ")";
                    exprstack.push(r);
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
                         || t == "neg"
                         || t == "reciprocal"
                         || t == "rsqrt"
                         || t == "sign"
                         || t == "sin"
                         || t == "sinh"
                         || t == "sqrt"
                         || t == "square"
                         || t == "tan"
                         || t == "tanh")
                {
                    std::string a = exprstack.top();
                    exprstack.pop();

                    std::string r = t + "(" + a + ")";
                    exprstack.push(r);
                }
                else if (t == "+"
                         || t == "-"
                         || t == "*"
                         || t == "/"
                         || t == "//"
                         || t == "atan2"
                         || t == "max"
                         || t == "min"
                         || t == "fmod"
                         || t == "pow"
                         || t == "remainder"
                         || t == "logaddexp")
                {
                    std::string a = exprstack.top();
                    exprstack.pop();
                    std::string b = exprstack.top();
                    exprstack.pop();

                    std::string r = t + "(" + a + "," + b + ")";
                    exprstack.push(r);
                }
                else if (t == "and" || t == "or" || t == "xor" || t == "lshift" || t == "rshift")
                {
                    std::string a = exprstack.top();
                    exprstack.pop();
                    std::string b = exprstack.top();
                    exprstack.pop();

                    std::string r = t + "(" + a + "," + b + ")";
                    exprstack.push(r);
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

                    if (op->type == "Tensor.reshape")
                    {
                        // drop output batch index
                        const int batch_index = op->outputs[0]->params["__batch_index"].i;
                        // fprintf(stderr, "batch_index = %d\n", batch_index);

                        if (batch_index != 233)
                        {
                            for (int j = batch_index; j + 1 < (int)elements.size(); j++)
                            {
                                elements[j] = elements[j + 1];
                            }
                            elements.resize(elements.size() - 1);
                        }
                    }

                    // reverse order
                    std::string r;
                    for (int j = (int)elements.size() - 1; j >= 0; j--)
                    {
                        r += elements[j];
                        if (j != 0)
                            r += ",";
                    }

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

            if (op->type == "Tensor.reshape")
            {
                fprintf(stderr, "convert reshape expression %s => %s\n", expr.c_str(), r.c_str());

                op->type = "Reshape";

                op->params.clear();
                op->params["6"] = r;
            }
            else
            {
                fprintf(stderr, "convert interp expression %s => %s\n", expr.c_str(), r.c_str());

                std::string mode = "nearest";
                bool align_corners = false;

                if (op->has_param("mode"))
                    mode = op->params.at("mode").s;

                if (op->has_param("align_corners"))
                    align_corners = op->params.at("align_corners").b;

                if (op->type == "F.upsample_nearest")
                {
                    mode = "nearest";
                }
                if (op->type == "F.upsample_bilinear")
                {
                    mode = "bilinear";
                }

                op->type = "Interp";

                op->params.clear();

                if (mode == "nearest")
                    op->params["0"] = 1;
                if (mode == "bilinear" || mode == "linear")
                    op->params["0"] = 2;
                if (mode == "bicubic")
                    op->params["0"] = 3;

                op->params["5"] = 1; // dynamic_target_size
                op->params["6"] = align_corners ? 1 : 0;
                op->params["9"] = r;
            }

            // link references to reshape
            {
                op->inputs = ordered_references;

                for (size_t i = 1; i < op->inputs.size(); i++)
                {
                    op->inputs[i]->consumers.push_back(op);
                }
            }

            // drop expression
            {
                Operand* expr_out = op_expr->outputs[0];
                expr_out->remove_consumer(op);

                if (expr_out->consumers.empty())
                {
                    for (auto& x : op_expr->inputs)
                    {
                        x->remove_consumer(op_expr);
                    }

                    graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), expr_out));
                    delete expr_out;

                    graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op_expr));
                    delete op_expr;
                }
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
