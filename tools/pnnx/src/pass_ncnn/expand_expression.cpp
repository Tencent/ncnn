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

#include "pass_ncnn.h"

#include <math.h>
#include <string.h>

#include <set>
#include <iostream>
#include <sstream>
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

static std::string expand_expression(Graph& graph, const Operator* op, int& pnnx_expr_index)
{
    std::string expr = op->params.at("expr").s;

    //     fprintf(stderr, "ncnn expand_expression %s\n", expr.c_str());

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
            // not supported
            return std::string();
        }
        else if (t == "int")
        {
            // not supported
            return std::string();
        }
        else if (t == "abs"
                 || t == "acos"
                 || t == "asin"
                 || t == "atan"
                 || t == "ceil"
                 || t == "cos"
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
                 || t == "sqrt"
                 || t == "square"
                 || t == "tan"
                 || t == "tanh"
                 || t == "trunc")
        {
            std::string a = exprstack.top();
            exprstack.pop();

            std::string r = t + "(" + (token_is_argument(a) ? op->inputs[std::stoi(a.substr(1))]->name : a) + ")";
            exprstack.push(r);

            Operator* op_unary = graph.new_operator_before("UnaryOp", t + "_" + std::to_string(pnnx_expr_index++), op);

            if (t == "abs") op_unary->params["0"] = 0;
            if (t == "acos") op_unary->params["0"] = 13;
            if (t == "asin") op_unary->params["0"] = 12;
            if (t == "atan") op_unary->params["0"] = 14;
            if (t == "ceil") op_unary->params["0"] = 3;
            if (t == "cos") op_unary->params["0"] = 10;
            if (t == "exp") op_unary->params["0"] = 7;
            if (t == "floor") op_unary->params["0"] = 2;
            if (t == "log") op_unary->params["0"] = 8;
            if (t == "log10") op_unary->params["0"] = 17;
            if (t == "neg") op_unary->params["0"] = 1;
            if (t == "reciprocal") op_unary->params["0"] = 15;
            if (t == "round") op_unary->params["0"] = 18;
            if (t == "rsqrt") op_unary->params["0"] = 6;
            if (t == "sign") fprintf(stderr, "UnaryOp sign not supported yet\n"); // TODO
            if (t == "sin") op_unary->params["0"] = 9;
            if (t == "sqrt") op_unary->params["0"] = 5;
            if (t == "square") op_unary->params["0"] = 4;
            if (t == "tan") op_unary->params["0"] = 11;
            if (t == "tanh") op_unary->params["0"] = 16;
            if (t == "trunc") op_unary->params["0"] = 19;

            Operand* op_unary_in = token_is_argument(a) ? op->inputs[std::stoi(a.substr(1))] : graph.get_operand(op->name + "_" + a);
            op_unary_in->consumers.push_back(op_unary);

            Operand* op_unary_out = graph.new_operand(op->name + "_" + r);
            op_unary_out->producer = op_unary;

            op_unary_out->shape = op_unary_in->shape;

            op_unary->inputs.push_back(op_unary_in);
            op_unary->outputs.push_back(op_unary_out);
        }
        else if (t == "add"
                 || t == "atan2"
                 || t == "div"
                 || t == "floor_divide"
                 || t == "fmod"
                 || t == "max"
                 || t == "maximum"
                 || t == "min"
                 || t == "minimum"
                 || t == "mul"
                 || t == "pow"
                 || t == "remainder"
                 || t == "sub")
        {
            std::string a = exprstack.top();
            exprstack.pop();
            std::string b = exprstack.top();
            exprstack.pop();

            std::string r = t + "(" + (token_is_argument(a) ? op->inputs[std::stoi(a.substr(1))]->name : a) + "," + (token_is_argument(b) ? op->inputs[std::stoi(b.substr(1))]->name : b) + ")";
            exprstack.push(r);

            Operator* op_binary = graph.new_operator_before("BinaryOp", t + "_" + std::to_string(pnnx_expr_index++), op);

            // default todo type mark  :[
            op_binary->params["0"] = -1;

            if (t == "add") op_binary->params["0"] = 0;
            if (t == "sub") op_binary->params["0"] = 1;
            if (t == "mul") op_binary->params["0"] = 2;
            if (t == "div") op_binary->params["0"] = 3;
            if (t == "max" || t == "maximum") op_binary->params["0"] = 4;
            if (t == "min" || t == "minimum") op_binary->params["0"] = 5;
            if (t == "floor_divide") fprintf(stderr, "BinaryOp floor_divide not supported yet\n"); // TODO
            if (t == "fmod") fprintf(stderr, "BinaryOp fmod not supported yet\n");                 // TODO
            if (t == "remainder") fprintf(stderr, "BinaryOp remainder not supported yet\n");       // TODO
            if (t == "pow") op_binary->params["0"] = 6;
            if (t == "atan2") op_binary->params["0"] = 10;

            if (token_is_literal(a))
            {
                if (t == "sub") op_binary->params["0"] = 7;
                if (t == "div") op_binary->params["0"] = 8;
                if (t == "pow") op_binary->params["0"] = 9;
                if (t == "atan2") op_binary->params["0"] = 11;

                Operand* op_binary_inb = token_is_argument(b) ? op->inputs[std::stoi(b.substr(1))] : graph.get_operand(op->name + "_" + b);
                op_binary_inb->consumers.push_back(op_binary);

                op_binary->params["1"] = 1; // with_scalar
                op_binary->params["2"] = std::stof(a);

                Operand* op_binary_out = graph.new_operand(op->name + "_" + r);
                op_binary_out->producer = op_binary;

                op_binary_out->shape = op_binary_inb->shape;

                op_binary->inputs.push_back(op_binary_inb);
                op_binary->outputs.push_back(op_binary_out);
            }
            else if (token_is_literal(b))
            {
                Operand* op_binary_ina = token_is_argument(a) ? op->inputs[std::stoi(a.substr(1))] : graph.get_operand(op->name + "_" + a);
                op_binary_ina->consumers.push_back(op_binary);

                op_binary->params["1"] = 1; // with_scalar
                op_binary->params["2"] = std::stof(b);

                if (t == "pow" && std::stof(b) == 2)
                {
                    // replace pow 2 with square
                    op_binary->type = "UnaryOp";
                    op_binary->params.clear();
                    op_binary->params["0"] = 4;
                }

                Operand* op_binary_out = graph.new_operand(op->name + "_" + r);
                op_binary_out->producer = op_binary;

                op_binary_out->shape = op_binary_ina->shape;

                op_binary->inputs.push_back(op_binary_ina);
                op_binary->outputs.push_back(op_binary_out);
            }
            else
            {
                Operand* op_binary_ina = token_is_argument(a) ? op->inputs[std::stoi(a.substr(1))] : graph.get_operand(op->name + "_" + a);
                op_binary_ina->consumers.push_back(op_binary);

                Operand* op_binary_inb = token_is_argument(b) ? op->inputs[std::stoi(b.substr(1))] : graph.get_operand(op->name + "_" + b);
                op_binary_inb->consumers.push_back(op_binary);

                Operand* op_binary_out = graph.new_operand(op->name + "_" + r);
                op_binary_out->producer = op_binary;

                // resolve out shape
                std::vector<int> out_shape;
                {
                    std::vector<int> a_shape = op_binary_ina->shape;
                    std::vector<int> b_shape = op_binary_inb->shape;
                    int outrank = (int)std::max(a_shape.size(), b_shape.size());
                    for (int k = (int)a_shape.size(); k < outrank; k++)
                    {
                        a_shape.insert(a_shape.begin(), 1);
                    }
                    for (int k = (int)b_shape.size(); k < outrank; k++)
                    {
                        b_shape.insert(b_shape.begin(), 1);
                    }
                    out_shape.resize(outrank);
                    for (int k = 0; k < outrank; k++)
                    {
                        out_shape[k] = std::max(a_shape[k], b_shape[k]);
                    }
                }
                op_binary_out->shape = out_shape;

                op_binary->inputs.push_back(op_binary_ina);
                op_binary->inputs.push_back(op_binary_inb);
                op_binary->outputs.push_back(op_binary_out);
            }
        }
        else if (t == "[") // list
        {
            // not supported
            return std::string();
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

    //     fprintf(stderr, "expand_expression return %s\n", r.c_str());

    return r;
}

void expand_expression(Graph& graph)
{
    int pnnx_expr_index = 0;

    std::set<Operator*> nonsupported_expr_ops;

    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];
            if (op->type != "pnnx.Expression")
                continue;

            if (nonsupported_expr_ops.find(op) != nonsupported_expr_ops.end())
                continue;

            matched = true;

            std::string outname = expand_expression(graph, op, pnnx_expr_index);

            if (outname.empty())
            {
                // not supported expr
                nonsupported_expr_ops.insert(op);
                break;
            }

            // link new output
            Operand* old_output_operand = op->outputs[0];
            Operand* new_output_operand = graph.get_operand(op->name + "_" + outname);

            if (!new_output_operand)
            {
                // not supported expr
                nonsupported_expr_ops.insert(op);
                break;
            }

            for (auto r : op->inputs)
            {
                r->remove_consumer(op);
            }

            for (auto& x : old_output_operand->consumers)
            {
                new_output_operand->consumers.push_back(x);

                for (size_t j = 0; j < x->inputs.size(); j++)
                {
                    if (x->inputs[j] == old_output_operand)
                    {
                        x->inputs[j] = new_output_operand;
                    }
                }
            }

            new_output_operand->type = old_output_operand->type;
            new_output_operand->shape = old_output_operand->shape;
            new_output_operand->params = old_output_operand->params;

            old_output_operand->producer = 0;
            old_output_operand->consumers.clear();

            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));
            delete op;

            graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), old_output_operand));
            delete old_output_operand;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
