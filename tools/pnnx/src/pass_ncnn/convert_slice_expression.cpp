// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_slice_expression.h"

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

static std::vector<std::string> split_into_tokens(const std::string& expr)
{
    std::vector<std::string> tokens;

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

    return tokens;
}

static std::string transform_nchw_annotation_and_drop_batch_index(const std::vector<std::string>& tokens, const std::vector<Operand*>& ordered_references, int output_batch_index)
{
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
                            fprintf(stderr, "slice expression refer to batch axis %d is not supported\n", a_batch_index);
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
                                fprintf(stderr, "slice expression refer to %d-rank dim %d is not supported\n", a_rank, bi);
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

            // drop output batch index
            if (output_batch_index != 233)
            {
                for (int j = output_batch_index; j + 1 < (int)elements.size(); j++)
                {
                    elements[j] = elements[j + 1];
                }
                elements.resize(elements.size() - 1);
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

    return r;
}

static void drop_expression_op(Graph& graph, const Operator* op_this, Operator* op_expr)
{
    if (!op_expr)
        return;

    Operand* expr_out = op_expr->outputs[0];
    expr_out->remove_consumer(op_this);

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

void convert_slice_expression_single_axis_ranged(Graph& graph)
{
    int op_index = 0;

    // single-axis ranged slice
    //  pnnx.Expression
    //  pnnx.Expression
    //  Tensor.slice

    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (op->type != "Tensor.slice")
                continue;

            if (op->inputs.size() == 1)
                continue;

            if (!op->has_param("dim"))
                continue;

            const int dim = op->params.at("dim").i;

            int start = 0;
            int end = 0;
            int step = 0;
            int select = 0;
            Operator* op_start = 0;
            Operator* op_end = 0;
            Operator* op_step = 0;
            Operator* op_select = 0;

            if (op->has_param("start"))
            {
                start = op->params.at("start").i;
            }
            else if (op->has_input("start"))
            {
                op_start = op->named_input("start")->producer;
                if (op_start->type != "pnnx.Expression")
                    continue;
            }
            else
            {
                continue;
            }

            if (op->has_param("end"))
            {
                end = op->params.at("end").i;
            }
            else if (op->has_input("end"))
            {
                op_end = op->named_input("end")->producer;
                if (op_end->type != "pnnx.Expression")
                    continue;
            }
            else
            {
                continue;
            }

            if (op->has_param("step"))
            {
                step = op->params.at("step").i;
            }
            else if (op->has_input("step"))
            {
                op_step = op->named_input("step")->producer;
                if (op_step->type != "pnnx.Expression")
                    continue;
            }
            else
            {
                continue;
            }

            if (op->has_param("select"))
            {
                select = op->params.at("select").i;
            }
            else if (op->has_input("select"))
            {
                op_select = op->named_input("select")->producer;
                if (op_select->type != "pnnx.Expression")
                    continue;
            }

            fprintf(stderr, "----------------------------convert_slice_expression_single_axis_ranged\n");

            matched = true;

            std::string start_expr = op_start ? op_start->params["expr"].s : std::to_string(start);
            std::string end_expr = op_end ? op_end->params["expr"].s : std::to_string(end);
            std::string step_expr = op_step ? op_step->params["expr"].s : std::to_string(step);
            std::string select_expr = op_select ? op_select->params["expr"].s : std::to_string(select);

            bool has_select = !op_step && step == 0;
            if (has_select)
            {
                // simulate select as slice
                start_expr = select_expr;
                end_expr = std::string("add(") + select_expr + ",1)";
                step_expr = "1";
            }

            // split into tokens
            std::vector<std::string> start_tokens = split_into_tokens(start_expr);
            std::vector<std::string> end_tokens = split_into_tokens(end_expr);
            std::vector<std::string> step_tokens = split_into_tokens(step_expr);

            // collect inputs and references
            std::map<Operand*, int> references;

            // begin with input blob
            int reference_index = 0;
            {
                references[op->inputs[0]] = reference_index++;
            }

            for (size_t i = 0; i < start_tokens.size(); i++)
            {
                std::string& t = start_tokens[i];

                if (t[0] != '@')
                    continue;

                int input_index = std::stoi(t.substr(1));
                Operand* r = op_start->inputs[input_index];

                if (references.find(r) == references.end())
                {
                    references[r] = reference_index++;
                }

                t = "@" + std::to_string(references[r]);
            }
            for (size_t i = 0; i < end_tokens.size(); i++)
            {
                std::string& t = end_tokens[i];

                if (t[0] != '@')
                    continue;

                int input_index = std::stoi(t.substr(1));
                Operand* r = op_end->inputs[input_index];

                if (references.find(r) == references.end())
                {
                    references[r] = reference_index++;
                }

                t = "@" + std::to_string(references[r]);
            }
            for (size_t i = 0; i < step_tokens.size(); i++)
            {
                std::string& t = step_tokens[i];

                if (t[0] != '@')
                    continue;

                int input_index = std::stoi(t.substr(1));
                Operand* r = op_step->inputs[input_index];

                if (references.find(r) == references.end())
                {
                    references[r] = reference_index++;
                }

                // reuse the same reference
                t = "@" + std::to_string(references[r]);
            }

            std::vector<Operand*> ordered_references(references.size());
            for (auto x : references)
            {
                ordered_references[x.second] = x.first;
            }

            // change nchw annotation to w,h,c / w,h,d,c with batch index dropped

            const int batch_index = op->outputs[0]->params["__batch_index"].i;

            std::string starts_expr = transform_nchw_annotation_and_drop_batch_index(start_tokens, ordered_references, batch_index);
            std::string ends_expr = transform_nchw_annotation_and_drop_batch_index(end_tokens, ordered_references, batch_index);
            std::string steps_expr = transform_nchw_annotation_and_drop_batch_index(step_tokens, ordered_references, batch_index);

            if (steps_expr != std::to_string(1))
            {
                fprintf(stderr, "slice with step expression %s is not supported\n", steps_expr.c_str());
            }

            op->type = "Crop";
            op->name = std::string("slice1_") + std::to_string(op_index++);

            op->params.clear();
            op->params["19"] = starts_expr;
            op->params["20"] = ends_expr;
            op->params["21"] = std::to_string(dim > batch_index ? dim - 1 : dim);

            // link references to reshape
            {
                op->inputs = ordered_references;

                for (size_t i = 1; i < op->inputs.size(); i++)
                {
                    op->inputs[i]->consumers.push_back(op);
                }
            }

            // drop expression
            drop_expression_op(graph, op, op_start);
            drop_expression_op(graph, op, op_end);
            drop_expression_op(graph, op, op_step);
            drop_expression_op(graph, op, op_select);

            // reshape for output, squeezing the slice dim
            if (has_select)
            {
                Operand* out = op->outputs[0];

                Operator* reshape = graph.new_operator_after("Tensor.reshape", op->name + "_ncnnreshape", op);

                Operand* reshape_in = graph.new_operand(op->name + "_ncnnreshape_in");

                reshape_in->params["__batch_index"] = batch_index;

                reshape->inputs.push_back(reshape_in);
                reshape->outputs.push_back(out);

                op->outputs[0] = reshape_in;

                out->producer = reshape;
                reshape_in->producer = op;
                reshape_in->consumers.push_back(reshape);

                reshape->params["shape"] = out->shape;
            }

            break;
        }

        if (!matched)
            break;
    }
}

void convert_slice_expression_single_axis_select(Graph& graph)
{
    int op_index = 0;

    // single-axis one slice
    //  pnnx.Expression
    //  Tensor.select

    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (op->type != "Tensor.select")
                continue;

            if (op->inputs.size() == 1)
                continue;

            if (!op->has_param("dim"))
                continue;

            const int dim = op->params.at("dim").i;

            int start = 0;
            Operator* op_start = 0;

            if (op->has_param("index"))
            {
                start = op->params.at("index").i;
            }
            else if (op->has_input("index"))
            {
                op_start = op->named_input("index")->producer;
                if (op_start->type != "pnnx.Expression")
                    continue;
            }
            else
            {
                continue;
            }

            fprintf(stderr, "----------------------------convert_slice_expression_single_axis_select\n");

            matched = true;

            std::string start_expr = op_start ? op_start->params["expr"].s : std::to_string(start);

            // split into tokens
            std::vector<std::string> start_tokens = split_into_tokens(start_expr);

            // collect inputs and references
            std::map<Operand*, int> references;

            // begin with input blob
            int reference_index = 0;
            {
                references[op->inputs[0]] = reference_index++;
            }

            for (size_t i = 0; i < start_tokens.size(); i++)
            {
                std::string& t = start_tokens[i];

                if (t[0] != '@')
                    continue;

                int input_index = std::stoi(t.substr(1));
                Operand* r = op_start->inputs[input_index];

                if (references.find(r) == references.end())
                {
                    references[r] = reference_index++;
                }

                t = "@" + std::to_string(references[r]);
            }

            std::vector<Operand*> ordered_references(references.size());
            for (auto x : references)
            {
                ordered_references[x.second] = x.first;
            }

            // change nchw annotation to w,h,c / w,h,d,c with batch index dropped

            const int batch_index = op->outputs[0]->params["__batch_index"].i;

            std::string starts_expr = transform_nchw_annotation_and_drop_batch_index(start_tokens, ordered_references, batch_index);

            op->type = "Crop";
            op->name = std::string("slice2_") + std::to_string(op_index++);

            op->params.clear();
            op->params["19"] = starts_expr;
            op->params["20"] = std::string("+(") + starts_expr + ",1)";
            op->params["21"] = std::to_string(dim > batch_index ? dim - 1 : dim);

            // link references to reshape
            {
                op->inputs = ordered_references;

                for (size_t i = 1; i < op->inputs.size(); i++)
                {
                    op->inputs[i]->consumers.push_back(op);
                }
            }

            // drop expression
            drop_expression_op(graph, op, op_start);

            // squeezing the select dim
            {
                Operand* out = op->outputs[0];

                Operator* squeeze = graph.new_operator_after("torch.squeeze", op->name + "_ncnnsqueeze", op);

                Operand* squeeze_in = graph.new_operand(op->name + "_ncnnsqueeze_in");

                squeeze->inputs.push_back(squeeze_in);
                squeeze->outputs.push_back(out);

                op->outputs[0] = squeeze_in;

                out->producer = squeeze;
                squeeze_in->producer = op;
                squeeze_in->consumers.push_back(squeeze);

                squeeze->params["dim"] = dim;

                squeeze_in->params["__batch_index"] = batch_index;
            }

            break;
        }

        if (!matched)
            break;
    }
}

static std::vector<std::string> split_into_raw_tokens(const std::string& expr)
{
    std::vector<std::string> tokens;

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

            std::string tt;
            tt += ch;
            tokens.push_back(tt);
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

    return tokens;
}

static void make_slice_indexes_expression(Graph& graph)
{
    // pnnx.Expression          pnnx_expr_24    2 1 0 1 13 expr=sub(floor_divide(size(@0,0),64),floor_divide(size(@1,1),128)) #0=(?)f32 #1=(?,?)f32
    // pnnx.Expression          pnnx_expr_18    1 1 12 14 expr=sub(size(@0,3),3) #12=(1,15,?,?)f32
    // pnnx.Expression          pnnx_expr_13    1 1 12 15 expr=floor_divide(neg(size(@0,2)),7) #12=(1,15,?,?)f32
    // pnnx.Expression          pnnx_expr_8     1 1 12 16 expr=floor_divide(size(@0,2),3) #12=(1,15,?,?)f32
    // pnnx.SliceIndexes        ncnnstarts      1 1 14 17 indexes=(0,@0,0)
    // pnnx.SliceIndexes        ncnnends        1 1 15 18 indexes=(0,@0,0)
    // pnnx.SliceIndexes        ncnnselects     2 1 13 16 19 indexes=(@0,2147483647,@1)

    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (op->type != "pnnx.SliceIndexes")
                continue;

            bool slice_index_expr = true;
            for (size_t i = 0; i < op->inputs.size(); i++)
            {
                if (op->inputs[i]->producer->type != "pnnx.Expression")
                {
                    slice_index_expr = false;
                    break;
                }
            }
            if (!slice_index_expr)
                continue;

            matched = true;

            const std::vector<std::string>& indexes = op->params["indexes"].as;

            std::map<Operand*, int> references;
            std::vector<Operator*> op_expr_si;

            int reference_index = 0;

            std::vector<std::string> new_indexes;

            for (size_t i = 0; i < indexes.size(); i++)
            {
                std::string si_expr = indexes[i];
                Operator* op_si = 0;
                if (si_expr[0] == '@')
                {
                    int si = std::stoi(si_expr.substr(1));
                    op_si = op->inputs[si]->producer;
                    si_expr = op_si->params.at("expr").s;

                    op_expr_si.push_back(op_si);
                }

                // split into tokens
                std::vector<std::string> si_tokens = split_into_raw_tokens(si_expr);

                // collect inputs and references
                for (size_t j = 0; j < si_tokens.size(); j++)
                {
                    std::string& t = si_tokens[j];

                    if (t[0] != '@')
                        continue;

                    int input_index = std::stoi(t.substr(1));
                    Operand* r = op_si->inputs[input_index];

                    if (references.find(r) == references.end())
                    {
                        references[r] = reference_index++;
                    }

                    t = "@" + std::to_string(references[r]);
                }

                std::string expr;
                for (const std::string& t : si_tokens)
                {
                    expr += t;
                }

                new_indexes.push_back(expr);
            }

            std::vector<Operand*> ordered_references(references.size());
            for (auto x : references)
            {
                ordered_references[x.second] = x.first;
            }

            op->params["indexes"] = new_indexes;

            // for (auto x : new_indexes)
            // {
            //     fprintf(stderr, "%s  ", x.c_str());
            // }
            // fprintf(stderr, "\n");

            // link references to slice indexes expression
            {
                op->inputs = ordered_references;

                for (size_t i = 1; i < op->inputs.size(); i++)
                {
                    op->inputs[i]->consumers.push_back(op);
                }
            }

            // drop expression
            for (auto op_si : op_expr_si)
            {
                drop_expression_op(graph, op, op_si);
            }

            break;
        }

        if (!matched)
            break;
    }
}

void convert_slice_expression_multi_axis_ranged(Graph& graph)
{
    int op_index = 0;

    // multi-axis ranged slice
    //  pnnx.SliceIndexes
    //  pnnx.SliceIndexes
    //  pnnx.SliceIndexes
    //  Tensor.slice

    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (op->type != "Tensor.slice")
                continue;

            if (op->inputs.size() == 1)
                continue;

            if (!op->has_param("dims"))
                continue;

            const std::vector<int>& dims = op->params.at("dims").ai;

            std::vector<int> starts;
            std::vector<int> ends;
            std::vector<int> steps;
            std::vector<int> selects;
            Operator* op_starts = 0;
            Operator* op_ends = 0;
            Operator* op_steps = 0;
            Operator* op_selects = 0;

            if (op->has_param("starts"))
            {
                starts = op->params.at("starts").ai;
            }
            else if (op->has_input("starts"))
            {
                op_starts = op->named_input("starts")->producer;
                if (op_starts->type != "pnnx.SliceIndexes")
                    continue;
            }
            else
            {
                continue;
            }

            if (op->has_param("ends"))
            {
                ends = op->params.at("ends").ai;
            }
            else if (op->has_input("ends"))
            {
                op_ends = op->named_input("ends")->producer;
                if (op_ends->type != "pnnx.SliceIndexes")
                    continue;
            }
            else
            {
                continue;
            }

            if (op->has_param("steps"))
            {
                steps = op->params.at("steps").ai;
            }
            else if (op->has_input("steps"))
            {
                op_steps = op->named_input("steps")->producer;
                if (op_steps->type != "pnnx.SliceIndexes")
                    continue;
            }
            else
            {
                continue;
            }

            if (op->has_param("selects"))
            {
                selects = op->params.at("selects").ai;
            }
            else if (op->has_input("selects"))
            {
                op_selects = op->named_input("selects")->producer;
                if (op_selects->type != "pnnx.SliceIndexes")
                    continue;
            }
            else
            {
                continue;
            }

            fprintf(stderr, "----------------------------convert_slice_expression_multi_axis_ranged\n");

            matched = true;

            std::vector<std::string> starts_expr;
            std::vector<std::string> ends_expr;
            std::vector<std::string> steps_expr;
            std::vector<std::string> selects_expr;
            if (op_starts)
            {
                starts_expr = op_starts->params["indexes"].as;
            }
            else
            {
                for (int i : starts)
                {
                    starts_expr.push_back(std::to_string(i));
                }
            }
            if (op_ends)
            {
                ends_expr = op_ends->params["indexes"].as;
            }
            else
            {
                for (int i : ends)
                {
                    ends_expr.push_back(std::to_string(i));
                }
            }
            if (op_steps)
            {
                steps_expr = op_steps->params["indexes"].as;
            }
            else
            {
                for (int i : steps)
                {
                    steps_expr.push_back(std::to_string(i));
                }
            }
            if (op_selects)
            {
                selects_expr = op_selects->params["indexes"].as;
            }
            else
            {
                for (int i : selects)
                {
                    selects_expr.push_back(std::to_string(i));
                }
            }

            // collect inputs and references
            std::map<Operand*, int> references;

            // begin with input blob
            int reference_index = 0;
            {
                references[op->inputs[0]] = reference_index++;
            }

            bool has_select = false;

            const size_t dims_count = dims.size();

            for (size_t i = 0; i < dims_count; i++)
            {
                const std::string& start_expr = starts_expr[i];
                const std::string& end_expr = ends_expr[i];
                const std::string& step_expr = steps_expr[i];
                const std::string& select_expr = selects_expr[i];

                // split into tokens
                std::vector<std::string> start_tokens = split_into_raw_tokens(start_expr);
                std::vector<std::string> end_tokens = split_into_raw_tokens(end_expr);
                std::vector<std::string> step_tokens = split_into_raw_tokens(step_expr);
                std::vector<std::string> select_tokens = split_into_raw_tokens(select_expr);

                bool is_select = true;
                if (select_tokens.size() == 1 && select_tokens[0] == std::to_string(INT_MAX))
                {
                    is_select = false;
                }

                if (is_select)
                {
                    has_select = true;

                    // simulate select as slice
                    for (size_t j = 0; j < select_tokens.size(); j++)
                    {
                        std::string& t = select_tokens[j];

                        if (t[0] != '@')
                            continue;

                        int input_index = std::stoi(t.substr(1));
                        Operand* r = op_selects->inputs[input_index];

                        if (references.find(r) == references.end())
                        {
                            references[r] = reference_index++;
                        }

                        t = "@" + std::to_string(references[r]);
                    }

                    start_tokens = select_tokens;
                    end_tokens.clear();
                    step_tokens.clear();
                    end_tokens.push_back("add");
                    end_tokens.push_back("(");
                    for (auto t : select_tokens)
                    {
                        end_tokens.push_back(t);
                        step_tokens.push_back("1");
                    }
                    end_tokens.push_back(",");
                    end_tokens.push_back("1");
                    end_tokens.push_back(")");
                }
                else
                {
                    for (size_t j = 0; j < start_tokens.size(); j++)
                    {
                        std::string& t = start_tokens[j];

                        if (t[0] != '@')
                            continue;

                        int input_index = std::stoi(t.substr(1));
                        Operand* r = op_starts->inputs[input_index];

                        if (references.find(r) == references.end())
                        {
                            references[r] = reference_index++;
                        }

                        t = "@" + std::to_string(references[r]);
                    }
                    for (size_t j = 0; j < end_tokens.size(); j++)
                    {
                        std::string& t = end_tokens[j];

                        if (t[0] != '@')
                            continue;

                        int input_index = std::stoi(t.substr(1));
                        Operand* r = op_ends->inputs[input_index];

                        if (references.find(r) == references.end())
                        {
                            references[r] = reference_index++;
                        }

                        t = "@" + std::to_string(references[r]);
                    }
                    for (size_t j = 0; j < step_tokens.size(); j++)
                    {
                        std::string& t = step_tokens[j];

                        if (t[0] != '@')
                            continue;

                        int input_index = std::stoi(t.substr(1));
                        Operand* r = op_steps->inputs[input_index];

                        if (references.find(r) == references.end())
                        {
                            references[r] = reference_index++;
                        }

                        // reuse the same reference
                        t = "@" + std::to_string(references[r]);
                    }
                }

                std::string new_start_expr;
                std::string new_end_expr;
                std::string new_step_expr;
                for (const std::string& t : start_tokens)
                {
                    new_start_expr += t;
                }
                for (const std::string& t : end_tokens)
                {
                    new_end_expr += t;
                }
                for (const std::string& t : step_tokens)
                {
                    new_step_expr += t;
                }
                starts_expr[i] = new_start_expr;
                ends_expr[i] = new_end_expr;
                steps_expr[i] = new_step_expr;
            }

            std::vector<Operand*> ordered_references(references.size());
            for (auto x : references)
            {
                ordered_references[x.second] = x.first;
            }

            // change nchw annotation to w,h,c / w,h,d,c with batch index dropped

            const int batch_index = op->outputs[0]->params["__batch_index"].i;

            std::string new_starts_expr;
            std::string new_ends_expr;
            std::string new_steps_expr;
            std::string new_dims_expr;

            for (size_t i = 0; i < dims_count; i++)
            {
                const std::string& start_expr = starts_expr[i];
                const std::string& end_expr = ends_expr[i];
                const std::string& step_expr = steps_expr[i];

                // split into tokens
                std::vector<std::string> start_tokens = split_into_tokens(start_expr);
                std::vector<std::string> end_tokens = split_into_tokens(end_expr);
                std::vector<std::string> step_tokens = split_into_tokens(step_expr);

                std::string new_start_expr = transform_nchw_annotation_and_drop_batch_index(start_tokens, ordered_references, batch_index);
                std::string new_end_expr = transform_nchw_annotation_and_drop_batch_index(end_tokens, ordered_references, batch_index);
                std::string new_step_expr = transform_nchw_annotation_and_drop_batch_index(step_tokens, ordered_references, batch_index);

                if (new_step_expr != std::to_string(1))
                {
                    fprintf(stderr, "slice with step expression %s is not supported\n", new_step_expr.c_str());
                }

                new_starts_expr += new_start_expr;
                new_ends_expr += new_end_expr;
                new_steps_expr += new_step_expr;
                new_dims_expr += std::to_string(dims[i] > batch_index ? dims[i] - 1 : dims[i]);

                if (i + 1 != dims_count)
                {
                    new_starts_expr += ",";
                    new_ends_expr += ",";
                    new_steps_expr += ",";
                    new_dims_expr += ",";
                }
            }

            op->type = "Crop";
            op->name = std::string("slice3_") + std::to_string(op_index++);

            op->params.clear();
            op->params["19"] = new_starts_expr;
            op->params["20"] = new_ends_expr;
            op->params["21"] = new_dims_expr;

            // link references to reshape
            {
                op->inputs = ordered_references;

                for (size_t i = 1; i < op->inputs.size(); i++)
                {
                    op->inputs[i]->consumers.push_back(op);
                }
            }

            // drop expression
            drop_expression_op(graph, op, op_starts);
            drop_expression_op(graph, op, op_ends);
            drop_expression_op(graph, op, op_steps);
            drop_expression_op(graph, op, op_selects);

            // reshape for output, squeezing the slice dim
            if (has_select)
            {
                Operand* out = op->outputs[0];

                Operator* reshape = graph.new_operator_after("Tensor.reshape", op->name + "_ncnnreshape", op);

                Operand* reshape_in = graph.new_operand(op->name + "_ncnnreshape_in");

                reshape_in->params["__batch_index"] = batch_index;

                reshape->inputs.push_back(reshape_in);
                reshape->outputs.push_back(out);

                op->outputs[0] = reshape_in;

                out->producer = reshape;
                reshape_in->producer = op;
                reshape_in->consumers.push_back(reshape);

                reshape->params["shape"] = out->shape;
            }

            break;
        }

        if (!matched)
            break;
    }
}

void convert_slice_expression(Graph& graph)
{
    convert_slice_expression_single_axis_ranged(graph);

    convert_slice_expression_single_axis_select(graph);

    make_slice_indexes_expression(graph);

    convert_slice_expression_multi_axis_ranged(graph);
}

} // namespace ncnn

} // namespace pnnx
