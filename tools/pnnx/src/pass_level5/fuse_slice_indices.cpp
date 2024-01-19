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

#include "fuse_slice_indices.h"

#include <string.h>
#include <algorithm>
#include <stack>
// #include <vector>
#include "pass_level2.h"

namespace pnnx {

class fuse_slice_indices_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
Tensor.slice            op_0        1 1 input a dim=%dim0 end=%end0 start=%start0 step=%step0
Tensor.slice            op_1        1 1 a b dim=%dim1 end=%end1 start=%start1 step=%step1
Tensor.slice            op_2        1 1 b c dim=%dim2 end=%end2 start=%start2 step=%step2
Tensor.slice            op_3        1 1 c d dim=%dim3 end=%end3 start=%start3 step=%step3
Tensor.slice            op_4        1 1 d out dim=%dim4 end=%end4 start=%start4 step=%step4
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    const char* name_str() const
    {
        return "slice";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;
        int dim2 = captured_params.at("dim2").i;
        int dim3 = captured_params.at("dim3").i;
        int dim4 = captured_params.at("dim4").i;

        return dim0 < dim1 && dim1 < dim2 && dim2 < dim3 && dim3 < dim4;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;
        int dim2 = captured_params.at("dim2").i;
        int dim3 = captured_params.at("dim3").i;
        int dim4 = captured_params.at("dim4").i;

        int start0 = captured_params.at("start0").i;
        int start1 = captured_params.at("start1").i;
        int start2 = captured_params.at("start2").i;
        int start3 = captured_params.at("start3").i;
        int start4 = captured_params.at("start4").i;

        int end0 = captured_params.at("end0").i;
        int end1 = captured_params.at("end1").i;
        int end2 = captured_params.at("end2").i;
        int end3 = captured_params.at("end3").i;
        int end4 = captured_params.at("end4").i;

        int step0 = captured_params.at("step0").i;
        int step1 = captured_params.at("step1").i;
        int step2 = captured_params.at("step2").i;
        int step3 = captured_params.at("step3").i;
        int step4 = captured_params.at("step4").i;

        op->params["dims"] = Parameter{dim0, dim1, dim2, dim3, dim4};
        op->params["starts"] = Parameter{start0, start1, start2, start3, start4};
        op->params["ends"] = Parameter{end0, end1, end2, end3, end4};
        op->params["steps"] = Parameter{step0, step1, step2, step3, step4};
    }
};

class fuse_slice_indices_pass_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
Tensor.slice            op_0        1 1 input a dim=%dim0 end=%end0 start=%start0 step=%step0
Tensor.slice            op_1        1 1 a b dim=%dim1 end=%end1 start=%start1 step=%step1
Tensor.slice            op_2        1 1 b c dim=%dim2 end=%end2 start=%start2 step=%step2
Tensor.slice            op_3        1 1 c out dim=%dim3 end=%end3 start=%start3 step=%step3
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    const char* name_str() const
    {
        return "slice";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;
        int dim2 = captured_params.at("dim2").i;
        int dim3 = captured_params.at("dim3").i;

        return dim0 < dim1 && dim1 < dim2 && dim2 < dim3;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;
        int dim2 = captured_params.at("dim2").i;
        int dim3 = captured_params.at("dim3").i;

        int start0 = captured_params.at("start0").i;
        int start1 = captured_params.at("start1").i;
        int start2 = captured_params.at("start2").i;
        int start3 = captured_params.at("start3").i;

        int end0 = captured_params.at("end0").i;
        int end1 = captured_params.at("end1").i;
        int end2 = captured_params.at("end2").i;
        int end3 = captured_params.at("end3").i;

        int step0 = captured_params.at("step0").i;
        int step1 = captured_params.at("step1").i;
        int step2 = captured_params.at("step2").i;
        int step3 = captured_params.at("step3").i;

        op->params["dims"] = Parameter{dim0, dim1, dim2, dim3};
        op->params["starts"] = Parameter{start0, start1, start2, start3};
        op->params["ends"] = Parameter{end0, end1, end2, end3};
        op->params["steps"] = Parameter{step0, step1, step2, step3};
    }
};

class fuse_slice_indices_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
Tensor.slice            op_0        1 1 input a dim=%dim0 end=%end0 start=%start0 step=%step0
Tensor.slice            op_1        1 1 a b dim=%dim1 end=%end1 start=%start1 step=%step1
Tensor.slice            op_2        1 1 b out dim=%dim2 end=%end2 start=%start2 step=%step2
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    const char* name_str() const
    {
        return "slice";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;
        int dim2 = captured_params.at("dim2").i;

        return dim0 < dim1 && dim1 < dim2;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;
        int dim2 = captured_params.at("dim2").i;

        int start0 = captured_params.at("start0").i;
        int start1 = captured_params.at("start1").i;
        int start2 = captured_params.at("start2").i;

        int end0 = captured_params.at("end0").i;
        int end1 = captured_params.at("end1").i;
        int end2 = captured_params.at("end2").i;

        int step0 = captured_params.at("step0").i;
        int step1 = captured_params.at("step1").i;
        int step2 = captured_params.at("step2").i;

        op->params["dims"] = Parameter{dim0, dim1, dim2};
        op->params["starts"] = Parameter{start0, start1, start2};
        op->params["ends"] = Parameter{end0, end1, end2};
        op->params["steps"] = Parameter{step0, step1, step2};
    }
};

class fuse_slice_indices_pass_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
Tensor.slice            op_0        1 1 input a dim=%dim0 end=%end0 start=%start0 step=%step0
Tensor.slice            op_1        1 1 a out dim=%dim1 end=%end1 start=%start1 step=%step1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    const char* name_str() const
    {
        return "slice";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;

        return dim0 < dim1;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int dim1 = captured_params.at("dim1").i;

        int start0 = captured_params.at("start0").i;
        int start1 = captured_params.at("start1").i;

        int end0 = captured_params.at("end0").i;
        int end1 = captured_params.at("end1").i;

        int step0 = captured_params.at("step0").i;
        int step1 = captured_params.at("step1").i;

        op->params["dims"] = Parameter{dim0, dim1};
        op->params["starts"] = Parameter{start0, start1};
        op->params["ends"] = Parameter{end0, end1};
        op->params["steps"] = Parameter{step0, step1};
    }
};

class fuse_slice_indices_pass_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tensor.slice            op_0        1 1 input out dim=%dim0 end=%end0 start=%start0 step=%step0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    const char* name_str() const
    {
        return "slice";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim0 = captured_params.at("dim0").i;
        int start0 = captured_params.at("start0").i;
        int end0 = captured_params.at("end0").i;
        int step0 = captured_params.at("step0").i;

        op->params["dims"] = Parameter{dim0};
        op->params["starts"] = Parameter{start0};
        op->params["ends"] = Parameter{end0};
        op->params["steps"] = Parameter{step0};
    }
};

void fuse_slice_indices(Graph& graph)
{
    // fuse_slice_indices_pass a;
    // fuse_slice_indices_pass_1 b;
    // fuse_slice_indices_pass_2 c;
    // fuse_slice_indices_pass_3 d;
    // fuse_slice_indices_pass_4 e;
    // int opindex = 0;
    //
    // pnnx_graph_rewrite(graph, &a, opindex);
    // pnnx_graph_rewrite(graph, &b, opindex);
    // pnnx_graph_rewrite(graph, &c, opindex);
    // pnnx_graph_rewrite(graph, &d, opindex);
    // pnnx_graph_rewrite(graph, &e, opindex);
    while (1)
    {
        bool matched = false;

        // for (size_t i = 0; i < graph.ops.size(); i++)
        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (op->type != "Tensor.slice")
                continue;

            if (!op->has_param("dim"))
            {
                // fprintf(stderr, "dynamic dim in slice chain is not supported\n");
                continue;
            }

            bool static_starts = true;
            bool static_ends = true;
            bool static_steps = true;

            if (!op->has_param("start")) static_starts = false;
            if (!op->has_param("end")) static_ends = false;
            if (!op->has_param("step")) static_steps = false;

            int descent_dim_current = op->params.at("dim").i;

            // collect slice op chain
            std::stack<Operator*> slice_ops;
            slice_ops.push(op);
            const Operand* in0 = op->inputs[0];
            while (in0->producer->type == "Tensor.slice")
            {
                Operator* sop = in0->producer;
                if (in0->consumers.size() != 1)
                {
                    // not single chain
                    break;
                }

                if (!sop->has_param("dim"))
                {
                    // fprintf(stderr, "dynamic dim in slice chain is not supported\n");

                    for (auto x : sop->params)
                    {
                        fprintf(stderr, "%s\n", x.first.c_str());
                    }

                    break;
                }

                if (!sop->has_param("start")) static_starts = false;
                if (!sop->has_param("end")) static_ends = false;
                if (!sop->has_param("step")) static_steps = false;

                int dim = sop->params.at("dim").i;
                if (descent_dim_current <= dim)
                {
                    break;
                }

                descent_dim_current = dim;

                slice_ops.push(sop);
                in0 = sop->inputs[0];
            }

            if (slice_ops.empty())
            {
                // single orphaned slice
                continue;
            }

            matched = true;

            // construct one-step slice
            std::vector<int> new_dims;
            std::vector<int> new_starts;
            std::vector<int> new_ends;
            std::vector<int> new_steps;
            Operator* op_starts = 0;
            Operator* op_ends = 0;
            Operator* op_steps = 0;
            if (!static_starts) op_starts = graph.new_operator_before("pnnx.Expression", op->name + "_ncnnstarts", op);
            if (!static_ends) op_ends = graph.new_operator_before("pnnx.Expression", op->name + "_ncnnends", op);
            if (!static_steps) op_steps = graph.new_operator_before("pnnx.Expression", op->name + "_ncnnsteps", op);

            // std::string starts_expr;
            // std::string ends_expr;
            // std::string steps_expr;

            std::vector<std::string> starts_expr;
            std::vector<std::string> ends_expr;
            std::vector<std::string> steps_expr;

            Operator* top_sop = slice_ops.top();
            slice_ops.pop();

            new_dims.push_back(top_sop->params.at("dim").i);

            if (static_starts)
            {
                new_starts.push_back(top_sop->params.at("start").type == 0 ? 0 : top_sop->params.at("start").i);
            }
            else if (top_sop->has_param("start"))
            {
                char tmp[32];
                if (top_sop->params.at("start").type == 0)
                {
                    sprintf(tmp, "0");
                }
                else
                {
                    sprintf(tmp, "%d", top_sop->params.at("start").i);
                }
                // starts_expr += tmp;
                starts_expr.push_back(tmp);
            }
            else
            {
                char tmp[32];
                sprintf(tmp, "@%d", (int)op_starts->inputs.size());
                // starts_expr += tmp;
                starts_expr.push_back(tmp);
                Operand* start = top_sop->named_input("start");
                op_starts->inputs.push_back(start);
                start->remove_consumer(top_sop);
                start->consumers.push_back(op_starts);
            }

            if (static_ends)
            {
                new_ends.push_back(top_sop->params.at("end").type == 0 ? INT_MAX : top_sop->params.at("end").i);
            }
            else if (top_sop->has_param("end"))
            {
                char tmp[32];
                if (top_sop->params.at("end").type == 0)
                {
                    sprintf(tmp, "%d", INT_MAX);
                }
                else
                {
                    sprintf(tmp, "%d", top_sop->params.at("end").i);
                }
                // ends_expr += tmp;
                ends_expr.push_back(tmp);
            }
            else
            {
                char tmp[32];
                sprintf(tmp, "@%d", (int)op_ends->inputs.size());
                // ends_expr += tmp;
                ends_expr.push_back(tmp);
                Operand* end = top_sop->named_input("end");
                op_ends->inputs.push_back(end);
                end->remove_consumer(top_sop);
                end->consumers.push_back(op_ends);
            }

            if (static_steps)
            {
                new_steps.push_back(top_sop->params.at("step").type == 0 ? 1 : top_sop->params.at("step").i);
            }
            else if (top_sop->has_param("step"))
            {
                char tmp[32];
                if (top_sop->params.at("step").type == 0)
                {
                    sprintf(tmp, "1");
                }
                else
                {
                    sprintf(tmp, "%d", top_sop->params.at("step").i);
                }
                // steps_expr += tmp;
                steps_expr.push_back(tmp);
            }
            else
            {
                char tmp[32];
                sprintf(tmp, "@%d", (int)op_steps->inputs.size());
                // steps_expr += tmp;
                steps_expr.push_back(tmp);
                Operand* step = top_sop->named_input("step");
                op_steps->inputs.push_back(step);
                step->remove_consumer(top_sop);
                step->consumers.push_back(op_steps);
            }

            while (!slice_ops.empty())
            {
                Operator* sop = slice_ops.top();
                slice_ops.pop();

                new_dims.push_back(sop->params.at("dim").i);

                if (static_starts)
                {
                    new_starts.push_back(sop->params.at("start").type == 0 ? 0 : sop->params.at("start").i);
                }
                else if (sop->has_param("start"))
                {
                    char tmp[32];
                    if (sop->params.at("start").type == 0)
                    {
                        sprintf(tmp, "0");
                    }
                    else
                    {
                        sprintf(tmp, "%d", sop->params.at("start").i);
                    }
                    // starts_expr += tmp;
                    starts_expr.push_back(tmp);
                }
                else
                {
                    char tmp[32];
                    sprintf(tmp, "@%d", (int)op_starts->inputs.size());
                    // starts_expr += tmp;
                    starts_expr.push_back(tmp);
                    Operand* start = sop->named_input("start");
                    op_starts->inputs.push_back(start);
                    start->remove_consumer(sop);
                    start->consumers.push_back(op_starts);
                }

                if (static_ends)
                {
                    new_ends.push_back(sop->params.at("end").type == 0 ? INT_MAX : sop->params.at("end").i);
                }
                else if (sop->has_param("end"))
                {
                    char tmp[32];
                    if (sop->params.at("end").type == 0)
                    {
                        sprintf(tmp, "%d", INT_MAX);
                    }
                    else
                    {
                        sprintf(tmp, "%d", sop->params.at("end").i);
                    }
                    // ends_expr += tmp;
                    ends_expr.push_back(tmp);
                }
                else
                {
                    char tmp[32];
                    sprintf(tmp, "@%d", (int)op_ends->inputs.size());
                    // ends_expr += tmp;
                    ends_expr.push_back(tmp);
                    Operand* end = sop->named_input("end");
                    op_ends->inputs.push_back(end);
                    end->remove_consumer(sop);
                    end->consumers.push_back(op_ends);
                }

                if (static_steps)
                {
                    new_steps.push_back(sop->params.at("step").type == 0 ? 1 : sop->params.at("step").i);
                }
                else if (sop->has_param("step"))
                {
                    char tmp[32];
                    if (sop->params.at("step").type == 0)
                    {
                        sprintf(tmp, "1");
                    }
                    else
                    {
                        sprintf(tmp, "%d", sop->params.at("step").i);
                    }
                    // steps_expr += tmp;
                    steps_expr.push_back(tmp);
                }
                else
                {
                    char tmp[32];
                    sprintf(tmp, "@%d", (int)op_steps->inputs.size());
                    // steps_expr += tmp;
                    steps_expr.push_back(tmp);
                    Operand* step = sop->named_input("step");
                    op_steps->inputs.push_back(step);
                    step->remove_consumer(sop);
                    step->consumers.push_back(op_steps);
                }

                if (!slice_ops.empty())
                {
                    // drop sop and sop output
                    Operand* sop_out = sop->outputs[0];

                    graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), sop_out));

                    delete sop_out;

                    graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), sop));

                    delete sop;
                }
            }

            op->params.clear();
            op->params["dims"] = new_dims;

            op->inputs.clear();
            op->inputnames.clear();

            op->inputs.push_back(top_sop->inputs[0]);
            op->inputnames.push_back("input");

            top_sop->inputs[0]->remove_consumer(top_sop);
            top_sop->inputs[0]->consumers.push_back(op);

            {
                // drop top_sop and top_sop output
                Operand* top_sop_out = top_sop->outputs[0];

                graph.operands.erase(std::find(graph.operands.begin(), graph.operands.end(), top_sop_out));

                delete top_sop_out;

                graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), top_sop));

                delete top_sop;
            }

            if (static_starts)
            {
                op->params["starts"] = new_starts;
            }
            else
            {
                // starts_expr += "]";
                op_starts->params["expr"] = starts_expr;

                Operand* starts_out = graph.new_operand(op->name + "_ncnnstarts_out");
                starts_out->producer = op_starts;
                op_starts->outputs.push_back(starts_out);
                starts_out->consumers.push_back(op);
                op->inputs.push_back(starts_out);
                op->inputnames.push_back("starts");
            }

            if (static_ends)
            {
                op->params["ends"] = new_ends;
            }
            else
            {
                // ends_expr += "]";
                op_ends->params["expr"] = ends_expr;

                Operand* ends_out = graph.new_operand(op->name + "_ncnnends_out");
                ends_out->producer = op_ends;
                op_ends->outputs.push_back(ends_out);
                ends_out->consumers.push_back(op);
                op->inputs.push_back(ends_out);
                op->inputnames.push_back("ends");
            }

            if (static_steps)
            {
                op->params["steps"] = new_steps;
            }
            else
            {
                // steps_expr += "]";
                op_steps->params["expr"] = steps_expr;

                Operand* steps_out = graph.new_operand(op->name + "_ncnnsteps_out");
                steps_out->producer = op_steps;
                op_steps->outputs.push_back(steps_out);
                steps_out->consumers.push_back(op);
                op->inputs.push_back(steps_out);
                op->inputnames.push_back("steps");
            }

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace pnnx
