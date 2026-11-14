// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_log_softmax : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.log_softmax           op          1 1 input out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
F.softmax               softmax     1 1 input softmax
UnaryOp                 log         1 1 softmax out 0=8
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F_log_softmax";
    }

    const char* name_str() const
    {
        return "f_logsoftmax";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        ops.at("softmax")->params["dim"] = captured_params.at("dim");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_log_softmax, 19)

class F_log_softmax_stable : public F_log_softmax
{
public:
    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operand* input = matched_operators.at("op")->inputs[0];
        const int input_rank = input->shape.size();
        if (input->type != 1 || input_rank == 0)
            return false;

        for (int size : input->shape)
        {
            if (size <= 0)
                return false;
        }

        const auto batch_index = input->params.find("__batch_index");
        const auto batch_axis = input->params.find("__ncnn_batch_axis");
        if (batch_index == input->params.end() || batch_axis == input->params.end()
                || batch_index->second.type != 2 || batch_axis->second.type != 2)
            return false;

        if (batch_index->second.i != 233 && (batch_index->second.i < 0 || batch_index->second.i >= input_rank))
            return false;

        const int ncnn_batch_axis = batch_axis->second.i;
        if (ncnn_batch_axis != 233 && (ncnn_batch_axis < 0 || ncnn_batch_axis >= input_rank))
            return false;

        const int native_rank = input_rank - (ncnn_batch_axis != 233 ? 1 : 0);
        if (native_rank < 1 || native_rank > 3 || captured_params.at("dim").type != 2)
            return false;

        int dim = captured_params.at("dim").i;
        if (dim < 0)
            dim += input_rank;

        return dim >= 0 && dim < input_rank && (ncnn_batch_axis == 233 || dim != ncnn_batch_axis);
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input       0 1 input
Reduction               amax        1 1 input amax 0=4 1=0 4=1 5=1
BinaryOp                shift       2 1 input amax shifted 0=1
UnaryOp                 exp         1 1 shifted exp 0=7
Reduction               sum         1 1 exp sum 0=0 1=0 4=1 5=1
UnaryOp                 log         1 1 sum log 0=8
BinaryOp                sub         2 1 shifted log out 0=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const Operand* input = ops.at("amax")->inputs[0];
        int dim = captured_params.at("dim").i;
        if (dim < 0)
            dim += input->shape.size();

        const int ncnn_batch_axis = input->params.at("__ncnn_batch_axis").i;
        const int axis = ncnn_batch_axis != 233 && dim > ncnn_batch_axis ? dim - 1 : dim;
        ops.at("amax")->params["3"] = std::vector<int>(1, axis);
        ops.at("sum")->params["3"] = std::vector<int>(1, axis);

        for (const auto& x : ops)
        {
            if (x.first == "sub")
                continue;

            Operand* operand = x.second->outputs[0];
            operand->type = input->type;
            operand->shape = input->shape;
            operand->params["__batch_index"] = input->params.at("__batch_index");
            operand->params["__ncnn_batch_axis"] = input->params.at("__ncnn_batch_axis");
            if (x.first == "amax" || x.first == "sum" || x.first == "log")
                operand->shape[dim] = 1;
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_log_softmax_stable, 18)


} // namespace ncnn

} // namespace pnnx
