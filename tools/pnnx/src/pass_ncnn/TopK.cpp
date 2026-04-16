// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

static int parameter_to_bool(const Parameter& p, int default_value)
{
    if (p.type == 1)
        return p.b ? 1 : 0;
    if (p.type == 2)
        return p.i ? 1 : 0;

    return default_value;
}

class torch_topk : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
torch.topk              op_0        1 2 input values indices k=%k dim=%dim largest=%largest sorted=%sorted
pnnx.Output             output      2 0 values indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "TopK";
    }

    const char* name_str() const
    {
        return "topk";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int axis = -1;
        if (captured_params.find("dim") != captured_params.end())
        {
            const Parameter& dim_p = captured_params.at("dim");
            if (dim_p.type == 2)
                axis = dim_p.i;
            else if (dim_p.type == 5 && !dim_p.ai.empty())
                axis = dim_p.ai[0];
        }

        int largest = 1;
        if (captured_params.find("largest") != captured_params.end())
            largest = parameter_to_bool(captured_params.at("largest"), 1);

        int sorted = 1;
        if (captured_params.find("sorted") != captured_params.end())
            sorted = parameter_to_bool(captured_params.at("sorted"), 1);

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        if (axis == batch_index)
        {
            fprintf(stderr, "TopK along batch axis is not supported\n");
            return;
        }

        int new_axis = axis;
        if (axis >= 0)
            new_axis = axis > batch_index ? axis - 1 : axis;

        int k_val = 1;
        if (captured_params.find("k") != captured_params.end())
        {
            const Parameter& k_p = captured_params.at("k");
            if (k_p.type == 2)
                k_val = k_p.i;
            else if (k_p.type == 5 && !k_p.ai.empty())
                k_val = k_p.ai[0];
        }

        op->params["0"] = new_axis;
        op->params["1"] = largest;
        op->params["2"] = sorted;
        op->params["3"] = k_val;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_topk, 20)

class torch_topk_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 1
pnnx.Input              input_0     0 1 input
torch.topk              op_0        1 1 input values k=%k dim=%dim largest=%largest sorted=%sorted
pnnx.Output             output      1 0 values
)PNNXIR";
    }

    const char* type_str() const
    {
        return "TopK";
    }

    const char* name_str() const
    {
        return "topk";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int axis = -1;
        if (captured_params.find("dim") != captured_params.end())
        {
            const Parameter& dim_p = captured_params.at("dim");
            if (dim_p.type == 2)
                axis = dim_p.i;
            else if (dim_p.type == 5 && !dim_p.ai.empty())
                axis = dim_p.ai[0];
        }

        int largest = 1;
        if (captured_params.find("largest") != captured_params.end())
            largest = parameter_to_bool(captured_params.at("largest"), 1);

        int sorted = 1;
        if (captured_params.find("sorted") != captured_params.end())
            sorted = parameter_to_bool(captured_params.at("sorted"), 1);

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        if (axis == batch_index)
        {
            fprintf(stderr, "TopK along batch axis is not supported\n");
            return;
        }

        int new_axis = axis;
        if (axis >= 0)
            new_axis = axis > batch_index ? axis - 1 : axis;

        int k_val = 1;
        if (captured_params.find("k") != captured_params.end())
        {
            const Parameter& k_p = captured_params.at("k");
            if (k_p.type == 2)
                k_val = k_p.i;
            else if (k_p.type == 5 && !k_p.ai.empty())
                k_val = k_p.ai[0];
        }

        op->params["0"] = new_axis;
        op->params["1"] = largest;
        op->params["2"] = sorted;
        op->params["3"] = k_val;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_topk_0, 20)

} // namespace ncnn

} // namespace pnnx
