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

class TopK : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 k
TopK                    op_0        2 2 input k values indices axis=%axis largest=%largest sorted=%sorted
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
        if (captured_params.find("axis") != captured_params.end())
            axis = captured_params.at("axis").i;

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

        op->params["0"] = new_axis;
        op->params["1"] = largest;
        op->params["2"] = sorted;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(TopK, 20)

class TopK_0 : public TopK
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 2
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 k
TopK                    op_0        2 1 input k values axis=%axis largest=%largest sorted=%sorted
pnnx.Output             output      1 0 values
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(TopK_0, 20)

} // namespace ncnn

} // namespace pnnx
