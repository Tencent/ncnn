// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_grid_sample : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 grid
prim::Constant          op_0        0 1 mode value=%mode
prim::Constant          op_1        0 1 padding_mode value=%padding_mode
prim::Constant          op_2        0 1 align_corners value=%align_corners
aten::grid_sampler      op_3        5 1 input grid mode padding_mode align_corners out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.grid_sample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("mode").i == 0)
            op->params["mode"] = "bilinear";
        if (captured_params.at("mode").i == 1)
            op->params["mode"] = "nearest";
        if (captured_params.at("mode").i == 2)
            op->params["mode"] = "bicubic";

        if (captured_params.at("padding_mode").i == 0)
            op->params["padding_mode"] = "zeros";
        if (captured_params.at("padding_mode").i == 1)
            op->params["padding_mode"] = "border";
        if (captured_params.at("padding_mode").i == 2)
            op->params["padding_mode"] = "reflection";

        op->params["align_corners"] = captured_params.at("align_corners");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_grid_sample, 140)

class F_grid_sample_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 grid
GridSample              op_0        2 1 input grid out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.grid_sample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int align_corners = 0;
        std::string mode = "linear";
        std::string padding_mode = "zeros";
        if (captured_params.find("op_0.align_corners") != captured_params.end())
        {
            align_corners = captured_params.at("op_0.align_corners").i;
        }
        if (captured_params.find("op_0.mode") != captured_params.end())
        {
            mode = captured_params.at("op_0.mode").s;
        }
        if (captured_params.find("op_0.padding_mode") != captured_params.end())
        {
            padding_mode = captured_params.at("op_0.padding_mode").s;
        }

        op->params["mode"] = mode;
        op->params["padding_mode"] = padding_mode;
        op->params["align_corners"] = align_corners ? true : false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_grid_sample_onnx, 140)

} // namespace pnnx
