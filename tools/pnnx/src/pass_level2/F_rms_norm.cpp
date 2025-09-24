// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_rms_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 normalized_shape
prim::Constant          op_0        0 1 eps value=%eps
aten::rms_norm          op_1        4 1 input normalized_shape weight eps out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.rms_norm";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_rms_norm, 130)

class F_rms_norm_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
RMSNormalization        op_0        2 1 input weight out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.rms_norm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int input_rank = op->inputs[0]->shape.size();

        int axis = -1;
        if (captured_params.find("op_0.axis") != captured_params.end())
        {
            axis = captured_params.at("op_0.axis").i;
        }

        float epsilon = 1e-05;
        if (captured_params.find("op_0.epsilon") != captured_params.end())
        {
            epsilon = captured_params.at("op_0.epsilon").f;
        }

        if (axis < 0)
        {
            axis = input_rank + axis;
        }

        std::vector<int> normalized_shape;
        for (int i = axis; i < input_rank; i++)
        {
            normalized_shape.push_back(op->inputs[0]->shape[i]);
        }

        op->params["normalized_shape"] = normalized_shape;
        op->params["eps"] = epsilon;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_rms_norm_onnx, 130)

} // namespace pnnx
