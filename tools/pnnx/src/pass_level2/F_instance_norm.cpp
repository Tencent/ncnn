// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_instance_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input_1     0 1 input
pnnx.Input              input_2     0 1 running_mean
pnnx.Input              input_3     0 1 running_var
pnnx.Input              input_4     0 1 weight
pnnx.Input              input_5     0 1 bias
prim::Constant          op_0        0 1 use_input_stats value=True
prim::Constant          op_1        0 1 momentum value=*
prim::Constant          op_2        0 1 eps value=%eps
prim::Constant          op_3        0 1 cudnn_enabled value=*
aten::instance_norm     op_4        9 1 input weight bias running_mean running_var use_input_stats momentum eps cudnn_enabled out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.instance_norm";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_instance_norm, 130)

class F_instance_norm_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
InstanceNormalization   op_0        3 1 input weight bias out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.instance_norm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float epsilon = 1e-05;
        if (captured_params.find("op_0.epsilon") != captured_params.end())
        {
            epsilon = captured_params.at("op_0.epsilon").f;
        }

        op->params["eps"] = epsilon;
        op->params["running_mean"] = Parameter();
        op->params["running_var"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_instance_norm_onnx, 131)

} // namespace pnnx
