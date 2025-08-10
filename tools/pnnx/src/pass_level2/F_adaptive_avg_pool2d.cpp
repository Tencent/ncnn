// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_adaptive_avg_pool2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 output_size
aten::adaptive_avg_pool2d op_0      2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_avg_pool2d";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_adaptive_avg_pool2d, 120)

class F_adaptive_avg_pool2d_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input #input=(?,?,?,?)f32
GlobalAveragePool       op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_avg_pool2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["output_size"] = {1, 1};
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_adaptive_avg_pool2d_onnx, 120)

class F_adaptive_avg_pool2d_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.Pooling             op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_avg_pool2d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int pool_type = captured_params.at("op_0.arg0").i;
        if (pool_type != 1)
            return false;

        const int pad_h = captured_params.at("op_0.arg5").i;
        const int pad_w = captured_params.at("op_0.arg6").i;
        if (pad_h != 0 || pad_w != 0)
            return false;

        const int kernel_h = captured_params.at("op_0.arg1").i;
        const int kernel_w = captured_params.at("op_0.arg2").i;
        if (kernel_h == 0 && kernel_w == 0)
            return true;

        if (captured_params.find("op_0.arg11") != captured_params.end())
        {
            const int is_adaptive = captured_params.at("op_0.arg11").i;
            if (is_adaptive == 1)
                return true;
        }

        return false;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // captured_params.at("op_0.arg0"); // pool_type
        const int kernel_h = captured_params.at("op_0.arg1").i;
        const int kernel_w = captured_params.at("op_0.arg2").i;
        if (kernel_h == 0 && kernel_w == 0)
        {
            // global pooling
            op->params["output_size"] = {1, 1};
        }

        // captured_params.at("op_0.arg3"); // stride_h
        // captured_params.at("op_0.arg4"); // stride_w
        // captured_params.at("op_0.arg5"); // pad_h
        // captured_params.at("op_0.arg6"); // pad_w
        // captured_params.at("op_0.arg7"); // kernel_index_h
        // captured_params.at("op_0.arg8"); // kernel_index_w
        // captured_params.at("op_0.arg9"); // pad_type
        // captured_params.at("op_0.arg10"); // ceil_mode

        if (captured_params.find("op_0.arg11") != captured_params.end())
        {
            const int is_adaptive = captured_params.at("op_0.arg11").i;
            if (is_adaptive == 1)
            {
                op->params["output_size"] = {captured_params.at("op_0.arg12").i, captured_params.at("op_0.arg13").i};
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_adaptive_avg_pool2d_tnn, 120)

} // namespace pnnx
