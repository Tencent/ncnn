// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_prod : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
prim::Constant          op_0        0 1 keepdim value=%keepdim
prim::Constant          op_1        0 1 dtype value=*
aten::prod              op_2        4 1 input dim keepdim dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.prod";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_prod, 50)

class torch_prod_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
ReduceProd              op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.prod";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axes") == captured_params.end())
            return false;

        if (captured_params.at("op_0.axes").type != 2 && captured_params.at("op_0.axes").type != 5)
            return false;

        if (captured_params.at("op_0.axes").type == 5 && captured_params.at("op_0.axes").ai.size() > 1)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim;
        if (captured_params.at("op_0.axes").type == 2)
        {
            dim = captured_params.at("op_0.axes").i;
        }
        else // if (captured_params.at("op_0.axes").type == 5)
        {
            dim = captured_params.at("op_0.axes").ai[0];
        }

        op->params["dim"] = dim;

        if (captured_params.find("op_0.keepdims") != captured_params.end())
        {
            op->params["keepdim"] = captured_params.at("op_0.keepdims").i ? true : false;
        }
        else
        {
            op->params["keepdim"] = true;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_prod_onnx, 50)

} // namespace pnnx
