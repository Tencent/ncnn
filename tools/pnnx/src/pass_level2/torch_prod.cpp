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

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axes") != captured_params.end())
        {
            if (captured_params.at("op_0.axes").type == 2)
            {
                op->params["dim"] = captured_params.at("op_0.axes");
            }
            else
            {
                const std::vector<int>& dim = captured_params.at("op_0.axes").ai;
                if (dim.size() == 1)
                {
                    // prefer single dim for compatible with torch
                    op->params["dim"] = dim[0];
                }
                else
                {
                    op->params["dim"] = dim;
                }
            }
        }
        else
        {
            // reduce all
            const int input_rank = (int)op->inputs[0]->shape.size();
            std::vector<int> dim(input_rank);
            for (int i = 0; i < input_rank; i++)
            {
                dim[i] = i;
            }
            op->params["dim"] = dim;
        }

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
