// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_unsqueeze : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 dim value=%dim
aten::unsqueeze         op_1        2 1 input dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.unsqueeze";
    }
};

class torch_unsqueeze_dynamic : public torch_unsqueeze
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
aten::unsqueeze         op_0        2 1 input dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_unsqueeze, 60)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_unsqueeze_dynamic, 61)

class torch_unsqueeze_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
Unsqueeze               op_0        2 1 input dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.unsqueeze";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_unsqueeze_onnx, 60)

class torch_unsqueeze_onnx_1 : public torch_unsqueeze_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Unsqueeze               op_0        1 1 input out axes=%axes
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("axes").type == 5 && captured_params.at("axes").ai.size() == 1)
        {
            op->params["dim"] = captured_params.at("axes").ai[0];
        }
        else
        {
            op->params["dim"] = captured_params.at("axes");
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_unsqueeze_onnx_1, 60)

class torch_unsqueeze_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.Unsqueeze           op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.unsqueeze";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int dims_count = captured_params.at("op_0.arg0").i;
        if (dims_count == 1)
        {
            op->params["dim"] = captured_params.at("op_0.arg1").i;
        }
        else
        {
            std::vector<int> dims(dims_count);
            for (int i = 0; i < dims_count; i++)
            {
                dims[i] = captured_params.at("op_0.arg" + std::to_string(i + 1)).i;
            }
            op->params["dim"] = dims;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_unsqueeze_tnn, 60)

} // namespace pnnx
