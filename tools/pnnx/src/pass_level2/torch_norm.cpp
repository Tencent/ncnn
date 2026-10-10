// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 dim value=%dim
prim::Constant          op_1        0 1 p value=%p
prim::Constant          op_2        0 1 keepdim value=%keepdim
aten::norm              op_3        4 1 input p dim keepdim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.norm";
    }
};

class torch_norm_2 : public torch_norm
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 dim value=%dim
prim::Constant          op_1        0 1 p value=%p
prim::Constant          op_2        0 1 keepdim value=%keepdim
prim::Constant          op_3        0 1 dtype value=*
aten::linalg_vector_norm op_4       5 1 input p dim keepdim dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class torch_norm_dims : public torch_norm
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 dim value=%dim
prim::Constant          op_1        0 1 p value=%p
prim::Constant          op_2        0 1 keepdim value=%keepdim
aten::norm              op_3        4 1 input p dim keepdim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class torch_norm_dims_2 : public torch_norm
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 dim value=%dim
prim::Constant          op_1        0 1 p value=%p
prim::Constant          op_2        0 1 keepdim value=%keepdim
prim::Constant          op_3        0 1 dtype value=*
aten::linalg_vector_norm op_4       5 1 input p dim keepdim dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_norm, 90)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_norm_2, 90)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_norm_dims, 90)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_norm_dims_2, 90)

class torch_norm_fro : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 dim value=%dim
prim::Constant          op_1        0 1 keepdim value=%keepdim
aten::frobenius_norm    op_2        3 1 input dim keepdim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.norm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["dim"] = captured_params.at("dim");
        op->params["p"] = "fro";
        op->params["keepdim"] = captured_params.at("keepdim");
    }
};

class torch_norm_fro_dims : public torch_norm_fro
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 dim value=%dim
prim::Constant          op_1        0 1 p value=%p
prim::Constant          op_2        0 1 keepdim value=%keepdim
aten::norm              op_3        4 1 input p dim keepdim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_norm_fro, 90)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_norm_fro_dims, 90)

class torch_norm_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input       0 1 input
aten::abs               op_0        1 1 input 4
prim::Constant          op_1        0 1 v2 value=2.0
aten::pow               op_2        2 1 4 v2 5
torch.sum               op_3        1 1 5 6 dim=%dim keepdim=%keepdim
prim::Constant          op_4        0 1 v0p5 value=0.5
aten::pow               op_5        2 1 6 v0p5 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.norm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["dim"] = captured_params.at("dim");
        op->params["keepdim"] = captured_params.at("keepdim");
        op->params["p"] = 2;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_norm_onnx, 90)

class torch_norm_onnx_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input       0 1 input
aten::abs               op_0        1 1 input 7
prim::Constant          op_1        0 1 v1 value=1.0
aten::pow               op_2        2 1 7 v1 8
torch.sum               op_3        1 1 8 9 dim=%dim keepdim=%keepdim
prim::Constant          op_4        0 1 v1_1 value=1.0
aten::pow               op_5        2 1 9 v1_1 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.norm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["dim"] = captured_params.at("dim");
        op->params["keepdim"] = captured_params.at("keepdim");
        op->params["p"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_norm_onnx_2, 90)

class torch_norm_onnx_l2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
ReduceL2                op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.norm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        bool keepdim = true;
        if (captured_params.find("op_0.keepdims") != captured_params.end())
        {
            keepdim = captured_params.at("op_0.keepdims").i == 1 ? true : false;
        }

        if (captured_params.find("op_0.axes") == captured_params.end())
        {
            op->params["dim"] = Parameter();
        }
        else
        {
            op->params["dim"] = captured_params.at("op_0.axes");
        }
        op->params["keepdim"] = keepdim;
        op->params["p"] = 2;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_norm_onnx_l2, 90)

class torch_norm_onnx_l1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
ReduceL1                op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.norm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        bool keepdim = true;
        if (captured_params.find("op_0.keepdims") != captured_params.end())
        {
            keepdim = captured_params.at("op_0.keepdims").i == 1 ? true : false;
        }

        if (captured_params.find("op_0.axes") == captured_params.end())
        {
            op->params["dim"] = Parameter();
        }
        else
        {
            op->params["dim"] = captured_params.at("op_0.axes");
        }
        op->params["keepdim"] = keepdim;
        op->params["p"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_norm_onnx_l1, 90)

class torch_norm_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.ReduceL2            op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.norm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::vector<int> dim;
        for (int i = 1;; i++)
        {
            if (captured_params.find("op_0.arg" + std::to_string(i)) == captured_params.end())
                break;

            dim.push_back(captured_params.at("op_0.arg" + std::to_string(i)).i);
        }

        op->params["dim"] = dim;
        op->params["keepdim"] = captured_params.at("op_0.arg0").i ? true : false;
        op->params["p"] = 2;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_norm_tnn, 90)

} // namespace pnnx
