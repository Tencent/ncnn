// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_addmm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mat1
pnnx.Input              input_2     0 1 mat2
pnnx.Input              input_3     0 1 beta
pnnx.Input              input_4     0 1 alpha
aten::addmm             op_0        5 1 input mat1 mat2 beta alpha out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.addmm";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_addmm, 90)

class torch_addmm_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mat1
pnnx.Input              input_2     0 1 mat2
aten::addmm             op_0        3 1 input mat1 mat2 out beta=%beta alpha=%alpha
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.addmm";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_addmm_1, 90)

class torch_addmm_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mat1
pnnx.Input              input_2     0 1 mat2
Gemm                    op_0        3 1 mat1 mat2 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.addmm";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        int transA = 0;
        if (captured_params.find("op_0.transA") != captured_params.end())
        {
            transA = captured_params.at("op_0.transA").i;
        }

        int transB = 0;
        if (captured_params.find("op_0.transB") != captured_params.end())
        {
            transB = captured_params.at("op_0.transB").i;
        }

        return transA == 0 && transB == 0;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float alpha = 1.f;
        if (captured_params.find("op_0.alpha") != captured_params.end())
        {
            alpha = captured_params.at("op_0.alpha").f;
        }

        float beta = 1.f;
        if (captured_params.find("op_0.beta") != captured_params.end())
        {
            beta = captured_params.at("op_0.beta").f;
        }

        op->params["alpha"] = alpha;
        op->params["beta"] = beta;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_addmm_onnx, 90)

class torch_addmm_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mat1
pnnx.Input              input_2     0 1 mat2
Gemm                    op_0        3 1 mat1 mat2 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (transA && !transB)
            return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mat1
pnnx.Input              input_2     0 1 mat2
torch.transpose         at          1 1 mat1 at dim0=1 dim1=0
torch.addmm             addmm       3 1 input at mat2 out
pnnx.Output             output      1 0 out
)PNNXIR";
        else if (!transA && transB)
            return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mat1
pnnx.Input              input_2     0 1 mat2
torch.transpose         bt          1 1 mat2 bt dim0=1 dim1=0
torch.addmm             addmm       3 1 input mat1 bt out
pnnx.Output             output      1 0 out
)PNNXIR";
        else // if (transA && transB)
            return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mat1
pnnx.Input              input_2     0 1 mat2
torch.transpose         at          1 1 mat1 at dim0=1 dim1=0
torch.transpose         bt          1 1 mat2 bt dim0=1 dim1=0
torch.addmm             addmm       3 1 input at bt out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        transA = 0;
        if (captured_params.find("op_0.transA") != captured_params.end())
        {
            transA = captured_params.at("op_0.transA").i;
        }

        transB = 0;
        if (captured_params.find("op_0.transB") != captured_params.end())
        {
            transB = captured_params.at("op_0.transB").i;
        }

        return transA == 1 || transB == 1;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        float alpha = 1.f;
        if (captured_params.find("op_0.alpha") != captured_params.end())
        {
            alpha = captured_params.at("op_0.alpha").f;
        }

        float beta = 1.f;
        if (captured_params.find("op_0.beta") != captured_params.end())
        {
            beta = captured_params.at("op_0.beta").f;
        }

        ops.at("addmm")->params["alpha"] = alpha;
        ops.at("addmm")->params["beta"] = beta;

        ops.at("addmm")->inputnames = {"input", "mat1", "mat2"};

        if (transA)
            ops.at("at")->inputnames = {"input"};
        if (transB)
            ops.at("bt")->inputnames = {"input"};
    }

protected:
    mutable int transA;
    mutable int transB;
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_addmm_onnx_1, 90)

} // namespace pnnx
