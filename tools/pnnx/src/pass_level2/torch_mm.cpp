// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_mm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mat2
aten::mm                op_0        2 1 input mat2 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.mm";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_mm, 90)

class torch_mm_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mat2
Gemm                    op_0        2 1 input mat2 out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.mm";
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

        float alpha = 1.f;
        if (captured_params.find("op_0.alpha") != captured_params.end())
        {
            alpha = captured_params.at("op_0.alpha").f;
        }

        return transA == 0 && transB == 0 && alpha == 1.f;
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_mm_onnx, 90)

class torch_mm_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 a
pnnx.Input              input_1     0 1 b
Gemm                    op_0        2 1 a b out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (transA && !transB)
            return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 a
pnnx.Input              input_1     0 1 b
torch.transpose         at          1 1 a at dim0=1 dim1=0
torch.mm                mm          2 1 at b out
pnnx.Output             output      1 0 out
)PNNXIR";
        else if (!transA && transB)
            return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 a
pnnx.Input              input_1     0 1 b
torch.transpose         bt          1 1 b bt dim0=1 dim1=0
torch.mm                mm          2 1 a bt out
pnnx.Output             output      1 0 out
)PNNXIR";
        else // if (transA && transB)
            return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 a
pnnx.Input              input_1     0 1 b
torch.transpose         at          1 1 a at dim0=1 dim1=0
torch.transpose         bt          1 1 b bt dim0=1 dim1=0
torch.mm                mm          2 1 at bt out
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

        float alpha = 1.f;
        if (captured_params.find("op_0.alpha") != captured_params.end())
        {
            alpha = captured_params.at("op_0.alpha").f;
        }

        return (transA == 1 || transB == 1) && alpha == 1.f;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        ops.at("mm")->inputnames = {"input", "mat2"};

        if (transA)
            ops.at("at")->inputnames = {"input"};
        if (transB)
            ops.at("bt")->inputnames = {"input"};
    }

protected:
    mutable int transA;
    mutable int transB;
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_mm_onnx_1, 90)

} // namespace pnnx
