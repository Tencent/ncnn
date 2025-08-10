// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_embedding : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
prim::Constant          op_0        0 1 padding_idx value=*
prim::Constant          op_1        0 1 scale_grad_by_freq value=%scale_grad_by_freq
prim::Constant          op_2        0 1 sparse value=%sparse
aten::embedding         op_3        5 1 weight input padding_idx scale_grad_by_freq sparse out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.embedding";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_embedding, 140)

class F_embedding_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
Gather                  op_0        2 1 weight input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.embedding";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["scale_grad_by_freq"] = false;
        op->params["sparse"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_embedding_onnx, 140)

class F_embedding_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
tnn.Gather              op_0        2 1 input weight out arg0=0 arg1=1 arg2=0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.embedding";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["scale_grad_by_freq"] = false;
        op->params["sparse"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_embedding_tnn, 140)

} // namespace pnnx
