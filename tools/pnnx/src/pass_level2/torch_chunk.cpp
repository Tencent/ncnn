// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_chunk : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 chunks
pnnx.Input              input_2     0 1 dim
aten::chunk             op_0        3 1 input chunks dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.chunk";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_chunk, 60)

class torch_chunk_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::split             op_0        1 1 input out dim=%dim indices=None num_outputs=%chunks
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.chunk";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_chunk_onnx_1, 60)

} // namespace pnnx
