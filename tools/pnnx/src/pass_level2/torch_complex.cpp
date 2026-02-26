// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_complex : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 real
pnnx.Input              input_1     0 1 imag
aten::complex           op_0        2 1 real imag out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.complex";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_complex, 60)

} // namespace pnnx
