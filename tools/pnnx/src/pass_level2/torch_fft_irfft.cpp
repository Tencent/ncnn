// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_fft_irfft : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 n
pnnx.Input              input_2     0 1 dim
pnnx.Input              input_3     0 1 norm
aten::fft_irfft         op_0        4 1 input n dim norm out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.fft.irfft";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_fft_irfft, 80)

} // namespace pnnx
