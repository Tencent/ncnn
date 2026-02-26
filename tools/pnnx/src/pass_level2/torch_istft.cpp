// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_istft : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
12 11
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 window
prim::Constant          op_0        0 1 n_fft value=%n_fft
prim::Constant          op_1        0 1 hop_length value=%hop_length
prim::Constant          op_2        0 1 win_length value=%win_length
prim::Constant          op_3        0 1 center value=%center
prim::Constant          op_4        0 1 normalized value=%normalized
prim::Constant          op_5        0 1 onesided value=%onesided
prim::Constant          op_6        0 1 length value=%length
prim::Constant          op_7        0 1 return_complex value=%return_complex
aten::istft             op_8        10 1 input n_fft hop_length win_length window center normalized onesided length return_complex out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.istft";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_istft, 80)

} // namespace pnnx
