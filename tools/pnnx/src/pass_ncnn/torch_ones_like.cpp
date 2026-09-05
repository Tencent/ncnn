// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_ones_like : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.ones_like         op_0        1 1 input out dtype=%dtype
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 4
pnnx.Input              input       0 1 input
BinaryOp                op_0        1 1 input zero_out 0=2 1=1 2=0.0
BinaryOp                op_1        1 1 zero_out out 0=0 1=1 2=1.0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        // the BinaryOp lowering produces storage in the input's arithmetic
        // type; only apply it when the output dtype matches (no explicit
        // dtype override, or torch.float), otherwise keep torch.ones_like
        const std::map<std::string, Parameter>::const_iterator it = captured_params.find("dtype");
        if (it == captured_params.end())
            return true;

        const Parameter& dt = it->second;
        if (dt.type == 0)
            return true; // dtype=None: inherits the input dtype
        if (dt.type == 2 && dt.i == 6)
            return true; // torch.float

        return false;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_ones_like, 19)

} // namespace ncnn

} // namespace pnnx
