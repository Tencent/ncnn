// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_div_rounding : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Input              other       0 1 other
prim::Constant          mode        0 1 mode value=%rounding_mode
aten::div               op          3 1 input other mode out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const Parameter& mode = captured_params.at("rounding_mode");
        return mode.type == 4 && (mode.s == "floor" || mode.s == "trunc");
    }

    const char* type_str() const
    {
        return "torch.div";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_div_rounding, 60)

} // namespace pnnx
