// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_adaptive_max_pool3d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 output_size
aten::adaptive_max_pool3d op_0      2 2 input output_size out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.adaptive_max_pool3d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["return_indices"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_adaptive_max_pool3d, 120)

} // namespace pnnx
