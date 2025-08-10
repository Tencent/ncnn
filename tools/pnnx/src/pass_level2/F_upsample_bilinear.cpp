// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_upsample_bilinear : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 align_corners value=1
prim::Constant          op_1        0 1 scale_h value=None
prim::Constant          op_2        0 1 scale_w value=None
aten::upsample_bilinear2d op_3      5 1 input size align_corners scale_h scale_w out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample_bilinear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_bilinear, 110)

class F_upsample_bilinear_1_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 align_corners value=1
prim::Constant          op_1        0 1 scale_factor value=None
aten::upsample_bilinear2d op_2      4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample_bilinear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_bilinear_1_1, 110)

class F_upsample_bilinear_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 scale_factor
prim::Constant          op_0        0 1 size value=None
prim::Constant          op_1        0 1 align_corners value=1
aten::upsample_bilinear2d op_2      4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample_bilinear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_bilinear_1, 110)

} // namespace pnnx
