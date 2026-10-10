// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_affine_grid : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 theta
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 align_corners value=%align_corners
aten::affine_grid_generator op_1    3 1 theta size align_corners out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.affine_grid";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_affine_grid, 110)

} // namespace pnnx
