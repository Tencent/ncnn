// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_pixel_unshuffle : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 downscale_factor
aten::pixel_unshuffle   op_0        2 1 input downscale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pixel_unshuffle";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pixel_unshuffle, 110)

class F_pixel_unshuffle_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
SpaceToDepth            op_0        1 1 input out blocksize=%downscale_factor
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pixel_unshuffle";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pixel_unshuffle_onnx, 110)

} // namespace pnnx
