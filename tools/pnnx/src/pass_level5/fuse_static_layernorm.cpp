// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_static_layernorm.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_static_Flayernorm_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data
pnnx.Attribute          op_bias     0 1 bias @data
F.layer_norm            op_0        3 1 input weight bias out normalized_shape=%normalized_shape eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.LayerNorm            ln          1 1 input out normalized_shape=%normalized_shape eps=%eps elementwise_affine=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_static_Flayernorm_pass_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data
F.layer_norm            op_0        2 1 input weight out normalized_shape=%normalized_shape eps=%eps bias=None
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.LayerNorm            ln          1 1 input out normalized_shape=%normalized_shape eps=%eps elementwise_affine=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& /*matched_operators*/, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        auto weight_data = captured_attrs.at("op_weight.data");
        std::vector<float> weight_data_fp32 = weight_data.get_float32_data();
        for (auto w : weight_data_fp32)
        {
            if (w != 1.f)
                return false;
        }

        return true;
    }
};

void fuse_static_layernorm(Graph& graph)
{
    fuse_static_Flayernorm_pass a;
    fuse_static_Flayernorm_pass_onnx b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace pnnx
