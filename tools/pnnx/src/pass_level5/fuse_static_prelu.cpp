// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_static_prelu.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_static_Fprelu_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%num_parameters)f32
F.prelu                 op_0        2 1 input weight out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.PReLU                prelu       1 1 input out num_parameters=%num_parameters @weight=%op_weight.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class convert_prelu_to_leakyrelu : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.PReLU                op_0        1 1 input out num_parameters=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.LeakyReLU";
    }

    const char* name_str() const
    {
        return "leakyrelu";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const Attribute& weight = captured_attrs.at("op_0.weight");
        op->params["negative_slope"] = weight.get_float32_data()[0];
    }
};

void fuse_static_prelu(Graph& graph)
{
    fuse_static_Fprelu_pass a;
    convert_prelu_to_leakyrelu b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace pnnx
