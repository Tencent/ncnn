// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_convert_rotaryembed.h"

#include "pass_level2.h"

namespace pnnx {

namespace ncnn {

class fuse_rotaryembed_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 cos_cache
pnnx.Input              input_2     0 1 sin_cache
torch.tensor_split      op_0        1 2 input 19 20 dim=%split_dim indices=(%embed_dim_half)
pnnx.Expression         op_1        1 1 20 21 expr=neg(@0)
torch.cat               op_2        2 1 21 19 22 dim=%cat_dim
pnnx.Expression         op_3        4 1 input cos_cache 22 sin_cache out expr=add(mul(@0,@1),mul(@2,@3))
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "RotaryEmbed";
    }

    const char* name_str() const
    {
        return "rope";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operand* input = matched_operators.at("op_0")->inputs[0];
        if (!input->shape.empty())
        {
            const int embed_dim = input->shape[input->shape.size() - 1];
            const int embed_dim_half = captured_params.at("embed_dim_half").i;
            if (embed_dim != embed_dim_half * 2)
                return false;
        }

        const int split_dim = captured_params.at("split_dim").i;
        if (split_dim != 3 && split_dim != -1)
            return false;

        const int cat_dim = captured_params.at("cat_dim").i;
        if (cat_dim != 3 && cat_dim != -1)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["0"] = 0; // non-interleaved
    }
};

class fuse_rotaryembed_pass_interleaved : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 11
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 cos_cache
pnnx.Input              input_2     0 1 sin_cache
Tensor.reshape          op_0        1 1 input 22 shape=(%batch,%num_heads,%seqlen,%embed_dim_half,2)
torch.transpose         op_1        1 1 22 23 dim0=%interleave_dim0 dim1=%interleave_dim1
Tensor.reshape          op_2        1 1 23 24 shape=(%batch,%num_heads,%seqlen,%embed_dim)
torch.tensor_split      op_3        1 2 24 28 29 dim=%split_dim indices=(%embed_dim_half)
pnnx.Expression         op_4        1 1 29 30 expr=neg(@0)
torch.cat               op_5        2 1 30 28 31 dim=%cat_dim
pnnx.Expression         op_6        4 1 24 cos_cache 31 sin_cache out expr=add(mul(@0,@1),mul(@2,@3))
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "RotaryEmbed";
    }

    const char* name_str() const
    {
        return "rope";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int embed_dim_half = captured_params.at("embed_dim_half").i;
        const int embed_dim = captured_params.at("embed_dim").i;
        if (embed_dim != embed_dim_half * 2)
            return false;

        const int interleave_dim0 = captured_params.at("interleave_dim0").i;
        const int interleave_dim1 = captured_params.at("interleave_dim1").i;
        if (!((interleave_dim0 == 4 && interleave_dim1 == 3) || (interleave_dim0 == 3 && interleave_dim1 == 4)))
            return false;

        const int split_dim = captured_params.at("split_dim").i;
        if (split_dim != 3 && split_dim != -1)
            return false;

        const int cat_dim = captured_params.at("cat_dim").i;
        if (cat_dim != 3 && cat_dim != -1)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["0"] = 1; // interleaved
    }
};

void fuse_convert_rotaryembed(Graph& graph)
{
    fuse_rotaryembed_pass_interleaved a;
    fuse_rotaryembed_pass b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace ncnn

} // namespace pnnx
