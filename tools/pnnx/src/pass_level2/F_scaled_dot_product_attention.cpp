// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_scaled_dot_product_attention : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 attn_mask
prim::Constant          op_0        0 1 dropout_p value=%dropout_p
prim::Constant          op_1        0 1 is_causal value=%is_causal
aten::scaled_dot_product_attention  op_2 6 1 query key value attn_mask dropout_p is_causal out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.scaled_dot_product_attention";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention, 140)

class F_scaled_dot_product_attention_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 attn_mask
prim::Constant          op_0        0 1 dropout_p value=%dropout_p
prim::Constant          op_1        0 1 is_causal value=%is_causal
prim::Constant          op_2        0 1 scale value=%scale
aten::scaled_dot_product_attention  op_3 7 1 query key value attn_mask dropout_p is_causal scale out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.scaled_dot_product_attention";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(op, captured_params, captured_attrs);

        if (captured_params.at("scale").type == 0)
        {
            // drop scale=None for compatibility with old torch
            op->params.erase("scale");
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_1, 140)

class F_scaled_dot_product_attention_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 attn_mask
prim::Constant          op_0        0 1 dropout_p value=%dropout_p
prim::Constant          op_1        0 1 is_causal value=%is_causal
prim::Constant          op_2        0 1 scale value=%scale
prim::Constant          op_3        0 1 enable_gqa value=%enable_gqa
aten::scaled_dot_product_attention  op_4 8 1 query key value attn_mask dropout_p is_causal scale enable_gqa out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.scaled_dot_product_attention";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(op, captured_params, captured_attrs);

        if (captured_params.at("scale").type == 0)
        {
            // drop scale=None for compatibility with old torch
            op->params.erase("scale");
        }

        if (captured_params.at("enable_gqa").type == 1 && captured_params.at("enable_gqa").b == false)
        {
            // drop enable_gqa=False for compatibility with old torch
            op->params.erase("enable_gqa");
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_2, 140)

static bool NearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    float diff = (float)fabs(a - b);
    if (diff <= epsilon)
        return true;

    // relative error
    return diff < epsilon * std::max(fabs(a), fabs(b));
}

class F_scaled_dot_product_attention_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
12 11
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
Tensor.permute          op_0        1 1 key kt dims=(0,1,3,2)
prim::Constant          op_1        0 1 scale value=%sqrt_scale
aten::mul               op_2        2 1 query scale q
prim::Constant          op_3        0 1 scale2 value=%sqrt_scale
aten::mul               op_4        2 1 kt scale2 k
torch.matmul            op_5        2 1 q k qk
F.softmax               op_6        1 1 qk 4 dim=-1
torch.matmul            op_7        2 1 4 value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.scaled_dot_product_attention";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["dropout_p"] = 0.f;
        op->params["is_causal"] = false;

        const float sqrt_scale = captured_params.at("sqrt_scale").f;
        const float scale = sqrt_scale * sqrt_scale;

        op->params["scale"] = scale;

        if (!op->inputs[0]->shape.empty())
        {
            const int embed_dim = op->inputs[0]->shape[op->inputs[0]->shape.size() - 1];
            if (NearlyEqual(scale, 1.f / sqrt(embed_dim), 0.001))
            {
                // drop scale=None for compatibility with old torch
                op->params.erase("scale");
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_onnx, 140)

class F_scaled_dot_product_attention_onnx_1 : public F_scaled_dot_product_attention_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
14 13
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 attn_mask
Tensor.permute          op_0        1 1 key kt dims=(0,1,3,2)
prim::Constant          op_1        0 1 scale value=%sqrt_scale
aten::mul               op_2        2 1 query scale q
prim::Constant          op_3        0 1 scale2 value=%sqrt_scale
aten::mul               op_4        2 1 kt scale2 k
torch.matmul            op_5        2 1 q k qk
aten::add               op_6        2 1 qk attn_mask qkm
F.softmax               op_7        1 1 qkm 4 dim=-1
torch.matmul            op_8        2 1 4 value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_onnx_1, 140)

class F_scaled_dot_product_attention_onnx_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
14 13
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
Tensor.reshape          op_0        1 1 key pnnx_0 shape=(-1,%kvsize,%embed_per_head)
Tensor.permute          op_1        1 1 pnnx_0 pnnx_1 dims=(0,2,1)
Tensor.reshape          op_2        1 1 pnnx_1 pnnx_2 shape=(%batch,%num_heads,%embed_per_head,%kvsize)
prim::Constant          op_3        0 1 val_23 value=%sqrt_scale
aten::mul               op_4        2 1 query val_23 pnnx_3
prim::Constant          op_5        0 1 val_23_pnnxshadow1 value=%sqrt_scale
aten::mul               op_6        2 1 pnnx_2 val_23_pnnxshadow1 pnnx_4
torch.matmul            op_7        2 1 pnnx_3 pnnx_4 pnnx_5
F.softmax               op_8        1 1 pnnx_5 pnnx_6 dim=-1
torch.matmul            op_9        2 1 pnnx_6 value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.scaled_dot_product_attention";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["dropout_p"] = 0.f;
        op->params["is_causal"] = false;

        const float sqrt_scale = captured_params.at("sqrt_scale").f;
        const float scale = sqrt_scale * sqrt_scale;

        op->params["scale"] = scale;

        if (!op->inputs[0]->shape.empty())
        {
            const int embed_dim = op->inputs[0]->shape[op->inputs[0]->shape.size() - 1];
            if (NearlyEqual(scale, 1.f / sqrt(embed_dim), 0.001))
            {
                // drop scale=None for compatibility with old torch
                op->params.erase("scale");
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_onnx_2, 140)

class F_scaled_dot_product_attention_onnx_3 : public F_scaled_dot_product_attention_onnx_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
16 15
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 attn_mask
Tensor.reshape          op_0        1 1 key pnnx_10 shape=(-1,%kvsize,%embed_per_head)
Tensor.permute          op_1        1 1 pnnx_10 pnnx_11 dims=(0,2,1)
Tensor.reshape          op_2        1 1 pnnx_11 pnnx_12 shape=(%batch,%num_heads,%embed_per_head,%kvsize)
prim::Constant          op_3        0 1 val_23_pnnxshadow2 value=%sqrt_scale
aten::mul               op_4        2 1 query val_23_pnnxshadow2 pnnx_13
prim::Constant          op_5        0 1 val_23_pnnxshadow3 value=%sqrt_scale
aten::mul               op_6        2 1 pnnx_12 val_23_pnnxshadow3 pnnx_14
torch.matmul            op_7        2 1 pnnx_13 pnnx_14 pnnx_15
aten::add               op_8        2 1 pnnx_15 attn_mask pnnx_16
F.softmax               op_9        1 1 pnnx_16 pnnx_17 dim=-1
torch.matmul            op_10       2 1 pnnx_17 value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_onnx_3, 140)

} // namespace pnnx
