// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_scaled_dot_product_attention : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
16 15
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 attn_mask
nn.Linear               op_0        1 1 input q bias=%qbias in_features=%qdim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input k bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input v bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 q 10 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 k 12 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 v 14 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 10 16 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 12 17 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 14 18 dims=(0,2,1,3)
F.scaled_dot_product_attention sdpa 4 1 16 17 18 attn_mask 19 %*=%*
Tensor.permute          op_10       1 1 19 20 dims=(0,2,1,3)
Tensor.reshape          op_11       1 1 20 21 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%qdim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "MultiHeadAttention";
    }

    const char* name_str() const
    {
        return "sdpa_attention";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("sdpa.dropout_p") != captured_params.end())
        {
            if (captured_params.at("sdpa.dropout_p").type != 3 || captured_params.at("sdpa.dropout_p").f != 0.f)
                return false;
        }

        if (captured_params.find("sdpa.is_causal") != captured_params.end())
        {
            if (captured_params.at("sdpa.is_causal").type != 1 || captured_params.at("sdpa.is_causal").b != false)
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = captured_params.at("embed_dim");
        op->params["1"] = captured_params.at("num_heads");

        const int embed_dim = captured_params.at("embed_dim").i;
        const int qdim = captured_params.at("qdim").i;
        const int kdim = captured_params.at("kdim").i;
        const int vdim = captured_params.at("vdim").i;

        op->params["2"] = embed_dim * qdim;
        op->params["3"] = kdim;
        op->params["4"] = vdim;
        op->params["5"] = 1;
        if (captured_params.find("sdpa.scale") != captured_params.end())
            op->params["6"] = captured_params.at("sdpa.scale");

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = captured_attrs.at("op_0.weight");
        if (captured_params.at("qbias").b)
        {
            op->attrs["2"] = captured_attrs.at("op_0.bias");
        }
        else
        {
            op->attrs["2"] = Attribute({embed_dim}, std::vector<float>(embed_dim, 0.f));
        }
        op->attrs["3"] = Attribute();
        op->attrs["3"].data = {0, 0, 0, 0};
        op->attrs["4"] = captured_attrs.at("op_1.weight");
        if (captured_params.at("kbias").b)
        {
            op->attrs["5"] = captured_attrs.at("op_1.bias");
        }
        else
        {
            op->attrs["5"] = Attribute({embed_dim}, std::vector<float>(embed_dim, 0.f));
        }
        op->attrs["6"] = Attribute();
        op->attrs["6"].data = {0, 0, 0, 0};
        op->attrs["7"] = captured_attrs.at("op_2.weight");
        if (captured_params.at("vbias").b)
        {
            op->attrs["8"] = captured_attrs.at("op_2.bias");
        }
        else
        {
            op->attrs["8"] = Attribute({embed_dim}, std::vector<float>(embed_dim, 0.f));
        }
        op->attrs["9"] = Attribute();
        op->attrs["9"].data = {0, 0, 0, 0};
        op->attrs["a"] = captured_attrs.at("out_proj.weight");
        if (captured_params.at("outbias").b)
        {
            op->attrs["b"] = captured_attrs.at("out_proj.bias");
        }
        else
        {
            op->attrs["b"] = Attribute({qdim}, std::vector<float>(qdim, 0.f));
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention, 10)

class F_scaled_dot_product_attention_1 : public F_scaled_dot_product_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
17 16
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 kv
pnnx.Input              input_2     0 1 attn_mask
nn.Linear               op_0        1 1 input q bias=%qbias in_features=%qdim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 kv k bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 kv v bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 q 10 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 k 12 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 v 14 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 10 16 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 12 17 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 14 18 dims=(0,2,1,3)
F.scaled_dot_product_attention sdpa 4 1 16 17 18 attn_mask 19 %*=%*
Tensor.permute          op_10       1 1 19 20 dims=(0,2,1,3)
Tensor.reshape          op_11       1 1 20 21 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%qdim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_1, 10)

class F_scaled_dot_product_attention_2 : public F_scaled_dot_product_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
15 14
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input q bias=%qbias in_features=%qdim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input k bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input v bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 q 10 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 k 12 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 v 14 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 10 16 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 12 17 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 14 18 dims=(0,2,1,3)
F.scaled_dot_product_attention sdpa 3 1 16 17 18 19 %*=%*
Tensor.permute          op_10       1 1 19 20 dims=(0,2,1,3)
Tensor.reshape          op_11       1 1 20 21 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%qdim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        F_scaled_dot_product_attention::write(op, captured_params, captured_attrs);
        op->params["5"] = 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_2, 10)

class F_scaled_dot_product_attention_3 : public F_scaled_dot_product_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
16 15
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 kv
nn.Linear               op_0        1 1 input q bias=%qbias in_features=%qdim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 kv k bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 kv v bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 q 10 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 k 12 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 v 14 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 10 16 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 12 17 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 14 18 dims=(0,2,1,3)
F.scaled_dot_product_attention sdpa 3 1 16 17 18 19 %*=%*
Tensor.permute          op_10       1 1 19 20 dims=(0,2,1,3)
Tensor.reshape          op_11       1 1 20 21 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%qdim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        F_scaled_dot_product_attention::write(op, captured_params, captured_attrs);
        op->params["5"] = 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_3, 10)

class F_scaled_dot_product_attention_4 : public F_scaled_dot_product_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
15 14
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input q bias=%qbias in_features=%qdim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input k bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input v bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 q 10 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 k 12 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 v 14 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 10 16 dim0=1 dim1=2
torch.transpose         op_7        1 1 12 17 dim0=1 dim1=2
torch.transpose         op_8        1 1 14 18 dim0=1 dim1=2
F.scaled_dot_product_attention sdpa 3 1 16 17 18 19 %*=%*
torch.transpose         op_10       1 1 19 20 dim0=1 dim1=2
Tensor.reshape          op_11       1 1 20 21 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%qdim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        F_scaled_dot_product_attention::write(op, captured_params, captured_attrs);
        op->params["5"] = 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_4, 10)

class F_scaled_dot_product_attention_fb : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
F.scaled_dot_product_attention sdpa 3 1 query key value out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "SDPA";
    }

    const char* name_str() const
    {
        return "sdpa";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("sdpa.dropout_p") != captured_params.end())
        {
            if (captured_params.at("sdpa.dropout_p").type != 3 || captured_params.at("sdpa.dropout_p").f != 0.f)
                return false;
        }

        if (captured_params.find("sdpa.is_causal") != captured_params.end())
        {
            if (captured_params.at("sdpa.is_causal").type != 1 || captured_params.at("sdpa.is_causal").b != false)
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["5"] = 0;
        if (captured_params.find("sdpa.scale") != captured_params.end())
            op->params["6"] = captured_params.at("sdpa.scale");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_fb, 31)

class F_scaled_dot_product_attention_fb_mask : public F_scaled_dot_product_attention_fb
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 6
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 attn_mask
F.scaled_dot_product_attention sdpa 4 1 query key value attn_mask out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["5"] = 1;
        if (captured_params.find("sdpa.scale") != captured_params.end())
            op->params["6"] = captured_params.at("sdpa.scale");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_fb_mask, 31)

class F_scaled_dot_product_attention_fb_gqa : public F_scaled_dot_product_attention_fb
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 7
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
torch.repeat_interleave op_0        1 1 key k2 dim=-3 repeats=%repeats
torch.repeat_interleave op_1        1 1 value v2 dim=-3 repeats=%repeats
F.scaled_dot_product_attention sdpa 3 1 query k2 v2 out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_fb_gqa, 30)

class F_scaled_dot_product_attention_fb_mask_gqa : public F_scaled_dot_product_attention_fb_mask
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 8
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 attn_mask
torch.repeat_interleave op_0        1 1 key k2 dim=-3 repeats=%repeats
torch.repeat_interleave op_1        1 1 value v2 dim=-3 repeats=%repeats
F.scaled_dot_product_attention sdpa 4 1 query k2 v2 attn_mask out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_fb_mask_gqa, 30)

} // namespace ncnn

} // namespace pnnx
