// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_multiheadattention.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

#include <torch/csrc/api/include/torch/version.h>

namespace pnnx {

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

class fuse_transformers_attention : public GraphRewriterPass
{
public:
    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.MultiheadAttention   attn_ht     1 1 input out embed_dim=%embed_dim kdim=%embed_dim vdim=%embed_dim batch_first=True add_zero_attn=False add_bias_kv=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const int embed_dim = captured_params.at("embed_dim").i;

        int num_heads;
        if (captured_params.find("num_heads") != captured_params.end())
        {
            num_heads = captured_params.at("num_heads").i;
            if (captured_params.find("feat_per_head") != captured_params.end())
            {
                const int feat_per_head = captured_params.at("feat_per_head").i;
                if (embed_dim != num_heads * feat_per_head)
                    return false;
            }
        }
        else // if (captured_params.find("feat_per_head") != captured_params.end())
        {
            const int feat_per_head = captured_params.at("feat_per_head").i;
            num_heads = embed_dim / feat_per_head;
        }

        if (captured_params.find("batch") != captured_params.end() && captured_params.find("batch_mul_num_heads") != captured_params.end())
        {
            const int batch = captured_params.at("batch").i;
            const int batch_mul_num_heads = captured_params.at("batch_mul_num_heads").i;
            if (batch_mul_num_heads != batch * num_heads)
                return false;
        }

        if (captured_params.find("sqrt_feat_per_head") != captured_params.end() && captured_params.find("feat_per_head") != captured_params.end())
        {
            const int feat_per_head = captured_params.at("feat_per_head").i;
            const float sqrt_feat_per_head = captured_params.at("sqrt_feat_per_head").f;
            if (!NearlyEqual(sqrt_feat_per_head, sqrt(feat_per_head), 0.001))
                return false;
        }

        if (captured_params.find("inv_sqrt_embed_dim_per_head") != captured_params.end() && captured_params.find("feat_per_head") != captured_params.end())
        {
            const int feat_per_head = captured_params.at("feat_per_head").i;
            const float inv_sqrt_embed_dim_per_head = captured_params.at("inv_sqrt_embed_dim_per_head").f;
            if (!NearlyEqual(inv_sqrt_embed_dim_per_head, 1.f / sqrt(feat_per_head), 0.001))
                return false;
        }

        if (captured_params.find("inv_sqrt_sqrt_embed_dim_per_head") != captured_params.end() && captured_params.find("feat_per_head") != captured_params.end())
        {
            const int feat_per_head = captured_params.at("feat_per_head").i;
            const float inv_sqrt_sqrt_embed_dim_per_head = captured_params.at("inv_sqrt_sqrt_embed_dim_per_head").f;
            if (!NearlyEqual(inv_sqrt_sqrt_embed_dim_per_head, 1.f / sqrt(sqrt(feat_per_head)), 0.001))
                return false;
        }

        if (captured_params.find("softmax_dim") != captured_params.end())
        {
            const int softmax_dim = captured_params.at("softmax_dim").i;
            int softmax_input_rank = (int)matched_operators.at("softmax")->inputs[0]->shape.size();
            if (softmax_dim != -1 && softmax_dim != softmax_input_rank - 1)
                return false;
        }

        batch = -1;
        if (captured_params.find("batch") != captured_params.end())
            batch = captured_params.at("batch").i;

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Operator* op = ops.at("attn_ht");

        const int embed_dim = captured_params.at("embed_dim").i;

        int num_heads;
        if (captured_params.find("num_heads") != captured_params.end())
        {
            num_heads = captured_params.at("num_heads").i;
        }
        else // if (captured_params.find("feat_per_head") != captured_params.end())
        {
            const int feat_per_head = captured_params.at("feat_per_head").i;
            num_heads = embed_dim / feat_per_head;
        }
        op->params["num_heads"] = num_heads;

        const bool qbias = captured_params.at("qbias").b;
        const bool kbias = captured_params.at("kbias").b;
        const bool vbias = captured_params.at("vbias").b;
        const bool outbias = captured_params.at("outbias").b;
        const bool bias = qbias || kbias || vbias || outbias;

        op->params["bias"] = bias;

        op->attrs["in_proj_weight"] = captured_attrs.at("op_0.weight") + captured_attrs.at("op_1.weight") + captured_attrs.at("op_2.weight");

        op->attrs["out_proj.weight"] = captured_attrs.at("out_proj.weight");

        if (bias)
        {
            op->attrs["in_proj_bias"] = Attribute();
            op->attrs["in_proj_bias"].type = op->attrs["in_proj_weight"].type;
            op->attrs["in_proj_bias"].shape = {embed_dim * 3};
            // combine qkv bias
            std::vector<float> in_proj_bias(embed_dim * 3);
            {
                float* in_proj_bias_ptr = (float*)in_proj_bias.data();
                if (qbias)
                {
                    auto qb = captured_attrs.at("op_0.bias").get_float32_data();
                    memcpy(in_proj_bias_ptr, (const void*)qb.data(), embed_dim * sizeof(float));
                }
                else
                {
                    memset(in_proj_bias_ptr, 0, embed_dim * sizeof(float));
                }
                in_proj_bias_ptr += embed_dim;
                if (kbias)
                {
                    auto kb = captured_attrs.at("op_1.bias").get_float32_data();
                    memcpy(in_proj_bias_ptr, (const void*)kb.data(), embed_dim * sizeof(float));
                }
                else
                {
                    memset(in_proj_bias_ptr, 0, embed_dim * sizeof(float));
                }
                in_proj_bias_ptr += embed_dim;
                if (vbias)
                {
                    auto vb = captured_attrs.at("op_2.bias").get_float32_data();
                    memcpy(in_proj_bias_ptr, (const void*)vb.data(), embed_dim * sizeof(float));
                }
                else
                {
                    memset(in_proj_bias_ptr, 0, embed_dim * sizeof(float));
                }
            }
            op->attrs["in_proj_bias"].set_float32_data(in_proj_bias);

            if (outbias)
            {
                op->attrs["out_proj.bias"] = captured_attrs.at("out_proj.bias");
            }
            else
            {
                // init bias as zero
                op->attrs["out_proj.bias"] = Attribute();
                op->attrs["out_proj.bias"].type = op->attrs["out_proj.weight"].type;
                op->attrs["out_proj.bias"].shape = {embed_dim};
                op->attrs["out_proj.bias"].set_float32_data(std::vector<float>(embed_dim, 0.f));
            }
        }
    }

protected:
    mutable int batch;
};

class fuse_transformers_cross_attention : public fuse_transformers_attention
{
public:
    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.MultiheadAttention   attn_ht     3 1 query key value out embed_dim=%embed_dim kdim=%kdim vdim=%vdim batch_first=True add_zero_attn=False add_bias_kv=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Operator* op = ops.at("attn_ht");

        const int embed_dim = captured_params.at("embed_dim").i;

        int num_heads;
        if (captured_params.find("num_heads") != captured_params.end())
        {
            num_heads = captured_params.at("num_heads").i;
        }
        else // if (captured_params.find("feat_per_head") != captured_params.end())
        {
            const int feat_per_head = captured_params.at("feat_per_head").i;
            num_heads = embed_dim / feat_per_head;
        }
        op->params["num_heads"] = num_heads;

        const int kdim = captured_params.at("kdim").i;
        const int vdim = captured_params.at("vdim").i;
        const bool qbias = captured_params.at("qbias").b;
        const bool kbias = captured_params.at("kbias").b;
        const bool vbias = captured_params.at("vbias").b;
        const bool outbias = captured_params.at("outbias").b;
        const bool bias = qbias || kbias || vbias || outbias;
        const bool same_qkv_dim = (embed_dim == kdim && embed_dim == vdim);

        op->params["bias"] = bias;

        if (same_qkv_dim)
        {
            // same qkv dim, merge into in_proj_weight
            op->attrs["in_proj_weight"] = captured_attrs.at("op_0.weight") + captured_attrs.at("op_1.weight") + captured_attrs.at("op_2.weight");
        }
        else
        {
            op->attrs["q_proj_weight"] = captured_attrs.at("op_0.weight");
            op->attrs["k_proj_weight"] = captured_attrs.at("op_1.weight");
            op->attrs["v_proj_weight"] = captured_attrs.at("op_2.weight");
        }

        op->attrs["out_proj.weight"] = captured_attrs.at("out_proj.weight");

        if (bias)
        {
            op->attrs["in_proj_bias"] = Attribute();
            op->attrs["in_proj_bias"].type = same_qkv_dim ? op->attrs["in_proj_weight"].type : op->attrs["q_proj_weight"].type;
            op->attrs["in_proj_bias"].shape = {embed_dim * 3};
            // combine qkv bias
            std::vector<float> in_proj_bias(embed_dim * 3);
            {
                float* in_proj_bias_ptr = (float*)in_proj_bias.data();
                if (qbias)
                {
                    auto qb = captured_attrs.at("op_0.bias").get_float32_data();
                    memcpy(in_proj_bias_ptr, (const void*)qb.data(), embed_dim * sizeof(float));
                }
                else
                {
                    memset(in_proj_bias_ptr, 0, embed_dim * sizeof(float));
                }
                in_proj_bias_ptr += embed_dim;
                if (kbias)
                {
                    auto kb = captured_attrs.at("op_1.bias").get_float32_data();
                    memcpy(in_proj_bias_ptr, (const void*)kb.data(), embed_dim * sizeof(float));
                }
                else
                {
                    memset(in_proj_bias_ptr, 0, embed_dim * sizeof(float));
                }
                in_proj_bias_ptr += embed_dim;
                if (vbias)
                {
                    auto vb = captured_attrs.at("op_2.bias").get_float32_data();
                    memcpy(in_proj_bias_ptr, (const void*)vb.data(), embed_dim * sizeof(float));
                }
                else
                {
                    memset(in_proj_bias_ptr, 0, embed_dim * sizeof(float));
                }
            }
            op->attrs["in_proj_bias"].set_float32_data(in_proj_bias);

            if (outbias)
            {
                op->attrs["out_proj.bias"] = captured_attrs.at("out_proj.bias");
            }
            else
            {
                // init bias as zero
                op->attrs["out_proj.bias"] = Attribute();
                op->attrs["out_proj.bias"].type = op->attrs["out_proj.weight"].type;
                op->attrs["out_proj.bias"].shape = {embed_dim};
                op->attrs["out_proj.bias"].set_float32_data(std::vector<float>(embed_dim, 0.f));
            }
        }
    }
};

class fuse_transformers_mask_attention : public fuse_transformers_attention
{
public:
    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 attn_mask
nn.MultiheadAttention   attn_ht     2 1 input attn_mask out embed_dim=%embed_dim kdim=%embed_dim vdim=%embed_dim batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_cross_mask_attention : public fuse_transformers_cross_attention
{
public:
    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 attn_mask
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_albert_attention : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 256 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 257 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 260 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 256 263 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 257 258 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 260 261 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 263 264 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 258 259 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 261 262 dims=(0,2,1,3)
torch.transpose         op_9        1 1 259 265 dim0=-1 dim1=-2
torch.matmul            op_10       2 1 264 265 266
pnnx.Expression         op_11       1 1 266 267 expr=div(@0,%sqrt_feat_per_head)
F.softmax               softmax     1 1 267 268 dim=%softmax_dim
torch.matmul            op_13       2 1 268 262 269
torch.transpose         op_14       1 1 269 270 dim0=2 dim1=1
torch.flatten           op_15       1 1 270 271 end_dim=-1 start_dim=2
nn.Linear               out_proj    1 1 271 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_albert_attention_1 : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 3 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 4 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 2 5 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 3 7 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 4 9 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 5 6 dim0=1 dim1=2
torch.transpose         op_7        1 1 7 8 dim0=1 dim1=2
torch.transpose         op_8        1 1 9 10 dim0=1 dim1=2
torch.transpose         op_9        1 1 8 11 dim0=-1 dim1=-2
torch.matmul            op_10       2 1 6 11 12
pnnx.Expression         op_11       1 1 12 13 expr=div(@0,%sqrt_feat_per_head)
F.softmax               softmax     1 1 13 14 dim=-1
torch.matmul            op_13       2 1 14 10 15
torch.transpose         op_14       1 1 15 16 dim0=2 dim1=1
torch.flatten           op_15       1 1 16 17 end_dim=-1 start_dim=2
nn.Linear               out_proj    1 1 17 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_albert_attention_onnx : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
20 19
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 256 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 257 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 260 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 256 263 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 257 258 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 260 261 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 263 264 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 258 265 dims=(0,2,3,1)
Tensor.permute          op_8        1 1 261 262 dims=(0,2,1,3)
torch.matmul            op_9        2 1 264 265 266
pnnx.Expression         op_10       1 1 266 267 expr=div(@0,%sqrt_feat_per_head)
F.softmax               softmax     1 1 267 268 dim=%softmax_dim
torch.matmul            op_12       2 1 268 262 269
Tensor.permute          op_13       1 1 269 270 dims=(0,2,1,3)
Tensor.reshape          op_14       1 1 270 271 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 271 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_bart_attention : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 4 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 6 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 2 3 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_4        1 1 3 8 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 4 5 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 6 7 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_7        1 1 8 9 dim0=1 dim1=2
torch.transpose         op_8        1 1 5 10 dim0=1 dim1=2
torch.transpose         op_9        1 1 7 11 dim0=1 dim1=2
Tensor.reshape          op_10       1 1 9 14 shape=(%batch_mul_num_heads,%qsize,%feat_per_head)
Tensor.reshape          op_11       1 1 10 12 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.reshape          op_12       1 1 11 17 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
torch.transpose         op_13       1 1 12 13 dim0=1 dim1=2
torch.bmm               op_14       2 1 14 13 15
F.softmax               softmax     1 1 15 16 dim=%softmax_dim
torch.bmm               op_16       2 1 16 17 18
Tensor.reshape          op_17       1 1 18 19 shape=(%batch,%num_heads,%qsize,%feat_per_head)
torch.transpose         op_18       1 1 19 20 dim0=1 dim1=2
Tensor.reshape          op_19       1 1 20 21 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_bart_attention_2 : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 6 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 7 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 2 3 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 6 8 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 7 10 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 3 4 dim0=1 dim1=2
torch.transpose         op_7        1 1 8 9 dim0=1 dim1=2
torch.transpose         op_8        1 1 10 11 dim0=1 dim1=2
pnnx.Expression         op_9        1 1 4 5 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_10       1 1 5 12 shape=(%batch_mul_num_heads,%qsize,%feat_per_head)
Tensor.reshape          op_11       1 1 9 13 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.reshape          op_12       1 1 11 14 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
torch.transpose         op_13       1 1 13 15 dim0=1 dim1=2
torch.bmm               op_14       2 1 12 15 16
F.softmax               softmax     1 1 16 17 dim=%softmax_dim
torch.bmm               op_16       2 1 17 14 18
Tensor.reshape          op_17       1 1 18 19 shape=(%batch,%num_heads,%qsize,%feat_per_head)
torch.transpose         op_18       1 1 19 20 dim0=1 dim1=2
Tensor.reshape          op_19       1 1 20 21 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_bart_attention_onnx : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 4 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 6 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 2 3 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_4        1 1 3 8 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 4 5 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 6 7 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_7        1 1 8 9 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 5 10 dims=(0,2,1,3)
Tensor.permute          op_9        1 1 7 11 dims=(0,2,1,3)
Tensor.reshape          op_10       1 1 9 14 shape=(%batch_mul_num_heads,%qsize,%feat_per_head)
Tensor.reshape          op_11       1 1 10 12 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.reshape          op_12       1 1 11 17 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.permute          op_13       1 1 12 13 dims=(0,2,1)
torch.matmul            op_14       2 1 14 13 15
F.softmax               softmax     1 1 15 16 dim=%softmax_dim
torch.matmul            op_16       2 1 16 17 18
Tensor.reshape          op_17       1 1 18 19 shape=(%batch,%num_heads,%qsize,%feat_per_head)
Tensor.permute          op_18       1 1 19 20 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 20 21 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_bart_attention_onnx_2 : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 6 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 7 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 2 3 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 6 8 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 7 10 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 3 4 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 8 9 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 10 11 dims=(0,2,1,3)
pnnx.Expression         op_9        1 1 4 5 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_10       1 1 5 12 shape=(%batch_mul_num_heads,%qsize,%feat_per_head)
Tensor.reshape          op_11       1 1 9 13 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.reshape          op_12       1 1 11 14 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.permute          op_13       1 1 13 15 dims=(0,2,1)
torch.matmul            op_14       2 1 12 15 16
F.softmax               softmax     1 1 16 17 dim=%softmax_dim
torch.matmul            op_16       2 1 17 14 18
Tensor.reshape          op_17       1 1 18 19 shape=(%batch,%num_heads,%qsize,%feat_per_head)
Tensor.permute          op_18       1 1 19 20 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 20 21 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_bart_sdpa_attention : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
17 16
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 4 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 6 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 2 8 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 4 5 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 6 7 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 8 9 dim0=1 dim1=2
torch.transpose         op_7        1 1 5 10 dim0=1 dim1=2
torch.transpose         op_8        1 1 7 11 dim0=1 dim1=2
F.scaled_dot_product_attention sdpa 3 1 9 10 11 19 attn_mask=None dropout_p=0.000000e+00 is_causal=False
torch.transpose         op_10       1 1 19 20 dim0=1 dim1=2
Tensor.reshape          op_11       1 1 20 21 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_bart_sdpa_attention_3 : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
17 16
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 5 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 6 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 2 3 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 5 7 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 6 9 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 3 4 dim0=1 dim1=2
torch.transpose         op_7        1 1 7 8 dim0=1 dim1=2
torch.transpose         op_8        1 1 9 10 dim0=1 dim1=2
F.scaled_dot_product_attention sdpa 3 1 4 8 10 11 attn_mask=None dropout_p=0.0 is_causal=False scale=%inv_sqrt_embed_dim_per_head
torch.transpose         op_10       1 1 11 12 dim0=1 dim1=2
Tensor.reshape          op_11       1 1 12 14 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 14 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_bart_sdpa_attention_onnx : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 3 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 4 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 6 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 3 9 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 4 5 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 6 7 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 9 10 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 5 11 dims=(0,2,3,1)
Tensor.permute          op_8        1 1 7 8 dims=(0,2,1,3)
pnnx.Expression         op_9        1 1 10 12 expr=mul(@0,%inv_sqrt_sqrt_embed_dim_per_head)
pnnx.Expression         op_10       1 1 11 13 expr=mul(@0,%inv_sqrt_sqrt_embed_dim_per_head)
torch.matmul            op_11       2 1 12 13 14
F.softmax               softmax     1 1 14 15 dim=%softmax_dim
torch.matmul            op_13       2 1 15 8 16
Tensor.permute          op_14       1 1 16 17 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 17 18 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 18 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_bart_sdpa_attention_onnx_1 : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
24 23
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 3 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 6 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 7 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 3 4 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 6 8 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 7 10 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 4 5 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 8 9 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 10 11 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 9 12 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.permute          op_10       1 1 12 13 dims=(0,2,1)
Tensor.reshape          op_11       1 1 13 14 shape=(%batch,%num_heads,%feat_per_head,%kvsize)
pnnx.Expression         op_12       1 1 5 15 expr=mul(@0,%inv_sqrt_sqrt_embed_dim_per_head)
pnnx.Expression         op_13       1 1 14 16 expr=mul(@0,%inv_sqrt_sqrt_embed_dim_per_head)
torch.matmul            op_14       2 1 15 16 17
F.softmax               softmax     1 1 17 18 dim=%softmax_dim
torch.matmul            op_16       2 1 18 11 19
Tensor.permute          op_17       1 1 19 20 dims=(0,2,1,3)
Tensor.reshape          op_18       1 1 20 21 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_bert_attention : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 5 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 8 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 2 3 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 5 6 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 8 9 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 3 4 dim0=1 dim1=2
torch.transpose         op_7        1 1 6 7 dim0=1 dim1=2
torch.transpose         op_8        1 1 9 10 dim0=1 dim1=2
torch.transpose         op_9        1 1 7 11 dim0=-1 dim1=-2
torch.matmul            op_10       2 1 4 11 12
pnnx.Expression         op_11       1 1 12 13 expr=div(@0,%sqrt_embed_dim_per_head)
F.softmax               softmax     1 1 13 14 dim=-1
torch.matmul            op_13       2 1 14 10 15
Tensor.permute          op_14       1 1 15 16 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 16 17 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 17 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_clip_attention : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 4 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 6 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 2 3 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_4        1 1 3 8 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 4 5 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 6 7 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_7        1 1 8 9 dim0=1 dim1=2
torch.transpose         op_8        1 1 5 10 dim0=1 dim1=2
torch.transpose         op_9        1 1 7 11 dim0=1 dim1=2
Tensor.reshape          op_10       1 1 9 14 shape=(%batch_mul_num_heads,%qsize,%feat_per_head)
Tensor.reshape          op_11       1 1 10 12 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.reshape          op_12       1 1 11 17 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
torch.transpose         op_13       1 1 12 13 dim0=1 dim1=2
torch.bmm               op_14       2 1 14 13 15
F.softmax               softmax     1 1 15 16 dim=%softmax_dim
torch.bmm               op_16       2 1 16 17 18
Tensor.reshape          op_17       1 1 18 19 shape=(%batch,%num_heads,%qsize,%feat_per_head)
torch.transpose         op_18       1 1 19 20 dim0=1 dim1=2
Tensor.reshape          op_19       1 1 20 21 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_clip_attention_2 : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 5 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 6 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 7 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 5 8 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 6 10 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 7 12 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 8 9 dim0=1 dim1=2
torch.transpose         op_7        1 1 10 11 dim0=1 dim1=2
torch.transpose         op_8        1 1 12 13 dim0=1 dim1=2
torch.transpose         op_9        1 1 11 14 dim0=-1 dim1=-2
torch.matmul            op_10       2 1 9 14 15
pnnx.Expression         op_11       1 1 15 16 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
F.softmax               softmax     1 1 16 17 dim=%softmax_dim
torch.matmul            op_13       2 1 17 13 18
torch.transpose         op_14       1 1 18 19 dim0=1 dim1=2
Tensor.reshape          op_16       1 1 19 21 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_clip_sdpa_attention : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
17 16
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 6 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 7 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 8 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 6 9 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 7 11 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 8 13 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 9 10 dim0=1 dim1=2
torch.transpose         op_7        1 1 11 12 dim0=1 dim1=2
torch.transpose         op_8        1 1 13 14 dim0=1 dim1=2
F.scaled_dot_product_attention sdpa 3 1 10 12 14 18 attn_mask=None dropout_p=0.000000e+00 is_causal=False scale=%inv_sqrt_embed_dim_per_head
torch.transpose         op_13       1 1 18 19 dim0=1 dim1=2
Tensor.reshape          op_15       1 1 19 21 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_clip_attention_onnx_2 : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
20 19
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 5 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 6 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 7 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 5 8 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 6 10 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 7 11 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 8 9 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 10 13 dims=(0,2,3,1)
Tensor.permute          op_8        1 1 11 12 dims=(0,2,1,3)
torch.matmul            op_9        2 1 9 13 14
pnnx.Expression         op_10       1 1 14 15 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
F.softmax               softmax     1 1 15 16 dim=%softmax_dim
torch.matmul            op_12       2 1 16 12 17
Tensor.permute          op_13       1 1 17 18 dims=(0,2,1,3)
Tensor.reshape          op_14       1 1 18 19 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 19 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_chinese_clip_attention : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 256 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 257 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 260 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 256 263 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 257 258 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 260 261 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 263 264 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 258 259 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 261 262 dims=(0,2,1,3)
torch.transpose         op_9        1 1 259 265 dim0=-1 dim1=-2
torch.matmul            op_10       2 1 264 265 266
pnnx.Expression         op_11       1 1 266 267 expr=div(@0,%sqrt_feat_per_head)
F.softmax               softmax     1 1 267 268 dim=%softmax_dim
torch.matmul            op_13       2 1 268 262 269
Tensor.permute          op_14       1 1 269 270 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 270 271 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 271 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_chinese_clip_attention_1 : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 5 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 9 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 12 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 5 6 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 9 10 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 12 13 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 6 7 dim0=1 dim1=2
torch.transpose         op_7        1 1 10 11 dim0=1 dim1=2
torch.transpose         op_8        1 1 13 14 dim0=1 dim1=2
torch.transpose         op_9        1 1 11 15 dim0=2 dim1=3
pnnx.Expression         op_10       1 1 7 8 expr=mul(@0,%inv_sqrt_feat_per_head)
torch.matmul            op_11       2 1 8 15 16
F.softmax               softmax     1 1 16 17 dim=%softmax_dim
torch.matmul            op_13       2 1 17 14 18
torch.transpose         op_14       1 1 18 19 dim0=1 dim1=2
Tensor.reshape          op_16       1 1 19 21 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_chinese_clip_attention_onnx : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
20 19
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 5 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 9 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 11 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 5 6 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 9 10 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 11 12 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 6 7 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 12 13 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 10 14 dims=(0,2,3,1)
pnnx.Expression         op_9        1 1 7 8 expr=mul(@0,%inv_sqrt_feat_per_head)
torch.matmul            op_10       2 1 8 14 15
F.softmax               softmax     1 1 15 16 dim=%softmax_dim
torch.matmul            op_12       2 1 16 13 17
Tensor.permute          op_13       1 1 17 18 dims=(0,2,1,3)
Tensor.reshape          op_14       1 1 18 19 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 19 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_ctrl_attention : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 3 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 4 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 2 5 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 3 7 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 4 9 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 5 6 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 7 8 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 9 10 dims=(0,2,1,3)
Tensor.permute          op_9        1 1 8 11 dims=(0,1,3,2)
torch.matmul            op_10       2 1 6 11 12
pnnx.Expression         op_11       1 1 12 13 expr=div(@0,%sqrt_feat_per_head)
F.softmax               softmax     1 1 13 14 dim=%softmax_dim
torch.matmul            op_13       2 1 14 10 15
Tensor.permute          op_14       1 1 15 16 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 16 17 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 17 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_fsmt_attention : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 4 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 5 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 2 3 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_4        1 1 3 6 shape=(%qsize,%batch_mul_num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 4 8 shape=(%kvsize,%batch_mul_num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 5 10 shape=(%kvsize,%batch_mul_num_heads,%feat_per_head)
torch.transpose         op_7        1 1 6 7 dim0=0 dim1=1
torch.transpose         op_8        1 1 8 9 dim0=0 dim1=1
torch.transpose         op_9        1 1 10 11 dim0=0 dim1=1
torch.transpose         op_10       1 1 9 12 dim0=1 dim1=2
torch.bmm               op_11       2 1 7 12 13
F.softmax               softmax     1 1 13 14 dim=%softmax_dim
torch.bmm               op_13       2 1 14 11 15
torch.transpose         op_14       1 1 15 16 dim0=0 dim1=1
Tensor.reshape          op_15       1 1 16 17 shape=(%qsize,%batch,%embed_dim)
nn.Linear               out_proj    1 1 17 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_attention::write(ops, captured_params, captured_attrs);

        Operator* op = ops.at("attn_ht");
        op->params["batch_first"] = false;
    }
};

class fuse_transformers_fsmt_attention_1 : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 8 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 2 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 3 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 8 9 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_4        1 1 2 4 shape=(%kvsize,%batch,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 3 6 shape=(%kvsize,%batch,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 9 10 shape=(%qsize,%batch_mul_num_heads,%feat_per_head)
Tensor.permute          op_7        1 1 4 5 dims=(1,2,0,3)
Tensor.permute          op_8        1 1 6 7 dims=(1,2,0,3)
Tensor.reshape          op_9        1 1 5 12 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.reshape          op_10       1 1 7 13 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
torch.transpose         op_11       1 1 10 11 dim0=0 dim1=1
torch.transpose         op_12       1 1 12 14 dim0=1 dim1=2
torch.bmm               op_13       2 1 11 14 15
F.softmax               softmax     1 1 15 16 dim=%softmax_dim
torch.bmm               op_15       2 1 16 13 17
torch.transpose         op_16       1 1 17 18 dim0=0 dim1=1
Tensor.reshape          op_17       1 1 18 19 shape=(%qsize,%batch,%embed_dim)
nn.Linear               out_proj    1 1 19 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_attention::write(ops, captured_params, captured_attrs);

        Operator* op = ops.at("attn_ht");
        op->params["batch_first"] = false;
    }
};

class fuse_transformers_fsmt_attention_onnx : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
20 19
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 4 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 5 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 2 3 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_4        1 1 3 6 shape=(%qsize,%batch_mul_num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 4 8 shape=(%kvsize,%batch_mul_num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 5 9 shape=(%kvsize,%batch_mul_num_heads,%feat_per_head)
Tensor.permute          op_7        1 1 6 7 dims=(1,0,2)
Tensor.permute          op_8        1 1 9 10 dims=(1,0,2)
Tensor.permute          op_9        1 1 8 11 dims=(1,2,0)
torch.matmul            op_10       2 1 7 11 12
F.softmax               softmax     1 1 12 13 dim=%softmax_dim
torch.matmul            op_12       2 1 13 10 14
Tensor.permute          op_13       1 1 14 15 dims=(1,0,2)
Tensor.reshape          op_14       1 1 15 16 shape=(%qsize,%batch,%embed_dim)
nn.Linear               out_proj    1 1 16 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_attention::write(ops, captured_params, captured_attrs);

        Operator* op = ops.at("attn_ht");
        op->params["batch_first"] = false;
    }
};

class fuse_transformers_fsmt_attention_onnx_1 : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 8 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 2 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 3 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 8 9 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_4        1 1 2 4 shape=(%kvsize,%batch,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 3 6 shape=(%kvsize,%batch,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 9 10 shape=(%qsize,%batch_mul_num_heads,%feat_per_head)
Tensor.permute          op_7        1 1 4 5 dims=(1,2,0,3)
Tensor.permute          op_8        1 1 6 7 dims=(1,2,0,3)
Tensor.permute          op_9        1 1 10 11 dims=(1,0,2)
Tensor.reshape          op_10       1 1 5 12 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.reshape          op_11       1 1 7 13 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.permute          op_12       1 1 12 14 dims=(0,2,1)
torch.matmul            op_13       2 1 11 14 15
F.softmax               softmax     1 1 15 16 dim=%softmax_dim
torch.matmul            op_15       2 1 16 13 17
Tensor.permute          op_16       1 1 17 18 dims=(1,0,2)
Tensor.reshape          op_17       1 1 18 19 shape=(%qsize,%batch,%embed_dim)
nn.Linear               out_proj    1 1 19 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_attention::write(ops, captured_params, captured_attrs);

        Operator* op = ops.at("attn_ht");
        op->params["batch_first"] = false;
    }
};

class fuse_transformers_m2m_100_attention : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 5 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 6 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 2 3 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 5 7 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 6 9 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 3 4 dim0=1 dim1=2
torch.transpose         op_7        1 1 7 8 dim0=1 dim1=2
torch.transpose         op_8        1 1 9 10 dim0=1 dim1=2
torch.transpose         op_9        1 1 8 11 dim0=2 dim1=3
torch.matmul            op_10       2 1 4 11 12
pnnx.Expression         op_11       1 1 12 13 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
F.softmax               softmax     1 1 13 14 dim=-1
torch.matmul            op_13       2 1 14 10 15
torch.transpose         op_14       1 1 15 16 dim0=1 dim1=2
Tensor.reshape          op_15       1 1 16 18 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 18 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_prophet_attention : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 3 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 5 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 8 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 3 4 expr=div(@0,%sqrt_feat_per_head)
Tensor.reshape          op_4        1 1 5 6 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 8 9 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 4 11 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_7        1 1 6 7 dim0=1 dim1=2
torch.transpose         op_8        1 1 9 10 dim0=1 dim1=2
torch.transpose         op_9        1 1 11 12 dim0=1 dim1=2
torch.transpose         op_10       1 1 7 13 dim0=2 dim1=3
torch.einsum            op_11       2 1 12 13 14 equation=ijkm,ijml->ijkl
F.softmax               softmax     1 1 14 15 dim=%softmax_dim
torch.einsum            op_13       2 1 15 10 16 equation=ijkm,ijml->ijkl
torch.transpose         op_14       1 1 16 17 dim0=1 dim1=2
Tensor.reshape          op_15       1 1 17 18 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 18 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_prophet_attention_onnx : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 3 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 5 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 8 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 3 4 expr=div(@0,%sqrt_feat_per_head)
Tensor.reshape          op_4        1 1 5 6 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 8 9 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 4 11 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_7        1 1 6 7 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 9 10 dims=(0,2,1,3)
Tensor.permute          op_9        1 1 11 12 dims=(0,2,1,3)
Tensor.permute          op_10       1 1 7 13 dims=(0,1,3,2)
torch.einsum            op_11       2 1 12 13 14 equation=ijkm,ijml->ijkl
F.softmax               softmax     1 1 14 15 dim=%softmax_dim
torch.einsum            op_13       2 1 15 10 16 equation=ijkm,ijml->ijkl
Tensor.permute          op_14       1 1 16 17 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 17 18 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 18 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_reformer_attention : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
22 21
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 4 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 5 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 6 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 4 7 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 5 9 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 6 11 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 7 8 dim0=2 dim1=1
torch.transpose         op_7        1 1 9 10 dim0=2 dim1=1
torch.transpose         op_8        1 1 11 12 dim0=2 dim1=1
pnnx.Expression         op_9        1 1 10 13 expr=div(@0,%sqrt_feat_per_head)
torch.transpose         op_10       1 1 13 14 dim0=-1 dim1=-2
torch.matmul            op_11       2 1 8 14 15
torch.logsumexp         softmax     1 1 15 16 dim=(%softmax_dim) keepdim=True
pnnx.Expression         op_13       2 1 15 16 17 expr=exp(sub(@0,@1))
torch.matmul            op_14       2 1 17 12 18
Tensor.permute          op_15       1 1 18 19 dims=(0,2,1,3)
Tensor.reshape          op_16       1 1 19 20 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 20 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_reformer_attention_onnx : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
22 21
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 4 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 5 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 6 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 4 7 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 5 9 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 6 11 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 7 8 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 9 10 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 11 12 dims=(0,2,1,3)
pnnx.Expression         op_9        1 1 10 13 expr=div(@0,%sqrt_feat_per_head)
Tensor.permute          op_10       1 1 13 14 dims=(0,1,3,2)
torch.matmul            op_11       2 1 8 14 15
torch.logsumexp         softmax     1 1 15 16 dim=(%softmax_dim) keepdim=True
pnnx.Expression         op_13       2 1 15 16 17 expr=exp(sub(@0,@1))
torch.matmul            op_14       2 1 17 12 18
Tensor.permute          op_15       1 1 18 19 dims=(0,2,1,3)
Tensor.reshape          op_16       1 1 19 20 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 20 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_reformer_attention_onnx_1 : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
22 21
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 4 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 5 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 6 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 4 7 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 5 9 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_7        1 1 6 11 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_4        1 1 7 8 dims=(0,2,1,3)
Tensor.permute          op_6        1 1 9 10 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 11 12 dims=(0,2,1,3)
pnnx.Expression         op_9        1 1 10 13 expr=div(@0,%sqrt_feat_per_head)
Tensor.permute          op_10       1 1 13 14 dims=(0,1,3,2)
torch.matmul            op_11       2 1 8 14 15
torch.logsumexp         softmax     1 1 15 16 dim=%softmax_dim keepdim=True
pnnx.Expression         op_13       2 1 15 16 17 expr=exp(sub(@0,@1))
torch.matmul            op_14       2 1 17 12 18
Tensor.permute          op_15       1 1 18 19 dims=(0,2,1,3)
Tensor.reshape          op_16       1 1 19 20 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 20 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_bart_mask_sdpa_attention : public fuse_transformers_cross_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
18 17
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
nn.Linear               op_0        1 1 query 4 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 7 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 8 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 4 5 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 7 9 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 8 11 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 5 6 dim0=1 dim1=2
torch.transpose         op_7        1 1 9 10 dim0=1 dim1=2
torch.transpose         op_8        1 1 11 12 dim0=1 dim1=2
F.scaled_dot_product_attention sdpa 4 1 6 10 12 mask 13 dropout_p=0.0 is_causal=False scale=%inv_sqrt_embed_dim_per_head
torch.transpose         op_10       1 1 13 14 dim0=1 dim1=2
Tensor.reshape          op_11       1 1 14 16 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 16 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (batch == 1)
        {
            return R"PNNXIR(7767517
7 6
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(1,1,%qsize,%kvsize)f32
Tensor.reshape          attn_ht_0   1 1 mask attn_mask shape=(%qsize,%kvsize)
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
        }

        return R"PNNXIR(7767517
8 7
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
Tensor.expand           attn_ht_0   1 1 mask 18 sizes=(%batch,%num_heads,%qsize,%kvsize)
Tensor.reshape          attn_ht_1   1 1 18 attn_mask
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_mask_attention::write(ops, captured_params, captured_attrs);

        if (batch == 1)
            return;

        const int batch = captured_params.at("batch").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int qsize = captured_params.at("qsize").i;
        const int kvsize = captured_params.at("kvsize").i;

        // set attn_mask shape
        Operator* reshape = ops.at("attn_ht_1");
        reshape->params["shape"] = std::vector<int>{batch * num_heads, qsize, kvsize};
    }
};

class fuse_transformers_clip_mask_attention : public fuse_transformers_cross_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
29 28
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
nn.Linear               op_0        1 1 query 7 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 9 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 12 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 7 8 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_4        1 1 8 15 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 9 10 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 12 13 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_7        1 1 15 16 dim0=1 dim1=2
torch.transpose         op_8        1 1 10 11 dim0=1 dim1=2
torch.transpose         op_9        1 1 13 14 dim0=1 dim1=2
Tensor.reshape          op_10       1 1 16 17 shape=(%batch_mul_num_heads,%qsize,%feat_per_head)
Tensor.reshape          op_11       1 1 11 18 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.reshape          op_12       1 1 14 19 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
torch.transpose         op_13       1 1 18 20 dim0=1 dim1=2
torch.bmm               op_14       2 1 17 20 21
Tensor.reshape          op_15       1 1 21 22 shape=(%batch,%num_heads,%qsize,%kvsize)
pnnx.Expression         op_16       2 1 22 mask 23 expr=add(@0,@1)
Tensor.reshape          op_17       1 1 23 24 shape=(%batch_mul_num_heads,%qsize,%kvsize)
F.softmax               softmax     1 1 24 25 dim=%softmax_dim
torch.bmm               op_19       2 1 25 19 26
Tensor.reshape          op_20       1 1 26 27 shape=(%batch,%num_heads,%qsize,%feat_per_head)
torch.transpose         op_21       1 1 27 28 dim0=1 dim1=2
Tensor.reshape          op_22       1 1 28 29 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 29 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (batch == 1)
        {
            return R"PNNXIR(7767517
7 6
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(1,1,%qsize,%kvsize)f32
Tensor.reshape          attn_ht_0   1 1 mask attn_mask shape=(%qsize,%kvsize)
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
        }

        return R"PNNXIR(7767517
8 7
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
Tensor.expand           attn_ht_0   1 1 mask 18 sizes=(%batch,%num_heads,%qsize,%kvsize)
Tensor.reshape          attn_ht_1   1 1 18 attn_mask shape=(%batch_mul_num_heads,%qsize,%kvsize)
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_clip_mask_attention_2 : public fuse_transformers_cross_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
31 30
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
pnnx.Input              input_cm    0 1 casual_mask #casual_mask=(%batch,1,%qsize,%kvsize)f32
nn.Linear               op_0        1 1 query 7 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 9 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 12 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 7 8 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_4        1 1 8 15 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 9 10 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 12 13 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_7        1 1 15 16 dim0=1 dim1=2
torch.transpose         op_8        1 1 10 11 dim0=1 dim1=2
torch.transpose         op_9        1 1 13 14 dim0=1 dim1=2
Tensor.reshape          op_10       1 1 16 17 shape=(%batch_mul_num_heads,%qsize,%feat_per_head)
Tensor.reshape          op_11       1 1 11 18 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.reshape          op_12       1 1 14 19 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
torch.transpose         op_13       1 1 18 20 dim0=1 dim1=2
torch.bmm               op_14       2 1 17 20 21
Tensor.reshape          op_15       1 1 21 22 shape=(%batch,%num_heads,%qsize,%kvsize)
pnnx.Expression         op_16       2 1 22 mask 223 expr=add(@0,@1)
pnnx.Expression         op_17       2 1 223 casual_mask 23 expr=add(@0,@1)
Tensor.reshape          op_18       1 1 23 24 shape=(%batch_mul_num_heads,%qsize,%kvsize)
F.softmax               softmax     1 1 24 25 dim=%softmax_dim
torch.bmm               op_20       2 1 25 19 26
Tensor.reshape          op_21       1 1 26 27 shape=(%batch,%num_heads,%qsize,%feat_per_head)
torch.transpose         op_22       1 1 27 28 dim0=1 dim1=2
Tensor.reshape          op_23       1 1 28 29 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 29 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (batch == 1)
        {
            return R"PNNXIR(7767517
9 8
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(1,1,%qsize,%kvsize)f32
pnnx.Input              input_cm    0 1 casual_mask #casual_mask=(1,1,%qsize,%kvsize)f32
pnnx.Expression         attn_ht_0   2 1 mask casual_mask 17 expr=add(@0,@1)
Tensor.reshape          attn_ht_1   1 1 17 attn_mask shape=(%qsize,%kvsize)
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
        }

        return R"PNNXIR(7767517
10 9
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
pnnx.Input              input_cm    0 1 casual_mask #casual_mask=(%batch,1,%qsize,%kvsize)f32
pnnx.Expression         attn_ht_0   2 1 mask casual_mask 17 expr=add(@0,@1)
Tensor.expand           attn_ht_1   1 1 17 18 sizes=(%batch,%num_heads,%qsize,%kvsize)
Tensor.reshape          attn_ht_2   1 1 18 attn_mask shape=(%batch_mul_num_heads,%qsize,%kvsize)
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_clip_mask_attention_3 : public fuse_transformers_cross_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
22 21
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
nn.Linear               op_0        1 1 query 9 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 10 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 11 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 9 12 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 10 14 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 11 16 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 12 13 dim0=1 dim1=2
torch.transpose         op_7        1 1 14 15 dim0=1 dim1=2
torch.transpose         op_8        1 1 16 17 dim0=1 dim1=2
torch.transpose         op_9        1 1 15 18 dim0=-1 dim1=-2
torch.matmul            op_10       2 1 13 18 19
pnnx.Expression         op_11       2 1 19 mask 20 expr=add(mul(@0,%inv_sqrt_embed_dim_per_head),@1)
F.softmax               softmax     1 1 20 21 dim=%softmax_dim
torch.matmul            op_13       2 1 21 17 22
torch.transpose         op_14       1 1 22 23 dim0=1 dim1=2
Tensor.reshape          op_16       1 1 23 25 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 25 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (batch == 1)
        {
            return R"PNNXIR(7767517
7 6
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(1,1,%qsize,%kvsize)f32
Tensor.reshape          attn_ht_0   1 1 mask attn_mask shape=(%qsize,%kvsize)
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
        }

        return R"PNNXIR(7767517
8 7
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
Tensor.expand           attn_ht_0   1 1 mask 18 sizes=(%batch,%num_heads,%qsize,%kvsize)
Tensor.reshape          attn_ht_1   1 1 18 attn_mask
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_mask_attention::write(ops, captured_params, captured_attrs);

        if (batch == 1)
            return;

        const int batch = captured_params.at("batch").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int qsize = captured_params.at("qsize").i;
        const int kvsize = captured_params.at("kvsize").i;

        // set attn_mask shape
        Operator* reshape = ops.at("attn_ht_1");
        reshape->params["shape"] = std::vector<int>{batch * num_heads, qsize, kvsize};
    }
};

class fuse_transformers_clip_mask_attention_4 : public fuse_transformers_cross_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
pnnx.Input              input_cm    0 1 casual_mask #casual_mask=(%batch,1,%qsize,%kvsize)f32
nn.Linear               op_0        1 1 query 21 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 22 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 23 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 21 24 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 22 26 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 23 28 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 24 25 dim0=1 dim1=2
torch.transpose         op_7        1 1 26 27 dim0=1 dim1=2
torch.transpose         op_8        1 1 28 29 dim0=1 dim1=2
torch.transpose         op_9        1 1 27 30 dim0=-1 dim1=-2
torch.matmul            op_10       2 1 25 30 31
pnnx.Expression         op_11       3 1 31 mask casual_mask 32 expr=add(mul(@0,%inv_sqrt_embed_dim_per_head),add(@1,@2))
F.softmax               softmax     1 1 32 33 dim=%softmax_dim
torch.matmul            op_13       2 1 33 29 34
torch.transpose         op_14       1 1 34 35 dim0=1 dim1=2
Tensor.reshape          op_15       1 1 35 37 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 37 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (batch == 1)
        {
            return R"PNNXIR(7767517
9 8
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(1,1,%qsize,%kvsize)f32
pnnx.Input              input_cm    0 1 casual_mask #casual_mask=(1,1,%qsize,%kvsize)f32
pnnx.Expression         attn_ht_0   2 1 mask casual_mask 17 expr=add(@0,@1)
Tensor.reshape          attn_ht_1   1 1 17 attn_mask shape=(%qsize,%kvsize)
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
        }

        return R"PNNXIR(7767517
10 9
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
pnnx.Input              input_cm    0 1 casual_mask #casual_mask=(%batch,1,%qsize,%kvsize)f32
pnnx.Expression         attn_ht_0   2 1 mask casual_mask 17 expr=add(@0,@1)
Tensor.expand           attn_ht_1   1 1 17 18 sizes=(%batch,%num_heads,%qsize,%kvsize)
Tensor.reshape          attn_ht_2   1 1 18 attn_mask
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_mask_attention::write(ops, captured_params, captured_attrs);

        if (batch == 1)
            return;

        const int batch = captured_params.at("batch").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int qsize = captured_params.at("qsize").i;
        const int kvsize = captured_params.at("kvsize").i;

        // set attn_mask shape
        Operator* reshape = ops.at("attn_ht_2");
        reshape->params["shape"] = std::vector<int>{batch * num_heads, qsize, kvsize};
    }
};

class fuse_transformers_clip_mask_attention_onnx : public fuse_transformers_clip_mask_attention_3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
nn.Linear               op_0        1 1 query 9 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 10 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 11 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 9 12 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 10 14 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 11 15 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 12 13 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 15 16 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 14 17 dims=(0,2,3,1)
torch.matmul            op_9        2 1 13 17 18
pnnx.Expression         op_10       2 1 18 mask 19 expr=add(mul(@0,%inv_sqrt_embed_dim_per_head),@1)
F.softmax               softmax     1 1 19 20 dim=%softmax_dim
torch.matmul            op_12       2 1 20 16 21
Tensor.permute          op_13       1 1 21 22 dims=(0,2,1,3)
Tensor.reshape          op_14       1 1 22 23 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 23 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_clip_mask_attention_onnx_2 : public fuse_transformers_clip_mask_attention_3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
nn.Linear               op_0        1 1 query 12 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 13 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 14 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 12 15 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 13 17 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 14 18 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 15 16 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 17 20 dims=(0,2,3,1)
Tensor.permute          op_8        1 1 18 19 dims=(0,2,1,3)
pnnx.Expression         op_9        1 1 16 21 expr=mul(@0,%inv_sqrt_sqrt_embed_dim_per_head)
pnnx.Expression         op_10       1 1 20 22 expr=mul(@0,%inv_sqrt_sqrt_embed_dim_per_head)
torch.matmul            op_11       2 1 21 22 23
pnnx.Expression         op_12       2 1 23 mask 24 expr=add(@0,@1)
F.softmax               softmax     1 1 24 25 dim=%softmax_dim
torch.matmul            op_14       2 1 25 19 26
Tensor.permute          op_15       1 1 26 27 dims=(0,2,1,3)
Tensor.reshape          op_16       1 1 27 28 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 28 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_clip_mask_attention_onnx_3 : public fuse_transformers_clip_mask_attention_3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
26 25
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
nn.Linear               op_0        1 1 query 20 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 21 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 22 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 20 23 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 21 25 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 22 27 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 23 24 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 25 26 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 27 28 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 26 29 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.permute          op_10       1 1 29 30 dims=(0,2,1)
Tensor.reshape          op_11       1 1 30 31 shape=(%batch,%num_heads,%feat_per_head,%kvsize)
pnnx.Expression         op_12       1 1 24 32 expr=mul(@0,%inv_sqrt_sqrt_embed_dim_per_head)
pnnx.Expression         op_13       1 1 31 33 expr=mul(@0,%inv_sqrt_sqrt_embed_dim_per_head)
torch.matmul            op_14       2 1 32 33 34
pnnx.Expression         op_15       2 1 34 mask 35 expr=add(@0,@1)
F.softmax               softmax     1 1 35 36 dim=%softmax_dim
torch.matmul            op_17       2 1 36 28 37
Tensor.permute          op_18       1 1 37 38 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 38 39 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 39 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_clip_mask_sdpa_attention : public fuse_transformers_cross_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
18 17
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
nn.Linear               op_0        1 1 query 12 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 13 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 14 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 12 15 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 13 17 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 14 19 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 15 16 dim0=1 dim1=2
torch.transpose         op_7        1 1 17 18 dim0=1 dim1=2
torch.transpose         op_8        1 1 19 20 dim0=1 dim1=2
F.scaled_dot_product_attention sdpa 4 1 16 18 20 mask 24 dropout_p=0.000000e+00 is_causal=False scale=%inv_sqrt_embed_dim_per_head
torch.transpose         op_13       1 1 24 25 dim0=1 dim1=2
Tensor.reshape          op_15       1 1 25 27 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 27 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (batch == 1)
        {
            return R"PNNXIR(7767517
7 6
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(1,1,%qsize,%kvsize)f32
Tensor.reshape          attn_ht_0   1 1 mask attn_mask shape=(%qsize,%kvsize)
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
        }

        return R"PNNXIR(7767517
8 7
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,1,%qsize,%kvsize)f32
Tensor.expand           attn_ht_0   1 1 mask 18 sizes=(%batch,%num_heads,%qsize,%kvsize)
Tensor.reshape          attn_ht_1   1 1 18 attn_mask
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_mask_attention::write(ops, captured_params, captured_attrs);

        if (batch == 1)
            return;

        const int batch = captured_params.at("batch").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int qsize = captured_params.at("qsize").i;
        const int kvsize = captured_params.at("kvsize").i;

        // set attn_mask shape
        Operator* reshape = ops.at("attn_ht_1");
        reshape->params["shape"] = std::vector<int>{batch * num_heads, qsize, kvsize};
    }
};

class fuse_transformers_distilbert_mask_sdpa_attention : public fuse_transformers_cross_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
18 17
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,%kvsize)f32
nn.Linear               op_0        1 1 query 107 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 111 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 115 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 107 109 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 111 113 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 115 117 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 109 q dim0=1 dim1=2
torch.transpose         op_7        1 1 113 k dim0=1 dim1=2
torch.transpose         op_8        1 1 117 v dim0=1 dim1=2
F.scaled_dot_product_attention sdpa 4 1 q k v mask x dropout_p=0.0 is_causal=False
torch.transpose         op_10       1 1 x 120 dim0=1 dim1=2
Tensor.reshape          op_11       1 1 120 a shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 a out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (batch == 1)
        {
            return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 mask
Tensor.expand           attn_ht_0   1 1 mask attn_mask sizes=(%qsize,%kvsize) #attn_mask=(%qsize,%kvsize)f32
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
        }

        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 mask
Tensor.reshape          attn_ht_0   1 1 mask 17 shape=(%batch,1,1,%kvsize) #17=(%batch,1,1,%kvsize)f32
Tensor.expand           attn_ht_1   1 1 17 18 sizes=(%batch,%num_heads,%qsize,%kvsize) #18=(%batch,%num_heads,%qsize,%kvsize)f32
Tensor.reshape          attn_ht_2   1 1 18 attn_mask
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_mask_attention::write(ops, captured_params, captured_attrs);

        if (batch == 1)
            return;

        const int batch = captured_params.at("batch").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int kvsize = captured_params.at("kvsize").i;
        const int qsize = captured_params.at("qsize").i;

        // set attn_mask shape
        Operator* reshape = ops.at("attn_ht_2");
        reshape->params["shape"] = std::vector<int>{batch * num_heads, qsize, kvsize};
    }
};

class fuse_transformers_distilbert_mask_attention_onnx : public fuse_transformers_distilbert_mask_sdpa_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
26 25
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,%kvsize)f32
nn.Linear               op_0        1 1 query 9 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 12 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 15 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 9 10 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 12 13 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 15 16 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 10 11 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 13 14 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 16 17 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 14 18 shape=(%batch_mul_num_heads,%kvsize,%feat_per_head)
Tensor.permute          op_10       1 1 18 19 dims=(0,2,1)
Tensor.reshape          op_11       1 1 19 20 shape=(%batch,%num_heads,%feat_per_head,%kvsize)
pnnx.Expression         op_12       1 1 11 21 expr=mul(@0,%inv_sqrt_sqrt_embed_dim_per_head)
pnnx.Expression         op_13       1 1 20 22 expr=mul(@0,%inv_sqrt_sqrt_embed_dim_per_head)
torch.matmul            op_14       2 1 21 22 23
pnnx.Expression         op_15       2 1 23 mask 24 expr=add(@0,@1)
F.softmax               softmax     1 1 24 25 dim=%softmax_dim
torch.matmul            op_17       2 1 25 17 26
Tensor.permute          op_18       1 1 26 27 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 27 28 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 28 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_flaubert_mask_attention : public fuse_transformers_cross_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask
nn.Linear               op_0        1 1 query 4 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 7 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 10 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 4 5 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 7 8 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 10 11 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 5 6 dim0=1 dim1=2
torch.transpose         op_7        1 1 8 9 dim0=1 dim1=2
torch.transpose         op_8        1 1 11 12 dim0=1 dim1=2
pnnx.Expression         op_9        1 1 6 13 expr=div(@0,%sqrt_feat_per_head)
torch.transpose         op_10       1 1 9 14 dim0=2 dim1=3
torch.matmul            op_11       2 1 13 14 15
Tensor.reshape          op_12       1 1 mask 17 shape=(%batch,1,%qsize,%kvsize)
Tensor.expand_as        op_13       2 1 17 15 18
Tensor.masked_fill      op_14       2 1 15 18 19 value=-3.402823e+38
F.softmax               softmax     1 1 19 20 dim=%softmax_dim
torch.matmul            op_16       2 1 20 12 21
torch.transpose         op_17       1 1 21 22 dim0=1 dim1=2
Tensor.reshape          op_18       1 1 22 23 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 23 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (batch == 1)
        {
            return R"PNNXIR(7767517
7 6
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask
Tensor.reshape          attn_ht_0   1 1 mask attn_mask shape=(%qsize,%kvsize)
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
        }

        return R"PNNXIR(7767517
9 8
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask
Tensor.reshape          attn_ht_0   1 1 mask 17 shape=(%batch,1,%qsize,%kvsize) #17=(%batch,1,%qsize,%kvsize)bool
Tensor.expand           attn_ht_1   1 1 17 18 sizes=(%batch,%num_heads,%qsize,%kvsize) #18=(%batch,%num_heads,%qsize,%kvsize)bool
Tensor.reshape          attn_ht_2   1 1 18 attn_mask
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_mask_attention::write(ops, captured_params, captured_attrs);

        if (batch == 1)
            return;

        const int batch = captured_params.at("batch").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int qsize = captured_params.at("qsize").i;
        const int kvsize = captured_params.at("kvsize").i;

        // set attn_mask shape
        Operator* reshape = ops.at("attn_ht_2");
        reshape->params["shape"] = std::vector<int>{batch * num_heads, qsize, kvsize};
    }
};

class fuse_transformers_flaubert_mask_attention_onnx : public fuse_transformers_cross_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
22 21
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask
nn.Linear               op_0        1 1 query 4 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 7 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 9 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 4 5 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 7 8 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 9 10 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 5 6 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 10 11 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 8 13 dims=(0,2,3,1)
pnnx.Expression         op_9        1 1 6 12 expr=div(@0,%sqrt_feat_per_head)
torch.matmul            op_10       2 1 12 13 14
torch.where             op_11       2 1 mask 14 18 input=-3.402823e+38
F.softmax               softmax     1 1 18 19 dim=%softmax_dim
torch.matmul            op_13       2 1 19 11 20
Tensor.permute          op_14       1 1 20 21 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 21 22 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 22 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #18=(%batch,%num_heads,%qsize,%kvsize)bool
Tensor.reshape          attn_ht_0   1 1 mask attn_mask
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_mask_attention::write(ops, captured_params, captured_attrs);

        const int batch = captured_params.at("batch").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int qsize = captured_params.at("qsize").i;
        const int kvsize = captured_params.at("kvsize").i;

        // set attn_mask shape
        Operator* reshape = ops.at("attn_ht_0");
        reshape->params["shape"] = std::vector<int>{batch * num_heads, qsize, kvsize};
    }
};

class fuse_transformers_prophet_mask_attention : public fuse_transformers_cross_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,%num_heads,1,%qsize)f32
nn.Linear               op_0        1 1 query 4 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 6 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 9 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 4 5 expr=div(@0,%sqrt_feat_per_head)
Tensor.reshape          op_4        1 1 6 7 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 9 10 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 5 12 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_7        1 1 7 8 dim0=1 dim1=2
torch.transpose         op_8        1 1 10 11 dim0=1 dim1=2
torch.transpose         op_9        1 1 12 13 dim0=1 dim1=2
torch.transpose         op_10       1 1 8 14 dim0=2 dim1=3
torch.einsum            op_11       2 1 13 14 15 equation=ijkm,ijml->ijkl
pnnx.Expression         op_12       2 1 15 mask 16 expr=add(@0,@1)
F.softmax               softmax     1 1 16 17 dim=%softmax_dim
torch.einsum            op_14       2 1 17 11 18 equation=ijkm,ijml->ijkl
torch.transpose         op_15       1 1 18 19 dim0=1 dim1=2
Tensor.reshape          op_16       1 1 19 20 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 20 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask
Tensor.expand           attn_ht_0   1 1 mask 18 sizes=(%batch,%num_heads,%qsize,%kvsize) #18=(%batch,%num_heads,%qsize,%kvsize)f32
Tensor.reshape          attn_ht_1   1 1 18 attn_mask
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_mask_attention::write(ops, captured_params, captured_attrs);

        const int batch = captured_params.at("batch").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int qsize = captured_params.at("qsize").i;
        const int kvsize = captured_params.at("kvsize").i;

        // set attn_mask shape
        Operator* reshape = ops.at("attn_ht_1");
        reshape->params["shape"] = std::vector<int>{batch * num_heads, qsize, kvsize};
    }
};

class fuse_transformers_prophet_mask_attention_onnx : public fuse_transformers_prophet_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
pnnx.Input              input_m     0 1 mask #mask=(%batch,%num_heads,1,%qsize)f32
nn.Linear               op_0        1 1 query 4 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 6 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 9 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 4 5 expr=div(@0,%sqrt_feat_per_head)
Tensor.reshape          op_4        1 1 6 7 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 9 10 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 5 12 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_7        1 1 7 8 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 10 11 dims=(0,2,1,3)
Tensor.permute          op_9        1 1 12 13 dims=(0,2,1,3)
Tensor.permute          op_10       1 1 8 14 dims=(0,1,3,2)
torch.einsum            op_11       2 1 13 14 15 equation=ijkm,ijml->ijkl
pnnx.Expression         op_12       2 1 15 mask 16 expr=add(@0,@1)
F.softmax               softmax     1 1 16 17 dim=%softmax_dim
torch.einsum            op_14       2 1 17 11 18 equation=ijkm,ijml->ijkl
Tensor.permute          op_15       1 1 18 19 dims=(0,2,1,3)
Tensor.reshape          op_16       1 1 19 20 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 20 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_xlm_mask_attention : public fuse_transformers_cross_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 mask
nn.Linear               op_0        1 1 query 5 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 8 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 11 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 5 6 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 8 9 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 11 12 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 6 7 dim0=1 dim1=2
torch.transpose         op_7        1 1 9 10 dim0=1 dim1=2
torch.transpose         op_8        1 1 12 13 dim0=1 dim1=2
pnnx.Expression         op_9        1 1 7 14 expr=div(@0,%sqrt_feat_per_head)
torch.transpose         op_10       1 1 10 15 dim0=2 dim1=3
torch.matmul            op_11       2 1 14 15 16
Tensor.reshape          op_12       1 1 mask 18 shape=(%batch,1,1,%kvsize)
Tensor.expand_as        op_13       2 1 18 16 19
Tensor.masked_fill      op_14       2 1 16 19 20 value=-3.402823e+38
F.softmax               softmax     1 1 20 21 dim=%softmax_dim
torch.matmul            op_16       2 1 21 13 22
torch.transpose         op_17       1 1 22 23 dim0=1 dim1=2
Tensor.reshape          op_18       1 1 23 24 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 24 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (batch == 1)
        {
            return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 mask
Tensor.reshape          attn_ht_0   1 1 mask 17 shape=(1,%kvsize) #17=(1,%kvsize)bool
Tensor.expand           attn_ht_1   1 1 17 attn_mask sizes=(%qsize,%kvsize) #attn_mask=(%qsize,%kvsize)bool
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
        }

        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 mask
Tensor.reshape          attn_ht_0   1 1 mask 17 shape=(%batch,1,1,%kvsize) #17=(%batch,1,1,%kvsize)bool
Tensor.expand           attn_ht_1   1 1 17 18 sizes=(%batch,%num_heads,%qsize,%kvsize) #18=(%batch,%num_heads,%qsize,%kvsize)bool
Tensor.reshape          attn_ht_2   1 1 18 attn_mask
nn.MultiheadAttention   attn_ht     4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_mask_attention::write(ops, captured_params, captured_attrs);

        if (batch == 1)
            return;

        const int batch = captured_params.at("batch").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int kvsize = captured_params.at("kvsize").i;
        const int qsize = captured_params.at("qsize").i;

        // set attn_mask shape
        Operator* reshape = ops.at("attn_ht_2");
        reshape->params["shape"] = std::vector<int>{batch * num_heads, qsize, kvsize};
    }
};

void fuse_transformers_multiheadattention(Graph& graph)
{
#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 9)
    fuse_transformers_albert_attention a;
    fuse_transformers_albert_attention_1 a1;
    fuse_transformers_albert_attention_onnx a2;
    fuse_transformers_bart_attention b;
    fuse_transformers_bart_attention_2 b2;
    fuse_transformers_bart_attention_onnx b3;
    fuse_transformers_bart_attention_onnx_2 b4;
    fuse_transformers_bart_sdpa_attention b5;
    fuse_transformers_bart_sdpa_attention_3 b7;
    fuse_transformers_bart_sdpa_attention_onnx b8;
    fuse_transformers_bart_sdpa_attention_onnx_1 b9;
    fuse_transformers_bert_attention bx;
    fuse_transformers_clip_attention y;
    fuse_transformers_clip_attention_2 y2;
    fuse_transformers_clip_attention_onnx_2 y3;
    fuse_transformers_clip_sdpa_attention y4;
    fuse_transformers_chinese_clip_attention z;
    fuse_transformers_chinese_clip_attention_1 z1;
    fuse_transformers_chinese_clip_attention_onnx z2;
    fuse_transformers_ctrl_attention c;
    fuse_transformers_fsmt_attention d;
    fuse_transformers_fsmt_attention_1 d1;
    fuse_transformers_fsmt_attention_onnx d2;
    fuse_transformers_fsmt_attention_onnx_1 d3;
    fuse_transformers_m2m_100_attention e;
    fuse_transformers_prophet_attention e1;
    fuse_transformers_prophet_attention_onnx e2;
    fuse_transformers_reformer_attention f;
    fuse_transformers_reformer_attention_onnx f2;
    fuse_transformers_reformer_attention_onnx_1 f3;

    fuse_transformers_bart_mask_sdpa_attention my0;
    fuse_transformers_clip_mask_attention my;
    fuse_transformers_clip_mask_attention_2 my2;
    fuse_transformers_clip_mask_attention_3 my3;
    fuse_transformers_clip_mask_attention_4 my4;
    fuse_transformers_clip_mask_attention_onnx my5;
    fuse_transformers_clip_mask_attention_onnx_2 my6;
    fuse_transformers_clip_mask_attention_onnx_3 my7;
    fuse_transformers_clip_mask_sdpa_attention my8;
    fuse_transformers_distilbert_mask_sdpa_attention myx;
    fuse_transformers_distilbert_mask_attention_onnx myx1;
    fuse_transformers_flaubert_mask_attention ma;
    fuse_transformers_flaubert_mask_attention_onnx ma2;
    fuse_transformers_prophet_mask_attention me;
    fuse_transformers_prophet_mask_attention_onnx me2;
    fuse_transformers_xlm_mask_attention mf;

    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &a1, opindex);
    pnnx_graph_rewrite(graph, &a2, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &b2, opindex);
    pnnx_graph_rewrite(graph, &b3, opindex);
    pnnx_graph_rewrite(graph, &b4, opindex);
    pnnx_graph_rewrite(graph, &b5, opindex);
    pnnx_graph_rewrite(graph, &b7, opindex);
    pnnx_graph_rewrite(graph, &b8, opindex);
    pnnx_graph_rewrite(graph, &b9, opindex);
    pnnx_graph_rewrite(graph, &bx, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
    pnnx_graph_rewrite(graph, &d, opindex);
    pnnx_graph_rewrite(graph, &d1, opindex);
    pnnx_graph_rewrite(graph, &d2, opindex);
    pnnx_graph_rewrite(graph, &d3, opindex);
    pnnx_graph_rewrite(graph, &e, opindex);
    pnnx_graph_rewrite(graph, &e1, opindex);
    pnnx_graph_rewrite(graph, &e2, opindex);
    pnnx_graph_rewrite(graph, &f, opindex);
    pnnx_graph_rewrite(graph, &f2, opindex);
    pnnx_graph_rewrite(graph, &f3, opindex);

    pnnx_graph_rewrite(graph, &y, opindex);
    pnnx_graph_rewrite(graph, &y2, opindex);
    pnnx_graph_rewrite(graph, &y3, opindex);
    pnnx_graph_rewrite(graph, &y4, opindex);
    pnnx_graph_rewrite(graph, &z, opindex);
    pnnx_graph_rewrite(graph, &z1, opindex);
    pnnx_graph_rewrite(graph, &z2, opindex);

    pnnx_graph_rewrite(graph, &ma, opindex);
    pnnx_graph_rewrite(graph, &ma2, opindex);
    pnnx_graph_rewrite(graph, &me, opindex);
    pnnx_graph_rewrite(graph, &me2, opindex);
    pnnx_graph_rewrite(graph, &mf, opindex);

    pnnx_graph_rewrite(graph, &my0, opindex);
    pnnx_graph_rewrite(graph, &my, opindex);
    pnnx_graph_rewrite(graph, &my2, opindex);
    pnnx_graph_rewrite(graph, &my3, opindex);
    pnnx_graph_rewrite(graph, &my4, opindex);
    pnnx_graph_rewrite(graph, &my5, opindex);
    pnnx_graph_rewrite(graph, &my6, opindex);
    pnnx_graph_rewrite(graph, &my7, opindex);
    pnnx_graph_rewrite(graph, &my8, opindex);
    pnnx_graph_rewrite(graph, &myx, opindex);
    pnnx_graph_rewrite(graph, &myx1, opindex);
#endif
}

} // namespace pnnx
