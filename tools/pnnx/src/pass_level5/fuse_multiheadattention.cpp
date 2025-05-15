// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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
nn.MultiheadAttention   attention   1 1 input out embed_dim=%embed_dim kdim=%embed_dim vdim=%embed_dim batch_first=True add_zero_attn=False add_bias_kv=False
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

        if (captured_params.find("softmax_dim") != captured_params.end())
        {
            const int softmax_dim = captured_params.at("softmax_dim").i;
            int softmax_input_rank = (int)matched_operators.at("softmax")->inputs[0]->shape.size();
            if (softmax_dim != -1 && softmax_dim != softmax_input_rank - 1)
                return false;
        }

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Operator* op = ops.at("attention");

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
nn.MultiheadAttention   attention   3 1 query key value out embed_dim=%embed_dim kdim=%kdim vdim=%vdim batch_first=True add_zero_attn=False add_bias_kv=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Operator* op = ops.at("attention");

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
nn.MultiheadAttention   attention   2 1 input attn_mask out embed_dim=%embed_dim kdim=%embed_dim vdim=%embed_dim batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
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
nn.MultiheadAttention   attention   4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_albert_attention : public fuse_transformers_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
19 18
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 256 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 257 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 260 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
Tensor.view             op_3        1 1 256 263 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_4        1 1 257 258 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 260 261 shape=(%batch,%size,%num_heads,%feat_per_head)
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

class fuse_transformers_bart_attention : public fuse_transformers_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 4 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 6 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 2 3 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.view             op_4        1 1 3 8 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 4 5 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_6        1 1 6 7 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.transpose         op_7        1 1 8 9 dim0=1 dim1=2
torch.transpose         op_8        1 1 5 10 dim0=1 dim1=2
torch.transpose         op_9        1 1 7 11 dim0=1 dim1=2
Tensor.contiguous       op_8_       1 1 10 10_ memory_format=*
Tensor.contiguous       op_9_       1 1 11 11_ memory_format=*
Tensor.reshape          op_10       1 1 9 14 shape=(%batch_mul_num_heads,%size,%feat_per_head)
Tensor.reshape          op_11       1 1 10_ 12 shape=(%batch_mul_num_heads,%size,%feat_per_head)
Tensor.reshape          op_12       1 1 11_ 17 shape=(%batch_mul_num_heads,%size,%feat_per_head)
torch.transpose         op_13       1 1 12 13 dim0=1 dim1=2
torch.bmm               op_14       2 1 14 13 15
F.softmax               softmax     1 1 15 16 dim=%softmax_dim
torch.bmm               op_16       2 1 16 17 18
Tensor.view             op_17       1 1 18 19 shape=(%batch,%num_heads,%size,%feat_per_head)
torch.transpose         op_18       1 1 19 20 dim0=1 dim1=2
Tensor.reshape          op_19       1 1 20 21 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_bart_sdpa_attention : public fuse_transformers_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
18 17
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 4 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 6 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
Tensor.view             op_4        1 1 2 8 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 4 5 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_6        1 1 6 7 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.transpose         op_7        1 1 8 9 dim0=1 dim1=2
torch.transpose         op_8        1 1 5 10 dim0=1 dim1=2
torch.transpose         op_9        1 1 7 11 dim0=1 dim1=2
Tensor.contiguous       op_7_       1 1 9 9_ memory_format=*
Tensor.contiguous       op_8_       1 1 10 10_ memory_format=*
Tensor.contiguous       op_9_       1 1 11 11_ memory_format=*
F.scaled_dot_product_attention sdpa 3 1 9_ 10_ 11_ 19 attn_mask=None dropout_p=0.000000e+00 is_causal=False
torch.transpose         op_18       1 1 19 20 dim0=1 dim1=2
Tensor.reshape          op_19       1 1 20 21 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_clip_attention : public fuse_transformers_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 4 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 6 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 2 3 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.view             op_4        1 1 3 8 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 4 5 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_6        1 1 6 7 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.transpose         op_7        1 1 8 9 dim0=1 dim1=2
torch.transpose         op_8        1 1 5 10 dim0=1 dim1=2
torch.transpose         op_9        1 1 7 11 dim0=1 dim1=2
Tensor.reshape          op_10       1 1 9 14 shape=(%batch_mul_num_heads,%size,%feat_per_head)
Tensor.reshape          op_11       1 1 10 12 shape=(%batch_mul_num_heads,%size,%feat_per_head)
Tensor.reshape          op_12       1 1 11 17 shape=(%batch_mul_num_heads,%size,%feat_per_head)
torch.transpose         op_13       1 1 12 13 dim0=1 dim1=2
torch.bmm               op_14       2 1 14 13 15
F.softmax               softmax     1 1 15 16 dim=%softmax_dim
torch.bmm               op_16       2 1 16 17 18
Tensor.view             op_17       1 1 18 19 shape=(%batch,%num_heads,%size,%feat_per_head)
torch.transpose         op_18       1 1 19 20 dim0=1 dim1=2
Tensor.reshape          op_19       1 1 20 21 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_chinese_clip_attention : public fuse_transformers_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
19 18
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 256 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 257 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 260 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
Tensor.view             op_3        1 1 256 263 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_4        1 1 257 258 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 260 261 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 263 264 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 258 259 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 261 262 dims=(0,2,1,3)
torch.transpose         op_9        1 1 259 265 dim0=-1 dim1=-2
torch.matmul            op_10       2 1 264 265 266
pnnx.Expression         op_11       1 1 266 267 expr=div(@0,%sqrt_feat_per_head)
F.softmax               softmax     1 1 267 268 dim=%softmax_dim
torch.matmul            op_13       2 1 268 262 269
Tensor.permute          op_14       1 1 269 270 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 270 271 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 271 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_ctrl_attention : public fuse_transformers_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
19 18
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 3 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 4 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 2 5 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 3 7 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 4 9 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 5 6 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 7 8 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 9 10 dims=(0,2,1,3)
Tensor.permute          op_9        1 1 8 11 dims=(0,1,3,2)
torch.matmul            op_10       2 1 6 11 12
pnnx.Expression         op_11       1 1 12 13 expr=div(@0,%sqrt_feat_per_head)
F.softmax               softmax     1 1 13 14 dim=%softmax_dim
torch.matmul            op_13       2 1 14 10 15
Tensor.permute          op_14       1 1 15 16 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 16 17 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 17 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_fsmt_attention : public fuse_transformers_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
19 18
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 2 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 4 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 5 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 2 3 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_4        1 1 3 6 shape=(%size,%batch_mul_num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 4 8 shape=(%size,%batch_mul_num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 5 10 shape=(%size,%batch_mul_num_heads,%feat_per_head)
torch.transpose         op_7        1 1 6 7 dim0=0 dim1=1
torch.transpose         op_8        1 1 8 9 dim0=0 dim1=1
torch.transpose         op_9        1 1 10 11 dim0=0 dim1=1
torch.transpose         op_10       1 1 9 12 dim0=1 dim1=2
torch.bmm               op_11       2 1 7 12 13
F.softmax               softmax     1 1 13 14 dim=%softmax_dim
torch.bmm               op_13       2 1 14 11 15
torch.transpose         op_14       1 1 15 16 dim0=0 dim1=1
Tensor.reshape          op_15       1 1 16 17 shape=(%size,%batch,%embed_dim)
nn.Linear               out_proj    1 1 17 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_attention::write(ops, captured_params, captured_attrs);

        Operator* op = ops.at("attention");
        op->params["batch_first"] = false;
    }
};

class fuse_transformers_prophet_attention : public fuse_transformers_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
19 18
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 3 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 5 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 8 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 3 4 expr=div(@0,%sqrt_feat_per_head)
Tensor.view             op_4        1 1 5 6 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 8 9 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_6        1 1 4 11 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.transpose         op_7        1 1 6 7 dim0=1 dim1=2
torch.transpose         op_8        1 1 9 10 dim0=1 dim1=2
torch.transpose         op_9        1 1 11 12 dim0=1 dim1=2
torch.transpose         op_10       1 1 7 13 dim0=2 dim1=3
torch.einsum            op_11       2 1 12 13 14 equation=ijkm,ijml->ijkl
F.softmax               softmax     1 1 14 15 dim=%softmax_dim
torch.einsum            op_13       2 1 15 10 16 equation=ijkm,ijml->ijkl
torch.transpose         op_14       1 1 16 17 dim0=1 dim1=2
Tensor.reshape          op_15       1 1 17 18 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 18 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_reformer_attention : public fuse_transformers_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
20 19
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 4 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 5 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 6 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
Tensor.view             op_3        1 1 4 7 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_4        1 1 5 9 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 6 11 shape=(%batch,%size,%num_heads,%feat_per_head)
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
Tensor.reshape          op_16       1 1 19 20 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 20 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_lxmert_cross_attention : public fuse_transformers_cross_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 6 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 7 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 8 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.view             op_3        1 1 6 9 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.view             op_4        1 1 7 11 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 8 13 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.permute          op_6        1 1 9 10 dims=(0,2,1,3)
Tensor.permute          op_7        1 1 11 12 dims=(0,2,1,3)
Tensor.permute          op_8        1 1 13 14 dims=(0,2,1,3)
torch.transpose         op_9        1 1 12 15 dim0=-1 dim1=-2
torch.matmul            op_10       2 1 10 15 16
pnnx.Expression         op_11       1 1 16 17 expr=div(@0,%sqrt_feat_per_head)
F.softmax               softmax     1 1 17 18 dim=%softmax_dim
torch.matmul            op_13       2 1 18 14 19
Tensor.permute          op_14       1 1 19 20 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 20 21 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_flaubert_mask_attention : public fuse_transformers_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mask
nn.Linear               op_0        1 1 input 4 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 7 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 10 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
Tensor.view             op_3        1 1 4 5 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_4        1 1 7 8 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 10 11 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 5 6 dim0=1 dim1=2
torch.transpose         op_7        1 1 8 9 dim0=1 dim1=2
torch.transpose         op_8        1 1 11 12 dim0=1 dim1=2
pnnx.Expression         op_9        1 1 6 13 expr=div(@0,%sqrt_feat_per_head)
torch.transpose         op_10       1 1 9 14 dim0=2 dim1=3
torch.matmul            op_11       2 1 13 14 15
Tensor.view             op_12       1 1 mask 17 shape=(%batch,1,%size,%size)
Tensor.expand_as        op_13       2 1 17 15 18
Tensor.masked_fill      op_14       2 1 15 18 19 value=-3.402823e+38
F.softmax               softmax     1 1 19 20 dim=%softmax_dim
torch.matmul            op_16       2 1 20 12 21
torch.transpose         op_17       1 1 21 22 dim0=1 dim1=2
Tensor.reshape          op_18       1 1 22 23 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 23 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mask
Tensor.view             attention_0 1 1 mask 17 shape=(%batch,1,%size,%size) #17=(%batch,1,%size,%size)bool
Tensor.expand           attention_1 1 1 17 18 shape=(%batch,%num_heads,%size,%size) #18=(%batch,%num_heads,%size,%size)bool
Tensor.reshape          attention_2 1 1 18 attn_mask
nn.MultiheadAttention   attention   2 1 input attn_mask out embed_dim=%embed_dim kdim=%embed_dim vdim=%embed_dim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_mask_attention::write(ops, captured_params, captured_attrs);

        const int batch = captured_params.at("batch").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int size = captured_params.at("size").i;

        // set attn_mask shape
        Operator* reshape = ops.at("attention_2");
        reshape->params["shape"] = std::vector<int>{batch * num_heads, size, size};
    }
};

class fuse_transformers_prophet_mask_attention : public fuse_transformers_mask_attention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mask #mask=(%batch,%num_heads,1,%size)f32
nn.Linear               op_0        1 1 input 4 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 6 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 9 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 4 5 expr=div(@0,%sqrt_feat_per_head)
Tensor.view             op_4        1 1 6 7 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 9 10 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_6        1 1 5 12 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.transpose         op_7        1 1 7 8 dim0=1 dim1=2
torch.transpose         op_8        1 1 10 11 dim0=1 dim1=2
torch.transpose         op_9        1 1 12 13 dim0=1 dim1=2
torch.transpose         op_10       1 1 8 14 dim0=2 dim1=3
torch.einsum            op_11       2 1 13 14 15 equation=ijkm,ijml->ijkl
pnnx.Expression         op_12       2 1 15 mask 16 expr=add(@0,@1)
F.softmax               softmax     1 1 16 17 dim=%softmax_dim
torch.einsum            op_14       2 1 17 11 18 equation=ijkm,ijml->ijkl
torch.transpose         op_15       1 1 18 19 dim0=1 dim1=2
Tensor.reshape          op_16       1 1 19 20 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 20 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mask
Tensor.expand           attention_0 1 1 mask 18 shape=(%batch,%num_heads,%size,%size) #18=(%batch,%num_heads,%size,%size)f32
Tensor.reshape          attention_1 1 1 18 attn_mask
nn.MultiheadAttention   attention   2 1 input attn_mask out embed_dim=%embed_dim kdim=%embed_dim vdim=%embed_dim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_mask_attention::write(ops, captured_params, captured_attrs);

        const int batch = captured_params.at("batch").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int size = captured_params.at("size").i;

        // set attn_mask shape
        Operator* reshape = ops.at("attention_1");
        reshape->params["shape"] = std::vector<int>{batch * num_heads, size, size};
    }
};

class fuse_transformers_xlm_cross_mask_attention : public fuse_transformers_cross_mask_attention
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
Tensor.view             op_3        1 1 5 6 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.view             op_4        1 1 8 9 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 11 12 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 6 7 dim0=1 dim1=2
torch.transpose         op_7        1 1 9 10 dim0=1 dim1=2
torch.transpose         op_8        1 1 12 13 dim0=1 dim1=2
pnnx.Expression         op_9        1 1 7 14 expr=div(@0,%sqrt_feat_per_head)
torch.transpose         op_10       1 1 10 15 dim0=2 dim1=3
torch.matmul            op_11       2 1 14 15 16
Tensor.view             op_12       1 1 mask 18 shape=(%batch,1,1,%qsize)
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
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 mask
Tensor.view             attention_0 1 1 mask 17 shape=(%batch,1,1,%qsize) #17=(%batch,1,1,%qsize)bool
Tensor.expand           attention_1 1 1 17 18 shape=(%batch,%num_heads,%kvsize,%qsize) #18=(%batch,%num_heads,%kvsize,%qsize)bool
Tensor.reshape          attention_2 1 1 18 attn_mask
nn.MultiheadAttention   attention   4 1 query key value attn_mask out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_transformers_cross_mask_attention::write(ops, captured_params, captured_attrs);

        const int batch = captured_params.at("batch").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int kvsize = captured_params.at("kvsize").i;
        const int qsize = captured_params.at("qsize").i;

        // set attn_mask shape
        Operator* reshape = ops.at("attention_2");
        reshape->params["shape"] = std::vector<int>{batch * num_heads, kvsize, qsize};
    }
};

void fuse_multiheadattention(Graph& graph)
{
#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 9)
    fuse_transformers_albert_attention a;
    fuse_transformers_bart_attention b;
    fuse_transformers_bart_sdpa_attention b2;
    fuse_transformers_clip_attention y;
    fuse_transformers_chinese_clip_attention z;
    fuse_transformers_ctrl_attention c;
    fuse_transformers_fsmt_attention d;
    fuse_transformers_prophet_attention e;
    fuse_transformers_reformer_attention f;

    fuse_transformers_lxmert_cross_attention ca;

    fuse_transformers_flaubert_mask_attention ma;
    fuse_transformers_prophet_mask_attention me;

    fuse_transformers_xlm_cross_mask_attention cma;

    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &b2, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
    pnnx_graph_rewrite(graph, &d, opindex);
    pnnx_graph_rewrite(graph, &e, opindex);
    pnnx_graph_rewrite(graph, &f, opindex);

    pnnx_graph_rewrite(graph, &y, opindex);
    pnnx_graph_rewrite(graph, &z, opindex);

    pnnx_graph_rewrite(graph, &ca, opindex);

    pnnx_graph_rewrite(graph, &ma, opindex);
    pnnx_graph_rewrite(graph, &me, opindex);

    pnnx_graph_rewrite(graph, &cma, opindex);
#endif
}

} // namespace pnnx
