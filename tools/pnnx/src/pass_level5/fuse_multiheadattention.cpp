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

class fuse_multiheadattention_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
14 13
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 1 bias=%qkvbias in_features=%embed_dim out_features=%qkv_out_features @bias @weight
Tensor.reshape          op_1        1 1 1 2 shape=(%batch,%size,3,%num_heads,%feat_per_head)
torch.permute           op_2        1 1 2 3 dims=(2,0,3,1,4)
torch.unbind            op_3        1 3 3 4 5 6 dim=0
pnnx.Expression         op_4        1 1 4 7 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
torch.permute           op_5        1 1 5 8 dims=(0,1,3,2)
torch.matmul            op_6        2 1 7 8 9
F.softmax               op_7        1 1 9 10 dim=-1
torch.matmul            op_8        2 1 10 6 11
torch.permute           op_9        1 1 11 12 dims=(0,2,1,3)
Tensor.reshape          op_10       1 1 12 13 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 13 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.MultiheadAttention   attention   1 1 input out embed_dim=%embed_dim kdim=%embed_dim vdim=%embed_dim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int embed_dim = captured_params.at("embed_dim").i;
        const int qkv_out_features = captured_params.at("qkv_out_features").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int feat_per_head = captured_params.at("feat_per_head").i;
        const float inv_sqrt_embed_dim_per_head = captured_params.at("inv_sqrt_embed_dim_per_head").f;

        if (qkv_out_features != embed_dim * 3)
            return false;

        if (embed_dim != num_heads * feat_per_head)
            return false;

        if (!NearlyEqual(inv_sqrt_embed_dim_per_head, 1.f / sqrt(feat_per_head), 0.001))
            return false;

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Operator* op = ops.at("attention");

        const int embed_dim = captured_params.at("embed_dim").i;
        const bool qkvbias = captured_params.at("qkvbias").b;
        const bool outbias = captured_params.at("outbias").b;
        const bool bias = qkvbias || outbias;

        op->params["bias"] = bias;

        op->attrs["in_proj_weight"] = captured_attrs.at("op_0.weight");
        if (bias)
        {
            if (qkvbias)
            {
                op->attrs["in_proj_bias"] = captured_attrs.at("op_0.bias");
            }
            else
            {
                // init bias as zero
                op->attrs["in_proj_bias"] = Attribute();
                op->attrs["in_proj_bias"].type = op->attrs["in_proj_weight"].type;
                op->attrs["in_proj_bias"].shape = {embed_dim * 3};
                op->attrs["in_proj_bias"].set_float32_data(std::vector<float>(embed_dim * 3, 0.f));
            }
        }

        op->attrs["out_proj.weight"] = captured_attrs.at("out_proj.weight");
        if (bias)
        {
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

class fuse_multiheadattention_pass_11 : public fuse_multiheadattention_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
18 17
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 1 bias=%qkvbias in_features=%embed_dim out_features=%qkv_out_features @bias @weight
torch.chunk             op_1        1 3 1 2 3 4 chunks=3 dim=-1
Tensor.reshape          op_2        1 1 2 5 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_3        1 1 3 6 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 4 7 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.permute           op_5        1 1 6 8 dims=(0,2,1,3)
torch.permute           op_6        1 1 5 9 dims=(0,2,1,3)
torch.transpose         op_7        1 1 8 10 dim0=-1 dim1=-2
torch.matmul            op_8        2 1 9 10 11
pnnx.Expression         op_9        1 1 11 12 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
nn.Softmax              op_10       1 1 12 13 dim=-1
torch.permute           op_11       1 1 7 14 dims=(0,2,1,3)
torch.matmul            op_12       2 1 13 14 15
torch.permute           op_13       1 1 15 16 dims=(0,2,1,3)
Tensor.reshape          op_14       1 1 16 17 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 17 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_sameqkv : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 31 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 32 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 33 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 32 34 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_4        1 1 31 35 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 34 36 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 33 37 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.permute           op_7        1 1 36 38 dims=(0,2,1,3)
Tensor.reshape          op_8        1 1 38 39 shape=(%num_heads,%size,%feat_per_head)
torch.permute           op_9        1 1 35 40 dims=(0,2,1,3)
Tensor.reshape          op_10       1 1 40 41 shape=(%num_heads,%size,%feat_per_head)
torch.permute           op_11       1 1 39 42 dims=(0,2,1)
torch.matmul            op_12       2 1 41 42 43
F.softmax               op_13       1 1 43 44 dim=-1
torch.permute           op_14       1 1 37 45 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 45 46 shape=(%num_heads,%size,%feat_per_head)
torch.matmul            op_16       2 1 44 46 47
Tensor.reshape          op_17       1 1 47 48 shape=(%batch,%num_heads,%size,%feat_per_head)
torch.permute           op_18       1 1 48 49 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 49 50 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 50 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.MultiheadAttention   attention   1 1 input out embed_dim=%embed_dim kdim=%embed_dim vdim=%embed_dim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int embed_dim = captured_params.at("embed_dim").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int feat_per_head = captured_params.at("feat_per_head").i;
        const float inv_sqrt_embed_dim_per_head = captured_params.at("inv_sqrt_embed_dim_per_head").f;

        if (embed_dim != num_heads * feat_per_head)
            return false;

        if (!NearlyEqual(inv_sqrt_embed_dim_per_head, 1.f / sqrt(feat_per_head), 0.001))
            return false;

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Operator* op = ops.at("attention");

        const int embed_dim = captured_params.at("embed_dim").i;
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

class fuse_multiheadattention_pass_qkv : public fuse_multiheadattention_pass_sameqkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 32 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 33 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 34 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 33 35 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_4        1 1 32 36 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 35 37 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 34 38 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.permute           op_7        1 1 37 39 dims=(0,2,1,3)
Tensor.reshape          op_8        1 1 39 40 shape=(%num_heads,%kvsize,%feat_per_head)
torch.permute           op_9        1 1 36 41 dims=(0,2,1,3)
Tensor.reshape          op_10       1 1 41 42 shape=(%num_heads,%qsize,%feat_per_head)
torch.permute           op_11       1 1 40 43 dims=(0,2,1)
torch.matmul            op_12       2 1 42 43 44
F.softmax               op_13       1 1 44 45 dim=-1
torch.permute           op_14       1 1 38 46 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 46 47 shape=(%num_heads,%kvsize,%feat_per_head)
torch.matmul            op_16       2 1 45 47 48
Tensor.reshape          op_17       1 1 48 49 shape=(%batch,%num_heads,%qsize,%feat_per_head)
torch.permute           op_18       1 1 49 50 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 50 51 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 51 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
nn.MultiheadAttention   attention   3 1 query key value out embed_dim=%embed_dim kdim=%kdim vdim=%vdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Operator* op = ops.at("attention");

        const int embed_dim = captured_params.at("embed_dim").i;
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

class fuse_multiheadattention_pass_q_samekv : public fuse_multiheadattention_pass_qkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
24 23
pnnx.Input              input_q     0 1 query
pnnx.Input              input_kv    0 1 kv
nn.Linear               op_0        1 1 query 32 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 kv 33 bias=%kbias in_features=%kvdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 kv 34 bias=%vbias in_features=%kvdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 33 35 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.reshape          op_4        1 1 32 36 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 35 37 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_6        1 1 34 38 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.permute           op_7        1 1 37 39 dims=(0,2,1,3)
Tensor.reshape          op_8        1 1 39 40 shape=(%num_heads,%kvsize,%feat_per_head)
torch.permute           op_9        1 1 36 41 dims=(0,2,1,3)
Tensor.reshape          op_10       1 1 41 42 shape=(%num_heads,%qsize,%feat_per_head)
torch.permute           op_11       1 1 40 43 dims=(0,2,1)
torch.matmul            op_12       2 1 42 43 44
F.softmax               op_13       1 1 44 45 dim=-1
torch.permute           op_14       1 1 38 46 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 46 47 shape=(%num_heads,%kvsize,%feat_per_head)
torch.matmul            op_16       2 1 45 47 48
Tensor.reshape          op_17       1 1 48 49 shape=(%batch,%num_heads,%qsize,%feat_per_head)
torch.permute           op_18       1 1 49 50 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 50 51 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 51 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 kv
nn.MultiheadAttention   attention   2 1 query kv out embed_dim=%embed_dim kdim=%kvdim vdim=%kvdim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_1 : public fuse_multiheadattention_pass_sameqkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
22 21
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 31 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 32 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 33 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 31 35 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 32 36 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 33 37 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.permute           op_6        1 1 36 38 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 38 39 shape=(%num_heads,%size,%feat_per_head)
torch.permute           op_8        1 1 35 40 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 40 41 shape=(%num_heads,%size,%feat_per_head)
torch.einsum            op_10       2 1 41 39 42 equation=ijl,ikl->ijk
pnnx.Expression         op_11       1 1 42 43 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
F.softmax               op_12       1 1 43 44 dim=-1
torch.permute           op_13       1 1 37 45 dims=(0,2,1,3)
Tensor.reshape          op_14       1 1 45 46 shape=(%num_heads,%size,%feat_per_head)
torch.einsum            op_15       2 1 44 46 47 equation=ijl,ilk->ijk
Tensor.reshape          op_16       1 1 47 48 shape=(%batch,%num_heads,%size,%feat_per_head)
torch.permute           op_17       1 1 48 49 dims=(0,2,1,3)
Tensor.reshape          op_18       1 1 49 50 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 50 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_1_1 : public fuse_multiheadattention_pass_sameqkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
19 18
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 47 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 48 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 49 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 47 50 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 48 51 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 49 52 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 51 53 dim0=1 dim1=2
torch.permute           op_7        1 1 53 54 dims=(0,1,3,2)
torch.transpose         op_8        1 1 50 55 dim0=1 dim1=2
torch.matmul            op_9        2 1 55 54 56
pnnx.Expression         op_10       1 1 56 57 expr=div(@0,%sqrt_feat_per_head)
F.softmax               op_11       1 1 57 58 dim=-1
torch.transpose         op_12       1 1 52 59 dim0=1 dim1=2
torch.matmul            op_13       2 1 58 59 60
torch.transpose         op_14       1 1 60 61 dim0=1 dim1=2
Tensor.reshape          op_15       1 1 61 62 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 62 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int embed_dim = captured_params.at("embed_dim").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int feat_per_head = captured_params.at("feat_per_head").i;
        const float sqrt_feat_per_head = captured_params.at("sqrt_feat_per_head").f;

        if (embed_dim != num_heads * feat_per_head)
            return false;

        if (!NearlyEqual(sqrt_feat_per_head, sqrt(feat_per_head), 0.001))
            return false;

        return true;
    }
};

class fuse_multiheadattention_pass_1_2 : public fuse_multiheadattention_pass_qkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
21 20
pnnx.Input              input_0     0 1 query #query=(%batch,%qsize,%embed_dim)f32
pnnx.Input              input_1     0 1 key #key=(%batch,%kvsize,%kdim)f32
pnnx.Input              input_2     0 1 value #value=(%batch,%kvsize,%kdim)f32
nn.Linear               op_0        1 1 query 47 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 48 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 49 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 47 50 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 48 51 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 49 52 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 51 53 dim0=1 dim1=2
torch.permute           op_7        1 1 53 54 dims=(0,1,3,2)
torch.transpose         op_8        1 1 50 55 dim0=1 dim1=2
torch.matmul            op_9        2 1 55 54 56
pnnx.Expression         op_10       1 1 56 57 expr=div(@0,%sqrt_feat_per_head)
F.softmax               op_11       1 1 57 58 dim=-1
torch.transpose         op_12       1 1 52 59 dim0=1 dim1=2
torch.matmul            op_13       2 1 58 59 60
torch.transpose         op_14       1 1 60 61 dim0=1 dim1=2
Tensor.reshape          op_15       1 1 61 62 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 62 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int embed_dim = captured_params.at("embed_dim").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int feat_per_head = captured_params.at("feat_per_head").i;
        const float sqrt_feat_per_head = captured_params.at("sqrt_feat_per_head").f;

        if (embed_dim != num_heads * feat_per_head)
            return false;

        if (!NearlyEqual(sqrt_feat_per_head, sqrt(feat_per_head), 0.001))
            return false;

        return true;
    }
};

class fuse_multiheadattention_pass_2 : public fuse_multiheadattention_pass_qkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
24 23
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 32 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 33 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 34 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 32 36 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 33 37 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 34 38 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.permute           op_6        1 1 37 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=(%num_heads,%kvsize,%feat_per_head)
torch.permute           op_8        1 1 36 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=(%num_heads,%qsize,%feat_per_head)
torch.einsum            op_10       2 1 42 40 43 equation=ijl,ikl->ijk
pnnx.Expression         op_11       1 1 43 44 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
F.softmax               op_12       1 1 44 45 dim=-1
torch.permute           op_13       1 1 38 46 dims=(0,2,1,3)
Tensor.reshape          op_14       1 1 46 47 shape=(%num_heads,%kvsize,%feat_per_head)
torch.einsum            op_15       2 1 45 47 48 equation=ijl,ilk->ijk
Tensor.reshape          op_16       1 1 48 49 shape=(%batch,%num_heads,%qsize,%feat_per_head)
torch.permute           op_17       1 1 49 50 dims=(0,2,1,3)
Tensor.reshape          op_18       1 1 50 51 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 51 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_3 : public fuse_multiheadattention_pass_q_samekv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input_q     0 1 query
pnnx.Input              input_kv    0 1 kv
nn.Linear               op_0        1 1 query 32 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 kv 33 bias=%kbias in_features=%kvdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 kv 34 bias=%vbias in_features=%kvdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 32 36 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 33 37 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 34 38 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.permute           op_6        1 1 37 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=(%num_heads,%kvsize,%feat_per_head)
torch.permute           op_8        1 1 36 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=(%num_heads,%qsize,%feat_per_head)
torch.einsum            op_10       2 1 42 40 43 equation=ijl,ikl->ijk
pnnx.Expression         op_11       1 1 43 44 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
F.softmax               op_12       1 1 44 45 dim=-1
torch.permute           op_13       1 1 38 46 dims=(0,2,1,3)
Tensor.reshape          op_14       1 1 46 47 shape=(%num_heads,%kvsize,%feat_per_head)
torch.einsum            op_15       2 1 45 47 48 equation=ijl,ilk->ijk
Tensor.reshape          op_16       1 1 48 49 shape=(%batch,%num_heads,%qsize,%feat_per_head)
torch.permute           op_17       1 1 49 50 dims=(0,2,1,3)
Tensor.reshape          op_18       1 1 50 51 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 51 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_5 : public fuse_multiheadattention_pass_sameqkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 33 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 34 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 35 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 33 36 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 34 37 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 35 38 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.permute           op_6        1 1 36 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=(%num_heads,%size,%feat_per_head)
torch.permute           op_8        1 1 37 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=(%num_heads,%size,%feat_per_head)
pnnx.Attribute          op_10       0 1 43 @data
torch.transpose         op_11       1 1 42 44 dim0=-1 dim1=-2
torch.baddbmm           op_12       3 1 43 40 44 45 alpha=%inv_sqrt_embed_dim_per_head beta=0
F.softmax               op_13       1 1 45 46 dim=-1
torch.permute           op_14       1 1 38 47 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 47 48 shape=(%num_heads,%size,%feat_per_head)
torch.bmm               op_16       2 1 46 48 49
Tensor.reshape          op_17       1 1 49 50 shape=(%batch,%num_heads,%size,%feat_per_head)
torch.permute           op_18       1 1 50 51 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 51 52 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 52 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    // TODO match data zero
};

class fuse_multiheadattention_pass_6 : public fuse_multiheadattention_pass_sameqkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
24 23
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 33 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 34 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 35 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 33 36 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 34 37 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 35 38 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.permute           op_6        1 1 36 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=(%num_heads,%size,%feat_per_head)
torch.permute           op_8        1 1 37 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=(%num_heads,%size,%feat_per_head)
pnnx.Expression         op_10       2 1 40 42 43 expr=%expr_zero_shape
torch.empty             op_11       1 1 43 zeros
torch.transpose         op_12       1 1 42 44 dim0=-1 dim1=-2
torch.baddbmm           op_13       3 1 zeros 40 44 45 alpha=%inv_sqrt_embed_dim_per_head beta=0
F.softmax               op_14       1 1 45 46 dim=-1
torch.permute           op_15       1 1 38 47 dims=(0,2,1,3)
Tensor.reshape          op_16       1 1 47 48 shape=(%num_heads,%size,%feat_per_head)
torch.bmm               op_17       2 1 46 48 49
Tensor.reshape          op_18       1 1 49 50 shape=(%batch,%num_heads,%size,%feat_per_head)
torch.permute           op_19       1 1 50 51 dims=(0,2,1,3)
Tensor.reshape          op_20       1 1 51 52 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 52 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    // TODO match expr_zero_shape
};

class fuse_multiheadattention_pass_7 : public fuse_multiheadattention_pass_qkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 33 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 34 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 35 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 33 36 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 34 37 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 35 38 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.permute           op_6        1 1 36 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=(%num_heads,%qsize,%feat_per_head)
torch.permute           op_8        1 1 37 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=(%num_heads,%kvsize,%feat_per_head)
pnnx.Attribute          op_10       0 1 43 @data
torch.transpose         op_11       1 1 42 44 dim0=-1 dim1=-2
torch.baddbmm           op_12       3 1 43 40 44 45 alpha=%inv_sqrt_embed_dim_per_head beta=0
F.softmax               op_13       1 1 45 46 dim=-1
torch.permute           op_14       1 1 38 47 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 47 48 shape=(%num_heads,%kvsize,%feat_per_head)
torch.bmm               op_16       2 1 46 48 49
Tensor.reshape          op_17       1 1 49 50 shape=(%batch,%num_heads,%qsize,%feat_per_head)
torch.permute           op_18       1 1 50 51 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 51 52 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 52 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    // TODO match data zero
};

class fuse_multiheadattention_pass_8 : public fuse_multiheadattention_pass_q_samekv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
24 23
pnnx.Input              input_q     0 1 query
pnnx.Input              input_kv    0 1 kv
nn.Linear               op_0        1 1 query 33 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 kv 34 bias=%kbias in_features=%kvdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 kv 35 bias=%vbias in_features=%kvdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 33 36 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 34 37 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 35 38 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.permute           op_6        1 1 36 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=(%num_heads,%qsize,%feat_per_head)
torch.permute           op_8        1 1 37 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=(%num_heads,%kvsize,%feat_per_head)
pnnx.Attribute          op_10       0 1 43 @data
torch.transpose         op_11       1 1 42 44 dim0=-1 dim1=-2
torch.baddbmm           op_12       3 1 43 40 44 45 alpha=%inv_sqrt_embed_dim_per_head beta=0
F.softmax               op_13       1 1 45 46 dim=-1
torch.permute           op_14       1 1 38 47 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 47 48 shape=(%num_heads,%kvsize,%feat_per_head)
torch.bmm               op_16       2 1 46 48 49
Tensor.reshape          op_17       1 1 49 50 shape=(%batch,%num_heads,%qsize,%feat_per_head)
torch.permute           op_18       1 1 50 51 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 51 52 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 52 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    // TODO match data zero
};

class fuse_multiheadattention_pass_9 : public fuse_multiheadattention_pass_qkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
26 25
pnnx.Input              input_q     0 1 query
pnnx.Input              input_k     0 1 key
pnnx.Input              input_v     0 1 value
nn.Linear               op_0        1 1 query 33 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 34 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 35 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 33 36 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 34 37 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 35 38 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.permute           op_6        1 1 36 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=(%num_heads,%qsize,%feat_per_head)
torch.permute           op_8        1 1 37 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=(%num_heads,%kvsize,%feat_per_head)
pnnx.Expression         op_10       1 1 40 43 expr=%expr_zero_shape
torch.empty             op_11       1 1 43 zeros
torch.transpose         op_12       1 1 42 44 dim0=-1 dim1=-2
torch.baddbmm           op_13       3 1 zeros 40 44 45 alpha=%inv_sqrt_embed_dim_per_head beta=0
F.softmax               op_14       1 1 45 46 dim=-1
torch.permute           op_15       1 1 38 47 dims=(0,2,1,3)
Tensor.reshape          op_16       1 1 47 48 shape=(%num_heads,%kvsize,%feat_per_head)
torch.bmm               op_17       2 1 46 48 49
Tensor.reshape          op_18       1 1 49 50 shape=(%batch,%num_heads,%qsize,%feat_per_head)
torch.permute           op_19       1 1 50 51 dims=(0,2,1,3)
Tensor.reshape          op_20       1 1 51 52 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 52 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    // TODO match expr_zero_shape
};

class fuse_multiheadattention_pass_10 : public fuse_multiheadattention_pass_q_samekv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 query
pnnx.Input              input_kv    0 1 kv
nn.Linear               op_0        1 1 query 33 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 kv 34 bias=%kbias in_features=%kvdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 kv 35 bias=%vbias in_features=%kvdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 33 36 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 34 37 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 35 38 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.permute           op_6        1 1 36 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=(%num_heads,%qsize,%feat_per_head)
torch.permute           op_8        1 1 37 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=(%num_heads,%kvsize,%feat_per_head)
pnnx.Expression         op_10       1 1 40 43 expr=%expr_zero_shape
torch.empty             op_11       1 1 43 zeros
torch.transpose         op_12       1 1 42 44 dim0=-1 dim1=-2
torch.baddbmm           op_13       3 1 zeros 40 44 45 alpha=%alpha beta=0
F.softmax               op_14       1 1 45 46 dim=-1
torch.permute           op_15       1 1 38 47 dims=(0,2,1,3)
Tensor.reshape          op_16       1 1 47 48 shape=(%num_heads,%kvsize,%feat_per_head)
torch.bmm               op_17       2 1 46 48 49
Tensor.reshape          op_18       1 1 49 50 shape=(%batch,%num_heads,%qsize,%feat_per_head)
torch.permute           op_19       1 1 50 51 dims=(0,2,1,3)
Tensor.reshape          op_20       1 1 51 52 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 52 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_12 : public fuse_multiheadattention_pass_sameqkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
15 14
pnnx.Input              input_0     0 1 input
nn.Linear               op_0        1 1 input 33 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 34 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 35 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
Tensor.view             op_3        1 1 33 36 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_4        1 1 34 37 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 35 38 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 38 39 dim0=1 dim1=2
torch.transpose         op_7        1 1 37 40 dim0=1 dim1=2
torch.transpose         op_8        1 1 36 41 dim0=1 dim1=2
F.scaled_dot_product_attention op_9 3 1 41 40 39 42 attn_mask=None dropout_p=0.000000e+00 is_causal=False
torch.transpose         op_10       1 1 42 43 dim0=1 dim1=2
Tensor.reshape          op_11       1 1 43 44 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 44 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int embed_dim = captured_params.at("embed_dim").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int feat_per_head = captured_params.at("feat_per_head").i;

        if (embed_dim != num_heads * feat_per_head)
            return false;

        return true;
    }
};

class fuse_multiheadattention_pass_12_1 : public fuse_multiheadattention_pass_12
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
15 14
pnnx.Input              input_0     0 1 input
nn.Linear               op_0        1 1 input 14 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 15 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 16 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 14 17 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_4        1 1 15 18 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.reshape          op_5        1 1 16 19 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.permute           op_6        1 1 19 20 dims=(0,2,1,3)
torch.permute           op_7        1 1 18 21 dims=(0,2,1,3)
torch.permute           op_8        1 1 17 22 dims=(0,2,1,3)
F.scaled_dot_product_attention op_9 3 1 22 21 20 23 attn_mask=None dropout_p=0.000000e+00 is_causal=False
torch.permute           op_10       1 1 23 24 dims=(0,2,1,3)
Tensor.reshape          op_11       1 1 24 25 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 25 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_13 : public fuse_multiheadattention_pass_qkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
17 16
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
nn.Linear               op_0        1 1 query 33 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 key 34 bias=%kbias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 value 35 bias=%vbias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.view             op_3        1 1 33 36 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.view             op_4        1 1 34 37 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 35 38 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 38 39 dim0=1 dim1=2
torch.transpose         op_7        1 1 37 40 dim0=1 dim1=2
torch.transpose         op_8        1 1 36 41 dim0=1 dim1=2
F.scaled_dot_product_attention op_9 3 1 41 40 39 42 attn_mask=None dropout_p=0.000000e+00 is_causal=False
torch.transpose         op_10       1 1 42 43 dim0=1 dim1=2
Tensor.reshape          op_11       1 1 43 44 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 44 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int embed_dim = captured_params.at("embed_dim").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int feat_per_head = captured_params.at("feat_per_head").i;

        if (embed_dim != num_heads * feat_per_head)
            return false;

        return true;
    }
};

class fuse_multiheadattention_pass_14 : public fuse_multiheadattention_pass_q_samekv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
16 15
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 kv
nn.Linear               op_0        1 1 query 33 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 kv 34 bias=%kbias in_features=%kvdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 kv 35 bias=%vbias in_features=%kvdim out_features=%embed_dim @bias @weight
Tensor.view             op_3        1 1 33 36 shape=(%batch,%qsize,%num_heads,%feat_per_head)
Tensor.view             op_4        1 1 34 37 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 35 38 shape=(%batch,%kvsize,%num_heads,%feat_per_head)
torch.transpose         op_6        1 1 38 39 dim0=1 dim1=2
torch.transpose         op_7        1 1 37 40 dim0=1 dim1=2
torch.transpose         op_8        1 1 36 41 dim0=1 dim1=2
F.scaled_dot_product_attention op_9 3 1 41 40 39 42 attn_mask=None dropout_p=0.000000e+00 is_causal=False
torch.transpose         op_10       1 1 42 43 dim0=1 dim1=2
Tensor.reshape          op_11       1 1 43 44 shape=(%batch,%qsize,%embed_dim)
nn.Linear               out_proj    1 1 44 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int embed_dim = captured_params.at("embed_dim").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int feat_per_head = captured_params.at("feat_per_head").i;

        if (embed_dim != num_heads * feat_per_head)
            return false;

        return true;
    }
};

class fuse_multiheadattention_pass_15 : public fuse_multiheadattention_pass_sameqkv
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
Tensor.reshape          op_10       1 1 9 14 shape=(%num_heads,%batch_mul_size,%feat_per_head)
Tensor.reshape          op_11       1 1 10 12 shape=(%num_heads,%batch_mul_size,%feat_per_head)
Tensor.reshape          op_12       1 1 11 17 shape=(%num_heads,%batch_mul_size,%feat_per_head)
torch.transpose         op_13       1 1 12 13 dim0=1 dim1=2
torch.bmm               op_14       2 1 14 13 15
F.softmax               op_15       1 1 15 16 dim=-1
torch.bmm               op_16       2 1 16 17 18
Tensor.view             op_17       1 1 18 19 shape=(%batch,%num_heads,%size,%feat_per_head)
torch.transpose         op_18       1 1 19 20 dim0=1 dim1=2
Tensor.reshape          op_19       1 1 20 21 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 21 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_16 : public fuse_multiheadattention_pass_sameqkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
27 26
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 3 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 5 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 7 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 3 4 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
Tensor.view             op_4        1 1 4 9 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 5 6 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_6        1 1 7 8 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.transpose         op_7        1 1 9 10 dim0=1 dim1=2
torch.transpose         op_8        1 1 6 11 dim0=1 dim1=2
torch.transpose         op_9        1 1 8 12 dim0=1 dim1=2
Tensor.reshape          op_10       1 1 10 15 shape=(%num_heads,%batch_mul_size,%feat_per_head)
Tensor.reshape          op_11       1 1 11 13 shape=(%num_heads,%batch_mul_size,%feat_per_head)
Tensor.reshape          op_12       1 1 12 21 shape=(%num_heads,%batch_mul_size,%feat_per_head)
torch.transpose         op_13       1 1 13 14 dim0=1 dim1=2
torch.bmm               op_14       2 1 15 14 16
Tensor.view             op_15       1 1 16 17 shape=(%batch,%num_heads,%size,%size)
pnnx.Attribute          attn_mask   0 1 attn_mask @data=(1,1,%size,%size)f32
pnnx.Expression         op_16       2 1 17 attn_mask 18 expr=add(@0,@1)
Tensor.view             op_17       1 1 18 19 shape=(%num_heads,%size,%size)
F.softmax               op_18       1 1 19 20 dim=-1
torch.bmm               op_19       2 1 20 21 22
Tensor.view             op_20       1 1 22 23 shape=(%batch,%num_heads,%size,%feat_per_head)
torch.transpose         op_21       1 1 23 24 dim0=1 dim1=2
Tensor.reshape          op_22       1 1 24 25 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 25 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          attn_mask   0 1 attn_mask @data=%attn_mask.data
nn.MultiheadAttention   attention   2 1 input attn_mask out embed_dim=%embed_dim kdim=%embed_dim vdim=%embed_dim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_multiheadattention_pass_sameqkv::write(ops, captured_params, captured_attrs);

        const int size = captured_params.at("size").i;

        ops.at("attn_mask")->attrs["data"].shape = {size, size};
    }
};

class fuse_multiheadattention_pass_16_1 : public fuse_multiheadattention_pass_sameqkv
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
20 19
pnnx.Input              input_0     0 1 input
nn.Linear               op_0        1 1 input 31 bias=%qbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 32 bias=%kbias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 34 bias=%vbias in_features=%embed_dim out_features=%embed_dim @bias @weight
Tensor.view             op_3        1 1 31 36 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_4        1 1 32 33 shape=(%batch,%size,%num_heads,%feat_per_head)
Tensor.view             op_5        1 1 34 35 shape=(%batch,%size,%num_heads,%feat_per_head)
torch.permute           op_6        1 1 36 38 dims=(0,2,1,3)
torch.permute           op_7        1 1 33 37 dims=(0,2,1,3)
torch.permute           op_8        1 1 35 43 dims=(0,2,1,3)
torch.transpose         op_9        1 1 37 39 dim0=-1 dim1=-2
torch.matmul            op_10       2 1 38 39 40
pnnx.Attribute          attn_mask   0 1 attn_mask @data=(1,1,1,%size)f32
pnnx.Expression         op_11       2 1 40 attn_mask 41 expr=add(div(@0,%sqrt_feat_per_head),@1)
F.softmax               op_12       1 1 41 42 dim=-1
torch.matmul            op_13       2 1 42 43 44
torch.permute           op_14       1 1 44 45 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 45 46 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 46 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Attribute          attn_mask   0 1 attn_mask @data=%attn_mask.data
nn.MultiheadAttention   attention   2 1 input attn_mask out embed_dim=%embed_dim kdim=%embed_dim vdim=%embed_dim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int embed_dim = captured_params.at("embed_dim").i;
        const int num_heads = captured_params.at("num_heads").i;
        const int feat_per_head = captured_params.at("feat_per_head").i;
        const float sqrt_feat_per_head = captured_params.at("sqrt_feat_per_head").f;

        if (embed_dim != num_heads * feat_per_head)
            return false;

        if (!NearlyEqual(sqrt_feat_per_head, sqrt(feat_per_head), 0.001))
            return false;

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_multiheadattention_pass_sameqkv::write(ops, captured_params, captured_attrs);

        const int size = captured_params.at("size").i;

        Operator* op_attr = ops.at("attn_mask");

        fprintf(stderr, "op_attr->attrs[data] type %d\n", op_attr->attrs["data"].type);

        // hack attn_mask shape
        op_attr->attrs["data"].shape = {size, size};

        // hack attn_mask value
        std::vector<char>& data = op_attr->attrs["data"].data;
        size_t len = data.size();
        data.resize(len * size);
        for (int i = 1; i < size; i++)
        {
            memcpy(&data[len * i], &data[0], len);
        }
    }
};

class fuse_multiheadattention_pass_17 : public fuse_multiheadattention_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
17 16
pnnx.Input              input_0     0 1 input
nn.Linear               op_0        1 1 input 8 bias=%qkvbias in_features=%embed_dim out_features=%qkv_out_features @bias @weight
Tensor.reshape          op_1        1 1 8 9 shape=(%batch,%size,3,%num_heads,%feat_per_head)
torch.permute           op_2        1 1 9 10 dims=(2,0,3,1,4)
torch.unbind            op_3        1 3 10 11 12 13 dim=0
pnnx.Expression         op_4        1 1 11 14 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
torch.transpose         op_5        1 1 12 15 dim0=-2 dim1=-1
torch.matmul            op_6        2 1 14 15 16
pnnx.Attribute          attn_mask   0 1 attn_mask @data=(1,%num_heads,%size,%size)f32
pnnx.Expression         op_7        2 1 16 attn_mask 18 expr=add(@0,@1)
F.softmax               op_8        1 1 18 19 dim=-1
torch.matmul            op_9        2 1 19 13 20
torch.transpose         op_10       1 1 20 21 dim0=1 dim1=2
Tensor.reshape          op_11       1 1 21 22 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 22 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Attribute          attn_mask   0 1 attn_mask @data=%attn_mask.data
nn.MultiheadAttention   attention   2 1 input attn_mask out embed_dim=%embed_dim kdim=%embed_dim vdim=%embed_dim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_multiheadattention_pass::write(ops, captured_params, captured_attrs);

        const int batch = captured_params.at("batch").i;
        const int size = captured_params.at("size").i;
        const int num_heads = captured_params.at("num_heads").i;

        Operator* op_attr = ops.at("attn_mask");

        // hack attn_mask shape
        op_attr->attrs["data"].shape = {batch * num_heads, size, size};

        // hack attn_mask value
        std::vector<char>& data = op_attr->attrs["data"].data;
        size_t len = data.size();
        data.resize(len * batch);
        for (int i = 1; i < batch; i++)
        {
            memcpy(&data[len * i], &data[0], len);
        }
    }
};

class fuse_multiheadattention_pass_18 : public fuse_multiheadattention_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
20 19
pnnx.Input              input_0     0 1 input
nn.Linear               op_0        1 1 input 25 bias=%qkvbias in_features=%embed_dim out_features=%qkv_out_features @bias @weight
Tensor.reshape          op_1        1 1 25 26 shape=(%batch,%size,3,%num_heads,%feat_per_head)
torch.permute           op_2        1 1 26 27 dims=(2,0,3,1,4)
torch.unbind            op_3        1 3 27 28 29 30 dim=0
pnnx.Expression         op_4        1 1 28 31 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
torch.transpose         op_5        1 1 29 32 dim0=-2 dim1=-1
torch.matmul            op_6        2 1 31 32 33
pnnx.Attribute          attn_mask   0 1 attn_mask @data=(1,%num_heads,%size,%size)f32
pnnx.Expression         op_7        2 1 33 attn_mask 35 expr=add(@0,@1)
Tensor.view             op_8        1 1 35 36 shape=(1,%batch,%num_heads,%size,%size)
pnnx.Attribute          op_9        0 1 37 @data=(1,%batch,1,%size,%size)f32
pnnx.Expression         op_10       2 1 36 37 38 expr=add(@0,@1)
Tensor.view             op_11       1 1 38 39 shape=(%batch,%num_heads,%size,%size)
F.softmax               op_12       1 1 39 40 dim=-1
torch.matmul            op_13       2 1 40 30 41
torch.transpose         op_14       1 1 41 42 dim0=1 dim1=2
Tensor.reshape          op_15       1 1 42 43 shape=(%batch,%size,%embed_dim)
nn.Linear               out_proj    1 1 43 out bias=%outbias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Attribute          attn_mask   0 1 attn_mask @data=%attn_mask.data
nn.MultiheadAttention   attention   2 1 input attn_mask out embed_dim=%embed_dim kdim=%embed_dim vdim=%embed_dim num_heads=%num_heads batch_first=True add_zero_attn=False add_bias_kv=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        fuse_multiheadattention_pass::write(ops, captured_params, captured_attrs);

        const int batch = captured_params.at("batch").i;
        const int size = captured_params.at("size").i;
        const int num_heads = captured_params.at("num_heads").i;

        Operator* op_attr = ops.at("attn_mask");

        // hack attn_mask shape
        op_attr->attrs["data"].shape = {batch * num_heads, size, size};

        // hack attn_mask value
        std::vector<char>& data = op_attr->attrs["data"].data;
        size_t len = data.size();
        data.resize(len * batch);
        for (int i = 1; i < batch; i++)
        {
            memcpy(&data[len * i], &data[0], len);
        }

        // add mask2
        {
            auto mask2 = captured_attrs.at("op_9.data");
            auto maskdata = op_attr->attrs["data"].get_float32_data();
            const int ls = mask2.shape[3] * mask2.shape[4];

            for (int i = 0; i < batch; i++)
            {
                for (int n = 0; n < num_heads; n++)
                {
                    float* p = (float*)maskdata.data() + ls * (i * num_heads + n);
                    const float* p2 = ((float*)mask2.data.data()) + ls * i;
                    for (int k = 0; k < ls; k++)
                    {
                        p[k] += p2[k];
                    }
                }
            }

            op_attr->attrs["data"].set_float32_data(maskdata);
        }
    }
};

void fuse_multiheadattention(Graph& graph)
{
#if TORCH_VERSION_MAJOR >= 2 || (TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 9)
    fuse_multiheadattention_pass a;
    fuse_multiheadattention_pass_11 a1;
    fuse_multiheadattention_pass_sameqkv b;
    fuse_multiheadattention_pass_qkv c;
    fuse_multiheadattention_pass_q_samekv d;
    fuse_multiheadattention_pass_1 b1;
    fuse_multiheadattention_pass_1_1 b11;
    fuse_multiheadattention_pass_1_2 b12;
    fuse_multiheadattention_pass_2 c1;
    fuse_multiheadattention_pass_3 d1;
    fuse_multiheadattention_pass_5 e;
    fuse_multiheadattention_pass_6 f;
    fuse_multiheadattention_pass_7 g;
    fuse_multiheadattention_pass_8 h;
    fuse_multiheadattention_pass_9 i;
    fuse_multiheadattention_pass_10 j;
    fuse_multiheadattention_pass_12 k;
    fuse_multiheadattention_pass_12_1 k1;
    fuse_multiheadattention_pass_13 l;
    fuse_multiheadattention_pass_14 m;
    fuse_multiheadattention_pass_15 n;
    fuse_multiheadattention_pass_16 o;
    fuse_multiheadattention_pass_16_1 o1;
    fuse_multiheadattention_pass_17 p;
    fuse_multiheadattention_pass_18 q;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &a1, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
    pnnx_graph_rewrite(graph, &d, opindex);
    pnnx_graph_rewrite(graph, &b1, opindex);
    pnnx_graph_rewrite(graph, &b11, opindex);
    pnnx_graph_rewrite(graph, &b12, opindex);
    pnnx_graph_rewrite(graph, &c1, opindex);
    pnnx_graph_rewrite(graph, &d1, opindex);
    pnnx_graph_rewrite(graph, &e, opindex);
    pnnx_graph_rewrite(graph, &f, opindex);
    pnnx_graph_rewrite(graph, &g, opindex);
    pnnx_graph_rewrite(graph, &h, opindex);
    pnnx_graph_rewrite(graph, &i, opindex);
    pnnx_graph_rewrite(graph, &j, opindex);
    pnnx_graph_rewrite(graph, &k, opindex);
    pnnx_graph_rewrite(graph, &k1, opindex);
    pnnx_graph_rewrite(graph, &l, opindex);
    pnnx_graph_rewrite(graph, &m, opindex);
    pnnx_graph_rewrite(graph, &n, opindex);
    pnnx_graph_rewrite(graph, &o, opindex);
    pnnx_graph_rewrite(graph, &o1, opindex);
    pnnx_graph_rewrite(graph, &p, opindex);
    pnnx_graph_rewrite(graph, &q, opindex);
#endif
}

} // namespace pnnx
