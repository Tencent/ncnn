// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_scaled_dot_product_attention.h"

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

class fuse_transformers_sdpa : public GraphRewriterPass
{
public:
    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 attn_mask
F.scaled_dot_product_attention sdpa_ht 4 1 query key value attn_mask out dropout_p=0.0 is_causal=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int feat_per_head = captured_params.at("feat_per_head").i;

        if (captured_params.find("sqrt_embed_dim_per_head") != captured_params.end())
        {
            const float sqrt_embed_dim_per_head = captured_params.at("sqrt_embed_dim_per_head").f;
            if (!NearlyEqual(sqrt_embed_dim_per_head, sqrt(feat_per_head), 0.001))
                return false;
        }

        return true;
    }
};

class fuse_transformers_deberta_sdpa : public fuse_transformers_sdpa
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
12 11
pnnx.Input              input_0     0 1 query #query=(%batch,%num_heads,%qsize,%feat_per_head)f32
pnnx.Input              input_1     0 1 key #key=(%batch,%num_heads,%kvsize,%feat_per_head)f32
pnnx.Input              input_2     0 1 value #value=(%batch,%num_heads,%kvsize,%feat_per_head)f32
pnnx.Input              input_3     0 1 attn_mask #attn_mask=(%qsize,%kvsize)bool
pnnx.Expression         op_0        1 1 query 12 expr=div(@0,%sqrt_embed_dim_per_head)
torch.transpose         op_1        1 1 key 13 dim0=-1 dim1=-2
torch.matmul            op_2        2 1 12 13 14
torch.bitwise_not       op_4        1 1 attn_mask 18
Tensor.masked_fill      op_5        2 1 14 18 16 value=-3.402823e+38
F.softmax               op_6        1 1 16 17 dim=-1
torch.matmul            op_7        2 1 17 value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_deberta_sdpa_onnx : public fuse_transformers_sdpa
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
12 11
pnnx.Input              input_0     0 1 query #query=(%batch,%num_heads,%qsize,%feat_per_head)f32
pnnx.Input              input_1     0 1 key #key=(%batch,%num_heads,%kvsize,%feat_per_head)f32
pnnx.Input              input_2     0 1 value #value=(%batch,%num_heads,%kvsize,%feat_per_head)f32
pnnx.Input              input_3     0 1 attn_mask #attn_mask=(%qsize,%kvsize)bool
pnnx.Expression         op_0        1 1 query 12 expr=div(@0,%sqrt_embed_dim_per_head)
Tensor.permute          op_1        1 1 key 13 dims=(0,1,3,2)
torch.matmul            op_2        2 1 12 13 14
torch.logical_not       op_4        1 1 attn_mask 18
torch.where             op_5        2 1 18 14 16 input=-3.402823e+38
F.softmax               op_6        1 1 16 17 dim=-1
torch.matmul            op_7        2 1 17 value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_distilbert_sdpa : public fuse_transformers_sdpa
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
13 12
pnnx.Input              input_0     0 1 query #query=(%batch,%num_heads,%qsize,%feat_per_head)f32
pnnx.Input              input_1     0 1 key #key=(%batch,%num_heads,%kvsize,%feat_per_head)f32
pnnx.Input              input_2     0 1 value #value=(%batch,%num_heads,%kvsize,%feat_per_head)f32
pnnx.Input              input_3     0 1 input #input=(%batch,%kvsize)bool
pnnx.Expression         op_0        1 1 query 13 expr=div(@0,%sqrt_embed_dim_per_head)
torch.transpose         op_1        1 1 key 14 dim0=2 dim1=3
torch.matmul            op_2        2 1 13 14 15
Tensor.reshape          op_3        1 1 input 17 shape=(%batch,1,1,%kvsize)
Tensor.expand_as        op_4        2 1 17 15 18
Tensor.masked_fill      op_5        2 1 15 18 19 value=-3.402823e+38
F.softmax               op_6        1 1 19 20 dim=-1
torch.matmul            op_7        2 1 20 value out
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
pnnx.Input              input_3     0 1 input
torch.bitwise_not       sdpa_ht_0   1 1 input 16
Tensor.reshape          sdpa_ht_1   1 1 16 17 shape=(%batch,1,1,%kvsize) #17=(%batch,1,1,%kvsize)bool
Tensor.expand           sdpa_ht_2   1 1 17 attn_mask sizes=(%batch,%num_heads,%qsize,%kvsize) #attn_mask=(%batch,%num_heads,%qsize,%kvsize)bool
F.scaled_dot_product_attention sdpa_ht 4 1 query key value attn_mask out dropout_p=0.0 is_causal=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_transformers_distilbert_sdpa_onnx : public fuse_transformers_sdpa
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
13 12
pnnx.Input              input_0     0 1 query #query=(%batch,%num_heads,%qsize,%feat_per_head)f32
pnnx.Input              input_1     0 1 key #key=(%batch,%feat_per_head,%num_heads,%kvsize)f32
pnnx.Input              input_2     0 1 value #value=(%batch,%num_heads,%kvsize,%feat_per_head)f32
pnnx.Input              input_3     0 1 input #input=(%batch,%kvsize)bool
pnnx.Expression         op_0        1 1 query 13 expr=div(@0,%sqrt_embed_dim_per_head)
Tensor.permute          op_1        1 1 key 14 dims=(0,2,3,1)
torch.matmul            op_2        2 1 13 14 15
Tensor.reshape          op_3        1 1 input 17 shape=(%batch,1,1,%kvsize)
Tensor.expand           op_4        1 1 17 18 sizes=(%batch,%num_heads,%qsize,%kvsize)
torch.where             op_5        2 1 18 15 19 input=-3.402823e+38
F.softmax               op_6        1 1 19 20 dim=-1
torch.matmul            op_7        2 1 20 value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 input
Tensor.permute          sdpa_ht_0   1 1 key 14 dims=(0,2,1,3)
torch.bitwise_not       sdpa_ht_1   1 1 input 16
Tensor.reshape          sdpa_ht_2   1 1 16 17 shape=(%batch,1,1,%kvsize) #17=(%batch,1,1,%kvsize)bool
Tensor.expand           sdpa_ht_3   1 1 17 attn_mask sizes=(%batch,%num_heads,%qsize,%kvsize) #attn_mask=(%batch,%num_heads,%qsize,%kvsize)bool
F.scaled_dot_product_attention sdpa_ht 4 1 query 14 value attn_mask out dropout_p=0.0 is_causal=False $attn_mask=attn_mask $key=14
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void fuse_transformers_scaled_dot_product_attention(Graph& graph)
{
#if TORCH_VERSION_MAJOR >= 2
    fuse_transformers_deberta_sdpa a;
    fuse_transformers_deberta_sdpa_onnx a2;
    fuse_transformers_distilbert_sdpa b;
    fuse_transformers_distilbert_sdpa_onnx b2;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &a2, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &b2, opindex);
#endif
}

} // namespace pnnx
