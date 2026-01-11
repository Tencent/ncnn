// Copyright 2023 Tencent
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

class fuse_scaled_dot_product_attention_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 query #query=(%batch,%num_heads,%qsize,%feat_per_head)f32
pnnx.Input              input_1     0 1 key #key=(%batch,%num_heads,%kvsize,%feat_per_head)f32
pnnx.Input              input_2     0 1 value #value=(%batch,%num_heads,%kvsize,%feat_per_head)f32
Tensor.permute          op_0        1 1 key 59 dims=(0,1,3,2)
torch.matmul            op_1        2 1 query 59 61
pnnx.Expression         op_2        1 1 61 62 expr=div(@0,%sqrt_embed_dim_per_head)
F.softmax               op_3        1 1 62 63 dim=-1
torch.matmul            op_4        2 1 63 value out
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
F.scaled_dot_product_attention sdpa 3 1 query key value out attn_mask=None dropout_p=0.0 is_causal=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int feat_per_head = captured_params.at("feat_per_head").i;
        const float sqrt_embed_dim_per_head = captured_params.at("sqrt_embed_dim_per_head").f;

        if (!NearlyEqual(sqrt_embed_dim_per_head, sqrt(feat_per_head), 0.001))
            return false;

        return true;
    }
};

class fuse_scaled_dot_product_attention_pass_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input_0     0 1 query #query=(%batch,%num_heads,%qsize,%feat_per_head)f32
pnnx.Input              input_1     0 1 key #key=(%batch,%num_heads,%kvsize,%feat_per_head)f32
pnnx.Input              input_2     0 1 value #value=(%batch,%num_heads,%kvsize,%feat_per_head)f32
pnnx.Input              input_3     0 1 attn_mask
torch.transpose         op_0        1 1 key 23 dim0=2 dim1=3
torch.matmul            op_1        2 1 query 23 24
pnnx.Expression         op_2        2 1 24 attn_mask 25 expr=add(mul(@0,%inv_sqrt_embed_dim_per_head),@1)
F.softmax               op_3        1 1 25 26 dim=-1
torch.matmul            op_4        2 1 26 value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 attn_mask
F.scaled_dot_product_attention sdpa 4 1 query key value attn_mask out dropout_p=0.0 is_causal=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int feat_per_head = captured_params.at("feat_per_head").i;
        const float inv_sqrt_embed_dim_per_head = captured_params.at("inv_sqrt_embed_dim_per_head").f;

        if (!NearlyEqual(inv_sqrt_embed_dim_per_head, 1.f / sqrt(feat_per_head), 0.001))
            return false;

        return true;
    }
};

class fuse_scaled_dot_product_attention_pass_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
14 13
pnnx.Input              input_0     0 1 query #query=(%batch,%qsize,%feat_per_head)f32
pnnx.Input              input_1     0 1 key #key=(%batch,%kvsize,%feat_per_head)f32
pnnx.Input              input_2     0 1 value #value=(%batch,%kvsize,%feat_per_head)f32
pnnx.Input              input_Rh    0 1 Rh #Rh=(%batch,%h,%w,%h,1)f32
pnnx.Input              input_Rw    0 1 Rw #Rw=(%batch,%h,%w,1,%w)f32
pnnx.Expression         op_0        1 1 query 17 expr=mul(@0,%inv_sqrt_embed_dim_per_head)
torch.transpose         op_1        1 1 key 22 dim0=-2 dim1=-1
torch.matmul            op_2        2 1 17 22 23
Tensor.reshape          op_3        1 1 23 24 shape=(%batch,%h,%w,%h,%w)
pnnx.Expression         op_4        3 1 24 Rh Rw 28 expr=add(add(@0,@1),@2)
Tensor.reshape          op_5        1 1 28 29 shape=(%batch,%qsize,%qsize)
F.softmax               op_6        1 1 29 30 dim=-1
torch.matmul            op_7        2 1 30 value out
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
pnnx.Input              input_Rh    0 1 Rh
pnnx.Input              input_Rw    0 1 Rw
pnnx.Expression         RhRw        2 1 Rh Rw RhRw expr=add(@0,@1) #RhRw=(%batch,%h,%w,%h,%w)f32
Tensor.reshape          attn_mask   1 1 RhRw attn_mask shape=(%batch,%qsize,%qsize) #attn_mask=(%batch,%qsize,%qsize)f32
F.scaled_dot_product_attention sdpa 4 1 query key value attn_mask out dropout_p=0.0 is_causal=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int qsize = captured_params.at("qsize").i;
        const int h = captured_params.at("h").i;
        const int w = captured_params.at("w").i;
        const int feat_per_head = captured_params.at("feat_per_head").i;
        const float inv_sqrt_embed_dim_per_head = captured_params.at("inv_sqrt_embed_dim_per_head").f;

        if (qsize != h * w)
            return false;

        if (!NearlyEqual(inv_sqrt_embed_dim_per_head, 1.f / sqrt(feat_per_head), 0.001))
            return false;

        return true;
    }
};

class fuse_scaled_dot_product_attention_pass_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
12 11
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 attn_mask
Tensor.permute          op_0        1 1 query 13 dims=(0,2,1,3)
Tensor.permute          op_1        1 1 key 20 dims=(0,2,3,1)
Tensor.permute          op_2        1 1 value 19 dims=(0,2,1,3)
torch.matmul            op_3        2 1 13 20 21
pnnx.Expression         op_4        2 1 21 attn_mask 23 expr=add(@0,@1)
F.softmax               softmax     1 1 23 24 dim=%softmax_dim
torch.matmul            op_6        2 1 24 19 out
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
pnnx.Input              input_3     0 1 attn_mask
Tensor.permute          op_0        1 1 query q dims=(0,2,1,3)
Tensor.permute          op_1        1 1 key k dims=(0,2,1,3)
Tensor.permute          op_2        1 1 value v dims=(0,2,1,3)
F.scaled_dot_product_attention sdpa 4 1 q k v attn_mask out dropout_p=0.0 is_causal=False $attn_mask=attn_mask
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const int softmax_dim = captured_params.at("softmax_dim").i;

        int softmax_input_rank = (int)matched_operators.at("softmax")->inputs[0]->shape.size();
        if (softmax_dim != -1 && softmax_dim != softmax_input_rank - 1)
            return false;

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        Operator* op = ops.at("sdpa");

        op->params["scale"] = 1.f;

        // rewrite qkv shape
        {
            std::vector<int> q_shape = ops.at("op_0")->inputs[0]->shape;
            std::vector<int> k_shape = ops.at("op_1")->inputs[0]->shape;
            std::vector<int> v_shape = ops.at("op_2")->inputs[0]->shape;

            if (!q_shape.empty())
                std::swap(q_shape[1], q_shape[2]);
            if (!k_shape.empty())
                std::swap(k_shape[1], k_shape[2]);
            if (!v_shape.empty())
                std::swap(v_shape[1], v_shape[2]);

            ops.at("op_0")->outputs[0]->shape = q_shape;
            ops.at("op_0")->outputs[0]->type = ops.at("op_0")->inputs[0]->type;
            ops.at("op_1")->outputs[0]->shape = k_shape;
            ops.at("op_1")->outputs[0]->type = ops.at("op_1")->inputs[0]->type;
            ops.at("op_2")->outputs[0]->shape = v_shape;
            ops.at("op_2")->outputs[0]->type = ops.at("op_2")->inputs[0]->type;
        }
    }
};

void fuse_scaled_dot_product_attention(Graph& graph)
{
#if TORCH_VERSION_MAJOR >= 2
    fuse_scaled_dot_product_attention_pass a;
    fuse_scaled_dot_product_attention_pass_0 a0;
    fuse_scaled_dot_product_attention_pass_1 b;
    fuse_scaled_dot_product_attention_pass_onnx onnx0;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &a0, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &onnx0, opindex);
#endif
}

} // namespace pnnx
