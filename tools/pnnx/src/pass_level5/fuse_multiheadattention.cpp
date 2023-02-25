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

namespace pnnx {

class fuse_multiheadattention_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
14 13
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 1 bias=%qkv_bias in_features=%embed_dim out_features=%qkv_out_features @bias @weight
Tensor.reshape          op_1        1 1 1 2 shape=%expr
torch.permute           op_2        1 1 2 3 dims=(2,0,3,1,4)
torch.unbind            op_3        1 3 3 4 5 6 dim=0
pnnx.Expression         op_4        1 1 4 7 expr=%expr2
torch.permute           op_5        1 1 5 8 dims=(0,1,3,2)
torch.matmul            op_6        2 1 7 8 9
F.softmax               op_7        1 1 9 10 dim=-1
torch.matmul            op_8        2 1 10 6 11
torch.permute           op_9        1 1 11 12 dims=(0,2,1,3)
Tensor.reshape          op_10       1 1 12 13 shape=%expr3
nn.Linear               out_proj    1 1 13 out bias=%out_proj_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.MultiheadAttention";
    }

    const char* name_str() const
    {
        return "attention";
    }

    bool match_captured_params_attrs(const std::map<std::string, Parameter>& captured_params) const
    {
        // (1,-1,3,4,16)
        // mul(@0,2.581989e-01)
        // (1,-1,64)

        // TODO stricter rules here

        if (captured_params.at("expr").type != 5)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        int num_heads = captured_params.at("expr").ai[3];

        bool qkv_bias = captured_params.at("qkv_bias").b;
        bool out_proj_bias = captured_params.at("out_proj_bias").b;
        bool bias = qkv_bias || out_proj_bias;

        op->params["num_heads"] = num_heads;
        op->params["batch_first"] = true;
        op->params["add_zero_attn"] = false;
        op->params["add_bias_kv"] = false;
        op->params["bias"] = bias;

        int embed_dim = captured_params.at("embed_dim").i;

        op->params["embed_dim"] = embed_dim;
        op->params["kdim"] = embed_dim;
        op->params["vdim"] = embed_dim;

        op->attrs["in_proj_weight"] = captured_attrs.at("op_0.weight");
        if (bias)
        {
            if (qkv_bias)
            {
                op->attrs["in_proj_bias"] = captured_attrs.at("op_0.bias");
            }
            else
            {
                // init bias as zero
                op->attrs["in_proj_bias"] = Attribute();
                op->attrs["in_proj_bias"].type = 1;
                op->attrs["in_proj_bias"].shape = {embed_dim * 3};

                op->attrs["in_proj_bias"].data.resize(embed_dim * 3 * sizeof(float));
                memset(op->attrs["in_proj_bias"].data.data(), 0, embed_dim * 3 * sizeof(float));
            }
        }

        op->attrs["out_proj.weight"] = captured_attrs.at("out_proj.weight");
        if (bias)
        {
            if (out_proj_bias)
            {
                op->attrs["out_proj.bias"] = captured_attrs.at("out_proj.bias");
            }
            else
            {
                // init bias as zero
                op->attrs["out_proj.bias"] = Attribute();
                op->attrs["out_proj.bias"].type = 1;
                op->attrs["out_proj.bias"].shape = {embed_dim};

                op->attrs["out_proj.bias"].data.resize(embed_dim * sizeof(float));
                memset(op->attrs["out_proj.bias"].data.data(), 0, embed_dim * sizeof(float));
            }
        }
    }
};

class fuse_multiheadattention_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 31 bias=%q_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 32 bias=%k_bias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 33 bias=%v_bias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 32 34 expr=%expr
Tensor.reshape          op_4        1 1 31 35 shape=%q_shape
Tensor.reshape          op_5        1 1 34 36 shape=%k_shape
Tensor.reshape          op_6        1 1 33 37 shape=%v_shape
torch.permute           op_7        1 1 36 38 dims=(0,2,1,3)
Tensor.reshape          op_8        1 1 38 39 shape=%k2_shape
torch.permute           op_9        1 1 35 40 dims=(0,2,1,3)
Tensor.reshape          op_10       1 1 40 41 shape=%q2_shape
torch.permute           op_11       1 1 39 42 dims=(0,2,1)
torch.matmul            op_12       2 1 41 42 43
F.softmax               op_13       1 1 43 44 dim=-1
torch.permute           op_14       1 1 37 45 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 45 46 shape=%v2_shape
torch.matmul            op_16       2 1 44 46 47
Tensor.reshape          op_18       1 1 47 48 shape=%qkv_shape
torch.permute           op_19       1 1 48 49 dims=(0,2,1,3)
Tensor.reshape          op_20       1 1 49 50 shape=%qkv2_shape
nn.Linear               out_proj    1 1 50 out bias=%out_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.MultiheadAttention";
    }

    const char* name_str() const
    {
        return "attention";
    }

    bool match_captured_params_attrs(const std::map<std::string, Parameter>& captured_params) const
    {
        // mul(@0,1.581139e-01)
        // q_shape = (1,-1,8,40)
        // k_shape = (1,-1,8,40)
        // v_shape = (1,-1,8,40)
        // k2_shape = (8,-1,40)
        // q2_shape = (8,-1,40)
        // v2_shape = (8,-1,40)
        // qkv_shape = (1,8,-1,40)
        // qkv2_shape = (1,-1,320)

        // TODO stricter rules here

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        int embed_dim = captured_params.at("embed_dim").i;
        int kdim = captured_params.at("kdim").i;
        int vdim = captured_params.at("vdim").i;

        // (1,*,8,40)
        int num_heads = captured_params.at("q_shape").ai[2];

        bool q_bias = captured_params.at("q_bias").b;
        bool k_bias = captured_params.at("k_bias").b;
        bool v_bias = captured_params.at("v_bias").b;
        bool out_bias = captured_params.at("out_bias").b;
        bool bias = q_bias || k_bias || v_bias || out_bias;

        op->params["embed_dim"] = embed_dim;
        op->params["kdim"] = kdim;
        op->params["vdim"] = vdim;

        op->params["num_heads"] = num_heads;
        op->params["batch_first"] = true;
        op->params["add_zero_attn"] = false;
        op->params["add_bias_kv"] = false;
        op->params["bias"] = bias;

        op->attrs["in_proj_weight"] = Attribute();
        op->attrs["in_proj_weight"].type = 1;
        op->attrs["in_proj_weight"].shape = {embed_dim * 3, embed_dim};
        op->attrs["in_proj_weight"].data.resize(embed_dim * 3 * embed_dim * sizeof(float));

        // combine qkv weight
        {
            float* in_proj_weight_ptr = (float*)op->attrs["in_proj_weight"].data.data();
            memcpy(in_proj_weight_ptr, captured_attrs.at("op_0.weight").data.data(), embed_dim * embed_dim * sizeof(float));
            in_proj_weight_ptr += embed_dim * embed_dim;
            memcpy(in_proj_weight_ptr, captured_attrs.at("op_1.weight").data.data(), embed_dim * embed_dim * sizeof(float));
            in_proj_weight_ptr += embed_dim * embed_dim;
            memcpy(in_proj_weight_ptr, captured_attrs.at("op_2.weight").data.data(), embed_dim * embed_dim * sizeof(float));
        }

        op->attrs["out_proj.weight"] = captured_attrs.at("out_proj.weight");

        if (bias)
        {
            op->attrs["in_proj_bias"] = Attribute();
            op->attrs["in_proj_bias"].type = 1;
            op->attrs["in_proj_bias"].shape = {embed_dim * 3};
            op->attrs["in_proj_bias"].data.resize(embed_dim * 3 * sizeof(float));

            // combine qkv bias
            {
                float* in_proj_bias_ptr = (float*)op->attrs["in_proj_bias"].data.data();
                if (q_bias)
                {
                    memcpy(in_proj_bias_ptr, captured_attrs.at("op_0.bias").data.data(), embed_dim * sizeof(float));
                }
                else
                {
                    memset(in_proj_bias_ptr, 0, embed_dim * sizeof(float));
                }
                in_proj_bias_ptr += embed_dim;
                if (k_bias)
                {
                    memcpy(in_proj_bias_ptr, captured_attrs.at("op_1.bias").data.data(), embed_dim * sizeof(float));
                }
                else
                {
                    memset(in_proj_bias_ptr, 0, embed_dim * sizeof(float));
                }
                in_proj_bias_ptr += embed_dim;
                if (v_bias)
                {
                    memcpy(in_proj_bias_ptr, captured_attrs.at("op_2.bias").data.data(), embed_dim * sizeof(float));
                }
                else
                {
                    memset(in_proj_bias_ptr, 0, embed_dim * sizeof(float));
                }
            }

            if (out_bias)
            {
                op->attrs["out_proj.bias"] = captured_attrs.at("out_proj.bias");
            }
            else
            {
                // init bias as zero
                op->attrs["out_proj.bias"] = Attribute();
                op->attrs["out_proj.bias"].type = 1;
                op->attrs["out_proj.bias"].shape = {embed_dim};

                op->attrs["out_proj.bias"].data.resize(embed_dim * sizeof(float));
                memset(op->attrs["out_proj.bias"].data.data(), 0, embed_dim * sizeof(float));
            }
        }
    }
};

class fuse_multiheadattention_pass_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 q
pnnx.Input              input_k     0 1 k
pnnx.Input              input_v     0 1 v
nn.Linear               op_0        1 1 q 32 bias=%q_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 k 33 bias=%k_bias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 v 34 bias=%v_bias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 33 35 expr=%expr
Tensor.reshape          op_4        1 1 32 36 shape=%q_shape
Tensor.reshape          op_5        1 1 35 37 shape=%k_shape
Tensor.reshape          op_6        1 1 34 38 shape=%v_shape
torch.permute           op_7        1 1 37 39 dims=(0,2,1,3)
Tensor.reshape          op_8        1 1 39 40 shape=%k2_shape
torch.permute           op_9        1 1 36 41 dims=(0,2,1,3)
Tensor.reshape          op_10       1 1 41 42 shape=%q2_shape
torch.permute           op_11       1 1 40 43 dims=(0,2,1)
torch.matmul            op_12       2 1 42 43 44
F.softmax               op_13       1 1 44 45 dim=-1
torch.permute           op_14       1 1 38 46 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 46 47 shape=%v2_shape
torch.matmul            op_16       2 1 45 47 48
Tensor.reshape          op_17       1 1 48 49 shape=%qkv_shape
torch.permute           op_18       1 1 49 50 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 50 51 shape=%qkv2_shape
nn.Linear               out_proj    1 1 51 out bias=%out_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.MultiheadAttention";
    }

    const char* name_str() const
    {
        return "attention";
    }

    bool match_captured_params_attrs(const std::map<std::string, Parameter>& captured_params) const
    {
        // mul(@0,1.581139e-01)
        // q_shape = (1,*,8,40)
        // k_shape = (1,*,8,40)
        // v_shape = (1,*,8,40)
        // k2_shape = (8,*,40)
        // q2_shape = (8,*,40)
        // v2_shape = (8,*,40)
        // qkv_shape = (1,*,8,40)
        // qkv2_shape = (1,*,320)

        // TODO stricter rules here

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        int embed_dim = captured_params.at("embed_dim").i;
        int kdim = captured_params.at("kdim").i;
        int vdim = captured_params.at("vdim").i;

        // (1,*,8,40)
        int num_heads = captured_params.at("q_shape").ai[2];

        bool q_bias = captured_params.at("q_bias").b;
        bool k_bias = captured_params.at("k_bias").b;
        bool v_bias = captured_params.at("v_bias").b;
        bool out_bias = captured_params.at("out_bias").b;
        bool bias = q_bias || k_bias || v_bias || out_bias;

        op->params["embed_dim"] = embed_dim;
        op->params["kdim"] = kdim;
        op->params["vdim"] = vdim;

        op->params["num_heads"] = num_heads;
        op->params["batch_first"] = true;
        op->params["add_zero_attn"] = false;
        op->params["add_bias_kv"] = false;
        op->params["bias"] = bias;

        op->attrs["q_proj_weight"] = captured_attrs.at("op_0.weight");
        op->attrs["k_proj_weight"] = captured_attrs.at("op_1.weight");
        op->attrs["v_proj_weight"] = captured_attrs.at("op_2.weight");
        op->attrs["out_proj.weight"] = captured_attrs.at("out_proj.weight");

        if (bias)
        {
            op->attrs["in_proj_bias"] = Attribute();
            op->attrs["in_proj_bias"].type = 1;
            op->attrs["in_proj_bias"].shape = {embed_dim * 3};
            op->attrs["in_proj_bias"].data.resize(embed_dim * 3 * sizeof(float));

            // combine qkv bias
            {
                float* in_proj_bias_ptr = (float*)op->attrs["in_proj_bias"].data.data();
                if (q_bias)
                {
                    memcpy(in_proj_bias_ptr, captured_attrs.at("op_0.bias").data.data(), embed_dim * sizeof(float));
                }
                else
                {
                    memset(in_proj_bias_ptr, 0, embed_dim * sizeof(float));
                }
                in_proj_bias_ptr += embed_dim;
                if (k_bias)
                {
                    memcpy(in_proj_bias_ptr, captured_attrs.at("op_1.bias").data.data(), embed_dim * sizeof(float));
                }
                else
                {
                    memset(in_proj_bias_ptr, 0, embed_dim * sizeof(float));
                }
                in_proj_bias_ptr += embed_dim;
                if (v_bias)
                {
                    memcpy(in_proj_bias_ptr, captured_attrs.at("op_2.bias").data.data(), embed_dim * sizeof(float));
                }
                else
                {
                    memset(in_proj_bias_ptr, 0, embed_dim * sizeof(float));
                }
            }

            if (out_bias)
            {
                op->attrs["out_proj.bias"] = captured_attrs.at("out_proj.bias");
            }
            else
            {
                // init bias as zero
                op->attrs["out_proj.bias"] = Attribute();
                op->attrs["out_proj.bias"].type = 1;
                op->attrs["out_proj.bias"].shape = {embed_dim};

                op->attrs["out_proj.bias"].data.resize(embed_dim * sizeof(float));
                memset(op->attrs["out_proj.bias"].data.data(), 0, embed_dim * sizeof(float));
            }
        }
    }
};

class fuse_multiheadattention_pass_4 : public fuse_multiheadattention_pass_3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
24 23
pnnx.Input              input_q     0 1 q
pnnx.Input              input_kv    0 1 kv
nn.Linear               op_0        1 1 q 32 bias=%q_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 kv 33 bias=%k_bias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 kv 34 bias=%v_bias in_features=%vdim out_features=%embed_dim @bias @weight
pnnx.Expression         op_3        1 1 33 35 expr=%expr
Tensor.reshape          op_4        1 1 32 36 shape=%q_shape
Tensor.reshape          op_5        1 1 35 37 shape=%k_shape
Tensor.reshape          op_6        1 1 34 38 shape=%v_shape
torch.permute           op_7        1 1 37 39 dims=(0,2,1,3)
Tensor.reshape          op_8        1 1 39 40 shape=%k2_shape
torch.permute           op_9        1 1 36 41 dims=(0,2,1,3)
Tensor.reshape          op_10       1 1 41 42 shape=%q2_shape
torch.permute           op_11       1 1 40 43 dims=(0,2,1)
torch.matmul            op_12       2 1 42 43 44
F.softmax               op_13       1 1 44 45 dim=-1
torch.permute           op_14       1 1 38 46 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 46 47 shape=%v2_shape
torch.matmul            op_16       2 1 45 47 48
Tensor.reshape          op_17       1 1 48 49 shape=%qkv_shape
torch.permute           op_18       1 1 49 50 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 50 51 shape=%qkv2_shape
nn.Linear               out_proj    1 1 51 out bias=%out_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_5 : public fuse_multiheadattention_pass_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 33 bias=%q_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 34 bias=%k_bias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 35 bias=%v_bias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 33 36 shape=%q_shape
Tensor.reshape          op_4        1 1 34 37 shape=%k_shape
Tensor.reshape          op_5        1 1 35 38 shape=%v_shape
torch.permute           op_6        1 1 36 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=%q2_shape
torch.permute           op_8        1 1 37 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=%k2_shape
pnnx.Attribute          op_10       0 1 43 @zeros
torch.transpose         op_11       1 1 42 44 dim0=-1 dim1=-2
torch.baddbmm           op_12       3 1 43 40 44 45 alpha=%alpha beta=0
F.softmax               op_13       1 1 45 46 dim=-1
torch.permute           op_14       1 1 38 47 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 47 48 shape=%v2_shape
torch.bmm               op_16       2 1 46 48 49
Tensor.reshape          op_17       1 1 49 50 shape=%qkv_shape
torch.permute           op_18       1 1 50 51 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 51 52 shape=%qkv2_shape
nn.Linear               out_proj    1 1 52 out bias=%out_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_6 : public fuse_multiheadattention_pass_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
24 23
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 33 bias=%q_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 input 34 bias=%k_bias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 input 35 bias=%v_bias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 33 36 shape=%q_shape
Tensor.reshape          op_4        1 1 34 37 shape=%k_shape
Tensor.reshape          op_5        1 1 35 38 shape=%v_shape
torch.permute           op_6        1 1 36 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=%q2_shape
torch.permute           op_8        1 1 37 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=%k2_shape
pnnx.Expression         op_10       2 1 40 42 43 expr=%expr_zero_shape
torch.empty             op_11       1 1 43 zeros
torch.transpose         op_12       1 1 42 44 dim0=-1 dim1=-2
torch.baddbmm           op_13       3 1 zeros 40 44 45 alpha=%alpha beta=0
F.softmax               op_14       1 1 45 46 dim=-1
torch.permute           op_15       1 1 38 47 dims=(0,2,1,3)
Tensor.reshape          op_16       1 1 47 48 shape=%v2_shape
torch.bmm               op_17       2 1 46 48 49
Tensor.reshape          op_18       1 1 49 50 shape=%qkv_shape
torch.permute           op_19       1 1 50 51 dims=(0,2,1,3)
Tensor.reshape          op_20       1 1 51 52 shape=%qkv2_shape
nn.Linear               out_proj    1 1 52 out bias=%out_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_7 : public fuse_multiheadattention_pass_3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 q
pnnx.Input              input_k     0 1 k
pnnx.Input              input_v     0 1 v
nn.Linear               op_0        1 1 q 33 bias=%q_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 k 34 bias=%k_bias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 v 35 bias=%v_bias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 33 36 shape=%q_shape
Tensor.reshape          op_4        1 1 34 37 shape=%k_shape
Tensor.reshape          op_5        1 1 35 38 shape=%v_shape
torch.permute           op_6        1 1 36 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=%q2_shape
torch.permute           op_8        1 1 37 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=%k2_shape
pnnx.Attribute          op_10       0 1 43 @zeros
torch.transpose         op_11       1 1 42 44 dim0=-1 dim1=-2
torch.baddbmm           op_12       3 1 43 40 44 45 alpha=%alpha beta=0
F.softmax               op_13       1 1 45 46 dim=-1
torch.permute           op_14       1 1 38 47 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 47 48 shape=%v2_shape
torch.bmm               op_16       2 1 46 48 49
Tensor.reshape          op_17       1 1 49 50 shape=%qkv_shape
torch.permute           op_18       1 1 50 51 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 51 52 shape=%qkv2_shape
nn.Linear               out_proj    1 1 52 out bias=%out_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_8 : public fuse_multiheadattention_pass_3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
24 23
pnnx.Input              input_q     0 1 q
pnnx.Input              input_kv    0 1 kv
nn.Linear               op_0        1 1 q 33 bias=%q_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 kv 34 bias=%k_bias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 kv 35 bias=%v_bias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 33 36 shape=%q_shape
Tensor.reshape          op_4        1 1 34 37 shape=%k_shape
Tensor.reshape          op_5        1 1 35 38 shape=%v_shape
torch.permute           op_6        1 1 36 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=%q2_shape
torch.permute           op_8        1 1 37 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=%k2_shape
pnnx.Attribute          op_10       0 1 43 @zeros
torch.transpose         op_11       1 1 42 44 dim0=-1 dim1=-2
torch.baddbmm           op_12       3 1 43 40 44 45 alpha=%alpha beta=0
F.softmax               op_13       1 1 45 46 dim=-1
torch.permute           op_14       1 1 38 47 dims=(0,2,1,3)
Tensor.reshape          op_15       1 1 47 48 shape=%v2_shape
torch.bmm               op_16       2 1 46 48 49
Tensor.reshape          op_17       1 1 49 50 shape=%qkv_shape
torch.permute           op_18       1 1 50 51 dims=(0,2,1,3)
Tensor.reshape          op_19       1 1 51 52 shape=%qkv2_shape
nn.Linear               out_proj    1 1 52 out bias=%out_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_9 : public fuse_multiheadattention_pass_3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
26 25
pnnx.Input              input_q     0 1 q
pnnx.Input              input_k     0 1 k
pnnx.Input              input_v     0 1 v
nn.Linear               op_0        1 1 q 33 bias=%q_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 k 34 bias=%k_bias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 v 35 bias=%v_bias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 33 36 shape=%q_shape
Tensor.reshape          op_4        1 1 34 37 shape=%k_shape
Tensor.reshape          op_5        1 1 35 38 shape=%v_shape
torch.permute           op_6        1 1 36 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=%q2_shape
torch.permute           op_8        1 1 37 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=%k2_shape
pnnx.Expression         op_10       1 1 40 43 expr=%expr_zero_shape
torch.empty             op_11       1 1 43 zeros
torch.transpose         op_12       1 1 42 44 dim0=-1 dim1=-2
torch.baddbmm           op_13       3 1 zeros 40 44 45 alpha=%alpha beta=0
F.softmax               op_14       1 1 45 46 dim=-1
torch.permute           op_15       1 1 38 47 dims=(0,2,1,3)
Tensor.reshape          op_16       1 1 47 48 shape=%v2_shape
torch.bmm               op_17       2 1 46 48 49
Tensor.reshape          op_18       1 1 49 50 shape=%qkv_shape
torch.permute           op_19       1 1 50 51 dims=(0,2,1,3)
Tensor.reshape          op_20       1 1 51 52 shape=%qkv2_shape
nn.Linear               out_proj    1 1 52 out bias=%out_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class fuse_multiheadattention_pass_10 : public fuse_multiheadattention_pass_3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
25 24
pnnx.Input              input_q     0 1 q
pnnx.Input              input_kv    0 1 kv
nn.Linear               op_0        1 1 q 33 bias=%q_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
nn.Linear               op_1        1 1 kv 34 bias=%k_bias in_features=%kdim out_features=%embed_dim @bias @weight
nn.Linear               op_2        1 1 kv 35 bias=%v_bias in_features=%vdim out_features=%embed_dim @bias @weight
Tensor.reshape          op_3        1 1 33 36 shape=%q_shape
Tensor.reshape          op_4        1 1 34 37 shape=%k_shape
Tensor.reshape          op_5        1 1 35 38 shape=%v_shape
torch.permute           op_6        1 1 36 39 dims=(0,2,1,3)
Tensor.reshape          op_7        1 1 39 40 shape=%q2_shape
torch.permute           op_8        1 1 37 41 dims=(0,2,1,3)
Tensor.reshape          op_9        1 1 41 42 shape=%k2_shape
pnnx.Expression         op_10       1 1 40 43 expr=%expr_zero_shape
torch.empty             op_11       1 1 43 zeros
torch.transpose         op_12       1 1 42 44 dim0=-1 dim1=-2
torch.baddbmm           op_13       3 1 zeros 40 44 45 alpha=%alpha beta=0
F.softmax               op_14       1 1 45 46 dim=-1
torch.permute           op_15       1 1 38 47 dims=(0,2,1,3)
Tensor.reshape          op_16       1 1 47 48 shape=%v2_shape
torch.bmm               op_17       2 1 46 48 49
Tensor.reshape          op_18       1 1 49 50 shape=%qkv_shape
torch.permute           op_19       1 1 50 51 dims=(0,2,1,3)
Tensor.reshape          op_20       1 1 51 52 shape=%qkv2_shape
nn.Linear               out_proj    1 1 52 out bias=%out_bias in_features=%embed_dim out_features=%embed_dim @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void fuse_multiheadattention(Graph& graph)
{
    fuse_multiheadattention_pass a;
    fuse_multiheadattention_pass_2 c;
    fuse_multiheadattention_pass_3 d;
    fuse_multiheadattention_pass_4 e;
    fuse_multiheadattention_pass_5 f;
    fuse_multiheadattention_pass_6 g;
    fuse_multiheadattention_pass_7 h;
    fuse_multiheadattention_pass_8 i;
    fuse_multiheadattention_pass_9 j;
    fuse_multiheadattention_pass_10 k;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
    pnnx_graph_rewrite(graph, &d, opindex);
    pnnx_graph_rewrite(graph, &e, opindex);
    pnnx_graph_rewrite(graph, &f, opindex);
    pnnx_graph_rewrite(graph, &g, opindex);
    pnnx_graph_rewrite(graph, &h, opindex);
    pnnx_graph_rewrite(graph, &i, opindex);
    pnnx_graph_rewrite(graph, &j, opindex);
    pnnx_graph_rewrite(graph, &k, opindex);
}

} // namespace pnnx
