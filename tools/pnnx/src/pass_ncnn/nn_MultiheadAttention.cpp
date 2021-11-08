// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "pass_ncnn.h"

#include <math.h>
#include <string.h>

namespace pnnx {

namespace ncnn {

class nn_MultiheadAttention : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.MultiheadAttention   op_0        1 1 input out num_heads=%num_heads batch_first=%batch_first add_zero_attn=%add_zero_attn embed_dim=%embed_dim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "MultiHeadAttention";
    }

    const char* name_str() const
    {
        return "attention";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = captured_params.at("embed_dim");
        op->params["1"] = captured_params.at("num_heads");

        if (captured_params.at("add_bias_kv").b)
        {
            fprintf(stderr, "MultiheadAttention add_bias_kv=True not supported\n");
        }

        const int embed_dim = captured_params.at("embed_dim").i;

        // split in_proj_weight and in_proj_bias into q k v
        std::vector<float> q_weight(embed_dim * embed_dim);
        std::vector<float> q_bias(embed_dim);
        std::vector<float> k_weight(embed_dim * embed_dim);
        std::vector<float> k_bias(embed_dim);
        std::vector<float> v_weight(embed_dim * embed_dim);
        std::vector<float> v_bias(embed_dim);
        {
            // qkv - embed_dim - embed_dim
            const float* wptr = (const float*)captured_attrs.at("op_0.in_proj_weight").data.data();
            // qkv - embed_dim
            const float* bptr = (const float*)captured_attrs.at("op_0.in_proj_bias").data.data();

            {
                memcpy(q_weight.data(), wptr, embed_dim * embed_dim * sizeof(float));
                memcpy(q_bias.data(), bptr, embed_dim * sizeof(float));
                wptr += embed_dim * embed_dim;
                bptr += embed_dim;
            }

            {
                memcpy(k_weight.data(), wptr, embed_dim * embed_dim * sizeof(float));
                memcpy(k_bias.data(), bptr, embed_dim * sizeof(float));
                wptr += embed_dim * embed_dim;
                bptr += embed_dim;
            }

            {
                memcpy(v_weight.data(), wptr, embed_dim * embed_dim * sizeof(float));
                memcpy(v_bias.data(), bptr, embed_dim * sizeof(float));
            }
        }

        op->params["2"] = embed_dim * embed_dim;

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = Attribute({embed_dim, embed_dim}, q_weight);
        op->attrs["2"] = Attribute({embed_dim}, q_bias);
        op->attrs["3"] = Attribute();
        op->attrs["3"].data = {0, 0, 0, 0};
        op->attrs["4"] = Attribute({embed_dim, embed_dim}, k_weight);
        op->attrs["5"] = Attribute({embed_dim}, k_bias);
        op->attrs["6"] = Attribute();
        op->attrs["6"].data = {0, 0, 0, 0};
        op->attrs["7"] = Attribute({embed_dim, embed_dim}, v_weight);
        op->attrs["8"] = Attribute({embed_dim}, v_bias);
        op->attrs["9"] = Attribute();
        op->attrs["9"].data = {0, 0, 0, 0};
        op->attrs["a"] = captured_attrs.at("op_0.out_proj.weight");
        op->attrs["b"] = captured_attrs.at("op_0.out_proj.bias");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention, 20)

class nn_MultiheadAttention_1 : public nn_MultiheadAttention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.MultiheadAttention   op_0        1 1 input out num_heads=%num_heads add_zero_attn=%add_zero_attn embed_dim=%embed_dim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention_1, 20)

} // namespace ncnn

} // namespace pnnx
