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

#include "fuse_scaled_dot_product_attention.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

#include <torch/csrc/api/include/torch/torch.h>

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
torch.permute           op_0        1 1 key 59 dims=(0,1,3,2)
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
F.scaled_dot_product_attention op_0 3 1 query key value out attn_mask=None dropout_p=0.0 is_causal=False
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

void fuse_scaled_dot_product_attention(Graph& graph)
{
#if TORCH_VERSION_MAJOR >= 2
    fuse_scaled_dot_product_attention_pass a;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
#endif
}

} // namespace pnnx
