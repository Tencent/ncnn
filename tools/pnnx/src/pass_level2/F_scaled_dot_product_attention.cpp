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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention, 10)

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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_scaled_dot_product_attention_1, 10)

} // namespace pnnx
