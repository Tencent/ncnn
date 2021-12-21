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

#include "pass_level2.h"

namespace pnnx {

class F_lp_pool1d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
20 19
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 kernel_size
pnnx.Input              input_2     0 1 stride
pnnx.Input              input_3     0 1 norm_type
prim::ListConstruct     op_0        1 1 kernel_size kernel_size_tuple
aten::pow               op_1        2 1 input norm_type 4
prim::Constant          op_2        0 1 padding_w value=0
prim::ListConstruct     op_3        1 1 padding_w padding
prim::Constant          op_4        0 1 ceil_mode value=%ceil_mode
prim::Constant          op_5        0 1 count_include_pad value=True
aten::avg_pool1d        op_6        6 1 4 kernel_size_tuple stride padding ceil_mode count_include_pad out.1
aten::sign              op_7        1 1 out.1 14
aten::abs               op_8        1 1 out.1 input.1
aten::relu              op_9        1 1 input.1 19
aten::mul               op_10       2 1 14 19 20
prim::Constant          op_11       0 1 21 value=*
aten::mul               op_12       2 1 20 21 22
prim::Constant          op_13       0 1 24 value=*
aten::pow               op_14       2 1 22 24 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.lp_pool1d";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_lp_pool1d, 7)

class F_lp_pool1d_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
19 18
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 kernel_size
pnnx.Input              input_2     0 1 stride
pnnx.Input              input_3     0 1 norm_type
aten::pow               op_0        2 1 input norm_type 4
prim::Constant          op_1        0 1 padding_w value=0
prim::ListConstruct     op_2        1 1 padding_w padding
prim::Constant          op_3        0 1 ceil_mode value=%ceil_mode
prim::Constant          op_4        0 1 count_include_pad value=True
aten::avg_pool1d        op_5        6 1 4 kernel_size stride padding ceil_mode count_include_pad out.1
aten::sign              op_6        1 1 out.1 14
aten::abs               op_7        1 1 out.1 input.1
aten::relu              op_8        1 1 input.1 19
aten::mul               op_9        2 1 14 19 20
prim::Constant          op_10       0 1 21 value=*
aten::mul               op_11       2 1 20 21 22
prim::Constant          op_12       0 1 24 value=*
aten::pow               op_13       2 1 22 24 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.lp_pool1d";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_lp_pool1d_1, 8)

} // namespace pnnx
