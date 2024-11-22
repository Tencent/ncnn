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
18 17
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 kernel_size
pnnx.Input              input_2     0 1 stride
pnnx.Input              input_3     0 1 norm_type
prim::ListConstruct     op_0        1 1 kernel_size kernel_size_tuple
aten::pow               op_1        2 1 input norm_type 4
prim::Constant          op_2        0 1 padding_w value=0
prim::ListConstruct     op_3        1 1 padding_w padding
F.avg_pool1d            op_4        4 1 4 kernel_size_tuple stride padding out.1 ceil_mode=%ceil_mode count_include_pad=True
aten::sign              op_5        1 1 out.1 14
aten::abs               op_6        1 1 out.1 input.1
F.relu                  op_7        1 1 input.1 19
aten::mul               op_8        2 1 14 19 20
prim::Constant          op_9        0 1 21 value=*
aten::mul               op_10       2 1 20 21 22
prim::Constant          op_11       0 1 24 value=*
aten::pow               op_12       2 1 22 24 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.lp_pool1d";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_lp_pool1d, 121)

class F_lp_pool1d_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
17 16
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 kernel_size
pnnx.Input              input_2     0 1 stride
pnnx.Input              input_3     0 1 norm_type
aten::pow               op_0        2 1 input norm_type 4
prim::Constant          op_1        0 1 padding_w value=0
prim::ListConstruct     op_2        1 1 padding_w padding
F.avg_pool1d            op_3        4 1 4 kernel_size stride padding out.1 ceil_mode=%ceil_mode count_include_pad=True
aten::sign              op_4        1 1 out.1 14
aten::abs               op_5        1 1 out.1 input.1
F.relu                  op_6        1 1 input.1 19
aten::mul               op_7        2 1 14 19 20
prim::Constant          op_8        0 1 21 value=*
aten::mul               op_9        2 1 20 21 22
prim::Constant          op_10       0 1 24 value=*
aten::pow               op_11       2 1 22 24 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.lp_pool1d";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_lp_pool1d_1, 121)

} // namespace pnnx
