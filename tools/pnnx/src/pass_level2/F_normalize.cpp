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

class F_normalize : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 keepdim value=True
prim::Constant          op_1        0 1 p value=%p
prim::Constant          op_2        0 1 dim value=%dim
prim::Constant          op_3        0 1 eps value=%eps
prim::ListConstruct     op_4        1 1 dim dims
aten::norm              op_5        4 1 input p dims keepdim 9
aten::clamp_min         op_6        2 1 9 eps 11
aten::expand_as         op_7        2 1 11 input denorm
aten::div               op_8        2 1 input denorm out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.normalize";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_normalize, 10)

class F_normalize_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
12 11
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 dtype value=*
prim::Constant          op_1        0 1 keepdim value=True
prim::Constant          op_2        0 1 p value=%p
prim::Constant          op_3        0 1 dim value=%dim
prim::Constant          op_4        0 1 eps value=%eps
prim::ListConstruct     op_5        1 1 dim dims
aten::linalg_vector_norm op_6       5 1 input p dims keepdim dtype 10
aten::clamp_min         op_7        2 1 10 eps 13
aten::expand_as         op_8        2 1 13 input denorm
aten::div               op_9        2 1 input denorm out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.normalize";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_normalize_1, 10)

} // namespace pnnx
