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

class F_mish : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::mish              op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.mish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_mish, 10)

class F_mish_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 11 value=1
prim::Constant          op_1        0 1 12 value=20
aten::softplus          op_2        3 1 input 11 12 a
aten::tanh              op_3        1 1 a b
aten::mul               op_4        2 1 input b out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.mish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_mish_1, 9)

} // namespace pnnx
