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

class F_hardswish : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::hardswish         op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish, 10)

class F_hardswish_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 392 value=3
prim::Constant          op_1        0 1 393 value=1
aten::add               op_2        3 1 input 392 393 a
prim::Constant          op_3        0 1 394 value=0.000000e+00
prim::Constant          op_4        0 1 395 value=6.000000e+00
aten::hardtanh_         op_5        3 1 a 394 395 b
aten::mul               op_6        2 1 input b c
prim::Constant          op_7        0 1 391 value=6
aten::div               op_8        2 1 c 391 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_1, 8)

class F_hardswish_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
aten::hardsigmoid       op_0        1 1 input a
aten::mul               op_1        2 1 input a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_2, 9)

class F_hardswish_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 12 value=3
prim::Constant          op_1        0 1 13 value=1
aten::add               op_2        3 1 input 12 13 a
prim::Constant          op_3        0 1 17 value=0.000000e+00
prim::Constant          op_4        0 1 18 value=6.000000e+00
aten::hardtanh          op_5        3 1 a 17 18 b
aten::mul               op_6        2 1 input b c
prim::Constant          op_7        0 1 22 value=6.000000e+00
aten::div               op_8        2 1 c 22 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_3, 8)

class F_hardswish_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 25 value=3.000000e+00
prim::Constant          op_1        0 1 47 value=1
aten::add               op_2        3 1 input 25 47 a
prim::Constant          op_3        0 1 48 value=0.000000e+00
prim::Constant          op_4        0 1 49 value=6.000000e+00
aten::hardtanh_         op_5        3 1 a 48 49 b
prim::Constant          op_6        0 1 50 value=6.000000e+00
aten::div               op_7        2 1 b 50 c
aten::mul               op_8        2 1 c input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_4, 8)

class F_hardswish_5 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 25 value=3.000000e+00
prim::Constant          op_1        0 1 48 value=1
aten::add               op_2        3 1 input 25 48 a
aten::relu6_            op_3        1 1 a b
prim::Constant          op_4        0 1 49 value=6.000000e+00
aten::div               op_5        2 1 b 49 c
aten::mul               op_6        2 1 c input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardswish";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardswish_5, 8)

} // namespace pnnx
