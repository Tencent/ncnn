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

class F_upsample_nearest : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 scale_h value=None
prim::Constant          op_1        0 1 scale_w value=None
aten::upsample_nearest2d op_2       4 1 input size scale_h scale_w out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample_nearest";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_nearest, 10)

class F_upsample_nearest_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 scale_factor
prim::Constant          op_0        0 1 size value=None
aten::upsample_nearest2d op_1       3 1 input size scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample_nearest";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_nearest_1, 10)

class F_upsample_nearest_1_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 scale_factor value=None
aten::upsample_nearest2d op_1       3 1 input size scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample_nearest";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_nearest_1_1, 10)

class F_upsample_nearest_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 scale_d value=None
prim::Constant          op_1        0 1 scale_h value=None
prim::Constant          op_2        0 1 scale_w value=None
aten::upsample_nearest3d op_3       5 1 input size scale_d scale_h scale_w out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample_nearest";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_nearest_2, 10)

class F_upsample_nearest_2_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 scale_factor value=None
aten::upsample_nearest3d op_1       3 1 input size scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample_nearest";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_nearest_2_1, 10)

class F_upsample_nearest_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 scale_factor
prim::Constant          op_0        0 1 size value=None
aten::upsample_nearest3d op_1       3 1 input size scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample_nearest";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_nearest_3, 10)

} // namespace pnnx
