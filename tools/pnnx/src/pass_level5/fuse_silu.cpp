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

#include "fuse_silu.h"

#include "pass_level2.h"

namespace pnnx {

class fuse_silu_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
F.sigmoid               op_0        1 1 input a
pnnx.Expression         op_1        2 1 input a out expr=mul(@0,@1)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.silu";
    }

    const char* name_str() const
    {
        return "silu";
    }
};

class fuse_silu_pass_1 : public fuse_silu_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
nn.Sigmoid              op_0        1 1 input a
pnnx.Expression         op_1        2 1 input a out expr=mul(@0,@1)
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void fuse_silu(Graph& graph)
{
    fuse_silu_pass a;
    fuse_silu_pass_1 b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace pnnx
