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

#include "fuse_contiguous_view.h"

#include "pass_level2.h"

namespace pnnx {

class fuse_contiguous_view_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
Tensor.contiguous       op_0        1 1 input a memory_format=*
Tensor.view             op_1        1 1 a out shape=%shape
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.reshape";
    }

    const char* name_str() const
    {
        return "view_shape";
    }
};

class fuse_contiguous_view_pass_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_1     0 1 input
pnnx.Input              input_2     0 1 shape
Tensor.contiguous       op_0        1 1 input a memory_format=*
Tensor.view             op_1        2 1 a shape out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.reshape";
    }

    const char* name_str() const
    {
        return "view_shape";
    }
};

void fuse_contiguous_view(Graph& graph)
{
    fuse_contiguous_view_pass a;
    fuse_contiguous_view_pass_1 b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace pnnx
