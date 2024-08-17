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

class torch_split : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 tensor
pnnx.Input              input_1     0 1 split_size_or_sections
pnnx.Input              input_2     0 1 dim
aten::split             op_0        3 1 tensor split_size_or_sections dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.split";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_split, 20)

class torch_split_01 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 tensor
pnnx.Input              input_1     0 1 split_size_or_sections
aten::split             op_0        2 1 tensor split_size_or_sections out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.split";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_split_01, 20)

class torch_split_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 tensor
pnnx.Input              input_1     0 1 split_size_or_sections
pnnx.Input              input_2     0 1 dim
aten::split_with_sizes  op_0        3 1 tensor split_size_or_sections dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.split";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_split_1, 20)

class torch_split_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 tensor
aten::split             op_0        1 1 tensor out dim=%dim indices=%split_size_or_sections
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.split";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_split_onnx, 20)

class torch_split_onnx_1 : public torch_split_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 tensor
pnnx.Input              input_1     0 1 split_size_or_sections
aten::split             op_0        2 1 tensor split_size_or_sections out dim=%dim indices=None
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_split_onnx_1, 20)

} // namespace pnnx
