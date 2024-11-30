// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

class torch_tensor_split_indices : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 dim value=%dim
prim::Constant          op_1        0 1 indices value=%indices
aten::tensor_split      op_2        3 1 input indices dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.tensor_split";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("indices").type == 5;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_tensor_split_indices, 60)

class torch_tensor_split_sections : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 dim value=%dim
prim::Constant          op_1        0 1 sections value=%sections
aten::tensor_split      op_2        3 1 input sections dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.tensor_split";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("sections").type == 2;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_tensor_split_sections, 60)

} // namespace pnnx
