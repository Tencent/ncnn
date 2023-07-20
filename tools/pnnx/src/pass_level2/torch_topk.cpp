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

#include "pass_level2.h"

namespace pnnx {

class torch_topk : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 k
pnnx.Input              input_2     0 1 dim
pnnx.Input              input_3     0 1 largest
pnnx.Input              input_4     0 1 sorted
aten::topk              op_0        5 2 input k dim largest sorted values indices
pnnx.Output             output      2 0 values indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.topk";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_topk, 20)

} // namespace pnnx
