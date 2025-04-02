// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "lower_power.h"

#include "pass_level2.h"

namespace pnnx {

namespace tnn2pnnx {

class lower_power_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.Power               op_0        1 1 input out arg0=%exponent arg1=%alpha arg2=%beta
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input       0 1 input
prim::Constant          alpha       0 1 alpha value=%alpha
prim::Constant          beta        0 1 beta value=%beta
prim::Constant          exponent    0 1 exponent value=%exponent
aten::mul               scale       2 1 input alpha a
aten::add               shift       2 1 a beta b
aten::pow               pow         2 1 b exponent out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void lower_power(Graph& graph)
{
    lower_power_pass a;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace tnn2pnnx

} // namespace pnnx
