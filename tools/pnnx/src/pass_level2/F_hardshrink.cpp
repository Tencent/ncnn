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

class F_hardshrink : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 lambd
aten::hardshrink        op_0        2 1 input lambd out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardshrink";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardshrink, 100)

static bool NearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    float diff = (float)fabs(a - b);
    if (diff <= epsilon)
        return true;

    // relative error
    return diff < epsilon * std::max(fabs(a), fabs(b));
}

class F_hardshrink_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 a value=%lambd
torch.gt                op_1        2 1 input a aa
prim::Constant          op_2        0 1 b value=%lambd2
torch.lt                op_3        2 1 input b bb
aten::__or__            op_4        2 1 aa bb ab
prim::Constant          op_5        0 1 zero value=0
torch.where             op_6        3 1 ab input zero out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.hardshrink";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        float lambd = captured_params.at("lambd").f;
        float lambd2 = captured_params.at("lambd2").f;
        return NearlyEqual(lambd, -lambd2, 0.001);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["lambd"] = captured_params.at("lambd");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_hardshrink_1, 100)

} // namespace pnnx
