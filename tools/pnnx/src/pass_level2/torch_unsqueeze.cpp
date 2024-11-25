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

class torch_unsqueeze : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 dim value=%dim
aten::unsqueeze         op_1        2 1 input dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.unsqueeze";
    }
};

class torch_unsqueeze_dynamic : public torch_unsqueeze
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
aten::unsqueeze         op_0        2 1 input dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_unsqueeze, 60)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_unsqueeze_dynamic, 61)

class torch_unsqueeze_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
Unsqueeze               op_0        2 1 input dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.unsqueeze";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_unsqueeze_onnx, 60)

class torch_unsqueeze_onnx_1 : public torch_unsqueeze_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Unsqueeze               op_0        1 1 input out axes=%axes
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("axes").type == 5 && captured_params.at("axes").ai.size() == 1)
        {
            op->params["dim"] = captured_params.at("axes").ai[0];
        }
        else
        {
            op->params["dim"] = captured_params.at("axes");
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_unsqueeze_onnx_1, 60)

} // namespace pnnx
