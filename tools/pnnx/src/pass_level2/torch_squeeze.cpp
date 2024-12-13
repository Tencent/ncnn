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

class torch_squeeze : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 dim value=%dim
aten::squeeze           op_1        2 1 input dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.squeeze";
    }
};

class torch_squeeze_0 : public torch_squeeze
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
aten::squeeze_dim       op_0        1 1 input out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class torch_squeeze_1 : public torch_squeeze
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
aten::squeeze           op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class torch_squeeze_dynamic : public torch_squeeze
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
aten::squeeze           op_0        2 1 input dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_squeeze, 60)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_squeeze_0, 60)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_squeeze_1, 60)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_squeeze_dynamic, 61)

class torch_squeeze_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
Squeeze                 op_0        2 1 input dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.squeeze";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_squeeze_onnx, 60)

class torch_squeeze_onnx_1 : public torch_squeeze_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Squeeze                 op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axes") != captured_params.end())
        {
            if (captured_params.at("op_0.axes").type == 5 && captured_params.at("op_0.axes").ai.size() == 1)
            {
                op->params["dim"] = captured_params.at("op_0.axes").ai[0];
            }
            else
            {
                op->params["dim"] = captured_params.at("op_0.axes");
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_squeeze_onnx_1, 60)

} // namespace pnnx
