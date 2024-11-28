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

class Tensor_reshape : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 shape
aten::reshape           op_0        2 1 input shape out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.reshape";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_reshape, 60)

class Tensor_reshape_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 shape
aten::cat               op_0        1 1 shape cat dim=0
Reshape                 op_1        2 1 input cat out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.reshape";
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_reshape_onnx, 60)

class Tensor_reshape_onnx_1 : public Tensor_reshape_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 shape
Reshape                 op_0        2 1 input shape out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_reshape_onnx_1, 61)

class Tensor_reshape_onnx_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Reshape                 op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.reshape";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.shape") == captured_params.end())
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("op_0.shape").type == 5)
        {
            op->params["shape"] = captured_params.at("op_0.shape");
        }
        else // if (captured_params.at("op_0.shape").type == 2)
        {
            op->params["shape"] = std::vector<int>{captured_params.at("op_0.shape").i};
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_reshape_onnx_2, 61)

} // namespace pnnx
