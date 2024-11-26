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

class F_softmax : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 dim value=%dim
prim::Constant          op_1        0 1 dtype value=*
aten::softmax           op_2        3 1 input dim dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax, 100)

class F_softmax_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::softmax_no_dtype  op_0        1 1 input out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax_1, 100)

class F_softmax_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
Softmax                 op_0        1 1 input out axis=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax_onnx, 101)

class F_softmax_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
Tensor.permute          op_0        1 1 input a dims=%dims
Softmax                 op_1        1 1 a b axis=%axis
Tensor.permute          op_2        1 1 b out dims=%dims
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& dims = captured_params.at("dims").ai;
        const int axis = captured_params.at("axis").i;

        if (axis >= (int)dims.size())
            return false;

        int excount = 0;
        for (int i = 0; i < (int)dims.size(); i++)
        {
            if (dims[i] != i)
                excount++;
        }

        if (excount != 2)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& dims = captured_params.at("dims").ai;
        const int axis = captured_params.at("axis").i;

        op->params["dim"] = dims[axis];
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax_onnx_1, 100)

} // namespace pnnx
