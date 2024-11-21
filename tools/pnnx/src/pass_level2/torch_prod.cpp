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

class torch_prod : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
prim::Constant          op_0        0 1 keepdim value=%keepdim
prim::Constant          op_1        0 1 dtype value=*
aten::prod              op_2        4 1 input dim keepdim dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.prod";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_prod, 20)

class torch_prod_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
ReduceProd              op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.prod";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axes") == captured_params.end())
            return false;

        if (captured_params.at("op_0.axes").type != 2 && captured_params.at("op_0.axes").type != 5)
            return false;

        if (captured_params.at("op_0.axes").type == 5 && captured_params.at("op_0.axes").ai.size() > 1)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim;
        if (captured_params.at("op_0.axes").type == 2)
        {
            dim = captured_params.at("op_0.axes").i;
        }
        else // if (captured_params.at("op_0.axes").type == 5)
        {
            dim = captured_params.at("op_0.axes").ai[0];
        }

        op->params["dim"] = dim;

        if (captured_params.find("op_0.keepdims") != captured_params.end())
        {
            op->params["keepdim"] = captured_params.at("op_0.keepdims").i ? true : false;
        }
        else
        {
            op->params["keepdim"] = true;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_prod_onnx, 20)

} // namespace pnnx
