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

class torch_min : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
prim::Constant          op_0        0 1 keepdim value=%keepdim
aten::min               op_1        3 2 input dim keepdim out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.min";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);

        // drop indices if not used
        if (op->outputs[1]->consumers.empty())
        {
            op->outputs[1]->producer = 0;
            op->outputs.resize(1);
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_min, 50)

class torch_min_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
aten::min               op_1        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.min";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_min_1, 50)

class torch_min_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
ReduceMin               op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.min";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axes") != captured_params.end())
        {
            if (captured_params.at("op_0.axes").type != 5 || captured_params.at("op_0.axes").ai.size() != 1)
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axes") != captured_params.end())
        {
            op->params["dim"] = captured_params.at("op_0.axes").ai[0];

            if (captured_params.find("op_0.keepdims") != captured_params.end())
            {
                op->params["keepdim"] = captured_params.at("op_0.keepdims").i ? true : false;
            }
            else
            {
                op->params["keepdim"] = true;
            }
        }
        else
        {
            // reduce all
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_min_onnx, 51)

class torch_min_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
ReduceMin               op_0        1 1 input out %*=%*
ArgMin                  op_1        1 1 input indices %*=%*
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.min";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axes") == captured_params.end())
            return false;

        if (captured_params.find("op_0.keepdims") == captured_params.end())
            return false;

        if (captured_params.find("op_1.axis") == captured_params.end())
            return false;

        if (captured_params.find("op_1.keepdims") == captured_params.end())
            return false;

        if (captured_params.at("op_0.axes").type != 5 || captured_params.at("op_0.axes").ai.size() != 1)
            return false;

        if (captured_params.at("op_1.axis").type != 2)
            return false;

        if (captured_params.at("op_0.axes").ai[0] != captured_params.at("op_1.axis").i)
            return false;

        if (captured_params.at("op_0.keepdims").i != captured_params.at("op_1.keepdims").i)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["dim"] = captured_params.at("op_0.axes").ai[0];
        op->params["keepdim"] = captured_params.at("op_0.keepdims").i ? true : false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_min_onnx_1, 50)

} // namespace pnnx
