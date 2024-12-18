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

class Tensor_slice : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
pnnx.Input              input_2     0 1 start
pnnx.Input              input_3     0 1 end
pnnx.Input              input_4     0 1 step
aten::slice             op_0        5 1 input dim start end step out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_slice, 70)

class Tensor_slice_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Slice                   op_0        1 1 input out axes=%axes starts=%starts ends=%ends
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("axes").type == 2)
        {
            op->params["dim"] = captured_params.at("axes");
            op->params["start"] = captured_params.at("starts");
            op->params["end"] = captured_params.at("ends");
            op->params["step"] = 1;
        }
        else // if (captured_params.at("axes").type == 5)
        {
            const std::vector<int>& axes = captured_params.at("axes").ai;
            const std::vector<int>& starts = captured_params.at("starts").ai;
            const std::vector<int>& ends = captured_params.at("ends").ai;

            if (axes.size() == 1)
            {
                op->params["dim"] = axes[0];
                op->params["start"] = starts[0];
                op->params["end"] = ends[0];
                op->params["step"] = 1;
            }
            else
            {
                op->params["dims"] = axes;
                op->params["starts"] = starts;
                op->params["ends"] = ends;
                op->params["steps"] = std::vector<int>(axes.size(), 1);
                op->params["selects"] = std::vector<int>(axes.size(), INT_MAX);
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_slice_onnx, 70)

class Tensor_slice_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Slice                   op_0        1 1 input out axes=%axes starts=%starts ends=%ends steps=%steps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("axes").type == 2)
        {
            op->params["dim"] = captured_params.at("axes");
            op->params["start"] = captured_params.at("starts");
            op->params["end"] = captured_params.at("ends");
            op->params["step"] = captured_params.at("steps");
        }
        else // if (captured_params.at("axes").type == 5)
        {
            const std::vector<int>& axes = captured_params.at("axes").ai;
            const std::vector<int>& starts = captured_params.at("starts").ai;
            const std::vector<int>& ends = captured_params.at("ends").ai;
            const std::vector<int>& steps = captured_params.at("steps").ai;

            if (axes.size() == 1)
            {
                op->params["dim"] = axes[0];
                op->params["start"] = starts[0];
                op->params["end"] = ends[0];
                op->params["step"] = steps[0];
            }
            else
            {
                op->params["dims"] = axes;
                op->params["starts"] = starts;
                op->params["ends"] = ends;
                op->params["steps"] = steps;
                op->params["selects"] = std::vector<int>(axes.size(), INT_MAX);
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_slice_onnx_1, 70)

} // namespace pnnx
