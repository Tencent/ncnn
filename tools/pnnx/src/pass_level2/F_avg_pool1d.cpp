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

class F_avg_pool1d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 kernel_size
pnnx.Input              input_2     0 1 stride
pnnx.Input              input_3     0 1 padding
prim::Constant          op_0        0 1 ceil_mode value=%ceil_mode
prim::Constant          op_1        0 1 count_include_pad value=%count_include_pad
aten::avg_pool1d        op_2        6 1 input kernel_size stride padding ceil_mode count_include_pad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.avg_pool1d";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_avg_pool1d, 10)

class F_avg_pool1d_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
AveragePool             op_0        1 1 input out kernel_shape=%kernel_shape strides=%strides pads=%pads ceil_mode=%ceil_mode count_include_pad=%count_include_pad auto_pad=*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.avg_pool1d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("kernel_shape").type != 5)
            return false;

        if (captured_params.at("kernel_shape").ai.size() != 1)
            return false;

        if (captured_params.at("strides").type != 5)
            return false;

        if (captured_params.at("strides").ai.size() != 1)
            return false;

        if (captured_params.at("pads").type != 5)
            return false;

        const std::vector<int>& pads = captured_params.at("pads").ai;
        if (pads.size() != 2 || pads[0] != pads[1])
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& kernel_shape = captured_params.at("kernel_shape").ai;
        const std::vector<int>& strides = captured_params.at("strides").ai;
        std::vector<int> pads = captured_params.at("pads").ai;
        int ceil_mode = captured_params.at("ceil_mode").i;
        int count_include_pad = captured_params.at("count_include_pad").i;

        if (pads.size() == 2)
        {
            pads = {pads[0]};
        }

        op->params["kernel_size"] = kernel_shape;
        op->params["stride"] = strides;
        op->params["padding"] = pads;
        op->params["ceil_mode"] = (ceil_mode != 0);
        op->params["count_include_pad"] = (count_include_pad != 0);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_avg_pool1d_onnx, 10)

class F_avg_pool1d_onnx_1 : public F_avg_pool1d_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
AveragePool             op_0        1 1 input out kernel_shape=%kernel_shape strides=%strides pads=%pads ceil_mode=%ceil_mode count_include_pad=%count_include_pad
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_avg_pool1d_onnx_1, 10)

} // namespace pnnx
