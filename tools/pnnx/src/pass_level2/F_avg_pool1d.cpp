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

class F_avg_pool1d_onnx_opset1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
AveragePool             op_0        1 1 input out kernel_shape=%kernel_shape strides=%strides pads=%pads
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.avg_pool1d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("kernel_shape").type != 5 || captured_params.at("kernel_shape").ai.size() != 1)
            return false;

        if (captured_params.at("strides").type != 5 || captured_params.at("strides").ai.size() != 1)
            return false;

        if (captured_params.at("pads").type != 5 || captured_params.at("pads").ai.size() != 2)
            return false;

        const std::vector<int>& pads = captured_params.at("pads").ai;
        if (pads[0] != pads[1])
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& pads = captured_params.at("pads").ai;

        op->params["kernel_size"] = captured_params.at("kernel_shape");
        op->params["stride"] = captured_params.at("strides");
        op->params["padding"] = {pads[0]};
        op->params["ceil_mode"] = false;
        op->params["count_include_pad"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_avg_pool1d_onnx_opset1, 10)

class F_avg_pool1d_onnx_opset7 : public F_avg_pool1d_onnx_opset1
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
AveragePool             op_0        1 1 input out kernel_shape=%kernel_shape strides=%strides pads=%pads count_include_pad=%count_include_pad
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        F_avg_pool1d_onnx_opset1::write(op, captured_params);

        int count_include_pad = captured_params.at("count_include_pad").i;
        op->params["count_include_pad"] = (count_include_pad != 0);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_avg_pool1d_onnx_opset7, 10)

class F_avg_pool1d_onnx_opset10 : public F_avg_pool1d_onnx_opset7
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

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        F_avg_pool1d_onnx_opset7::write(op, captured_params);

        int ceil_mode = captured_params.at("ceil_mode").i;
        op->params["ceil_mode"] = (ceil_mode != 0);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_avg_pool1d_onnx_opset10, 10)

class F_avg_pool1d_onnx_opset10_1 : public F_avg_pool1d_onnx_opset10
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
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_avg_pool1d_onnx_opset10_1, 10)

class F_avg_pool1d_onnx_opset19 : public F_avg_pool1d_onnx_opset10
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
AveragePool             op_0        1 1 input out kernel_shape=%kernel_shape dilations=%dilations strides=%strides pads=%pads ceil_mode=%ceil_mode count_include_pad=%count_include_pad
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (F_avg_pool1d_onnx_opset10::match(captured_params))
            return false;

        if (captured_params.at("dilations").type != 5 || captured_params.at("dilations").ai.size() != 1)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        F_avg_pool1d_onnx_opset10::write(op, captured_params);

        op->params["dilation"] = captured_params.at("dilations");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_avg_pool1d_onnx_opset19, 10)

class F_avg_pool1d_onnx_opset19_1 : public F_avg_pool1d_onnx_opset19
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
AveragePool             op_0        1 1 input out kernel_shape=%kernel_shape dilations=%dilations strides=%strides pads=%pads ceil_mode=%ceil_mode count_include_pad=%count_include_pad auto_pad=*
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_avg_pool1d_onnx_opset19_1, 10)

} // namespace pnnx
