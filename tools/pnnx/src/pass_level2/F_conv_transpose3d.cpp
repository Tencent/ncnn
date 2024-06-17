// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

class F_conv_transpose3d_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
ConvTranspose           op_0        3 1 input weight bias out kernel_shape=%kernel_shape strides=%strides pads=%pads dilations=%dilations group=%group
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv_transpose3d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("kernel_shape").type != 5)
            return false;

        if (captured_params.at("kernel_shape").ai.size() != 3)
            return false;

        if (captured_params.at("strides").type != 5)
            return false;

        if (captured_params.at("strides").ai.size() != 3)
            return false;

        if (captured_params.at("dilations").type != 5)
            return false;

        if (captured_params.at("dilations").ai.size() != 3)
            return false;

        if (captured_params.at("group").type != 2)
            return false;

        if (captured_params.at("pads").type != 5)
            return false;

        const std::vector<int>& pads = captured_params.at("pads").ai;
        if (pads.size() != 6 || pads[0] != pads[3] || pads[1] != pads[4] || pads[2] != pads[5])
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& pads = captured_params.at("pads").ai;

        op->params["stride"] = captured_params.at("strides");
        op->params["dilation"] = captured_params.at("dilations");
        op->params["groups"] = captured_params.at("group");
        op->params["padding"] = {pads[0], pads[1], pads[2]};
        op->params["output_padding"] = {0, 0, 0};
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv_transpose3d_onnx, 10)

class F_conv_transpose3d_onnx_0 : public F_conv_transpose3d_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
ConvTranspose           op_0        2 1 input weight out kernel_shape=%kernel_shape strides=%strides pads=%pads dilations=%dilations group=%group
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
F.conv_transpose3d      conv        2 1 input weight out bias=None
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv_transpose3d_onnx_0, 10)

class F_conv_transpose3d_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
ConvTranspose           op_0        3 1 input weight bias out strides=%strides pads=%pads dilations=%dilations group=%group auto_pad=NOTSET
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv_transpose3d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("strides").type != 5)
            return false;

        if (captured_params.at("strides").ai.size() != 3)
            return false;

        if (captured_params.at("dilations").type != 5)
            return false;

        if (captured_params.at("dilations").ai.size() != 3)
            return false;

        if (captured_params.at("group").type != 2)
            return false;

        if (captured_params.at("pads").type != 5)
            return false;

        const std::vector<int>& pads = captured_params.at("pads").ai;
        if (pads.size() != 6 || pads[0] != pads[3] || pads[1] != pads[4] || pads[2] != pads[5])
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& pads = captured_params.at("pads").ai;

        op->params["stride"] = captured_params.at("strides");
        op->params["dilation"] = captured_params.at("dilations");
        op->params["groups"] = captured_params.at("group");
        op->params["padding"] = {pads[0], pads[1], pads[2]};
        op->params["output_padding"] = {0, 0, 0};
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv_transpose3d_onnx_1, 10)

class F_conv_transpose3d_onnx_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
ConvTranspose           op_0        3 1 input weight bias out kernel_shape=%kernel_shape strides=%strides pads=%pads dilations=%dilations group=%group output_padding=%output_padding
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv_transpose3d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("kernel_shape").type != 5)
            return false;

        if (captured_params.at("kernel_shape").ai.size() != 3)
            return false;

        if (captured_params.at("strides").type != 5)
            return false;

        if (captured_params.at("strides").ai.size() != 3)
            return false;

        if (captured_params.at("dilations").type != 5)
            return false;

        if (captured_params.at("dilations").ai.size() != 3)
            return false;

        if (captured_params.at("output_padding").type != 5)
            return false;

        if (captured_params.at("output_padding").ai.size() != 3)
            return false;

        if (captured_params.at("group").type != 2)
            return false;

        if (captured_params.at("pads").type != 5)
            return false;

        const std::vector<int>& pads = captured_params.at("pads").ai;
        if (pads.size() != 6 || pads[0] != pads[3] || pads[1] != pads[4] || pads[2] != pads[5])
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& pads = captured_params.at("pads").ai;

        op->params["stride"] = captured_params.at("strides");
        op->params["dilation"] = captured_params.at("dilations");
        op->params["groups"] = captured_params.at("group");
        op->params["padding"] = {pads[0], pads[1], pads[2]};
        op->params["output_padding"] = captured_params.at("output_padding");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv_transpose3d_onnx_2, 10)

class F_conv_transpose3d_onnx_3 : public F_conv_transpose3d_onnx_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
ConvTranspose           op_0        2 1 input weight out kernel_shape=%kernel_shape strides=%strides pads=%pads dilations=%dilations group=%group output_padding=%output_padding
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
F.conv_transpose3d      conv        2 1 input weight out bias=None
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv_transpose3d_onnx_3, 10)

} // namespace pnnx
