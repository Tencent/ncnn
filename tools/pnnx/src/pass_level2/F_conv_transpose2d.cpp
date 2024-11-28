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

class F_conv_transpose2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
15 14
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
prim::Constant          op_0        0 1 stride value=%stride
prim::Constant          op_1        0 1 padding value=%padding
prim::Constant          op_2        0 1 dilation value=%dilation
prim::Constant          op_3        0 1 transposed value=True
prim::Constant          op_4        0 1 output_padding value=%output_padding
prim::Constant          op_5        0 1 groups value=%groups
prim::Constant          op_6        0 1 benchmark value=*
prim::Constant          op_7        0 1 deterministic value=*
prim::Constant          op_8        0 1 cudnn_enabled value=*
prim::Constant          op_9        0 1 allow_tf32 value=*
aten::_convolution      op_10       13 1 input weight bias stride padding dilation transposed output_padding groups benchmark deterministic cudnn_enabled allow_tf32 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv_transpose2d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("stride").type == 5 && captured_params.at("stride").ai.size() == 2;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv_transpose2d, 140)

class F_conv_transpose2d_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
ConvTranspose           op_0        3 1 input weight bias out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv_transpose2d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.kernel_shape") != captured_params.end())
        {
            if (captured_params.at("op_0.kernel_shape").type != 5 || captured_params.at("op_0.kernel_shape").ai.size() != 2)
                return false;
        }

        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            if (captured_params.at("op_0.dilations").type != 5 || captured_params.at("op_0.dilations").ai.size() != 2)
                return false;
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            if (captured_params.at("op_0.strides").type != 5 || captured_params.at("op_0.strides").ai.size() != 2)
                return false;
        }

        if (captured_params.find("op_0.output_padding") != captured_params.end())
        {
            if (captured_params.at("op_0.output_padding").type != 5 || captured_params.at("op_0.output_padding").ai.size() != 2)
                return false;
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            if (captured_params.at("op_0.pads").type != 5 || captured_params.at("op_0.pads").ai.size() != 4)
                return false;

            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            if (pads[0] != pads[2] || pads[1] != pads[3])
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            op->params["dilation"] = captured_params.at("op_0.dilations");
        }
        else
        {
            op->params["dilation"] = {1, 1};
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            op->params["stride"] = captured_params.at("op_0.strides");
        }
        else
        {
            op->params["stride"] = {1, 1};
        }

        if (captured_params.find("op_0.output_padding") != captured_params.end())
        {
            op->params["output_padding"] = captured_params.at("op_0.output_padding");
        }
        else
        {
            op->params["output_padding"] = {0, 0};
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            op->params["padding"] = {pads[0], pads[1]};
        }
        else
        {
            op->params["padding"] = {0, 0};
        }

        if (captured_params.find("op_0.group") != captured_params.end())
        {
            op->params["groups"] = captured_params.at("op_0.group");
        }
        else
        {
            op->params["groups"] = 1;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv_transpose2d_onnx, 140)

class F_conv_transpose2d_onnx_1 : public F_conv_transpose2d_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
ConvTranspose           op_0        2 1 input weight out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        F_conv_transpose2d_onnx::write(op, captured_params);

        op->params["bias"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv_transpose2d_onnx_1, 140)

} // namespace pnnx
