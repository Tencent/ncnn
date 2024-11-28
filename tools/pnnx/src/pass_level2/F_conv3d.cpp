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

class F_conv3d : public GraphRewriterPass
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
prim::Constant          op_3        0 1 transposed value=False
prim::Constant          op_4        0 1 output_padding value=(0,0,0)
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
        return "F.conv3d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("stride").type == 5 && captured_params.at("stride").ai.size() == 3;
    }
};

class F_conv3d_mode : public F_conv3d
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
prim::Constant          op_0        0 1 stride value=%stride
prim::Constant          op_1        0 1 padding value=%padding
prim::Constant          op_2        0 1 dilation value=%dilation
prim::Constant          op_3        0 1 groups value=%groups
aten::_convolution_mode op_4        7 1 input weight bias stride padding dilation groups out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv3d, 140)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv3d_mode, 140)

class F_conv3d_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
prim::Constant          op_0        0 1 transposed value=False
aten::convolution_onnx  op_1        4 1 input weight bias transposed out dilations=%dilations groups=%groups output_padding=(0,0,0) pads=%pads strides=%strides
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv3d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& dilations = captured_params.at("dilations").ai;
        const std::vector<int>& strides = captured_params.at("strides").ai;
        const std::vector<int>& pads = captured_params.at("pads").ai;
        return dilations.size() == 3 && strides.size() == 3 && pads.size() == 6;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::vector<int> pads = captured_params.at("pads").ai;
        if (pads.size() == 6)
        {
            pads = {pads[0], pads[1], pads[2]};
        }

        op->params["dilation"] = captured_params.at("dilations");
        op->params["stride"] = captured_params.at("strides");
        op->params["padding"] = pads;
        op->params["groups"] = captured_params.at("groups");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv3d_0, 140)

class F_conv3d_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
Conv                    op_0        3 1 input weight bias out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv3d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.kernel_shape") != captured_params.end())
        {
            if (captured_params.at("op_0.kernel_shape").type != 5 || captured_params.at("op_0.kernel_shape").ai.size() != 3)
                return false;
        }

        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            if (captured_params.at("op_0.dilations").type != 5 || captured_params.at("op_0.dilations").ai.size() != 3)
                return false;
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            if (captured_params.at("op_0.strides").type != 5 || captured_params.at("op_0.strides").ai.size() != 3)
                return false;
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            if (captured_params.at("op_0.pads").type != 5 || captured_params.at("op_0.pads").ai.size() != 6)
                return false;

            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            if (pads[0] != pads[3] || pads[1] != pads[4] || pads[2] != pads[5])
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
            op->params["dilation"] = {1, 1, 1};
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            op->params["stride"] = captured_params.at("op_0.strides");
        }
        else
        {
            op->params["stride"] = {1, 1, 1};
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            op->params["padding"] = {pads[0], pads[1], pads[2]};
        }
        else
        {
            op->params["padding"] = {0, 0, 0};
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv3d_onnx, 140)

class F_conv3d_onnx_1 : public F_conv3d_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
Conv                    op_0        2 1 input weight out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        F_conv3d_onnx::write(op, captured_params);

        op->params["bias"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv3d_onnx_1, 140)

} // namespace pnnx
