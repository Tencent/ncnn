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

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torchvision_DeformConv2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 offset
torchvision.ops.DeformConv2d op_0   2 1 input offset out in_channels=%in_channels out_channels=%out_channels kernel_size=%kernel_size stride=%stride padding=%padding dilation=%dilation groups=1 bias=%bias @weight @bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeformableConv2D";
    }

    const char* name_str() const
    {
        return "deformconv2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = captured_params.at("out_channels");
        op->params["1"] = captured_params.at("kernel_size").ai[1];
        op->params["11"] = captured_params.at("kernel_size").ai[0];
        op->params["2"] = captured_params.at("dilation").ai[1];
        op->params["12"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[1];
        op->params["13"] = captured_params.at("stride").ai[0];
        op->params["4"] = captured_params.at("padding").ai[1];
        op->params["14"] = captured_params.at("padding").ai[0];
        op->params["5"] = captured_params.at("bias").b ? 1 : 0;
        op->params["6"] = captured_attrs.at("op_0.weight").elemcount();

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = captured_attrs.at("op_0.weight");
        if (captured_params.at("bias").b)
            op->attrs["2"] = captured_attrs.at("op_0.bias");
    }
};

class torchvision_DeformConv2d_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 offset
pnnx.Input              input_2     0 1 mask
torchvision.ops.DeformConv2d op_0   3 1 input offset mask out in_channels=%in_channels out_channels=%out_channels kernel_size=%kernel_size stride=%stride padding=%padding dilation=%dilation groups=1 bias=%bias @weight @bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeformableConv2D";
    }

    const char* name_str() const
    {
        return "deformconv2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = captured_params.at("out_channels");
        op->params["1"] = captured_params.at("kernel_size").ai[1];
        op->params["11"] = captured_params.at("kernel_size").ai[0];
        op->params["2"] = captured_params.at("dilation").ai[1];
        op->params["12"] = captured_params.at("dilation").ai[0];
        op->params["3"] = captured_params.at("stride").ai[1];
        op->params["13"] = captured_params.at("stride").ai[0];
        op->params["4"] = captured_params.at("padding").ai[1];
        op->params["14"] = captured_params.at("padding").ai[0];
        op->params["5"] = captured_params.at("bias").b ? 1 : 0;
        op->params["6"] = captured_attrs.at("op_0.weight").elemcount();

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = captured_attrs.at("op_0.weight");
        if (captured_params.at("bias").b)
            op->attrs["2"] = captured_attrs.at("op_0.bias");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torchvision_DeformConv2d, 20)
REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torchvision_DeformConv2d_1, 20)

} // namespace ncnn

} // namespace pnnx
