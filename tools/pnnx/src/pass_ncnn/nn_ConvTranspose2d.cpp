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

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class nn_ConvTranspose2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.ConvTranspose2d      op_0        1 1 input out in_channels=%in_channels out_channels=%out_channels kernel_size=%kernel_size stride=%stride padding=%padding dilation=%dilation output_padding=%output_padding groups=1 bias=%bias @weight @bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Deconvolution";
    }

    const char* name_str() const
    {
        return "deconv";
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
        op->params["18"] = captured_params.at("output_padding").ai[1];
        op->params["19"] = captured_params.at("output_padding").ai[0];
        op->params["5"] = captured_params.at("bias").b ? 1 : 0;
        op->params["6"] = captured_attrs.at("op_0.weight").elemcount();

        // transpose inch-outch-kh-kw to outch-inch-kh-kw
        const int inch = captured_params.at("in_channels").i;
        const int outch = captured_params.at("out_channels").i;
        const int kh = captured_params.at("kernel_size").ai[0];
        const int kw = captured_params.at("kernel_size").ai[1];
        std::vector<float> new_weight;
        {
            auto w = captured_attrs.at("op_0.weight").get_float32_data();

            new_weight.resize(outch * inch * kh * kw);
            float* w2 = (float*)new_weight.data();
            const int maxk = kh * kw;

            // reorder weight from inch-outch to outch-inch
            for (int i = 0; i < outch; i++)
            {
                for (int j = 0; j < inch; j++)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        w2[(i * inch + j) * maxk + k] = w[(j * outch + i) * maxk + k];
                    }
                }
            }
        }

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = Attribute({outch, inch, kh, kw}, new_weight);
        if (captured_params.at("bias").b)
            op->attrs["2"] = captured_attrs.at("op_0.bias");
    }
};

class nn_ConvTranspose2d_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.ConvTranspose2d      op_0        1 1 input out in_channels=%in_channels out_channels=%out_channels kernel_size=%kernel_size stride=%stride padding=%padding dilation=%dilation output_padding=%output_padding groups=%groups bias=%bias @weight @bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "DeconvolutionDepthWise";
    }

    const char* name_str() const
    {
        return "deconvdw";
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
        op->params["18"] = captured_params.at("output_padding").ai[1];
        op->params["19"] = captured_params.at("output_padding").ai[0];
        op->params["5"] = captured_params.at("bias").b ? 1 : 0;
        op->params["6"] = captured_attrs.at("op_0.weight").elemcount();
        op->params["7"] = captured_params.at("groups");

        // transpose group-inch/group-outch/group-kh-kw to group-outch/group-inch/group-kh-kw
        const int inch = captured_params.at("in_channels").i;
        const int outch = captured_params.at("out_channels").i;
        const int groups = captured_params.at("groups").i;
        const int kh = captured_params.at("kernel_size").ai[0];
        const int kw = captured_params.at("kernel_size").ai[1];
        std::vector<float> new_weight;
        {
            auto w = captured_attrs.at("op_0.weight").get_float32_data();

            new_weight.resize(outch / groups * inch * kh * kw);
            float* w2 = (float*)new_weight.data();
            const int outch_g = outch / groups;
            const int inch_g = inch / groups;
            const int maxk = kh * kw;

            for (int g = 0; g < groups; g++)
            {
                // reorder weight from inch-outch to outch-inch
                float* wg2 = w2 + g * outch_g * inch_g * maxk;
                const float* wg = (const float*)w.data() + g * inch_g * outch_g * maxk;
                for (int i = 0; i < outch_g; i++)
                {
                    for (int j = 0; j < inch_g; j++)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            wg2[(i * inch_g + j) * maxk + k] = wg[(j * outch_g + i) * maxk + k];
                        }
                    }
                }
            }
        }

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = Attribute({outch / groups, inch, kh, kw}, new_weight);
        if (captured_params.at("bias").b)
            op->attrs["2"] = captured_attrs.at("op_0.bias");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_ConvTranspose2d, 20)
REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_ConvTranspose2d_1, 21)

} // namespace ncnn

} // namespace pnnx
