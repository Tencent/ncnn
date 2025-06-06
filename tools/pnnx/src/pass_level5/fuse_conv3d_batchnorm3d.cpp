// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "fuse_conv3d_batchnorm3d.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_conv3d_batchnorm3d_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
nn.Conv3d               op_0        1 1 input a in_channels=%in_channels out_channels=%out_channels kernel_size=%kernel_size stride=%stride padding_mode=%padding_mode padding=%padding dilation=%dilation groups=%groups bias=%bias @weight @bias
nn.BatchNorm3d          op_1        1 1 a out num_features=%num_features eps=%eps affine=%affine @running_mean @running_var @weight @bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.Conv3d";
    }

    const char* name_str() const
    {
        return "convbn3d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["in_channels"] = captured_params.at("in_channels");
        op->params["out_channels"] = captured_params.at("out_channels");
        op->params["kernel_size"] = captured_params.at("kernel_size");
        op->params["padding_mode"] = captured_params.at("padding_mode");
        op->params["stride"] = captured_params.at("stride");
        op->params["padding"] = captured_params.at("padding");
        op->params["dilation"] = captured_params.at("dilation");
        op->params["groups"] = captured_params.at("groups");
        op->params["bias"] = true;

        // resolve merged conv3d weight and bias
        int channels = captured_params.at("num_features").i;
        float bn_eps = captured_params.at("eps").f;
        bool has_bn_affine = captured_params.at("affine").b;
        bool has_conv_bias = captured_params.at("bias").b;

        auto bn_running_mean = captured_attrs.at("op_1.running_mean").get_float32_data();
        auto bn_running_var = captured_attrs.at("op_1.running_var").get_float32_data();
        auto bn_weight = has_bn_affine ? captured_attrs.at("op_1.weight").get_float32_data() : std::vector<float>();
        auto bn_bias = has_bn_affine ? captured_attrs.at("op_1.bias").get_float32_data() : std::vector<float>();

        // a = bias - slope * mean / sqrt(var + eps)
        // b = slope / sqrt(var + eps)
        // value = value * b + a

        std::vector<float> a(channels);
        std::vector<float> b(channels);
        for (int i = 0; i < channels; i++)
        {
            double sqrt_var = sqrt(bn_running_var[i] + bn_eps);

            if (has_bn_affine)
            {
                a[i] = (float)(bn_bias[i] - bn_weight[i] * bn_running_mean[i] / sqrt_var);
                b[i] = (float)(bn_weight[i] / sqrt_var);
            }
            else
            {
                a[i] = (float)(-bn_running_mean[i] / sqrt_var);
                b[i] = (float)(1.f / sqrt_var);
            }
        }

        op->attrs["weight"] = captured_attrs.at("op_0.weight");

        if (has_conv_bias)
        {
            op->attrs["bias"] = captured_attrs.at("op_0.bias");
        }
        else
        {
            // init bias as zero
            op->attrs["bias"] = Attribute();
            op->attrs["bias"].type = op->attrs["weight"].type;
            op->attrs["bias"].shape = {channels};
            op->attrs["bias"].set_float32_data(std::vector<float>(channels, 0.f));
        }

        auto conv_weight = op->attrs["weight"].get_float32_data();
        auto conv_bias = op->attrs["bias"].get_float32_data();

        const int outch = captured_params.at("out_channels").i;
        const int weight_per_outch = op->attrs["weight"].elemcount() / outch;

        for (int i = 0; i < channels; i++)
        {
            float* conv_weight_outch = conv_weight.data() + weight_per_outch * i;
            for (int j = 0; j < weight_per_outch; j++)
            {
                conv_weight_outch[j] *= b[i];
            }

            conv_bias[i] = conv_bias[i] * b[i] + a[i];
        }

        op->attrs["weight"].set_float32_data(conv_weight);
        op->attrs["bias"].set_float32_data(conv_bias);
    }
};

void fuse_conv3d_batchnorm3d(Graph& graph)
{
    fuse_conv3d_batchnorm3d_pass a;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace pnnx
