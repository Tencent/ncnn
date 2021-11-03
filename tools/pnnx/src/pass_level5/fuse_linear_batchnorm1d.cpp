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

#include "fuse_linear_batchnorm1d.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_linear_batchnorm1d_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input a in_features=%in_features out_features=%out_features bias=%bias @weight @bias
nn.BatchNorm1d          op_1        1 1 a out num_features=%num_features eps=%eps affine=%affine @running_mean @running_var @weight @bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.Linear";
    }

    const char* name_str() const
    {
        return "linearbn1d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["in_features"] = captured_params.at("in_features");
        op->params["out_features"] = captured_params.at("out_features");
        op->params["bias"] = true;

        // resolve merged linear weight and bias
        int channels = captured_params.at("out_features").i;
        float bn_eps = captured_params.at("eps").f;
        bool has_bn_affine = captured_params.at("affine").b;
        bool has_conv_bias = captured_params.at("bias").b;

        const float* bn_running_mean = (const float*)captured_attrs.at("op_1.running_mean").data.data();
        const float* bn_running_var = (const float*)captured_attrs.at("op_1.running_var").data.data();
        const float* bn_weight = has_bn_affine ? (const float*)captured_attrs.at("op_1.weight").data.data() : 0;
        const float* bn_bias = has_bn_affine ? (const float*)captured_attrs.at("op_1.bias").data.data() : 0;

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
                a[i] = bn_bias[i] - bn_weight[i] * bn_running_mean[i] / sqrt_var;
                b[i] = bn_weight[i] / sqrt_var;
            }
            else
            {
                a[i] = -bn_running_mean[i] / sqrt_var;
                b[i] = 1.f / sqrt_var;
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
            op->attrs["bias"].type = 1;
            op->attrs["bias"].shape = {channels};

            op->attrs["bias"].data.resize(channels * sizeof(float));
            memset(op->attrs["bias"].data.data(), 0, channels * sizeof(float));
        }

        float* conv_weight = (float*)op->attrs["weight"].data.data();
        float* conv_bias = (float*)op->attrs["bias"].data.data();

        const int weight_per_outch = op->params["in_features"].i;

        for (int i = 0; i < channels; i++)
        {
            float* conv_weight_outch = conv_weight + weight_per_outch * i;
            for (int j = 0; j < weight_per_outch; j++)
            {
                conv_weight_outch[j] *= b[i];
            }

            conv_bias[i] = conv_bias[i] * b[i] + a[i];
        }
    }
};

void fuse_linear_batchnorm1d(Graph& graph)
{
    fuse_linear_batchnorm1d_pass a;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace pnnx
