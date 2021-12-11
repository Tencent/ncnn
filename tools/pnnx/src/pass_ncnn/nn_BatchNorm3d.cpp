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

class nn_BatchNorm3d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.BatchNorm3d          op_0        1 1 input out affine=%affine eps=%eps num_features=%num_features @running_mean @running_var @weight @bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "BatchNorm";
    }

    const char* name_str() const
    {
        return "bn";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = captured_params.at("num_features");
        op->params["1"] = captured_params.at("eps");

        op->attrs["1"] = captured_attrs.at("op_0.running_mean");
        op->attrs["2"] = captured_attrs.at("op_0.running_var");

        if (captured_params.at("affine").b)
        {
            op->attrs["0"] = captured_attrs.at("op_0.weight");
            op->attrs["3"] = captured_attrs.at("op_0.bias");
        }
        else
        {
            const int num_features = captured_params.at("num_features").i;
            std::vector<float> weight(num_features, 1.f);
            std::vector<float> bias(num_features, 0.f);
            op->attrs["0"] = Attribute({num_features}, weight);
            op->attrs["3"] = Attribute({num_features}, bias);
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_BatchNorm3d, 20)

} // namespace ncnn

} // namespace pnnx
