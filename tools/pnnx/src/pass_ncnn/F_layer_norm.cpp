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

class F_layer_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.layer_norm            op_0        1 1 input out weight=None bias=None normalized_shape=%normalized_shape eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "LayerNorm";
    }

    const char* name_str() const
    {
        return "ln";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& normalized_shape = captured_params.at("normalized_shape").ai;
        int affine_size = normalized_shape[0];
        for (size_t i = 1; i < normalized_shape.size(); i++)
        {
            affine_size *= normalized_shape[i];
        }

        op->params["0"] = affine_size;
        op->params["1"] = captured_params.at("eps");
        op->params["2"] = 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_layer_norm, 20)

class F_layer_norm_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @qwq
pnnx.Attribute          op_bias     0 1 bias @qwq
F.layer_norm            op_0        3 1 input weight bias out normalized_shape=%normalized_shape eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "LayerNorm";
    }

    const char* name_str() const
    {
        return "ln";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        Attribute weight;
        Attribute bias;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 10) == "op_weight.")
                weight = x.second;
            if (x.first.substr(0, 8) == "op_bias.")
                bias = x.second;
        }

        const std::vector<int>& normalized_shape = captured_params.at("normalized_shape").ai;
        int affine_size = normalized_shape[0];
        for (size_t i = 1; i < normalized_shape.size(); i++)
        {
            affine_size *= normalized_shape[i];
        }

        op->params["0"] = affine_size;
        op->params["1"] = captured_params.at("eps");
        op->params["2"] = 1;

        op->attrs["0"] = weight;
        op->attrs["1"] = bias;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_layer_norm_1, 20)

} // namespace ncnn

} // namespace pnnx
