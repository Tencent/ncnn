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

class F_batch_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @qwq
pnnx.Attribute          op_var      0 1 running_var @qwq
F.batch_norm            op_0        3 1 input running_mean running_var out weight=None bias=None eps=%eps
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
        Attribute running_mean;
        Attribute running_var;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 8) == "op_mean.")
                running_mean = x.second;
            if (x.first.substr(0, 7) == "op_var.")
                running_var = x.second;
        }

        op->params["0"] = running_mean.shape[0];
        op->params["1"] = captured_params.at("eps");

        const int channels = running_mean.shape[0];

        op->attrs["0"] = Attribute({channels}, std::vector<float>(channels, 1.f));
        op->attrs["1"] = running_mean;
        op->attrs["2"] = running_var;
        op->attrs["3"] = Attribute({channels}, std::vector<float>(channels, 0.f));
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_batch_norm, 20)

class F_batch_norm_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
pnnx.Attribute          op_mean     0 1 running_mean @qwq
pnnx.Attribute          op_var      0 1 running_var @qwq
pnnx.Attribute          op_weight   0 1 weight @qwq
pnnx.Attribute          op_bias     0 1 bias @qwq
F.batch_norm            op_0        5 1 input running_mean running_var weight bias out eps=%eps
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
        Attribute running_mean;
        Attribute running_var;
        Attribute weight;
        Attribute bias;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 8) == "op_mean.")
                running_mean = x.second;
            if (x.first.substr(0, 7) == "op_var.")
                running_var = x.second;
            if (x.first.substr(0, 10) == "op_weight.")
                weight = x.second;
            if (x.first.substr(0, 8) == "op_bias.")
                bias = x.second;
        }

        op->params["0"] = running_mean.shape[0];
        op->params["1"] = captured_params.at("eps");

        op->attrs["0"] = weight;
        op->attrs["1"] = running_mean;
        op->attrs["2"] = running_var;
        op->attrs["3"] = bias;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_batch_norm_1, 20)

} // namespace ncnn

} // namespace pnnx
