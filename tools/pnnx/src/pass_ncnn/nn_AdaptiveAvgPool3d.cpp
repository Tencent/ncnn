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

class nn_AdaptiveAvgPool3d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.AdaptiveAvgPool3d    op_0        1 1 input out output_size=(1,1,1)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Pooling3D";
    }

    const char* name_str() const
    {
        return "gap3d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["0"] = 1;
        op->params["4"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_AdaptiveAvgPool3d, 20)

class nn_AdaptiveAvgPool3d_n : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.AdaptiveAvgPool3d    op_0        1 1 input out output_size=%output_size
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Pooling3D";
    }

    const char* name_str() const
    {
        return "aap3d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& output_size = captured_params.at("output_size").ai;

        op->params["0"] = 1;
        op->params["7"] = 1;
        op->params["8"] = output_size[2] == 0 ? -233 : output_size[2];
        op->params["18"] = output_size[1] == 0 ? -233 : output_size[1];
        op->params["28"] = output_size[0] == 0 ? -233 : output_size[0];
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_AdaptiveAvgPool3d_n, 21)

} // namespace ncnn

} // namespace pnnx
