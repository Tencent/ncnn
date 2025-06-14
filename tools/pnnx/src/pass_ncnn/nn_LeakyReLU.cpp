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

class nn_LeakyReLU : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.LeakyReLU            op_0        1 1 input out negative_slope=%negative_slope
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "ReLU";
    }

    const char* name_str() const
    {
        return "leakyrelu";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        float negative_slope = 0.f;

        if (captured_params.at("negative_slope").type == 2)
        {
            negative_slope = (float)captured_params.at("negative_slope").i;
        }
        if (captured_params.at("negative_slope").type == 3)
        {
            negative_slope = captured_params.at("negative_slope").f;
        }

        op->params["0"] = negative_slope;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_LeakyReLU, 20)

} // namespace ncnn

} // namespace pnnx
