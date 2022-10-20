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

class F_grid_sample : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0       0 1 input0
pnnx.Input              input_1       0 1 input1
F.grid_sample           op_0          2 1 input0 input1 out mode=%mode padding_mode=%padding_mode align_corners=%align_corners
pnnx.Output             output        1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "GridSample";
    }

    const char* name_str() const
    {
        return "gridsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& mode = captured_params.at("mode").s;
        if (mode == "bilinear")
            op->params["0"] = 1;
        if (mode == "nearest")
            op->params["0"] = 2;
        if (mode == "bicubic")
            op->params["0"] = 3;

        const std::string& padding_mode = captured_params.at("padding_mode").s;
        if (padding_mode == "zeros")
            op->params["1"] = 1;
        if (padding_mode == "border")
            op->params["1"] = 2;
        if (padding_mode == "reflection")
            op->params["1"] = 3;

        op->params["2"] = captured_params.at("align_corners").b ? 1 : 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_grid_sample, 20)

} // namespace ncnn

} // namespace pnnx
