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

#include "pass_level2.h"

namespace pnnx {

class F_grid_sample : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_1     0 1 input
pnnx.Input              input_2     0 1 grid
prim::Constant          op_0        0 1 mode value=%mode
prim::Constant          op_1        0 1 padding_mode value=%padding_mode
prim::Constant          op_2        0 1 align_corners value=%align_corners
aten::grid_sampler      op_3        5 1 input grid mode padding_mode align_corners out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.grid_sample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("mode").i == 0)
            op->params["mode"] = "bilinear";
        if (captured_params.at("mode").i == 1)
            op->params["mode"] = "nearest";
        if (captured_params.at("mode").i == 2)
            op->params["mode"] = "bicubic";

        if (captured_params.at("padding_mode").i == 0)
            op->params["padding_mode"] = "zeros";
        if (captured_params.at("padding_mode").i == 1)
            op->params["padding_mode"] = "border";
        if (captured_params.at("padding_mode").i == 2)
            op->params["padding_mode"] = "reflection";

        op->params["align_corners"] = captured_params.at("align_corners");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_grid_sample, 10)

} // namespace pnnx
