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

class F_upsample_bilinear : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.upsample_bilinear     op_0        1 1 input out align_corners=%align_corners size=%size
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Interp";
    }

    const char* name_str() const
    {
        return "upsample_bilinear";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& size = captured_params.at("size").ai;

        op->params["0"] = 2; // bilinear

        if (size.size() == 2)
        {
            op->params["3"] = size[0];
            op->params["4"] = size[1];
        }
        else
        {
            fprintf(stderr, "unsupported upsample_bilinear size\n");
        }

        op->params["6"] = captured_params.at("align_corners").b ? 1 : 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_upsample_bilinear, 20)

class F_upsample_bilinear_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.upsample_bilinear     op_0        1 1 input out align_corners=%align_corners scale_factor=%scale_factor
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Interp";
    }

    const char* name_str() const
    {
        return "upsample_bilinear";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<float>& scale_factor = captured_params.at("scale_factor").af;

        op->params["0"] = 2; // bilinear

        if (scale_factor.size() == 2)
        {
            op->params["1"] = scale_factor[0];
            op->params["2"] = scale_factor[1];
        }
        else
        {
            fprintf(stderr, "unsupported upsample_bilinear scale_factor\n");
        }

        op->params["6"] = captured_params.at("align_corners").b ? 1 : 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_upsample_bilinear_1, 20)

} // namespace ncnn

} // namespace pnnx
