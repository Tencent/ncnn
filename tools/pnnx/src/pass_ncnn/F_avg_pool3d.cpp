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

class F_avg_pool3d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.avg_pool3d            op_0        1 1 input out kernel_size=%kernel_size stride=%stride padding=%padding ceil_mode=%ceil_mode count_include_pad=%count_include_pad divisor_override=%divisor_override
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Pooling3D";
    }

    const char* name_str() const
    {
        return "avgpool3d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("divisor_override").type != 0)
        {
            fprintf(stderr, "unsupported avgpool3d divisor_override\n");
            return;
        }

        std::vector<int> stride;
        if (captured_params.at("stride").type == 0)
        {
            stride = captured_params.at("kernel_size").ai;
        }
        else
        {
            stride = captured_params.at("stride").ai;
        }

        op->params["0"] = 1;
        op->params["1"] = captured_params.at("kernel_size").ai[2];
        op->params["11"] = captured_params.at("kernel_size").ai[1];
        op->params["21"] = captured_params.at("kernel_size").ai[0];
        op->params["2"] = stride[2];
        op->params["12"] = stride[1];
        op->params["22"] = stride[0];
        op->params["3"] = captured_params.at("padding").ai[2];
        op->params["13"] = captured_params.at("padding").ai[1];
        op->params["23"] = captured_params.at("padding").ai[0];
        op->params["5"] = captured_params.at("ceil_mode").b ? 0 : 1;
        op->params["6"] = captured_params.at("count_include_pad").b ? 1 : 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_avg_pool3d, 20)

} // namespace ncnn

} // namespace pnnx
