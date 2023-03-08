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

class F_max_pool1d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.max_pool1d            op_0        1 1 input out kernel_size=%kernel_size stride=%stride dilation=(1) padding=%padding ceil_mode=%ceil_mode return_indices=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Pooling1D";
    }

    const char* name_str() const
    {
        return "maxpool1d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::vector<int> stride;
        if (captured_params.at("stride").type == 0)
        {
            stride = captured_params.at("kernel_size").ai;
        }
        else
        {
            stride = captured_params.at("stride").ai;
        }

        op->params["0"] = 0;
        op->params["1"] = captured_params.at("kernel_size").ai[0];
        op->params["2"] = stride[0];
        op->params["3"] = captured_params.at("padding").ai[0];
        op->params["5"] = captured_params.at("ceil_mode").b ? 0 : 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_max_pool1d, 20)

} // namespace ncnn

} // namespace pnnx
