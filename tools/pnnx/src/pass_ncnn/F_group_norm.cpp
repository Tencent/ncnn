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

class F_group_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.group_norm            op_0        1 1 input out weight=None bias=None num_groups=%num_groups eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "GroupNorm";
    }

    const char* name_str() const
    {
        return "gn";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int input_rank = op->inputs[0]->shape.size();

        if (input_rank <= 2)
        {
            fprintf(stderr, "group_norm not possible for %d-rank tensor\n", input_rank);
            return;
        }

        op->params["0"] = captured_params.at("num_groups");
        op->params["1"] = op->inputs[0]->shape[1];
        op->params["2"] = captured_params.at("eps");
        op->params["3"] = 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_group_norm, 20)

} // namespace ncnn

} // namespace pnnx
