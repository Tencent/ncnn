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

class F_pairwise_distance : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input0
pnnx.Input              input_1     0 1 input1
F.pairwise_distance     op_0        2 1 input0 input1 out p=%p eps=%eps keepdim=%keepdim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "PairwiseDistance";
    }

    const char* name_str() const
    {
        return "pairwisedistance";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["0"] = captured_params.at("p").f;
        op->params["1"] = captured_params.at("eps").f;
		op->params["2"] = captured_params.at("keepdim").b ? 1 : 0;

    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_pairwise_distance, 20)

} // namespace ncnn

} // namespace pnnx
