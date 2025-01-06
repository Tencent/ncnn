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

class torch_topk : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.topk              op_0        1 2 input out indices dim=%dim k=%k largest=%largest sorted=%sorted
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "TopK";
    }

    const char* name_str() const
    {
        return "topk";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int k = captured_params.at("k").i;
        int dim = captured_params.at("dim").i;
        int largest = captured_params.at("largest").b ? 1 : 0;
        int sorted = captured_params.at("sorted").b ? 1 : 0;

        // 设置参数
        op->params["0"] = k;
        op->params["1"] = dim;
        op->params["2"] = largest;
        op->params["3"] = sorted;

        // 未完成说明
        int input_rank = (int)op->inputs[0]->shape.size();
        if (input_rank == 4 && (dim == 0 || dim == 1))
        {
            printf("error: 4D with dim = 0 or 1 is not supported yet\n");
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_topk, 20)

} // namespace ncnn

} // namespace pnnx