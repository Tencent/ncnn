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

class torch_unsqueeze : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.unsqueeze         op_0        1 1 input out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "ExpandDims";
    }

    const char* name_str() const
    {
        return "unsqueeze";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        int dim = captured_params.at("dim").i;
        if (dim == batch_index)
        {
            fprintf(stderr, "unsqueeze batch dim %d is not supported yet!\n", batch_index);
            return;
        }

        int input_rank = op->inputs[0]->shape.size();

        if (input_rank > 4)
        {
            fprintf(stderr, "unsqueeze %d-rank tensor is not supported yet!\n", input_rank);
            return;
        }

        if (dim > batch_index)
            dim -= 1;

        std::vector<int> axes = {dim};
        op->params["3"] = axes;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_unsqueeze, 20)

} // namespace ncnn

} // namespace pnnx
