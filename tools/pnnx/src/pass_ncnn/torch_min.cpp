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

class torch_min : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 3
pnnx.Input              input       0 1 input
torch.min               op_0        1 2 input out indices dim=%dim keepdim=%keepdim
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reduction";
    }

    const char* name_str() const
    {
        return "min";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int dim = captured_params.at("dim").i;

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        if (dim == batch_index)
        {
            fprintf(stderr, "min along batch axis is not supported\n");
            return;
        }

        int new_dim = dim > batch_index ? dim - 1 : dim;

        op->params["0"] = 5;
        op->params["1"] = 0;
        op->params["3"] = std::vector<int>{new_dim};
        op->params["4"] = captured_params.at("keepdim").b ? 1 : 0;
        op->params["5"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_min, 20)

class torch_min_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.min               op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reduction";
    }

    const char* name_str() const
    {
        return "min";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["0"] = 5;
        op->params["1"] = 1;
        op->params["4"] = 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_min_1, 20)

} // namespace ncnn

} // namespace pnnx
