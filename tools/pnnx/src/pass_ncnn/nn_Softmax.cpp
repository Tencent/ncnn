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

class nn_Softmax : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Softmax              op_0        1 1 input out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Softmax";
    }

    const char* name_str() const
    {
        return "softmax";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        int axis = captured_params.at("dim").i;
        if (axis == batch_index)
        {
            fprintf(stderr, "softmax along batch axis %d is not supported\n", batch_index);
            return;
        }

        if (axis < 0)
        {
            int input_rank = op->inputs[0]->shape.size();
            axis = input_rank + axis;
        }

        if (axis > batch_index)
            axis -= 1;

        op->params["0"] = axis;
        op->params["1"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Softmax, 20)

} // namespace ncnn

} // namespace pnnx
