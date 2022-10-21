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

class F_normalize : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.normalize             op_0        1 1 input out dim=%dim eps=%eps p=%p
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Normalize";
    }

    const char* name_str() const
    {
        return "normalize";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        int axis = captured_params.at("dim").i;
        if (axis == batch_index)
        {
            fprintf(stderr, "normalize along batch axis %d is not supported\n", batch_index);
            return;
        }

        if (axis < 0)
        {
            int input_rank = op->inputs[0]->shape.size();
            axis = input_rank + axis;
        }

        if (axis > batch_index)
            axis -= 1;

        float p = 0.f;
        if (captured_params.at("p").type == 2)
            p = captured_params.at("p").i;
        if (captured_params.at("p").type == 3)
            p = captured_params.at("p").f;

        if (p != 2.f)
        {
            fprintf(stderr, "unsupported normalize p=%f\n", p);
            return;
        }

        int input_rank = op->inputs[0]->shape.size();

        if (batch_index >= 0 && batch_index < input_rank)
            input_rank -= 1;

        if (input_rank == 2 || axis != 0)
        {
            fprintf(stderr, "unsupported normalize for %d-rank tensor with axis %d\n", input_rank, axis);
            return;
        }

        if (input_rank == 1 && axis == 0)
        {
            op->params["0"] = 1; // across_spatial
            op->params["4"] = 1; // across_channel
        }

        if (input_rank == 3 && axis == 0)
        {
            op->params["0"] = 0; // across_spatial
            op->params["4"] = 1; // across_channel
        }

        op->params["1"] = 1; // channel_shared
        op->params["2"] = captured_params.at("eps");
        op->params["3"] = 1; // scale_data_size
        op->params["9"] = 1; // eps_mode

        op->attrs["0"] = Attribute({1}, std::vector<float>(1, 1.f));
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_normalize, 20)

} // namespace ncnn

} // namespace pnnx
