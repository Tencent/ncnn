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

class nn_LSTM : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.LSTM                 op_0        1 1 input out input_size=%input_size hidden_size=%hidden_size num_layers=1 bias=%bias batch_first=%batch_first bidirectional=%bidirectional @weight_ih_l0 @weight_hh_l0 @bias_ih_l0 @bias_hh_l0 @weight_ih_l0_reverse @weight_hh_l0_reverse @bias_ih_l0_reverse @bias_hh_l0_reverse
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "LSTM";
    }

    const char* name_str() const
    {
        return "lstm";
    }

    void write(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs, Operator* op) const
    {
        const bool bidirectional = captured_params.at("bidirectional").b;
        const int num_directions = bidirectional ? 2 : 1;
        const int num_output = captured_params.at("hidden_size").i;
        const int input_size = captured_params.at("input_size").i;

        int weight_data_size = num_directions * num_output * input_size * 4;

        op->params["0"] = num_output;
        op->params["1"] = weight_data_size;
        op->params["2"] = bidirectional ? 2 : 0;

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};

        // reorder IFGO to IFOG
        {
            op->attrs["1"] = captured_attrs.at("op_0.weight_ih_l0");
            if (bidirectional)
                op->attrs["2"] = captured_attrs.at("op_0.weight_ih_l0_reverse");
        }

        if (captured_params.at("bias").b)
        {
            // reduce bias_ih and bias_hh
            std::vector<float> new_bias;
            {
                const float* bias_ih = (const float*)captured_attrs.at("op_0.bias_ih_l0").data.data();
                const float* bias_hh = (const float*)captured_attrs.at("op_0.bias_hh_l0").data.data();

                new_bias.resize(num_output);
                float* bias = (float*)new_bias.data();
                for (int i = 0; i < num_output; i++)
                {
                    bias[i] = bias_ih[i] + bias_hh[i];
                }
            }

            op->attrs["3"] = Attribute({num_output}, new_bias);

            if (bidirectional)
            {
                std::vector<float> new_bias_reverse;
                {
                    const float* bias_ih = (const float*)captured_attrs.at("op_0.bias_ih_l0_reverse").data.data();
                    const float* bias_hh = (const float*)captured_attrs.at("op_0.bias_hh_l0_reverse").data.data();

                    new_bias_reverse.resize(num_output);
                    float* bias = (float*)new_bias_reverse.data();
                    for (int i = 0; i < num_output; i++)
                    {
                        bias[i] = bias_ih[i] + bias_hh[i];
                    }
                }

                op->attrs["4"] = Attribute({num_output}, new_bias_reverse);
            }
        }

        op->attrs["5"] = Attribute();
        op->attrs["5"].data = {0, 0, 0, 0};

        // reorder IFGO to IFOG
        {
            op->attrs["6"] = captured_attrs.at("op_0.weight_hh_l0");
            if (bidirectional)
                op->attrs["7"] = captured_attrs.at("op_0.weight_hh_l0_reverse");
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_LSTM, 20)

} // namespace ncnn

} // namespace pnnx
