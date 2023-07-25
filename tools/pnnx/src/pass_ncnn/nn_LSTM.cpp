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
#include <string.h>

namespace pnnx {

namespace ncnn {

class nn_LSTM : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 4
pnnx.Input              input       0 1 input
nn.LSTM                 op_0        1 3 input out out_hidden out_cell input_size=%input_size hidden_size=%hidden_size num_layers=1 bias=%bias batch_first=%batch_first bidirectional=%bidirectional proj_size=%proj_size @weight_ih_l0 @weight_hh_l0 @bias_ih_l0 @bias_hh_l0 @weight_hr_l0 @weight_ih_l0_reverse @weight_hh_l0_reverse @bias_ih_l0_reverse @bias_hh_l0_reverse @weight_hr_l0_reverse
pnnx.Output             output      3 0 out out_hidden out_cell
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

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const bool bidirectional = captured_params.at("bidirectional").b;
        const int num_directions = bidirectional ? 2 : 1;
        const int hidden_size = captured_params.at("hidden_size").i;
        const int input_size = captured_params.at("input_size").i;

        int proj_size = captured_params.at("proj_size").i;
        if (proj_size == 0)
            proj_size = hidden_size;

        int weight_data_size = num_directions * hidden_size * input_size * 4;

        op->params["0"] = proj_size;
        op->params["1"] = weight_data_size;
        op->params["2"] = bidirectional ? 2 : 0;
        op->params["3"] = hidden_size;

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};

        // reorder IFGO-hidden-input_size to IFOG-hidden-input_size
        {
            std::vector<float> new_weight_ih;
            {
                const int weight_data_size_g = hidden_size * input_size;

                auto weight_ih = captured_attrs.at("op_0.weight_ih_l0").get_float32_data();
                const float* iptr = (const float*)weight_ih.data();
                const float* fptr = (const float*)weight_ih.data() + weight_data_size_g;
                const float* gptr = (const float*)weight_ih.data() + weight_data_size_g * 2;
                const float* optr = (const float*)weight_ih.data() + weight_data_size_g * 3;

                new_weight_ih.resize(4 * hidden_size * input_size);
                float* weight = (float*)new_weight_ih.data();
                float* w_iptr = weight;
                float* w_fptr = weight + weight_data_size_g;
                float* w_optr = weight + weight_data_size_g * 2;
                float* w_gptr = weight + weight_data_size_g * 3;
                memcpy(w_iptr, iptr, weight_data_size_g * sizeof(float));
                memcpy(w_fptr, fptr, weight_data_size_g * sizeof(float));
                memcpy(w_optr, optr, weight_data_size_g * sizeof(float));
                memcpy(w_gptr, gptr, weight_data_size_g * sizeof(float));
            }

            if (bidirectional)
            {
                std::vector<float> new_weight_ih_reverse;
                {
                    const int weight_data_size_g = hidden_size * input_size;

                    auto weight_ih = captured_attrs.at("op_0.weight_ih_l0_reverse").get_float32_data();
                    const float* iptr = (const float*)weight_ih.data();
                    const float* fptr = (const float*)weight_ih.data() + weight_data_size_g;
                    const float* gptr = (const float*)weight_ih.data() + weight_data_size_g * 2;
                    const float* optr = (const float*)weight_ih.data() + weight_data_size_g * 3;

                    new_weight_ih_reverse.resize(4 * hidden_size * input_size);
                    float* weight = (float*)new_weight_ih_reverse.data();
                    float* w_iptr = weight;
                    float* w_fptr = weight + weight_data_size_g;
                    float* w_optr = weight + weight_data_size_g * 2;
                    float* w_gptr = weight + weight_data_size_g * 3;
                    memcpy(w_iptr, iptr, weight_data_size_g * sizeof(float));
                    memcpy(w_fptr, fptr, weight_data_size_g * sizeof(float));
                    memcpy(w_optr, optr, weight_data_size_g * sizeof(float));
                    memcpy(w_gptr, gptr, weight_data_size_g * sizeof(float));
                }
                op->attrs["1"] = Attribute({4, hidden_size, input_size}, new_weight_ih) + Attribute({4, hidden_size, input_size}, new_weight_ih_reverse);
            }
            else
            {
                op->attrs["1"] = Attribute({4, hidden_size, input_size}, new_weight_ih);
            }
        }

        op->attrs["2"] = Attribute();
        op->attrs["2"].data = {0, 0, 0, 0};
        if (captured_params.at("bias").b)
        {
            // reduce bias_ih and bias_hh
            // reorder IFGO-hidden to IFOG-hidden
            std::vector<float> new_bias;
            {
                auto bias_ih = captured_attrs.at("op_0.bias_ih_l0").get_float32_data();
                auto bias_hh = captured_attrs.at("op_0.bias_hh_l0").get_float32_data();
                const float* bias_ih_iptr = (const float*)bias_ih.data();
                const float* bias_ih_fptr = (const float*)bias_ih.data() + hidden_size;
                const float* bias_ih_gptr = (const float*)bias_ih.data() + hidden_size * 2;
                const float* bias_ih_optr = (const float*)bias_ih.data() + hidden_size * 3;
                const float* bias_hh_iptr = (const float*)bias_hh.data();
                const float* bias_hh_fptr = (const float*)bias_hh.data() + hidden_size;
                const float* bias_hh_gptr = (const float*)bias_hh.data() + hidden_size * 2;
                const float* bias_hh_optr = (const float*)bias_hh.data() + hidden_size * 3;

                new_bias.resize(4 * hidden_size);
                float* bias = (float*)new_bias.data();
                float* b_iptr = bias;
                float* b_fptr = bias + hidden_size;
                float* b_optr = bias + hidden_size * 2;
                float* b_gptr = bias + hidden_size * 3;
                for (int i = 0; i < hidden_size; i++)
                {
                    b_iptr[i] = bias_ih_iptr[i] + bias_hh_iptr[i];
                }
                for (int i = 0; i < hidden_size; i++)
                {
                    b_fptr[i] = bias_ih_fptr[i] + bias_hh_fptr[i];
                }
                for (int i = 0; i < hidden_size; i++)
                {
                    b_optr[i] = bias_ih_optr[i] + bias_hh_optr[i];
                }
                for (int i = 0; i < hidden_size; i++)
                {
                    b_gptr[i] = bias_ih_gptr[i] + bias_hh_gptr[i];
                }
            }

            if (bidirectional)
            {
                std::vector<float> new_bias_reverse;
                {
                    auto bias_ih = captured_attrs.at("op_0.bias_ih_l0_reverse").get_float32_data();
                    auto bias_hh = captured_attrs.at("op_0.bias_hh_l0_reverse").get_float32_data();
                    const float* bias_ih_iptr = (const float*)bias_ih.data();
                    const float* bias_ih_fptr = (const float*)bias_ih.data() + hidden_size;
                    const float* bias_ih_gptr = (const float*)bias_ih.data() + hidden_size * 2;
                    const float* bias_ih_optr = (const float*)bias_ih.data() + hidden_size * 3;
                    const float* bias_hh_iptr = (const float*)bias_hh.data();
                    const float* bias_hh_fptr = (const float*)bias_hh.data() + hidden_size;
                    const float* bias_hh_gptr = (const float*)bias_hh.data() + hidden_size * 2;
                    const float* bias_hh_optr = (const float*)bias_hh.data() + hidden_size * 3;

                    new_bias_reverse.resize(4 * hidden_size);
                    float* bias = (float*)new_bias_reverse.data();
                    float* b_iptr = bias;
                    float* b_fptr = bias + hidden_size;
                    float* b_optr = bias + hidden_size * 2;
                    float* b_gptr = bias + hidden_size * 3;
                    for (int i = 0; i < hidden_size; i++)
                    {
                        b_iptr[i] = bias_ih_iptr[i] + bias_hh_iptr[i];
                    }
                    for (int i = 0; i < hidden_size; i++)
                    {
                        b_fptr[i] = bias_ih_fptr[i] + bias_hh_fptr[i];
                    }
                    for (int i = 0; i < hidden_size; i++)
                    {
                        b_optr[i] = bias_ih_optr[i] + bias_hh_optr[i];
                    }
                    for (int i = 0; i < hidden_size; i++)
                    {
                        b_gptr[i] = bias_ih_gptr[i] + bias_hh_gptr[i];
                    }
                }

                op->attrs["3"] = Attribute({4, hidden_size}, new_bias) + Attribute({4, hidden_size}, new_bias_reverse);
            }
            else
            {
                op->attrs["3"] = Attribute({4, hidden_size}, new_bias);
            }
        }
        else
        {
            std::vector<float> bias(4 * hidden_size, 0.f);

            if (bidirectional)
                op->attrs["3"] = Attribute({4, hidden_size}, bias) + Attribute({4, hidden_size}, bias);
            else
                op->attrs["3"] = Attribute({4, hidden_size}, bias);
        }

        op->attrs["4"] = Attribute();
        op->attrs["4"].data = {0, 0, 0, 0};

        // reorder IFGO-hidden-proj to IFOG-hidden-proj
        {
            std::vector<float> new_weight_hh;
            {
                const int weight_data_size_g = hidden_size * proj_size;

                auto weight_hh = captured_attrs.at("op_0.weight_hh_l0").get_float32_data();
                const float* iptr = (const float*)weight_hh.data();
                const float* fptr = (const float*)weight_hh.data() + weight_data_size_g;
                const float* gptr = (const float*)weight_hh.data() + weight_data_size_g * 2;
                const float* optr = (const float*)weight_hh.data() + weight_data_size_g * 3;

                new_weight_hh.resize(4 * hidden_size * proj_size);
                float* weight = (float*)new_weight_hh.data();
                float* w_iptr = weight;
                float* w_fptr = weight + weight_data_size_g;
                float* w_optr = weight + weight_data_size_g * 2;
                float* w_gptr = weight + weight_data_size_g * 3;
                memcpy(w_iptr, iptr, weight_data_size_g * sizeof(float));
                memcpy(w_fptr, fptr, weight_data_size_g * sizeof(float));
                memcpy(w_optr, optr, weight_data_size_g * sizeof(float));
                memcpy(w_gptr, gptr, weight_data_size_g * sizeof(float));
            }

            if (bidirectional)
            {
                std::vector<float> new_weight_hh_reverse;
                {
                    const int weight_data_size_g = hidden_size * proj_size;

                    auto weight_hh = captured_attrs.at("op_0.weight_hh_l0_reverse").get_float32_data();
                    const float* iptr = (const float*)weight_hh.data();
                    const float* fptr = (const float*)weight_hh.data() + weight_data_size_g;
                    const float* gptr = (const float*)weight_hh.data() + weight_data_size_g * 2;
                    const float* optr = (const float*)weight_hh.data() + weight_data_size_g * 3;

                    new_weight_hh_reverse.resize(4 * hidden_size * proj_size);
                    float* weight = (float*)new_weight_hh_reverse.data();
                    float* w_iptr = weight;
                    float* w_fptr = weight + weight_data_size_g;
                    float* w_optr = weight + weight_data_size_g * 2;
                    float* w_gptr = weight + weight_data_size_g * 3;
                    memcpy(w_iptr, iptr, weight_data_size_g * sizeof(float));
                    memcpy(w_fptr, fptr, weight_data_size_g * sizeof(float));
                    memcpy(w_optr, optr, weight_data_size_g * sizeof(float));
                    memcpy(w_gptr, gptr, weight_data_size_g * sizeof(float));
                }
                op->attrs["5"] = Attribute({4, hidden_size, proj_size}, new_weight_hh) + Attribute({4, hidden_size, proj_size}, new_weight_hh_reverse);
            }
            else
            {
                op->attrs["5"] = Attribute({4, hidden_size, proj_size}, new_weight_hh);
            }
        }

        if (proj_size != hidden_size)
        {
            op->attrs["6"] = Attribute();
            op->attrs["6"].data = {0, 0, 0, 0};

            if (bidirectional)
            {
                op->attrs["7"] = captured_attrs.at("op_0.weight_hr_l0") + captured_attrs.at("op_0.weight_hr_l0_reverse");
            }
            else
            {
                op->attrs["7"] = captured_attrs.at("op_0.weight_hr_l0");
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_LSTM, 20)

class nn_LSTM_1 : public nn_LSTM
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 6
pnnx.Input              input       0 1 input
pnnx.Input              in_hidden   0 1 in_hidden
pnnx.Input              in_hidden   0 1 in_cell
nn.LSTM                 op_0        3 3 input in_hidden in_cell out out_hidden out_cell input_size=%input_size hidden_size=%hidden_size num_layers=1 bias=%bias batch_first=%batch_first bidirectional=%bidirectional proj_size=%proj_size @weight_ih_l0 @weight_hh_l0 @bias_ih_l0 @bias_hh_l0 @weight_hr_l0 @weight_ih_l0_reverse @weight_hh_l0_reverse @bias_ih_l0_reverse @bias_hh_l0_reverse @weight_hr_l0_reverse
pnnx.Output             output      3 0 out out_hidden out_cell
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_LSTM_1, 20)

class nn_LSTM_2 : public nn_LSTM
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.LSTM                 op_0        1 1 input out input_size=%input_size hidden_size=%hidden_size num_layers=1 bias=%bias batch_first=%batch_first bidirectional=%bidirectional proj_size=%proj_size @weight_ih_l0 @weight_hh_l0 @bias_ih_l0 @bias_hh_l0 @weight_hr_l0 @weight_ih_l0_reverse @weight_hh_l0_reverse @bias_ih_l0_reverse @bias_hh_l0_reverse @weight_hr_l0_reverse
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_LSTM_2, 20)

class nn_LSTM_3 : public nn_LSTM
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Input              in_hidden   0 1 in_hidden
pnnx.Input              in_hidden   0 1 in_cell
nn.LSTM                 op_0        3 1 input in_hidden in_cell out input_size=%input_size hidden_size=%hidden_size num_layers=1 bias=%bias batch_first=%batch_first bidirectional=%bidirectional proj_size=%proj_size @weight_ih_l0 @weight_hh_l0 @bias_ih_l0 @bias_hh_l0 @weight_hr_l0 @weight_ih_l0_reverse @weight_hh_l0_reverse @bias_ih_l0_reverse @bias_hh_l0_reverse @weight_hr_l0_reverse
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_LSTM_3, 20)

} // namespace ncnn

} // namespace pnnx
