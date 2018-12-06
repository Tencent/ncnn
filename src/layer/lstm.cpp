// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "lstm.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(LSTM)

LSTM::LSTM()
{
    one_blob_only = false;
    support_inplace = false;
}

int LSTM::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    weight_data_size = pd.get(1, 0);

    return 0;
}

int LSTM::load_model(const ModelBin& mb)
{
    int size = weight_data_size / num_output / 4;

    // raw weight data
    weight_hc_data = mb.load(size, num_output * 4, 0);
    if (weight_hc_data.empty())
        return -100;

    weight_xc_data = mb.load(size, num_output * 4, 0);
    if (weight_xc_data.empty())
        return -100;

    bias_c_data = mb.load(4, num_output, 0);
    if (bias_c_data.empty())
        return -100;

    return 0;
}

int LSTM::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // size x T
    const Mat& input_blob = bottom_blobs[0];
    size_t elemsize = input_blob.elemsize;

    // T, 0 or 1 each
    const Mat& cont_blob = bottom_blobs[1];

    int T = input_blob.h;
    int size = input_blob.w;

    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    // internal cell state
    Mat cell(num_output, 4u, opt.workspace_allocator);
    if (cell.empty())
        return -100;
    // 4 x num_output
    Mat gates(4, num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output, T, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // unroll
    for (int t=0; t<T; t++)
    {
        // clip hidden by continuation indicator
        // h_cont_{t-1} = cont_t * h_{t-1}
        // h_cont_{t-1} = h_{t-1} if cont_t == 1
        //                0       otherwise
        // calculate hidden
        // gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
        const int cont = ((const int*)cont_blob)[t];
        const float* x = input_blob.row(t);
        for (int q=0; q<num_output; q++)
        {
            float h_cont = cont ? hidden[q] : 0.f;

            const float* bias_c_data_ptr = (const float*)bias_c_data + 4 * q;
            float* gates_data = (float*)gates + 4 * q;

            // gate I F O G
            const float* weight_hc_data_I = (const float*)weight_hc_data + weight_hc_data.w * q;
            const float* weight_xc_data_I = (const float*)weight_xc_data + weight_xc_data.w * q;
            const float* weight_hc_data_F = (const float*)weight_hc_data + weight_hc_data.w * q + size;
            const float* weight_xc_data_F = (const float*)weight_xc_data + weight_xc_data.w * q + size;
            const float* weight_hc_data_O = (const float*)weight_hc_data + weight_hc_data.w * q + size*2;
            const float* weight_xc_data_O = (const float*)weight_xc_data + weight_xc_data.w * q + size*2;
            const float* weight_hc_data_G = (const float*)weight_hc_data + weight_hc_data.w * q + size*3;
            const float* weight_xc_data_G = (const float*)weight_xc_data + weight_xc_data.w * q + size*3;

            float I = bias_c_data_ptr[0];
            float F = bias_c_data_ptr[1];
            float O = bias_c_data_ptr[2];
            float G = bias_c_data_ptr[3];
            for (int i=0; i<size; i++)
            {
                I += weight_hc_data_I[i] * h_cont + weight_xc_data_I[i] * x[i];
                F += weight_hc_data_F[i] * h_cont + weight_xc_data_F[i] * x[i];
                O += weight_hc_data_O[i] * h_cont + weight_xc_data_O[i] * x[i];
                G += weight_hc_data_G[i] * h_cont + weight_xc_data_G[i] * x[i];
            }

            gates_data[0] = I;
            gates_data[1] = F;
            gates_data[2] = O;
            gates_data[3] = G;
        }

        // lstm unit
        // sigmoid(I)
        // sigmoid(F)
        // sigmoid(O)
        // tanh(G)
        // c_t := f_t .* c_{t-1} + i_t .* g_t
        // h_t := o_t .* tanh[c_t]
        float* output_data = top_blob.row(t);
        for (int q=0; q<num_output; q++)
        {
            float* gates_data = (float*)gates + 4 * q;

            float I = gates_data[0];
            float F = gates_data[1];
            float O = gates_data[2];
            float G = gates_data[3];

            I = 1.f / (1.f + exp(-I));
            F = cont ? 1.f / (1.f + exp(-F)) : 0.f;
            O = 1.f / (1.f + exp(-O));
            G = tanh(G);

            float cell2 = F * cell[q] + I * G;
            float H = O * tanh(cell2);

            cell[q] = cell2;
            hidden[q] = H;
            output_data[q] = H;
        }

        // no cell output here
    }

    return 0;
}

} // namespace ncnn
