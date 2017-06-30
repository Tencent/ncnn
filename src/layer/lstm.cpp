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

LSTM::~LSTM()
{
}

#if NCNN_STDIO
#if NCNN_STRING
int LSTM::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d", &num_output, &weight_data_size);
    if (nscan != 2)
    {
        fprintf(stderr, "LSTM load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int LSTM::load_param_bin(FILE* paramfp)
{
    fread(&num_output, sizeof(int), 1, paramfp);

    fread(&weight_data_size, sizeof(int), 1, paramfp);

    return 0;
}

int LSTM::load_model(FILE* binfp)
{
    int nread;

    int size = weight_data_size / 2 / num_output / 4;

    // raw weight data
    weight_hc_data.create(size * 4, num_output);
    if (weight_hc_data.empty())
        return -100;
    nread = fread(weight_hc_data.data, size * 4 * num_output * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "LSTM read weight_hc_data failed %d\n", nread);
        return -1;
    }

    weight_xc_data.create(size * 4, num_output);
    if (weight_xc_data.empty())
        return -100;
    nread = fread(weight_xc_data.data, size * 4 * num_output * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "LSTM read weight_xc_data failed %d\n", nread);
        return -1;
    }

    bias_c_data.create(4, num_output);
    if (bias_c_data.empty())
        return -100;
    nread = fread(bias_c_data.data, 4 * num_output * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "LSTM read bias_c_data failed %d\n", nread);
        return -1;
    }

    return 0;
}
#endif // NCNN_STDIO

int LSTM::load_param(const unsigned char*& mem)
{
    num_output = *(int*)(mem);
    mem += 4;

    weight_data_size = *(int*)(mem);
    mem += 4;

    return 0;
}

int LSTM::load_model(const unsigned char*& mem)
{
    int size = weight_data_size / 2 / num_output / 4;

    // raw weight data
    weight_hc_data = Mat(size * 4, num_output, (float*)mem);
    mem += size * 4 * num_output * sizeof(float);

    weight_xc_data = Mat(size * 4, num_output, (float*)mem);
    mem += size * 4 * num_output * sizeof(float);

    bias_c_data = Mat(4, num_output, (float*)mem);
    mem += 4 * num_output * sizeof(float);

    return 0;
}

int LSTM::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    // size x 1 x T
    const Mat& input_blob = bottom_blobs[0];

    // T, 0 or 1 each
    const Mat& cont_blob = bottom_blobs[1];

    int T = input_blob.c;
    int size = input_blob.w;

    // initial hidden state
    Mat hidden(num_output);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    // internal cell state
    Mat cell(num_output);
    if (cell.empty())
        return -100;
    // 4 x num_output
    Mat gates(4, num_output);
    if (gates.empty())
        return -100;

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output, 1, T);
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
        const float cont = cont_blob.data[t];
        const Mat x = input_blob.channel(t);
        float* hidden_data = hidden;
        for (int q=0; q<num_output; q++)
        {
            float h_cont = cont ? hidden_data[q] : 0.f;

            const float* x_data = x;
            const float* bias_c_data_ptr = bias_c_data.data + 4 * q;
            float* gates_data = gates.data + 4 * q;

            // gate I F O G
            const float* weight_hc_data_I = weight_hc_data.data + weight_hc_data.w * q;
            const float* weight_xc_data_I = weight_xc_data.data + weight_xc_data.w * q;
            const float* weight_hc_data_F = weight_hc_data.data + weight_hc_data.w * q + size;
            const float* weight_xc_data_F = weight_xc_data.data + weight_xc_data.w * q + size;
            const float* weight_hc_data_O = weight_hc_data.data + weight_hc_data.w * q + size*2;
            const float* weight_xc_data_O = weight_xc_data.data + weight_xc_data.w * q + size*2;
            const float* weight_hc_data_G = weight_hc_data.data + weight_hc_data.w * q + size*3;
            const float* weight_xc_data_G = weight_xc_data.data + weight_xc_data.w * q + size*3;

            float I = bias_c_data_ptr[0];
            float F = bias_c_data_ptr[1];
            float O = bias_c_data_ptr[2];
            float G = bias_c_data_ptr[3];
            for (int i=0; i<size; i++)
            {
                I += weight_hc_data_I[i] * h_cont + weight_xc_data_I[i] * x_data[i];
                F += weight_hc_data_F[i] * h_cont + weight_xc_data_F[i] * x_data[i];
                O += weight_hc_data_O[i] * h_cont + weight_xc_data_O[i] * x_data[i];
                G += weight_hc_data_G[i] * h_cont + weight_xc_data_G[i] * x_data[i];
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
        float* cell_data = cell;
        Mat output = top_blob.channel(t);
        float* output_data = output;
        for (int q=0; q<num_output; q++)
        {
            float* gates_data = gates.data + 4 * q;

            float I = gates_data[0];
            float F = gates_data[1];
            float O = gates_data[2];
            float G = gates_data[3];

            I = 1.f / (1.f + exp(-I));
            F = cont ? 0.f : 1.f / (1.f + exp(-F));
            O = 1.f / (1.f + exp(-O));
            G = tanh(G);

            float cell = F * cell_data[q] + I * G;
            float H = O * tanh(cell);

            cell_data[q] = cell;
            hidden_data[q] = H;
            output_data[q] = H;
        }

        // no cell output here
    }

    return 0;
}

} // namespace ncnn
