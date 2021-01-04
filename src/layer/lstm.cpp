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

LSTM::LSTM()
{
    one_blob_only = false;
    support_inplace = false;
}

int LSTM::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    weight_data_size = pd.get(1, 0);
    direction = pd.get(2, 0);
    if (direction == 2)
        one_blob_only = true;
    return 0;
}

int LSTM::load_model(const ModelBin& mb)
{
    int num_directions = direction == 2 ? 2 : 1;

    int size = weight_data_size / num_directions / num_output / 4;

    // raw weight data
    weight_xc_data = mb.load(size, num_output * 4, num_directions, 0);
    if (weight_xc_data.empty())
        return -100;

    bias_c_data = mb.load(num_output, 4, num_directions, 0);
    if (bias_c_data.empty())
        return -100;

    weight_hc_data = mb.load(num_output, num_output * 4, num_directions, 0);
    if (weight_hc_data.empty())
        return -100;

    return 0;
}

static int lstm(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // 4 x num_output
    Mat gates(4, num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        // clip hidden by continuation indicator
        // h_cont_{t-1} = cont_t * h_{t-1}
        // h_cont_{t-1} = h_{t-1} if cont_t == 1
        //                0       otherwise
        // calculate hidden
        // gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c

        int ti = reverse ? T - 1 - t : t;

        const float* x = bottom_blob.row(ti);
        for (int q = 0; q < num_output; q++)
        {
            const float* bias_c_I = bias_c.row(0);
            const float* bias_c_F = bias_c.row(1);
            const float* bias_c_O = bias_c.row(2);
            const float* bias_c_G = bias_c.row(3);

            float* gates_data = gates.row(q);

            // gate I F O G
            const float* weight_xc_I = weight_xc.row(num_output * 0 + q);
            const float* weight_xc_F = weight_xc.row(num_output * 1 + q);
            const float* weight_xc_O = weight_xc.row(num_output * 2 + q);
            const float* weight_xc_G = weight_xc.row(num_output * 3 + q);

            const float* weight_hc_I = weight_hc.row(num_output * 0 + q);
            const float* weight_hc_F = weight_hc.row(num_output * 1 + q);
            const float* weight_hc_O = weight_hc.row(num_output * 2 + q);
            const float* weight_hc_G = weight_hc.row(num_output * 3 + q);

            float I = bias_c_I[q];
            float F = bias_c_F[q];
            float O = bias_c_O[q];
            float G = bias_c_G[q];

            for (int i = 0; i < size; i++)
            {
                float xi = x[i];

                I += weight_xc_I[i] * xi;
                F += weight_xc_F[i] * xi;
                O += weight_xc_O[i] * xi;
                G += weight_xc_G[i] * xi;
            }

            for (int i = 0; i < num_output; i++)
            {
                float h_cont = hidden_state[i];

                I += weight_hc_I[i] * h_cont;
                F += weight_hc_F[i] * h_cont;
                O += weight_hc_O[i] * h_cont;
                G += weight_hc_G[i] * h_cont;
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
        float* output_data = top_blob.row(ti);
        for (int q = 0; q < num_output; q++)
        {
            const float* gates_data = gates.row(q);

            float I = gates_data[0];
            float F = gates_data[1];
            float O = gates_data[2];
            float G = gates_data[3];

            I = 1.f / (1.f + exp(-I));
            F = 1.f / (1.f + exp(-F));
            O = 1.f / (1.f + exp(-O));
            G = tanh(G);

            float cell2 = F * cell_state[q] + I * G;
            float H = O * tanh(cell2);
            cell_state[q] = cell2;
            hidden_state[q] = H;
            output_data[q] = H;
        }
    }

    return 0;
}

int LSTM::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int T = bottom_blob.h;

    int num_directions = direction == 2 ? 2 : 1;

    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    Mat cell(num_output, 4u, opt.workspace_allocator);
    if (cell.empty())
        return -100;
    cell.fill(0.f);

    top_blob.create(num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = lstm(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, cell, opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        int ret0 = lstm(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, cell, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.0f);
        cell.fill(0.0f);

        int ret1 = lstm(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), hidden, cell, opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i = 0; i < T; i++)
        {
            const float* pf = top_blob_forward.row(i);
            const float* pr = top_blob_reverse.row(i);
            float* ptr = top_blob.row(i);

            memcpy(ptr, pf, num_output * sizeof(float));
            memcpy(ptr + num_output, pr, num_output * sizeof(float));
        }
    }

    return 0;
}

int LSTM::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() != 3 || top_blobs.size() != 3)
    {
        return forward(bottom_blobs[0], top_blobs[0], opt);
    }
    const Mat& bottom_blob = bottom_blobs[0];
    int T = bottom_blob.h;
    Mat& top_blob = top_blobs[0];
    Mat& hidden_state = top_blobs[1];
    Mat& cell_state = top_blobs[2];

    //Copy previous states
    hidden_state = bottom_blobs[1].clone(opt.blob_allocator);
    cell_state = bottom_blobs[2].clone(opt.blob_allocator);

    top_blob.create(num_output, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = lstm(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden_state, cell_state, opt);
        if (ret != 0)
            return ret;
    }

    return 0;
}

} // namespace ncnn
