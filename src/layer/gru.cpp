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

#include "gru.h"

#include <math.h>

namespace ncnn {

GRU::GRU()
{
    one_blob_only = false;
    support_inplace = false;
}

int GRU::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    weight_data_size = pd.get(1, 0);
    direction = pd.get(2, 0);
    if (direction == 2)
        one_blob_only = true;
    return 0;
}

int GRU::load_model(const ModelBin& mb)
{
    int num_directions = direction == 2 ? 2 : 1;

    int size = weight_data_size / num_directions / num_output / 3;

    // raw weight data
    weight_xc_data = mb.load(size, num_output * 3, num_directions, 0);
    if (weight_xc_data.empty())
        return -100;

    bias_c_data = mb.load(num_output, 4, num_directions, 0);
    if (bias_c_data.empty())
        return -100;

    weight_hc_data = mb.load(num_output, num_output * 3, num_directions, 0);
    if (weight_hc_data.empty())
        return -100;

    return 0;
}

static int gru(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // 2 x num_output
    Mat gates(2, num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        const float* x = bottom_blob.row(ti);
        for (int q = 0; q < num_output; q++)
        {
            float* gates_data = gates.row(q);

            // gate reset update
            const float* bias_c_R = bias_c.row(0);
            const float* bias_c_U = bias_c.row(1);

            const float* weight_xc_R = weight_xc.row(num_output * 0 + q);
            const float* weight_xc_U = weight_xc.row(num_output * 1 + q);
            const float* weight_hc_R = weight_hc.row(num_output * 0 + q);
            const float* weight_hc_U = weight_hc.row(num_output * 1 + q);

            float R = bias_c_R[q];
            float U = bias_c_U[q];

            for (int i = 0; i < size; i++)
            {
                float xi = x[i];

                R += weight_xc_R[i] * xi;
                U += weight_xc_U[i] * xi;
            }

            for (int i = 0; i < num_output; i++)
            {
                float h_cont = hidden_state[i];

                R += weight_hc_R[i] * h_cont;
                U += weight_hc_U[i] * h_cont;
            }

            // sigmoid(R)
            // sigmoid(U)
            R = 1.f / (1.f + exp(-R));
            U = 1.f / (1.f + exp(-U));

            // gate new
            const float* bias_c_WN = bias_c.row(2);
            const float* bias_c_BN = bias_c.row(3);

            const float* weight_xc_N = weight_xc.row(num_output * 2 + q);
            const float* weight_hc_N = weight_hc.row(num_output * 2 + q);

            float N = bias_c_BN[q];

            for (int i = 0; i < num_output; i++)
            {
                float h_cont = hidden_state[i];

                N += weight_hc_N[i] * h_cont;
            }

            N = bias_c_WN[q] + R * N;

            for (int i = 0; i < size; i++)
            {
                float xi = x[i];

                N += weight_xc_N[i] * xi;
            }

            // tanh(N)
            N = tanh(N);

            gates_data[0] = U;
            gates_data[1] = N;
        }

        // h_t := (1 - update) .* new + update .* h_{t-1}
        float* output_data = top_blob.row(ti);
        for (int q = 0; q < num_output; q++)
        {
            const float* gates_data = gates.row(q);

            float U = gates_data[0];
            float N = gates_data[1];

            float H = (1 - U) * N + U * hidden_state[q];

            hidden_state[q] = H;
            output_data[q] = H;
        }
    }

    return 0;
}

int GRU::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int T = bottom_blob.h;

    int num_directions = direction == 2 ? 2 : 1;

    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    top_blob.create(num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = gru(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, opt);
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

        int ret0 = gru(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.0f);

        int ret1 = gru(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), hidden, opt);
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

int GRU::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() != 2 || top_blobs.size() != 2)
    {
        return forward(bottom_blobs[0], top_blobs[0], opt);
    }
    const Mat& bottom_blob = bottom_blobs[0];
    int T = bottom_blob.h;
    Mat& top_blob = top_blobs[0];
    Mat& hidden_state = top_blobs[1];

    //Copy previous states
    hidden_state = bottom_blobs[1].clone(opt.blob_allocator);

    top_blob.create(num_output, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = gru(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden_state, opt);
        if (ret != 0)
            return ret;
    }

    return 0;
}

} // namespace ncnn
