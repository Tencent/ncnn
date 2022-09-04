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

#include "lstm2.h"

#include <math.h>

namespace ncnn {

LSTM2::LSTM2()
{
    one_blob_only = false;
    support_inplace = false;
}

int LSTM2::load_param(const ParamDict& pd)
{
    hidden_size = pd.get(0, 0); // hidden
    weight_data_size = pd.get(1, 0);
    direction = pd.get(2, 0);
    proj_size = pd.get(3, 0);

    if (proj_size == 0 || direction > 0)
      // bi-LSTM with proj_size > 0 is not implemented
      return -100;


    return 0;
}

int LSTM2::load_model(const ModelBin& mb)
{
    int input_size = weight_data_size / hidden_size / 4;

    // raw weight data
    weight_xc_data = mb.load(input_size, hidden_size * 4, 0);
    if (weight_xc_data.empty())
        return -100;

    bias_c_data = mb.load(hidden_size, 4, 0);
    if (bias_c_data.empty())
        return -100;

    weight_hc_data = mb.load(proj_size, hidden_size * 4, 0);
    if (weight_hc_data.empty())
        return -100;

    weight_hr_data = mb.load(hidden_size, proj_size, 0);
    if (weight_hr_data.empty())
        return -100;

    return 0;
}

static int lstm(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    int input_size = bottom_blob.w;
    int T = bottom_blob.h;

    int hidden_size = weight_hr.w;
    int proj_size = weight_hr.h;

    // 4 x hidden
    Mat gates(4, hidden_size, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    Mat tmp_top(hidden_size, 4u, opt.workspace_allocator);
    if (tmp_top.empty())
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

        int ti = t;

        const float* x = bottom_blob.row(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < hidden_size; q++)
        {
            const float* bias_c_I = bias_c.row(0);
            const float* bias_c_F = bias_c.row(1);
            const float* bias_c_O = bias_c.row(2);
            const float* bias_c_G = bias_c.row(3);

            float* gates_data = gates.row(q);

            // gate I F O G
            const float* weight_xc_I = weight_xc.row(hidden_size * 0 + q);
            const float* weight_xc_F = weight_xc.row(hidden_size * 1 + q);
            const float* weight_xc_O = weight_xc.row(hidden_size * 2 + q);
            const float* weight_xc_G = weight_xc.row(hidden_size * 3 + q);

            const float* weight_hc_I = weight_hc.row(hidden_size * 0 + q);
            const float* weight_hc_F = weight_hc.row(hidden_size * 1 + q);
            const float* weight_hc_O = weight_hc.row(hidden_size * 2 + q);
            const float* weight_hc_G = weight_hc.row(hidden_size * 3 + q);

            float I = bias_c_I[q];
            float F = bias_c_F[q];
            float O = bias_c_O[q];
            float G = bias_c_G[q];

            for (int i = 0; i < input_size; i++)
            {
                float xi = x[i];

                I += weight_xc_I[i] * xi;
                F += weight_xc_F[i] * xi;
                O += weight_xc_O[i] * xi;
                G += weight_xc_G[i] * xi;
            }

            for (int i = 0; i < proj_size; i++)
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
        // float* output_data = top_blob.row(ti);
        float* tmp_output_data = tmp_top;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < hidden_size; q++)
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
            tmp_output_data[q] = H;
            // hidden_state[q] = H;
            // output_data[q] = H;
        }

        float* output_data = top_blob.row(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < proj_size; q++)
        {
          const float* hr = weight_hr.row(q);
          float s = 0;
          for (int i = 0; i < hidden_size; i++)
          {
            s += tmp_output_data[i] * hr[i];
          }
          output_data[q] = s;
          hidden_state[q] = s;
        }
    }

    return 0;
}

int LSTM2::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int T = bottom_blob.h;

    Mat hidden;
    Mat cell;
    Allocator* hidden_cell_allocator = top_blobs.size() == 3 ? opt.blob_allocator : opt.workspace_allocator;
    if (bottom_blobs.size() == 3)
    {
        hidden = bottom_blobs[1].clone(hidden_cell_allocator);
        cell = bottom_blobs[2].clone(hidden_cell_allocator);
    }
    else
    {
        hidden.create(proj_size, 4u, hidden_cell_allocator);
        if (hidden.empty())
            return -100;
        hidden.fill(0.f);

        cell.create(hidden_size, 4u, hidden_cell_allocator);
        if (cell.empty())
            return -100;
        cell.fill(0.f);
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(proj_size , T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    int ret = lstm(bottom_blob, top_blob, weight_xc_data, bias_c_data, weight_hc_data, weight_hr_data, hidden, cell, opt);
    if (ret != 0)
        return ret;

    if (top_blobs.size() == 3)
    {
        top_blobs[1] = hidden;
        top_blobs[2] = cell;
    }

    return 0;
}

} // namespace ncnn
