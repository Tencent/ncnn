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

#include "lstm2_x86.h"

#include "x86_activation.h"
#include "x86_usability.h"

#include <math.h>
#include "layer_type.h"

namespace ncnn {

LSTM2_x86::LSTM2_x86()
{
    one_blob_only = false;
    support_inplace = false;
}

int LSTM2_x86::create_pipeline(const Option& opt)
{
    (void)(opt);

    return 0;
}
#ifdef __AVX__
static int lstm(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    int input_size = bottom_blob.w;
    int T = bottom_blob.h;

    int hidden_size = weight_hr.w;
    int proj_size = weight_hr.h;

    // 4 x hidden
    Mat gates(hidden_size, 4, 4u, opt.workspace_allocator);
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

        int nn_hidden_size = hidden_size >> 1;
        int remain_hidden_size_start = nn_hidden_size << 1;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = qq * 2;

            const float* x = bottom_blob.row(ti);  // input_size
            const float* hidden_ptr_r = hidden_state;  // proj_size
            const float* bias_c_I = bias_c.row(0);  // hidden_size
            const float* bias_c_F = bias_c.row(1);  // hidden_size
            const float* bias_c_O = bias_c.row(2);  // hidden_size
            const float* bias_c_G = bias_c.row(3);  // hidden_size

            float* gates_data_I = gates.row(0); // hidden_size
            float* gates_data_F = gates.row(1); // hidden_size
            float* gates_data_O = gates.row(2); // hidden_size
            float* gates_data_G = gates.row(3); // hidden_size
            // gate I F O G
            const float* weight_xc_I_0 = weight_xc.row(hidden_size * 0 + q);  // input_size
            const float* weight_xc_F_0 = weight_xc.row(hidden_size * 1 + q);  // input_size
            const float* weight_xc_O_0 = weight_xc.row(hidden_size * 2 + q);  // input_size
            const float* weight_xc_G_0 = weight_xc.row(hidden_size * 3 + q);  // input_size
            const float* weight_xc_I_1 = weight_xc.row(hidden_size * 0 + (q + 1));  // input_size
            const float* weight_xc_F_1 = weight_xc.row(hidden_size * 1 + (q + 1));  // input_size
            const float* weight_xc_O_1 = weight_xc.row(hidden_size * 2 + (q + 1));  // input_size
            const float* weight_xc_G_1 = weight_xc.row(hidden_size * 3 + (q + 1));  // input_size

            const float* weight_hc_I_0 = weight_hc.row(hidden_size * 0 + q);  // proj_size
            const float* weight_hc_F_0 = weight_hc.row(hidden_size * 1 + q);  // proj_size
            const float* weight_hc_O_0 = weight_hc.row(hidden_size * 2 + q);  // proj_size
            const float* weight_hc_G_0 = weight_hc.row(hidden_size * 3 + q);  // proj_size
            const float* weight_hc_I_1 = weight_hc.row(hidden_size * 0 + (q + 1));  // proj_size
            const float* weight_hc_F_1 = weight_hc.row(hidden_size * 1 + (q + 1));  // proj_size
            const float* weight_hc_O_1 = weight_hc.row(hidden_size * 2 + (q + 1));  // proj_size
            const float* weight_hc_G_1 = weight_hc.row(hidden_size * 3 + (q + 1));  // proj_size

            // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.htm
            // float I = bias_c_I[q];  // hidden_size
            // float F = bias_c_F[q];  // hidden_size
            // float O = bias_c_O[q];  // hidden_size
            // float G = bias_c_G[q];  // hidden_size
            __m256 _sumI_0 = _mm256_setzero_ps();
            __m256 _sumF_0 = _mm256_setzero_ps();
            __m256 _sumO_0 = _mm256_setzero_ps();
            __m256 _sumG_0 = _mm256_setzero_ps();
            __m256 _sumI_1 = _mm256_setzero_ps();
            __m256 _sumF_1 = _mm256_setzero_ps();
            __m256 _sumO_1 = _mm256_setzero_ps();
            __m256 _sumG_1 = _mm256_setzero_ps();
            int nn_num_size = input_size >> 3;
            int remain_size = input_size & 7;
            for (; nn_num_size > 0; nn_num_size--)
            {
                __m256 xi = _mm256_loadu_ps(x);
                _sumI_0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_xc_I_0), xi, _sumI_0);
                _sumF_0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_xc_F_0), xi, _sumF_0);
                _sumO_0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_xc_O_0), xi, _sumO_0);
                _sumG_0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_xc_G_0), xi, _sumG_0);
                _sumI_1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_xc_I_1), xi, _sumI_1);
                _sumF_1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_xc_F_1), xi, _sumF_1);
                _sumO_1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_xc_O_1), xi, _sumO_1);
                _sumG_1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_xc_G_1), xi, _sumG_1);
                x += 8;
                weight_xc_I_0 += 8;
                weight_xc_F_0 += 8;
                weight_xc_O_0 += 8;
                weight_xc_G_0 += 8;
                weight_xc_I_1 += 8;
                weight_xc_F_1 += 8;
                weight_xc_O_1 += 8;
                weight_xc_G_1 += 8;
            }
            int nn_proj_size = proj_size >> 3;
            int remain_proj_size = proj_size & 7;
            for (; nn_proj_size > 0; nn_proj_size--)
            {
                __m256 h_cont = _mm256_loadu_ps(hidden_ptr_r);

                _sumI_0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_hc_I_0), h_cont, _sumI_0);
                _sumF_0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_hc_F_0), h_cont, _sumF_0);
                _sumO_0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_hc_O_0), h_cont, _sumO_0);
                _sumG_0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_hc_G_0), h_cont, _sumG_0);
                _sumI_1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_hc_I_1), h_cont, _sumI_1);
                _sumF_1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_hc_F_1), h_cont, _sumF_1);
                _sumO_1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_hc_O_1), h_cont, _sumO_1);
                _sumG_1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_hc_G_1), h_cont, _sumG_1);
                hidden_ptr_r += 8;
                weight_hc_I_0 += 8;
                weight_hc_F_0 += 8;
                weight_hc_O_0 += 8;
                weight_hc_G_0 += 8;
                weight_hc_I_1 += 8;
                weight_hc_F_1 += 8;
                weight_hc_O_1 += 8;
                weight_hc_G_1 += 8;
            }
            float sums[8];
            _mm256_storeu_ps(sums, HorizontalSums(_sumI_0, _sumF_0, _sumO_0, _sumG_0, _sumI_1, _sumF_1, _sumO_1, _sumG_1));
            sums[0] += bias_c_I[q];
            sums[1] += bias_c_F[q];
            sums[2] += bias_c_O[q];
            sums[3] += bias_c_G[q];
            sums[4] += bias_c_I[q + 1];
            sums[5] += bias_c_F[q + 1];
            sums[6] += bias_c_O[q + 1];
            sums[7] += bias_c_G[q + 1];

            for (; remain_size > 0; remain_size--)
            {
                float xi = *x;
                sums[0] += *weight_xc_I_0 * xi;
                sums[1] += *weight_xc_F_0 * xi;
                sums[2] += *weight_xc_O_0 * xi;
                sums[3] += *weight_xc_G_0 * xi;
                sums[4] += *weight_xc_I_1 * xi;
                sums[5] += *weight_xc_F_1 * xi;
                sums[6] += *weight_xc_O_1 * xi;
                sums[7] += *weight_xc_G_1 * xi;
                x++;
                weight_xc_I_0++;
                weight_xc_F_0++;
                weight_xc_O_0++;
                weight_xc_G_0++;
                weight_xc_I_1++;
                weight_xc_F_1++;
                weight_xc_O_1++;
                weight_xc_G_1++;
            }

            for (; remain_proj_size > 0; remain_proj_size--)
            {
                float h_cont = *hidden_ptr_r;
                sums[0] += *weight_hc_I_0 * h_cont;
                sums[1] += *weight_hc_F_0 * h_cont;
                sums[2] += *weight_hc_O_0 * h_cont;
                sums[3] += *weight_hc_G_0 * h_cont;
                sums[4] += *weight_hc_I_1 * h_cont;
                sums[5] += *weight_hc_F_1 * h_cont;
                sums[6] += *weight_hc_O_1 * h_cont;
                sums[7] += *weight_hc_G_1 * h_cont;
                hidden_ptr_r++;
                weight_hc_I_0++;
                weight_hc_F_0++;
                weight_hc_O_0++;
                weight_hc_G_0++;
                weight_hc_I_1++;
                weight_hc_F_1++;
                weight_hc_O_1++;
                weight_hc_G_1++;
            }
            gates_data_I[q] = sums[0];
            gates_data_F[q] = sums[1];
            gates_data_O[q] = sums[2];
            gates_data_G[q] = sums[3];
            gates_data_I[q + 1] = sums[4];
            gates_data_F[q + 1] = sums[5];
            gates_data_O[q + 1] = sums[6];
            gates_data_G[q + 1] = sums[7];
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_hidden_size_start; q < hidden_size; q++)
        {
            const float* x = bottom_blob.row(ti);  // input_size
            const float* hidden_ptr_r = hidden_state;  // proj_size
            const float* bias_c_I = bias_c.row(0);  // hidden_size
            const float* bias_c_F = bias_c.row(1);  // hidden_size
            const float* bias_c_O = bias_c.row(2);  // hidden_size
            const float* bias_c_G = bias_c.row(3);  // hidden_size

            float* gates_data_I = gates.row(0);  // hidden_size
            float* gates_data_F = gates.row(1);  // hidden_size
            float* gates_data_O = gates.row(2);  // hidden_size
            float* gates_data_G = gates.row(3);  // hidden_size
            // gate I F O G
            const float* weight_xc_I = weight_xc.row(hidden_size * 0 + q);  // input_size
            const float* weight_xc_F = weight_xc.row(hidden_size * 1 + q);  // input_size
            const float* weight_xc_O = weight_xc.row(hidden_size * 2 + q);  // input_size
            const float* weight_xc_G = weight_xc.row(hidden_size * 3 + q);  // input_size

            const float* weight_hc_I = weight_hc.row(hidden_size * 0 + q);  // proj_size
            const float* weight_hc_F = weight_hc.row(hidden_size * 1 + q);  // proj_size
            const float* weight_hc_O = weight_hc.row(hidden_size * 2 + q);  // proj_size
            const float* weight_hc_G = weight_hc.row(hidden_size * 3 + q);  // proj_size

            // float I = bias_c_I[q];
            // float F = bias_c_F[q];
            // float O = bias_c_O[q];
            // float G = bias_c_G[q];
            __m256 _sumI = _mm256_setzero_ps();
            __m256 _sumF = _mm256_setzero_ps();
            __m256 _sumO = _mm256_setzero_ps();
            __m256 _sumG = _mm256_setzero_ps();
            int nn_num_size = input_size >> 3;
            int remain_size = input_size & 7;
            for (; nn_num_size > 0; nn_num_size--)
            {
                __m256 xi = _mm256_loadu_ps(x);
                _sumI = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_xc_I), xi, _sumI);
                _sumF = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_xc_F), xi, _sumF);
                _sumO = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_xc_O), xi, _sumO);
                _sumG = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_xc_G), xi, _sumG);
                x += 8;
                weight_xc_I += 8;
                weight_xc_F += 8;
                weight_xc_O += 8;
                weight_xc_G += 8;
            }
            int nn_proj_size = proj_size >> 3;
            int remain_proj_size = proj_size & 7;
            for (; nn_proj_size > 0; nn_proj_size--)
            {
                __m256 h_cont = _mm256_loadu_ps(hidden_ptr_r);

                _sumI = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_hc_I), h_cont, _sumI);
                _sumF = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_hc_F), h_cont, _sumF);
                _sumO = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_hc_O), h_cont, _sumO);
                _sumG = _mm256_comp_fmadd_ps(_mm256_loadu_ps(weight_hc_G), h_cont, _sumG);
                hidden_ptr_r += 8;
                weight_hc_I += 8;
                weight_hc_F += 8;
                weight_hc_O += 8;
                weight_hc_G += 8;
            }
            float sums[4];
            _mm_storeu_ps(sums, HorizontalSums(_sumI, _sumF, _sumO, _sumG));
            sums[0] += bias_c_I[q];
            sums[1] += bias_c_F[q];
            sums[2] += bias_c_O[q];
            sums[3] += bias_c_G[q];

            for (; remain_size > 0; remain_size--)
            {
                float xi = *x;
                sums[0] += *weight_xc_I * xi;
                sums[1] += *weight_xc_F * xi;
                sums[2] += *weight_xc_O * xi;
                sums[3] += *weight_xc_G * xi;
                x++;
                weight_xc_I++;
                weight_xc_F++;
                weight_xc_O++;
                weight_xc_G++;
            }

            for (; remain_proj_size > 0; remain_proj_size--)
            {
                float h_cont = *hidden_ptr_r;
                sums[0] += *weight_hc_I * h_cont;
                sums[1] += *weight_hc_F * h_cont;
                sums[2] += *weight_hc_O * h_cont;
                sums[3] += *weight_hc_G * h_cont;
                hidden_ptr_r++;
                weight_hc_I++;
                weight_hc_F++;
                weight_hc_O++;
                weight_hc_G++;
            }
            gates_data_I[q] = sums[0];
            gates_data_F[q] = sums[1];
            gates_data_O[q] = sums[2];
            gates_data_G[q] = sums[3];
        }

        // lstm unit
        // sigmoid(I)
        // sigmoid(F)
        // sigmoid(O)
        // tanh(G)
        // c_t := f_t .* c_{t-1} + i_t .* g_t
        // h_t := o_t .* tanh[c_t]
        // float* output_data = top_blob.row(ti);
        float* tmp_output_data = tmp_top;  // hidden_size
        float* cell_ptr = cell_state;  // hidden_size
        const float* gates_data_I = gates.row(0);  // hidden_size
        const float* gates_data_F = gates.row(1);  // hidden_size
        const float* gates_data_O = gates.row(2);  // hidden_size
        const float* gates_data_G = gates.row(3);  // hidden_size
        int nn_activation = hidden_size >> 3;
        int remain_activations = hidden_size & 7;
        for (; nn_activation > 0; nn_activation--)
        {
            __m256 I = sigmoid_avx(_mm256_loadu_ps(gates_data_I));
            __m256 F = sigmoid_avx(_mm256_loadu_ps(gates_data_F));
            __m256 O = sigmoid_avx(_mm256_loadu_ps(gates_data_O));
            __m256 G = tanh_avx(_mm256_loadu_ps(gates_data_G));
            __m256 cell2 = _mm256_add_ps(_mm256_mul_ps(F, _mm256_loadu_ps(cell_ptr)), _mm256_mul_ps(I, G));
            __m256 H = _mm256_mul_ps(O, tanh_avx(cell2));
            _mm256_storeu_ps(cell_ptr, cell2);
            _mm256_storeu_ps(tmp_output_data, H);
            cell_ptr += 8;
            tmp_output_data += 8;
            gates_data_I += 8;
            gates_data_F += 8;
            gates_data_O += 8;
            gates_data_G += 8;
        }
        for (; remain_activations > 0; remain_activations--)
        {
            float I = *gates_data_I;
            float F = *gates_data_F;
            float O = *gates_data_O;
            float G = *gates_data_G;

            I = 1.f / (1.f + exp(-I));
            F = 1.f / (1.f + exp(-F));
            O = 1.f / (1.f + exp(-O));
            G = tanh(G);
            float cell2 = F * *cell_ptr + I * G;
            float H = O * tanh(cell2);
            *cell_ptr = cell2;
            *tmp_output_data = H;
            cell_ptr++;
            tmp_output_data++;
            gates_data_I++;
            gates_data_F++;
            gates_data_O++;
            gates_data_G++;
        }

        float* output_data = top_blob.row(ti);  // proj_size
        float* hidden_ptr = hidden_state;  // proj_size
        tmp_output_data = tmp_top;  // hidden_size
#if 1
        for (int q = 0; q < proj_size; q++)
        {
          const float* hr = weight_hr.row(q);
          float s = 0;
          for (int i = 0; i < hidden_size; i++)
          {
            s += tmp_output_data[i] * hr[i];
          }
          output_data[q] = s;
          hidden_ptr[q] = s;
        }
#else
        int nn_proj_size = proj_size >> 3;
        int remain_proj_size = proj_size & 7;

        #pragma omp parallel for num_threads(opt.num_threads)
        for(int i = 0; i < nn_proj_size; i++) {
            tmp_output_data = tmp_top;  // hidden_size

            __m256 _sum_0 = _mm256_setzero_ps();
            __m256 _sum_1 = _mm256_setzero_ps();
            __m256 _sum_2 = _mm256_setzero_ps();
            __m256 _sum_3 = _mm256_setzero_ps();
            __m256 _sum_4 = _mm256_setzero_ps();
            __m256 _sum_5 = _mm256_setzero_ps();
            __m256 _sum_6 = _mm256_setzero_ps();
            __m256 _sum_7 = _mm256_setzero_ps();

            const float* hr0 = weight_hr.row(i*8 + 0);
            const float* hr1 = weight_hr.row(i*8 + 1);
            const float* hr2 = weight_hr.row(i*8 + 2);
            const float* hr3 = weight_hr.row(i*8 + 3);
            const float* hr4 = weight_hr.row(i*8 + 4);
            const float* hr5 = weight_hr.row(i*8 + 5);
            const float* hr6 = weight_hr.row(i*8 + 6);
            const float* hr7 = weight_hr.row(i*8 + 7);

            int nn_hidden_size = hidden_size >> 3;
            int remain_size = hidden_size & 7;
            for (; nn_hidden_size > 0; nn_hidden_size--) {
                __m256 xi = _mm256_loadu_ps(tmp_output_data);
                _sum_0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(hr0), xi, _sum_0);
                _sum_1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(hr1), xi, _sum_1);
                _sum_2 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(hr2), xi, _sum_2);
                _sum_3 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(hr3), xi, _sum_3);
                _sum_4 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(hr4), xi, _sum_4);
                _sum_5 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(hr5), xi, _sum_5);
                _sum_6 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(hr6), xi, _sum_6);
                _sum_7 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(hr7), xi, _sum_7);

                tmp_output_data += 8;
                hr0 += 8;
                hr1 += 8;
                hr2 += 8;
                hr3 += 8;
                hr4 += 8;
                hr5 += 8;
                hr6 += 8;
                hr7 += 8;
            }

            float sums[8];
            _mm256_storeu_ps(sums, HorizontalSums(_sum_0, _sum_1, _sum_2, _sum_3, _sum_4, _sum_5, _sum_6, _sum_7));

            for (; remain_size > 0; remain_size--) {
                float xi = *tmp_output_data;
                sums[0] += *hr0 * xi;
                sums[1] += *hr1 * xi;
                sums[2] += *hr2 * xi;
                sums[3] += *hr3 * xi;
                sums[4] += *hr4 * xi;
                sums[5] += *hr5 * xi;
                sums[6] += *hr6 * xi;
                sums[7] += *hr7 * xi;
                tmp_output_data++;
                hr0++;
                hr1++;
                hr2++;
                hr3++;
                hr4++;
                hr5++;
                hr6++;
                hr7++;
            }

          output_data[i*8+0] = sums[0];
          output_data[i*8+1] = sums[1];
          output_data[i*8+2] = sums[2];
          output_data[i*8+3] = sums[3];
          output_data[i*8+4] = sums[4];
          output_data[i*8+5] = sums[5];
          output_data[i*8+6] = sums[6];
          output_data[i*8+7] = sums[7];

          hidden_ptr[i*8+0] = sums[0];
          hidden_ptr[i*8+1] = sums[1];
          hidden_ptr[i*8+2] = sums[2];
          hidden_ptr[i*8+3] = sums[3];
          hidden_ptr[i*8+4] = sums[4];
          hidden_ptr[i*8+5] = sums[5];
          hidden_ptr[i*8+6] = sums[6];
          hidden_ptr[i*8+7] = sums[7];
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for(int i = 0; i < remain_proj_size; ++i) {
            tmp_output_data = tmp_top;  // hidden_size
            const float* hr = weight_hr.row(nn_proj_size*8 + i);
            __m256 _sum = _mm256_setzero_ps();
            int nn_num_size = input_size >> 3;
            int remain_size = input_size & 7;
            for (; nn_num_size > 0; nn_num_size--) {
                __m256 xi = _mm256_loadu_ps(tmp_output_data);
                _sum = _mm256_comp_fmadd_ps(_mm256_loadu_ps(hr), xi, _sum);
                tmp_output_data += 8;
                hr += 8;
            }
            float s = _mm256_reduce_add_ps(_sum);
            for(; remain_size > 0; remain_size--) {
              float xi = *tmp_output_data;
              s += *hr * xi;
              tmp_output_data++;
              hr++;
            }

            output_data[nn_proj_size*8 + i] = s;
            hidden_ptr[nn_proj_size*8 + i] = s;
        }
#endif
    }

    return 0;
}
#endif

int LSTM2_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if __AVX__
    // return LSTM2::forward(bottom_blobs, top_blobs, opt);

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
#else
    return LSTM2::forward(bottom_blobs, top_blobs, opt);
#endif
}

} // namespace ncnn
