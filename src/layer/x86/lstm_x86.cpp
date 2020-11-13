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
#ifdef __AVX__
#include "avx_activation.h"
#include "avx_usability.h"
#endif // NCNN_AVX2

#include "lstm_x86.h"

#include <math.h>
#include "layer_type.h"

namespace ncnn {

LSTM_x86::LSTM_x86()
{
#ifdef __AVX__
    support_weight_fp16_storage = true;
#endif
    one_blob_only = false;
    support_inplace = false;
}

int LSTM_x86::create_pipeline(const Option& opt)
{
#if __AVX__
    if (opt.use_weight_fp16_storage)
    {
        ncnn::cast_float32_to_float16(weight_xc_data, weight_xc_data_fp16, opt);
        ncnn::cast_float32_to_float16(weight_hc_data, weight_hc_data_fp16, opt);
    }
#else
    (void)(opt);
#endif // __AVX__

    return 0;
}
#ifdef __AVX__

static int lstm_fp16(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;
    // fprintf(stderr, "bottom_blob = %d x %d x %d num_output = %d \n", bottom_blob.w,bottom_blob.h,bottom_blob.c,num_output);
    // 4 x num_output
    Mat gates(num_output, 4, 4u, opt.workspace_allocator);
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
        int remain_output = (num_output >> 1) << 1;
        for (int q = 0; q + 1 < num_output; q += 2)
        {
            const float* x = bottom_blob.row(ti);
            const float* hidden_ptr_r = hidden_state;
            const float* bias_c_I = bias_c.row(0);
            const float* bias_c_F = bias_c.row(1);
            const float* bias_c_O = bias_c.row(2);
            const float* bias_c_G = bias_c.row(3);

            float* gates_data_I = gates.row(0);
            float* gates_data_F = gates.row(1);
            float* gates_data_O = gates.row(2);
            float* gates_data_G = gates.row(3);
            // gate I F O G
            const unsigned short* weight_xc_I_0 = (const unsigned short*)weight_xc.row(num_output * 0 + q);
            const unsigned short* weight_xc_F_0 = (const unsigned short*)weight_xc.row(num_output * 1 + q);
            const unsigned short* weight_xc_O_0 = (const unsigned short*)weight_xc.row(num_output * 2 + q);
            const unsigned short* weight_xc_G_0 = (const unsigned short*)weight_xc.row(num_output * 3 + q);
            const unsigned short* weight_xc_I_1 = (const unsigned short*)weight_xc.row(num_output * 0 + (q + 1));
            const unsigned short* weight_xc_F_1 = (const unsigned short*)weight_xc.row(num_output * 1 + (q + 1));
            const unsigned short* weight_xc_O_1 = (const unsigned short*)weight_xc.row(num_output * 2 + (q + 1));
            const unsigned short* weight_xc_G_1 = (const unsigned short*)weight_xc.row(num_output * 3 + (q + 1));

            const unsigned short* weight_hc_I_0 = (const unsigned short*)weight_hc.row(num_output * 0 + q);
            const unsigned short* weight_hc_F_0 = (const unsigned short*)weight_hc.row(num_output * 1 + q);
            const unsigned short* weight_hc_O_0 = (const unsigned short*)weight_hc.row(num_output * 2 + q);
            const unsigned short* weight_hc_G_0 = (const unsigned short*)weight_hc.row(num_output * 3 + q);
            const unsigned short* weight_hc_I_1 = (const unsigned short*)weight_hc.row(num_output * 0 + (q + 1));
            const unsigned short* weight_hc_F_1 = (const unsigned short*)weight_hc.row(num_output * 1 + (q + 1));
            const unsigned short* weight_hc_O_1 = (const unsigned short*)weight_hc.row(num_output * 2 + (q + 1));
            const unsigned short* weight_hc_G_1 = (const unsigned short*)weight_hc.row(num_output * 3 + (q + 1));

            // float I = bias_c_I[q];
            // float F = bias_c_F[q];
            // float O = bias_c_O[q];
            // float G = bias_c_G[q];
            __m256 _sumI_0 = _mm256_setzero_ps();
            __m256 _sumF_0 = _mm256_setzero_ps();
            __m256 _sumO_0 = _mm256_setzero_ps();
            __m256 _sumG_0 = _mm256_setzero_ps();
            __m256 _sumI_1 = _mm256_setzero_ps();
            __m256 _sumF_1 = _mm256_setzero_ps();
            __m256 _sumO_1 = _mm256_setzero_ps();
            __m256 _sumG_1 = _mm256_setzero_ps();
            int nn_num_size = size >> 3;
            int remain_size = size & 7;
            for (; nn_num_size > 0; nn_num_size--)
            {
                __m256 xi = _mm256_loadu_ps(x);
                _sumI_0 = _mm256_fmadd_ps(loadfp16(weight_xc_I_0), xi, _sumI_0);
                _sumF_0 = _mm256_fmadd_ps(loadfp16(weight_xc_F_0), xi, _sumF_0);
                _sumO_0 = _mm256_fmadd_ps(loadfp16(weight_xc_O_0), xi, _sumO_0);
                _sumG_0 = _mm256_fmadd_ps(loadfp16(weight_xc_G_0), xi, _sumG_0);
                _sumI_1 = _mm256_fmadd_ps(loadfp16(weight_xc_I_1), xi, _sumI_1);
                _sumF_1 = _mm256_fmadd_ps(loadfp16(weight_xc_F_1), xi, _sumF_1);
                _sumO_1 = _mm256_fmadd_ps(loadfp16(weight_xc_O_1), xi, _sumO_1);
                _sumG_1 = _mm256_fmadd_ps(loadfp16(weight_xc_G_1), xi, _sumG_1);
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
            int nn_num_output = num_output >> 3;
            int remain_num_output = num_output & 7;
            for (; nn_num_output > 0; nn_num_output--)
            {
                __m256 h_cont = _mm256_loadu_ps(hidden_ptr_r);

                _sumI_0 = _mm256_fmadd_ps(loadfp16(weight_hc_I_0), h_cont, _sumI_0);
                _sumF_0 = _mm256_fmadd_ps(loadfp16(weight_hc_F_0), h_cont, _sumF_0);
                _sumO_0 = _mm256_fmadd_ps(loadfp16(weight_hc_O_0), h_cont, _sumO_0);
                _sumG_0 = _mm256_fmadd_ps(loadfp16(weight_hc_G_0), h_cont, _sumG_0);
                _sumI_1 = _mm256_fmadd_ps(loadfp16(weight_hc_I_1), h_cont, _sumI_1);
                _sumF_1 = _mm256_fmadd_ps(loadfp16(weight_hc_F_1), h_cont, _sumF_1);
                _sumO_1 = _mm256_fmadd_ps(loadfp16(weight_hc_O_1), h_cont, _sumO_1);
                _sumG_1 = _mm256_fmadd_ps(loadfp16(weight_hc_G_1), h_cont, _sumG_1);
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
            if (remain_size != 0)
            {
                unsigned short fp16_weights[8][8] = {{0}};
                float _xi_f[8] = {0};
                // No fast way to convert to fp32 one element at the time
                // so batch an 8 lane vector.
                for (int i = 0; i < remain_size; i++)
                {
                    _xi_f[i] = *x;
                    fp16_weights[0][i] = *weight_xc_I_0;
                    fp16_weights[1][i] = *weight_xc_F_0;
                    fp16_weights[2][i] = *weight_xc_O_0;
                    fp16_weights[3][i] = *weight_xc_G_0;
                    fp16_weights[4][i] = *weight_xc_I_1;
                    fp16_weights[5][i] = *weight_xc_F_1;
                    fp16_weights[6][i] = *weight_xc_O_1;
                    fp16_weights[7][i] = *weight_xc_G_1;
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
                __m256 xi = _mm256_loadu_ps(_xi_f);
                _sumI_0 = _mm256_fmadd_ps(loadfp16(fp16_weights[0]), xi, _sumI_0);
                _sumF_0 = _mm256_fmadd_ps(loadfp16(fp16_weights[1]), xi, _sumF_0);
                _sumO_0 = _mm256_fmadd_ps(loadfp16(fp16_weights[2]), xi, _sumO_0);
                _sumG_0 = _mm256_fmadd_ps(loadfp16(fp16_weights[3]), xi, _sumG_0);
                _sumI_1 = _mm256_fmadd_ps(loadfp16(fp16_weights[4]), xi, _sumI_1);
                _sumF_1 = _mm256_fmadd_ps(loadfp16(fp16_weights[5]), xi, _sumF_1);
                _sumO_1 = _mm256_fmadd_ps(loadfp16(fp16_weights[6]), xi, _sumO_1);
                _sumG_1 = _mm256_fmadd_ps(loadfp16(fp16_weights[7]), xi, _sumG_1);
            }
            if (remain_num_output != 0)
            {
                unsigned short fp16_weights[8][8] = {{0}};
                float _hcont_f[8] = {0};
                // No fast way to convert to fp32 one element at the time
                // so batch an 8 lane vector.
                for (int i = 0; i < remain_num_output; i++)
                {
                    _hcont_f[i] = *hidden_ptr_r;
                    fp16_weights[0][i] = *weight_hc_I_0;
                    fp16_weights[1][i] = *weight_hc_F_0;
                    fp16_weights[2][i] = *weight_hc_O_0;
                    fp16_weights[3][i] = *weight_hc_G_0;
                    fp16_weights[4][i] = *weight_hc_I_1;
                    fp16_weights[5][i] = *weight_hc_F_1;
                    fp16_weights[6][i] = *weight_hc_O_1;
                    fp16_weights[7][i] = *weight_hc_G_1;
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
                __m256 h_cont = _mm256_loadu_ps(_hcont_f);
                _sumI_0 = _mm256_fmadd_ps(loadfp16(fp16_weights[0]), h_cont, _sumI_0);
                _sumF_0 = _mm256_fmadd_ps(loadfp16(fp16_weights[1]), h_cont, _sumF_0);
                _sumO_0 = _mm256_fmadd_ps(loadfp16(fp16_weights[2]), h_cont, _sumO_0);
                _sumG_0 = _mm256_fmadd_ps(loadfp16(fp16_weights[3]), h_cont, _sumG_0);
                _sumI_1 = _mm256_fmadd_ps(loadfp16(fp16_weights[4]), h_cont, _sumI_1);
                _sumF_1 = _mm256_fmadd_ps(loadfp16(fp16_weights[5]), h_cont, _sumF_1);
                _sumO_1 = _mm256_fmadd_ps(loadfp16(fp16_weights[6]), h_cont, _sumO_1);
                _sumG_1 = _mm256_fmadd_ps(loadfp16(fp16_weights[7]), h_cont, _sumG_1);
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
            gates_data_I[q] = sums[0];
            gates_data_F[q] = sums[1];
            gates_data_O[q] = sums[2];
            gates_data_G[q] = sums[3];
            gates_data_I[q + 1] = sums[4];
            gates_data_F[q + 1] = sums[5];
            gates_data_O[q + 1] = sums[6];
            gates_data_G[q + 1] = sums[7];
        }

        for (int q = remain_output; q < num_output; q++)
        {
            const float* x = bottom_blob.row(ti);
            const float* hidden_ptr_r = hidden_state;
            const float* bias_c_I = bias_c.row(0);
            const float* bias_c_F = bias_c.row(1);
            const float* bias_c_O = bias_c.row(2);
            const float* bias_c_G = bias_c.row(3);

            float* gates_data_I = gates.row(0);
            float* gates_data_F = gates.row(1);
            float* gates_data_O = gates.row(2);
            float* gates_data_G = gates.row(3);
            // gate I F O G
            const unsigned short* weight_xc_I = (const unsigned short*)weight_xc.row(num_output * 0 + q);
            const unsigned short* weight_xc_F = (const unsigned short*)weight_xc.row(num_output * 1 + q);
            const unsigned short* weight_xc_O = (const unsigned short*)weight_xc.row(num_output * 2 + q);
            const unsigned short* weight_xc_G = (const unsigned short*)weight_xc.row(num_output * 3 + q);

            const unsigned short* weight_hc_I = (const unsigned short*)weight_hc.row(num_output * 0 + q);
            const unsigned short* weight_hc_F = (const unsigned short*)weight_hc.row(num_output * 1 + q);
            const unsigned short* weight_hc_O = (const unsigned short*)weight_hc.row(num_output * 2 + q);
            const unsigned short* weight_hc_G = (const unsigned short*)weight_hc.row(num_output * 3 + q);

            // float I = bias_c_I[q];
            // float F = bias_c_F[q];
            // float O = bias_c_O[q];
            // float G = bias_c_G[q];
            __m256 _sumI = _mm256_setzero_ps();
            __m256 _sumF = _mm256_setzero_ps();
            __m256 _sumO = _mm256_setzero_ps();
            __m256 _sumG = _mm256_setzero_ps();
            int nn_num_size = size >> 3;
            int remain_size = size & 7;
            for (; nn_num_size > 0; nn_num_size--)
            {
                __m256 xi = _mm256_loadu_ps(x);
                _sumI = _mm256_fmadd_ps(loadfp16(weight_xc_I), xi, _sumI);
                _sumF = _mm256_fmadd_ps(loadfp16(weight_xc_F), xi, _sumF);
                _sumO = _mm256_fmadd_ps(loadfp16(weight_xc_O), xi, _sumO);
                _sumG = _mm256_fmadd_ps(loadfp16(weight_xc_G), xi, _sumG);
                x += 8;
                weight_xc_I += 8;
                weight_xc_F += 8;
                weight_xc_O += 8;
                weight_xc_G += 8;
            }
            int nn_num_output = num_output >> 3;
            int remain_num_output = num_output & 7;
            for (; nn_num_output > 0; nn_num_output--)
            {
                __m256 h_cont = _mm256_loadu_ps(hidden_ptr_r);

                _sumI = _mm256_fmadd_ps(loadfp16(weight_hc_I), h_cont, _sumI);
                _sumF = _mm256_fmadd_ps(loadfp16(weight_hc_F), h_cont, _sumF);
                _sumO = _mm256_fmadd_ps(loadfp16(weight_hc_O), h_cont, _sumO);
                _sumG = _mm256_fmadd_ps(loadfp16(weight_hc_G), h_cont, _sumG);
                hidden_ptr_r += 8;
                weight_hc_I += 8;
                weight_hc_F += 8;
                weight_hc_O += 8;
                weight_hc_G += 8;
            }
            if (remain_size != 0)
            {
                unsigned short fp16_weights[4][8] = {{0}};
                float _xi_f[8] = {0};
                // No fast way to convert to fp32 one element at the time
                // so batch an 8 lane vector.
                for (int i = 0; i < remain_size; i++)
                {
                    _xi_f[i] = *x;
                    fp16_weights[0][i] = *weight_xc_I;
                    fp16_weights[1][i] = *weight_xc_F;
                    fp16_weights[2][i] = *weight_xc_O;
                    fp16_weights[3][i] = *weight_xc_G;
                    x++;
                    weight_xc_I++;
                    weight_xc_F++;
                    weight_xc_O++;
                    weight_xc_G++;
                }
                __m256 xi = _mm256_loadu_ps(_xi_f);
                _sumI = _mm256_fmadd_ps(loadfp16(fp16_weights[0]), xi, _sumI);
                _sumF = _mm256_fmadd_ps(loadfp16(fp16_weights[1]), xi, _sumF);
                _sumO = _mm256_fmadd_ps(loadfp16(fp16_weights[2]), xi, _sumO);
                _sumG = _mm256_fmadd_ps(loadfp16(fp16_weights[3]), xi, _sumG);
            }
            if (remain_num_output != 0)
            {
                unsigned short fp16_weights[4][8] = {{0}};
                float _hcont_f[8] = {0};
                // No fast way to convert to fp32 one element at the time
                // so batch an 8 lane vector.
                for (int i = 0; i < remain_num_output; i++)
                {
                    _hcont_f[i] = *hidden_ptr_r;
                    fp16_weights[0][i] = *weight_hc_I;
                    fp16_weights[1][i] = *weight_hc_F;
                    fp16_weights[2][i] = *weight_hc_O;
                    fp16_weights[3][i] = *weight_hc_G;
                    hidden_ptr_r++;
                    weight_hc_I++;
                    weight_hc_F++;
                    weight_hc_O++;
                    weight_hc_G++;
                }
                __m256 h_cont = _mm256_loadu_ps(_hcont_f);
                _sumI = _mm256_fmadd_ps(loadfp16(fp16_weights[0]), h_cont, _sumI);
                _sumF = _mm256_fmadd_ps(loadfp16(fp16_weights[1]), h_cont, _sumF);
                _sumO = _mm256_fmadd_ps(loadfp16(fp16_weights[2]), h_cont, _sumO);
                _sumG = _mm256_fmadd_ps(loadfp16(fp16_weights[3]), h_cont, _sumG);
            }

            float sums[4];
            _mm_storeu_ps(sums, HorizontalSums(_sumI, _sumF, _sumO, _sumG));
            sums[0] += bias_c_I[q];
            sums[1] += bias_c_F[q];
            sums[2] += bias_c_O[q];
            sums[3] += bias_c_G[q];
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
        float* output_data = top_blob.row(ti);
        float* cell_ptr = cell_state;
        float* hidden_ptr = hidden_state;
        const float* gates_data_I = gates.row(0);
        const float* gates_data_F = gates.row(1);
        const float* gates_data_O = gates.row(2);
        const float* gates_data_G = gates.row(3);
        int nn_activation = num_output >> 3;
        int remain_activations = num_output & 7;
        for (; nn_activation > 0; nn_activation--)
        {
            __m256 I = sigmoid_avx(_mm256_loadu_ps(gates_data_I));
            __m256 F = sigmoid_avx(_mm256_loadu_ps(gates_data_F));
            __m256 O = sigmoid_avx(_mm256_loadu_ps(gates_data_O));
            __m256 G = tanh_avx(_mm256_loadu_ps(gates_data_G));
            __m256 cell2 = _mm256_add_ps(_mm256_mul_ps(F, _mm256_loadu_ps(cell_ptr)), _mm256_mul_ps(I, G));
            __m256 H = _mm256_mul_ps(O, tanh_avx(cell2));
            _mm256_storeu_ps(cell_ptr, cell2);
            _mm256_storeu_ps(hidden_ptr, H);
            _mm256_storeu_ps(output_data, H);
            cell_ptr += 8;
            output_data += 8;
            hidden_ptr += 8;
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
            *hidden_ptr = H;
            *output_data = H;
            cell_ptr++;
            output_data++;
            hidden_ptr++;
            gates_data_I++;
            gates_data_F++;
            gates_data_O++;
            gates_data_G++;
        }

        // no cell output here
    }

    return 0;
}

static int lstm(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // 4 x num_output
    Mat gates(num_output, 4, 4u, opt.workspace_allocator);
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
        int remain_output = (num_output >> 1) << 1;
        for (int q = 0; q + 1 < num_output; q += 2)
        {
            const float* x = bottom_blob.row(ti);
            const float* hidden_ptr_r = hidden_state;
            const float* bias_c_I = bias_c.row(0);
            const float* bias_c_F = bias_c.row(1);
            const float* bias_c_O = bias_c.row(2);
            const float* bias_c_G = bias_c.row(3);

            float* gates_data_I = gates.row(0);
            float* gates_data_F = gates.row(1);
            float* gates_data_O = gates.row(2);
            float* gates_data_G = gates.row(3);
            // gate I F O G
            const float* weight_xc_I_0 = weight_xc.row(num_output * 0 + q);
            const float* weight_xc_F_0 = weight_xc.row(num_output * 1 + q);
            const float* weight_xc_O_0 = weight_xc.row(num_output * 2 + q);
            const float* weight_xc_G_0 = weight_xc.row(num_output * 3 + q);
            const float* weight_xc_I_1 = weight_xc.row(num_output * 0 + (q + 1));
            const float* weight_xc_F_1 = weight_xc.row(num_output * 1 + (q + 1));
            const float* weight_xc_O_1 = weight_xc.row(num_output * 2 + (q + 1));
            const float* weight_xc_G_1 = weight_xc.row(num_output * 3 + (q + 1));

            const float* weight_hc_I_0 = weight_hc.row(num_output * 0 + q);
            const float* weight_hc_F_0 = weight_hc.row(num_output * 1 + q);
            const float* weight_hc_O_0 = weight_hc.row(num_output * 2 + q);
            const float* weight_hc_G_0 = weight_hc.row(num_output * 3 + q);
            const float* weight_hc_I_1 = weight_hc.row(num_output * 0 + (q + 1));
            const float* weight_hc_F_1 = weight_hc.row(num_output * 1 + (q + 1));
            const float* weight_hc_O_1 = weight_hc.row(num_output * 2 + (q + 1));
            const float* weight_hc_G_1 = weight_hc.row(num_output * 3 + (q + 1));

            // float I = bias_c_I[q];
            // float F = bias_c_F[q];
            // float O = bias_c_O[q];
            // float G = bias_c_G[q];
            __m256 _sumI_0 = _mm256_setzero_ps();
            __m256 _sumF_0 = _mm256_setzero_ps();
            __m256 _sumO_0 = _mm256_setzero_ps();
            __m256 _sumG_0 = _mm256_setzero_ps();
            __m256 _sumI_1 = _mm256_setzero_ps();
            __m256 _sumF_1 = _mm256_setzero_ps();
            __m256 _sumO_1 = _mm256_setzero_ps();
            __m256 _sumG_1 = _mm256_setzero_ps();
            int nn_num_size = size >> 3;
            int remain_size = size & 7;
            for (; nn_num_size > 0; nn_num_size--)
            {
                __m256 xi = _mm256_loadu_ps(x);
                _sumI_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_I_0), xi, _sumI_0);
                _sumF_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_F_0), xi, _sumF_0);
                _sumO_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_O_0), xi, _sumO_0);
                _sumG_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_G_0), xi, _sumG_0);
                _sumI_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_I_1), xi, _sumI_1);
                _sumF_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_F_1), xi, _sumF_1);
                _sumO_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_O_1), xi, _sumO_1);
                _sumG_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_G_1), xi, _sumG_1);
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
            int nn_num_output = num_output >> 3;
            int remain_num_output = num_output & 7;
            for (; nn_num_output > 0; nn_num_output--)
            {
                __m256 h_cont = _mm256_loadu_ps(hidden_ptr_r);

                _sumI_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_I_0), h_cont, _sumI_0);
                _sumF_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_F_0), h_cont, _sumF_0);
                _sumO_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_O_0), h_cont, _sumO_0);
                _sumG_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_G_0), h_cont, _sumG_0);
                _sumI_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_I_1), h_cont, _sumI_1);
                _sumF_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_F_1), h_cont, _sumF_1);
                _sumO_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_O_1), h_cont, _sumO_1);
                _sumG_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_G_1), h_cont, _sumG_1);
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

            for (; remain_num_output > 0; remain_num_output--)
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

        for (int q = remain_output; q < num_output; q++)
        {
            const float* x = bottom_blob.row(ti);
            const float* hidden_ptr_r = hidden_state;
            const float* bias_c_I = bias_c.row(0);
            const float* bias_c_F = bias_c.row(1);
            const float* bias_c_O = bias_c.row(2);
            const float* bias_c_G = bias_c.row(3);

            float* gates_data_I = gates.row(0);
            float* gates_data_F = gates.row(1);
            float* gates_data_O = gates.row(2);
            float* gates_data_G = gates.row(3);
            // gate I F O G
            const float* weight_xc_I = weight_xc.row(num_output * 0 + q);
            const float* weight_xc_F = weight_xc.row(num_output * 1 + q);
            const float* weight_xc_O = weight_xc.row(num_output * 2 + q);
            const float* weight_xc_G = weight_xc.row(num_output * 3 + q);

            const float* weight_hc_I = weight_hc.row(num_output * 0 + q);
            const float* weight_hc_F = weight_hc.row(num_output * 1 + q);
            const float* weight_hc_O = weight_hc.row(num_output * 2 + q);
            const float* weight_hc_G = weight_hc.row(num_output * 3 + q);

            // float I = bias_c_I[q];
            // float F = bias_c_F[q];
            // float O = bias_c_O[q];
            // float G = bias_c_G[q];
            __m256 _sumI = _mm256_setzero_ps();
            __m256 _sumF = _mm256_setzero_ps();
            __m256 _sumO = _mm256_setzero_ps();
            __m256 _sumG = _mm256_setzero_ps();
            int nn_num_size = size >> 3;
            int remain_size = size & 7;
            for (; nn_num_size > 0; nn_num_size--)
            {
                __m256 xi = _mm256_loadu_ps(x);
                _sumI = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_I), xi, _sumI);
                _sumF = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_F), xi, _sumF);
                _sumO = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_O), xi, _sumO);
                _sumG = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_G), xi, _sumG);
                x += 8;
                weight_xc_I += 8;
                weight_xc_F += 8;
                weight_xc_O += 8;
                weight_xc_G += 8;
            }
            int nn_num_output = num_output >> 3;
            int remain_num_output = num_output & 7;
            for (; nn_num_output > 0; nn_num_output--)
            {
                __m256 h_cont = _mm256_loadu_ps(hidden_ptr_r);

                _sumI = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_I), h_cont, _sumI);
                _sumF = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_F), h_cont, _sumF);
                _sumO = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_O), h_cont, _sumO);
                _sumG = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_G), h_cont, _sumG);
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

            for (; remain_num_output > 0; remain_num_output--)
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
        float* output_data = top_blob.row(ti);
        float* cell_ptr = cell_state;
        float* hidden_ptr = hidden_state;
        const float* gates_data_I = gates.row(0);
        const float* gates_data_F = gates.row(1);
        const float* gates_data_O = gates.row(2);
        const float* gates_data_G = gates.row(3);
        int nn_activation = num_output >> 3;
        int remain_activations = num_output & 7;
        for (; nn_activation > 0; nn_activation--)
        {
            __m256 I = sigmoid_avx(_mm256_loadu_ps(gates_data_I));
            __m256 F = sigmoid_avx(_mm256_loadu_ps(gates_data_F));
            __m256 O = sigmoid_avx(_mm256_loadu_ps(gates_data_O));
            __m256 G = tanh_avx(_mm256_loadu_ps(gates_data_G));
            __m256 cell2 = _mm256_add_ps(_mm256_mul_ps(F, _mm256_loadu_ps(cell_ptr)), _mm256_mul_ps(I, G));
            __m256 H = _mm256_mul_ps(O, tanh_avx(cell2));
            _mm256_storeu_ps(cell_ptr, cell2);
            _mm256_storeu_ps(hidden_ptr, H);
            _mm256_storeu_ps(output_data, H);
            cell_ptr += 8;
            output_data += 8;
            hidden_ptr += 8;
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
            *hidden_ptr = H;
            *output_data = H;
            cell_ptr++;
            output_data++;
            hidden_ptr++;
            gates_data_I++;
            gates_data_F++;
            gates_data_O++;
            gates_data_G++;
        }

        // no cell output here
    }

    return 0;
}
#endif

int LSTM_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __AVX__
    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);
    // internal cell state
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
        if (opt.use_weight_fp16_storage)
        {
            // Uni directional
            int ret = lstm_fp16(bottom_blob, top_blob, direction, weight_xc_data_fp16.channel(0), bias_c_data.channel(0), weight_hc_data_fp16.channel(0), hidden, cell, opt);
            if (ret != 0)
                return ret;
        }
        else
        {
            // Uni directional
            int ret = lstm(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, cell, opt);
            if (ret != 0)
                return ret;
        }
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        if (opt.use_weight_fp16_storage)
        {
            // Uni directional
            int ret0 = lstm_fp16(bottom_blob, top_blob_forward, 0, weight_xc_data_fp16.channel(0), bias_c_data.channel(0), weight_hc_data_fp16.channel(0), hidden, cell, opt);
            if (ret0 != 0)
                return ret0;
        }
        else
        {
            // Uni directional
            int ret0 = lstm(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden, cell, opt);
            if (ret0 != 0)
                return ret0;
        }

        hidden.fill(0.0f);
        cell.fill(0.0f);
        if (opt.use_weight_fp16_storage)
        {
            // Uni directional
            int ret1 = lstm_fp16(bottom_blob, top_blob_reverse, 1, weight_xc_data_fp16.channel(1), bias_c_data.channel(1), weight_hc_data_fp16.channel(1), hidden, cell, opt);
            if (ret1 != 0)
                return ret1;
        }
        else
        {
            // Uni directional
            int ret1 = lstm(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), hidden, cell, opt);
            if (ret1 != 0)
                return ret1;
        }

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
#else
    return LSTM::forward(bottom_blob, top_blob, opt);
#endif
}

int LSTM_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if __AVX__
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

    if (opt.use_weight_fp16_storage)
    {
        // Uni directional
        int ret = lstm_fp16(bottom_blob, top_blob, direction, weight_xc_data_fp16.channel(0), bias_c_data.channel(0), weight_hc_data_fp16.channel(0), hidden_state, cell_state, opt);
        if (ret != 0)
            return ret;
    }
    else
    {
        // Uni directional
        int ret = lstm(bottom_blob, top_blob, direction, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), hidden_state, cell_state, opt);
        if (ret != 0)
            return ret;
    }
    return 0;
#else
    return LSTM::forward(bottom_blobs, top_blobs, opt);
#endif
}

} // namespace ncnn
