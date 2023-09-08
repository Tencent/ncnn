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

#include "lstm_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE2__

#include "x86_activation.h"
#include "x86_usability.h"

#include <math.h>
#include "layer_type.h"

namespace ncnn {

LSTM_x86::LSTM_x86()
{
    one_blob_only = false;
    support_inplace = false;
}

int LSTM_x86::create_pipeline(const Option& opt)
{
    // pack IFOG
    int num_directions = direction == 2 ? 2 : 1;
    int size = weight_data_size / num_directions / hidden_size / 4;

#if __AVX__
    weight_xc_data_packed.create(size, hidden_size / 2 + hidden_size % 2, num_directions, 32u, 8);
    bias_c_data_packed.create(hidden_size, 1, num_directions, 16u, 4);
    weight_hc_data_packed.create(num_output, hidden_size / 2 + hidden_size % 2, num_directions, 32u, 8);
#else
    weight_xc_data_packed.create(size, hidden_size, num_directions, 16u, 4);
    bias_c_data_packed.create(hidden_size, 1, num_directions, 16u, 4);
    weight_hc_data_packed.create(num_output, hidden_size, num_directions, 16u, 4);
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int dr = 0; dr < num_directions; dr++)
    {
        const Mat weight_xc = weight_xc_data.channel(dr);
        const Mat bias_c = bias_c_data.channel(dr);
        const Mat weight_hc = weight_hc_data.channel(dr);

        Mat weight_xc_data_packed_dr = weight_xc_data_packed.channel(dr);
        Mat bias_c_data_packed_dr = bias_c_data_packed.channel(dr);
        Mat weight_hc_data_packed_dr = weight_hc_data_packed.channel(dr);

        const float* bias_c_I = bias_c.row(0);
        const float* bias_c_F = bias_c.row(1);
        const float* bias_c_O = bias_c.row(2);
        const float* bias_c_G = bias_c.row(3);

        float* bias_c_IFOG = bias_c_data_packed_dr.row(0);

        int q = 0;
#if __AVX__
        for (; q + 1 < hidden_size; q += 2)
        {
            bias_c_IFOG[0] = bias_c_I[q];
            bias_c_IFOG[1] = bias_c_F[q];
            bias_c_IFOG[2] = bias_c_O[q];
            bias_c_IFOG[3] = bias_c_G[q];
            bias_c_IFOG[4] = bias_c_I[q + 1];
            bias_c_IFOG[5] = bias_c_F[q + 1];
            bias_c_IFOG[6] = bias_c_O[q + 1];
            bias_c_IFOG[7] = bias_c_G[q + 1];

            bias_c_IFOG += 8;

            const float* weight_xc_I = weight_xc.row(hidden_size * 0 + q);
            const float* weight_xc_F = weight_xc.row(hidden_size * 1 + q);
            const float* weight_xc_O = weight_xc.row(hidden_size * 2 + q);
            const float* weight_xc_G = weight_xc.row(hidden_size * 3 + q);
            const float* weight_xc_I_1 = weight_xc.row(hidden_size * 0 + q + 1);
            const float* weight_xc_F_1 = weight_xc.row(hidden_size * 1 + q + 1);
            const float* weight_xc_O_1 = weight_xc.row(hidden_size * 2 + q + 1);
            const float* weight_xc_G_1 = weight_xc.row(hidden_size * 3 + q + 1);

            const float* weight_hc_I = weight_hc.row(hidden_size * 0 + q);
            const float* weight_hc_F = weight_hc.row(hidden_size * 1 + q);
            const float* weight_hc_O = weight_hc.row(hidden_size * 2 + q);
            const float* weight_hc_G = weight_hc.row(hidden_size * 3 + q);
            const float* weight_hc_I_1 = weight_hc.row(hidden_size * 0 + q + 1);
            const float* weight_hc_F_1 = weight_hc.row(hidden_size * 1 + q + 1);
            const float* weight_hc_O_1 = weight_hc.row(hidden_size * 2 + q + 1);
            const float* weight_hc_G_1 = weight_hc.row(hidden_size * 3 + q + 1);

            float* weight_xc_IFOG = weight_xc_data_packed_dr.row(q / 2);
            float* weight_hc_IFOG = weight_hc_data_packed_dr.row(q / 2);

            for (int i = 0; i < size; i++)
            {
                weight_xc_IFOG[0] = weight_xc_I[i];
                weight_xc_IFOG[1] = weight_xc_F[i];
                weight_xc_IFOG[2] = weight_xc_O[i];
                weight_xc_IFOG[3] = weight_xc_G[i];
                weight_xc_IFOG[4] = weight_xc_I_1[i];
                weight_xc_IFOG[5] = weight_xc_F_1[i];
                weight_xc_IFOG[6] = weight_xc_O_1[i];
                weight_xc_IFOG[7] = weight_xc_G_1[i];

                weight_xc_IFOG += 8;
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc_IFOG[0] = weight_hc_I[i];
                weight_hc_IFOG[1] = weight_hc_F[i];
                weight_hc_IFOG[2] = weight_hc_O[i];
                weight_hc_IFOG[3] = weight_hc_G[i];
                weight_hc_IFOG[4] = weight_hc_I_1[i];
                weight_hc_IFOG[5] = weight_hc_F_1[i];
                weight_hc_IFOG[6] = weight_hc_O_1[i];
                weight_hc_IFOG[7] = weight_hc_G_1[i];

                weight_hc_IFOG += 8;
            }
        }
#endif // __AVX__
        for (; q < hidden_size; q++)
        {
            bias_c_IFOG[0] = bias_c_I[q];
            bias_c_IFOG[1] = bias_c_F[q];
            bias_c_IFOG[2] = bias_c_O[q];
            bias_c_IFOG[3] = bias_c_G[q];

            bias_c_IFOG += 4;

            const float* weight_xc_I = weight_xc.row(hidden_size * 0 + q);
            const float* weight_xc_F = weight_xc.row(hidden_size * 1 + q);
            const float* weight_xc_O = weight_xc.row(hidden_size * 2 + q);
            const float* weight_xc_G = weight_xc.row(hidden_size * 3 + q);

            const float* weight_hc_I = weight_hc.row(hidden_size * 0 + q);
            const float* weight_hc_F = weight_hc.row(hidden_size * 1 + q);
            const float* weight_hc_O = weight_hc.row(hidden_size * 2 + q);
            const float* weight_hc_G = weight_hc.row(hidden_size * 3 + q);

#if __AVX__
            float* weight_xc_IFOG = weight_xc_data_packed_dr.row(q / 2 + q % 2);
            float* weight_hc_IFOG = weight_hc_data_packed_dr.row(q / 2 + q % 2);
#else
            float* weight_xc_IFOG = weight_xc_data_packed_dr.row(q);
            float* weight_hc_IFOG = weight_hc_data_packed_dr.row(q);
#endif

            for (int i = 0; i < size; i++)
            {
                weight_xc_IFOG[0] = weight_xc_I[i];
                weight_xc_IFOG[1] = weight_xc_F[i];
                weight_xc_IFOG[2] = weight_xc_O[i];
                weight_xc_IFOG[3] = weight_xc_G[i];

                weight_xc_IFOG += 4;
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc_IFOG[0] = weight_hc_I[i];
                weight_hc_IFOG[1] = weight_hc_F[i];
                weight_hc_IFOG[2] = weight_hc_O[i];
                weight_hc_IFOG[3] = weight_hc_G[i];

                weight_hc_IFOG += 4;
            }
        }
    }

    if (opt.lightmode)
    {
        weight_xc_data.release();
        bias_c_data.release();
        weight_hc_data.release();
    }

    return 0;
}

static int lstm(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;
    int hidden_size = cell_state.w;

    // 4 x hidden_size
    Mat gates(4, hidden_size, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    Mat tmp_hidden_state;
    if (num_output != hidden_size)
    {
        tmp_hidden_state.create(hidden_size, 4u, opt.workspace_allocator);
        if (tmp_hidden_state.empty())
            return -100;
    }

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

#if __AVX__
        int nn_hidden_size = hidden_size >> 1;
        int remain_hidden_size_start = nn_hidden_size << 1;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = qq * 2;

            const float* bias_c_IFOG = (const float*)bias_c + q * 4;

            // gate I F O G
            const float* weight_xc_IFOG = weight_xc.row(q / 2);
            const float* weight_hc_IFOG = weight_hc.row(q / 2);

            __m256 _IFOG = _mm256_loadu_ps(bias_c_IFOG);
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();

            const float* x = bottom_blob.row(ti);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                __m256 _xi0 = _mm256_broadcast_ss(x);
                __m256 _xi1 = _mm256_broadcast_ss(x + 1);
                __m256 _xi2 = _mm256_broadcast_ss(x + 2);
                __m256 _xi3 = _mm256_broadcast_ss(x + 3);
                __m256 _weight_xc_IFOG0 = _mm256_loadu_ps(weight_xc_IFOG);
                __m256 _weight_xc_IFOG1 = _mm256_loadu_ps(weight_xc_IFOG + 8);
                __m256 _weight_xc_IFOG2 = _mm256_loadu_ps(weight_xc_IFOG + 16);
                __m256 _weight_xc_IFOG3 = _mm256_loadu_ps(weight_xc_IFOG + 24);
                _IFOG = _mm256_comp_fmadd_ps(_weight_xc_IFOG0, _xi0, _IFOG);
                _sum1 = _mm256_comp_fmadd_ps(_weight_xc_IFOG1, _xi1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_weight_xc_IFOG2, _xi2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_weight_xc_IFOG3, _xi3, _sum3);

                x += 4;
                weight_xc_IFOG += 32;
            }
            for (; i < size; i++)
            {
                __m256 _xi = _mm256_broadcast_ss(x);
                __m256 _weight_xc_IFOG = _mm256_loadu_ps(weight_xc_IFOG);
                _IFOG = _mm256_comp_fmadd_ps(_weight_xc_IFOG, _xi, _IFOG);

                x += 1;
                weight_xc_IFOG += 8;
            }

            const float* hidden_ptr = hidden_state;

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                __m256 _h_cont0 = _mm256_broadcast_ss(hidden_ptr);
                __m256 _h_cont1 = _mm256_broadcast_ss(hidden_ptr + 1);
                __m256 _h_cont2 = _mm256_broadcast_ss(hidden_ptr + 2);
                __m256 _h_cont3 = _mm256_broadcast_ss(hidden_ptr + 3);
                __m256 _weight_hc_IFOG0 = _mm256_loadu_ps(weight_hc_IFOG);
                __m256 _weight_hc_IFOG1 = _mm256_loadu_ps(weight_hc_IFOG + 8);
                __m256 _weight_hc_IFOG2 = _mm256_loadu_ps(weight_hc_IFOG + 16);
                __m256 _weight_hc_IFOG3 = _mm256_loadu_ps(weight_hc_IFOG + 24);
                _IFOG = _mm256_comp_fmadd_ps(_weight_hc_IFOG0, _h_cont0, _IFOG);
                _sum1 = _mm256_comp_fmadd_ps(_weight_hc_IFOG1, _h_cont1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_weight_hc_IFOG2, _h_cont2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_weight_hc_IFOG3, _h_cont3, _sum3);

                hidden_ptr += 4;
                weight_hc_IFOG += 32;
            }
            for (; i < num_output; i++)
            {
                __m256 _h_cont = _mm256_broadcast_ss(hidden_ptr);
                __m256 _weight_hc_IFOG = _mm256_loadu_ps(weight_hc_IFOG);
                _IFOG = _mm256_comp_fmadd_ps(_weight_hc_IFOG, _h_cont, _IFOG);

                hidden_ptr += 1;
                weight_hc_IFOG += 8;
            }

            float* gates_data = gates.row(q);

            _IFOG = _mm256_add_ps(_IFOG, _sum1);
            _sum2 = _mm256_add_ps(_sum2, _sum3);
            _IFOG = _mm256_add_ps(_IFOG, _sum2);

            _mm256_storeu_ps(gates_data, _IFOG);
        }
#else
        int nn_hidden_size = 0;
        int remain_hidden_size_start = 0;
#endif // __AVX__

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_hidden_size_start; q < hidden_size; q++)
        {
            const float* bias_c_IFOG = (const float*)bias_c + q * 4;

            // gate I F O G
#if __AVX__
            const float* weight_xc_IFOG = weight_xc.row(q / 2 + q % 2);
            const float* weight_hc_IFOG = weight_hc.row(q / 2 + q % 2);
#else
            const float* weight_xc_IFOG = weight_xc.row(q);
            const float* weight_hc_IFOG = weight_hc.row(q);
#endif

#if __SSE2__
            __m128 _IFOG = _mm_loadu_ps(bias_c_IFOG);
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();
#else  // __SSE2__
            float I = bias_c_IFOG[0];
            float F = bias_c_IFOG[1];
            float O = bias_c_IFOG[2];
            float G = bias_c_IFOG[3];
#endif // __SSE2__

            const float* x = bottom_blob.row(ti);

            int i = 0;
#if __SSE2__
            for (; i + 3 < size; i += 4)
            {
                __m128 _xi0 = _mm_load1_ps(x);
                __m128 _xi1 = _mm_load1_ps(x + 1);
                __m128 _xi2 = _mm_load1_ps(x + 2);
                __m128 _xi3 = _mm_load1_ps(x + 3);
                __m128 _weight_xc_IFOG0 = _mm_loadu_ps(weight_xc_IFOG);
                __m128 _weight_xc_IFOG1 = _mm_loadu_ps(weight_xc_IFOG + 4);
                __m128 _weight_xc_IFOG2 = _mm_loadu_ps(weight_xc_IFOG + 8);
                __m128 _weight_xc_IFOG3 = _mm_loadu_ps(weight_xc_IFOG + 12);
                _IFOG = _mm_comp_fmadd_ps(_weight_xc_IFOG0, _xi0, _IFOG);
                _sum1 = _mm_comp_fmadd_ps(_weight_xc_IFOG1, _xi1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_weight_xc_IFOG2, _xi2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_weight_xc_IFOG3, _xi3, _sum3);

                x += 4;
                weight_xc_IFOG += 16;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
#if __SSE2__
                __m128 _xi = _mm_load1_ps(x);
                __m128 _weight_xc_IFOG = _mm_loadu_ps(weight_xc_IFOG);
                _IFOG = _mm_comp_fmadd_ps(_weight_xc_IFOG, _xi, _IFOG);
#else  // __SSE2__
                float xi = x[0];
                I += xi * weight_xc_IFOG[0];
                F += xi * weight_xc_IFOG[1];
                O += xi * weight_xc_IFOG[2];
                G += xi * weight_xc_IFOG[3];
#endif // __SSE2__

                x += 1;
                weight_xc_IFOG += 4;
            }

            const float* hidden_ptr = hidden_state;

            i = 0;
#if __SSE2__
            for (; i + 3 < num_output; i += 4)
            {
                __m128 _h_cont0 = _mm_load1_ps(hidden_ptr);
                __m128 _h_cont1 = _mm_load1_ps(hidden_ptr + 1);
                __m128 _h_cont2 = _mm_load1_ps(hidden_ptr + 2);
                __m128 _h_cont3 = _mm_load1_ps(hidden_ptr + 3);
                __m128 _weight_hc_IFOG0 = _mm_loadu_ps(weight_hc_IFOG);
                __m128 _weight_hc_IFOG1 = _mm_loadu_ps(weight_hc_IFOG + 4);
                __m128 _weight_hc_IFOG2 = _mm_loadu_ps(weight_hc_IFOG + 8);
                __m128 _weight_hc_IFOG3 = _mm_loadu_ps(weight_hc_IFOG + 12);
                _IFOG = _mm_comp_fmadd_ps(_weight_hc_IFOG0, _h_cont0, _IFOG);
                _sum1 = _mm_comp_fmadd_ps(_weight_hc_IFOG1, _h_cont1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_weight_hc_IFOG2, _h_cont2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_weight_hc_IFOG3, _h_cont3, _sum3);

                hidden_ptr += 4;
                weight_hc_IFOG += 16;
            }
#endif // __SSE2__
            for (; i < num_output; i++)
            {
#if __SSE2__
                __m128 _h_cont = _mm_load1_ps(hidden_ptr);
                __m128 _weight_hc_IFOG = _mm_loadu_ps(weight_hc_IFOG);
                _IFOG = _mm_comp_fmadd_ps(_weight_hc_IFOG, _h_cont, _IFOG);
#else  // __SSE2__
                float h_cont = hidden_ptr[0];
                I += h_cont * weight_hc_IFOG[0];
                F += h_cont * weight_hc_IFOG[1];
                O += h_cont * weight_hc_IFOG[2];
                G += h_cont * weight_hc_IFOG[3];
#endif // __SSE2__

                hidden_ptr += 1;
                weight_hc_IFOG += 4;
            }

            float* gates_data = gates.row(q);

#if __SSE2__
            _IFOG = _mm_add_ps(_IFOG, _sum1);
            _sum2 = _mm_add_ps(_sum2, _sum3);
            _IFOG = _mm_add_ps(_IFOG, _sum2);

            _mm_storeu_ps(gates_data, _IFOG);
#else  // __SSE2__
            gates_data[0] = I;
            gates_data[1] = F;
            gates_data[2] = O;
            gates_data[3] = G;
#endif // __SSE2__
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
        float* tmp_hidden_ptr = tmp_hidden_state;

#if __SSE2__
        nn_hidden_size = hidden_size >> 2;
        remain_hidden_size_start = nn_hidden_size << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = qq * 4;

            const float* gates_data = gates.row(q);

            __m128 _IFOG_4x4_0 = _mm_loadu_ps(gates_data);
            __m128 _IFOG_4x4_1 = _mm_loadu_ps(gates_data + 4);
            __m128 _IFOG_4x4_2 = _mm_loadu_ps(gates_data + 8);
            __m128 _IFOG_4x4_3 = _mm_loadu_ps(gates_data + 12);

            _MM_TRANSPOSE4_PS(_IFOG_4x4_0, _IFOG_4x4_1, _IFOG_4x4_2, _IFOG_4x4_3);

            __m128 _lstm_I = sigmoid_sse(_IFOG_4x4_0);
            __m128 _lstm_F = sigmoid_sse(_IFOG_4x4_1);
            __m128 _lstm_O = sigmoid_sse(_IFOG_4x4_2);
            __m128 _lstm_G = tanh_sse(_IFOG_4x4_3);

            __m128 _cell2 = _mm_add_ps(_mm_mul_ps(_lstm_F, _mm_loadu_ps(cell_ptr + q)), _mm_mul_ps(_lstm_I, _lstm_G));
            __m128 _lstm_H = _mm_mul_ps(_lstm_O, tanh_sse(_cell2));

            _mm_storeu_ps(cell_ptr + q, _cell2);

            if (num_output == hidden_size)
            {
                _mm_storeu_ps(hidden_ptr + q, _lstm_H);
                _mm_storeu_ps(output_data + q, _lstm_H);
            }
            else
            {
                _mm_storeu_ps(tmp_hidden_ptr + q, _lstm_H);
            }
        }
#else  // __SSE2__
        remain_hidden_size_start = 0;
#endif // __SSE2__
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_hidden_size_start; q < hidden_size; q++)
        {
            const float* gates_data = gates.row(q);

            float I = gates_data[0];
            float F = gates_data[1];
            float O = gates_data[2];
            float G = gates_data[3];

            I = 1.f / (1.f + expf(-I));
            F = 1.f / (1.f + expf(-F));
            O = 1.f / (1.f + expf(-O));
            G = tanhf(G);

            float cell2 = F * cell_ptr[q] + I * G;
            float H = O * tanhf(cell2);

            cell_ptr[q] = cell2;
            if (num_output == hidden_size)
            {
                hidden_ptr[q] = H;
                output_data[q] = H;
            }
            else
            {
                tmp_hidden_ptr[q] = H;
            }
        }

        if (num_output != hidden_size)
        {
            // int nn_num_output = num_output >> 2;
            // int remain_num_output_start = nn_num_output << 2;
            // #pragma omp parallel for num_threads(opt.num_threads)
            // for (int qq = 0; qq < nn_num_output; qq++)
            // {
            //     int q = qq * 4;
            //
            // }
            int remain_num_output_start = 0;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = remain_num_output_start; q < num_output; q++)
            {
                const float* hr = weight_hr.row(q);
                const float* tmp_hidden_ptr = tmp_hidden_state;

                float H = 0;
                for (int i = 0; i < hidden_size; i++)
                {
                    H += tmp_hidden_ptr[i] * hr[i];
                }

                output_data[q] = H;
                hidden_ptr[q] = H;
            }
        }
    }

    return 0;
}

int LSTM_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int T = bottom_blob.h;

    int num_directions = direction == 2 ? 2 : 1;

    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    Mat cell(hidden_size, 4u, opt.workspace_allocator);
    if (cell.empty())
        return -100;
    cell.fill(0.f);

    top_blob.create(num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = lstm(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
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

        int ret0 = lstm(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.0f);
        cell.fill(0.0f);

        int ret1 = lstm(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1), hidden, cell, opt);
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

int LSTM_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

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
        hidden.create(num_output, num_directions, 4u, hidden_cell_allocator);
        if (hidden.empty())
            return -100;
        hidden.fill(0.f);

        cell.create(hidden_size, num_directions, 4u, hidden_cell_allocator);
        if (cell.empty())
            return -100;
        cell.fill(0.f);
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = lstm(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
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

        Mat hidden0 = hidden.row_range(0, 1);
        Mat cell0 = cell.row_range(0, 1);
        int ret0 = lstm(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden0, cell0, opt);
        if (ret0 != 0)
            return ret0;

        Mat hidden1 = hidden.row_range(1, 1);
        Mat cell1 = cell.row_range(1, 1);
        int ret1 = lstm(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1), hidden1, cell1, opt);
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

    if (top_blobs.size() == 3)
    {
        top_blobs[1] = hidden;
        top_blobs[2] = cell;
    }

    return 0;
}

} // namespace ncnn
