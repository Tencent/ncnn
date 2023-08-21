// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "gru_arm.h"

#include <math.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static int gru_fp16s(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // 2 x num_output
    Mat gates(4 * 2, num_output / 4 + num_output % 4, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        int nn_num_output = num_output >> 2;
        int remain_num_output_start = nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            const __fp16* x = bottom_blob.row<const __fp16>(ti);

            // gate reset update
            const __fp16* bias_c_RUBNWN = (const __fp16*)bias_c + q * 4;

            const __fp16* weight_xc_RUN = weight_xc.row<const __fp16>(q / 4);
            const __fp16* weight_hc_RUN = weight_hc.row<const __fp16>(q / 4);

            float32x4_t _gru_R = vcvt_f32_f16(vld1_f16(bias_c_RUBNWN));
            float32x4_t _gru_U = vcvt_f32_f16(vld1_f16(bias_c_RUBNWN + 4));
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);
            float32x4_t _sum4 = vdupq_n_f32(0.f);
            float32x4_t _sum5 = vdupq_n_f32(0.f);
            float32x4_t _sum6 = vdupq_n_f32(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _xi = vcvt_f32_f16(vld1_f16(x + i));
                float32x4_t _weight_xc_R = vcvt_f32_f16(vld1_f16(weight_xc_RUN));
                float32x4_t _weight_xc_U = vcvt_f32_f16(vld1_f16(weight_xc_RUN + 4));
                float32x4_t _weight_xc_R_1 = vcvt_f32_f16(vld1_f16(weight_xc_RUN + 8));
                float32x4_t _weight_xc_U_1 = vcvt_f32_f16(vld1_f16(weight_xc_RUN + 12));
                float32x4_t _weight_xc_R_2 = vcvt_f32_f16(vld1_f16(weight_xc_RUN + 16));
                float32x4_t _weight_xc_U_2 = vcvt_f32_f16(vld1_f16(weight_xc_RUN + 20));
                float32x4_t _weight_xc_R_3 = vcvt_f32_f16(vld1_f16(weight_xc_RUN + 24));
                float32x4_t _weight_xc_U_3 = vcvt_f32_f16(vld1_f16(weight_xc_RUN + 28));
                _gru_R = vfmaq_laneq_f32(_gru_R, _weight_xc_R, _xi, 0);
                _gru_U = vfmaq_laneq_f32(_gru_U, _weight_xc_U, _xi, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_xc_R_1, _xi, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_xc_U_1, _xi, 1);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_xc_R_2, _xi, 2);
                _sum4 = vfmaq_laneq_f32(_sum4, _weight_xc_U_2, _xi, 2);
                _sum5 = vfmaq_laneq_f32(_sum5, _weight_xc_R_3, _xi, 3);
                _sum6 = vfmaq_laneq_f32(_sum6, _weight_xc_U_3, _xi, 3);

                weight_xc_RUN += 32;
            }
            for (; i < size; i++)
            {
                __fp16 xi = x[i];

                float32x4_t _xi = vcvt_f32_f16(vdup_n_f16(xi));
                float32x4_t _weight_xc_R = vcvt_f32_f16(vld1_f16(weight_xc_RUN));
                float32x4_t _weight_xc_U = vcvt_f32_f16(vld1_f16(weight_xc_RUN + 4));
                _gru_R = vmlaq_f32(_gru_R, _weight_xc_R, _xi);
                _gru_U = vmlaq_f32(_gru_U, _weight_xc_U, _xi);

                weight_xc_RUN += 8;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float32x4_t _h_cont = vld1q_f32((const float*)hidden_state + i);
                float32x4_t _weight_hc_R = vcvt_f32_f16(vld1_f16(weight_hc_RUN));
                float32x4_t _weight_hc_U = vcvt_f32_f16(vld1_f16(weight_hc_RUN + 4));
                float32x4_t _weight_hc_R_1 = vcvt_f32_f16(vld1_f16(weight_hc_RUN + 8));
                float32x4_t _weight_hc_U_1 = vcvt_f32_f16(vld1_f16(weight_hc_RUN + 12));
                float32x4_t _weight_hc_R_2 = vcvt_f32_f16(vld1_f16(weight_hc_RUN + 16));
                float32x4_t _weight_hc_U_2 = vcvt_f32_f16(vld1_f16(weight_hc_RUN + 20));
                float32x4_t _weight_hc_R_3 = vcvt_f32_f16(vld1_f16(weight_hc_RUN + 24));
                float32x4_t _weight_hc_U_3 = vcvt_f32_f16(vld1_f16(weight_hc_RUN + 28));
                _gru_R = vfmaq_laneq_f32(_gru_R, _weight_hc_R, _h_cont, 0);
                _gru_U = vfmaq_laneq_f32(_gru_U, _weight_hc_U, _h_cont, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_hc_R_1, _h_cont, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_hc_U_1, _h_cont, 1);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_hc_R_2, _h_cont, 2);
                _sum4 = vfmaq_laneq_f32(_sum4, _weight_hc_U_2, _h_cont, 2);
                _sum5 = vfmaq_laneq_f32(_sum5, _weight_hc_R_3, _h_cont, 3);
                _sum6 = vfmaq_laneq_f32(_sum6, _weight_hc_U_3, _h_cont, 3);

                weight_hc_RUN += 32;
            }
            for (; i < num_output; i++)
            {
                float h_cont = hidden_state[i];

                float32x4_t _h_cont = vdupq_n_f32(h_cont);
                float32x4_t _weight_hc_R = vcvt_f32_f16(vld1_f16(weight_hc_RUN));
                float32x4_t _weight_hc_U = vcvt_f32_f16(vld1_f16(weight_hc_RUN + 4));
                _gru_R = vmlaq_f32(_gru_R, _weight_hc_R, _h_cont);
                _gru_U = vmlaq_f32(_gru_U, _weight_hc_U, _h_cont);

                weight_hc_RUN += 8;
            }

            _gru_R = vaddq_f32(_gru_R, _sum1);
            _gru_U = vaddq_f32(_gru_U, _sum2);
            _sum3 = vaddq_f32(_sum3, _sum5);
            _sum4 = vaddq_f32(_sum4, _sum6);
            _gru_R = vaddq_f32(_gru_R, _sum3);
            _gru_U = vaddq_f32(_gru_U, _sum4);

            // sigmoid(R)
            // sigmoid(U)
            _gru_R = sigmoid_ps(_gru_R);
            _gru_U = sigmoid_ps(_gru_U);

            // gate new
            float32x4_t _gru_N = vcvt_f32_f16(vld1_f16(bias_c_RUBNWN + 8));
            _sum1 = vdupq_n_f32(0.f);
            _sum2 = vdupq_n_f32(0.f);
            _sum3 = vdupq_n_f32(0.f);

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float32x4_t _h_cont = vld1q_f32((const float*)hidden_state + i);
                float32x4_t _weight_hc_N = vcvt_f32_f16(vld1_f16(weight_hc_RUN));
                float32x4_t _weight_hc_N_1 = vcvt_f32_f16(vld1_f16(weight_hc_RUN + 4));
                float32x4_t _weight_hc_N_2 = vcvt_f32_f16(vld1_f16(weight_hc_RUN + 8));
                float32x4_t _weight_hc_N_3 = vcvt_f32_f16(vld1_f16(weight_hc_RUN + 12));
                _gru_N = vfmaq_laneq_f32(_gru_N, _weight_hc_N, _h_cont, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_hc_N_1, _h_cont, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_hc_N_2, _h_cont, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_hc_N_3, _h_cont, 3);

                weight_hc_RUN += 16;
            }
            for (; i < num_output; i++)
            {
                float h_cont = hidden_state[i];

                float32x4_t _h_cont = vdupq_n_f32(h_cont);
                float32x4_t _weight_hc_N = vcvt_f32_f16(vld1_f16(weight_hc_RUN));
                _gru_N = vmlaq_f32(_gru_N, _weight_hc_N, _h_cont);

                weight_hc_RUN += 4;
            }

            _gru_N = vaddq_f32(_gru_N, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _gru_N = vaddq_f32(_gru_N, _sum2);

            _gru_N = vmlaq_f32(vcvt_f32_f16(vld1_f16(bias_c_RUBNWN + 12)), _gru_R, _gru_N);
            _sum1 = vdupq_n_f32(0.f);
            _sum2 = vdupq_n_f32(0.f);
            _sum3 = vdupq_n_f32(0.f);

            i = 0;
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _xi = vcvt_f32_f16(vld1_f16(x + i));
                float32x4_t _weight_xc_N = vcvt_f32_f16(vld1_f16(weight_xc_RUN));
                float32x4_t _weight_xc_N_1 = vcvt_f32_f16(vld1_f16(weight_xc_RUN + 4));
                float32x4_t _weight_xc_N_2 = vcvt_f32_f16(vld1_f16(weight_xc_RUN + 8));
                float32x4_t _weight_xc_N_3 = vcvt_f32_f16(vld1_f16(weight_xc_RUN + 12));
                _gru_N = vfmaq_laneq_f32(_gru_N, _weight_xc_N, _xi, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_xc_N_1, _xi, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_xc_N_2, _xi, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_xc_N_3, _xi, 3);

                weight_xc_RUN += 16;
            }
            for (; i < size; i++)
            {
                __fp16 xi = x[i];

                float32x4_t _xi = vcvt_f32_f16(vdup_n_f16(xi));
                float32x4_t _weight_xc_N = vcvt_f32_f16(vld1_f16(weight_xc_RUN));
                _gru_N = vmlaq_f32(_gru_N, _weight_xc_N, _xi);

                weight_xc_RUN += 4;
            }

            _gru_N = vaddq_f32(_gru_N, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _gru_N = vaddq_f32(_gru_N, _sum2);

            // tanh(N)
            _gru_N = tanh_ps(_gru_N);

            float* gates_data = gates.row(q / 4);

            vst1q_f32(gates_data, _gru_U);
            vst1q_f32(gates_data + 4, _gru_N);
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            const __fp16* x = bottom_blob.row<const __fp16>(ti);

            // gate reset update
            const __fp16* bias_c_RUBNWN = (const __fp16*)bias_c + q * 4;

            const __fp16* weight_xc_RUN = weight_xc.row<const __fp16>(q / 4 + q % 4);
            const __fp16* weight_hc_RUN = weight_hc.row<const __fp16>(q / 4 + q % 4);

            float R = (float)bias_c_RUBNWN[0];
            float U = (float)bias_c_RUBNWN[1];

            for (int i = 0; i < size; i++)
            {
                float xi = (float)x[i];

                R += (float)weight_xc_RUN[0] * xi;
                U += (float)weight_xc_RUN[1] * xi;

                weight_xc_RUN += 2;
            }

            for (int i = 0; i < num_output; i++)
            {
                float h_cont = hidden_state[i];

                R += (float)weight_hc_RUN[0] * h_cont;
                U += (float)weight_hc_RUN[1] * h_cont;

                weight_hc_RUN += 2;
            }

            // sigmoid(R)
            // sigmoid(U)
            R = 1.f / (1.f + expf(-R));
            U = 1.f / (1.f + expf(-U));

            // gate new
            float N = (float)bias_c_RUBNWN[2];

            for (int i = 0; i < num_output; i++)
            {
                float h_cont = hidden_state[i];

                N += (float)weight_hc_RUN[0] * h_cont;

                weight_hc_RUN += 1;
            }

            N = (float)bias_c_RUBNWN[3] + R * N;

            for (int i = 0; i < size; i++)
            {
                float xi = (float)x[i];

                N += (float)weight_xc_RUN[0] * xi;

                weight_xc_RUN += 1;
            }

            // tanh(N)
            N = tanhf(N);

            float* gates_data = gates.row(q / 4 + q % 4);

            gates_data[0] = U;
            gates_data[1] = N;
        }

        // h_t := (1 - update) .* new + update .* h_{t-1}
        __fp16* output_data = top_blob.row<__fp16>(ti);

        float* hidden_ptr = hidden_state;

        nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            const float* gates_data = gates.row(q / 4);

            float32x4_t _gru_U = vld1q_f32(gates_data);
            float32x4_t _gru_N = vld1q_f32(gates_data + 4);

            float32x4_t _gru_H = vaddq_f32(vmulq_f32(vsubq_f32(vdupq_n_f32(1.f), _gru_U), _gru_N), vmulq_f32(_gru_U, vld1q_f32(hidden_ptr + q)));

            vst1q_f32(hidden_ptr + q, _gru_H);
            vst1_f16(output_data + q, vcvt_f16_f32(_gru_H));
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            const float* gates_data = gates.row(q / 4 + q % 4);

            float U = gates_data[0];
            float N = gates_data[1];

            float H = (1 - U) * N + U * hidden_ptr[q];

            hidden_ptr[q] = H;
            output_data[q] = (__fp16)H;
        }
    }

    return 0;
}

static int gru_fp16sa(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // 2 x num_output
    Mat gates(4 * 2, num_output / 4 + num_output % 4, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        int nn_num_output = num_output >> 2;
        int remain_num_output_start = nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            const __fp16* x = bottom_blob.row<const __fp16>(ti);

            // gate reset update
            const __fp16* bias_c_RUBNWN = (const __fp16*)bias_c + q * 4;

            const __fp16* weight_xc_RUN = weight_xc.row<const __fp16>(q / 4);
            const __fp16* weight_hc_RUN = weight_hc.row<const __fp16>(q / 4);

            float16x8_t _RU = vld1q_f16(bias_c_RUBNWN);
            float16x8_t _sum1 = vdupq_n_f16((__fp16)0.f);
            float16x8_t _sum2 = vdupq_n_f16((__fp16)0.f);
            float16x8_t _sum3 = vdupq_n_f16((__fp16)0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                asm volatile(
                    "ld1    {v4.4h}, [%0], #8       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    "fmla   %2.8h, v0.8h, v4.h[0]   \n"
                    "fmla   %3.8h, v1.8h, v4.h[1]   \n"
                    "fmla   %4.8h, v2.8h, v4.h[2]   \n"
                    "fmla   %5.8h, v3.8h, v4.h[3]   \n"
                    : "=r"(x),
                    "=r"(weight_xc_RUN),
                    "=w"(_RU),
                    "=w"(_sum1),
                    "=w"(_sum2),
                    "=w"(_sum3)
                    : "0"(x),
                    "1"(weight_xc_RUN),
                    "2"(_RU),
                    "3"(_sum1),
                    "4"(_sum2),
                    "5"(_sum3)
                    : "memory", "v0", "v1", "v2", "v3", "v4");
            }
            for (; i < size; i++)
            {
                __fp16 xi = *x++;

                float16x8_t _xi = vdupq_n_f16(xi);
                float16x8_t _weight_xc_RU = vld1q_f16(weight_xc_RUN);
                _RU = vfmaq_f16(_RU, _weight_xc_RU, _xi);

                weight_xc_RUN += 8;
            }

            const float* hidden_ptr = hidden_state;

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                asm volatile(
                    "ld1    {v4.4s}, [%0], #16      \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    "fcvtn  v4.4h, v4.4s            \n"
                    "fmla   %2.8h, v0.8h, v4.h[0]   \n"
                    "fmla   %3.8h, v1.8h, v4.h[1]   \n"
                    "fmla   %4.8h, v2.8h, v4.h[2]   \n"
                    "fmla   %5.8h, v3.8h, v4.h[3]   \n"
                    : "=r"(hidden_ptr),
                    "=r"(weight_hc_RUN),
                    "=w"(_RU),
                    "=w"(_sum1),
                    "=w"(_sum2),
                    "=w"(_sum3)
                    : "0"(hidden_ptr),
                    "1"(weight_hc_RUN),
                    "2"(_RU),
                    "3"(_sum1),
                    "4"(_sum2),
                    "5"(_sum3)
                    : "memory", "v0", "v1", "v2", "v3", "v4");
            }
            for (; i < num_output; i++)
            {
                float h_cont = *hidden_ptr++;

                float16x8_t _h_cont = vdupq_n_f16((__fp16)h_cont);
                float16x8_t _weight_hc_RU = vld1q_f16(weight_hc_RUN);
                _RU = vfmaq_f16(_RU, _weight_hc_RU, _h_cont);

                weight_hc_RUN += 8;
            }

            _RU = vaddq_f16(_RU, _sum1);
            _sum2 = vaddq_f16(_sum2, _sum3);
            _RU = vaddq_f16(_RU, _sum2);

            // sigmoid(R)
            // sigmoid(U)
            float32x4_t _R32 = sigmoid_ps(vcvt_f32_f16(vget_low_f16(_RU)));
            float32x4_t _U32 = sigmoid_ps(vcvt_f32_f16(vget_high_f16(_RU)));

            x -= size;
            hidden_ptr = hidden_state;

            // gate new
            float16x4_t _gru_N = vld1_f16(bias_c_RUBNWN + 8);
            float16x4_t _sum4 = vdup_n_f16((__fp16)0.f);
            float16x4_t _sum5 = vdup_n_f16((__fp16)0.f);
            float16x4_t _sum6 = vdup_n_f16((__fp16)0.f);

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                asm volatile(
                    "ld1    {v4.4s}, [%0], #16      \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n"
                    "fcvtn  v4.4h, v4.4s            \n"
                    "fmla   %2.4h, v0.4h, v4.h[0]   \n"
                    "fmla   %3.4h, v1.4h, v4.h[1]   \n"
                    "fmla   %4.4h, v2.4h, v4.h[2]   \n"
                    "fmla   %5.4h, v3.4h, v4.h[3]   \n"
                    : "=r"(hidden_ptr),
                    "=r"(weight_hc_RUN),
                    "=w"(_gru_N),
                    "=w"(_sum4),
                    "=w"(_sum5),
                    "=w"(_sum6)
                    : "0"(hidden_ptr),
                    "1"(weight_hc_RUN),
                    "2"(_gru_N),
                    "3"(_sum4),
                    "4"(_sum5),
                    "5"(_sum6)
                    : "memory", "v0", "v1", "v2", "v3", "v4");
            }
            for (; i < num_output; i++)
            {
                float h_cont = *hidden_ptr++;

                float16x4_t _h_cont = vdup_n_f16((__fp16)h_cont);
                float16x4_t _weight_hc_N = vld1_f16(weight_hc_RUN);
                _gru_N = vfma_f16(_gru_N, _weight_hc_N, _h_cont);

                weight_hc_RUN += 4;
            }

            _gru_N = vadd_f16(_gru_N, _sum4);
            _sum5 = vadd_f16(_sum5, _sum6);
            _gru_N = vadd_f16(_gru_N, _sum5);

            _gru_N = vfma_f16(vld1_f16(bias_c_RUBNWN + 12), vcvt_f16_f32(_R32), _gru_N);
            _sum4 = vdup_n_f16((__fp16)0.f);
            _sum5 = vdup_n_f16((__fp16)0.f);
            _sum6 = vdup_n_f16((__fp16)0.f);

            i = 0;
            for (; i + 3 < size; i += 4)
            {
                asm volatile(
                    "ld1    {v4.4h}, [%0], #8       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n"
                    "fmla   %2.4h, v0.4h, v4.h[0]   \n"
                    "fmla   %3.4h, v1.4h, v4.h[1]   \n"
                    "fmla   %4.4h, v2.4h, v4.h[2]   \n"
                    "fmla   %5.4h, v3.4h, v4.h[3]   \n"
                    : "=r"(x),
                    "=r"(weight_xc_RUN),
                    "=w"(_gru_N),
                    "=w"(_sum4),
                    "=w"(_sum5),
                    "=w"(_sum6)
                    : "0"(x),
                    "1"(weight_xc_RUN),
                    "2"(_gru_N),
                    "3"(_sum4),
                    "4"(_sum5),
                    "5"(_sum6)
                    : "memory", "v0", "v1", "v2", "v3", "v4");
            }
            for (; i < size; i++)
            {
                __fp16 xi = *x++;

                float16x4_t _xi = vdup_n_f16(xi);
                float16x4_t _weight_xc_N = vld1_f16(weight_xc_RUN);
                _gru_N = vfma_f16(_gru_N, _weight_xc_N, _xi);

                weight_xc_RUN += 4;
            }

            _gru_N = vadd_f16(_gru_N, _sum4);
            _sum5 = vadd_f16(_sum5, _sum6);
            _gru_N = vadd_f16(_gru_N, _sum5);

            // tanh(N)
            float32x4_t _N32 = tanh_ps(vcvt_f32_f16(_gru_N));

            float* gates_data = gates.row(q / 4);

            vst1q_f32(gates_data, _U32);
            vst1q_f32(gates_data + 4, _N32);
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            const __fp16* x = bottom_blob.row<const __fp16>(ti);

            // gate reset update
            const __fp16* bias_c_RUBNWN = (const __fp16*)bias_c + q * 4;

            const __fp16* weight_xc_RUN = weight_xc.row<const __fp16>(q / 4 + q % 4);
            const __fp16* weight_hc_RUN = weight_hc.row<const __fp16>(q / 4 + q % 4);

            __fp16 R = bias_c_RUBNWN[0];
            __fp16 U = bias_c_RUBNWN[1];

            for (int i = 0; i < size; i++)
            {
                __fp16 xi = x[i];

                R += weight_xc_RUN[0] * xi;
                U += weight_xc_RUN[1] * xi;

                weight_xc_RUN += 2;
            }

            for (int i = 0; i < num_output; i++)
            {
                __fp16 h_cont = (__fp16)hidden_state[i];

                R += weight_hc_RUN[0] * h_cont;
                U += weight_hc_RUN[1] * h_cont;

                weight_hc_RUN += 2;
            }

            // sigmoid(R)
            // sigmoid(U)
            float R32 = 1.f / (1.f + expf((float)-R));
            float U32 = 1.f / (1.f + expf((float)-U));

            // gate new
            __fp16 N = bias_c_RUBNWN[2];

            for (int i = 0; i < num_output; i++)
            {
                __fp16 h_cont = (__fp16)hidden_state[i];

                N += weight_hc_RUN[0] * h_cont;

                weight_hc_RUN += 1;
            }

            N = bias_c_RUBNWN[3] + (__fp16)R32 * N;

            for (int i = 0; i < size; i++)
            {
                __fp16 xi = x[i];

                N += weight_xc_RUN[0] * xi;

                weight_xc_RUN += 1;
            }

            // tanh(N)
            float N32 = tanhf((float)N);

            float* gates_data = gates.row(q / 4 + q % 4);

            gates_data[0] = U32;
            gates_data[1] = N32;
        }

        // h_t := (1 - update) .* new + update .* h_{t-1}
        __fp16* output_data = top_blob.row<__fp16>(ti);

        float* hidden_ptr = hidden_state;

        nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            const float* gates_data = gates.row(q / 4);

            float32x4_t _gru_U = vld1q_f32(gates_data);
            float32x4_t _gru_N = vld1q_f32(gates_data + 4);

            float32x4_t _gru_H = vaddq_f32(vmulq_f32(vsubq_f32(vdupq_n_f32(1.f), _gru_U), _gru_N), vmulq_f32(_gru_U, vld1q_f32(hidden_ptr + q)));

            vst1q_f32(hidden_ptr + q, _gru_H);
            vst1_f16(output_data + q, vcvt_f16_f32(_gru_H));
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            const float* gates_data = gates.row(q / 4 + q % 4);

            float U = gates_data[0];
            float N = gates_data[1];

            float H = (1 - U) * N + U * hidden_ptr[q];

            hidden_ptr[q] = H;
            output_data[q] = (__fp16)H;
        }
    }

    return 0;
}

int GRU_arm::create_pipeline_fp16s(const Option& opt)
{
    // pack RUN
    int num_directions = direction == 2 ? 2 : 1;
    int size = weight_data_size / num_directions / num_output / 3;

    weight_xc_data_packed.create(size * 12, num_output / 4 + num_output % 4, num_directions, 2u, 1);
    bias_c_data_packed.create(num_output, 1, num_directions, 8u, 4);
    weight_hc_data_packed.create(num_output * 12, num_output / 4 + num_output % 4, num_directions, 2u, 1);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int dr = 0; dr < num_directions; dr++)
    {
        const Mat weight_xc = weight_xc_data.channel(dr);
        const Mat bias_c = bias_c_data.channel(dr);
        const Mat weight_hc = weight_hc_data.channel(dr);

        Mat weight_xc_data_packed_dr = weight_xc_data_packed.channel(dr);
        Mat bias_c_data_packed_dr = bias_c_data_packed.channel(dr);
        Mat weight_hc_data_packed_dr = weight_hc_data_packed.channel(dr);

        const float* bias_c_R = bias_c.row(0);
        const float* bias_c_U = bias_c.row(1);
        const float* bias_c_WN = bias_c.row(2);
        const float* bias_c_BN = bias_c.row(3);

        __fp16* bias_c_RUBNWN = bias_c_data_packed_dr.row<__fp16>(0);

        int q = 0;
        for (; q + 3 < num_output; q += 4)
        {
            bias_c_RUBNWN[0] = (__fp16)bias_c_R[q];
            bias_c_RUBNWN[1] = (__fp16)bias_c_R[q + 1];
            bias_c_RUBNWN[2] = (__fp16)bias_c_R[q + 2];
            bias_c_RUBNWN[3] = (__fp16)bias_c_R[q + 3];
            bias_c_RUBNWN[4] = (__fp16)bias_c_U[q];
            bias_c_RUBNWN[5] = (__fp16)bias_c_U[q + 1];
            bias_c_RUBNWN[6] = (__fp16)bias_c_U[q + 2];
            bias_c_RUBNWN[7] = (__fp16)bias_c_U[q + 3];
            bias_c_RUBNWN[8] = (__fp16)bias_c_BN[q];
            bias_c_RUBNWN[9] = (__fp16)bias_c_BN[q + 1];
            bias_c_RUBNWN[10] = (__fp16)bias_c_BN[q + 2];
            bias_c_RUBNWN[11] = (__fp16)bias_c_BN[q + 3];
            bias_c_RUBNWN[12] = (__fp16)bias_c_WN[q];
            bias_c_RUBNWN[13] = (__fp16)bias_c_WN[q + 1];
            bias_c_RUBNWN[14] = (__fp16)bias_c_WN[q + 2];
            bias_c_RUBNWN[15] = (__fp16)bias_c_WN[q + 3];

            bias_c_RUBNWN += 16;

            const float* weight_xc_R = weight_xc.row(num_output * 0 + q);
            const float* weight_xc_U = weight_xc.row(num_output * 1 + q);
            const float* weight_xc_N = weight_xc.row(num_output * 2 + q);

            const float* weight_xc_R_1 = weight_xc.row(num_output * 0 + q + 1);
            const float* weight_xc_U_1 = weight_xc.row(num_output * 1 + q + 1);
            const float* weight_xc_N_1 = weight_xc.row(num_output * 2 + q + 1);

            const float* weight_xc_R_2 = weight_xc.row(num_output * 0 + q + 2);
            const float* weight_xc_U_2 = weight_xc.row(num_output * 1 + q + 2);
            const float* weight_xc_N_2 = weight_xc.row(num_output * 2 + q + 2);

            const float* weight_xc_R_3 = weight_xc.row(num_output * 0 + q + 3);
            const float* weight_xc_U_3 = weight_xc.row(num_output * 1 + q + 3);
            const float* weight_xc_N_3 = weight_xc.row(num_output * 2 + q + 3);

            const float* weight_hc_R = weight_hc.row(num_output * 0 + q);
            const float* weight_hc_U = weight_hc.row(num_output * 1 + q);
            const float* weight_hc_N = weight_hc.row(num_output * 2 + q);

            const float* weight_hc_R_1 = weight_hc.row(num_output * 0 + q + 1);
            const float* weight_hc_U_1 = weight_hc.row(num_output * 1 + q + 1);
            const float* weight_hc_N_1 = weight_hc.row(num_output * 2 + q + 1);

            const float* weight_hc_R_2 = weight_hc.row(num_output * 0 + q + 2);
            const float* weight_hc_U_2 = weight_hc.row(num_output * 1 + q + 2);
            const float* weight_hc_N_2 = weight_hc.row(num_output * 2 + q + 2);

            const float* weight_hc_R_3 = weight_hc.row(num_output * 0 + q + 3);
            const float* weight_hc_U_3 = weight_hc.row(num_output * 1 + q + 3);
            const float* weight_hc_N_3 = weight_hc.row(num_output * 2 + q + 3);

            __fp16* weight_xc_RUN = weight_xc_data_packed_dr.row<__fp16>(q / 4);
            __fp16* weight_hc_RUN = weight_hc_data_packed_dr.row<__fp16>(q / 4);

            for (int i = 0; i < size; i++)
            {
                weight_xc_RUN[0] = (__fp16)weight_xc_R[i];
                weight_xc_RUN[1] = (__fp16)weight_xc_R_1[i];
                weight_xc_RUN[2] = (__fp16)weight_xc_R_2[i];
                weight_xc_RUN[3] = (__fp16)weight_xc_R_3[i];
                weight_xc_RUN[4] = (__fp16)weight_xc_U[i];
                weight_xc_RUN[5] = (__fp16)weight_xc_U_1[i];
                weight_xc_RUN[6] = (__fp16)weight_xc_U_2[i];
                weight_xc_RUN[7] = (__fp16)weight_xc_U_3[i];

                weight_xc_RUN += 8;
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc_RUN[0] = (__fp16)weight_hc_R[i];
                weight_hc_RUN[1] = (__fp16)weight_hc_R_1[i];
                weight_hc_RUN[2] = (__fp16)weight_hc_R_2[i];
                weight_hc_RUN[3] = (__fp16)weight_hc_R_3[i];
                weight_hc_RUN[4] = (__fp16)weight_hc_U[i];
                weight_hc_RUN[5] = (__fp16)weight_hc_U_1[i];
                weight_hc_RUN[6] = (__fp16)weight_hc_U_2[i];
                weight_hc_RUN[7] = (__fp16)weight_hc_U_3[i];

                weight_hc_RUN += 8;
            }

            for (int i = 0; i < size; i++)
            {
                weight_xc_RUN[0] = (__fp16)weight_xc_N[i];
                weight_xc_RUN[1] = (__fp16)weight_xc_N_1[i];
                weight_xc_RUN[2] = (__fp16)weight_xc_N_2[i];
                weight_xc_RUN[3] = (__fp16)weight_xc_N_3[i];

                weight_xc_RUN += 4;
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc_RUN[0] = (__fp16)weight_hc_N[i];
                weight_hc_RUN[1] = (__fp16)weight_hc_N_1[i];
                weight_hc_RUN[2] = (__fp16)weight_hc_N_2[i];
                weight_hc_RUN[3] = (__fp16)weight_hc_N_3[i];

                weight_hc_RUN += 4;
            }
        }
        for (; q < num_output; q++)
        {
            bias_c_RUBNWN[0] = (__fp16)bias_c_R[q];
            bias_c_RUBNWN[1] = (__fp16)bias_c_U[q];
            bias_c_RUBNWN[2] = (__fp16)bias_c_BN[q];
            bias_c_RUBNWN[3] = (__fp16)bias_c_WN[q];

            bias_c_RUBNWN += 4;

            const float* weight_xc_R = weight_xc.row(num_output * 0 + q);
            const float* weight_xc_U = weight_xc.row(num_output * 1 + q);
            const float* weight_xc_N = weight_xc.row(num_output * 2 + q);

            const float* weight_hc_R = weight_hc.row(num_output * 0 + q);
            const float* weight_hc_U = weight_hc.row(num_output * 1 + q);
            const float* weight_hc_N = weight_hc.row(num_output * 2 + q);

            __fp16* weight_xc_RUN = weight_xc_data_packed_dr.row<__fp16>(q / 4 + q % 4);
            __fp16* weight_hc_RUN = weight_hc_data_packed_dr.row<__fp16>(q / 4 + q % 4);

            for (int i = 0; i < size; i++)
            {
                weight_xc_RUN[0] = (__fp16)weight_xc_R[i];
                weight_xc_RUN[1] = (__fp16)weight_xc_U[i];

                weight_xc_RUN += 2;
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc_RUN[0] = (__fp16)weight_hc_R[i];
                weight_hc_RUN[1] = (__fp16)weight_hc_U[i];

                weight_hc_RUN += 2;
            }

            for (int i = 0; i < size; i++)
            {
                weight_xc_RUN[0] = (__fp16)weight_xc_N[i];

                weight_xc_RUN += 1;
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc_RUN[0] = (__fp16)weight_hc_N[i];

                weight_hc_RUN += 1;
            }
        }
    }

    return 0;
}

int GRU_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int T = bottom_blob.h;

    int num_directions = direction == 2 ? 2 : 1;

    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    top_blob.create(num_output * num_directions, T, 2u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = gru_fp16s(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        int ret0 = gru_fp16s(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.f);

        int ret1 = gru_fp16s(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden, opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i = 0; i < T; i++)
        {
            const __fp16* pf = top_blob_forward.row<const __fp16>(i);
            const __fp16* pr = top_blob_reverse.row<const __fp16>(i);
            __fp16* ptr = top_blob.row<__fp16>(i);

            memcpy(ptr, pf, num_output * sizeof(__fp16));
            memcpy(ptr + num_output, pr, num_output * sizeof(__fp16));
        }
    }

    return 0;
}

int GRU_arm::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

    Mat hidden;
    Allocator* hidden_allocator = top_blobs.size() == 2 ? opt.blob_allocator : opt.workspace_allocator;
    if (bottom_blobs.size() == 2)
    {
        Option opt_cast = opt;
        opt_cast.blob_allocator = hidden_allocator;
        cast_float16_to_float32(bottom_blobs[1], hidden, opt_cast);
    }
    else
    {
        hidden.create(num_output, num_directions, 4u, hidden_allocator);
        if (hidden.empty())
            return -100;
        hidden.fill(0.f);
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output * num_directions, T, 2u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = gru_fp16s(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        Mat hidden0 = hidden.row_range(0, 1);
        int ret0 = gru_fp16s(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden0, opt);
        if (ret0 != 0)
            return ret0;

        Mat hidden1 = hidden.row_range(1, 1);
        int ret1 = gru_fp16s(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden1, opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i = 0; i < T; i++)
        {
            const __fp16* pf = top_blob_forward.row<const __fp16>(i);
            const __fp16* pr = top_blob_reverse.row<const __fp16>(i);
            __fp16* ptr = top_blob.row<__fp16>(i);

            memcpy(ptr, pf, num_output * sizeof(__fp16));
            memcpy(ptr + num_output, pr, num_output * sizeof(__fp16));
        }
    }

    if (top_blobs.size() == 2)
    {
        cast_float32_to_float16(hidden, top_blobs[1], opt);
    }

    return 0;
}

int GRU_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int T = bottom_blob.h;

    int num_directions = direction == 2 ? 2 : 1;

    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    top_blob.create(num_output * num_directions, T, 2u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = gru_fp16sa(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        int ret0 = gru_fp16sa(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.f);

        int ret1 = gru_fp16sa(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden, opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i = 0; i < T; i++)
        {
            const __fp16* pf = top_blob_forward.row<const __fp16>(i);
            const __fp16* pr = top_blob_reverse.row<const __fp16>(i);
            __fp16* ptr = top_blob.row<__fp16>(i);

            memcpy(ptr, pf, num_output * sizeof(__fp16));
            memcpy(ptr + num_output, pr, num_output * sizeof(__fp16));
        }
    }

    return 0;
}

int GRU_arm::forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

    Mat hidden;
    Allocator* hidden_allocator = top_blobs.size() == 2 ? opt.blob_allocator : opt.workspace_allocator;
    if (bottom_blobs.size() == 2)
    {
        Option opt_cast = opt;
        opt_cast.blob_allocator = hidden_allocator;
        cast_float16_to_float32(bottom_blobs[1], hidden, opt_cast);
    }
    else
    {
        hidden.create(num_output, num_directions, 4u, hidden_allocator);
        if (hidden.empty())
            return -100;
        hidden.fill(0.f);
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output * num_directions, T, 2u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = gru_fp16sa(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        Mat hidden0 = hidden.row_range(0, 1);
        int ret0 = gru_fp16sa(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden0, opt);
        if (ret0 != 0)
            return ret0;

        Mat hidden1 = hidden.row_range(1, 1);
        int ret1 = gru_fp16sa(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden1, opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i = 0; i < T; i++)
        {
            const __fp16* pf = top_blob_forward.row<const __fp16>(i);
            const __fp16* pr = top_blob_reverse.row<const __fp16>(i);
            __fp16* ptr = top_blob.row<__fp16>(i);

            memcpy(ptr, pf, num_output * sizeof(__fp16));
            memcpy(ptr + num_output, pr, num_output * sizeof(__fp16));
        }
    }

    if (top_blobs.size() == 2)
    {
        top_blobs[1] = hidden;
    }

    return 0;
}
#endif

} // namespace ncnn
