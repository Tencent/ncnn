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

#include "lstm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static int lstm_fp16s(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
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

        const __fp16* x = bottom_blob.row<const __fp16>(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < hidden_size; q++)
        {
            const __fp16* bias_c_IFOG = (const __fp16*)bias_c + q * 4;

            // gate I F O G
            const __fp16* weight_xc_IFOG = weight_xc.row<const __fp16>(q);

            const __fp16* weight_hc_IFOG = weight_hc.row<const __fp16>(q);

            float32x4_t _IFOG = vcvt_f32_f16(vld1_f16(bias_c_IFOG));
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _xi = vcvt_f32_f16(vld1_f16(x + i));

                float32x4_t _weight_xc_IFOG_0 = vcvt_f32_f16(vld1_f16(weight_xc_IFOG));
                float32x4_t _weight_xc_IFOG_1 = vcvt_f32_f16(vld1_f16(weight_xc_IFOG + 4));
                float32x4_t _weight_xc_IFOG_2 = vcvt_f32_f16(vld1_f16(weight_xc_IFOG + 8));
                float32x4_t _weight_xc_IFOG_3 = vcvt_f32_f16(vld1_f16(weight_xc_IFOG + 12));

                _IFOG = vfmaq_laneq_f32(_IFOG, _weight_xc_IFOG_0, _xi, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_xc_IFOG_1, _xi, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_xc_IFOG_2, _xi, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_xc_IFOG_3, _xi, 3);

                weight_xc_IFOG += 16;
            }
            for (; i < size; i++)
            {
                __fp16 xi = x[i];

                float32x4_t _xi = vcvt_f32_f16(vdup_n_f16(xi));
                float32x4_t _weight_xc_IFOG = vcvt_f32_f16(vld1_f16(weight_xc_IFOG));
                _IFOG = vfmaq_f32(_IFOG, _weight_xc_IFOG, _xi);

                weight_xc_IFOG += 4;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float32x4_t _h_cont = vld1q_f32((const float*)hidden_state + i);

                float32x4_t _weight_hc_IFOG_0 = vcvt_f32_f16(vld1_f16(weight_hc_IFOG));
                float32x4_t _weight_hc_IFOG_1 = vcvt_f32_f16(vld1_f16(weight_hc_IFOG + 4));
                float32x4_t _weight_hc_IFOG_2 = vcvt_f32_f16(vld1_f16(weight_hc_IFOG + 8));
                float32x4_t _weight_hc_IFOG_3 = vcvt_f32_f16(vld1_f16(weight_hc_IFOG + 12));

                _IFOG = vfmaq_laneq_f32(_IFOG, _weight_hc_IFOG_0, _h_cont, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_hc_IFOG_1, _h_cont, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_hc_IFOG_2, _h_cont, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_hc_IFOG_3, _h_cont, 3);

                weight_hc_IFOG += 16;
            }
            for (; i < num_output; i++)
            {
                float h_cont = hidden_state[i];

                float32x4_t _h_cont = vdupq_n_f32(h_cont);
                float32x4_t _weight_hc_IFOG = vcvt_f32_f16(vld1_f16(weight_hc_IFOG));
                _IFOG = vfmaq_f32(_IFOG, _weight_hc_IFOG, _h_cont);

                weight_hc_IFOG += 4;
            }

            float* gates_data = gates.row(q);

            _IFOG = vaddq_f32(_IFOG, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _IFOG = vaddq_f32(_IFOG, _sum2);

            vst1q_f32(gates_data, _IFOG);
        }

        // lstm unit
        // sigmoid(I)
        // sigmoid(F)
        // sigmoid(O)
        // tanh(G)
        // c_t := f_t .* c_{t-1} + i_t .* g_t
        // h_t := o_t .* tanh[c_t]
        __fp16* output_data = top_blob.row<__fp16>(ti);

        float* cell_ptr = cell_state;
        float* hidden_ptr = hidden_state;
        float* tmp_hidden_ptr = tmp_hidden_state;

        int nn_hidden_size = hidden_size >> 2;
        int remain_hidden_size_start = nn_hidden_size << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = qq * 4;

            const float* gates_data = gates.row(q);

            float32x4x4_t _IFOG_4x4 = vld4q_f32(gates_data);

            float32x4_t _lstm_I = sigmoid_ps(_IFOG_4x4.val[0]);
            float32x4_t _lstm_F = sigmoid_ps(_IFOG_4x4.val[1]);
            float32x4_t _lstm_O = sigmoid_ps(_IFOG_4x4.val[2]);
            float32x4_t _lstm_G = tanh_ps(_IFOG_4x4.val[3]);

            float32x4_t _cell2 = vaddq_f32(vmulq_f32(_lstm_F, vld1q_f32(cell_ptr + q)), vmulq_f32(_lstm_I, _lstm_G));
            float32x4_t _lstm_H = vmulq_f32(_lstm_O, tanh_ps(_cell2));

            vst1q_f32(cell_ptr + q, _cell2);

            if (num_output == hidden_size)
            {
                vst1q_f32(hidden_ptr + q, _lstm_H);
                vst1_f16(output_data + q, vcvt_f16_f32(_lstm_H));
            }
            else
            {
                vst1q_f32(tmp_hidden_ptr + q, _lstm_H);
            }
        }
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
                output_data[q] = (__fp16)H;
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

                hidden_ptr[q] = H;
                output_data[q] = (__fp16)H;
            }
        }
    }

    return 0;
}

static int lstm_fp16sa(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;
    int hidden_size = cell_state.w;

    // 4 x hidden_size
    Mat gates(4, hidden_size, 2u, opt.workspace_allocator);
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

        int nn_hidden_size = hidden_size >> 1;
        int remain_hidden_size_start = nn_hidden_size << 1;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = qq * 2;

            const __fp16* bias_c_IFOG = (const __fp16*)bias_c + q * 4;

            // gate I F O G
            const __fp16* weight_xc_IFOG = weight_xc.row<const __fp16>(q / 2);

            const __fp16* weight_hc_IFOG = weight_hc.row<const __fp16>(q / 2);

            float16x8_t _IFOG = vld1q_f16(bias_c_IFOG);
            float16x8_t _sum1 = vdupq_n_f16((__fp16)0.f);
            float16x8_t _sum2 = vdupq_n_f16((__fp16)0.f);
            float16x8_t _sum3 = vdupq_n_f16((__fp16)0.f);

            const __fp16* x = bottom_blob.row<const __fp16>(ti);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "ld1    {v4.4h}, [%0], #8       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    "fmla   %2.8h, v0.8h, v4.h[0]   \n"
                    "fmla   %3.8h, v1.8h, v4.h[1]   \n"
                    "fmla   %4.8h, v2.8h, v4.h[2]   \n"
                    "fmla   %5.8h, v3.8h, v4.h[3]   \n"
                    : "=r"(x),
                    "=r"(weight_xc_IFOG),
                    "=w"(_IFOG),
                    "=w"(_sum1),
                    "=w"(_sum2),
                    "=w"(_sum3)
                    : "0"(x),
                    "1"(weight_xc_IFOG),
                    "2"(_IFOG),
                    "3"(_sum1),
                    "4"(_sum2),
                    "5"(_sum3)
                    : "memory", "v0", "v1", "v2", "v3", "v4");
#else  // NCNN_GNU_INLINE_ASM
                float16x4_t _x = vld1_f16(x);
                float16x8_t _w0 = vld1q_f16(weight_xc_IFOG);
                float16x8_t _w1 = vld1q_f16(weight_xc_IFOG + 8);
                float16x8_t _w2 = vld1q_f16(weight_xc_IFOG + 16);
                float16x8_t _w3 = vld1q_f16(weight_xc_IFOG + 24);
                _IFOG = vfmaq_lane_f16(_IFOG, _w0, _x, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _w1, _x, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _w2, _x, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _w3, _x, 3);

                x += 4;
                weight_xc_IFOG += 32;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; i < size; i++)
            {
                __fp16 xi = *x++;

                float16x8_t _xi = vdupq_n_f16(xi);
                float16x8_t _weight_xc_IFOG = vld1q_f16(weight_xc_IFOG);
                _IFOG = vfmaq_f16(_IFOG, _weight_xc_IFOG, _xi);

                weight_xc_IFOG += 8;
            }

            const float* hidden_ptr = hidden_state;

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "ld1    {v4.4s}, [%0], #16      \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    "fcvtn  v4.4h, v4.4s            \n"
                    "fmla   %2.8h, v0.8h, v4.h[0]   \n"
                    "fmla   %3.8h, v1.8h, v4.h[1]   \n"
                    "fmla   %4.8h, v2.8h, v4.h[2]   \n"
                    "fmla   %5.8h, v3.8h, v4.h[3]   \n"
                    : "=r"(hidden_ptr),
                    "=r"(weight_hc_IFOG),
                    "=w"(_IFOG),
                    "=w"(_sum1),
                    "=w"(_sum2),
                    "=w"(_sum3)
                    : "0"(hidden_ptr),
                    "1"(weight_hc_IFOG),
                    "2"(_IFOG),
                    "3"(_sum1),
                    "4"(_sum2),
                    "5"(_sum3)
                    : "memory", "v0", "v1", "v2", "v3", "v4");
#else  // NCNN_GNU_INLINE_ASM
                float16x4_t _h_cont = vcvt_f16_f32(vld1q_f32(hidden_ptr));
                float16x8_t _w0 = vld1q_f16(weight_hc_IFOG);
                float16x8_t _w1 = vld1q_f16(weight_hc_IFOG + 8);
                float16x8_t _w2 = vld1q_f16(weight_hc_IFOG + 16);
                float16x8_t _w3 = vld1q_f16(weight_hc_IFOG + 24);
                _IFOG = vfmaq_lane_f16(_IFOG, _w0, _h_cont, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _w1, _h_cont, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _w2, _h_cont, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _w3, _h_cont, 3);

                hidden_ptr += 4;
                weight_hc_IFOG += 32;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; i < num_output; i++)
            {
                float h_cont = *hidden_ptr++;

                float16x8_t _h_cont = vdupq_n_f16((__fp16)h_cont);
                float16x8_t _weight_hc_IFOG = vld1q_f16(weight_hc_IFOG);
                _IFOG = vfmaq_f16(_IFOG, _weight_hc_IFOG, _h_cont);

                weight_hc_IFOG += 8;
            }

            __fp16* gates_data = gates.row<__fp16>(q);

            _IFOG = vaddq_f16(_IFOG, _sum1);
            _sum2 = vaddq_f16(_sum2, _sum3);
            _IFOG = vaddq_f16(_IFOG, _sum2);

            vst1q_f16(gates_data, _IFOG);
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_hidden_size_start; q < hidden_size; q++)
        {
            const __fp16* bias_c_IFOG = (const __fp16*)bias_c + q * 4;

            // gate I F O G
            const __fp16* weight_xc_IFOG = weight_xc.row<const __fp16>(q / 2 + q % 2);

            const __fp16* weight_hc_IFOG = weight_hc.row<const __fp16>(q / 2 + q % 2);

            float16x4_t _IFOG = vld1_f16(bias_c_IFOG);
            float16x4_t _sum1 = vdup_n_f16((__fp16)0.f);
            float16x4_t _sum2 = vdup_n_f16((__fp16)0.f);
            float16x4_t _sum3 = vdup_n_f16((__fp16)0.f);

            const __fp16* x = bottom_blob.row<const __fp16>(ti);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "ld1    {v4.4h}, [%0], #8       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n"
                    "fmla   %2.4h, v0.4h, v4.h[0]   \n"
                    "fmla   %3.4h, v1.4h, v4.h[1]   \n"
                    "fmla   %4.4h, v2.4h, v4.h[2]   \n"
                    "fmla   %5.4h, v3.4h, v4.h[3]   \n"
                    : "=r"(x),
                    "=r"(weight_xc_IFOG),
                    "=w"(_IFOG),
                    "=w"(_sum1),
                    "=w"(_sum2),
                    "=w"(_sum3)
                    : "0"(x),
                    "1"(weight_xc_IFOG),
                    "2"(_IFOG),
                    "3"(_sum1),
                    "4"(_sum2),
                    "5"(_sum3)
                    : "memory", "v0", "v1", "v2", "v3", "v4");
#else  // NCNN_GNU_INLINE_ASM
                float16x4_t _x = vld1_f16(x);
                float16x4_t _w0 = vld1_f16(weight_xc_IFOG);
                float16x4_t _w1 = vld1_f16(weight_xc_IFOG + 4);
                float16x4_t _w2 = vld1_f16(weight_xc_IFOG + 8);
                float16x4_t _w3 = vld1_f16(weight_xc_IFOG + 12);
                _IFOG = vfma_lane_f16(_IFOG, _w0, _x, 0);
                _sum1 = vfma_lane_f16(_sum1, _w1, _x, 1);
                _sum2 = vfma_lane_f16(_sum2, _w2, _x, 2);
                _sum3 = vfma_lane_f16(_sum3, _w3, _x, 3);

                x += 4;
                weight_xc_IFOG += 16;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; i < size; i++)
            {
                __fp16 xi = *x++;

                float16x4_t _xi = vdup_n_f16(xi);
                float16x4_t _weight_xc_IFOG = vld1_f16(weight_xc_IFOG);
                _IFOG = vfma_f16(_IFOG, _weight_xc_IFOG, _xi);

                weight_xc_IFOG += 4;
            }

            const float* hidden_ptr = hidden_state;

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "ld1    {v4.4s}, [%0], #16      \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n"
                    "fcvtn  v4.4h, v4.4s            \n"
                    "fmla   %2.4h, v0.4h, v4.h[0]   \n"
                    "fmla   %3.4h, v1.4h, v4.h[1]   \n"
                    "fmla   %4.4h, v2.4h, v4.h[2]   \n"
                    "fmla   %5.4h, v3.4h, v4.h[3]   \n"
                    : "=r"(hidden_ptr),
                    "=r"(weight_hc_IFOG),
                    "=w"(_IFOG),
                    "=w"(_sum1),
                    "=w"(_sum2),
                    "=w"(_sum3)
                    : "0"(hidden_ptr),
                    "1"(weight_hc_IFOG),
                    "2"(_IFOG),
                    "3"(_sum1),
                    "4"(_sum2),
                    "5"(_sum3)
                    : "memory", "v0", "v1", "v2", "v3", "v4");
#else  // NCNN_GNU_INLINE_ASM
                float16x4_t _h_cont = vcvt_f16_f32(vld1q_f32(hidden_ptr));
                float16x4_t _w0 = vld1_f16(weight_hc_IFOG);
                float16x4_t _w1 = vld1_f16(weight_hc_IFOG + 4);
                float16x4_t _w2 = vld1_f16(weight_hc_IFOG + 8);
                float16x4_t _w3 = vld1_f16(weight_hc_IFOG + 12);
                _IFOG = vfma_lane_f16(_IFOG, _w0, _h_cont, 0);
                _sum1 = vfma_lane_f16(_sum1, _w1, _h_cont, 1);
                _sum2 = vfma_lane_f16(_sum2, _w2, _h_cont, 2);
                _sum3 = vfma_lane_f16(_sum3, _w3, _h_cont, 3);

                hidden_ptr += 4;
                weight_hc_IFOG += 16;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; i < num_output; i++)
            {
                float h_cont = *hidden_ptr++;

                float16x4_t _h_cont = vdup_n_f16((__fp16)h_cont);
                float16x4_t _weight_hc_IFOG = vld1_f16(weight_hc_IFOG);
                _IFOG = vfma_f16(_IFOG, _weight_hc_IFOG, _h_cont);

                weight_hc_IFOG += 4;
            }

            __fp16* gates_data = gates.row<__fp16>(q);

            _IFOG = vadd_f16(_IFOG, _sum1);
            _sum2 = vadd_f16(_sum2, _sum3);
            _IFOG = vadd_f16(_IFOG, _sum2);

            vst1_f16(gates_data, _IFOG);
        }

        // lstm unit
        // sigmoid(I)
        // sigmoid(F)
        // sigmoid(O)
        // tanh(G)
        // c_t := f_t .* c_{t-1} + i_t .* g_t
        // h_t := o_t .* tanh[c_t]
        __fp16* output_data = top_blob.row<__fp16>(ti);

        float* cell_ptr = cell_state;
        float* hidden_ptr = hidden_state;
        float* tmp_hidden_ptr = tmp_hidden_state;

        nn_hidden_size = hidden_size >> 2;
        remain_hidden_size_start = nn_hidden_size << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = qq * 4;

            const __fp16* gates_data = gates.row<const __fp16>(q);

            float16x4x4_t _IFOG_4x4 = vld4_f16(gates_data);

            float32x4_t _lstm_I = sigmoid_ps(vcvt_f32_f16(_IFOG_4x4.val[0]));
            float32x4_t _lstm_F = sigmoid_ps(vcvt_f32_f16(_IFOG_4x4.val[1]));
            float32x4_t _lstm_O = sigmoid_ps(vcvt_f32_f16(_IFOG_4x4.val[2]));
            float32x4_t _lstm_G = tanh_ps(vcvt_f32_f16(_IFOG_4x4.val[3]));

            float32x4_t _cell2 = vaddq_f32(vmulq_f32(_lstm_F, vld1q_f32(cell_ptr + q)), vmulq_f32(_lstm_I, _lstm_G));
            float32x4_t _lstm_H = vmulq_f32(_lstm_O, tanh_ps(_cell2));

            vst1q_f32(cell_ptr + q, _cell2);

            if (num_output == hidden_size)
            {
                vst1q_f32(hidden_ptr + q, _lstm_H);
                vst1_f16(output_data + q, vcvt_f16_f32(_lstm_H));
            }
            else
            {
                vst1q_f32(tmp_hidden_ptr + q, _lstm_H);
            }
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_hidden_size_start; q < hidden_size; q++)
        {
            const __fp16* gates_data = gates.row<const __fp16>(q);

            float I = (float)gates_data[0];
            float F = (float)gates_data[1];
            float O = (float)gates_data[2];
            float G = (float)gates_data[3];

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
                output_data[q] = (__fp16)H;
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

                hidden_ptr[q] = H;
                output_data[q] = (__fp16)H;
            }
        }
    }

    return 0;
}

int LSTM_arm::create_pipeline_fp16s(const Option& opt)
{
    // pack IFOG
    int num_directions = direction == 2 ? 2 : 1;
    int size = weight_data_size / num_directions / hidden_size / 4;

    if (opt.use_fp16_arithmetic)
    {
        weight_xc_data_packed.create(size, hidden_size / 2 + hidden_size % 2, num_directions, 16u, 8);
        bias_c_data_packed.create(hidden_size, 1, num_directions, 8u, 4);
        weight_hc_data_packed.create(num_output, hidden_size / 2 + hidden_size % 2, num_directions, 16u, 8);
    }
    else
    {
        weight_xc_data_packed.create(size, hidden_size, num_directions, 8u, 4);
        bias_c_data_packed.create(hidden_size, 1, num_directions, 8u, 4);
        weight_hc_data_packed.create(num_output, hidden_size, num_directions, 8u, 4);
    }

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

        __fp16* bias_c_IFOG = bias_c_data_packed_dr.row<__fp16>(0);

        if (opt.use_fp16_arithmetic)
        {
            int q = 0;
            for (; q + 1 < hidden_size; q += 2)
            {
                bias_c_IFOG[0] = (__fp16)bias_c_I[q];
                bias_c_IFOG[1] = (__fp16)bias_c_F[q];
                bias_c_IFOG[2] = (__fp16)bias_c_O[q];
                bias_c_IFOG[3] = (__fp16)bias_c_G[q];
                bias_c_IFOG[4] = (__fp16)bias_c_I[q + 1];
                bias_c_IFOG[5] = (__fp16)bias_c_F[q + 1];
                bias_c_IFOG[6] = (__fp16)bias_c_O[q + 1];
                bias_c_IFOG[7] = (__fp16)bias_c_G[q + 1];

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

                __fp16* weight_xc_IFOG = weight_xc_data_packed_dr.row<__fp16>(q / 2);
                __fp16* weight_hc_IFOG = weight_hc_data_packed_dr.row<__fp16>(q / 2);

                for (int i = 0; i < size; i++)
                {
                    weight_xc_IFOG[0] = (__fp16)weight_xc_I[i];
                    weight_xc_IFOG[1] = (__fp16)weight_xc_F[i];
                    weight_xc_IFOG[2] = (__fp16)weight_xc_O[i];
                    weight_xc_IFOG[3] = (__fp16)weight_xc_G[i];
                    weight_xc_IFOG[4] = (__fp16)weight_xc_I_1[i];
                    weight_xc_IFOG[5] = (__fp16)weight_xc_F_1[i];
                    weight_xc_IFOG[6] = (__fp16)weight_xc_O_1[i];
                    weight_xc_IFOG[7] = (__fp16)weight_xc_G_1[i];

                    weight_xc_IFOG += 8;
                }

                for (int i = 0; i < num_output; i++)
                {
                    weight_hc_IFOG[0] = (__fp16)weight_hc_I[i];
                    weight_hc_IFOG[1] = (__fp16)weight_hc_F[i];
                    weight_hc_IFOG[2] = (__fp16)weight_hc_O[i];
                    weight_hc_IFOG[3] = (__fp16)weight_hc_G[i];
                    weight_hc_IFOG[4] = (__fp16)weight_hc_I_1[i];
                    weight_hc_IFOG[5] = (__fp16)weight_hc_F_1[i];
                    weight_hc_IFOG[6] = (__fp16)weight_hc_O_1[i];
                    weight_hc_IFOG[7] = (__fp16)weight_hc_G_1[i];

                    weight_hc_IFOG += 8;
                }
            }
            for (; q < hidden_size; q++)
            {
                bias_c_IFOG[0] = (__fp16)bias_c_I[q];
                bias_c_IFOG[1] = (__fp16)bias_c_F[q];
                bias_c_IFOG[2] = (__fp16)bias_c_O[q];
                bias_c_IFOG[3] = (__fp16)bias_c_G[q];

                bias_c_IFOG += 4;

                const float* weight_xc_I = weight_xc.row(hidden_size * 0 + q);
                const float* weight_xc_F = weight_xc.row(hidden_size * 1 + q);
                const float* weight_xc_O = weight_xc.row(hidden_size * 2 + q);
                const float* weight_xc_G = weight_xc.row(hidden_size * 3 + q);

                const float* weight_hc_I = weight_hc.row(hidden_size * 0 + q);
                const float* weight_hc_F = weight_hc.row(hidden_size * 1 + q);
                const float* weight_hc_O = weight_hc.row(hidden_size * 2 + q);
                const float* weight_hc_G = weight_hc.row(hidden_size * 3 + q);

                __fp16* weight_xc_IFOG = weight_xc_data_packed_dr.row<__fp16>(q / 2 + q % 2);
                __fp16* weight_hc_IFOG = weight_hc_data_packed_dr.row<__fp16>(q / 2 + q % 2);

                for (int i = 0; i < size; i++)
                {
                    weight_xc_IFOG[0] = (__fp16)weight_xc_I[i];
                    weight_xc_IFOG[1] = (__fp16)weight_xc_F[i];
                    weight_xc_IFOG[2] = (__fp16)weight_xc_O[i];
                    weight_xc_IFOG[3] = (__fp16)weight_xc_G[i];

                    weight_xc_IFOG += 4;
                }

                for (int i = 0; i < num_output; i++)
                {
                    weight_hc_IFOG[0] = (__fp16)weight_hc_I[i];
                    weight_hc_IFOG[1] = (__fp16)weight_hc_F[i];
                    weight_hc_IFOG[2] = (__fp16)weight_hc_O[i];
                    weight_hc_IFOG[3] = (__fp16)weight_hc_G[i];

                    weight_hc_IFOG += 4;
                }
            }
        }
        else
        {
            for (int q = 0; q < hidden_size; q++)
            {
                bias_c_IFOG[0] = (__fp16)bias_c_I[q];
                bias_c_IFOG[1] = (__fp16)bias_c_F[q];
                bias_c_IFOG[2] = (__fp16)bias_c_O[q];
                bias_c_IFOG[3] = (__fp16)bias_c_G[q];

                bias_c_IFOG += 4;

                const float* weight_xc_I = weight_xc.row(hidden_size * 0 + q);
                const float* weight_xc_F = weight_xc.row(hidden_size * 1 + q);
                const float* weight_xc_O = weight_xc.row(hidden_size * 2 + q);
                const float* weight_xc_G = weight_xc.row(hidden_size * 3 + q);

                const float* weight_hc_I = weight_hc.row(hidden_size * 0 + q);
                const float* weight_hc_F = weight_hc.row(hidden_size * 1 + q);
                const float* weight_hc_O = weight_hc.row(hidden_size * 2 + q);
                const float* weight_hc_G = weight_hc.row(hidden_size * 3 + q);

                __fp16* weight_xc_IFOG = weight_xc_data_packed_dr.row<__fp16>(q);
                __fp16* weight_hc_IFOG = weight_hc_data_packed_dr.row<__fp16>(q);

                for (int i = 0; i < size; i++)
                {
                    weight_xc_IFOG[0] = (__fp16)weight_xc_I[i];
                    weight_xc_IFOG[1] = (__fp16)weight_xc_F[i];
                    weight_xc_IFOG[2] = (__fp16)weight_xc_O[i];
                    weight_xc_IFOG[3] = (__fp16)weight_xc_G[i];

                    weight_xc_IFOG += 4;
                }

                for (int i = 0; i < num_output; i++)
                {
                    weight_hc_IFOG[0] = (__fp16)weight_hc_I[i];
                    weight_hc_IFOG[1] = (__fp16)weight_hc_F[i];
                    weight_hc_IFOG[2] = (__fp16)weight_hc_O[i];
                    weight_hc_IFOG[3] = (__fp16)weight_hc_G[i];

                    weight_hc_IFOG += 4;
                }
            }
        }
    }

    weight_xc_data.release();
    bias_c_data.release();
    weight_hc_data.release();

    return 0;
}

int LSTM_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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

    top_blob.create(num_output * num_directions, T, 2u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = lstm_fp16s(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
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

        int ret0 = lstm_fp16s(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.f);
        cell.fill(0.f);

        int ret1 = lstm_fp16s(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1), hidden, cell, opt);
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

int LSTM_arm::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

    Mat hidden;
    Mat cell;
    Allocator* hidden_cell_allocator = top_blobs.size() == 3 ? opt.blob_allocator : opt.workspace_allocator;
    if (bottom_blobs.size() == 3)
    {
        Option opt_cast = opt;
        opt_cast.blob_allocator = hidden_cell_allocator;
        cast_float16_to_float32(bottom_blobs[1], hidden, opt_cast);
        cast_float16_to_float32(bottom_blobs[2], cell, opt_cast);
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
    top_blob.create(num_output * num_directions, T, 2u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = lstm_fp16s(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
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
        Mat cell0 = cell.row_range(0, 1);
        int ret0 = lstm_fp16s(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden0, cell0, opt);
        if (ret0 != 0)
            return ret0;

        Mat hidden1 = hidden.row_range(1, 1);
        Mat cell1 = cell.row_range(1, 1);
        int ret1 = lstm_fp16s(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1), hidden1, cell1, opt);
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

    if (top_blobs.size() == 3)
    {
        cast_float32_to_float16(hidden, top_blobs[1], opt);
        cast_float32_to_float16(cell, top_blobs[2], opt);
    }

    return 0;
}

int LSTM_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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

    top_blob.create(num_output * num_directions, T, 2u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = lstm_fp16sa(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
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

        int ret0 = lstm_fp16sa(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.f);
        cell.fill(0.f);

        int ret1 = lstm_fp16sa(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1), hidden, cell, opt);
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

int LSTM_arm::forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

    Mat hidden;
    Mat cell;
    Allocator* hidden_cell_allocator = top_blobs.size() == 3 ? opt.blob_allocator : opt.workspace_allocator;
    if (bottom_blobs.size() == 3)
    {
        Option opt_cast = opt;
        opt_cast.blob_allocator = hidden_cell_allocator;
        cast_float16_to_float32(bottom_blobs[1], hidden, opt_cast);
        cast_float16_to_float32(bottom_blobs[2], cell, opt_cast);
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
    top_blob.create(num_output * num_directions, T, 2u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = lstm_fp16sa(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
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
        Mat cell0 = cell.row_range(0, 1);
        int ret0 = lstm_fp16sa(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden0, cell0, opt);
        if (ret0 != 0)
            return ret0;

        Mat hidden1 = hidden.row_range(1, 1);
        Mat cell1 = cell.row_range(1, 1);
        int ret1 = lstm_fp16sa(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1), hidden1, cell1, opt);
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

    if (top_blobs.size() == 3)
    {
        cast_float32_to_float16(hidden, top_blobs[1], opt);
        cast_float32_to_float16(cell, top_blobs[2], opt);
    }

    return 0;
}
#endif

} // namespace ncnn
