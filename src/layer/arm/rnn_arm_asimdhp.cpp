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

#include "rnn_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static int rnn_fp16sa(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // num_output
    Mat gates(num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        const __fp16* x = bottom_blob.row<const __fp16>(ti);

        int nn_num_output = num_output >> 3;
        int remain_num_output_start = nn_num_output << 3;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 8;

            const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 8);
            const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 8);

            float16x8_t _rnn_H = vld1q_f16((const __fp16*)bias_c + q);
            float16x8_t _sum1 = vdupq_n_f16(0.f);
            float16x8_t _sum2 = vdupq_n_f16(0.f);
            float16x8_t _sum3 = vdupq_n_f16(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _x = vld1_f16(x + i);
                float16x8_t _weight_xc = vld1q_f16(weight_xc_ptr);
                float16x8_t _weight_xc_1 = vld1q_f16(weight_xc_ptr + 8);
                float16x8_t _weight_xc_2 = vld1q_f16(weight_xc_ptr + 16);
                float16x8_t _weight_xc_3 = vld1q_f16(weight_xc_ptr + 24);
                _rnn_H = vfmaq_lane_f16(_rnn_H, _weight_xc, _x, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _weight_xc_3, _x, 3);

                weight_xc_ptr += 32;
            }
            for (; i < size; i++)
            {
                float16x8_t _x = vdupq_n_f16(x[i]);
                float16x8_t _weight_xc = vld1q_f16(weight_xc_ptr);
                _rnn_H = vfmaq_f16(_rnn_H, _weight_xc, _x);

                weight_xc_ptr += 8;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float16x4_t _hidden_state = vcvt_f16_f32(vld1q_f32((const float*)hidden_state + i));
                float16x8_t _weight_hc = vld1q_f16(weight_hc_ptr);
                float16x8_t _weight_hc_1 = vld1q_f16(weight_hc_ptr + 8);
                float16x8_t _weight_hc_2 = vld1q_f16(weight_hc_ptr + 16);
                float16x8_t _weight_hc_3 = vld1q_f16(weight_hc_ptr + 24);
                _rnn_H = vfmaq_lane_f16(_rnn_H, _weight_hc, _hidden_state, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _weight_hc_3, _hidden_state, 3);

                weight_hc_ptr += 32;
            }
            for (; i < num_output; i++)
            {
                float16x8_t _hidden_state = vdupq_n_f16((__fp16)hidden_state[i]);
                float16x8_t _weight_hc = vld1q_f16(weight_hc_ptr);
                _rnn_H = vfmaq_f16(_rnn_H, _weight_hc, _hidden_state);

                weight_hc_ptr += 8;
            }

            _rnn_H = vaddq_f16(_rnn_H, _sum1);
            _sum2 = vaddq_f16(_sum2, _sum3);
            _rnn_H = vaddq_f16(_rnn_H, _sum2);

            float32x4_t _H32low = tanh_ps(vcvt_f32_f16(vget_low_f16(_rnn_H)));
            float32x4_t _H32high = tanh_ps(vcvt_f32_f16(vget_high_f16(_rnn_H)));

            vst1q_f32((float*)gates + q, _H32low);
            vst1q_f32((float*)gates + q + 4, _H32high);
        }
        nn_num_output = (num_output - remain_num_output_start) >> 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = remain_num_output_start + qq * 4;

            const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 8 + (q % 8) / 4);
            const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 8 + (q % 8) / 4);

            float16x4_t _rnn_H = vld1_f16((const __fp16*)bias_c + q);
            float16x4_t _sum1 = vdup_n_f16(0.f);
            float16x4_t _sum2 = vdup_n_f16(0.f);
            float16x4_t _sum3 = vdup_n_f16(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _x = vld1_f16(x + i);
                float16x4_t _weight_xc = vld1_f16(weight_xc_ptr);
                float16x4_t _weight_xc_1 = vld1_f16(weight_xc_ptr + 4);
                float16x4_t _weight_xc_2 = vld1_f16(weight_xc_ptr + 8);
                float16x4_t _weight_xc_3 = vld1_f16(weight_xc_ptr + 12);
                _rnn_H = vfma_lane_f16(_rnn_H, _weight_xc, _x, 0);
                _sum1 = vfma_lane_f16(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfma_lane_f16(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfma_lane_f16(_sum3, _weight_xc_3, _x, 3);

                weight_xc_ptr += 16;
            }
            for (; i < size; i++)
            {
                float16x4_t _x = vdup_n_f16(x[i]);
                float16x4_t _weight_xc = vld1_f16(weight_xc_ptr);
                _rnn_H = vfma_f16(_rnn_H, _weight_xc, _x);

                weight_xc_ptr += 4;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float16x4_t _hidden_state = vcvt_f16_f32(vld1q_f32((const float*)hidden_state + i));
                float16x4_t _weight_hc = vld1_f16(weight_hc_ptr);
                float16x4_t _weight_hc_1 = vld1_f16(weight_hc_ptr + 4);
                float16x4_t _weight_hc_2 = vld1_f16(weight_hc_ptr + 8);
                float16x4_t _weight_hc_3 = vld1_f16(weight_hc_ptr + 12);
                _rnn_H = vfma_lane_f16(_rnn_H, _weight_hc, _hidden_state, 0);
                _sum1 = vfma_lane_f16(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfma_lane_f16(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfma_lane_f16(_sum3, _weight_hc_3, _hidden_state, 3);

                weight_hc_ptr += 16;
            }
            for (; i < num_output; i++)
            {
                float16x4_t _hidden_state = vdup_n_f16((__fp16)hidden_state[i]);
                float16x4_t _weight_hc = vld1_f16(weight_hc_ptr);
                _rnn_H = vfma_f16(_rnn_H, _weight_hc, _hidden_state);

                weight_hc_ptr += 4;
            }

            _rnn_H = vadd_f16(_rnn_H, _sum1);
            _sum2 = vadd_f16(_sum2, _sum3);
            _rnn_H = vadd_f16(_rnn_H, _sum2);

            float32x4_t _H32 = tanh_ps(vcvt_f32_f16(_rnn_H));

            vst1q_f32((float*)gates + q, _H32);
        }
        remain_num_output_start += nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 8 + (q % 8) / 4 + q % 4);
            const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 8 + (q % 8) / 4 + q % 4);

            __fp16 H = ((const __fp16*)bias_c)[q];

            for (int i = 0; i < size; i++)
            {
                H += weight_xc_ptr[i] * x[i];
            }

            for (int i = 0; i < num_output; i++)
            {
                H += weight_hc_ptr[i] * (__fp16)hidden_state[i];
            }

            float H32 = tanhf((float)H);

            gates[q] = H32;
        }

        __fp16* output_data = top_blob.row<__fp16>(ti);

        float* hidden_ptr = hidden_state;

        nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            float32x4_t _rnn_H = vld1q_f32((float*)gates + q);

            vst1q_f32(hidden_ptr + q, _rnn_H);
            vst1_f16(output_data + q, vcvt_f16_f32(_rnn_H));
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            float H = gates[q];

            hidden_ptr[q] = H;
            output_data[q] = (__fp16)H;
        }
    }

    return 0;
}

static int rnn_fp16s(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
{
    if (opt.use_fp16_arithmetic)
        return rnn_fp16sa(bottom_blob, top_blob, reverse, weight_xc, bias_c, weight_hc, hidden_state, opt);

    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // num_output
    Mat gates(num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        const __fp16* x = bottom_blob.row<const __fp16>(ti);

        int nn_num_output = num_output >> 2;
        int remain_num_output_start = nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 4);
            const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 4);

            float32x4_t _rnn_H = vcvt_f32_f16(vld1_f16((const __fp16*)bias_c + q));
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _x = vcvt_f32_f16(vld1_f16(x + i));
                float32x4_t _weight_xc = vcvt_f32_f16(vld1_f16(weight_xc_ptr));
                float32x4_t _weight_xc_1 = vcvt_f32_f16(vld1_f16(weight_xc_ptr + 4));
                float32x4_t _weight_xc_2 = vcvt_f32_f16(vld1_f16(weight_xc_ptr + 8));
                float32x4_t _weight_xc_3 = vcvt_f32_f16(vld1_f16(weight_xc_ptr + 12));
                _rnn_H = vfmaq_laneq_f32(_rnn_H, _weight_xc, _x, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_xc_3, _x, 3);

                weight_xc_ptr += 16;
            }
            for (; i < size; i++)
            {
                float32x4_t _x = vcvt_f32_f16(vdup_n_f16(x[i]));
                float32x4_t _weight_xc = vcvt_f32_f16(vld1_f16(weight_xc_ptr));
                _rnn_H = vfmaq_f32(_rnn_H, _weight_xc, _x);

                weight_xc_ptr += 4;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float32x4_t _hidden_state = vld1q_f32((const float*)hidden_state + i);
                float32x4_t _weight_hc = vcvt_f32_f16(vld1_f16(weight_hc_ptr));
                float32x4_t _weight_hc_1 = vcvt_f32_f16(vld1_f16(weight_hc_ptr + 4));
                float32x4_t _weight_hc_2 = vcvt_f32_f16(vld1_f16(weight_hc_ptr + 8));
                float32x4_t _weight_hc_3 = vcvt_f32_f16(vld1_f16(weight_hc_ptr + 12));
                _rnn_H = vfmaq_laneq_f32(_rnn_H, _weight_hc, _hidden_state, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_hc_3, _hidden_state, 3);

                weight_hc_ptr += 16;
            }
            for (; i < num_output; i++)
            {
                float32x4_t _hidden_state = vdupq_n_f32(hidden_state[i]);
                float32x4_t _weight_hc = vcvt_f32_f16(vld1_f16(weight_hc_ptr));
                _rnn_H = vfmaq_f32(_rnn_H, _weight_hc, _hidden_state);

                weight_hc_ptr += 4;
            }

            _rnn_H = vaddq_f32(_rnn_H, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _rnn_H = vaddq_f32(_rnn_H, _sum2);

            _rnn_H = tanh_ps(_rnn_H);

            vst1q_f32((float*)gates + q, _rnn_H);
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 4 + q % 4);
            const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 4 + q % 4);

            float H = (float)(((const __fp16*)bias_c)[q]);

            for (int i = 0; i < size; i++)
            {
                H += (float)weight_xc_ptr[i] * (float)x[i];
            }

            for (int i = 0; i < num_output; i++)
            {
                H += (float)weight_hc_ptr[i] * hidden_state[i];
            }

            H = tanhf(H);

            gates[q] = H;
        }

        __fp16* output_data = top_blob.row<__fp16>(ti);

        float* hidden_ptr = hidden_state;

        nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            float32x4_t _rnn_H = vld1q_f32((float*)gates + q);

            vst1q_f32(hidden_ptr + q, _rnn_H);
            vst1_f16(output_data + q, vcvt_f16_f32(_rnn_H));
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            float H = gates[q];

            hidden_ptr[q] = H;
            output_data[q] = (__fp16)H;
        }
    }

    return 0;
}

#if NCNN_INT8
static int rnn_fp16sa_int8(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc_int8, const Mat& weight_xc_int8_descales, const Mat& bias_c, const Mat& weight_hc_int8, const Mat& weight_hc_int8_descales, Mat& hidden_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // num_output
    Mat gates(num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        const __fp16* x = bottom_blob.row<const __fp16>(ti);

        int nn_num_output = num_output >> 3;
        int remain_num_output_start = nn_num_output << 3;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 8;

            // const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 8);
            // const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 8);

            const signed char* weight_xc_int8_ptr = weight_xc_int8.row<const signed char>(q / 8);
            const signed char* weight_hc_int8_ptr = weight_hc_int8.row<const signed char>(q / 8);

            const __fp16* weight_xc_int8_descales_ptr = weight_xc_int8_descales.row<const __fp16>(q / 8);
            const __fp16* weight_hc_int8_descales_ptr = weight_hc_int8_descales.row<const __fp16>(q / 8);

            float16x8_t _descale_xc = vld1q_f16(weight_xc_int8_descales_ptr);
            float16x8_t _descale_hc = vld1q_f16(weight_hc_int8_descales_ptr);

            float16x8_t _rnn_H = vld1q_f16((const __fp16*)bias_c + q);
            float16x8_t _sum1 = vdupq_n_f16(0.f);
            float16x8_t _sum2 = vdupq_n_f16(0.f);
            float16x8_t _sum3 = vdupq_n_f16(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _x = vld1_f16(x + i);

                int8x16_t _weight_xc_01 = vld1q_s8(weight_xc_int8_ptr);
                int8x16_t _weight_xc_23 = vld1q_s8(weight_xc_int8_ptr + 16);

                float16x8_t _weight_xc_0 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(_weight_xc_01))), _descale_xc);
                float16x8_t _weight_xc_1 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(_weight_xc_01))), _descale_xc);
                float16x8_t _weight_xc_2 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(_weight_xc_23))), _descale_xc);
                float16x8_t _weight_xc_3 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(_weight_xc_23))), _descale_xc);

                _rnn_H = vfmaq_lane_f16(_rnn_H, _weight_xc_0, _x, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _weight_xc_3, _x, 3);

                weight_xc_int8_ptr += 32;
            }
            for (; i < size; i++)
            {
                float16x8_t _x = vdupq_n_f16(x[i]);
                float16x8_t _weight_xc = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vld1_s8(weight_xc_int8_ptr))), _descale_xc);
                _rnn_H = vfmaq_f16(_rnn_H, _weight_xc, _x);

                weight_xc_int8_ptr += 8;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float16x4_t _hidden_state = vcvt_f16_f32(vld1q_f32((const float*)hidden_state + i));

                int8x16_t _weight_hc_01 = vld1q_s8(weight_hc_int8_ptr);
                int8x16_t _weight_hc_23 = vld1q_s8(weight_hc_int8_ptr + 16);

                float16x8_t _weight_hc_0 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(_weight_hc_01))), _descale_hc);
                float16x8_t _weight_hc_1 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(_weight_hc_01))), _descale_hc);
                float16x8_t _weight_hc_2 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(_weight_hc_23))), _descale_hc);
                float16x8_t _weight_hc_3 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(_weight_hc_23))), _descale_hc);

                _rnn_H = vfmaq_lane_f16(_rnn_H, _weight_hc_0, _hidden_state, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _weight_hc_3, _hidden_state, 3);

                weight_hc_int8_ptr += 32;
            }
            for (; i < num_output; i++)
            {
                float16x8_t _hidden_state = vdupq_n_f16((__fp16)hidden_state[i]);
                float16x8_t _weight_hc = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vld1_s8(weight_hc_int8_ptr))), _descale_hc);
                _rnn_H = vfmaq_f16(_rnn_H, _weight_hc, _hidden_state);

                weight_hc_int8_ptr += 8;
            }

            _rnn_H = vaddq_f16(_rnn_H, _sum1);
            _sum2 = vaddq_f16(_sum2, _sum3);
            _rnn_H = vaddq_f16(_rnn_H, _sum2);

            float32x4_t _H32low = tanh_ps(vcvt_f32_f16(vget_low_f16(_rnn_H)));
            float32x4_t _H32high = tanh_ps(vcvt_f32_f16(vget_high_f16(_rnn_H)));

            vst1q_f32((float*)gates + q, _H32low);
            vst1q_f32((float*)gates + q + 4, _H32high);
        }
        nn_num_output = (num_output - remain_num_output_start) >> 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = remain_num_output_start + qq * 4;

            const signed char* weight_xc_int8_ptr = weight_xc_int8.row<const signed char>(q / 8 + (q % 8) / 4);
            const signed char* weight_hc_int8_ptr = weight_hc_int8.row<const signed char>(q / 8 + (q % 8) / 4);
            const __fp16* weight_xc_int8_descales_ptr = weight_xc_int8_descales.row<const __fp16>(q / 8 + (q % 8) / 4);
            const __fp16* weight_hc_int8_descales_ptr = weight_hc_int8_descales.row<const __fp16>(q / 8 + (q % 8) / 4);

            float16x4_t _descale_xc = vld1_f16(weight_xc_int8_descales_ptr);
            float16x4_t _descale_hc = vld1_f16(weight_hc_int8_descales_ptr);
            float16x8_t _descale_xc_2 = vcombine_f16(_descale_xc, _descale_xc);
            float16x8_t _descale_hc_2 = vcombine_f16(_descale_hc, _descale_hc);

            float16x4_t _rnn_H = vld1_f16((const __fp16*)bias_c + q);
            float16x4_t _sum1 = vdup_n_f16(0.f);
            float16x4_t _sum2 = vdup_n_f16(0.f);
            float16x4_t _sum3 = vdup_n_f16(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _x = vld1_f16(x + i);

                int8x16_t _weight_xc = vld1q_s8(weight_xc_int8_ptr);
                float16x8_t _weight_xc_01 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(_weight_xc))), _descale_xc_2);
                float16x8_t _weight_xc_23 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(_weight_xc))), _descale_xc_2);
                float16x4_t _weight_xc_0 = vget_low_f16(_weight_xc_01);
                float16x4_t _weight_xc_1 = vget_high_f16(_weight_xc_01);
                float16x4_t _weight_xc_2 = vget_low_f16(_weight_xc_23);
                float16x4_t _weight_xc_3 = vget_high_f16(_weight_xc_23);

                _rnn_H = vfma_lane_f16(_rnn_H, _weight_xc_0, _x, 0);
                _sum1 = vfma_lane_f16(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfma_lane_f16(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfma_lane_f16(_sum3, _weight_xc_3, _x, 3);

                weight_xc_int8_ptr += 16;
            }
            for (; i < size; i++)
            {
                float16x4_t _x = vdup_n_f16(x[i]);
                float16x4_t _weight_xc = vmul_f16(vcvt_f16_s16(vget_low_s16(vmovl_s8(vld1_s8(weight_xc_int8_ptr)))), _descale_xc);
                _rnn_H = vfma_f16(_rnn_H, _weight_xc, _x);

                weight_xc_int8_ptr += 4;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float16x4_t _hidden_state = vcvt_f16_f32(vld1q_f32((const float*)hidden_state + i));

                int8x16_t _weight_hc = vld1q_s8(weight_hc_int8_ptr);
                float16x8_t _weight_hc_01 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(_weight_hc))), _descale_hc_2);
                float16x8_t _weight_hc_23 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(_weight_hc))), _descale_hc_2);
                float16x4_t _weight_hc_0 = vget_low_f16(_weight_hc_01);
                float16x4_t _weight_hc_1 = vget_high_f16(_weight_hc_01);
                float16x4_t _weight_hc_2 = vget_low_f16(_weight_hc_23);
                float16x4_t _weight_hc_3 = vget_high_f16(_weight_hc_23);

                _rnn_H = vfma_lane_f16(_rnn_H, _weight_hc_0, _hidden_state, 0);
                _sum1 = vfma_lane_f16(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfma_lane_f16(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfma_lane_f16(_sum3, _weight_hc_3, _hidden_state, 3);

                weight_hc_int8_ptr += 16;
            }
            for (; i < num_output; i++)
            {
                float16x4_t _hidden_state = vdup_n_f16((__fp16)hidden_state[i]);
                float16x4_t _weight_hc = vmul_f16(vcvt_f16_s16(vget_low_s16(vmovl_s8(vld1_s8(weight_hc_int8_ptr)))), _descale_hc);
                _rnn_H = vfma_f16(_rnn_H, _weight_hc, _hidden_state);

                weight_hc_int8_ptr += 4;
            }

            _rnn_H = vadd_f16(_rnn_H, _sum1);
            _sum2 = vadd_f16(_sum2, _sum3);
            _rnn_H = vadd_f16(_rnn_H, _sum2);

            float32x4_t _H32 = tanh_ps(vcvt_f32_f16(_rnn_H));

            vst1q_f32((float*)gates + q, _H32);
        }
        remain_num_output_start += nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            const signed char* weight_xc_int8_ptr = weight_xc_int8.row<const signed char>(q / 8 + (q % 8) / 4 + q % 4);
            const signed char* weight_hc_int8_ptr = weight_hc_int8.row<const signed char>(q / 8 + (q % 8) / 4 + q % 4);
            const __fp16* weight_xc_int8_descales_ptr = weight_xc_int8_descales.row<const __fp16>(q / 8 + (q % 8) / 4 + q % 4);
            const __fp16* weight_hc_int8_descales_ptr = weight_hc_int8_descales.row<const __fp16>(q / 8 + (q % 8) / 4 + q % 4);

            const __fp16 descale_xc = weight_xc_int8_descales_ptr[0];
            const __fp16 descale_hc = weight_hc_int8_descales_ptr[0];

            __fp16 H = ((const __fp16*)bias_c)[q];

            for (int i = 0; i < size; i++)
            {
                H += (__fp16)weight_xc_int8_ptr[i] * descale_xc * x[i];
            }

            for (int i = 0; i < num_output; i++)
            {
                H += (__fp16)weight_hc_int8_ptr[i] * descale_hc * (__fp16)hidden_state[i];
            }

            float H32 = tanhf((float)H);

            gates[q] = H32;
        }

        __fp16* output_data = top_blob.row<__fp16>(ti);

        float* hidden_ptr = hidden_state;

        nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            float32x4_t _rnn_H = vld1q_f32((float*)gates + q);

            vst1q_f32(hidden_ptr + q, _rnn_H);
            vst1_f16(output_data + q, vcvt_f16_f32(_rnn_H));
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            float H = gates[q];

            hidden_ptr[q] = H;
            output_data[q] = (__fp16)H;
        }
    }

    return 0;
}

static int rnn_fp16s_int8(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc_int8, const Mat& weight_xc_int8_descales, const Mat& bias_c, const Mat& weight_hc_int8, const Mat& weight_hc_int8_descales, Mat& hidden_state, const Option& opt)
{
    if (opt.use_fp16_arithmetic)
        return rnn_fp16sa_int8(bottom_blob, top_blob, reverse, weight_xc_int8, weight_xc_int8_descales, bias_c, weight_hc_int8, weight_hc_int8_descales, hidden_state, opt);

    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // num_output
    Mat gates(num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        const __fp16* x = bottom_blob.row<const __fp16>(ti);

        int nn_num_output = num_output >> 2;
        int remain_num_output_start = nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            const signed char* weight_xc_int8_ptr = weight_xc_int8.row<const signed char>(q / 4);
            const signed char* weight_hc_int8_ptr = weight_hc_int8.row<const signed char>(q / 4);

            const float* weight_xc_int8_descales_ptr = weight_xc_int8_descales.row(q / 4);
            const float* weight_hc_int8_descales_ptr = weight_hc_int8_descales.row(q / 4);

            float32x4_t _descale_xc = vld1q_f32(weight_xc_int8_descales_ptr);
            float32x4_t _descale_hc = vld1q_f32(weight_hc_int8_descales_ptr);

            float32x4_t _rnn_H = vcvt_f32_f16(vld1_f16((const __fp16*)bias_c + q));
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _x = vcvt_f32_f16(vld1_f16(x + i));

                int8x16_t _weight_xc = vld1q_s8(weight_xc_int8_ptr);
                int16x8_t _weight_xc_01 = vmovl_s8(vget_low_s8(_weight_xc));
                int16x8_t _weight_xc_23 = vmovl_s8(vget_high_s8(_weight_xc));
                float32x4_t _weight_xc_0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(_weight_xc_01))), _descale_xc);
                float32x4_t _weight_xc_1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(_weight_xc_01))), _descale_xc);
                float32x4_t _weight_xc_2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(_weight_xc_23))), _descale_xc);
                float32x4_t _weight_xc_3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(_weight_xc_23))), _descale_xc);

                _rnn_H = vfmaq_laneq_f32(_rnn_H, _weight_xc_0, _x, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_xc_3, _x, 3);

                weight_xc_int8_ptr += 16;
            }
            for (; i < size; i++)
            {
                float32x4_t _x = vcvt_f32_f16(vdup_n_f16(x[i]));
                float32x4_t _weight_xc = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vld1_s8(weight_xc_int8_ptr))))), _descale_xc);
                _rnn_H = vfmaq_f32(_rnn_H, _weight_xc, _x);

                weight_xc_int8_ptr += 4;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float32x4_t _hidden_state = vld1q_f32((const float*)hidden_state + i);

                int8x16_t _weight_hc = vld1q_s8(weight_hc_int8_ptr);
                int16x8_t _weight_hc_01 = vmovl_s8(vget_low_s8(_weight_hc));
                int16x8_t _weight_hc_23 = vmovl_s8(vget_high_s8(_weight_hc));
                float32x4_t _weight_hc_0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(_weight_hc_01))), _descale_hc);
                float32x4_t _weight_hc_1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(_weight_hc_01))), _descale_hc);
                float32x4_t _weight_hc_2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(_weight_hc_23))), _descale_hc);
                float32x4_t _weight_hc_3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(_weight_hc_23))), _descale_hc);

                _rnn_H = vfmaq_laneq_f32(_rnn_H, _weight_hc_0, _hidden_state, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_hc_3, _hidden_state, 3);

                weight_hc_int8_ptr += 16;
            }
            for (; i < num_output; i++)
            {
                float32x4_t _hidden_state = vdupq_n_f32(hidden_state[i]);
                float32x4_t _weight_hc = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vld1_s8(weight_hc_int8_ptr))))), _descale_hc);
                _rnn_H = vfmaq_f32(_rnn_H, _weight_hc, _hidden_state);

                weight_hc_int8_ptr += 4;
            }

            _rnn_H = vaddq_f32(_rnn_H, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _rnn_H = vaddq_f32(_rnn_H, _sum2);

            _rnn_H = tanh_ps(_rnn_H);

            vst1q_f32((float*)gates + q, _rnn_H);
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            const signed char* weight_xc_int8_ptr = weight_xc_int8.row<const signed char>(q / 4 + q % 4);
            const signed char* weight_hc_int8_ptr = weight_hc_int8.row<const signed char>(q / 4 + q % 4);
            const float* weight_xc_int8_descales_ptr = weight_xc_int8_descales.row(q / 4 + q % 4);
            const float* weight_hc_int8_descales_ptr = weight_hc_int8_descales.row(q / 4 + q % 4);

            const float descale_xc = weight_xc_int8_descales_ptr[0];
            const float descale_hc = weight_hc_int8_descales_ptr[0];

            float H = (float)(((const __fp16*)bias_c)[q]);

            for (int i = 0; i < size; i++)
            {
                H += weight_xc_int8_ptr[i] * descale_xc * (float)x[i];
            }

            for (int i = 0; i < num_output; i++)
            {
                H += weight_hc_int8_ptr[i] * descale_hc * hidden_state[i];
            }

            H = tanhf(H);

            gates[q] = H;
        }

        __fp16* output_data = top_blob.row<__fp16>(ti);

        float* hidden_ptr = hidden_state;

        nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            float32x4_t _rnn_H = vld1q_f32((float*)gates + q);

            vst1q_f32(hidden_ptr + q, _rnn_H);
            vst1_f16(output_data + q, vcvt_f16_f32(_rnn_H));
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            float H = gates[q];

            hidden_ptr[q] = H;
            output_data[q] = (__fp16)H;
        }
    }

    return 0;
}
#endif // NCNN_INT8

int RNN_arm::create_pipeline_fp16s(const Option& opt)
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        const int num_directions = direction == 2 ? 2 : 1;
        const int size = weight_data_size / num_directions / num_output;

        if (opt.use_fp16_arithmetic)
        {
            weight_xc_data_packed.create(size * 8, num_output / 8 + (num_output % 8) / 4 + num_output % 4, num_directions, 1u, 1);
            weight_hc_data_packed.create(num_output * 8, num_output / 8 + (num_output % 8) / 4 + num_output % 4, num_directions, 1u, 1);
            weight_xc_data_int8_descales_packed.create(8, num_output / 8 + (num_output % 8) / 4 + num_output % 4, num_directions, 2u, 1);
            weight_hc_data_int8_descales_packed.create(8, num_output / 8 + (num_output % 8) / 4 + num_output % 4, num_directions, 2u, 1);
        }
        else
        {
            weight_xc_data_packed.create(size * 4, num_output / 4 + num_output % 4, num_directions, 1u, 1);
            weight_hc_data_packed.create(num_output * 4, num_output / 4 + num_output % 4, num_directions, 1u, 1);
            weight_xc_data_int8_descales_packed.create(4, num_output / 4 + num_output % 4, num_directions, 4u, 1);
            weight_hc_data_int8_descales_packed.create(4, num_output / 4 + num_output % 4, num_directions, 4u, 1);
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int dr = 0; dr < num_directions; dr++)
        {
            const Mat weight_xc = weight_xc_data.channel(dr);
            const Mat weight_hc = weight_hc_data.channel(dr);
            const float* weight_xc_int8_scales = weight_xc_data_int8_scales.row(dr);
            const float* weight_hc_int8_scales = weight_hc_data_int8_scales.row(dr);

            Mat weight_xc_data_packed_dr = weight_xc_data_packed.channel(dr);
            Mat weight_hc_data_packed_dr = weight_hc_data_packed.channel(dr);
            Mat weight_xc_data_int8_descales_packed_dr = weight_xc_data_int8_descales_packed.channel(dr);
            Mat weight_hc_data_int8_descales_packed_dr = weight_hc_data_int8_descales_packed.channel(dr);

            int q = 0;
            if (opt.use_fp16_arithmetic)
            {
                for (; q + 7 < num_output; q += 8)
                {
                    const signed char* weight_xc_0 = weight_xc.row<const signed char>(q);
                    const signed char* weight_xc_1 = weight_xc.row<const signed char>(q + 1);
                    const signed char* weight_xc_2 = weight_xc.row<const signed char>(q + 2);
                    const signed char* weight_xc_3 = weight_xc.row<const signed char>(q + 3);
                    const signed char* weight_xc_4 = weight_xc.row<const signed char>(q + 4);
                    const signed char* weight_xc_5 = weight_xc.row<const signed char>(q + 5);
                    const signed char* weight_xc_6 = weight_xc.row<const signed char>(q + 6);
                    const signed char* weight_xc_7 = weight_xc.row<const signed char>(q + 7);

                    const signed char* weight_hc_0 = weight_hc.row<const signed char>(q);
                    const signed char* weight_hc_1 = weight_hc.row<const signed char>(q + 1);
                    const signed char* weight_hc_2 = weight_hc.row<const signed char>(q + 2);
                    const signed char* weight_hc_3 = weight_hc.row<const signed char>(q + 3);
                    const signed char* weight_hc_4 = weight_hc.row<const signed char>(q + 4);
                    const signed char* weight_hc_5 = weight_hc.row<const signed char>(q + 5);
                    const signed char* weight_hc_6 = weight_hc.row<const signed char>(q + 6);
                    const signed char* weight_hc_7 = weight_hc.row<const signed char>(q + 7);

                    signed char* weight_xc_ptr = weight_xc_data_packed_dr.row<signed char>(q / 8);
                    signed char* weight_hc_ptr = weight_hc_data_packed_dr.row<signed char>(q / 8);
                    __fp16* weight_xc_int8_descales_ptr = weight_xc_data_int8_descales_packed_dr.row<__fp16>(q / 8);
                    __fp16* weight_hc_int8_descales_ptr = weight_hc_data_int8_descales_packed_dr.row<__fp16>(q / 8);

                    for (int i = 0; i < size; i++)
                    {
                        weight_xc_ptr[0] = weight_xc_0[i];
                        weight_xc_ptr[1] = weight_xc_1[i];
                        weight_xc_ptr[2] = weight_xc_2[i];
                        weight_xc_ptr[3] = weight_xc_3[i];
                        weight_xc_ptr[4] = weight_xc_4[i];
                        weight_xc_ptr[5] = weight_xc_5[i];
                        weight_xc_ptr[6] = weight_xc_6[i];
                        weight_xc_ptr[7] = weight_xc_7[i];

                        weight_xc_ptr += 8;
                    }

                    for (int i = 0; i < num_output; i++)
                    {
                        weight_hc_ptr[0] = weight_hc_0[i];
                        weight_hc_ptr[1] = weight_hc_1[i];
                        weight_hc_ptr[2] = weight_hc_2[i];
                        weight_hc_ptr[3] = weight_hc_3[i];
                        weight_hc_ptr[4] = weight_hc_4[i];
                        weight_hc_ptr[5] = weight_hc_5[i];
                        weight_hc_ptr[6] = weight_hc_6[i];
                        weight_hc_ptr[7] = weight_hc_7[i];

                        weight_hc_ptr += 8;
                    }

                    float32x4_t _xc0 = vld1q_f32(weight_xc_int8_scales + q);
                    float32x4_t _xc1 = vld1q_f32(weight_xc_int8_scales + q + 4);
                    float32x4_t _hc0 = vld1q_f32(weight_hc_int8_scales + q);
                    float32x4_t _hc1 = vld1q_f32(weight_hc_int8_scales + q + 4);

                    float32x4_t _one = vdupq_n_f32(1.f);
                    float16x4_t _reciprocal_xc0 = vcvt_f16_f32(vdivq_f32(_one, _xc0));
                    float16x4_t _reciprocal_xc1 = vcvt_f16_f32(vdivq_f32(_one, _xc1));
                    float16x4_t _reciprocal_hc0 = vcvt_f16_f32(vdivq_f32(_one, _hc0));
                    float16x4_t _reciprocal_hc1 = vcvt_f16_f32(vdivq_f32(_one, _hc1));

                    vst1q_f16(weight_xc_int8_descales_ptr, vcombine_f16(_reciprocal_xc0, _reciprocal_xc1));
                    vst1q_f16(weight_hc_int8_descales_ptr, vcombine_f16(_reciprocal_hc0, _reciprocal_hc1));
                }
            }
            for (; q + 3 < num_output; q += 4)
            {
                const signed char* weight_xc_0 = weight_xc.row<const signed char>(q);
                const signed char* weight_xc_1 = weight_xc.row<const signed char>(q + 1);
                const signed char* weight_xc_2 = weight_xc.row<const signed char>(q + 2);
                const signed char* weight_xc_3 = weight_xc.row<const signed char>(q + 3);

                const signed char* weight_hc_0 = weight_hc.row<const signed char>(q);
                const signed char* weight_hc_1 = weight_hc.row<const signed char>(q + 1);
                const signed char* weight_hc_2 = weight_hc.row<const signed char>(q + 2);
                const signed char* weight_hc_3 = weight_hc.row<const signed char>(q + 3);

                int qq = opt.use_fp16_arithmetic ? q / 8 + (q % 8) / 4 : q / 4;
                signed char* weight_xc_ptr = weight_xc_data_packed_dr.row<signed char>(qq);
                signed char* weight_hc_ptr = weight_hc_data_packed_dr.row<signed char>(qq);
                float* weight_xc_int8_descales_ptr = weight_xc_data_int8_descales_packed_dr.row(qq);
                float* weight_hc_int8_descales_ptr = weight_hc_data_int8_descales_packed_dr.row(qq);

                for (int i = 0; i < size; i++)
                {
                    weight_xc_ptr[0] = weight_xc_0[i];
                    weight_xc_ptr[1] = weight_xc_1[i];
                    weight_xc_ptr[2] = weight_xc_2[i];
                    weight_xc_ptr[3] = weight_xc_3[i];

                    weight_xc_ptr += 4;
                }

                for (int i = 0; i < num_output; i++)
                {
                    weight_hc_ptr[0] = weight_hc_0[i];
                    weight_hc_ptr[1] = weight_hc_1[i];
                    weight_hc_ptr[2] = weight_hc_2[i];
                    weight_hc_ptr[3] = weight_hc_3[i];

                    weight_hc_ptr += 4;
                }

                float32x4_t _xc = vld1q_f32(weight_xc_int8_scales + q);
                float32x4_t _hc = vld1q_f32(weight_hc_int8_scales + q);

                float32x4_t _one = vdupq_n_f32(1.f);
                float32x4_t _reciprocal_xc = vdivq_f32(_one, _xc);
                float32x4_t _reciprocal_hc = vdivq_f32(_one, _hc);

                if (opt.use_fp16_arithmetic)
                {
                    vst1_f16((__fp16*)weight_xc_int8_descales_ptr, vcvt_f16_f32(_reciprocal_xc));
                    vst1_f16((__fp16*)weight_hc_int8_descales_ptr, vcvt_f16_f32(_reciprocal_hc));
                }
                else
                {
                    vst1q_f32(weight_xc_int8_descales_ptr, _reciprocal_xc);
                    vst1q_f32(weight_hc_int8_descales_ptr, _reciprocal_hc);
                }
            }
            for (; q < num_output; q++)
            {
                const signed char* weight_xc_0 = weight_xc.row<const signed char>(q);
                const signed char* weight_hc_0 = weight_hc.row<const signed char>(q);

                int qq = opt.use_fp16_arithmetic ? q / 8 + (q % 8) / 4 + q % 4 : q / 4 + q % 4;
                signed char* weight_xc_ptr = weight_xc_data_packed_dr.row<signed char>(qq);
                signed char* weight_hc_ptr = weight_hc_data_packed_dr.row<signed char>(qq);
                float* weight_xc_int8_descales_ptr = weight_xc_data_int8_descales_packed_dr.row(qq);
                float* weight_hc_int8_descales_ptr = weight_hc_data_int8_descales_packed_dr.row(qq);

                for (int i = 0; i < size; i++)
                {
                    weight_xc_ptr[i] = weight_xc_0[i];
                }

                for (int i = 0; i < num_output; i++)
                {
                    weight_hc_ptr[i] = weight_hc_0[i];
                }

                if (opt.use_fp16_arithmetic)
                {
                    ((__fp16*)weight_xc_int8_descales_ptr)[0] = (__fp16)(1.f / weight_xc_int8_scales[q]);
                    ((__fp16*)weight_hc_int8_descales_ptr)[0] = (__fp16)(1.f / weight_hc_int8_scales[q]);
                }
                else
                {
                    weight_xc_int8_descales_ptr[0] = 1.f / weight_xc_int8_scales[q];
                    weight_hc_int8_descales_ptr[0] = 1.f / weight_hc_int8_scales[q];
                }
            }
        }

        cast_float32_to_float16(bias_c_data, bias_c_data_packed, opt);

        if (opt.lightmode)
        {
            weight_xc_data.release();
            bias_c_data.release();
            weight_hc_data.release();
            weight_xc_data_int8_scales.release();
            weight_hc_data_int8_scales.release();
        }

        return 0;
    }
#endif

    int num_directions = direction == 2 ? 2 : 1;
    int size = weight_data_size / num_directions / num_output;

    if (opt.use_fp16_arithmetic)
    {
        weight_xc_data_packed.create(size * 8, num_output / 8 + (num_output % 8) / 4 + num_output % 4, num_directions, 2u, 1);
        weight_hc_data_packed.create(num_output * 8, num_output / 8 + (num_output % 8) / 4 + num_output % 4, num_directions, 2u, 1);
    }
    else
    {
        weight_xc_data_packed.create(size * 4, num_output / 4 + num_output % 4, num_directions, 2u, 1);
        weight_hc_data_packed.create(num_output * 4, num_output / 4 + num_output % 4, num_directions, 2u, 1);
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int dr = 0; dr < num_directions; dr++)
    {
        const Mat weight_xc = weight_xc_data.channel(dr);
        const Mat weight_hc = weight_hc_data.channel(dr);

        Mat weight_xc_data_packed_dr = weight_xc_data_packed.channel(dr);
        Mat weight_hc_data_packed_dr = weight_hc_data_packed.channel(dr);

        int q = 0;
        if (opt.use_fp16_arithmetic)
        {
            for (; q + 7 < num_output; q += 8)
            {
                const float* weight_xc_0 = weight_xc.row(q);
                const float* weight_xc_1 = weight_xc.row(q + 1);
                const float* weight_xc_2 = weight_xc.row(q + 2);
                const float* weight_xc_3 = weight_xc.row(q + 3);
                const float* weight_xc_4 = weight_xc.row(q + 4);
                const float* weight_xc_5 = weight_xc.row(q + 5);
                const float* weight_xc_6 = weight_xc.row(q + 6);
                const float* weight_xc_7 = weight_xc.row(q + 7);

                const float* weight_hc_0 = weight_hc.row(q);
                const float* weight_hc_1 = weight_hc.row(q + 1);
                const float* weight_hc_2 = weight_hc.row(q + 2);
                const float* weight_hc_3 = weight_hc.row(q + 3);
                const float* weight_hc_4 = weight_hc.row(q + 4);
                const float* weight_hc_5 = weight_hc.row(q + 5);
                const float* weight_hc_6 = weight_hc.row(q + 6);
                const float* weight_hc_7 = weight_hc.row(q + 7);

                __fp16* weight_xc = weight_xc_data_packed_dr.row<__fp16>(q / 8);
                __fp16* weight_hc = weight_hc_data_packed_dr.row<__fp16>(q / 8);

                for (int i = 0; i < size; i++)
                {
                    weight_xc[0] = (__fp16)weight_xc_0[i];
                    weight_xc[1] = (__fp16)weight_xc_1[i];
                    weight_xc[2] = (__fp16)weight_xc_2[i];
                    weight_xc[3] = (__fp16)weight_xc_3[i];
                    weight_xc[4] = (__fp16)weight_xc_4[i];
                    weight_xc[5] = (__fp16)weight_xc_5[i];
                    weight_xc[6] = (__fp16)weight_xc_6[i];
                    weight_xc[7] = (__fp16)weight_xc_7[i];

                    weight_xc += 8;
                }

                for (int i = 0; i < num_output; i++)
                {
                    weight_hc[0] = (__fp16)weight_hc_0[i];
                    weight_hc[1] = (__fp16)weight_hc_1[i];
                    weight_hc[2] = (__fp16)weight_hc_2[i];
                    weight_hc[3] = (__fp16)weight_hc_3[i];
                    weight_hc[4] = (__fp16)weight_hc_4[i];
                    weight_hc[5] = (__fp16)weight_hc_5[i];
                    weight_hc[6] = (__fp16)weight_hc_6[i];
                    weight_hc[7] = (__fp16)weight_hc_7[i];

                    weight_hc += 8;
                }
            }
        }
        for (; q + 3 < num_output; q += 4)
        {
            const float* weight_xc_0 = weight_xc.row(q);
            const float* weight_xc_1 = weight_xc.row(q + 1);
            const float* weight_xc_2 = weight_xc.row(q + 2);
            const float* weight_xc_3 = weight_xc.row(q + 3);

            const float* weight_hc_0 = weight_hc.row(q);
            const float* weight_hc_1 = weight_hc.row(q + 1);
            const float* weight_hc_2 = weight_hc.row(q + 2);
            const float* weight_hc_3 = weight_hc.row(q + 3);

            __fp16* weight_xc = opt.use_fp16_arithmetic ? weight_xc_data_packed_dr.row<__fp16>(q / 8 + (q % 8) / 4) : weight_xc_data_packed_dr.row<__fp16>(q / 4);
            __fp16* weight_hc = opt.use_fp16_arithmetic ? weight_hc_data_packed_dr.row<__fp16>(q / 8 + (q % 8) / 4) : weight_hc_data_packed_dr.row<__fp16>(q / 4);

            for (int i = 0; i < size; i++)
            {
                weight_xc[0] = (__fp16)weight_xc_0[i];
                weight_xc[1] = (__fp16)weight_xc_1[i];
                weight_xc[2] = (__fp16)weight_xc_2[i];
                weight_xc[3] = (__fp16)weight_xc_3[i];

                weight_xc += 4;
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc[0] = (__fp16)weight_hc_0[i];
                weight_hc[1] = (__fp16)weight_hc_1[i];
                weight_hc[2] = (__fp16)weight_hc_2[i];
                weight_hc[3] = (__fp16)weight_hc_3[i];

                weight_hc += 4;
            }
        }
        for (; q < num_output; q++)
        {
            const float* weight_xc_0 = weight_xc.row(q);
            const float* weight_hc_0 = weight_hc.row(q);

            __fp16* weight_xc = opt.use_fp16_arithmetic ? weight_xc_data_packed_dr.row<__fp16>(q / 8 + (q % 8) / 4 + q % 4) : weight_xc_data_packed_dr.row<__fp16>(q / 4 + q % 4);
            __fp16* weight_hc = opt.use_fp16_arithmetic ? weight_hc_data_packed_dr.row<__fp16>(q / 8 + (q % 8) / 4 + q % 4) : weight_hc_data_packed_dr.row<__fp16>(q / 4 + q % 4);

            for (int i = 0; i < size; i++)
            {
                weight_xc[i] = (__fp16)weight_xc_0[i];
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc[i] = (__fp16)weight_hc_0[i];
            }
        }
    }

    cast_float32_to_float16(bias_c_data, bias_c_data_packed, opt);

    if (opt.lightmode)
    {
        weight_xc_data.release();
        bias_c_data.release();
        weight_hc_data.release();
    }

    return 0;
}

int RNN_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = rnn_fp16s_int8(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), weight_xc_data_int8_descales_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), weight_hc_data_int8_descales_packed.channel(0), hidden, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = rnn_fp16s(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
            if (ret != 0)
                return ret;
        }
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 2u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = rnn_fp16s_int8(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), weight_xc_data_int8_descales_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), weight_hc_data_int8_descales_packed.channel(0), hidden, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = rnn_fp16s(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
            if (ret != 0)
                return ret;
        }

        hidden.fill(0.f);

#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = rnn_fp16s_int8(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), weight_xc_data_int8_descales_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), weight_hc_data_int8_descales_packed.channel(1), hidden, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = rnn_fp16s(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden, opt);
            if (ret != 0)
                return ret;
        }

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

int RNN_arm::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = rnn_fp16s_int8(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), weight_xc_data_int8_descales_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), weight_hc_data_int8_descales_packed.channel(0), hidden, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = rnn_fp16s(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
            if (ret != 0)
                return ret;
        }
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
#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = rnn_fp16s_int8(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), weight_xc_data_int8_descales_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), weight_hc_data_int8_descales_packed.channel(0), hidden0, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = rnn_fp16s(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden0, opt);
            if (ret != 0)
                return ret;
        }

        Mat hidden1 = hidden.row_range(1, 1);
#if NCNN_INT8
        if (int8_scale_term)
        {
            int ret = rnn_fp16s_int8(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), weight_xc_data_int8_descales_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), weight_hc_data_int8_descales_packed.channel(1), hidden1, opt);
            if (ret != 0)
                return ret;
        }
        else
#endif
        {
            int ret = rnn_fp16s(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden1, opt);
            if (ret != 0)
                return ret;
        }

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
#endif

} // namespace ncnn
