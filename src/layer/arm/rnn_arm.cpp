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

#include "rnn_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"

#include <math.h>

namespace ncnn {

RNN_arm::RNN_arm()
{
#if __ARM_NEON
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int RNN_arm::create_pipeline(const Option& opt)
{
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage)
    {
        return create_pipeline_bf16s(opt);
    }
#endif

    int num_directions = direction == 2 ? 2 : 1;
    int size = weight_data_size / num_directions / num_output;

#if __ARM_NEON
    weight_xc_data_packed.create(size * 4, num_output / 4 + num_output % 4, num_directions);
    weight_hc_data_packed.create(num_output * 4, num_output / 4 + num_output % 4, num_directions);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int dr = 0; dr < num_directions; dr++)
    {
        const Mat weight_xc = weight_xc_data.channel(dr);
        const Mat weight_hc = weight_hc_data.channel(dr);

        Mat weight_xc_data_packed_dr = weight_xc_data_packed.channel(dr);
        Mat weight_hc_data_packed_dr = weight_hc_data_packed.channel(dr);

        int q = 0;
#if __ARM_NEON
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

            float* weight_xc = weight_xc_data_packed_dr.row(q / 4);
            float* weight_hc = weight_hc_data_packed_dr.row(q / 4);

            for (int i = 0; i < size; i++)
            {
                weight_xc[0] = weight_xc_0[i];
                weight_xc[1] = weight_xc_1[i];
                weight_xc[2] = weight_xc_2[i];
                weight_xc[3] = weight_xc_3[i];

                weight_xc += 4;
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc[0] = weight_hc_0[i];
                weight_hc[1] = weight_hc_1[i];
                weight_hc[2] = weight_hc_2[i];
                weight_hc[3] = weight_hc_3[i];

                weight_hc += 4;
            }
        }
#endif // __ARM_NEON
        for (; q < num_output; q++)
        {
            const float* weight_xc_0 = weight_xc.row(q);
            const float* weight_hc_0 = weight_hc.row(q);

#if __ARM_NEON
            float* weight_xc = weight_xc_data_packed_dr.row(q / 4 + q % 4);
            float* weight_hc = weight_hc_data_packed_dr.row(q / 4 + q % 4);
#else
            float* weight_xc = weight_xc_data_packed_dr.row(q);
            float* weight_hc = weight_hc_data_packed_dr.row(q);
#endif // __ARM_NEON

            for (int i = 0; i < size; i++)
            {
                weight_xc[i] = weight_xc_0[i];
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc[i] = weight_hc_0[i];
            }
        }
    }
#else
    weight_xc_data_packed = weight_xc_data;
    weight_hc_data_packed = weight_hc_data;
#endif

    bias_c_data_packed = bias_c_data;

    return 0;
}

static int rnn(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
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

        const float* x = bottom_blob.row(ti);

        int q = 0;
#if __ARM_NEON
        for (; q + 3 < num_output; q += 4)
        {
            const float* weight_xc_ptr = weight_xc.row(q / 4);
            const float* weight_hc_ptr = weight_hc.row(q / 4);

            float32x4_t _H = vld1q_f32((const float*)bias_c + q);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _x = vld1q_f32(x + i);
                float32x4_t _weight_xc = vld1q_f32(weight_xc_ptr);
                float32x4_t _weight_xc_1 = vld1q_f32(weight_xc_ptr + 4);
                float32x4_t _weight_xc_2 = vld1q_f32(weight_xc_ptr + 8);
                float32x4_t _weight_xc_3 = vld1q_f32(weight_xc_ptr + 12);
#if __aarch64__
                _H = vfmaq_laneq_f32(_H, _weight_xc, _x, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_xc_3, _x, 3);
#else
                _H = vmlaq_lane_f32(_H, _weight_xc, vget_low_f32(_x), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _weight_xc_1, vget_low_f32(_x), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _weight_xc_2, vget_high_f32(_x), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _weight_xc_3, vget_high_f32(_x), 1);
#endif

                weight_xc_ptr += 16;
            }
            for (; i < size; i++)
            {
                float32x4_t _x = vdupq_n_f32(x[i]);
                float32x4_t _weight_xc = vld1q_f32(weight_xc_ptr);
                _H = vmlaq_f32(_H, _weight_xc, _x);

                weight_xc_ptr += 4;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float32x4_t _hidden_state = vld1q_f32((const float*)hidden_state + i);
                float32x4_t _weight_hc = vld1q_f32(weight_hc_ptr);
                float32x4_t _weight_hc_1 = vld1q_f32(weight_hc_ptr + 4);
                float32x4_t _weight_hc_2 = vld1q_f32(weight_hc_ptr + 8);
                float32x4_t _weight_hc_3 = vld1q_f32(weight_hc_ptr + 12);
#if __aarch64__
                _H = vfmaq_laneq_f32(_H, _weight_hc, _hidden_state, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_hc_3, _hidden_state, 3);
#else
                _H = vmlaq_lane_f32(_H, _weight_hc, vget_low_f32(_hidden_state), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _weight_hc_1, vget_low_f32(_hidden_state), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _weight_hc_2, vget_high_f32(_hidden_state), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _weight_hc_3, vget_high_f32(_hidden_state), 1);
#endif

                weight_hc_ptr += 16;
            }
            for (; i < num_output; i++)
            {
                float32x4_t _hidden_state = vdupq_n_f32(hidden_state[i]);
                float32x4_t _weight_hc = vld1q_f32(weight_hc_ptr);
                _H = vmlaq_f32(_H, _weight_hc, _hidden_state);

                weight_hc_ptr += 4;
            }

            _H = vaddq_f32(_H, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _H = vaddq_f32(_H, _sum2);

            _H = tanh_ps(_H);

            vst1q_f32((float*)gates + q, _H);
        }
#endif // __ARM_NEON
        for (; q < num_output; q++)
        {
#if __ARM_NEON
            const float* weight_xc_ptr = weight_xc.row(q / 4 + q % 4);
            const float* weight_hc_ptr = weight_hc.row(q / 4 + q % 4);
#else
            const float* weight_xc_ptr = weight_xc.row(q);
            const float* weight_hc_ptr = weight_hc.row(q);
#endif // __ARM_NEON

            float H = bias_c[q];

            for (int i = 0; i < size; i++)
            {
                H += weight_xc_ptr[i] * x[i];
            }

            for (int i = 0; i < num_output; i++)
            {
                H += weight_hc_ptr[i] * hidden_state[i];
            }

            H = tanh(H);

            gates[q] = H;
        }

        float* output_data = top_blob.row(ti);

        float* hidden_ptr = hidden_state;

        q = 0;
#if __ARM_NEON
        for (; q + 3 < num_output; q += 4)
        {
            float32x4_t _H = vld1q_f32((float*)gates + q);

            vst1q_f32(hidden_ptr, _H);
            vst1q_f32(output_data, _H);

            hidden_ptr += 4;
            output_data += 4;
        }
#endif // __ARM_NEON
        for (; q < num_output; q++)
        {
            float H = gates[q];

            *hidden_ptr++ = H;
            *output_data++ = H;
        }
    }

    return 0;
}

int RNN_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blob, top_blob, opt);
#endif

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
        int ret = rnn(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
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

        int ret0 = rnn(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.0f);

        int ret1 = rnn(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden, opt);
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

int RNN_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int elembits = bottom_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blobs, top_blobs, opt);
        else
            return forward_fp16s(bottom_blobs, top_blobs, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif

    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

    Mat hidden;
    Allocator* hidden_allocator = top_blobs.size() == 2 ? opt.blob_allocator : opt.workspace_allocator;
    if (bottom_blobs.size() == 2)
    {
        hidden = bottom_blobs[1].clone(hidden_allocator);
    }
    else
    {
        hidden.create(num_output, num_directions, 4u, hidden_allocator);
        if (hidden.empty())
            return -100;
        hidden.fill(0.f);
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Uni directional
    if (direction == 0 || direction == 1)
    {
        int ret = rnn(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
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
        int ret0 = rnn(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden0, opt);
        if (ret0 != 0)
            return ret0;

        Mat hidden1 = hidden.row_range(1, 1);
        int ret1 = rnn(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden1, opt);
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

    if (top_blobs.size() == 2)
    {
        top_blobs[1] = hidden;
    }

    return 0;
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static int rnn_fp16s(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
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

        int q = 0;
        for (; q + 3 < num_output; q += 4)
        {
            const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 4);
            const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 4);

            float32x4_t _H = vcvt_f32_f16(vld1_f16((const __fp16*)bias_c + q));
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
                _H = vfmaq_laneq_f32(_H, _weight_xc, _x, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_xc_3, _x, 3);

                weight_xc_ptr += 16;
            }
            for (; i < size; i++)
            {
                float32x4_t _x = vcvt_f32_f16(vdup_n_f16(x[i]));
                float32x4_t _weight_xc = vcvt_f32_f16(vld1_f16(weight_xc_ptr));
                _H = vfmaq_f32(_H, _weight_xc, _x);

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
                _H = vfmaq_laneq_f32(_H, _weight_hc, _hidden_state, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_hc_3, _hidden_state, 3);

                weight_hc_ptr += 16;
            }
            for (; i < num_output; i++)
            {
                float32x4_t _hidden_state = vdupq_n_f32(hidden_state[i]);
                float32x4_t _weight_hc = vcvt_f32_f16(vld1_f16(weight_hc_ptr));
                _H = vfmaq_f32(_H, _weight_hc, _hidden_state);

                weight_hc_ptr += 4;
            }

            _H = vaddq_f32(_H, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _H = vaddq_f32(_H, _sum2);

            _H = tanh_ps(_H);

            vst1q_f32((float*)gates + q, _H);
        }
        for (; q < num_output; q++)
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

            H = tanh(H);

            gates[q] = H;
        }

        __fp16* output_data = top_blob.row<__fp16>(ti);

        float* hidden_ptr = hidden_state;

        q = 0;
        for (; q + 3 < num_output; q += 4)
        {
            float32x4_t _H = vld1q_f32((float*)gates + q);

            vst1q_f32(hidden_ptr, _H);
            vst1_f16(output_data, vcvt_f16_f32(_H));

            hidden_ptr += 4;
            output_data += 4;
        }
        for (; q < num_output; q++)
        {
            float H = gates[q];

            *hidden_ptr++ = H;
            *output_data++ = (__fp16)H;
        }
    }

    return 0;
}

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

        int q = 0;
        for (; q + 7 < num_output; q += 8)
        {
            const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 8);
            const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 8);

            float16x8_t _H = vld1q_f16((const __fp16*)bias_c + q);
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
                _H = vfmaq_lane_f16(_H, _weight_xc, _x, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _weight_xc_3, _x, 3);

                weight_xc_ptr += 32;
            }
            for (; i < size; i++)
            {
                float16x8_t _x = vdupq_n_f16(x[i]);
                float16x8_t _weight_xc = vld1q_f16(weight_xc_ptr);
                _H = vfmaq_f16(_H, _weight_xc, _x);

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
                _H = vfmaq_lane_f16(_H, _weight_hc, _hidden_state, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _weight_hc_3, _hidden_state, 3);

                weight_hc_ptr += 32;
            }
            for (; i < num_output; i++)
            {
                float16x8_t _hidden_state = vdupq_n_f16((__fp16)hidden_state[i]);
                float16x8_t _weight_hc = vld1q_f16(weight_hc_ptr);
                _H = vfmaq_f16(_H, _weight_hc, _hidden_state);

                weight_hc_ptr += 8;
            }

            _H = vaddq_f16(_H, _sum1);
            _sum2 = vaddq_f16(_sum2, _sum3);
            _H = vaddq_f16(_H, _sum2);

            float32x4_t _H32low = tanh_ps(vcvt_f32_f16(vget_low_f16(_H)));
            float32x4_t _H32high = tanh_ps(vcvt_f32_f16(vget_high_f16(_H)));

            vst1q_f32((float*)gates + q, _H32low);
            vst1q_f32((float*)gates + q + 4, _H32high);
        }
        for (; q + 3 < num_output; q += 4)
        {
            const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 8 + (q % 8) / 4);
            const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 8 + (q % 8) / 4);

            float16x4_t _H = vld1_f16((const __fp16*)bias_c + q);
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
                _H = vfma_lane_f16(_H, _weight_xc, _x, 0);
                _sum1 = vfma_lane_f16(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfma_lane_f16(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfma_lane_f16(_sum3, _weight_xc_3, _x, 3);

                weight_xc_ptr += 16;
            }
            for (; i < size; i++)
            {
                float16x4_t _x = vdup_n_f16(x[i]);
                float16x4_t _weight_xc = vld1_f16(weight_xc_ptr);
                _H = vfma_f16(_H, _weight_xc, _x);

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
                _H = vfma_lane_f16(_H, _weight_hc, _hidden_state, 0);
                _sum1 = vfma_lane_f16(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfma_lane_f16(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfma_lane_f16(_sum3, _weight_hc_3, _hidden_state, 3);

                weight_hc_ptr += 16;
            }
            for (; i < num_output; i++)
            {
                float16x4_t _hidden_state = vdup_n_f16((__fp16)hidden_state[i]);
                float16x4_t _weight_hc = vld1_f16(weight_hc_ptr);
                _H = vfma_f16(_H, _weight_hc, _hidden_state);

                weight_hc_ptr += 4;
            }

            _H = vadd_f16(_H, _sum1);
            _sum2 = vadd_f16(_sum2, _sum3);
            _H = vadd_f16(_H, _sum2);

            float32x4_t _H32 = tanh_ps(vcvt_f32_f16(_H));

            vst1q_f32((float*)gates + q, _H32);
        }
        for (; q < num_output; q++)
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

            float H32 = tanh((float)H);

            gates[q] = H32;
        }

        __fp16* output_data = top_blob.row<__fp16>(ti);

        float* hidden_ptr = hidden_state;

        q = 0;
        for (; q + 3 < num_output; q += 4)
        {
            float32x4_t _H = vld1q_f32((float*)gates + q);

            vst1q_f32(hidden_ptr, _H);
            vst1_f16(output_data, vcvt_f16_f32(_H));

            hidden_ptr += 4;
            output_data += 4;
        }
        for (; q < num_output; q++)
        {
            float H = gates[q];

            *hidden_ptr++ = H;
            *output_data++ = (__fp16)H;
        }
    }

    return 0;
}

int RNN_arm::create_pipeline_fp16s(const Option& opt)
{
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
        int ret = rnn_fp16s(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
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

        int ret0 = rnn_fp16s(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.f);

        int ret1 = rnn_fp16s(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden, opt);
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
        int ret = rnn_fp16s(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
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
        int ret0 = rnn_fp16s(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden0, opt);
        if (ret0 != 0)
            return ret0;

        Mat hidden1 = hidden.row_range(1, 1);
        int ret1 = rnn_fp16s(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden1, opt);
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

int RNN_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
        int ret = rnn_fp16sa(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
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

        int ret0 = rnn_fp16sa(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.f);

        int ret1 = rnn_fp16sa(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden, opt);
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

int RNN_arm::forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
        int ret = rnn_fp16sa(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
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
        int ret0 = rnn_fp16sa(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden0, opt);
        if (ret0 != 0)
            return ret0;

        Mat hidden1 = hidden.row_range(1, 1);
        int ret1 = rnn_fp16sa(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden1, opt);
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
#endif

#if NCNN_BF16
static int rnn_bf16s(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
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

        const unsigned short* x = bottom_blob.row<const unsigned short>(ti);

        int q = 0;
#if __ARM_NEON
        for (; q + 3 < num_output; q += 4)
        {
            const unsigned short* weight_xc_ptr = weight_xc.row<const unsigned short>(q / 4);
            const unsigned short* weight_hc_ptr = weight_hc.row<const unsigned short>(q / 4);

            float32x4_t _H = vcvt_f32_bf16(vld1_u16((const unsigned short*)bias_c + q));
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _x = vcvt_f32_bf16(vld1_u16(x + i));
                float32x4_t _weight_xc = vcvt_f32_bf16(vld1_u16(weight_xc_ptr));
                float32x4_t _weight_xc_1 = vcvt_f32_bf16(vld1_u16(weight_xc_ptr + 4));
                float32x4_t _weight_xc_2 = vcvt_f32_bf16(vld1_u16(weight_xc_ptr + 8));
                float32x4_t _weight_xc_3 = vcvt_f32_bf16(vld1_u16(weight_xc_ptr + 12));
#if __aarch64__
                _H = vfmaq_laneq_f32(_H, _weight_xc, _x, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_xc_3, _x, 3);
#else
                _H = vmlaq_lane_f32(_H, _weight_xc, vget_low_f32(_x), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _weight_xc_1, vget_low_f32(_x), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _weight_xc_2, vget_high_f32(_x), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _weight_xc_3, vget_high_f32(_x), 1);
#endif

                weight_xc_ptr += 16;
            }
            for (; i < size; i++)
            {
                float32x4_t _x = vcvt_f32_bf16(vdup_n_u16(x[i]));
                float32x4_t _weight_xc = vcvt_f32_bf16(vld1_u16(weight_xc_ptr));
                _H = vmlaq_f32(_H, _weight_xc, _x);

                weight_xc_ptr += 4;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float32x4_t _hidden_state = vld1q_f32((const float*)hidden_state + i);
                float32x4_t _weight_hc = vcvt_f32_bf16(vld1_u16(weight_hc_ptr));
                float32x4_t _weight_hc_1 = vcvt_f32_bf16(vld1_u16(weight_hc_ptr + 4));
                float32x4_t _weight_hc_2 = vcvt_f32_bf16(vld1_u16(weight_hc_ptr + 8));
                float32x4_t _weight_hc_3 = vcvt_f32_bf16(vld1_u16(weight_hc_ptr + 12));
#if __aarch64__
                _H = vfmaq_laneq_f32(_H, _weight_hc, _hidden_state, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_hc_3, _hidden_state, 3);
#else
                _H = vmlaq_lane_f32(_H, _weight_hc, vget_low_f32(_hidden_state), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _weight_hc_1, vget_low_f32(_hidden_state), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _weight_hc_2, vget_high_f32(_hidden_state), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _weight_hc_3, vget_high_f32(_hidden_state), 1);
#endif

                weight_hc_ptr += 16;
            }
            for (; i < num_output; i++)
            {
                float32x4_t _hidden_state = vdupq_n_f32(hidden_state[i]);
                float32x4_t _weight_hc = vcvt_f32_bf16(vld1_u16(weight_hc_ptr));
                _H = vmlaq_f32(_H, _weight_hc, _hidden_state);

                weight_hc_ptr += 4;
            }

            _H = vaddq_f32(_H, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _H = vaddq_f32(_H, _sum2);

            _H = tanh_ps(_H);

            vst1q_f32((float*)gates + q, _H);
        }
#endif // __ARM_NEON
        for (; q < num_output; q++)
        {
#if __ARM_NEON
            const unsigned short* weight_xc_ptr = weight_xc.row<const unsigned short>(q / 4 + q % 4);
            const unsigned short* weight_hc_ptr = weight_hc.row<const unsigned short>(q / 4 + q % 4);
#else
            const unsigned short* weight_xc_ptr = weight_xc.row<const unsigned short>(q);
            const unsigned short* weight_hc_ptr = weight_hc.row<const unsigned short>(q);
#endif // __ARM_NEON

            float H = bfloat16_to_float32(((const unsigned short*)bias_c)[q]);

            for (int i = 0; i < size; i++)
            {
                H += bfloat16_to_float32(weight_xc_ptr[i]) * bfloat16_to_float32(x[i]);
            }

            for (int i = 0; i < num_output; i++)
            {
                H += bfloat16_to_float32(weight_hc_ptr[i]) * hidden_state[i];
            }

            H = tanh(H);

            gates[q] = H;
        }

        unsigned short* output_data = top_blob.row<unsigned short>(ti);

        float* hidden_ptr = hidden_state;

        q = 0;
#if __ARM_NEON
        for (; q + 3 < num_output; q += 4)
        {
            float32x4_t _H = vld1q_f32((float*)gates + q);

            vst1q_f32(hidden_ptr, _H);
            vst1_u16(output_data, vcvt_bf16_f32(_H));

            hidden_ptr += 4;
            output_data += 4;
        }
#endif // __ARM_NEON
        for (; q < num_output; q++)
        {
            float H = gates[q];

            *hidden_ptr++ = H;
            *output_data++ = float32_to_bfloat16(H);
        }
    }

    return 0;
}

int RNN_arm::create_pipeline_bf16s(const Option& opt)
{
    int num_directions = direction == 2 ? 2 : 1;
    int size = weight_data_size / num_directions / num_output;

#if __ARM_NEON
    weight_xc_data_packed.create(size * 4, num_output / 4 + num_output % 4, num_directions, 2u, 1);
    weight_hc_data_packed.create(num_output * 4, num_output / 4 + num_output % 4, num_directions, 2u, 1);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int dr = 0; dr < num_directions; dr++)
    {
        const Mat weight_xc = weight_xc_data.channel(dr);
        const Mat weight_hc = weight_hc_data.channel(dr);

        Mat weight_xc_data_packed_dr = weight_xc_data_packed.channel(dr);
        Mat weight_hc_data_packed_dr = weight_hc_data_packed.channel(dr);

        int q = 0;
#if __ARM_NEON
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

            unsigned short* weight_xc = weight_xc_data_packed_dr.row<unsigned short>(q / 4);
            unsigned short* weight_hc = weight_hc_data_packed_dr.row<unsigned short>(q / 4);

            for (int i = 0; i < size; i++)
            {
                weight_xc[0] = float32_to_bfloat16(weight_xc_0[i]);
                weight_xc[1] = float32_to_bfloat16(weight_xc_1[i]);
                weight_xc[2] = float32_to_bfloat16(weight_xc_2[i]);
                weight_xc[3] = float32_to_bfloat16(weight_xc_3[i]);

                weight_xc += 4;
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc[0] = float32_to_bfloat16(weight_hc_0[i]);
                weight_hc[1] = float32_to_bfloat16(weight_hc_1[i]);
                weight_hc[2] = float32_to_bfloat16(weight_hc_2[i]);
                weight_hc[3] = float32_to_bfloat16(weight_hc_3[i]);

                weight_hc += 4;
            }
        }
#endif // __ARM_NEON
        for (; q < num_output; q++)
        {
            const float* weight_xc_0 = weight_xc.row(q);
            const float* weight_hc_0 = weight_hc.row(q);

#if __ARM_NEON
            unsigned short* weight_xc = weight_xc_data_packed_dr.row<unsigned short>(q / 4 + q % 4);
            unsigned short* weight_hc = weight_hc_data_packed_dr.row<unsigned short>(q / 4 + q % 4);
#else
            unsigned short* weight_xc = weight_xc_data_packed_dr.row<unsigned short>(q);
            unsigned short* weight_hc = weight_hc_data_packed_dr.row<unsigned short>(q);
#endif // __ARM_NEON

            for (int i = 0; i < size; i++)
            {
                weight_xc[i] = float32_to_bfloat16(weight_xc_0[i]);
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc[i] = float32_to_bfloat16(weight_hc_0[i]);
            }
        }
    }
#else
    cast_float32_to_bfloat16(weight_xc_data, weight_xc_data_packed, opt);
    cast_float32_to_bfloat16(weight_hc_data, weight_hc_data_packed, opt);
#endif

    cast_float32_to_bfloat16(bias_c_data, bias_c_data_packed, opt);

    return 0;
}

int RNN_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
        int ret = rnn_bf16s(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
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

        int ret0 = rnn_bf16s(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.f);

        int ret1 = rnn_bf16s(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden, opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i = 0; i < T; i++)
        {
            const unsigned short* pf = top_blob_forward.row<const unsigned short>(i);
            const unsigned short* pr = top_blob_reverse.row<const unsigned short>(i);
            unsigned short* ptr = top_blob.row<unsigned short>(i);

            memcpy(ptr, pf, num_output * sizeof(unsigned short));
            memcpy(ptr + num_output, pr, num_output * sizeof(unsigned short));
        }
    }

    return 0;
}

int RNN_arm::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
        cast_bfloat16_to_float32(bottom_blobs[1], hidden, opt_cast);
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
        int ret = rnn_bf16s(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden, opt);
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
        int ret0 = rnn_bf16s(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), hidden0, opt);
        if (ret0 != 0)
            return ret0;

        Mat hidden1 = hidden.row_range(1, 1);
        int ret1 = rnn_bf16s(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), hidden1, opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i = 0; i < T; i++)
        {
            const unsigned short* pf = top_blob_forward.row<const unsigned short>(i);
            const unsigned short* pr = top_blob_reverse.row<const unsigned short>(i);
            unsigned short* ptr = top_blob.row<unsigned short>(i);

            memcpy(ptr, pf, num_output * sizeof(unsigned short));
            memcpy(ptr + num_output, pr, num_output * sizeof(unsigned short));
        }
    }

    if (top_blobs.size() == 2)
    {
        cast_float32_to_bfloat16(hidden, top_blobs[1], opt);
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
