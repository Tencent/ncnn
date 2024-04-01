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
#include "arm_usability.h"

#include "cpu.h"

namespace ncnn {

RNN_arm::RNN_arm()
{
#if __ARM_NEON
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int RNN_arm::create_pipeline(const Option& opt)
{
#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage)
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

    weight_xc_data.release();
    bias_c_data.release();
    weight_hc_data.release();

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

        int remain_num_output_start = 0;
#if __ARM_NEON
        int nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            const float* weight_xc_ptr = weight_xc.row(q / 4);
            const float* weight_hc_ptr = weight_hc.row(q / 4);

            float32x4_t _rnn_H = vld1q_f32((const float*)bias_c + q);
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
                _rnn_H = vfmaq_laneq_f32(_rnn_H, _weight_xc, _x, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_xc_3, _x, 3);
#else
                _rnn_H = vmlaq_lane_f32(_rnn_H, _weight_xc, vget_low_f32(_x), 0);
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
                _rnn_H = vmlaq_f32(_rnn_H, _weight_xc, _x);

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
                _rnn_H = vfmaq_laneq_f32(_rnn_H, _weight_hc, _hidden_state, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_hc_3, _hidden_state, 3);
#else
                _rnn_H = vmlaq_lane_f32(_rnn_H, _weight_hc, vget_low_f32(_hidden_state), 0);
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
                _rnn_H = vmlaq_f32(_rnn_H, _weight_hc, _hidden_state);

                weight_hc_ptr += 4;
            }

            _rnn_H = vaddq_f32(_rnn_H, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _rnn_H = vaddq_f32(_rnn_H, _sum2);

            _rnn_H = tanh_ps(_rnn_H);

            vst1q_f32((float*)gates + q, _rnn_H);
        }
#endif // __ARM_NEON
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
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

            H = tanhf(H);

            gates[q] = H;
        }

        float* output_data = top_blob.row(ti);

        float* hidden_ptr = hidden_state;

#if __ARM_NEON
        nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            float32x4_t _rnn_H = vld1q_f32((float*)gates + q);

            vst1q_f32(hidden_ptr + q, _rnn_H);
            vst1q_f32(output_data + q, _rnn_H);
        }
#endif // __ARM_NEON
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            float H = gates[q];

            hidden_ptr[q] = H;
            output_data[q] = H;
        }
    }

    return 0;
}

int RNN_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
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

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
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

        int remain_num_output_start = 0;
#if __ARM_NEON
        int nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            const unsigned short* weight_xc_ptr = weight_xc.row<const unsigned short>(q / 4);
            const unsigned short* weight_hc_ptr = weight_hc.row<const unsigned short>(q / 4);

            float32x4_t _rnn_H = bfloat2float(vld1_u16((const unsigned short*)bias_c + q));
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _x = bfloat2float(vld1_u16(x + i));
                float32x4_t _weight_xc = bfloat2float(vld1_u16(weight_xc_ptr));
                float32x4_t _weight_xc_1 = bfloat2float(vld1_u16(weight_xc_ptr + 4));
                float32x4_t _weight_xc_2 = bfloat2float(vld1_u16(weight_xc_ptr + 8));
                float32x4_t _weight_xc_3 = bfloat2float(vld1_u16(weight_xc_ptr + 12));
#if __aarch64__
                _rnn_H = vfmaq_laneq_f32(_rnn_H, _weight_xc, _x, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_xc_3, _x, 3);
#else
                _rnn_H = vmlaq_lane_f32(_rnn_H, _weight_xc, vget_low_f32(_x), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _weight_xc_1, vget_low_f32(_x), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _weight_xc_2, vget_high_f32(_x), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _weight_xc_3, vget_high_f32(_x), 1);
#endif

                weight_xc_ptr += 16;
            }
            for (; i < size; i++)
            {
                float32x4_t _x = bfloat2float(vdup_n_u16(x[i]));
                float32x4_t _weight_xc = bfloat2float(vld1_u16(weight_xc_ptr));
                _rnn_H = vmlaq_f32(_rnn_H, _weight_xc, _x);

                weight_xc_ptr += 4;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float32x4_t _hidden_state = vld1q_f32((const float*)hidden_state + i);
                float32x4_t _weight_hc = bfloat2float(vld1_u16(weight_hc_ptr));
                float32x4_t _weight_hc_1 = bfloat2float(vld1_u16(weight_hc_ptr + 4));
                float32x4_t _weight_hc_2 = bfloat2float(vld1_u16(weight_hc_ptr + 8));
                float32x4_t _weight_hc_3 = bfloat2float(vld1_u16(weight_hc_ptr + 12));
#if __aarch64__
                _rnn_H = vfmaq_laneq_f32(_rnn_H, _weight_hc, _hidden_state, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_hc_3, _hidden_state, 3);
#else
                _rnn_H = vmlaq_lane_f32(_rnn_H, _weight_hc, vget_low_f32(_hidden_state), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _weight_hc_1, vget_low_f32(_hidden_state), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _weight_hc_2, vget_high_f32(_hidden_state), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _weight_hc_3, vget_high_f32(_hidden_state), 1);
#endif

                weight_hc_ptr += 16;
            }
            for (; i < num_output; i++)
            {
                float32x4_t _hidden_state = vdupq_n_f32(hidden_state[i]);
                float32x4_t _weight_hc = bfloat2float(vld1_u16(weight_hc_ptr));
                _rnn_H = vmlaq_f32(_rnn_H, _weight_hc, _hidden_state);

                weight_hc_ptr += 4;
            }

            _rnn_H = vaddq_f32(_rnn_H, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _rnn_H = vaddq_f32(_rnn_H, _sum2);

            _rnn_H = tanh_ps(_rnn_H);

            vst1q_f32((float*)gates + q, _rnn_H);
        }
#endif // __ARM_NEON
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
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

            H = tanhf(H);

            gates[q] = H;
        }

        unsigned short* output_data = top_blob.row<unsigned short>(ti);

        float* hidden_ptr = hidden_state;

#if __ARM_NEON
        nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            float32x4_t _rnn_H = vld1q_f32((float*)gates + q);

            vst1q_f32(hidden_ptr + q, _rnn_H);
            vst1_u16(output_data + q, float2bfloat(_rnn_H));
        }
#endif // __ARM_NEON
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            float H = gates[q];

            hidden_ptr[q] = H;
            output_data[q] = float32_to_bfloat16(H);
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

    weight_xc_data.release();
    bias_c_data.release();
    weight_hc_data.release();

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
