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

#include "innerproduct_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(InnerProduct_arm)

InnerProduct_arm::InnerProduct_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

int InnerProduct_arm::create_pipeline(const Option& opt)
{
    int num_input = weight_data_size / num_output;

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    // pack4
    if (num_input % 4 == 0 && num_output % 4 == 0)
    {
        // src = inch-outch
        // dst = 4a-4b-inch/4a-outch/4b
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_pack4.create(num_input/4, num_output/4, (size_t)4*16, 16);

            for (int q=0; q+3<num_output; q+=4)
            {
                const float* k0 = weight_data_r2.row(q);
                const float* k1 = weight_data_r2.row(q+1);
                const float* k2 = weight_data_r2.row(q+2);
                const float* k3 = weight_data_r2.row(q+3);

                float* g00 = weight_data_pack4.row(q/4);

                for (int p=0; p+3<num_input; p+=4)
                {
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];

                    g00[4] = k0[1];
                    g00[5] = k1[1];
                    g00[6] = k2[1];
                    g00[7] = k3[1];

                    g00[8] = k0[2];
                    g00[9] = k1[2];
                    g00[10] = k2[2];
                    g00[11] = k3[2];

                    g00[12] = k0[3];
                    g00[13] = k1[3];
                    g00[14] = k2[3];
                    g00[15] = k3[3];

                    k0 += 4;
                    k1 += 4;
                    k2 += 4;
                    k3 += 4;
                    g00 += 16;
                }
            }
        }
    }

    // pack1to4
    if (num_input % 4 != 0 && num_output % 4 == 0)
    {
        // src = inch-outch
        // dst = 4b-inch-outch/4b
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_pack1to4.create(num_input, num_output/4, (size_t)4*4, 4);

            for (int q=0; q+3<num_output; q+=4)
            {
                const float* k0 = weight_data_r2.row(q);
                const float* k1 = weight_data_r2.row(q+1);
                const float* k2 = weight_data_r2.row(q+2);
                const float* k3 = weight_data_r2.row(q+3);

                float* g00 = weight_data_pack1to4.row(q/4);

                for (int p=0; p<num_input; p++)
                {
                    g00[0] = k0[p];
                    g00[1] = k1[p];
                    g00[2] = k2[p];
                    g00[3] = k3[p];

                    g00 += 4;
                }
            }
        }
    }

    // pack4to1
    if (num_input % 4 == 0 && num_output % 4 != 0)
    {
        // src = inch-outch
        // dst = 4a-inch/4a-outch
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_pack4to1.create(num_input/4, num_output, (size_t)4*4, 4);

            for (int q=0; q<num_output; q++)
            {
                const float* k0 = weight_data_r2.row(q);

                float* g00 = weight_data_pack4to1.row(q);

                for (int p=0; p+3<num_input; p+=4)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[1];
                    g00[2] = k0[2];
                    g00[3] = k0[3];

                    k0 += 4;
                    g00 += 4;
                }
            }
        }
    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    return 0;
}

int InnerProduct_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (use_int8_inference)
    {
        // TODO
        return InnerProduct::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    int num_input = bottom_blob.w;

    int out_elempack = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (elempack == 4 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output / out_elempack; p++)
        {
            const float* w = (const float*)weight_data_pack4 + num_input * p * 16;
            const float* m = bottom_blob;

            float32x4_t _sum = vdupq_n_f32(0.f);

            if (bias_term)
            {
                _sum = vld1q_f32(((const float*)bias_data) + p * 4);
            }

            // num_input
            for (int i = 0; i < num_input; i++)
            {
                float32x4_t _val = vld1q_f32( m );

                float32x4_t _w0 = vld1q_f32( w );
                float32x4_t _w1 = vld1q_f32( w + 4 );
                float32x4_t _w2 = vld1q_f32( w + 8 );
                float32x4_t _w3 = vld1q_f32( w + 12 );

#if __aarch64__
                _sum = vmlaq_laneq_f32(_sum, _w0, _val, 0);
                _sum = vmlaq_laneq_f32(_sum, _w1, _val, 1);
                _sum = vmlaq_laneq_f32(_sum, _w2, _val, 2);
                _sum = vmlaq_laneq_f32(_sum, _w3, _val, 3);
#else
                _sum = vmlaq_lane_f32(_sum, _w0, vget_low_f32(_val), 0);
                _sum = vmlaq_lane_f32(_sum, _w1, vget_low_f32(_val), 1);
                _sum = vmlaq_lane_f32(_sum, _w2, vget_high_f32(_val), 0);
                _sum = vmlaq_lane_f32(_sum, _w3, vget_high_f32(_val), 1);
#endif

                w += 16;
                m += 4;
            }

            if (activation_type == 1)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                _sum = vmaxq_f32(_sum, _zero);
            }
            else if (activation_type == 2)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _slope = vdupq_n_f32(activation_params[0]);
                uint32x4_t _lemask = vcleq_f32(_sum, _zero);
                float32x4_t _ps = vmulq_f32(_sum, _slope);
                _sum = vbslq_f32(_lemask, _ps, _sum);
            }
            else if (activation_type == 3)
            {
                float32x4_t _min = vdupq_n_f32(activation_params[0]);
                float32x4_t _max = vdupq_n_f32(activation_params[1]);
                _sum = vmaxq_f32(_sum, _min);
                _sum = vminq_f32(_sum, _max);
            }
            else if (activation_type == 4)
            {
                float32x4_t _one = vdupq_n_f32(1.f);
                _sum = vnegq_f32(_sum);
                _sum = exp_ps(_sum);
                _sum = vaddq_f32(_sum, _one);
                float32x4_t _outp = vrecpeq_f32(_sum);
                _outp = vmulq_f32(vrecpsq_f32(_sum, _outp), _outp);
//                 _outp = vmulq_f32(vrecpsq_f32(_sum, _outp), _outp);
                _sum = _outp;
            }

            float* outptr = top_blob;
            vst1q_f32(outptr + p * 4, _sum);
        }

        return 0;
    }

    if (elempack == 1 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output / out_elempack; p++)
        {
            const float* w = (const float*)weight_data_pack1to4 + num_input * p * 4;
            const float* m = bottom_blob;

            float32x4_t _sum = vdupq_n_f32(0.f);

            if (bias_term)
            {
                _sum = vld1q_f32(((const float*)bias_data) + p * 4);
            }

            // num_input
            for (int i = 0; i < num_input; i++)
            {
                float32x4_t _val = vdupq_n_f32( m[i] );
                float32x4_t _w = vld1q_f32( w );
                _sum = vmlaq_f32(_sum, _val, _w);

                w += 4;
            }

            if (activation_type == 1)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                _sum = vmaxq_f32(_sum, _zero);
            }
            else if (activation_type == 2)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _slope = vdupq_n_f32(activation_params[0]);
                uint32x4_t _lemask = vcleq_f32(_sum, _zero);
                float32x4_t _ps = vmulq_f32(_sum, _slope);
                _sum = vbslq_f32(_lemask, _ps, _sum);
            }
            else if (activation_type == 3)
            {
                float32x4_t _min = vdupq_n_f32(activation_params[0]);
                float32x4_t _max = vdupq_n_f32(activation_params[1]);
                _sum = vmaxq_f32(_sum, _min);
                _sum = vminq_f32(_sum, _max);
            }
            else if (activation_type == 4)
            {
                float32x4_t _one = vdupq_n_f32(1.f);
                _sum = vnegq_f32(_sum);
                _sum = exp_ps(_sum);
                _sum = vaddq_f32(_sum, _one);
                float32x4_t _outp = vrecpeq_f32(_sum);
                _outp = vmulq_f32(vrecpsq_f32(_sum, _outp), _outp);
//                 _outp = vmulq_f32(vrecpsq_f32(_sum, _outp), _outp);
                _sum = _outp;
            }

            float* outptr = top_blob;
            vst1q_f32(outptr + p * 4, _sum);
        }

        return 0;
    }

    if (elempack == 4 && out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output; p++)
        {
            const float* w = (const float*)weight_data_pack4to1 + num_input * p * 4;
            const float* m = bottom_blob;

            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            // num_input
            for (int i = 0; i < num_input; i++)
            {
                float32x4_t _val = vld1q_f32( m );
                float32x4_t _w = vld1q_f32( w );
                float32x4_t _s4 = vmulq_f32(_val, _w);
#if __aarch64__
                sum += vaddvq_f32(_s4); // dot
#else
                float32x2_t _ss = vadd_f32(vget_low_f32(_s4), vget_high_f32(_s4));
                _ss = vpadd_f32(_ss, _ss);
                sum += vget_lane_f32(_ss, 0);
#endif

                w += 4;
                m += 4;
            }

            if (activation_type == 1)
            {
                sum = std::max(sum, 0.f);
            }
            else if (activation_type == 2)
            {
                float slope = activation_params[0];
                sum = sum > 0.f ? sum : sum * slope;
            }
            else if (activation_type == 3)
            {
                float min = activation_params[0];
                float max = activation_params[1];
                if (sum < min)
                    sum = min;
                if (sum > max)
                    sum = max;
            }
            else if (activation_type == 4)
            {
                sum = 1.f / (1.f + exp(-sum));
            }

            top_blob[p] = sum;
        }

        return 0;
    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* weight_data_ptr = weight_data;

    int nn_num_output = num_output >> 2;
    int remain_num_output_start = nn_num_output << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_num_output; pp++)
    {
        int p = pp * 4;

        float sum0 = 0.f;
        float sum1 = 0.f;
        float sum2 = 0.f;
        float sum3 = 0.f;

        if (bias_term)
        {
            sum0 = bias_data[p];
            sum1 = bias_data[p+1];
            sum2 = bias_data[p+2];
            sum3 = bias_data[p+3];
        }

        const float* w0 = weight_data_ptr + size * channels * p;
        const float* w1 = weight_data_ptr + size * channels * (p+1);
        const float* w2 = weight_data_ptr + size * channels * (p+2);
        const float* w3 = weight_data_ptr + size * channels * (p+3);

#if __ARM_NEON
        float32x4_t _sum0 = vdupq_n_f32(0.f);
        float32x4_t _sum1 = vdupq_n_f32(0.f);
        float32x4_t _sum2 = vdupq_n_f32(0.f);
        float32x4_t _sum3 = vdupq_n_f32(0.f);
#endif // __ARM_NEON

        // channels
        for (int q=0; q<channels; q++)
        {
            const float* m = bottom_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size & 3;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            for (; nn>0; nn--)
            {
                float32x4_t _m = vld1q_f32(m);

                float32x4_t _w0 = vld1q_f32(w0);
                _sum0 = vmlaq_f32(_sum0, _m, _w0);

                float32x4_t _w1 = vld1q_f32(w1);
                _sum1 = vmlaq_f32(_sum1, _m, _w1);

                float32x4_t _w2 = vld1q_f32(w2);
                _sum2 = vmlaq_f32(_sum2, _m, _w2);

                float32x4_t _w3 = vld1q_f32(w3);
                _sum3 = vmlaq_f32(_sum3, _m, _w3);

                m += 4;
                w0 += 4;
                w1 += 4;
                w2 += 4;
                w3 += 4;
            }
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                sum0 += *m * *w0;
                sum1 += *m * *w1;
                sum2 += *m * *w2;
                sum3 += *m * *w3;

                m++;
                w0++;
                w1++;
                w2++;
                w3++;
            }

        }

#if __ARM_NEON
        float32x2_t _sum0ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
        float32x2_t _sum1ss = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
        float32x2_t _sum2ss = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
        float32x2_t _sum3ss = vadd_f32(vget_low_f32(_sum3), vget_high_f32(_sum3));

        float32x2_t _sum01ss = vpadd_f32(_sum0ss, _sum1ss);
        float32x2_t _sum23ss = vpadd_f32(_sum2ss, _sum3ss);

        sum0 += vget_lane_f32(_sum01ss, 0);
        sum1 += vget_lane_f32(_sum01ss, 1);
        sum2 += vget_lane_f32(_sum23ss, 0);
        sum3 += vget_lane_f32(_sum23ss, 1);

#endif // __ARM_NEON

        if (activation_type == 1)
        {
            sum0 = std::max(sum0, 0.f);
            sum1 = std::max(sum1, 0.f);
            sum2 = std::max(sum2, 0.f);
            sum3 = std::max(sum3, 0.f);
        }
        else if (activation_type == 2)
        {
            float slope = activation_params[0];
            sum0 = sum0 > 0.f ? sum0 : sum0 * slope;
            sum1 = sum1 > 0.f ? sum1 : sum1 * slope;
            sum2 = sum2 > 0.f ? sum2 : sum2 * slope;
            sum3 = sum3 > 0.f ? sum3 : sum3 * slope;
        }
        else if (activation_type == 3)
        {
            float min = activation_params[0];
            float max = activation_params[1];
            if (sum0 < min) sum0 = min;
            if (sum0 > max) sum0 = max;
            if (sum1 < min) sum1 = min;
            if (sum1 > max) sum1 = max;
            if (sum2 < min) sum2 = min;
            if (sum2 > max) sum2 = max;
            if (sum3 < min) sum3 = min;
            if (sum3 > max) sum3 = max;
        }

        top_blob[p] = sum0;
        top_blob[p+1] = sum1;
        top_blob[p+2] = sum2;
        top_blob[p+3] = sum3;
    }

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_num_output_start; p<num_output; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        const float* w = weight_data_ptr + size * channels * p;

#if __ARM_NEON
        float32x4_t _sum = vdupq_n_f32(0.f);
        float32x4_t _sum2 = vdupq_n_f32(0.f);
#endif // __ARM_NEON

        // channels
        for (int q=0; q<channels; q++)
        {
            const float* m = bottom_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                "0:                                   \n"
                "prfm       pldl1keep, [%1, #256]     \n"
                "ld1        {v0.4s, v1.4s}, [%1], #32 \n"
                "prfm       pldl1keep, [%2, #256]     \n"
                "ld1        {v2.4s, v3.4s}, [%2], #32 \n"
                "fmla       %3.4s, v0.4s, v2.4s       \n"
                "subs       %w0, %w0, #1              \n"
                "fmla       %4.4s, v1.4s, v3.4s       \n"
                "bne        0b                        \n"
                : "=r"(nn),     // %0
                  "=r"(m),      // %1
                  "=r"(w),      // %2
                  "=w"(_sum),   // %3
                  "=w"(_sum2)   // %4
                : "0"(nn),
                  "1"(m),
                  "2"(w),
                  "3"(_sum),
                  "4"(_sum2)
                : "cc", "memory", "v0", "v1", "v2", "v3"
            );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.f32   {d0-d3}, [%1 :128]! \n"
                "pld        [%2, #256]          \n"
                "vld1.f32   {d4-d7}, [%2]!      \n"
                "vmla.f32   %q3, q0, q2         \n"
                "subs       %0, #1              \n"
                "vmla.f32   %q4, q1, q3         \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(m),      // %1
                  "=r"(w),      // %2
                  "=w"(_sum),   // %3
                  "=w"(_sum2)   // %4
                : "0"(nn),
                  "1"(m),
                  "2"(w),
                  "3"(_sum),
                  "4"(_sum2)
                : "cc", "memory", "q0", "q1", "q2", "q3"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                sum += *m * *w;

                m++;
                w++;
            }
        }

#if __ARM_NEON
        _sum = vaddq_f32(_sum, _sum2);
#if __aarch64__
        sum += vaddvq_f32(_sum);
#else
        float32x2_t _sumss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
        _sumss = vpadd_f32(_sumss, _sumss);
        sum += vget_lane_f32(_sumss, 0);
#endif // __aarch64__
#endif // __ARM_NEON

        if (activation_type == 1)
        {
            sum = std::max(sum, 0.f);
        }
        else if (activation_type == 2)
        {
            float slope = activation_params[0];
            sum = sum > 0.f ? sum : sum * slope;
        }
        else if (activation_type == 3)
        {
            float min = activation_params[0];
            float max = activation_params[1];
            if (sum < min)
                sum = min;
            if (sum > max)
                sum = max;
        }
        else if (activation_type == 4)
        {
            sum = 1.f / (1.f + exp(-sum));
        }

        top_blob[p] = sum;
    }

    return 0;
}

} // namespace ncnn
