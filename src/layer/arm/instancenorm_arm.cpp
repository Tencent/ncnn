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

#include "instancenorm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

InstanceNorm_arm::InstanceNorm_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int InstanceNorm_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = bottom_top_blob.channel(q);

            float32x4_t _div_size = vdupq_n_f32(1.f / size);

            // mean and var
            float32x4_t _sum = vdupq_n_f32(0.f);
            float32x4_t _sqsum = vdupq_n_f32(0.f);
            const float* ptr = ptr0;
            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _sum = vaddq_f32(_sum, _p);
                ptr += 4;
                //sqsum += ptr[i] * ptr[i];
            }
            float32x4_t _mean = vmulq_f32(_sum, _div_size);
            ptr = ptr0;
            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _tmp = vsubq_f32(_p, _mean);
                _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                ptr += 4;
            }
            float32x4_t _var_eps = vmlaq_f32(vdupq_n_f32(eps), _sqsum, _div_size);
            // the var maybe minus due to accuracy
            //float var = sqsum / size - mean * mean;

            float32x4_t _reciprocal = vrsqrteq_f32(_var_eps);
            _reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps, _reciprocal), _reciprocal), _reciprocal);
            // _reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps, _reciprocal), _reciprocal), _reciprocal);

            float32x4_t _a;
            float32x4_t _b;
            if (affine)
            {
                float32x4_t _gamma = vld1q_f32((const float*)gamma_data + q * 4);
                float32x4_t _beta = vld1q_f32((const float*)beta_data + q * 4);

                _a = vmulq_f32(_gamma, _reciprocal);
                _b = vmlsq_f32(_beta, _mean, _a);
            }
            else
            {
                _a = _reciprocal;
                _b = vnegq_f32(vmulq_f32(_mean, _a));
            }

            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = vld1q_f32(ptr0);
                _p = vmlaq_f32(_b, _p, _a);
                vst1q_f32(ptr0, _p);
                ptr0 += 4;
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr0 = bottom_top_blob.channel(q);

        // mean and var
        float sum = 0.f;
        float sqsum = 0.f;
        const float* ptr = ptr0;
        int i = 0;
#if __ARM_NEON
        float32x4_t _sum = vdupq_n_f32(0.f);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _sum = vaddq_f32(_sum, _p);
            ptr += 4;
        }
#if __aarch64__
        sum = vaddvq_f32(_sum);
#else
        float32x2_t _s2 = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
        _s2 = vpadd_f32(_s2, _s2);
        sum = vget_lane_f32(_s2, 0);
#endif
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            sum += *ptr++;
            //sqsum += ptr[i] * ptr[i];
        }
        float mean = sum / size;
        ptr = ptr0;
        i = 0;
#if __ARM_NEON
        float32x4_t _sqsum = vdupq_n_f32(0.f);
        float32x4_t _mean = vdupq_n_f32(mean);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _tmp = vsubq_f32(_p, _mean);
            _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
            ptr += 4;
        }
#if __aarch64__
        sqsum = vaddvq_f32(_sqsum);
#else
        float32x2_t _sq2 = vadd_f32(vget_low_f32(_sqsum), vget_high_f32(_sqsum));
        _sq2 = vpadd_f32(_sq2, _sq2);
        sqsum = vget_lane_f32(_sq2, 0);
#endif
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float tmp = *ptr++ - mean;
            sqsum += tmp * tmp;
        }
        float var = sqsum / size;
        // the var maybe minus due to accuracy
        //float var = sqsum / size - mean * mean;

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = (float)(gamma / (sqrtf(var + eps)));
            b = (float)(-mean * a + beta);
        }
        else
        {
            a = (float)(1.f / (sqrtf(var + eps)));
            b = (float)(-mean * a);
        }

        i = 0;
#if __ARM_NEON
        float32x4_t _a = vdupq_n_f32(a);
        float32x4_t _b = vdupq_n_f32(b);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr0);
            _p = vmlaq_f32(_b, _p, _a);
            vst1q_f32(ptr0, _p);
            ptr0 += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr0 = *ptr0 * a + b;
            ptr0++;
        }
    }

    return 0;
}

#if NCNN_BF16
int InstanceNorm_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr0 = bottom_top_blob.channel(q);

            float32x4_t _div_size = vdupq_n_f32(1.f / size);

            // mean and var
            float32x4_t _sum = vdupq_n_f32(0.f);
            float32x4_t _sqsum = vdupq_n_f32(0.f);
            const unsigned short* ptr = ptr0;
            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                _sum = vaddq_f32(_sum, _p);
                ptr += 4;
                //sqsum += ptr[i] * ptr[i];
            }
            float32x4_t _mean = vmulq_f32(_sum, _div_size);
            ptr = ptr0;
            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                float32x4_t _tmp = vsubq_f32(_p, _mean);
                _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                ptr += 4;
            }
            float32x4_t _var_eps = vmlaq_f32(vdupq_n_f32(eps), _sqsum, _div_size);
            // the var maybe minus due to accuracy
            //float var = sqsum / size - mean * mean;

            float32x4_t _reciprocal = vrsqrteq_f32(_var_eps);
            _reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps, _reciprocal), _reciprocal), _reciprocal);
            // _reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps, _reciprocal), _reciprocal), _reciprocal);

            float32x4_t _a;
            float32x4_t _b;
            if (affine)
            {
                float32x4_t _gamma = vld1q_f32((const float*)gamma_data + q * 4);
                float32x4_t _beta = vld1q_f32((const float*)beta_data + q * 4);

                _a = vmulq_f32(_gamma, _reciprocal);
                _b = vmlsq_f32(_beta, _mean, _a);
            }
            else
            {
                _a = _reciprocal;
                _b = vnegq_f32(vmulq_f32(_mean, _a));
            }

            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr0));
                _p = vmlaq_f32(_b, _p, _a);
                vst1_u16(ptr0, float2bfloat(_p));
                ptr0 += 4;
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr0 = bottom_top_blob.channel(q);

        // mean and var
        float sum = 0.f;
        float sqsum = 0.f;
        const unsigned short* ptr = ptr0;
        int i = 0;
#if __ARM_NEON
        float32x4_t _sum = vdupq_n_f32(0.f);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            _sum = vaddq_f32(_sum, _p);
            ptr += 4;
        }
#if __aarch64__
        sum = vaddvq_f32(_sum);
#else
        float32x2_t _s2 = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
        _s2 = vpadd_f32(_s2, _s2);
        sum = vget_lane_f32(_s2, 0);
#endif
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            sum += bfloat16_to_float32(*ptr++);
            //sqsum += ptr[i] * ptr[i];
        }
        float mean = sum / size;
        ptr = ptr0;
        i = 0;
#if __ARM_NEON
        float32x4_t _sqsum = vdupq_n_f32(0.f);
        float32x4_t _mean = vdupq_n_f32(mean);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            float32x4_t _tmp = vsubq_f32(_p, _mean);
            _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
            ptr += 4;
        }
#if __aarch64__
        sqsum = vaddvq_f32(_sqsum);
#else
        float32x2_t _sq2 = vadd_f32(vget_low_f32(_sqsum), vget_high_f32(_sqsum));
        _sq2 = vpadd_f32(_sq2, _sq2);
        sqsum = vget_lane_f32(_sq2, 0);
#endif
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float tmp = bfloat16_to_float32(*ptr++) - mean;
            sqsum += tmp * tmp;
        }
        float var = sqsum / size;
        // the var maybe minus due to accuracy
        //float var = sqsum / size - mean * mean;

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = (float)(gamma / (sqrtf(var + eps)));
            b = (float)(-mean * a + beta);
        }
        else
        {
            a = (float)(1.f / (sqrtf(var + eps)));
            b = (float)(-mean * a);
        }

        i = 0;
#if __ARM_NEON
        float32x4_t _a = vdupq_n_f32(a);
        float32x4_t _b = vdupq_n_f32(b);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr0));
            _p = vmlaq_f32(_b, _p, _a);
            vst1_u16(ptr0, float2bfloat(_p));
            ptr0 += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr0 = float32_to_bfloat16(bfloat16_to_float32(*ptr0) * a + b);
            ptr0++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
