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

namespace ncnn {

InstanceNorm_arm::InstanceNorm_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int InstanceNorm_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
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

            a = (float)(gamma / (sqrt(var + eps)));
            b = (float)(-mean * a + beta);
        }
        else
        {
            a = (float)(1.f / (sqrt(var + eps)));
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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int InstanceNorm_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

    if (elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr0 = bottom_top_blob.channel(q);

            float32x4_t _div_size = vdupq_n_f32(1.f / size);
            float32x4_t _eps = vdupq_n_f32(eps);

            // mean and var
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sqsum0 = vdupq_n_f32(0.f);
            float32x4_t _sqsum1 = vdupq_n_f32(0.f);
            const __fp16* ptr = ptr0;
            for (int i = 0; i < size; i++)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
                float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
                _sum0 = vaddq_f32(_sum0, _p0);
                _sum1 = vaddq_f32(_sum1, _p1);
                ptr += 8;
                //sqsum += ptr[i] * ptr[i];
            }
            float32x4_t _mean0 = vmulq_f32(_sum0, _div_size);
            float32x4_t _mean1 = vmulq_f32(_sum1, _div_size);
            ptr = ptr0;
            for (int i = 0; i < size; i++)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
                float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
                float32x4_t _tmp0 = vsubq_f32(_p0, _mean0);
                float32x4_t _tmp1 = vsubq_f32(_p1, _mean1);
                _sqsum0 = vfmaq_f32(_sqsum0, _tmp0, _tmp0);
                _sqsum1 = vfmaq_f32(_sqsum1, _tmp1, _tmp1);
                ptr += 8;
            }
            float32x4_t _var_eps0 = vfmaq_f32(_eps, _sqsum0, _div_size);
            float32x4_t _var_eps1 = vfmaq_f32(_eps, _sqsum1, _div_size);
            // the var maybe minus due to accuracy
            //float var = sqsum / size - mean * mean;

            float32x4_t _reciprocal0 = vrsqrteq_f32(_var_eps0);
            float32x4_t _reciprocal1 = vrsqrteq_f32(_var_eps1);
            _reciprocal0 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps0, _reciprocal0), _reciprocal0), _reciprocal0);
            _reciprocal1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps1, _reciprocal1), _reciprocal1), _reciprocal1);
            // _reciprocal0 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps0, _reciprocal0), _reciprocal0), _reciprocal0);
            // _reciprocal1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps1, _reciprocal1), _reciprocal1), _reciprocal1);

            float16x8_t _a;
            float16x8_t _b;
            if (affine)
            {
                float32x4_t _gamma0 = vld1q_f32((const float*)gamma_data + q * 8);
                float32x4_t _gamma1 = vld1q_f32((const float*)gamma_data + q * 8 + 4);
                float32x4_t _beta0 = vld1q_f32((const float*)beta_data + q * 8);
                float32x4_t _beta1 = vld1q_f32((const float*)beta_data + q * 8 + 4);

                float32x4_t _a320 = vmulq_f32(_gamma0, _reciprocal0);
                float32x4_t _a321 = vmulq_f32(_gamma1, _reciprocal1);
                float16x4_t _a0 = vcvt_f16_f32(_a320);
                float16x4_t _a1 = vcvt_f16_f32(_a321);
                float16x4_t _b0 = vcvt_f16_f32(vmlsq_f32(_beta0, _mean0, _a320));
                float16x4_t _b1 = vcvt_f16_f32(vmlsq_f32(_beta1, _mean1, _a321));

                _a = vcombine_f16(_a0, _a1);
                _b = vcombine_f16(_b0, _b1);
            }
            else
            {
                float16x4_t _a0 = vcvt_f16_f32(_reciprocal0);
                float16x4_t _a1 = vcvt_f16_f32(_reciprocal1);
                float16x4_t _b0 = vcvt_f16_f32(vnegq_f32(vmulq_f32(_mean0, _reciprocal0)));
                float16x4_t _b1 = vcvt_f16_f32(vnegq_f32(vmulq_f32(_mean1, _reciprocal1)));

                _a = vcombine_f16(_a0, _a1);
                _b = vcombine_f16(_b0, _b1);
            }

            for (int i = 0; i < size; i++)
            {
                float16x8_t _p = vld1q_f16(ptr0);
                _p = vfmaq_f16(_b, _p, _a);
                vst1q_f16(ptr0, _p);
                ptr0 += 8;
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr0 = bottom_top_blob.channel(q);

            float32x4_t _div_size = vdupq_n_f32(1.f / size);

            // mean and var
            float32x4_t _sum = vdupq_n_f32(0.f);
            float32x4_t _sqsum = vdupq_n_f32(0.f);
            const __fp16* ptr = ptr0;
            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                _sum = vaddq_f32(_sum, _p);
                ptr += 4;
                //sqsum += ptr[i] * ptr[i];
            }
            float32x4_t _mean = vmulq_f32(_sum, _div_size);
            ptr = ptr0;
            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                float32x4_t _tmp = vsubq_f32(_p, _mean);
                _sqsum = vfmaq_f32(_sqsum, _tmp, _tmp);
                ptr += 4;
            }
            float32x4_t _var_eps = vfmaq_f32(vdupq_n_f32(eps), _sqsum, _div_size);
            // the var maybe minus due to accuracy
            //float var = sqsum / size - mean * mean;

            float32x4_t _reciprocal = vrsqrteq_f32(_var_eps);
            _reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps, _reciprocal), _reciprocal), _reciprocal);
            // _reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps, _reciprocal), _reciprocal), _reciprocal);

            float16x4_t _a;
            float16x4_t _b;
            if (affine)
            {
                float32x4_t _gamma = vld1q_f32((const float*)gamma_data + q * 4);
                float32x4_t _beta = vld1q_f32((const float*)beta_data + q * 4);

                float32x4_t _a32 = vmulq_f32(_gamma, _reciprocal);
                _a = vcvt_f16_f32(_a32);
                _b = vcvt_f16_f32(vmlsq_f32(_beta, _mean, _a32));
            }
            else
            {
                _a = vcvt_f16_f32(_reciprocal);
                _b = vcvt_f16_f32(vnegq_f32(vmulq_f32(_mean, _reciprocal)));
            }

            for (int i = 0; i < size; i++)
            {
                float16x4_t _p = vld1_f16(ptr0);
                _p = vfma_f16(_b, _p, _a);
                vst1_f16(ptr0, _p);
                ptr0 += 4;
            }
        }

        return 0;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr0 = bottom_top_blob.channel(q);

        // mean and var
        float sum = 0.f;
        float sqsum = 0.f;
        const __fp16* ptr = ptr0;
        int i = 0;
#if __ARM_NEON
        float32x4_t _sum = vdupq_n_f32(0.f);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            _sum = vaddq_f32(_sum, _p);
            ptr += 4;
        }
        sum = vaddvq_f32(_sum);
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
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            float32x4_t _tmp = vsubq_f32(_p, _mean);
            _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
            ptr += 4;
        }
        sqsum = vaddvq_f32(_sqsum);
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float tmp = *ptr++ - mean;
            sqsum += tmp * tmp;
        }
        float var = sqsum / size;
        // the var maybe minus due to accuracy
        //float var = sqsum / size - mean * mean;

        __fp16 a;
        __fp16 b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = (__fp16)(gamma / (sqrt(var + eps)));
            b = (__fp16)(-mean * a + beta);
        }
        else
        {
            a = (__fp16)(1.f / (sqrt(var + eps)));
            b = (__fp16)(-mean * a);
        }

        i = 0;
#if __ARM_NEON
        float16x8_t _a = vdupq_n_f16(a);
        float16x8_t _b = vdupq_n_f16(b);
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr0);
            _p = vfmaq_f16(_b, _p, _a);
            vst1q_f16(ptr0, _p);
            ptr0 += 8;
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
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

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
                float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
                _sum = vaddq_f32(_sum, _p);
                ptr += 4;
                //sqsum += ptr[i] * ptr[i];
            }
            float32x4_t _mean = vmulq_f32(_sum, _div_size);
            ptr = ptr0;
            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
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
                float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr0));
                _p = vmlaq_f32(_b, _p, _a);
                vst1_u16(ptr0, vcvt_bf16_f32(_p));
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
            float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
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
            float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
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

            a = (float)(gamma / (sqrt(var + eps)));
            b = (float)(-mean * a + beta);
        }
        else
        {
            a = (float)(1.f / (sqrt(var + eps)));
            b = (float)(-mean * a);
        }

        i = 0;
#if __ARM_NEON
        float32x4_t _a = vdupq_n_f32(a);
        float32x4_t _b = vdupq_n_f32(b);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr0));
            _p = vmlaq_f32(_b, _p, _a);
            vst1_u16(ptr0, vcvt_bf16_f32(_p));
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
