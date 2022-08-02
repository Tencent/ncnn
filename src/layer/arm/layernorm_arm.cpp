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

#include "layernorm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

LayerNorm_arm::LayerNorm_arm()
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

int LayerNorm_arm::create_pipeline(const Option& opt)
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

    return 0;
}

#if __ARM_NEON
inline float sum_float32x4(float32x4_t _sum)
{
    float sum = 0.f;
#if __aarch64__
    sum += vaddvq_f32(_sum);
#else
    float32x2_t _sum2 = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
    float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
    sum += vget_lane_f32(_ss2, 0);
#endif
    return sum;
}
#endif

int LayerNorm_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;
    int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_inplace_fp16sa(bottom_top_blob, opt);
        else
            return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

#if __ARM_NEON
    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;
            float* ptr0 = bottom_top_blob;

            // sum
            float sum = 0.f;
            float32x4_t _sum = vdupq_n_f32(0.f);
            const float* ptr = ptr0;
            for (int i = 0; i < w; i++)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _sum = vaddq_f32(_sum, _p);
                ptr += 4;
            }
            sum += sum_float32x4(_sum);

            // mean
            float mean = 0.25 * sum / w;
            float32x4_t _mean = vdupq_n_f32(mean);

            // sum
            ptr = ptr0;
            float sqsum = 0.f;
            float32x4_t _sqsum = vdupq_n_f32(0.f);
            for (int i = 0; i < w; i++)
            {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _tmp = vsubq_f32(_p, _mean);
                _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                ptr += 4;
            }
            sqsum += sum_float32x4(_sqsum);

            // var
            float var = 0.25 * sqsum / w;

            // coefficient
            float a = static_cast<float>(1.f / (sqrt(var + eps)));
            float b = -mean * a;
            float32x4_t _a = vdupq_n_f32(a);
            float32x4_t _b = vdupq_n_f32(b);

            // affine
            if (affine)
            {
                const float* gptr = (const float*)gamma_data;
                const float* bptr = (const float*)beta_data;

                for (int i = 0; i < w; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr0);
                    _p = vmlaq_f32(_b, _p, _a);
                    float32x4_t _gamma = vld1q_f32(gptr);
                    float32x4_t _beta = vld1q_f32(bptr);
                    _p = vmlaq_f32(_beta, _p, _gamma);
                    vst1q_f32(ptr0, _p);
                    ptr0 += 4;
                    gptr += 4;
                    bptr += 4;
                }
            }
            else
            {
                for (int i = 0; i < w; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr0);
                    _p = vmlaq_f32(_b, _p, _a);
                    vst1q_f32(ptr0, _p);
                    ptr0 += 4;
                }
            }

            return 0;
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            float32x4_t _div_size = vdupq_n_f32(1.f / w);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr0 = bottom_top_blob.row(i);

                // sum
                float32x4_t _sum = vdupq_n_f32(0.f);
                const float* ptr = ptr0;
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    _sum = vaddq_f32(_sum, _p);
                    ptr += 4;
                }

                // mean
                float32x4_t _mean = vmulq_f32(_sum, _div_size);

                // sum
                ptr = ptr0;
                float32x4_t _sqsum = vdupq_n_f32(0.f);
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _tmp = vsubq_f32(_p, _mean);
                    _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                    ptr += 4;
                }

                // var
                float32x4_t _var_eps = vmlaq_f32(vdupq_n_f32(eps), _sqsum, _div_size);

                // coefficient
                float32x4_t _a = vrsqrteq_f32(_var_eps);
                _a = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps, _a), _a), _a);
                float32x4_t _b = vmlsq_f32(vdupq_n_f32(0.f), _mean, _a);

                // affine
                if (affine)
                {
                    const float* gptr = (const float*)gamma_data;
                    const float* bptr = (const float*)beta_data;

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        _p = vmlaq_f32(_b, _p, _a);
                        float32x4_t _gamma = vdupq_n_f32(*gptr);
                        float32x4_t _beta = vdupq_n_f32(*bptr);
                        _p = vmlaq_f32(_beta, _p, _gamma);
                        vst1q_f32(ptr0, _p);
                        ptr0 += 4;
                        gptr += 1;
                        bptr += 1;
                    }
                }
                else
                {
                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        _p = vmlaq_f32(_b, _p, _a);
                        vst1q_f32(ptr0, _p);
                        ptr0 += 4;
                    }
                }
            }

            return 0;
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            float32x4_t _div_size = vdupq_n_f32(1.f / affine_size);

            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr0 = bottom_top_blob.channel(q).row(i);

                        // sum
                        float32x4_t _sum = vdupq_n_f32(0.f);
                        const float* ptr = ptr0;
                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _p = vld1q_f32(ptr);
                            _sum = vaddq_f32(_sum, _p);
                            ptr += 4;
                        }

                        // mean
                        float32x4_t _mean = vmulq_f32(_sum, _div_size);

                        // sum
                        ptr = ptr0;
                        float32x4_t _sqsum = vdupq_n_f32(0.f);
                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _p = vld1q_f32(ptr);
                            float32x4_t _tmp = vsubq_f32(_p, _mean);
                            _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                            ptr += 4;
                        }

                        // var
                        float32x4_t _var_eps = vmlaq_f32(vdupq_n_f32(eps), _sqsum, _div_size);

                        // coefficient
                        float32x4_t _a = vrsqrteq_f32(_var_eps);
                        _a = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps, _a), _a), _a);
                        float32x4_t _b = vmlsq_f32(vdupq_n_f32(0.f), _mean, _a);

                        // affine
                        if (affine)
                        {
                            const float* gptr = (const float*)gamma_data;
                            const float* bptr = (const float*)beta_data;

                            for (int j = 0; j < w; j++)
                            {
                                float32x4_t _p = vld1q_f32(ptr0);
                                _p = vmlaq_f32(_b, _p, _a);
                                float32x4_t _gamma = vdupq_n_f32(*gptr);
                                float32x4_t _beta = vdupq_n_f32(*bptr);
                                _p = vmlaq_f32(_beta, _p, _gamma);
                                vst1q_f32(ptr0, _p);
                                ptr0 += 4;
                                gptr += 1;
                                bptr += 1;
                            }
                        }
                        else
                        {
                            for (int j = 0; j < w; j++)
                            {
                                float32x4_t _p = vld1q_f32(ptr0);
                                _p = vmlaq_f32(_b, _p, _a);
                                vst1q_f32(ptr0, _p);
                                ptr0 += 4;
                            }
                        }
                    }
                }

                return 0;
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr0 = bottom_top_blob.channel(q);

                    // sum
                    float32x4_t _sum = vdupq_n_f32(0.f);
                    const float* ptr = ptr0;
                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _sum = vaddq_f32(_sum, _p);
                        ptr += 4;
                    }

                    // mean
                    float32x4_t _mean = vmulq_f32(_sum, _div_size);

                    // sum
                    ptr = ptr0;
                    float32x4_t _sqsum = vdupq_n_f32(0.f);
                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        float32x4_t _tmp = vsubq_f32(_p, _mean);
                        _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                        ptr += 4;
                    }

                    // var
                    float32x4_t _var_eps = vmlaq_f32(vdupq_n_f32(eps), _sqsum, _div_size);

                    // coefficient
                    float32x4_t _a = vrsqrteq_f32(_var_eps);
                    _a = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps, _a), _a), _a);
                    float32x4_t _b = vmlsq_f32(vdupq_n_f32(0.f), _mean, _a);

                    // affine
                    if (affine)
                    {
                        const float* gptr = (const float*)gamma_data;
                        const float* bptr = (const float*)beta_data;

                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _p = vld1q_f32(ptr0);
                            _p = vmlaq_f32(_b, _p, _a);
                            float32x4_t _gamma = vdupq_n_f32(*gptr);
                            float32x4_t _beta = vdupq_n_f32(*bptr);
                            _p = vmlaq_f32(_beta, _p, _gamma);
                            vst1q_f32(ptr0, _p);
                            ptr0 += 4;
                            gptr += 1;
                            bptr += 1;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _p = vld1q_f32(ptr0);
                            _p = vmlaq_f32(_b, _p, _a);
                            vst1q_f32(ptr0, _p);
                            ptr0 += 4;
                        }
                    }
                }

                return 0;
            }
        }
    }

    if (elempack == 1)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;
            float* ptr0 = bottom_top_blob;

            // sum
            float sum = 0.f;
            float32x4_t _sum = vdupq_n_f32(0.f);
            const float* ptr = ptr0;
            int i = 0;
            for (; i + 3 < w; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _sum = vaddq_f32(_sum, _p);
                ptr += 4;
            }
            sum += sum_float32x4(_sum);
            for (; i < w; i++)
            {
                sum += *ptr;
                ptr++;
            }

            // mean
            float mean = sum / w;
            float32x4_t _mean = vdupq_n_f32(mean);

            // sum
            ptr = ptr0;
            float sqsum = 0.f;
            float tmp = 0.f;
            float32x4_t _sqsum = vdupq_n_f32(0.f);
            i = 0;
            for (; i + 3 < w; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _tmp = vsubq_f32(_p, _mean);
                _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                ptr += 4;
            }
            sqsum += sum_float32x4(_sqsum);
            for (; i < w; i++)
            {
                tmp = *ptr - mean;
                sqsum += tmp * tmp;
                ptr++;
            }

            // var
            float var = sqsum / w;

            // coefficient
            float a = static_cast<float>(1.f / (sqrt(var + eps)));
            float b = -mean * a;
            float32x4_t _a = vdupq_n_f32(a);
            float32x4_t _b = vdupq_n_f32(b);

            // affine
            if (affine)
            {
                const float* gptr = (const float*)gamma_data;
                const float* bptr = (const float*)beta_data;

                i = 0;
                for (; i + 3 < w; i += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr0);
                    _p = vmlaq_f32(_b, _p, _a);
                    float32x4_t _gamma = vld1q_f32(gptr);
                    float32x4_t _beta = vld1q_f32(bptr);
                    _p = vmlaq_f32(_beta, _p, _gamma);
                    vst1q_f32(ptr0, _p);
                    ptr0 += 4;
                    gptr += 4;
                    bptr += 4;
                }
                for (; i < w; i++)
                {
                    *ptr0 = ((*ptr0) * a + b) * (*gptr) + (*bptr);
                    ptr0++;
                    gptr++;
                    bptr++;
                }
            }
            else
            {
                i = 0;
                for (; i + 3 < w; i += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr0);
                    _p = vmlaq_f32(_b, _p, _a);
                    vst1q_f32(ptr0, _p);
                    ptr0 += 4;
                }
                for (; i < w; i++)
                {
                    *ptr0 = (*ptr0) * a + b;
                    ptr0++;
                }
            }

            return 0;
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr0 = bottom_top_blob.row(i);

                // sum
                float sum = 0.f;
                float32x4_t _sum = vdupq_n_f32(0.f);
                const float* ptr = ptr0;
                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    _sum = vaddq_f32(_sum, _p);
                    ptr += 4;
                }
                sum += sum_float32x4(_sum);
                for (; j < w; j++)
                {
                    sum += *ptr;
                    ptr++;
                }

                // mean
                float mean = sum / w;
                float32x4_t _mean = vdupq_n_f32(mean);

                // sum
                ptr = ptr0;
                float sqsum = 0.f;
                float tmp = 0.f;
                float32x4_t _sqsum = vdupq_n_f32(0.f);
                j = 0;
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _tmp = vsubq_f32(_p, _mean);
                    _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                    ptr += 4;
                }
                sqsum += sum_float32x4(_sqsum);
                for (; j < w; j++)
                {
                    tmp = *ptr - mean;
                    sqsum += tmp * tmp;
                    ptr++;
                }

                // var
                float var = sqsum / w;

                // coefficient
                float a = static_cast<float>(1.f / (sqrt(var + eps)));
                float b = -mean * a;
                float32x4_t _a = vdupq_n_f32(a);
                float32x4_t _b = vdupq_n_f32(b);

                // affine
                if (affine)
                {
                    const float* gptr = (const float*)gamma_data;
                    const float* bptr = (const float*)beta_data;

                    j = 0;
                    for (; j + 3 < w; j += 4)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        _p = vmlaq_f32(_b, _p, _a);
                        float32x4_t _gamma = vld1q_f32(gptr);
                        float32x4_t _beta = vld1q_f32(bptr);
                        _p = vmlaq_f32(_beta, _p, _gamma);
                        vst1q_f32(ptr0, _p);
                        ptr0 += 4;
                        gptr += 4;
                        bptr += 4;
                    }
                    for (; j < w; j++)
                    {
                        *ptr0 = ((*ptr0) * a + b) * (*gptr) + (*bptr);
                        ptr0++;
                        gptr++;
                        bptr++;
                    }
                }
                else
                {
                    j = 0;
                    for (; j + 3 < w; j += 4)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        _p = vmlaq_f32(_b, _p, _a);
                        vst1q_f32(ptr0, _p);
                        ptr0 += 4;
                    }
                    for (; j < w; j++)
                    {
                        *ptr0 = (*ptr0) * a + b;
                        ptr0++;
                    }
                }
            }

            return 0;
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr0 = bottom_top_blob.channel(q).row(i);

                        // sum
                        float sum = 0.f;
                        float32x4_t _sum = vdupq_n_f32(0.f);
                        const float* ptr = ptr0;
                        int j = 0;
                        for (; j + 3 < w; j += 4)
                        {
                            float32x4_t _p = vld1q_f32(ptr);
                            _sum = vaddq_f32(_sum, _p);
                            ptr += 4;
                        }
                        sum += sum_float32x4(_sum);
                        for (; j < w; j++)
                        {
                            sum += *ptr;
                            ptr++;
                        }

                        // mean
                        float mean = sum / w;
                        float32x4_t _mean = vdupq_n_f32(mean);

                        // sum
                        ptr = ptr0;
                        float sqsum = 0.f;
                        float tmp = 0.f;
                        float32x4_t _sqsum = vdupq_n_f32(0.f);
                        j = 0;
                        for (; j + 3 < w; j += 4)
                        {
                            float32x4_t _p = vld1q_f32(ptr);
                            float32x4_t _tmp = vsubq_f32(_p, _mean);
                            _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                            ptr += 4;
                        }
                        sqsum += sum_float32x4(_sqsum);
                        for (; j < w; j++)
                        {
                            tmp = *ptr - mean;
                            sqsum += tmp * tmp;
                            ptr++;
                        }

                        // var
                        float var = sqsum / w;

                        // coefficient
                        float a = static_cast<float>(1.f / (sqrt(var + eps)));
                        float b = -mean * a;
                        float32x4_t _a = vdupq_n_f32(a);
                        float32x4_t _b = vdupq_n_f32(b);

                        // affine
                        if (affine)
                        {
                            const float* gptr = (const float*)gamma_data;
                            const float* bptr = (const float*)beta_data;

                            j = 0;
                            for (; j + 3 < w; j += 4)
                            {
                                float32x4_t _p = vld1q_f32(ptr0);
                                _p = vmlaq_f32(_b, _p, _a);
                                float32x4_t _gamma = vld1q_f32(gptr);
                                float32x4_t _beta = vld1q_f32(bptr);
                                _p = vmlaq_f32(_beta, _p, _gamma);
                                vst1q_f32(ptr0, _p);
                                ptr0 += 4;
                                gptr += 4;
                                bptr += 4;
                            }
                            for (; j < w; j++)
                            {
                                *ptr0 = ((*ptr0) * a + b) * (*gptr) + (*bptr);
                                ptr0++;
                                gptr++;
                                bptr++;
                            }
                        }
                        else
                        {
                            j = 0;
                            for (; j + 3 < w; j += 4)
                            {
                                float32x4_t _p = vld1q_f32(ptr0);
                                _p = vmlaq_f32(_b, _p, _a);
                                vst1q_f32(ptr0, _p);
                                ptr0 += 4;
                            }
                            for (; j < w; j++)
                            {
                                *ptr0 = (*ptr0) * a + b;
                                ptr0++;
                            }
                        }
                    }
                }

                return 0;
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr0 = bottom_top_blob.channel(q);

                    // sum
                    float sum = 0.f;
                    float32x4_t _sum = vdupq_n_f32(0.f);
                    const float* ptr = ptr0;
                    int j = 0;
                    for (; j + 3 < size; j += 4)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _sum = vaddq_f32(_sum, _p);
                        ptr += 4;
                    }
                    sum += sum_float32x4(_sum);
                    for (; j < size; j++)
                    {
                        sum += *ptr;
                        ptr++;
                    }

                    // mean
                    float mean = sum / size;
                    float32x4_t _mean = vdupq_n_f32(mean);

                    // sum
                    ptr = ptr0;
                    float sqsum = 0.f;
                    float tmp = 0.f;
                    float32x4_t _sqsum = vdupq_n_f32(0.f);
                    j = 0;
                    for (; j + 3 < size; j += 4)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        float32x4_t _tmp = vsubq_f32(_p, _mean);
                        _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                        ptr += 4;
                    }
                    sqsum += sum_float32x4(_sqsum);
                    for (; j < size; j++)
                    {
                        tmp = *ptr - mean;
                        sqsum += tmp * tmp;
                        ptr++;
                    }

                    // var
                    float var = sqsum / size;

                    // coefficient
                    float a = static_cast<float>(1.f / (sqrt(var + eps)));
                    float b = -mean * a;
                    float32x4_t _a = vdupq_n_f32(a);
                    float32x4_t _b = vdupq_n_f32(b);

                    // affine
                    if (affine)
                    {
                        const float* gptr = (const float*)gamma_data;
                        const float* bptr = (const float*)beta_data;

                        j = 0;
                        for (; j + 3 < size; j += 4)
                        {
                            float32x4_t _p = vld1q_f32(ptr0);
                            _p = vmlaq_f32(_b, _p, _a);
                            float32x4_t _gamma = vld1q_f32(gptr);
                            float32x4_t _beta = vld1q_f32(bptr);
                            _p = vmlaq_f32(_beta, _p, _gamma);
                            vst1q_f32(ptr0, _p);
                            ptr0 += 4;
                            gptr += 4;
                            bptr += 4;
                        }
                        for (; j < size; j++)
                        {
                            *ptr0 = ((*ptr0) * a + b) * (*gptr) + (*bptr);
                            ptr0++;
                            gptr++;
                            bptr++;
                        }
                    }
                    else
                    {
                        j = 0;
                        for (; j + 3 < size; j += 4)
                        {
                            float32x4_t _p = vld1q_f32(ptr0);
                            _p = vmlaq_f32(_b, _p, _a);
                            vst1q_f32(ptr0, _p);
                            ptr0 += 4;
                        }
                        for (; j < size; j++)
                        {
                            *ptr0 = (*ptr0) * a + b;
                            ptr0++;
                        }
                    }
                }

                return 0;
            }
        }
    }

#endif

    // fallback to naive implement
    return LayerNorm::forward_inplace(bottom_top_blob, opt);
}

#if NCNN_BF16
int LayerNorm_arm::create_pipeline_bf16s(const Option& opt)
{
    ncnn::cast_float32_to_bfloat16(gamma_data, gamma_data_bf16, opt);
    ncnn::cast_float32_to_bfloat16(beta_data, beta_data_bf16, opt);

    if (opt.lightmode)
    {
        gamma_data.release();
        beta_data.release();
    }

    return 0;
}

#if __ARM_NEON
int LayerNorm_arm::forward_inplace_pack1_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        unsigned short* ptr0 = (unsigned short*)bottom_top_blob;

        // sum
        float sum = 0.f;
        float32x4_t _sum = vdupq_n_f32(0.f);
        const unsigned short* ptr = ptr0;
        int i = 0;
        for (; i + 3 < w; i += 4)
        {
            float32x4_t _p = float2bfloat(vld1_u16(ptr));
            _sum = vaddq_f32(_sum, _p);
            ptr += 4;
        }
        sum += sum_float32x4(_sum);
        for (; i < w; i++)
        {
            sum += bfloat16_to_float32(*ptr);
            ptr++;
        }

        // mean
        float mean = sum / w;
        float32x4_t _mean = vdupq_n_f32(mean);

        // sum
        ptr = ptr0;
        float sqsum = 0.f;
        float tmp = 0.f;
        float32x4_t _sqsum = vdupq_n_f32(0.f);
        i = 0;
        for (; i + 3 < w; i += 4)
        {
            float32x4_t _p = float2bfloat(vld1_u16(ptr));
            float32x4_t _tmp = vsubq_f32(_p, _mean);
            _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
            ptr += 4;
        }
        sqsum += sum_float32x4(_sqsum);
        for (; i < w; i++)
        {
            tmp = bfloat16_to_float32(*ptr) - mean;
            sqsum += tmp * tmp;
            ptr++;
        }

        // var
        float var = sqsum / w;

        // coefficient
        float a = static_cast<float>(1.f / (sqrt(var + eps)));
        float b = -mean * a;
        float32x4_t _a = vdupq_n_f32(a);
        float32x4_t _b = vdupq_n_f32(b);

        // affine
        if (affine)
        {
            const unsigned short* gptr = (const unsigned short*)gamma_data_bf16;
            const unsigned short* bptr = (const unsigned short*)beta_data_bf16;

            i = 0;
            for (; i + 3 < w; i += 4)
            {
                float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                _p = vmlaq_f32(_b, _p, _a);
                float32x4_t _gamma = float2bfloat(vld1_u16(gptr));
                float32x4_t _beta = float2bfloat(vld1_u16(bptr));
                _p = vmlaq_f32(_beta, _p, _gamma);
                vst1_u16(ptr0, bfloat2float(_p));
                ptr0 += 4;
                gptr += 4;
                bptr += 4;
            }
            for (; i < w; i++)
            {
                *ptr0 = float32_to_bfloat16((bfloat16_to_float32(*ptr0) * a + b) * bfloat16_to_float32(*gptr) + bfloat16_to_float32(*bptr));
                ptr0 += 1;
                gptr += 1;
                bptr += 1;
            }
        }
        else
        {
            i = 0;
            for (; i + 3 < w; i += 4)
            {
                float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                _p = vmlaq_f32(_b, _p, _a);
                vst1_u16(ptr0, bfloat2float(_p));
                ptr0 += 4;
            }
            for (; i < w; i++)
            {
                *ptr0 = float32_to_bfloat16(bfloat16_to_float32(*ptr0) * a + b);
                ptr0 += 1;
            }
        }

        return 0;
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr0 = bottom_top_blob.row<unsigned short>(i);

            // sum
            float sum = 0.f;
            float32x4_t _sum = vdupq_n_f32(0.f);
            const unsigned short* ptr = ptr0;
            int j = 0;
            for (; j + 3 < w; j += 4)
            {
                float32x4_t _p = float2bfloat(vld1_u16(ptr));
                _sum = vaddq_f32(_sum, _p);
                ptr += 4;
            }
            sum += sum_float32x4(_sum);
            for (; j < w; j++)
            {
                sum += bfloat16_to_float32(*ptr);
                ptr++;
            }

            // mean
            float mean = sum / w;
            float32x4_t _mean = vdupq_n_f32(mean);

            // sum
            ptr = ptr0;
            float sqsum = 0.f;
            float tmp = 0.f;
            float32x4_t _sqsum = vdupq_n_f32(0.f);
            j = 0;
            for (; j + 3 < w; j += 4)
            {
                float32x4_t _p = float2bfloat(vld1_u16(ptr));
                float32x4_t _tmp = vsubq_f32(_p, _mean);
                _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                ptr += 4;
            }
            sqsum += sum_float32x4(_sqsum);
            for (; j < w; j++)
            {
                tmp = bfloat16_to_float32(*ptr) - mean;
                sqsum += tmp * tmp;
                ptr++;
            }

            // var
            float var = sqsum / w;

            // coefficient
            float a = static_cast<float>(1.f / (sqrt(var + eps)));
            float b = -mean * a;
            float32x4_t _a = vdupq_n_f32(a);
            float32x4_t _b = vdupq_n_f32(b);

            // affine
            if (affine)
            {
                const unsigned short* gptr = (const unsigned short*)gamma_data_bf16;
                const unsigned short* bptr = (const unsigned short*)beta_data_bf16;

                j = 0;
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                    _p = vmlaq_f32(_b, _p, _a);
                    float32x4_t _gamma = float2bfloat(vld1_u16(gptr));
                    float32x4_t _beta = float2bfloat(vld1_u16(bptr));
                    _p = vmlaq_f32(_beta, _p, _gamma);
                    vst1_u16(ptr0, bfloat2float(_p));
                    ptr0 += 4;
                    gptr += 4;
                    bptr += 4;
                }
                for (; j < w; j++)
                {
                    *ptr0 = float32_to_bfloat16((bfloat16_to_float32(*ptr0) * a + b) * bfloat16_to_float32(*gptr) + bfloat16_to_float32(*bptr));
                    ptr0++;
                    gptr++;
                    bptr++;
                }
            }
            else
            {
                j = 0;
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                    _p = vmlaq_f32(_b, _p, _a);
                    vst1_u16(ptr0, bfloat2float(_p));
                    ptr0 += 4;
                }
                for (; j < w; j++)
                {
                    *ptr0 = float32_to_bfloat16(bfloat16_to_float32(*ptr0) * a + b);
                    ptr0++;
                }
            }
        }

        return 0;
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    unsigned short* ptr0 = bottom_top_blob.channel(q).row<unsigned short>(i);

                    // sum
                    float sum = 0.f;
                    float32x4_t _sum = vdupq_n_f32(0.f);
                    const unsigned short* ptr = ptr0;
                    int j = 0;
                    for (; j + 3 < w; j += 4)
                    {
                        float32x4_t _p = float2bfloat(vld1_u16(ptr));
                        _sum = vaddq_f32(_sum, _p);
                        ptr += 4;
                    }
                    sum += sum_float32x4(_sum);
                    for (; j < w; j++)
                    {
                        sum += bfloat16_to_float32(*ptr);
                        ptr++;
                    }

                    // mean
                    float mean = sum / w;
                    float32x4_t _mean = vdupq_n_f32(mean);

                    // sum
                    ptr = ptr0;
                    float sqsum = 0.f;
                    float tmp = 0.f;
                    float32x4_t _sqsum = vdupq_n_f32(0.f);
                    j = 0;
                    for (; j + 3 < w; j += 4)
                    {
                        float32x4_t _p = float2bfloat(vld1_u16(ptr));
                        float32x4_t _tmp = vsubq_f32(_p, _mean);
                        _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                        ptr += 4;
                    }
                    sqsum += sum_float32x4(_sqsum);
                    for (; j < w; j++)
                    {
                        tmp = bfloat16_to_float32(*ptr) - mean;
                        sqsum += tmp * tmp;
                        ptr++;
                    }

                    // var
                    float var = sqsum / w;

                    // coefficient
                    float a = static_cast<float>(1.f / (sqrt(var + eps)));
                    float b = -mean * a;
                    float32x4_t _a = vdupq_n_f32(a);
                    float32x4_t _b = vdupq_n_f32(b);

                    // affine
                    if (affine)
                    {
                        const unsigned short* gptr = (const unsigned short*)gamma_data_bf16;
                        const unsigned short* bptr = (const unsigned short*)beta_data_bf16;

                        j = 0;
                        for (; j + 3 < w; j += 4)
                        {
                            float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                            _p = vmlaq_f32(_b, _p, _a);
                            float32x4_t _gamma = float2bfloat(vld1_u16(gptr));
                            float32x4_t _beta = float2bfloat(vld1_u16(bptr));
                            _p = vmlaq_f32(_beta, _p, _gamma);
                            vst1_u16(ptr0, bfloat2float(_p));
                            ptr0 += 4;
                            gptr += 4;
                            bptr += 4;
                        }
                        for (; j < w; j++)
                        {
                            *ptr0 = float32_to_bfloat16((bfloat16_to_float32(*ptr0) * a + b) * bfloat16_to_float32(*gptr) + bfloat16_to_float32(*bptr));
                            ptr0++;
                            gptr++;
                            bptr++;
                        }
                    }
                    else
                    {
                        j = 0;
                        for (; j + 3 < w; j += 4)
                        {
                            float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                            _p = vmlaq_f32(_b, _p, _a);
                            vst1_u16(ptr0, bfloat2float(_p));
                            ptr0 += 4;
                        }
                        for (; j < w; j++)
                        {
                            *ptr0 = float32_to_bfloat16(bfloat16_to_float32(*ptr0) * a + b);
                            ptr0++;
                        }
                    }
                }
            }

            return 0;
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr0 = (unsigned short*)bottom_top_blob.channel(q);

                // sum
                float sum = 0.f;
                float32x4_t _sum = vdupq_n_f32(0.f);
                const unsigned short* ptr = ptr0;
                int j = 0;
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = float2bfloat(vld1_u16(ptr));
                    _sum = vaddq_f32(_sum, _p);
                    ptr += 4;
                }
                sum += sum_float32x4(_sum);
                for (; j < size; j++)
                {
                    sum += bfloat16_to_float32(*ptr);
                    ptr++;
                }

                // mean
                float mean = sum / size;
                float32x4_t _mean = vdupq_n_f32(mean);

                // sum
                ptr = ptr0;
                float sqsum = 0.f;
                float tmp = 0.f;
                float32x4_t _sqsum = vdupq_n_f32(0.f);
                j = 0;
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = float2bfloat(vld1_u16(ptr));
                    float32x4_t _tmp = vsubq_f32(_p, _mean);
                    _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                    ptr += 4;
                }
                sqsum += sum_float32x4(_sqsum);
                for (; j < size; j++)
                {
                    tmp = bfloat16_to_float32(*ptr) - mean;
                    sqsum += tmp * tmp;
                    ptr++;
                }

                // var
                float var = sqsum / size;

                // coefficient
                float a = static_cast<float>(1.f / (sqrt(var + eps)));
                float b = -mean * a;
                float32x4_t _a = vdupq_n_f32(a);
                float32x4_t _b = vdupq_n_f32(b);

                // affine
                if (affine)
                {
                    const unsigned short* gptr = (const unsigned short*)gamma_data_bf16;
                    const unsigned short* bptr = (const unsigned short*)beta_data_bf16;

                    j = 0;
                    for (; j + 3 < size; j += 4)
                    {
                        float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                        _p = vmlaq_f32(_b, _p, _a);
                        float32x4_t _gamma = float2bfloat(vld1_u16(gptr));
                        float32x4_t _beta = float2bfloat(vld1_u16(bptr));
                        _p = vmlaq_f32(_beta, _p, _gamma);
                        vst1_u16(ptr0, bfloat2float(_p));
                        ptr0 += 4;
                        gptr += 4;
                        bptr += 4;
                    }
                    for (; j < size; j++)
                    {
                        *ptr0 = float32_to_bfloat16((bfloat16_to_float32(*ptr0) * a + b) * bfloat16_to_float32(*gptr) + bfloat16_to_float32(*bptr));
                        ptr0++;
                        gptr++;
                        bptr++;
                    }
                }
                else
                {
                    j = 0;
                    for (; j + 3 < size; j += 4)
                    {
                        float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                        _p = vmlaq_f32(_b, _p, _a);
                        vst1_u16(ptr0, bfloat2float(_p));
                        ptr0 += 4;
                    }
                    for (; j < size; j++)
                    {
                        *ptr0 = float32_to_bfloat16(bfloat16_to_float32(*ptr0) * a + b);
                        ptr0++;
                    }
                }
            }

            return 0;
        }
    }

    return 0;
}

int LayerNorm_arm::forward_inplace_pack4_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        unsigned short* ptr0 = (unsigned short*)bottom_top_blob;

        // sum
        float sum = 0.f;
        float32x4_t _sum = vdupq_n_f32(0.f);
        const unsigned short* ptr = ptr0;
        for (int i = 0; i < w; i++)
        {
            float32x4_t _p = float2bfloat(vld1_u16(ptr));
            _sum = vaddq_f32(_sum, _p);
            ptr += 4;
        }
        sum += sum_float32x4(_sum);

        // mean
        float mean = 0.25 * sum / w;
        float32x4_t _mean = vdupq_n_f32(mean);

        // sum
        ptr = ptr0;
        float sqsum = 0.f;
        float32x4_t _sqsum = vdupq_n_f32(0.f);
        for (int i = 0; i < w; i++)
        {
            float32x4_t _p = float2bfloat(vld1_u16(ptr));
            float32x4_t _tmp = vsubq_f32(_p, _mean);
            _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
            ptr += 4;
        }
        sqsum += sum_float32x4(_sqsum);

        // var
        float var = 0.25 * sqsum / w;

        // coefficient
        float a = static_cast<float>(1.f / (sqrt(var + eps)));
        float b = -mean * a;
        float32x4_t _a = vdupq_n_f32(a);
        float32x4_t _b = vdupq_n_f32(b);

        // affine
        if (affine)
        {
            const unsigned short* gptr = (const unsigned short*)gamma_data_bf16;
            const unsigned short* bptr = (const unsigned short*)beta_data_bf16;

            for (int i = 0; i < w; i++)
            {
                float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                _p = vmlaq_f32(_b, _p, _a);
                float32x4_t _gamma = float2bfloat(vld1_u16(gptr));
                float32x4_t _beta = float2bfloat(vld1_u16(bptr));
                _p = vmlaq_f32(_beta, _p, _gamma);
                vst1_u16(ptr0, bfloat2float(_p));
                ptr0 += 4;
                gptr += 4;
                bptr += 4;
            }
        }
        else
        {
            for (int i = 0; i < w; i++)
            {
                float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                _p = vmlaq_f32(_b, _p, _a);
                vst1_u16(ptr0, bfloat2float(_p));
                ptr0 += 4;
            }
        }

        return 0;
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        float32x4_t _div_size = vdupq_n_f32(1.f / w);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr0 = bottom_top_blob.row<unsigned short>(i);

            // sum
            float32x4_t _sum = vdupq_n_f32(0.f);
            const unsigned short* ptr = ptr0;
            for (int j = 0; j < w; j++)
            {
                float32x4_t _p = float2bfloat(vld1_u16(ptr));
                _sum = vaddq_f32(_sum, _p);
                ptr += 4;
            }

            // mean
            float32x4_t _mean = vmulq_f32(_sum, _div_size);

            // sum
            ptr = ptr0;
            float32x4_t _sqsum = vdupq_n_f32(0.f);
            for (int j = 0; j < w; j++)
            {
                float32x4_t _p = float2bfloat(vld1_u16(ptr));
                float32x4_t _tmp = vsubq_f32(_p, _mean);
                _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                ptr += 4;
            }

            // var
            float32x4_t _var_eps = vmlaq_f32(vdupq_n_f32(eps), _sqsum, _div_size);

            // coefficient
            float32x4_t _a = vrsqrteq_f32(_var_eps);
            _a = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps, _a), _a), _a);
            float32x4_t _b = vmlsq_f32(vdupq_n_f32(0.f), _mean, _a);

            // affine
            if (affine)
            {
                const unsigned short* gptr = (const unsigned short*)gamma_data_bf16;
                const unsigned short* bptr = (const unsigned short*)beta_data_bf16;

                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                    _p = vmlaq_f32(_b, _p, _a);
                    float32x4_t _gamma = float2bfloat(vdup_n_u16(*gptr));
                    float32x4_t _beta = float2bfloat(vdup_n_u16(*bptr));
                    _p = vmlaq_f32(_beta, _p, _gamma);
                    vst1_u16(ptr0, bfloat2float(_p));
                    ptr0 += 4;
                    gptr += 1;
                    bptr += 1;
                }
            }
            else
            {
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                    _p = vmlaq_f32(_b, _p, _a);
                    vst1_u16(ptr0, bfloat2float(_p));
                    ptr0 += 4;
                }
            }
        }

        return 0;
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        float32x4_t _div_size = vdupq_n_f32(1.f / affine_size);

        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    unsigned short* ptr0 = bottom_top_blob.channel(q).row<unsigned short>(i);

                    // sum
                    float32x4_t _sum = vdupq_n_f32(0.f);
                    const unsigned short* ptr = ptr0;
                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = float2bfloat(vld1_u16(ptr));
                        _sum = vaddq_f32(_sum, _p);
                        ptr += 4;
                    }

                    // mean
                    float32x4_t _mean = vmulq_f32(_sum, _div_size);

                    // sum
                    ptr = ptr0;
                    float32x4_t _sqsum = vdupq_n_f32(0.f);
                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = float2bfloat(vld1_u16(ptr));
                        float32x4_t _tmp = vsubq_f32(_p, _mean);
                        _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                        ptr += 4;
                    }

                    // var
                    float32x4_t _var_eps = vmlaq_f32(vdupq_n_f32(eps), _sqsum, _div_size);

                    // coefficient
                    float32x4_t _a = vrsqrteq_f32(_var_eps);
                    _a = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps, _a), _a), _a);
                    float32x4_t _b = vmlsq_f32(vdupq_n_f32(0.f), _mean, _a);

                    // affine
                    if (affine)
                    {
                        const unsigned short* gptr = (const unsigned short*)gamma_data_bf16;
                        const unsigned short* bptr = (const unsigned short*)beta_data_bf16;

                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                            _p = vmlaq_f32(_b, _p, _a);
                            float32x4_t _gamma = float2bfloat(vdup_n_u16(*gptr));
                            float32x4_t _beta = float2bfloat(vdup_n_u16(*bptr));
                            _p = vmlaq_f32(_beta, _p, _gamma);
                            vst1_u16(ptr0, bfloat2float(_p));
                            ptr0 += 4;
                            gptr += 1;
                            bptr += 1;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                            _p = vmlaq_f32(_b, _p, _a);
                            vst1_u16(ptr0, bfloat2float(_p));
                            ptr0 += 4;
                        }
                    }
                }
            }

            return 0;
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr0 = (unsigned short*)bottom_top_blob.channel(q);

                // sum
                float32x4_t _sum = vdupq_n_f32(0.f);
                const unsigned short* ptr = ptr0;
                for (int j = 0; j < size; j++)
                {
                    float32x4_t _p = float2bfloat(vld1_u16(ptr));
                    _sum = vaddq_f32(_sum, _p);
                    ptr += 4;
                }

                // mean
                float32x4_t _mean = vmulq_f32(_sum, _div_size);

                // sum
                ptr = ptr0;
                float32x4_t _sqsum = vdupq_n_f32(0.f);
                for (int j = 0; j < size; j++)
                {
                    float32x4_t _p = float2bfloat(vld1_u16(ptr));
                    float32x4_t _tmp = vsubq_f32(_p, _mean);
                    _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                    ptr += 4;
                }

                // var
                float32x4_t _var_eps = vmlaq_f32(vdupq_n_f32(eps), _sqsum, _div_size);

                // coefficient
                float32x4_t _a = vrsqrteq_f32(_var_eps);
                _a = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps, _a), _a), _a);
                float32x4_t _b = vmlsq_f32(vdupq_n_f32(0.f), _mean, _a);

                // affine
                if (affine)
                {
                    const unsigned short* gptr = (const unsigned short*)gamma_data_bf16;
                    const unsigned short* bptr = (const unsigned short*)beta_data_bf16;

                    for (int j = 0; j < size; j++)
                    {
                        float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                        _p = vmlaq_f32(_b, _p, _a);
                        float32x4_t _gamma = float2bfloat(vdup_n_u16(*gptr));
                        float32x4_t _beta = float2bfloat(vdup_n_u16(*bptr));
                        _p = vmlaq_f32(_beta, _p, _gamma);
                        vst1_u16(ptr0, bfloat2float(_p));
                        ptr0 += 4;
                        gptr += 1;
                        bptr += 1;
                    }
                }
                else
                {
                    for (int j = 0; j < size; j++)
                    {
                        float32x4_t _p = float2bfloat(vld1_u16(ptr0));
                        _p = vmlaq_f32(_b, _p, _a);
                        vst1_u16(ptr0, bfloat2float(_p));
                        ptr0 += 4;
                    }
                }
            }

            return 0;
        }
    }

    return 0;
}
#endif

int LayerNorm_arm::forward_inplace_naive_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        unsigned short* ptr0 = (unsigned short*)bottom_top_blob;

        // sum
        float sum = 0.f;
        const unsigned short* ptr = ptr0;
        for (int i = 0; i < w; i++)
        {
            sum += bfloat16_to_float32(*ptr);
            ptr++;
        }

        // mean
        float mean = sum / w;

        // sum
        ptr = ptr0;
        float sqsum = 0.f;
        float tmp = 0.f;
        for (int i = 0; i < w; i++)
        {
            tmp = bfloat16_to_float32(*ptr) - mean;
            sqsum += tmp * tmp;
            ptr++;
        }

        // var
        float var = sqsum / w;

        // coefficient
        float a = static_cast<float>(1.f / (sqrt(var + eps)));
        float b = -mean * a;

        // affine
        if (affine)
        {
            const unsigned short* gptr = (const unsigned short*)gamma_data_bf16;
            const unsigned short* bptr = (const unsigned short*)beta_data_bf16;

            for (int i = 0; i < w; i++)
            {
                *ptr0 = float32_to_bfloat16((bfloat16_to_float32(*ptr0) * a + b) * bfloat16_to_float32(*gptr) + bfloat16_to_float32(*bptr));
                ptr0 += 1;
                gptr += 1;
                bptr += 1;
            }
        }
        else
        {
            for (int i = 0; i < w; i++)
            {
                *ptr0 = float32_to_bfloat16(bfloat16_to_float32(*ptr0) * a + b);
                ptr0 += 1;
            }
        }

        return 0;
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr0 = bottom_top_blob.row<unsigned short>(i);

            // sum
            float sum = 0.f;
            const unsigned short* ptr = ptr0;
            for (int j = 0; j < w; j++)
            {
                sum += bfloat16_to_float32(*ptr);
                ptr++;
            }

            // mean
            float mean = sum / w;

            // sum
            ptr = ptr0;
            float sqsum = 0.f;
            float tmp = 0.f;
            for (int j = 0; j < w; j++)
            {
                tmp = bfloat16_to_float32(*ptr) - mean;
                sqsum += tmp * tmp;
                ptr++;
            }

            // var
            float var = sqsum / w;

            // coefficient
            float a = static_cast<float>(1.f / (sqrt(var + eps)));
            float b = -mean * a;

            // affine
            if (affine)
            {
                const unsigned short* gptr = (const unsigned short*)gamma_data_bf16;
                const unsigned short* bptr = (const unsigned short*)beta_data_bf16;

                for (int j = 0; j < w; j++)
                {
                    *ptr0 = float32_to_bfloat16((bfloat16_to_float32(*ptr0) * a + b) * bfloat16_to_float32(*gptr) + bfloat16_to_float32(*bptr));
                    ptr0++;
                    gptr++;
                    bptr++;
                }
            }
            else
            {
                for (int j = 0; j < w; j++)
                {
                    *ptr0 = float32_to_bfloat16(bfloat16_to_float32(*ptr0) * a + b);
                    ptr0++;
                }
            }
        }

        return 0;
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    unsigned short* ptr0 = bottom_top_blob.channel(q).row<unsigned short>(i);

                    // sum
                    float sum = 0.f;
                    const unsigned short* ptr = ptr0;
                    for (int j = 0; j < w; j++)
                    {
                        sum += bfloat16_to_float32(*ptr);
                        ptr++;
                    }

                    // mean
                    float mean = sum / w;

                    // sum
                    ptr = ptr0;
                    float sqsum = 0.f;
                    float tmp = 0.f;
                    for (int j = 0; j < w; j++)
                    {
                        tmp = bfloat16_to_float32(*ptr) - mean;
                        sqsum += tmp * tmp;
                        ptr++;
                    }

                    // var
                    float var = sqsum / w;

                    // coefficient
                    float a = static_cast<float>(1.f / (sqrt(var + eps)));
                    float b = -mean * a;

                    // affine
                    if (affine)
                    {
                        const unsigned short* gptr = (const unsigned short*)gamma_data_bf16;
                        const unsigned short* bptr = (const unsigned short*)beta_data_bf16;

                        for (int j = 0; j < w; j++)
                        {
                            *ptr0 = float32_to_bfloat16((bfloat16_to_float32(*ptr0) * a + b) * bfloat16_to_float32(*gptr) + bfloat16_to_float32(*bptr));
                            ptr0++;
                            gptr++;
                            bptr++;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < w; j++)
                        {
                            *ptr0 = float32_to_bfloat16(bfloat16_to_float32(*ptr0) * a + b);
                            ptr0++;
                        }
                    }
                }
            }

            return 0;
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr0 = (unsigned short*)bottom_top_blob.channel(q);

                // sum
                float sum = 0.f;
                const unsigned short* ptr = ptr0;
                for (int j = 0; j < size; j++)
                {
                    sum += bfloat16_to_float32(*ptr);
                    ptr++;
                }

                // mean
                float mean = sum / size;

                // sum
                ptr = ptr0;
                float sqsum = 0.f;
                float tmp = 0.f;
                for (int j = 0; j < size; j++)
                {
                    tmp = bfloat16_to_float32(*ptr) - mean;
                    sqsum += tmp * tmp;
                    ptr++;
                }

                // var
                float var = sqsum / size;

                // coefficient
                float a = static_cast<float>(1.f / (sqrt(var + eps)));
                float b = -mean * a;

                // affine
                if (affine)
                {
                    const unsigned short* gptr = (const unsigned short*)gamma_data_bf16;
                    const unsigned short* bptr = (const unsigned short*)beta_data_bf16;

                    for (int j = 0; j < size; j++)
                    {
                        *ptr0 = float32_to_bfloat16((bfloat16_to_float32(*ptr0) * a + b) * bfloat16_to_float32(*gptr) + bfloat16_to_float32(*bptr));
                        ptr0++;
                        gptr++;
                        bptr++;
                    }
                }
                else
                {
                    for (int j = 0; j < size; j++)
                    {
                        *ptr0 = float32_to_bfloat16(bfloat16_to_float32(*ptr0) * a + b);
                        ptr0++;
                    }
                }
            }

            return 0;
        }
    }

    return 0;
}

int LayerNorm_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;
    int elembits = bottom_top_blob.elembits();

#if __ARM_NEON
    if (elempack == 4)
    {
        return forward_inplace_pack4_bf16s(bottom_top_blob, opt);
    }

    if (elempack == 1)
    {
        return forward_inplace_pack1_bf16s(bottom_top_blob, opt);
    }
#endif

    return forward_inplace_naive_bf16s(bottom_top_blob, opt);

    return 0;
}
#endif

} // namespace ncnn
