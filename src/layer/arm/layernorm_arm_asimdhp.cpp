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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

inline float sum_float16x4(float32x4_t _sum)
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

inline __fp16 sum_float16x4(float16x4_t _sum)
{
    float16x4_t _ss2 = vpadd_f16(_sum, _sum);
    _ss2 = vpadd_f16(_ss2, _ss2);
    __fp16 sum = vget_lane_f16(_ss2, 0);
    return sum;
}

inline __fp16 sum_float16x8(float16x8_t _sum)
{
    float16x4_t _sum2 = vadd_f16(vget_low_f16(_sum), vget_high_f16(_sum));
    float16x4_t _ss2 = vpadd_f16(_sum2, _sum2);
    _ss2 = vpadd_f16(_ss2, _ss2);
    __fp16 sum = vget_lane_f16(_ss2, 0);
    return sum;
}

int LayerNorm_arm::create_pipeline_fp16s(const Option& opt)
{
    ncnn::cast_float32_to_float16(gamma_data, gamma_data_fp16, opt);
    ncnn::cast_float32_to_float16(beta_data, beta_data_fp16, opt);

    if (opt.lightmode)
    {
        gamma_data.release();
        beta_data.release();
    }

    return 0;
}

int LayerNorm_arm::forward_inplace_pack1_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        __fp16* ptr0 = (__fp16*)bottom_top_blob;

        // sum
        float sum = 0.f;
        float32x4_t _sum = vdupq_n_f32(0.f);
        const __fp16* ptr = ptr0;
        int i = 0;
        for (; i + 3 < w; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            _sum = vaddq_f32(_sum, _p);
            ptr += 4;
        }
        sum += sum_float16x4(_sum);
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
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            float32x4_t _tmp = vsubq_f32(_p, _mean);
            _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
            ptr += 4;
        }
        sqsum += sum_float16x4(_sqsum);
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
            const __fp16* gptr = (const __fp16*)gamma_data_fp16;
            const __fp16* bptr = (const __fp16*)beta_data_fp16;

            i = 0;
            for (; i + 3 < w; i += 4)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                _p = vmlaq_f32(_b, _p, _a);
                float32x4_t _gamma = vcvt_f32_f16(vld1_f16(gptr));
                float32x4_t _beta = vcvt_f32_f16(vld1_f16(bptr));
                _p = vmlaq_f32(_beta, _p, _gamma);
                vst1_f16(ptr0, vcvt_f16_f32(_p));
                ptr0 += 4;
                gptr += 4;
                bptr += 4;
            }
            for (; i < w; i++)
            {
                *ptr0 = ((*ptr0) * a + b) * (*gptr) + (*bptr);
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
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                _p = vmlaq_f32(_b, _p, _a);
                vst1_f16(ptr0, vcvt_f16_f32(_p));
                ptr0 += 4;
            }
            for (; i < w; i++)
            {
                *ptr0 = (*ptr0) * a + b;
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
            __fp16* ptr0 = (__fp16*)bottom_top_blob.row<__fp16>(i);

            // sum
            float sum = 0.f;
            float32x4_t _sum = vdupq_n_f32(0.f);
            const __fp16* ptr = ptr0;
            int j = 0;
            for (; j + 3 < w; j += 4)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                _sum = vaddq_f32(_sum, _p);
                ptr += 4;
            }
            sum += sum_float16x4(_sum);
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
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                float32x4_t _tmp = vsubq_f32(_p, _mean);
                _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                ptr += 4;
            }
            sqsum += sum_float16x4(_sqsum);
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
                const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                const __fp16* bptr = (const __fp16*)beta_data_fp16;

                j = 0;
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                    _p = vmlaq_f32(_b, _p, _a);
                    float32x4_t _gamma = vcvt_f32_f16(vld1_f16(gptr));
                    float32x4_t _beta = vcvt_f32_f16(vld1_f16(bptr));
                    _p = vmlaq_f32(_beta, _p, _gamma);
                    vst1_f16(ptr0, vcvt_f16_f32(_p));
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
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                    _p = vmlaq_f32(_b, _p, _a);
                    vst1_f16(ptr0, vcvt_f16_f32(_p));
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
                    __fp16* ptr0 = (__fp16*)bottom_top_blob.channel(q).row<__fp16>(i);

                    // sum
                    float sum = 0.f;
                    float32x4_t _sum = vdupq_n_f32(0.f);
                    const __fp16* ptr = ptr0;
                    int j = 0;
                    for (; j + 3 < w; j += 4)
                    {
                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                        _sum = vaddq_f32(_sum, _p);
                        ptr += 4;
                    }
                    sum += sum_float16x4(_sum);
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
                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                        float32x4_t _tmp = vsubq_f32(_p, _mean);
                        _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                        ptr += 4;
                    }
                    sqsum += sum_float16x4(_sqsum);
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
                        const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                        const __fp16* bptr = (const __fp16*)beta_data_fp16;

                        j = 0;
                        for (; j + 3 < w; j += 4)
                        {
                            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                            _p = vmlaq_f32(_b, _p, _a);
                            float32x4_t _gamma = vcvt_f32_f16(vld1_f16(gptr));
                            float32x4_t _beta = vcvt_f32_f16(vld1_f16(bptr));
                            _p = vmlaq_f32(_beta, _p, _gamma);
                            vst1_f16(ptr0, vcvt_f16_f32(_p));
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
                            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                            _p = vmlaq_f32(_b, _p, _a);
                            vst1_f16(ptr0, vcvt_f16_f32(_p));
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
                __fp16* ptr0 = (__fp16*)bottom_top_blob.channel(q);

                // sum
                float sum = 0.f;
                float32x4_t _sum = vdupq_n_f32(0.f);
                const __fp16* ptr = ptr0;
                int j = 0;
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                    _sum = vaddq_f32(_sum, _p);
                    ptr += 4;
                }
                sum += sum_float16x4(_sum);
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
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                    float32x4_t _tmp = vsubq_f32(_p, _mean);
                    _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
                    ptr += 4;
                }
                sqsum += sum_float16x4(_sqsum);
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
                    const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                    const __fp16* bptr = (const __fp16*)beta_data_fp16;

                    j = 0;
                    for (; j + 3 < size; j += 4)
                    {
                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                        _p = vmlaq_f32(_b, _p, _a);
                        float32x4_t _gamma = vcvt_f32_f16(vld1_f16(gptr));
                        float32x4_t _beta = vcvt_f32_f16(vld1_f16(bptr));
                        _p = vmlaq_f32(_beta, _p, _gamma);
                        vst1_f16(ptr0, vcvt_f16_f32(_p));
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
                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                        _p = vmlaq_f32(_b, _p, _a);
                        vst1_f16(ptr0, vcvt_f16_f32(_p));
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

    return 0;
}

int LayerNorm_arm::forward_inplace_pack4_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        __fp16* ptr0 = (__fp16*)bottom_top_blob;

        // sum
        float sum = 0.f;
        float32x4_t _sum = vdupq_n_f32(0.f);
        const __fp16* ptr = ptr0;
        for (int i = 0; i < w; i++)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            _sum = vaddq_f32(_sum, _p);
            ptr += 4;
        }
        sum += sum_float16x4(_sum);

        // mean
        float mean = 0.25 * sum / w;
        float32x4_t _mean = vdupq_n_f32(mean);

        // sum
        ptr = ptr0;
        float sqsum = 0.f;
        float32x4_t _sqsum = vdupq_n_f32(0.f);
        for (int i = 0; i < w; i++)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            float32x4_t _tmp = vsubq_f32(_p, _mean);
            _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
            ptr += 4;
        }
        sqsum += sum_float16x4(_sqsum);

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
            const __fp16* gptr = (const __fp16*)gamma_data_fp16;
            const __fp16* bptr = (const __fp16*)beta_data_fp16;

            for (int i = 0; i < w; i++)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                _p = vmlaq_f32(_b, _p, _a);
                float32x4_t _gamma = vcvt_f32_f16(vld1_f16(gptr));
                float32x4_t _beta = vcvt_f32_f16(vld1_f16(bptr));
                _p = vmlaq_f32(_beta, _p, _gamma);
                vst1_f16(ptr0, vcvt_f16_f32(_p));
                ptr0 += 4;
                gptr += 4;
                bptr += 4;
            }
        }
        else
        {
            for (int i = 0; i < w; i++)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                _p = vmlaq_f32(_b, _p, _a);
                vst1_f16(ptr0, vcvt_f16_f32(_p));
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
            __fp16* ptr0 = (__fp16*)bottom_top_blob.row<__fp16>(i);

            // sum
            float32x4_t _sum = vdupq_n_f32(0.f);
            const __fp16* ptr = ptr0;
            for (int j = 0; j < w; j++)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
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
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
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
                const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                const __fp16* bptr = (const __fp16*)beta_data_fp16;

                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                    _p = vmlaq_f32(_b, _p, _a);
                    float32x4_t _gamma = vdupq_n_f32(*gptr);
                    float32x4_t _beta = vdupq_n_f32(*bptr);
                    _p = vmlaq_f32(_beta, _p, _gamma);
                    vst1_f16(ptr0, vcvt_f16_f32(_p));
                    ptr0 += 4;
                    gptr += 1;
                    bptr += 1;
                }
            }
            else
            {
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                    _p = vmlaq_f32(_b, _p, _a);
                    vst1_f16(ptr0, vcvt_f16_f32(_p));
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

        if (affine_size == w)
        {
            float32x4_t _div_size = vdupq_n_f32(1.f / w);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    __fp16* ptr0 = (__fp16*)bottom_top_blob.channel(q).row<__fp16>(i);

                    // sum
                    float32x4_t _sum = vdupq_n_f32(0.f);
                    const __fp16* ptr = ptr0;
                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
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
                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
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
                        const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                        const __fp16* bptr = (const __fp16*)beta_data_fp16;

                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                            _p = vmlaq_f32(_b, _p, _a);
                            float32x4_t _gamma = vdupq_n_f32(*gptr);
                            float32x4_t _beta = vdupq_n_f32(*bptr);
                            _p = vmlaq_f32(_beta, _p, _gamma);
                            vst1_f16(ptr0, vcvt_f16_f32(_p));
                            ptr0 += 4;
                            gptr += 1;
                            bptr += 1;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                            _p = vmlaq_f32(_b, _p, _a);
                            vst1_f16(ptr0, vcvt_f16_f32(_p));
                            ptr0 += 4;
                        }
                    }
                }
            }

            return 0;
        }
        else
        {
            float32x4_t _div_size = vdupq_n_f32(1.f / size);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                __fp16* ptr0 = (__fp16*)bottom_top_blob.channel(q);

                // sum
                float32x4_t _sum = vdupq_n_f32(0.f);
                const __fp16* ptr = ptr0;
                for (int j = 0; j < size; j++)
                {
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
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
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
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
                    const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                    const __fp16* bptr = (const __fp16*)beta_data_fp16;

                    for (int j = 0; j < size; j++)
                    {
                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                        _p = vmlaq_f32(_b, _p, _a);
                        float32x4_t _gamma = vdupq_n_f32(*gptr);
                        float32x4_t _beta = vdupq_n_f32(*bptr);
                        _p = vmlaq_f32(_beta, _p, _gamma);
                        vst1_f16(ptr0, vcvt_f16_f32(_p));
                        ptr0 += 4;
                        gptr += 1;
                        bptr += 1;
                    }
                }
                else
                {
                    for (int j = 0; j < size; j++)
                    {
                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                        _p = vmlaq_f32(_b, _p, _a);
                        vst1_f16(ptr0, vcvt_f16_f32(_p));
                        ptr0 += 4;
                    }
                }
            }

            return 0;
        }
    }

    return 0;
}

int LayerNorm_arm::forward_inplace_pack8_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        w += w;
        __fp16* ptr0 = (__fp16*)bottom_top_blob;

        // sum
        float sum = 0.f;
        float32x4_t _sum = vdupq_n_f32(0.f);
        const __fp16* ptr = ptr0;
        for (int i = 0; i < w; i++)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            _sum = vaddq_f32(_sum, _p);
            ptr += 4;
        }
        sum += sum_float16x4(_sum);

        // mean
        float mean = 0.25 * sum / w;
        float32x4_t _mean = vdupq_n_f32(mean);

        // sum
        ptr = ptr0;
        float sqsum = 0.f;
        float32x4_t _sqsum = vdupq_n_f32(0.f);
        for (int i = 0; i < w; i++)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            float32x4_t _tmp = vsubq_f32(_p, _mean);
            _sqsum = vmlaq_f32(_sqsum, _tmp, _tmp);
            ptr += 4;
        }
        sqsum += sum_float16x4(_sqsum);

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
            const __fp16* gptr = (const __fp16*)gamma_data_fp16;
            const __fp16* bptr = (const __fp16*)beta_data_fp16;

            for (int i = 0; i < w; i++)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                _p = vmlaq_f32(_b, _p, _a);
                float32x4_t _gamma = vcvt_f32_f16(vld1_f16(gptr));
                float32x4_t _beta = vcvt_f32_f16(vld1_f16(bptr));
                _p = vmlaq_f32(_beta, _p, _gamma);
                vst1_f16(ptr0, vcvt_f16_f32(_p));
                ptr0 += 4;
                gptr += 4;
                bptr += 4;
            }
        }
        else
        {
            for (int i = 0; i < w; i++)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                _p = vmlaq_f32(_b, _p, _a);
                vst1_f16(ptr0, vcvt_f16_f32(_p));
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
            __fp16* ptr0 = (__fp16*)bottom_top_blob.row<__fp16>(i);

            // sum
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            const __fp16* ptr = ptr0;
            for (int j = 0; j < w; j++)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                _sum0 = vaddq_f32(_sum0, _p);
                ptr += 4;

                _p = vcvt_f32_f16(vld1_f16(ptr));
                _sum1 = vaddq_f32(_sum1, _p);
                ptr += 4;
            }

            // mean
            float32x4_t _mean0 = vmulq_f32(_sum0, _div_size);
            float32x4_t _mean1 = vmulq_f32(_sum1, _div_size);

            // sum
            ptr = ptr0;
            float32x4_t _sqsum0 = vdupq_n_f32(0.f);
            float32x4_t _sqsum1 = vdupq_n_f32(0.f);
            for (int j = 0; j < w; j++)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                float32x4_t _tmp = vsubq_f32(_p, _mean0);
                _sqsum0 = vmlaq_f32(_sqsum0, _tmp, _tmp);
                ptr += 4;

                _p = vcvt_f32_f16(vld1_f16(ptr));
                _tmp = vsubq_f32(_p, _mean1);
                _sqsum1 = vmlaq_f32(_sqsum1, _tmp, _tmp);
                ptr += 4;
            }

            // var
            float32x4_t _var_eps0 = vmlaq_f32(vdupq_n_f32(eps), _sqsum0, _div_size);
            float32x4_t _var_eps1 = vmlaq_f32(vdupq_n_f32(eps), _sqsum1, _div_size);

            // coefficient
            float32x4_t _a0 = vrsqrteq_f32(_var_eps0);
            float32x4_t _a1 = vrsqrteq_f32(_var_eps1);
            _a0 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps0, _a0), _a0), _a0);
            _a1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps1, _a1), _a1), _a1);
            float32x4_t _b0 = vmlsq_f32(vdupq_n_f32(0.f), _mean0, _a0);
            float32x4_t _b1 = vmlsq_f32(vdupq_n_f32(0.f), _mean1, _a1);

            // affine
            if (affine)
            {
                const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                const __fp16* bptr = (const __fp16*)beta_data_fp16;

                for (int j = 0; j < w; j++)
                {
                    float32x4_t _gamma = vdupq_n_f32(*gptr);
                    float32x4_t _beta = vdupq_n_f32(*bptr);

                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                    _p = vmlaq_f32(_b0, _p, _a0);
                    _p = vmlaq_f32(_beta, _p, _gamma);
                    vst1_f16(ptr0, vcvt_f16_f32(_p));
                    ptr0 += 4;

                    _p = vcvt_f32_f16(vld1_f16(ptr0));
                    _p = vmlaq_f32(_b1, _p, _a1);
                    _p = vmlaq_f32(_beta, _p, _gamma);
                    vst1_f16(ptr0, vcvt_f16_f32(_p));
                    ptr0 += 4;

                    gptr += 1;
                    bptr += 1;
                }
            }
            else
            {
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                    _p = vmlaq_f32(_b0, _p, _a0);
                    vst1_f16(ptr0, vcvt_f16_f32(_p));
                    ptr0 += 4;

                    _p = vcvt_f32_f16(vld1_f16(ptr0));
                    _p = vmlaq_f32(_b1, _p, _a1);
                    vst1_f16(ptr0, vcvt_f16_f32(_p));
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

        if (affine_size == w)
        {
            float32x4_t _div_size = vdupq_n_f32(1.f / w);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    __fp16* ptr0 = (__fp16*)bottom_top_blob.channel(q).row<__fp16>(i);

                    // sum
                    float32x4_t _sum0 = vdupq_n_f32(0.f);
                    float32x4_t _sum1 = vdupq_n_f32(0.f);
                    const __fp16* ptr = ptr0;
                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                        _sum0 = vaddq_f32(_sum0, _p);
                        ptr += 4;

                        _p = vcvt_f32_f16(vld1_f16(ptr));
                        _sum1 = vaddq_f32(_sum1, _p);
                        ptr += 4;
                    }

                    // mean
                    float32x4_t _mean0 = vmulq_f32(_sum0, _div_size);
                    float32x4_t _mean1 = vmulq_f32(_sum1, _div_size);

                    // sum
                    ptr = ptr0;
                    float32x4_t _sqsum0 = vdupq_n_f32(0.f);
                    float32x4_t _sqsum1 = vdupq_n_f32(0.f);
                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                        float32x4_t _tmp = vsubq_f32(_p, _mean0);
                        _sqsum0 = vmlaq_f32(_sqsum0, _tmp, _tmp);
                        ptr += 4;

                        _p = vcvt_f32_f16(vld1_f16(ptr));
                        _tmp = vsubq_f32(_p, _mean1);
                        _sqsum1 = vmlaq_f32(_sqsum1, _tmp, _tmp);
                        ptr += 4;
                    }

                    // var
                    float32x4_t _var_eps0 = vmlaq_f32(vdupq_n_f32(eps), _sqsum0, _div_size);
                    float32x4_t _var_eps1 = vmlaq_f32(vdupq_n_f32(eps), _sqsum1, _div_size);

                    // coefficient
                    float32x4_t _a0 = vrsqrteq_f32(_var_eps0);
                    float32x4_t _a1 = vrsqrteq_f32(_var_eps1);
                    _a0 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps0, _a0), _a0), _a0);
                    _a1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps1, _a1), _a1), _a1);
                    float32x4_t _b0 = vmlsq_f32(vdupq_n_f32(0.f), _mean0, _a0);
                    float32x4_t _b1 = vmlsq_f32(vdupq_n_f32(0.f), _mean1, _a1);

                    // affine
                    if (affine)
                    {
                        const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                        const __fp16* bptr = (const __fp16*)beta_data_fp16;

                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _gamma = vdupq_n_f32(*gptr);
                            float32x4_t _beta = vdupq_n_f32(*bptr);

                            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                            _p = vmlaq_f32(_b0, _p, _a0);
                            _p = vmlaq_f32(_beta, _p, _gamma);
                            vst1_f16(ptr0, vcvt_f16_f32(_p));
                            ptr0 += 4;

                            _p = vcvt_f32_f16(vld1_f16(ptr0));
                            _p = vmlaq_f32(_b1, _p, _a1);
                            _p = vmlaq_f32(_beta, _p, _gamma);
                            vst1_f16(ptr0, vcvt_f16_f32(_p));
                            ptr0 += 4;

                            gptr += 1;
                            bptr += 1;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                            _p = vmlaq_f32(_b0, _p, _a0);
                            vst1_f16(ptr0, vcvt_f16_f32(_p));
                            ptr0 += 4;

                            _p = vcvt_f32_f16(vld1_f16(ptr0));
                            _p = vmlaq_f32(_b1, _p, _a1);
                            vst1_f16(ptr0, vcvt_f16_f32(_p));
                            ptr0 += 4;
                        }
                    }
                }
            }

            return 0;
        }
        else
        {
            float32x4_t _div_size = vdupq_n_f32(1.f / size);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                __fp16* ptr0 = (__fp16*)bottom_top_blob.channel(q);

                // sum
                float32x4_t _sum0 = vdupq_n_f32(0.f);
                float32x4_t _sum1 = vdupq_n_f32(0.f);
                const __fp16* ptr = ptr0;
                for (int j = 0; j < size; j++)
                {
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                    _sum0 = vaddq_f32(_sum0, _p);
                    ptr += 4;

                    _p = vcvt_f32_f16(vld1_f16(ptr));
                    _sum1 = vaddq_f32(_sum1, _p);
                    ptr += 4;
                }

                // mean
                float32x4_t _mean0 = vmulq_f32(_sum0, _div_size);
                float32x4_t _mean1 = vmulq_f32(_sum1, _div_size);

                // sum
                ptr = ptr0;
                float32x4_t _sqsum0 = vdupq_n_f32(0.f);
                float32x4_t _sqsum1 = vdupq_n_f32(0.f);
                for (int j = 0; j < size; j++)
                {
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                    float32x4_t _tmp = vsubq_f32(_p, _mean0);
                    _sqsum0 = vmlaq_f32(_sqsum0, _tmp, _tmp);
                    ptr += 4;

                    _p = vcvt_f32_f16(vld1_f16(ptr));
                    _tmp = vsubq_f32(_p, _mean1);
                    _sqsum1 = vmlaq_f32(_sqsum1, _tmp, _tmp);
                    ptr += 4;
                }

                // var
                float32x4_t _var_eps0 = vmlaq_f32(vdupq_n_f32(eps), _sqsum0, _div_size);
                float32x4_t _var_eps1 = vmlaq_f32(vdupq_n_f32(eps), _sqsum1, _div_size);

                // coefficient
                float32x4_t _a0 = vrsqrteq_f32(_var_eps0);
                float32x4_t _a1 = vrsqrteq_f32(_var_eps1);
                _a0 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps0, _a0), _a0), _a0);
                _a1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var_eps1, _a1), _a1), _a1);
                float32x4_t _b0 = vmlsq_f32(vdupq_n_f32(0.f), _mean0, _a0);
                float32x4_t _b1 = vmlsq_f32(vdupq_n_f32(0.f), _mean1, _a1);

                // affine
                if (affine)
                {
                    const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                    const __fp16* bptr = (const __fp16*)beta_data_fp16;

                    for (int j = 0; j < size; j++)
                    {
                        float32x4_t _gamma = vdupq_n_f32(*gptr);
                        float32x4_t _beta = vdupq_n_f32(*bptr);

                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                        _p = vmlaq_f32(_b0, _p, _a0);
                        _p = vmlaq_f32(_beta, _p, _gamma);
                        vst1_f16(ptr0, vcvt_f16_f32(_p));
                        ptr0 += 4;

                        _p = vcvt_f32_f16(vld1_f16(ptr0));
                        _p = vmlaq_f32(_b1, _p, _a1);
                        _p = vmlaq_f32(_beta, _p, _gamma);
                        vst1_f16(ptr0, vcvt_f16_f32(_p));
                        ptr0 += 4;

                        gptr += 1;
                        bptr += 1;
                    }
                }
                else
                {
                    for (int j = 0; j < size; j++)
                    {
                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
                        _p = vmlaq_f32(_b0, _p, _a0);
                        vst1_f16(ptr0, vcvt_f16_f32(_p));
                        ptr0 += 4;

                        _p = vcvt_f32_f16(vld1_f16(ptr0));
                        _p = vmlaq_f32(_b1, _p, _a1);
                        vst1_f16(ptr0, vcvt_f16_f32(_p));
                        ptr0 += 4;
                    }
                }
            }

            return 0;
        }
    }

    return 0;
}

int LayerNorm_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;
    int elembits = bottom_top_blob.elembits();

    // inplace操作，所以输入是什么pack，输出就是什么pack，有1，4，8要处理
    if (elempack == 8)
    {
        return forward_inplace_pack8_fp16s(bottom_top_blob, opt);
    }

    if (elempack == 4)
    {
        return forward_inplace_pack4_fp16s(bottom_top_blob, opt);
    }

    if (elempack == 1)
    {
        return forward_inplace_pack1_fp16s(bottom_top_blob, opt);
    }

    return 0;
}

int LayerNorm_arm::forward_inplace_pack1_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        __fp16* ptr0 = (__fp16*)bottom_top_blob;

        // sum
        __fp16 sum = 0.f;
        float16x4_t _sum = vdup_n_f16(0.f);
        const __fp16* ptr = ptr0;
        int i = 0;
        for (; i + 3 < w; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _sum = vadd_f16(_sum, _p);
            ptr += 4;
        }
        sum += sum_float16x4(_sum);
        for (; i < w; i++)
        {
            sum += *ptr;
            ptr++;
        }

        // mean
        __fp16 mean = sum / w;
        float16x4_t _mean = vdup_n_f16(mean);

        // sum
        ptr = ptr0;
        __fp16 sqsum = 0.f;
        __fp16 tmp = 0.f;
        float16x4_t _sqsum = vdup_n_f16(0.f);
        i = 0;
        for (; i + 3 < w; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            float16x4_t _tmp = vsub_f16(_p, _mean);
            _sqsum = vfma_f16(_sqsum, _tmp, _tmp);
            ptr += 4;
        }
        sqsum += sum_float16x4(_sqsum);
        for (; i < w; i++)
        {
            tmp = *ptr - mean;
            sqsum += tmp * tmp;
            ptr++;
        }

        // var
        __fp16 var = sqsum / w;

        // coefficient
        __fp16 a = static_cast<__fp16>(1.f / (sqrt(var + eps)));
        __fp16 b = -mean * a;
        float16x4_t _a = vdup_n_f16(a);
        float16x4_t _b = vdup_n_f16(b);

        // affine
        if (affine)
        {
            const __fp16* gptr = (const __fp16*)gamma_data_fp16;
            const __fp16* bptr = (const __fp16*)beta_data_fp16;

            i = 0;
            for (; i + 3 < w; i += 4)
            {
                float16x4_t _p = vld1_f16(ptr0);
                _p = vfma_f16(_b, _p, _a);
                float16x4_t _gamma = vld1_f16(gptr);
                float16x4_t _beta = vld1_f16(bptr);
                _p = vfma_f16(_beta, _p, _gamma);
                vst1_f16(ptr0, _p);
                ptr0 += 4;
                gptr += 4;
                bptr += 4;
            }
            for (; i < w; i++)
            {
                *ptr0 = ((*ptr0) * a + b) * (*gptr) + (*bptr);
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
                float16x4_t _p = vld1_f16(ptr0);
                _p = vfma_f16(_b, _p, _a);
                vst1_f16(ptr0, _p);
                ptr0 += 4;
            }
            for (; i < w; i++)
            {
                *ptr0 = (*ptr0) * a + b;
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
            __fp16* ptr0 = (__fp16*)bottom_top_blob.row<__fp16>(i);

            // sum
            __fp16 sum = 0.f;
            float16x4_t _sum = vdup_n_f16(0.f);
            const __fp16* ptr = ptr0;
            int j = 0;
            for (; j + 3 < w; j += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                _sum = vadd_f16(_sum, _p);
                ptr += 4;
            }
            sum += sum_float16x4(_sum);
            for (; j < w; j++)
            {
                sum += *ptr;
                ptr++;
            }

            // mean
            __fp16 mean = sum / w;
            float16x4_t _mean = vdup_n_f16(mean);

            // sum
            ptr = ptr0;
            __fp16 sqsum = 0.f;
            __fp16 tmp = 0.f;
            float16x4_t _sqsum = vdup_n_f16(0.f);
            j = 0;
            for (; j + 3 < w; j += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                float16x4_t _tmp = vsub_f16(_p, _mean);
                _sqsum = vfma_f16(_sqsum, _tmp, _tmp);
                ptr += 4;
            }
            sqsum += sum_float16x4(_sqsum);
            for (; j < w; j++)
            {
                tmp = *ptr - mean;
                sqsum += tmp * tmp;
                ptr++;
            }

            // var
            __fp16 var = sqsum / w;

            // coefficient
            __fp16 a = static_cast<__fp16>(1.f / (sqrt(var + eps)));
            __fp16 b = -mean * a;
            float16x4_t _a = vdup_n_f16(a);
            float16x4_t _b = vdup_n_f16(b);

            // affine
            if (affine)
            {
                const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                const __fp16* bptr = (const __fp16*)beta_data_fp16;

                j = 0;
                for (; j + 3 < w; j += 4)
                {
                    float16x4_t _p = vld1_f16(ptr0);
                    _p = vfma_f16(_b, _p, _a);
                    float16x4_t _gamma = vld1_f16(gptr);
                    float16x4_t _beta = vld1_f16(bptr);
                    _p = vfma_f16(_beta, _p, _gamma);
                    vst1_f16(ptr0, _p);
                    ptr0 += 4;
                    gptr += 4;
                    bptr += 4;
                }
                for (; j < w; j++)
                {
                    *ptr0 = ((*ptr0) * a + b) * (*gptr) + (*bptr);
                    ptr0 += 1;
                    gptr += 1;
                    bptr += 1;
                }
            }
            else
            {
                j = 0;
                for (; j + 3 < w; j += 4)
                {
                    float16x4_t _p = vld1_f16(ptr0);
                    _p = vfma_f16(_b, _p, _a);
                    vst1_f16(ptr0, _p);
                    ptr0 += 4;
                }
                for (; j < w; j++)
                {
                    *ptr0 = (*ptr0) * a + b;
                    ptr0 += 1;
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
                    __fp16* ptr0 = (__fp16*)bottom_top_blob.channel(q).row<__fp16>(i);

                    // sum
                    __fp16 sum = 0.f;
                    float16x4_t _sum = vdup_n_f16(0.f);
                    const __fp16* ptr = ptr0;
                    int j = 0;
                    for (; j + 3 < w; j += 4)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        _sum = vadd_f16(_sum, _p);
                        ptr += 4;
                    }
                    sum += sum_float16x4(_sum);
                    for (; j < w; j++)
                    {
                        sum += *ptr;
                        ptr++;
                    }

                    // mean
                    __fp16 mean = sum / w;
                    float16x4_t _mean = vdup_n_f16(mean);

                    // sum
                    ptr = ptr0;
                    __fp16 sqsum = 0.f;
                    __fp16 tmp = 0.f;
                    float16x4_t _sqsum = vdup_n_f16(0.f);
                    j = 0;
                    for (; j + 3 < w; j += 4)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _tmp = vsub_f16(_p, _mean);
                        _sqsum = vfma_f16(_sqsum, _tmp, _tmp);
                        ptr += 4;
                    }
                    sqsum += sum_float16x4(_sqsum);
                    for (; j < w; j++)
                    {
                        tmp = *ptr - mean;
                        sqsum += tmp * tmp;
                        ptr++;
                    }

                    // var
                    __fp16 var = sqsum / w;

                    // coefficient
                    __fp16 a = static_cast<__fp16>(1.f / (sqrt(var + eps)));
                    __fp16 b = -mean * a;
                    float16x4_t _a = vdup_n_f16(a);
                    float16x4_t _b = vdup_n_f16(b);

                    // affine
                    if (affine)
                    {
                        const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                        const __fp16* bptr = (const __fp16*)beta_data_fp16;

                        j = 0;
                        for (; j + 3 < w; j += 4)
                        {
                            float16x4_t _p = vld1_f16(ptr0);
                            _p = vfma_f16(_b, _p, _a);
                            float16x4_t _gamma = vld1_f16(gptr);
                            float16x4_t _beta = vld1_f16(bptr);
                            _p = vfma_f16(_beta, _p, _gamma);
                            vst1_f16(ptr0, _p);
                            ptr0 += 4;
                            gptr += 4;
                            bptr += 4;
                        }
                        for (; j < w; j++)
                        {
                            *ptr0 = ((*ptr0) * a + b) * (*gptr) + (*bptr);
                            ptr0 += 1;
                            gptr += 1;
                            bptr += 1;
                        }
                    }
                    else
                    {
                        j = 0;
                        for (; j + 3 < w; j += 4)
                        {
                            float16x4_t _p = vld1_f16(ptr0);
                            _p = vfma_f16(_b, _p, _a);
                            vst1_f16(ptr0, _p);
                            ptr0 += 4;
                        }
                        for (; j < w; j++)
                        {
                            *ptr0 = (*ptr0) * a + b;
                            ptr0 += 1;
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
                __fp16* ptr0 = (__fp16*)bottom_top_blob.channel(q);

                // sum
                __fp16 sum = 0.f;
                float16x4_t _sum = vdup_n_f16(0.f);
                const __fp16* ptr = ptr0;
                int j = 0;
                for (; j + 3 < size; j += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    _sum = vadd_f16(_sum, _p);
                    ptr += 4;
                }
                sum += sum_float16x4(_sum);
                for (; j < size; j++)
                {
                    sum += *ptr;
                    ptr++;
                }

                // mean
                __fp16 mean = sum / size;
                float16x4_t _mean = vdup_n_f16(mean);

                // sum
                ptr = ptr0;
                __fp16 sqsum = 0.f;
                __fp16 tmp = 0.f;
                float16x4_t _sqsum = vdup_n_f16(0.f);
                j = 0;
                for (; j + 3 < size; j += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _tmp = vsub_f16(_p, _mean);
                    _sqsum = vfma_f16(_sqsum, _tmp, _tmp);
                    ptr += 4;
                }
                sqsum += sum_float16x4(_sqsum);
                for (; j < size; j++)
                {
                    tmp = *ptr - mean;
                    sqsum += tmp * tmp;
                    ptr++;
                }

                // var
                __fp16 var = sqsum / size;

                // coefficient
                __fp16 a = static_cast<__fp16>(1.f / (sqrt(var + eps)));
                __fp16 b = -mean * a;
                float16x4_t _a = vdup_n_f16(a);
                float16x4_t _b = vdup_n_f16(b);

                // affine
                if (affine)
                {
                    const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                    const __fp16* bptr = (const __fp16*)beta_data_fp16;

                    j = 0;
                    for (; j + 3 < size; j += 4)
                    {
                        float16x4_t _p = vld1_f16(ptr0);
                        _p = vfma_f16(_b, _p, _a);
                        float16x4_t _gamma = vld1_f16(gptr);
                        float16x4_t _beta = vld1_f16(bptr);
                        _p = vfma_f16(_beta, _p, _gamma);
                        vst1_f16(ptr0, _p);
                        ptr0 += 4;
                        gptr += 4;
                        bptr += 4;
                    }
                    for (; j < size; j++)
                    {
                        *ptr0 = ((*ptr0) * a + b) * (*gptr) + (*bptr);
                        ptr0 += 1;
                        gptr += 1;
                        bptr += 1;
                    }
                }
                else
                {
                    j = 0;
                    for (; j + 3 < size; j += 4)
                    {
                        float16x4_t _p = vld1_f16(ptr0);
                        _p = vfma_f16(_b, _p, _a);
                        vst1_f16(ptr0, _p);
                        ptr0 += 4;
                    }
                    for (; j < size; j++)
                    {
                        *ptr0 = (*ptr0) * a + b;
                        ptr0 += 1;
                    }
                }
            }

            return 0;
        }
    }

    return 0;
}

int LayerNorm_arm::forward_inplace_pack4_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        __fp16* ptr0 = (__fp16*)bottom_top_blob;

        // sum
        __fp16 sum = 0.f;
        float16x4_t _sum = vdup_n_f16(0.f);
        const __fp16* ptr = ptr0;
        for (int i = 0; i < w; i++)
        {
            float16x4_t _p = vld1_f16(ptr);
            _sum = vadd_f16(_sum, _p);
            ptr += 4;
        }
        sum += sum_float16x4(_sum);

        // mean
        __fp16 mean = 0.25 * sum / w;
        float16x4_t _mean = vdup_n_f16(mean);

        // sum
        ptr = ptr0;
        __fp16 sqsum = 0.f;
        float16x4_t _sqsum = vdup_n_f16(0.f);
        for (int i = 0; i < w; i++)
        {
            float16x4_t _p = vld1_f16(ptr);
            float16x4_t _tmp = vsub_f16(_p, _mean);
            _sqsum = vfma_f16(_sqsum, _tmp, _tmp);
            ptr += 4;
        }
        sqsum += sum_float16x4(_sqsum);

        // var
        __fp16 var = 0.25 * sqsum / w;

        // coefficient
        __fp16 a = static_cast<__fp16>(1.f / (sqrt(var + eps)));
        __fp16 b = -mean * a;
        float16x4_t _a = vdup_n_f16(a);
        float16x4_t _b = vdup_n_f16(b);

        // affine
        if (affine)
        {
            const __fp16* gptr = (const __fp16*)gamma_data_fp16;
            const __fp16* bptr = (const __fp16*)beta_data_fp16;

            for (int i = 0; i < w; i++)
            {
                float16x4_t _p = vld1_f16(ptr0);
                _p = vfma_f16(_b, _p, _a);
                float16x4_t _gamma = vld1_f16(gptr);
                float16x4_t _beta = vld1_f16(bptr);
                _p = vfma_f16(_beta, _p, _gamma);
                vst1_f16(ptr0, _p);
                ptr0 += 4;
                gptr += 4;
                bptr += 4;
            }
        }
        else
        {
            for (int i = 0; i < w; i++)
            {
                float16x4_t _p = vld1_f16(ptr0);
                _p = vfma_f16(_b, _p, _a);
                vst1_f16(ptr0, _p);
                ptr0 += 4;
            }
        }

        return 0;
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        float16x4_t _div_size = vdup_n_f16(1.f / w);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr0 = (__fp16*)bottom_top_blob.row<__fp16>(i);

            // sum
            float16x4_t _sum = vdup_n_f16(0.f);
            const __fp16* ptr = ptr0;
            for (int j = 0; j < w; j++)
            {
                float16x4_t _p = vld1_f16(ptr);
                _sum = vadd_f16(_sum, _p);
                ptr += 4;
            }

            // mean
            float16x4_t _mean = vmul_f16(_sum, _div_size);

            // sum
            ptr = ptr0;
            float16x4_t _sqsum = vdup_n_f16(0.f);
            for (int j = 0; j < w; j++)
            {
                float16x4_t _p = vld1_f16(ptr);
                float16x4_t _tmp = vsub_f16(_p, _mean);
                _sqsum = vfma_f16(_sqsum, _tmp, _tmp);
                ptr += 4;
            }

            // var
            float16x4_t _var_eps = vfma_f16(vdup_n_f16(eps), _sqsum, _div_size);

            // coefficient
            float16x4_t _a = vrsqrte_f16(_var_eps);
            _a = vmul_f16(vrsqrts_f16(vmul_f16(_var_eps, _a), _a), _a);
            float16x4_t _b = vfms_f16(vdup_n_f16(0.f), _mean, _a);

            // affine
            if (affine)
            {
                const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                const __fp16* bptr = (const __fp16*)beta_data_fp16;

                for (int j = 0; j < w; j++)
                {
                    float16x4_t _p = vld1_f16(ptr0);
                    _p = vfma_f16(_b, _p, _a);
                    float16x4_t _gamma = vdup_n_f16(*gptr);
                    float16x4_t _beta = vdup_n_f16(*bptr);
                    _p = vfma_f16(_beta, _p, _gamma);
                    vst1_f16(ptr0, _p);
                    ptr0 += 4;
                    gptr += 1;
                    bptr += 1;
                }
            }
            else
            {
                for (int j = 0; j < w; j++)
                {
                    float16x4_t _p = vld1_f16(ptr0);
                    _p = vfma_f16(_b, _p, _a);
                    vst1_f16(ptr0, _p);
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

        if (affine_size == w)
        {
            float16x4_t _div_size = vdup_n_f16(1.f / w);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    __fp16* ptr0 = (__fp16*)bottom_top_blob.channel(q).row<__fp16>(i);

                    // sum
                    float16x4_t _sum = vdup_n_f16(0.f);
                    const __fp16* ptr = ptr0;
                    for (int j = 0; j < w; j++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        _sum = vadd_f16(_sum, _p);
                        ptr += 4;
                    }

                    // mean
                    float16x4_t _mean = vmul_f16(_sum, _div_size);

                    // sum
                    ptr = ptr0;
                    float16x4_t _sqsum = vdup_n_f16(0.f);
                    for (int j = 0; j < w; j++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _tmp = vsub_f16(_p, _mean);
                        _sqsum = vfma_f16(_sqsum, _tmp, _tmp);
                        ptr += 4;
                    }

                    // var
                    float16x4_t _var_eps = vfma_f16(vdup_n_f16(eps), _sqsum, _div_size);

                    // coefficient
                    float16x4_t _a = vrsqrte_f16(_var_eps);
                    _a = vmul_f16(vrsqrts_f16(vmul_f16(_var_eps, _a), _a), _a);
                    float16x4_t _b = vfms_f16(vdup_n_f16(0.f), _mean, _a);

                    // affine
                    if (affine)
                    {
                        const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                        const __fp16* bptr = (const __fp16*)beta_data_fp16;

                        for (int j = 0; j < w; j++)
                        {
                            float16x4_t _p = vld1_f16(ptr0);
                            _p = vfma_f16(_b, _p, _a);
                            float16x4_t _gamma = vdup_n_f16(*gptr);
                            float16x4_t _beta = vdup_n_f16(*bptr);
                            _p = vfma_f16(_beta, _p, _gamma);
                            vst1_f16(ptr0, _p);
                            ptr0 += 4;
                            gptr += 1;
                            bptr += 1;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < w; j++)
                        {
                            float16x4_t _p = vld1_f16(ptr0);
                            _p = vfma_f16(_b, _p, _a);
                            vst1_f16(ptr0, _p);
                            ptr0 += 4;
                        }
                    }
                }
            }

            return 0;
        }
        else
        {
            float16x4_t _div_size = vdup_n_f16(1.f / size);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                __fp16* ptr0 = (__fp16*)bottom_top_blob.channel(q);

                // sum
                float16x4_t _sum = vdup_n_f16(0.f);
                const __fp16* ptr = ptr0;
                for (int j = 0; j < size; j++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    _sum = vadd_f16(_sum, _p);
                    ptr += 4;
                }

                // mean
                float16x4_t _mean = vmul_f16(_sum, _div_size);

                // sum
                ptr = ptr0;
                float16x4_t _sqsum = vdup_n_f16(0.f);
                for (int j = 0; j < size; j++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _tmp = vsub_f16(_p, _mean);
                    _sqsum = vfma_f16(_sqsum, _tmp, _tmp);
                    ptr += 4;
                }

                // var
                float16x4_t _var_eps = vfma_f16(vdup_n_f16(eps), _sqsum, _div_size);

                // coefficient
                float16x4_t _a = vrsqrte_f16(_var_eps);
                _a = vmul_f16(vrsqrts_f16(vmul_f16(_var_eps, _a), _a), _a);
                float16x4_t _b = vfms_f16(vdup_n_f16(0.f), _mean, _a);

                // affine
                if (affine)
                {
                    const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                    const __fp16* bptr = (const __fp16*)beta_data_fp16;

                    for (int j = 0; j < size; j++)
                    {
                        float16x4_t _p = vld1_f16(ptr0);
                        _p = vfma_f16(_b, _p, _a);
                        float16x4_t _gamma = vdup_n_f16(*gptr);
                        float16x4_t _beta = vdup_n_f16(*bptr);
                        _p = vfma_f16(_beta, _p, _gamma);
                        vst1_f16(ptr0, _p);
                        ptr0 += 4;
                        gptr += 1;
                        bptr += 1;
                    }
                }
                else
                {
                    for (int j = 0; j < size; j++)
                    {
                        float16x4_t _p = vld1_f16(ptr0);
                        _p = vfma_f16(_b, _p, _a);
                        vst1_f16(ptr0, _p);
                        ptr0 += 4;
                    }
                }
            }

            return 0;
        }
    }

    return 0;
}

int LayerNorm_arm::forward_inplace_pack8_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        __fp16* ptr0 = (__fp16*)bottom_top_blob;

        // sum
        __fp16 sum = 0.f;
        float16x8_t _sum = vdupq_n_f16(0.f);
        const __fp16* ptr = ptr0;
        for (int i = 0; i < w; i++)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _sum = vaddq_f16(_sum, _p);
            ptr += 8;
        }
        sum += sum_float16x8(_sum);

        // mean
        __fp16 mean = 0.125 * sum / w;
        float16x8_t _mean = vdupq_n_f16(mean);

        // sum
        ptr = ptr0;
        __fp16 sqsum = 0.f;
        float16x8_t _sqsum = vdupq_n_f16(0.f);
        for (int i = 0; i < w; i++)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float16x8_t _tmp = vsubq_f16(_p, _mean);
            _sqsum = vfmaq_f16(_sqsum, _tmp, _tmp);
            ptr += 8;
        }
        sqsum += sum_float16x8(_sqsum);

        // var
        __fp16 var = 0.125 * sqsum / w;

        // coefficient
        __fp16 a = static_cast<__fp16>(1.f / (sqrt(var + eps)));
        __fp16 b = -mean * a;
        float16x8_t _a = vdupq_n_f16(a);
        float16x8_t _b = vdupq_n_f16(b);

        // affine
        if (affine)
        {
            const __fp16* gptr = (const __fp16*)gamma_data_fp16;
            const __fp16* bptr = (const __fp16*)beta_data_fp16;

            for (int i = 0; i < w; i++)
            {
                float16x8_t _p = vld1q_f16(ptr0);
                _p = vfmaq_f16(_b, _p, _a);
                float16x8_t _gamma = vld1q_f16(gptr);
                float16x8_t _beta = vld1q_f16(bptr);
                _p = vfmaq_f16(_beta, _p, _gamma);
                vst1q_f16(ptr0, _p);
                ptr0 += 8;
                gptr += 8;
                bptr += 8;
            }
        }
        else
        {
            for (int i = 0; i < w; i++)
            {
                float16x8_t _p = vld1q_f16(ptr0);
                _p = vfmaq_f16(_b, _p, _a);
                vst1q_f16(ptr0, _p);
                ptr0 += 8;
            }
        }

        return 0;
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        float16x8_t _div_size = vdupq_n_f16(1.f / w);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr0 = (__fp16*)bottom_top_blob.row<__fp16>(i);

            // sum
            float16x8_t _sum = vdupq_n_f16(0.f);
            const __fp16* ptr = ptr0;
            for (int j = 0; j < w; j++)
            {
                float16x8_t _p = vld1q_f16(ptr);
                _sum = vaddq_f16(_sum, _p);
                ptr += 8;
            }

            // mean
            float16x8_t _mean = vmulq_f16(_sum, _div_size);

            // sum
            ptr = ptr0;
            float16x8_t _sqsum = vdupq_n_f16(0.f);
            for (int j = 0; j < w; j++)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float16x8_t _tmp = vsubq_f16(_p, _mean);
                _sqsum = vfmaq_f16(_sqsum, _tmp, _tmp);
                ptr += 8;
            }

            // var
            float16x8_t _var_eps = vfmaq_f16(vdupq_n_f16(eps), _sqsum, _div_size);

            // coefficient
            float16x8_t _a = vrsqrteq_f16(_var_eps);
            _a = vmulq_f16(vrsqrtsq_f16(vmulq_f16(_var_eps, _a), _a), _a);
            float16x8_t _b = vfmsq_f16(vdupq_n_f16(0.f), _mean, _a);

            // affine
            if (affine)
            {
                const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                const __fp16* bptr = (const __fp16*)beta_data_fp16;

                for (int j = 0; j < w; j++)
                {
                    float16x8_t _p = vld1q_f16(ptr0);
                    _p = vfmaq_f16(_b, _p, _a);
                    float16x8_t _gamma = vdupq_n_f16(*gptr);
                    float16x8_t _beta = vdupq_n_f16(*bptr);
                    _p = vfmaq_f16(_beta, _p, _gamma);
                    vst1q_f16(ptr0, _p);
                    ptr0 += 8;
                    gptr += 1;
                    bptr += 1;
                }
            }
            else
            {
                for (int j = 0; j < w; j++)
                {
                    float16x8_t _p = vld1q_f16(ptr0);
                    _p = vfmaq_f16(_b, _p, _a);
                    vst1q_f16(ptr0, _p);
                    ptr0 += 8;
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
            float16x8_t _div_size = vdupq_n_f16(1.f / w);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    __fp16* ptr0 = (__fp16*)bottom_top_blob.channel(q).row<__fp16>(i);

                    // sum
                    float16x8_t _sum = vdupq_n_f16(0.f);
                    const __fp16* ptr = ptr0;
                    for (int j = 0; j < w; j++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        _sum = vaddq_f16(_sum, _p);
                        ptr += 8;
                    }

                    // mean
                    float16x8_t _mean = vmulq_f16(_sum, _div_size);

                    // sum
                    ptr = ptr0;
                    float16x8_t _sqsum = vdupq_n_f16(0.f);
                    for (int j = 0; j < w; j++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        float16x8_t _tmp = vsubq_f16(_p, _mean);
                        _sqsum = vfmaq_f16(_sqsum, _tmp, _tmp);
                        ptr += 8;
                    }

                    // var
                    float16x8_t _var_eps = vfmaq_f16(vdupq_n_f16(eps), _sqsum, _div_size);

                    // coefficient
                    float16x8_t _a = vrsqrteq_f16(_var_eps);
                    _a = vmulq_f16(vrsqrtsq_f16(vmulq_f16(_var_eps, _a), _a), _a);
                    float16x8_t _b = vfmsq_f16(vdupq_n_f16(0.f), _mean, _a);

                    // affine
                    if (affine)
                    {
                        const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                        const __fp16* bptr = (const __fp16*)beta_data_fp16;

                        for (int j = 0; j < w; j++)
                        {
                            float16x8_t _p = vld1q_f16(ptr0);
                            _p = vfmaq_f16(_b, _p, _a);
                            float16x8_t _gamma = vdupq_n_f16(*gptr);
                            float16x8_t _beta = vdupq_n_f16(*bptr);
                            _p = vfmaq_f16(_beta, _p, _gamma);
                            vst1q_f16(ptr0, _p);
                            ptr0 += 8;
                            gptr += 1;
                            bptr += 1;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < w; j++)
                        {
                            float16x8_t _p = vld1q_f16(ptr0);
                            _p = vfmaq_f16(_b, _p, _a);
                            vst1q_f16(ptr0, _p);
                            ptr0 += 8;
                        }
                    }
                }
            }

            return 0;
        }
        else
        {
            float16x8_t _div_size = vdupq_n_f16(1.f / size);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                __fp16* ptr0 = (__fp16*)bottom_top_blob.channel(q);

                // sum
                float16x8_t _sum = vdupq_n_f16(0.f);
                const __fp16* ptr = ptr0;
                for (int j = 0; j < size; j++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    _sum = vaddq_f16(_sum, _p);
                    ptr += 8;
                }

                // mean
                float16x8_t _mean = vmulq_f16(_sum, _div_size);

                // sum
                ptr = ptr0;
                float16x8_t _sqsum = vdupq_n_f16(0.f);
                for (int j = 0; j < size; j++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _tmp = vsubq_f16(_p, _mean);
                    _sqsum = vfmaq_f16(_sqsum, _tmp, _tmp);
                    ptr += 8;
                }

                // var
                float16x8_t _var_eps = vfmaq_f16(vdupq_n_f16(eps), _sqsum, _div_size);

                // coefficient
                float16x8_t _a = vrsqrteq_f16(_var_eps);
                _a = vmulq_f16(vrsqrtsq_f16(vmulq_f16(_var_eps, _a), _a), _a);
                float16x8_t _b = vfmsq_f16(vdupq_n_f16(0.f), _mean, _a);

                // affine
                if (affine)
                {
                    const __fp16* gptr = (const __fp16*)gamma_data_fp16;
                    const __fp16* bptr = (const __fp16*)beta_data_fp16;

                    for (int j = 0; j < size; j++)
                    {
                        float16x8_t _p = vld1q_f16(ptr0);
                        _p = vfmaq_f16(_b, _p, _a);
                        float16x8_t _gamma = vdupq_n_f16(*gptr);
                        float16x8_t _beta = vdupq_n_f16(*bptr);
                        _p = vfmaq_f16(_beta, _p, _gamma);
                        vst1q_f16(ptr0, _p);
                        ptr0 += 8;
                        gptr += 1;
                        bptr += 1;
                    }
                }
                else
                {
                    for (int j = 0; j < size; j++)
                    {
                        float16x8_t _p = vld1q_f16(ptr0);
                        _p = vfmaq_f16(_b, _p, _a);
                        vst1q_f16(ptr0, _p);
                        ptr0 += 8;
                    }
                }
            }

            return 0;
        }
    }

    return 0;
}

int LayerNorm_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;
    int elembits = bottom_top_blob.elembits();

    if (elempack == 8)
    {
        return forward_inplace_pack8_fp16sa(bottom_top_blob, opt);
    }

    if (elempack == 4)
    {
        return forward_inplace_pack4_fp16sa(bottom_top_blob, opt);
    }

    if (elempack == 1)
    {
        return forward_inplace_pack1_fp16sa(bottom_top_blob, opt);
    }

    return 0;
}
#endif

} // namespace ncnn
