// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "instancenorm_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

InstanceNorm_loongarch::InstanceNorm_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int InstanceNorm_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

#if __loongarch_sx
    int elempack = bottom_top_blob.elempack;
#if __loongarch_asx
    if (elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            __m256 _sum = (__m256)__lasx_xvreplfr2vr_s(0.f);
            const float* ptr0 = ptr;
            for (int i = 0; i < size; i++)
            {
                __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
                _sum = __lasx_xvfadd_s(_sum, _p);
                ptr0 += 8;
            }

            float sum_data[8];
            __lasx_xvst(_sum, sum_data, 0);

            float mean_data[8];
            for (int i = 0; i < 8; i++)
            {
                mean_data[i] = sum_data[i] / size;
            }
            __m256 _mean = (__m256)__lasx_xvld(mean_data, 0);

            __m256 _sqsum = (__m256)__lasx_xvreplfr2vr_s(0.f);
            ptr0 = ptr;
            for (int i = 0; i < size; i++)
            {
                __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
                _p = __lasx_xvfsub_s(_p, _mean);
                _sqsum = __lasx_xvfmadd_s(_p, _p, _sqsum);
                ptr0 += 8;
            }

            float sqsum_data[8];
            __lasx_xvst(_sqsum, sqsum_data, 0);

            float a_data[8];
            float b_data[8];
            if (affine)
            {
                const float* gamma_ptr = (const float*)gamma_data + q * 8;
                const float* beta_ptr = (const float*)beta_data + q * 8;

                for (int i = 0; i < 8; i++)
                {
                    float a = gamma_ptr[i] / sqrtf(sqsum_data[i] / size + eps);
                    a_data[i] = a;
                    b_data[i] = -mean_data[i] * a + beta_ptr[i];
                }
            }
            else
            {
                for (int i = 0; i < 8; i++)
                {
                    float a = 1.f / sqrtf(sqsum_data[i] / size + eps);
                    a_data[i] = a;
                    b_data[i] = -mean_data[i] * a;
                }
            }

            __m256 _a = (__m256)__lasx_xvld(a_data, 0);
            __m256 _b = (__m256)__lasx_xvld(b_data, 0);

            for (int i = 0; i < size; i++)
            {
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                _p = __lasx_xvfmadd_s(_p, _a, _b);
                __lasx_xvst(_p, ptr, 0);
                ptr += 8;
            }
        }

        return 0;
    }
#endif // __loongarch_asx

    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            __m128 _sum = (__m128)__lsx_vreplfr2vr_s(0.f);
            const float* ptr0 = ptr;
            for (int i = 0; i < size; i++)
            {
                __m128 _p = (__m128)__lsx_vld(ptr0, 0);
                _sum = __lsx_vfadd_s(_sum, _p);
                ptr0 += 4;
            }

            float sum_data[4];
            __lsx_vst(_sum, sum_data, 0);

            float mean_data[4];
            for (int i = 0; i < 4; i++)
            {
                mean_data[i] = sum_data[i] / size;
            }
            __m128 _mean = (__m128)__lsx_vld(mean_data, 0);

            __m128 _sqsum = (__m128)__lsx_vreplfr2vr_s(0.f);
            ptr0 = ptr;
            for (int i = 0; i < size; i++)
            {
                __m128 _p = (__m128)__lsx_vld(ptr0, 0);
                _p = __lsx_vfsub_s(_p, _mean);
                _sqsum = __lsx_vfmadd_s(_p, _p, _sqsum);
                ptr0 += 4;
            }

            float sqsum_data[4];
            __lsx_vst(_sqsum, sqsum_data, 0);

            float a_data[4];
            float b_data[4];
            if (affine)
            {
                const float* gamma_ptr = (const float*)gamma_data + q * 4;
                const float* beta_ptr = (const float*)beta_data + q * 4;

                for (int i = 0; i < 4; i++)
                {
                    float a = gamma_ptr[i] / sqrtf(sqsum_data[i] / size + eps);
                    a_data[i] = a;
                    b_data[i] = -mean_data[i] * a + beta_ptr[i];
                }
            }
            else
            {
                for (int i = 0; i < 4; i++)
                {
                    float a = 1.f / sqrtf(sqsum_data[i] / size + eps);
                    a_data[i] = a;
                    b_data[i] = -mean_data[i] * a;
                }
            }

            __m128 _a = (__m128)__lsx_vld(a_data, 0);
            __m128 _b = (__m128)__lsx_vld(b_data, 0);

            for (int i = 0; i < size; i++)
            {
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                _p = __lsx_vfmadd_s(_p, _a, _b);
                __lsx_vst(_p, ptr, 0);
                ptr += 4;
            }
        }

        return 0;
    }
#endif // __loongarch_sx

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        const float* ptr0 = ptr;

        // mean and var
        float sum = 0.f;
        float sqsum = 0.f;

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _sum_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
            _sum_lasx = __lasx_xvfadd_s(_sum_lasx, _p);
            ptr0 += 8;
        }
        sum += __lasx_reduce_fadd_s(_sum_lasx);
#endif // __loongarch_asx
        __m128 _sum_lsx = (__m128)__lsx_vreplfr2vr_s(0.f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
            _sum_lsx = __lsx_vfadd_s(_sum_lsx, _p);
            ptr0 += 4;
        }
        sum += __lsx_reduce_fadd_s(_sum_lsx);
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            sum += *ptr0++;
        }

        float mean = sum / size;

        ptr0 = ptr;
        i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _sqsum_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
        __m256 _mean_lasx = (__m256)__lasx_xvreplfr2vr_s(mean);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
            _p = __lasx_xvfsub_s(_p, _mean_lasx);
            _sqsum_lasx = __lasx_xvfmadd_s(_p, _p, _sqsum_lasx);
            ptr0 += 8;
        }
        sqsum += __lasx_reduce_fadd_s(_sqsum_lasx);
#endif // __loongarch_asx
        __m128 _sqsum_lsx = (__m128)__lsx_vreplfr2vr_s(0.f);
        __m128 _mean_lsx = (__m128)__lsx_vreplfr2vr_s(mean);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
            _p = __lsx_vfsub_s(_p, _mean_lsx);
            _sqsum_lsx = __lsx_vfmadd_s(_p, _p, _sqsum_lsx);
            ptr0 += 4;
        }
        sqsum += __lsx_reduce_fadd_s(_sqsum_lsx);
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float tmp = *ptr0++ - mean;
            sqsum += tmp * tmp;
        }

        float var = sqsum / size;

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = gamma / sqrtf(var + eps);
            b = -mean * a + beta;
        }
        else
        {
            a = 1.f / sqrtf(var + eps);
            b = -mean * a;
        }

        i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _a_lasx = (__m256)__lasx_xvreplfr2vr_s(a);
        __m256 _b_lasx = (__m256)__lasx_xvreplfr2vr_s(b);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            _p = __lasx_xvfmadd_s(_p, _a_lasx, _b_lasx);
            __lasx_xvst(_p, ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _a_lsx = (__m128)__lsx_vreplfr2vr_s(a);
        __m128 _b_lsx = (__m128)__lsx_vreplfr2vr_s(b);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _p = __lsx_vfmadd_s(_p, _a_lsx, _b_lsx);
            __lsx_vst(_p, ptr, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr = *ptr * a + b;
            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int InstanceNorm_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

#if __loongarch_sx
    int elempack = bottom_top_blob.elempack;
#if __loongarch_asx
    if (elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q);

            __m256 _sum = (__m256)__lasx_xvreplfr2vr_s(0.f);
            const unsigned short* ptr0 = ptr;
            for (int i = 0; i < size; i++)
            {
                __m256 _p = bfloat2float_lasx((__m128i*)ptr0);
                _sum = __lasx_xvfadd_s(_sum, _p);
                ptr0 += 8;
            }

            float sum_data[8];
            __lasx_xvst(_sum, sum_data, 0);

            float mean_data[8];
            for (int i = 0; i < 8; i++)
            {
                mean_data[i] = sum_data[i] / size;
            }
            __m256 _mean = (__m256)__lasx_xvld(mean_data, 0);

            __m256 _sqsum = (__m256)__lasx_xvreplfr2vr_s(0.f);
            ptr0 = ptr;
            for (int i = 0; i < size; i++)
            {
                __m256 _p = bfloat2float_lasx((__m128i*)ptr0);
                _p = __lasx_xvfsub_s(_p, _mean);
                _sqsum = __lasx_xvfmadd_s(_p, _p, _sqsum);
                ptr0 += 8;
            }

            float sqsum_data[8];
            __lasx_xvst(_sqsum, sqsum_data, 0);

            float a_data[8];
            float b_data[8];
            if (affine)
            {
                const float* gamma_ptr = (const float*)gamma_data + q * 8;
                const float* beta_ptr = (const float*)beta_data + q * 8;

                for (int i = 0; i < 8; i++)
                {
                    float a = gamma_ptr[i] / sqrtf(sqsum_data[i] / size + eps);
                    a_data[i] = a;
                    b_data[i] = -mean_data[i] * a + beta_ptr[i];
                }
            }
            else
            {
                for (int i = 0; i < 8; i++)
                {
                    float a = 1.f / sqrtf(sqsum_data[i] / size + eps);
                    a_data[i] = a;
                    b_data[i] = -mean_data[i] * a;
                }
            }

            __m256 _a = (__m256)__lasx_xvld(a_data, 0);
            __m256 _b = (__m256)__lasx_xvld(b_data, 0);

            for (int i = 0; i < size; i++)
            {
                __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                _p = __lasx_xvfmadd_s(_p, _a, _b);
                __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
                ptr += 8;
            }
        }

        return 0;
    }
#endif // __loongarch_asx

    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q);

            __m128 _sum = (__m128)__lsx_vreplfr2vr_s(0.f);
            const unsigned short* ptr0 = ptr;
            for (int i = 0; i < size; i++)
            {
                __m128 _p = bfloat2float_lsx((__m128i*)ptr0);
                _sum = __lsx_vfadd_s(_sum, _p);
                ptr0 += 4;
            }

            float sum_data[4];
            __lsx_vst(_sum, sum_data, 0);

            float mean_data[4];
            for (int i = 0; i < 4; i++)
            {
                mean_data[i] = sum_data[i] / size;
            }
            __m128 _mean = (__m128)__lsx_vld(mean_data, 0);

            __m128 _sqsum = (__m128)__lsx_vreplfr2vr_s(0.f);
            ptr0 = ptr;
            for (int i = 0; i < size; i++)
            {
                __m128 _p = bfloat2float_lsx((__m128i*)ptr0);
                _p = __lsx_vfsub_s(_p, _mean);
                _sqsum = __lsx_vfmadd_s(_p, _p, _sqsum);
                ptr0 += 4;
            }

            float sqsum_data[4];
            __lsx_vst(_sqsum, sqsum_data, 0);

            float a_data[4];
            float b_data[4];
            if (affine)
            {
                const float* gamma_ptr = (const float*)gamma_data + q * 4;
                const float* beta_ptr = (const float*)beta_data + q * 4;

                for (int i = 0; i < 4; i++)
                {
                    float a = gamma_ptr[i] / sqrtf(sqsum_data[i] / size + eps);
                    a_data[i] = a;
                    b_data[i] = -mean_data[i] * a + beta_ptr[i];
                }
            }
            else
            {
                for (int i = 0; i < 4; i++)
                {
                    float a = 1.f / sqrtf(sqsum_data[i] / size + eps);
                    a_data[i] = a;
                    b_data[i] = -mean_data[i] * a;
                }
            }

            __m128 _a = (__m128)__lsx_vld(a_data, 0);
            __m128 _b = (__m128)__lsx_vld(b_data, 0);

            for (int i = 0; i < size; i++)
            {
                __m128 _p = bfloat2float_lsx((__m128i*)ptr);
                _p = __lsx_vfmadd_s(_p, _a, _b);
                __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
                ptr += 4;
            }
        }

        return 0;
    }
#endif // __loongarch_sx

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        // mean and var
        float sum = 0.f;
        float sqsum = 0.f;
        const unsigned short* ptr0 = ptr;

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _sum_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr0, 0));
            _sum_lasx = __lasx_xvfadd_s(_sum_lasx, _p);
            ptr0 += 8;
        }
        sum += __lasx_reduce_fadd_s(_sum_lasx);
#endif // __loongarch_asx
        __m128 _sum_lsx = (__m128)__lsx_vreplfr2vr_s(0.f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx(ptr0);
            _sum_lsx = __lsx_vfadd_s(_sum_lsx, _p);
            ptr0 += 4;
        }
        sum += __lsx_reduce_fadd_s(_sum_lsx);
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            sum += bfloat16_to_float32(*ptr0++);
        }

        float mean = sum / size;

        ptr0 = ptr;
        i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _sqsum_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
        __m256 _mean_lasx = (__m256)__lasx_xvreplfr2vr_s(mean);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr0, 0));
            _p = __lasx_xvfsub_s(_p, _mean_lasx);
            _sqsum_lasx = __lasx_xvfmadd_s(_p, _p, _sqsum_lasx);
            ptr0 += 8;
        }
        sqsum += __lasx_reduce_fadd_s(_sqsum_lasx);
#endif // __loongarch_asx
        __m128 _sqsum_lsx = (__m128)__lsx_vreplfr2vr_s(0.f);
        __m128 _mean_lsx = (__m128)__lsx_vreplfr2vr_s(mean);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx(ptr0);
            _p = __lsx_vfsub_s(_p, _mean_lsx);
            _sqsum_lsx = __lsx_vfmadd_s(_p, _p, _sqsum_lsx);
            ptr0 += 4;
        }
        sqsum += __lsx_reduce_fadd_s(_sqsum_lsx);
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float tmp = bfloat16_to_float32(*ptr0++) - mean;
            sqsum += tmp * tmp;
        }

        float var = sqsum / size;

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = gamma / sqrtf(var + eps);
            b = -mean * a + beta;
        }
        else
        {
            a = 1.f / sqrtf(var + eps);
            b = -mean * a;
        }

        i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _a_lasx = (__m256)__lasx_xvreplfr2vr_s(a);
        __m256 _b_lasx = (__m256)__lasx_xvreplfr2vr_s(b);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
            _p = __lasx_xvfmadd_s(_p, _a_lasx, _b_lasx);
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _a_lsx = (__m128)__lsx_vreplfr2vr_s(a);
        __m128 _b_lsx = (__m128)__lsx_vreplfr2vr_s(b);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx(ptr);
            _p = __lsx_vfmadd_s(_p, _a_lsx, _b_lsx);
            __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * a + b);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
