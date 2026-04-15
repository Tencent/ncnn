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

#if __loongarch_asx
    if (bottom_top_blob.elempack == 8)
    {
        const int channels = bottom_top_blob.c;
        const int size = bottom_top_blob.w * bottom_top_blob.h;

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

#if __loongarch_sx
    if (bottom_top_blob.elempack == 4)
    {
        const int channels = bottom_top_blob.c;
        const int size = bottom_top_blob.w * bottom_top_blob.h;

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

    return InstanceNorm::forward_inplace(bottom_top_blob, opt);
}

#if NCNN_BF16
int InstanceNorm_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
#if __loongarch_asx
    if (bottom_top_blob.elempack == 8)
    {
        const int channels = bottom_top_blob.c;
        const int size = bottom_top_blob.w * bottom_top_blob.h;

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

#if __loongarch_sx
    if (bottom_top_blob.elempack == 4)
    {
        const int channels = bottom_top_blob.c;
        const int size = bottom_top_blob.w * bottom_top_blob.h;

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

    // scalar fallback
    Option opt_cast = opt;
    opt_cast.blob_allocator = opt.workspace_allocator;

    Mat bottom_top_blob_fp32;
    cast_bfloat16_to_float32(bottom_top_blob, bottom_top_blob_fp32, opt_cast);
    if (bottom_top_blob_fp32.empty())
        return -100;

    int ret = forward_inplace(bottom_top_blob_fp32, opt);
    if (ret != 0)
        return ret;

    cast_float32_to_bfloat16(bottom_top_blob_fp32, bottom_top_blob, opt);
    if (bottom_top_blob.empty())
        return -100;

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
