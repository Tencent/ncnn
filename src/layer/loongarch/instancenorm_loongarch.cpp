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

    // x = (x - mean) / (sqrt(var + eps)) * gamma + beta

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int c = bottom_top_blob.c;
    int size = w * h * d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; q++)
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
        __m128 _sum = (__m128)__lsx_vreplfr2vr_s(0.f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
            _sum = __lsx_vfadd_s(_sum, _p);
            ptr0 += 4;
        }
        sum += __lsx_reduce_fadd_s(_sum);
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            sum += ptr0[0];
            ptr0++;
        }

        float mean = sum / size;
        float tmp = 0.f;

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
        __m128 _sqsum = (__m128)__lsx_vreplfr2vr_s(0.f);
        __m128 _mean = (__m128)__lsx_vreplfr2vr_s(mean);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
            _p = __lsx_vfsub_s(_p, _mean);
            _sqsum = __lsx_vfmadd_s(_p, _p, _sqsum);
            ptr0 += 4;
        }
        sqsum += __lsx_reduce_fadd_s(_sqsum);
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            tmp = ptr0[0] - mean;
            sqsum += tmp * tmp;
            ptr0++;
        }

        float var = sqsum / size;
        // the var maybe minus due to accuracy

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = gamma / (sqrtf(var + eps));
            b = -mean * a + beta;
        }
        else
        {
            a = 1.f / (sqrtf(var + eps));
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
        __m128 _a = (__m128)__lsx_vreplfr2vr_s(a);
        __m128 _b = (__m128)__lsx_vreplfr2vr_s(b);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _p = __lsx_vfmadd_s(_p, _a, _b);
            __lsx_vst(_p, ptr, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * a + b;
            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int InstanceNorm_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / (sqrt(var + eps)) * gamma + beta

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int c = bottom_top_blob.c;
    int size = w * h * d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);
        const unsigned short* ptr0 = ptr;

        // mean and var
        float sum = 0.f;
        float sqsum = 0.f;

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _sum_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i*)ptr0);
            _sum_lasx = __lasx_xvfadd_s(_sum_lasx, _p);
            ptr0 += 8;
        }
        sum += __lasx_reduce_fadd_s(_sum_lasx);
#endif // __loongarch_asx
        __m128 _sum = (__m128)__lsx_vreplfr2vr_s(0.f);
        __m128i _zero_bf16 = __lsx_vreplgr2vr_w(0);
        for (; i + 7 < size; i += 8)
        {
            __m128i _p01 = __lsx_vld(ptr0, 0);
            __m128 _p0 = (__m128)__lsx_vilvl_h(_p01, _zero_bf16);
            __m128 _p1 = (__m128)__lsx_vilvh_h(_p01, _zero_bf16);
            _sum = __lsx_vfadd_s(_sum, _p0);
            _sum = __lsx_vfadd_s(_sum, _p1);
            ptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx(__lsx_vldrepl_d(ptr0, 0));
            _sum = __lsx_vfadd_s(_sum, _p);
            ptr0 += 4;
        }
        sum += __lsx_reduce_fadd_s(_sum);
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            sum += bfloat16_to_float32(ptr0[0]);
            ptr0++;
        }

        float mean = sum / size;
        float tmp = 0.f;

        ptr0 = ptr;
        i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _sqsum_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
        __m256 _mean_lasx = (__m256)__lasx_xvreplfr2vr_s(mean);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i*)ptr0);
            _p = __lasx_xvfsub_s(_p, _mean_lasx);
            _sqsum_lasx = __lasx_xvfmadd_s(_p, _p, _sqsum_lasx);
            ptr0 += 8;
        }
        sqsum += __lasx_reduce_fadd_s(_sqsum_lasx);
#endif // __loongarch_asx
        __m128 _sqsum = (__m128)__lsx_vreplfr2vr_s(0.f);
        __m128 _mean = (__m128)__lsx_vreplfr2vr_s(mean);
        for (; i + 7 < size; i += 8)
        {
            __m128i _p01 = __lsx_vld(ptr0, 0);
            __m128 _p0 = (__m128)__lsx_vilvl_h(_p01, _zero_bf16);
            __m128 _p1 = (__m128)__lsx_vilvh_h(_p01, _zero_bf16);
            _p0 = __lsx_vfsub_s(_p0, _mean);
            _p1 = __lsx_vfsub_s(_p1, _mean);
            _sqsum = __lsx_vfmadd_s(_p0, _p0, _sqsum);
            _sqsum = __lsx_vfmadd_s(_p1, _p1, _sqsum);
            ptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx(__lsx_vldrepl_d(ptr0, 0));
            _p = __lsx_vfsub_s(_p, _mean);
            _sqsum = __lsx_vfmadd_s(_p, _p, _sqsum);
            ptr0 += 4;
        }
        sqsum += __lsx_reduce_fadd_s(_sqsum);
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            tmp = bfloat16_to_float32(ptr0[0]) - mean;
            sqsum += tmp * tmp;
            ptr0++;
        }

        float var = sqsum / size;
        // the var maybe minus due to accuracy

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = gamma / (sqrtf(var + eps));
            b = -mean * a + beta;
        }
        else
        {
            a = 1.f / (sqrtf(var + eps));
            b = -mean * a;
        }

        i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _a_lasx = (__m256)__lasx_xvreplfr2vr_s(a);
        __m256 _b_lasx = (__m256)__lasx_xvreplfr2vr_s(b);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i*)ptr);
            _p = __lasx_xvfmadd_s(_p, _a_lasx, _b_lasx);
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _a = (__m128)__lsx_vreplfr2vr_s(a);
        __m128 _b = (__m128)__lsx_vreplfr2vr_s(b);
        for (; i + 7 < size; i += 8)
        {
            __m128i _p01 = __lsx_vld(ptr, 0);
            __m128 _p0 = (__m128)__lsx_vilvl_h(_p01, _zero_bf16);
            __m128 _p1 = (__m128)__lsx_vilvh_h(_p01, _zero_bf16);
            _p0 = __lsx_vfmadd_s(_p0, _a, _b);
            _p1 = __lsx_vfmadd_s(_p1, _a, _b);
            __lsx_vst(float2bfloat_lsx(_p0, _p1), ptr, 0);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx(__lsx_vldrepl_d(ptr, 0));
            _p = __lsx_vfmadd_s(_p, _a, _b);
            __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            ptr[0] = float32_to_bfloat16(bfloat16_to_float32(ptr[0]) * a + b);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
