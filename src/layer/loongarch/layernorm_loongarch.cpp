// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

LayerNorm_loongarch::LayerNorm_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void layernorm_loongarch_bf16(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        __m256 _sum = (__m256)__lasx_xvreplfr2vr_s(0.f);
        const unsigned short* ptr0 = ptr;
        for (int i = 0; i < size; i += 8)
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
            mean_data[i] = sum_data[i] / elemcount;
        }
        __m256 _mean = (__m256)__lasx_xvld(mean_data, 0);

        __m256 _sqsum = (__m256)__lasx_xvreplfr2vr_s(0.f);
        ptr0 = ptr;
        for (int i = 0; i < size; i += 8)
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
        for (int i = 0; i < 8; i++)
        {
            float a = 1.f / sqrtf(sqsum_data[i] / elemcount + eps);
            a_data[i] = a;
            b_data[i] = -mean_data[i] * a;
        }

        __m256 _a = (__m256)__lasx_xvld(a_data, 0);
        __m256 _b = (__m256)__lasx_xvld(b_data, 0);

        if (gamma_ptr && beta_ptr)
        {
            for (int i = 0; i < size; i += 8)
            {
                __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                _p = __lasx_xvfmadd_s(_p, _a, _b);
                __m256 _gamma = (__m256)__lasx_xvreplfr2vr_s(gamma_ptr[0]);
                __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta_ptr[0]);
                _p = __lasx_xvfmadd_s(_p, _gamma, _beta);
                __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
                ptr += 8;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        else
        {
            for (int i = 0; i < size; i += 8)
            {
                __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                _p = __lasx_xvfmadd_s(_p, _a, _b);
                __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
                ptr += 8;
            }
        }

        return;
    }
#endif // __loongarch_asx

    if (elempack == 4)
    {
        __m128 _sum = (__m128)__lsx_vreplfr2vr_s(0.f);
        const unsigned short* ptr0 = ptr;
        for (int i = 0; i < size; i += 4)
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
            mean_data[i] = sum_data[i] / elemcount;
        }
        __m128 _mean = (__m128)__lsx_vld(mean_data, 0);

        __m128 _sqsum = (__m128)__lsx_vreplfr2vr_s(0.f);
        ptr0 = ptr;
        for (int i = 0; i < size; i += 4)
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
        for (int i = 0; i < 4; i++)
        {
            float a = 1.f / sqrtf(sqsum_data[i] / elemcount + eps);
            a_data[i] = a;
            b_data[i] = -mean_data[i] * a;
        }

        __m128 _a = (__m128)__lsx_vld(a_data, 0);
        __m128 _b = (__m128)__lsx_vld(b_data, 0);

        if (gamma_ptr && beta_ptr)
        {
            for (int i = 0; i < size; i += 4)
            {
                __m128 _p = bfloat2float_lsx((__m128i*)ptr);
                _p = __lsx_vfmadd_s(_p, _a, _b);
                __m128 _gamma = __lsx_vreplfr2vr_s(gamma_ptr[0]);
                __m128 _beta = __lsx_vreplfr2vr_s(beta_ptr[0]);
                _p = __lsx_vfmadd_s(_p, _gamma, _beta);
                __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
                ptr += 4;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        else
        {
            for (int i = 0; i < size; i += 4)
            {
                __m128 _p = bfloat2float_lsx((__m128i*)ptr);
                _p = __lsx_vfmadd_s(_p, _a, _b);
                __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
                ptr += 4;
            }
        }

        return;
    }
#endif // __loongarch_sx

    // elempack == 1 or scalar fallback
    float mean = 0.f;
    {
        const unsigned short* ptr0 = ptr;
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _sum8 = (__m256)__lasx_xvreplfr2vr_s(0.f);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i*)ptr0);
            _sum8 = __lasx_xvfadd_s(_sum8, _p);
            ptr0 += 8;
        }
        mean += __lasx_reduce_fadd_s(_sum8);
#endif // __loongarch_asx
        __m128 _sum4 = (__m128)__lsx_vreplfr2vr_s(0.f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx((__m128i*)ptr0);
            _sum4 = __lsx_vfadd_s(_sum4, _p);
            ptr0 += 4;
        }
        mean += __lsx_reduce_fadd_s(_sum4);
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            mean += bfloat16_to_float32(ptr0[0]);
            ptr0++;
        }
        mean /= size;
    }

    float var = 0.f;
    {
        const unsigned short* ptr0 = ptr;
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _mean8 = __lasx_xvreplfr2vr_s(mean);
        __m256 _sqsum8 = (__m256)__lasx_xvreplfr2vr_s(0.f);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i*)ptr0);
            _p = __lasx_xvfsub_s(_p, _mean8);
            _sqsum8 = __lasx_xvfmadd_s(_p, _p, _sqsum8);
            ptr0 += 8;
        }
        var += __lasx_reduce_fadd_s(_sqsum8);
#endif // __loongarch_asx
        __m128 _mean4 = __lsx_vreplfr2vr_s(mean);
        __m128 _sqsum4 = (__m128)__lsx_vreplfr2vr_s(0.f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx((__m128i*)ptr0);
            _p = __lsx_vfsub_s(_p, _mean4);
            _sqsum4 = __lsx_vfmadd_s(_p, _p, _sqsum4);
            ptr0 += 4;
        }
        var += __lsx_reduce_fadd_s(_sqsum4);
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr0[0]) - mean;
            var += v * v;
            ptr0++;
        }
        var = 1.f / sqrtf(var / size + eps);
    }

    const float bias = -mean * var;

    if (gamma_ptr && beta_ptr)
    {
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _a8 = __lasx_xvreplfr2vr_s(var);
        __m256 _b8 = __lasx_xvreplfr2vr_s(bias);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i*)ptr);
            __m256 _gamma = (__m256)__lasx_xvld(gamma_ptr, 0);
            __m256 _beta = (__m256)__lasx_xvld(beta_ptr, 0);
            _p = __lasx_xvfmadd_s(_p, _a8, _b8);
            _p = __lasx_xvfmadd_s(_p, _gamma, _beta);
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            ptr += 8;
            gamma_ptr += 8;
            beta_ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _a4 = __lsx_vreplfr2vr_s(var);
        __m128 _b4 = __lsx_vreplfr2vr_s(bias);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx((__m128i*)ptr);
            __m128 _gamma = (__m128)__lsx_vld(gamma_ptr, 0);
            __m128 _beta = (__m128)__lsx_vld(beta_ptr, 0);
            _p = __lsx_vfmadd_s(_p, _a4, _b4);
            _p = __lsx_vfmadd_s(_p, _gamma, _beta);
            __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
            ptr += 4;
            gamma_ptr += 4;
            beta_ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            ptr[0] = float32_to_bfloat16((bfloat16_to_float32(ptr[0]) * var + bias) * gamma_ptr[0] + beta_ptr[0]);
            ptr++;
            gamma_ptr++;
            beta_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _a8 = __lasx_xvreplfr2vr_s(var);
        __m256 _b8 = __lasx_xvreplfr2vr_s(bias);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i*)ptr);
            _p = __lasx_xvfmadd_s(_p, _a8, _b8);
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _a4 = __lsx_vreplfr2vr_s(var);
        __m128 _b4 = __lsx_vreplfr2vr_s(bias);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx((__m128i*)ptr);
            _p = __lsx_vfmadd_s(_p, _a4, _b4);
            __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            ptr[0] = float32_to_bfloat16(bfloat16_to_float32(ptr[0]) * var + bias);
            ptr++;
        }
    }
}

static void layernorm_loongarch(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        __m256 _sum = (__m256)__lasx_xvreplfr2vr_s(0.f);
        const float* ptr0 = ptr;
        for (int i = 0; i < size; i += 8)
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
            mean_data[i] = sum_data[i] / elemcount;
        }
        __m256 _mean = (__m256)__lasx_xvld(mean_data, 0);

        __m256 _sqsum = (__m256)__lasx_xvreplfr2vr_s(0.f);
        ptr0 = ptr;
        for (int i = 0; i < size; i += 8)
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
        for (int i = 0; i < 8; i++)
        {
            float a = 1.f / sqrtf(sqsum_data[i] / elemcount + eps);
            a_data[i] = a;
            b_data[i] = -mean_data[i] * a;
        }

        __m256 _a = (__m256)__lasx_xvld(a_data, 0);
        __m256 _b = (__m256)__lasx_xvld(b_data, 0);

        if (gamma_ptr && beta_ptr)
        {
            for (int i = 0; i < size; i += 8)
            {
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                _p = __lasx_xvfmadd_s(_p, _a, _b);
                __m256 _gamma = (__m256)__lasx_xvreplfr2vr_s(gamma_ptr[0]);
                __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(beta_ptr[0]);
                _p = __lasx_xvfmadd_s(_p, _gamma, _beta);
                __lasx_xvst(_p, ptr, 0);
                ptr += 8;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        else
        {
            for (int i = 0; i < size; i += 8)
            {
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                _p = __lasx_xvfmadd_s(_p, _a, _b);
                __lasx_xvst(_p, ptr, 0);
                ptr += 8;
            }
        }

        return;
    }
#endif // __loongarch_asx

    if (elempack == 4)
    {
        __m128 _sum = (__m128)__lsx_vreplfr2vr_s(0.f);
        const float* ptr0 = ptr;
        for (int i = 0; i < size; i += 4)
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
            mean_data[i] = sum_data[i] / elemcount;
        }
        __m128 _mean = (__m128)__lsx_vld(mean_data, 0);

        __m128 _sqsum = (__m128)__lsx_vreplfr2vr_s(0.f);
        ptr0 = ptr;
        for (int i = 0; i < size; i += 4)
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
        for (int i = 0; i < 4; i++)
        {
            float a = 1.f / sqrtf(sqsum_data[i] / elemcount + eps);
            a_data[i] = a;
            b_data[i] = -mean_data[i] * a;
        }

        __m128 _a = (__m128)__lsx_vld(a_data, 0);
        __m128 _b = (__m128)__lsx_vld(b_data, 0);

        if (gamma_ptr && beta_ptr)
        {
            for (int i = 0; i < size; i += 4)
            {
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                _p = __lsx_vfmadd_s(_p, _a, _b);
                __m128 _gamma = __lsx_vreplfr2vr_s(gamma_ptr[0]);
                __m128 _beta = __lsx_vreplfr2vr_s(beta_ptr[0]);
                _p = __lsx_vfmadd_s(_p, _gamma, _beta);
                __lsx_vst(_p, ptr, 0);
                ptr += 4;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        else
        {
            for (int i = 0; i < size; i += 4)
            {
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                _p = __lsx_vfmadd_s(_p, _a, _b);
                __lsx_vst(_p, ptr, 0);
                ptr += 4;
            }
        }

        return;
    }
#endif // __loongarch_sx

    float mean = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _sum8 = (__m256)__lasx_xvreplfr2vr_s(0.f);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
            _sum8 = __lasx_xvfadd_s(_sum8, _p);
            ptr0 += 8;
        }
        mean += __lasx_reduce_fadd_s(_sum8);
#endif // __loongarch_asx
        __m128 _sum4 = (__m128)__lsx_vreplfr2vr_s(0.f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
            _sum4 = __lsx_vfadd_s(_sum4, _p);
            ptr0 += 4;
        }
        mean += __lsx_reduce_fadd_s(_sum4);
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            mean += ptr0[0];
            ptr0++;
        }
        mean /= size;
    }

    float var = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _mean8 = __lasx_xvreplfr2vr_s(mean);
        __m256 _sqsum8 = (__m256)__lasx_xvreplfr2vr_s(0.f);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
            _p = __lasx_xvfsub_s(_p, _mean8);
            _sqsum8 = __lasx_xvfmadd_s(_p, _p, _sqsum8);
            ptr0 += 8;
        }
        var += __lasx_reduce_fadd_s(_sqsum8);
#endif // __loongarch_asx
        __m128 _mean4 = __lsx_vreplfr2vr_s(mean);
        __m128 _sqsum4 = (__m128)__lsx_vreplfr2vr_s(0.f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
            _p = __lsx_vfsub_s(_p, _mean4);
            _sqsum4 = __lsx_vfmadd_s(_p, _p, _sqsum4);
            ptr0 += 4;
        }
        var += __lsx_reduce_fadd_s(_sqsum4);
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = ptr0[0] - mean;
            var += v * v;
            ptr0++;
        }
        var = 1.f / sqrtf(var / size + eps);
    }

    const float bias = -mean * var;

    if (gamma_ptr && beta_ptr)
    {
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _a8 = __lasx_xvreplfr2vr_s(var);
        __m256 _b8 = __lasx_xvreplfr2vr_s(bias);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __m256 _gamma = (__m256)__lasx_xvld(gamma_ptr, 0);
            __m256 _beta = (__m256)__lasx_xvld(beta_ptr, 0);
            _p = __lasx_xvfmadd_s(_p, _a8, _b8);
            _p = __lasx_xvfmadd_s(_p, _gamma, _beta);
            __lasx_xvst(_p, ptr, 0);
            ptr += 8;
            gamma_ptr += 8;
            beta_ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _a4 = __lsx_vreplfr2vr_s(var);
        __m128 _b4 = __lsx_vreplfr2vr_s(bias);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _gamma = (__m128)__lsx_vld(gamma_ptr, 0);
            __m128 _beta = (__m128)__lsx_vld(beta_ptr, 0);
            _p = __lsx_vfmadd_s(_p, _a4, _b4);
            _p = __lsx_vfmadd_s(_p, _gamma, _beta);
            __lsx_vst(_p, ptr, 0);
            ptr += 4;
            gamma_ptr += 4;
            beta_ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            ptr[0] = (ptr[0] * var + bias) * gamma_ptr[0] + beta_ptr[0];
            ptr++;
            gamma_ptr++;
            beta_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _a8 = __lasx_xvreplfr2vr_s(var);
        __m256 _b8 = __lasx_xvreplfr2vr_s(bias);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            _p = __lasx_xvfmadd_s(_p, _a8, _b8);
            __lasx_xvst(_p, ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _a4 = __lsx_vreplfr2vr_s(var);
        __m128 _b4 = __lsx_vreplfr2vr_s(bias);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _p = __lsx_vfmadd_s(_p, _a4, _b4);
            __lsx_vst(_p, ptr, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * var + bias;
            ptr++;
        }
    }
}

int LayerNorm_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        layernorm_loongarch(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            layernorm_loongarch(ptr, gamma_data, beta_data, eps, w, elempack);
        }
    }

    if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    layernorm_loongarch(ptr, gamma_data, beta_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                layernorm_loongarch(ptr, gamma_data, beta_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int LayerNorm_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;

    if (dims == 1)
    {
        unsigned short* ptr = (unsigned short*)bottom_top_blob;
        layernorm_loongarch_bf16(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = (unsigned short*)bottom_top_blob.row(i);
            layernorm_loongarch_bf16(ptr, gamma_data, beta_data, eps, w, elempack);
        }
    }

    if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q).row(i);
                    layernorm_loongarch_bf16(ptr, gamma_data, beta_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q);
                layernorm_loongarch_bf16(ptr, gamma_data, beta_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
