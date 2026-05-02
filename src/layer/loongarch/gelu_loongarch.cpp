// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gelu_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "loongarch_usability.h"
#include "lsx_mathfun.h"
#if __loongarch_asx
#include <lasxintrin.h>
#include "lasx_mathfun.h"
#endif // __loongarch_asx
#endif // __loongarch_sx

namespace ncnn {

GELU_loongarch::GELU_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int GELU_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
    int size = w * h * d * elempack;

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;

        if (fast_gelu)
        {
#if __loongarch_sx
#if __loongarch_asx
            __m256 _half8 = (__m256)__lasx_xvreplfr2vr_s(0.5f);
            __m256 _one8 = (__m256)__lasx_xvreplfr2vr_s(1.f);
            __m256 _fast1c8 = (__m256)__lasx_xvreplfr2vr_s(0.79788452f);
            __m256 _fast2c8 = (__m256)__lasx_xvreplfr2vr_s(0.044715f);
            for (; i + 7 < size; i += 8)
            {
                __builtin_prefetch(ptr + 32);

                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                __m256 _cube = __lasx_xvfmul_s(_p, _p);
                _cube = __lasx_xvfmul_s(_cube, _p);
                __m256 _blob = __lasx_xvfmul_s(_fast2c8, _cube);
                _blob = __lasx_xvfadd_s(_blob, _p);
                _blob = __lasx_xvfmul_s(_fast1c8, _blob);
                _blob = tanh256_ps(_blob);
                _blob = __lasx_xvfadd_s(_one8, _blob);
                _blob = __lasx_xvfmul_s(_half8, __lasx_xvfmul_s(_blob, _p));
                __lasx_xvst(_blob, ptr, 0);

                ptr += 8;
            }
#endif
            __m128 _half4 = (__m128)__lsx_vreplfr2vr_s(0.5f);
            __m128 _one4 = (__m128)__lsx_vreplfr2vr_s(1.f);
            __m128 _fast1c4 = (__m128)__lsx_vreplfr2vr_s(0.79788452f);
            __m128 _fast2c4 = (__m128)__lsx_vreplfr2vr_s(0.044715f);
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);

                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                __m128 _cube = __lsx_vfmul_s(_p, _p);
                _cube = __lsx_vfmul_s(_cube, _p);
                __m128 _blob = __lsx_vfmul_s(_fast2c4, _cube);
                _blob = __lsx_vfadd_s(_blob, _p);
                _blob = __lsx_vfmul_s(_fast1c4, _blob);
                _blob = tanh_ps(_blob);
                _blob = __lsx_vfadd_s(_one4, _blob);
                _blob = __lsx_vfmul_s(_half4, __lsx_vfmul_s(_blob, _p));
                __lsx_vst(_blob, ptr, 0);

                ptr += 4;
            }
#endif
            for (; i < size; i++)
            {
                *ptr = 0.5f * *ptr * (1.0f + tanhf(0.79788452f * (*ptr + 0.044715f * *ptr * *ptr * *ptr)));
                ptr++;
            }
        }
        else
        {
            for (; i < size; i++)
            {
                *ptr = 0.5f * *ptr * erfcf(-0.70710678f * *ptr);
                ptr++;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int GELU_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;

        if (fast_gelu)
        {
#if __loongarch_sx
#if __loongarch_asx
            __m256 _half8 = (__m256)__lasx_xvreplfr2vr_s(0.5f);
            __m256 _one8 = (__m256)__lasx_xvreplfr2vr_s(1.f);
            __m256 _fast1c8 = (__m256)__lasx_xvreplfr2vr_s(0.79788452f);
            __m256 _fast2c8 = (__m256)__lasx_xvreplfr2vr_s(0.044715f);
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                __m256 _cube = __lasx_xvfmul_s(_p, _p);
                _cube = __lasx_xvfmul_s(_cube, _p);
                __m256 _blob = __lasx_xvfmul_s(_fast2c8, _cube);
                _blob = __lasx_xvfadd_s(_blob, _p);
                _blob = __lasx_xvfmul_s(_fast1c8, _blob);
                _blob = tanh256_ps(_blob);
                _blob = __lasx_xvfadd_s(_one8, _blob);
                _blob = __lasx_xvfmul_s(_half8, __lasx_xvfmul_s(_blob, _p));
                __lsx_vst(float2bfloat_lasx(_blob), ptr, 0);

                ptr += 8;
            }
#endif
            __m128 _half4 = (__m128)__lsx_vreplfr2vr_s(0.5f);
            __m128 _one4 = (__m128)__lsx_vreplfr2vr_s(1.f);
            __m128 _fast1c4 = (__m128)__lsx_vreplfr2vr_s(0.79788452f);
            __m128 _fast2c4 = (__m128)__lsx_vreplfr2vr_s(0.044715f);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_lsx((__m128i*)ptr);
                __m128 _cube = __lsx_vfmul_s(_p, _p);
                _cube = __lsx_vfmul_s(_cube, _p);
                __m128 _blob = __lsx_vfmul_s(_fast2c4, _cube);
                _blob = __lsx_vfadd_s(_blob, _p);
                _blob = __lsx_vfmul_s(_fast1c4, _blob);
                _blob = tanh_ps(_blob);
                _blob = __lsx_vfadd_s(_one4, _blob);
                _blob = __lsx_vfmul_s(_half4, __lsx_vfmul_s(_blob, _p));
                __lsx_vstelm_d(float2bfloat_lsx(_blob), ptr, 0, 0);

                ptr += 4;
            }
#endif
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                v = 0.5f * v * (1.0f + tanhf(0.79788452f * (v + 0.044715f * v * v * v)));
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
        else
        {
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                v = 0.5f * v * erfcf(-0.70710678f * v);
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
