// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gelu_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#endif // __loongarch_sx

namespace ncnn {

GELU_loongarch::GELU_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
}

int GELU_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __loongarch_sx
        if (fast_gelu)
        {
            __m128 _half = (__m128)__lsx_vreplfr2vr_s(0.5f);
            __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
            __m128 _fast1c = (__m128)__lsx_vreplfr2vr_s(0.79788452f);
            __m128 _fast2c = (__m128)__lsx_vreplfr2vr_s(0.044715f);
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                __m128 _p = (__m128)__lsx_vld(ptr, 0);

                __m128 _cube = __lsx_vfmul_s(_p, _p);
                _cube = __lsx_vfmul_s(_p, _cube);
                __m128 _blob = __lsx_vfmul_s(_fast2c, _cube);
                _blob = __lsx_vfadd_s(_p, _blob);
                _blob = __lsx_vfmul_s(_fast1c, _blob);
                _blob = tanh_ps(_blob);
                _blob = __lsx_vfadd_s(_one, _blob);
                _blob = __lsx_vfmul_s(_half, __lsx_vfmul_s(_blob, _p));
                __lsx_vst(_blob, ptr, 0);

                ptr += 4;
            }
        }
        else
        {
            __m128 _half = (__m128)__lsx_vreplfr2vr_s(0.5f);
            __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
            __m128 _inv_sqrt2 = (__m128)__lsx_vreplfr2vr_s(0.70710678f);
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                __m128 _p = (__m128)__lsx_vld(ptr, 0);

                __m128 _blob = __lsx_vfmul_s(_inv_sqrt2, _p);
                _blob = erf_ps(_blob);
                _blob = __lsx_vfadd_s(_one, _blob);
                _blob = __lsx_vfmul_s(_half, __lsx_vfmul_s(_blob, _p));
                __lsx_vst(_blob, ptr, 0);

                ptr += 4;
            }
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            if (fast_gelu)
            {
                *ptr = 0.5f * *ptr * (1.0f + tanhf(0.79788452f * (*ptr + 0.044715f * *ptr * *ptr * *ptr)));
            }
            else
            {
                *ptr = 0.5f * *ptr * (1.0f + erff(0.70710678f * *ptr));
            }

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
