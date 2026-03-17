// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "selu_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#endif // __loongarch_sx

namespace ncnn {

SELU_loongarch::SELU_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
}

int SELU_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;
    float alphaxlambda = alpha * lambda;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __loongarch_sx
        __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
        __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
        __m128 _alphaxlambda = (__m128)__lsx_vreplfr2vr_s(alphaxlambda);
        __m128 _lambda = (__m128)__lsx_vreplfr2vr_s(lambda);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128i _lemask = __lsx_vfcmp_cle_s(_p, _zero);

            __m128 _nps = exp_ps(_p);
            _nps = __lsx_vfsub_s(_nps, _one);
            _nps = __lsx_vfmul_s(_nps, _alphaxlambda);

            _p = __lsx_vfmul_s(_p, _lambda);

            _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_nps, (__m128i)_lemask);
            __lsx_vst(_p, ptr, 0);

            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            if (*ptr < 0.f)
                *ptr = (expf(*ptr) - 1.f) * alphaxlambda;
            else
                *ptr *= lambda;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
