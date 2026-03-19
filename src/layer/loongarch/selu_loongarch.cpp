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
        __m128 _alpha = (__m128)__lsx_vreplfr2vr_s(alpha);
        __m128 _lambda = (__m128)__lsx_vreplfr2vr_s(lambda);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __m128 _p = (__m128)__lsx_vld(ptr, 0);

            __m128 _pos = __lsx_vfmax_s(_p, _zero);
            __m128 _neg = __lsx_vfmin_s(_p, _zero);

            __m128 _blob = exp_ps(_neg);
            _blob = __lsx_vfsub_s(_blob, _one);
            _blob = __lsx_vfmul_s(_alpha, _blob);
            _blob = __lsx_vfmul_s(_lambda, __lsx_vfadd_s(_pos, _blob));

            __lsx_vst(_blob, ptr, 0);

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
