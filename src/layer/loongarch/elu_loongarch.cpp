// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "elu_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#include "loongarch_activation.h"
#endif // __loongarch_sx

namespace ncnn {

ELU_loongarch::ELU_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
}

int ELU_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        __m128 _alpha = (__m128)__lsx_vreplfr2vr_s(alpha);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _p = elu_ps(_p, _alpha);
            __lsx_vst(_p, ptr, 0);

            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            if (*ptr < 0.f)
                *ptr = alpha * (expf(*ptr) - 1.f);

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
