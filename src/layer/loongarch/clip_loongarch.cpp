// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "clip_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "clip_bf16s.h"
#endif

Clip_loongarch::Clip_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#if NCNN_BF16
    support_bf16_storage = true;
#endif
#endif
}

int Clip_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

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
        __m128 _max = (__m128)__lsx_vreplfr2vr_s(max);
        __m128 _min = (__m128)__lsx_vreplfr2vr_s(min);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _p = __lsx_vfmax_s(_p, _min);
            _p = __lsx_vfmin_s(_p, _max);
            __lsx_vst(_p, ptr, 0);

            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            if (*ptr < min)
                *ptr = min;

            if (*ptr > max)
                *ptr = max;

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int Clip_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    clip_bf16s(bottom_top_blob, min, max, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
