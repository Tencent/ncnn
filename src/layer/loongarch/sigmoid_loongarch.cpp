// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "sigmoid_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#if __loongarch_asx
#include <lasxintrin.h>
#include "lasx_mathfun.h"
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

Sigmoid_loongarch::Sigmoid_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif
}

int Sigmoid_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
#if __loongarch_asx
        __m256 _one_lasx = (__m256)__lasx_xvreplfr2vr_s(1.f);
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(ptr + 32);
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            _p = (__m256)__lasx_xvbitrevi_w((__m256i)_p, 31);
            _p = exp256_ps(_p);
            _p = __lasx_xvfadd_s(_p, _one_lasx);
            __m256 _outp = __lasx_xvfdiv_s(_one_lasx, _p);
            __lasx_xvst(_outp, ptr, 0);

            ptr += 8;
        }
#endif // __loongarch_lasx
        __m128 _one_lsx = (__m128)__lsx_vreplfr2vr_s(1.f);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _p = (__m128)__lsx_vbitrevi_w((__m128i)_p, 31);
            _p = exp_ps(_p);
            _p = __lsx_vfadd_s(_p, _one_lsx);
            __m128 _outp = __lsx_vfdiv_s(_one_lsx, _p);
            __lsx_vst(_outp, ptr, 0);

            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr = 1.f / (1.f + expf(-*ptr));

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
