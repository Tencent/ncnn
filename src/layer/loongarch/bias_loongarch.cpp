// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "bias_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

int Bias_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    const float* bias_ptr = bias_data;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float bias = bias_ptr[q];

#if __loongarch_sx
#if __loongarch_asx
        int nn8 = size >> 3;
        int remain8 = size - (nn8 << 3);
#else
        int nn8 = 0;
        int remain8 = size;
#endif // __loongarch_asx
        int nn4 = remain8 >> 2;
        int remain = remain8 - (nn4 << 2);
#else
        int remain = size;
#endif // __loongarch_sx

#if __loongarch_sx
#if __loongarch_asx
        __m256 _bias256 = (__m256)__lasx_xvreplfr2vr_s(bias);
        for (; nn8 > 0; nn8--)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __m256 _outp = __lasx_xvfadd_s(_p, _bias256);
            __lasx_xvst(_outp, ptr, 0);

            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _bias = (__m128)__lsx_vreplfr2vr_s(bias);
        for (; nn4 > 0; nn4--)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _outp = __lsx_vfadd_s(_p, _bias);
            __lsx_vst(_outp, ptr, 0);

            ptr += 4;
        }
#endif // __loongarch_sx

        for (; remain > 0; remain--)
        {
            *ptr = *ptr + bias;
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
