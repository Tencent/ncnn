// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "clip_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {


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
#if __loongarch_asx
        __m256 _max256 = (__m256)__lasx_xvreplfr2vr_s(max);
        __m256 _min256 = (__m256)__lasx_xvreplfr2vr_s(min);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            _p = __lasx_xvfmax_s(_p, _min256);
            _p = __lasx_xvfmin_s(_p, _max256);
            __lasx_xvst(_p, ptr, 0);

            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _max = (__m128)__lsx_vreplfr2vr_s(max);
        __m128 _min = (__m128)__lsx_vreplfr2vr_s(min);
        for (; i + 3 < size; i += 4)
        {
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
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _min_lasx = (__m256)__lasx_xvreplfr2vr_s(min);
        __m256 _max_lasx = (__m256)__lasx_xvreplfr2vr_s(max);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
            _p = __lasx_xvfmax_s(_p, _min_lasx);
            _p = __lasx_xvfmin_s(_p, _max_lasx);
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _min = (__m128)__lsx_vreplfr2vr_s(min);
        __m128 _max = (__m128)__lsx_vreplfr2vr_s(max);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
            _p = __lsx_vfmax_s(_p, _min);
            _p = __lsx_vfmin_s(_p, _max);
            __lsx_vstelm_d(float2bfloat_lsx(_p, _p), ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < min) v = min;
            if (v > max) v = max;
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
