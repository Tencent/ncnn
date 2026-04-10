// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "hardswish_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

HardSwish_loongarch::HardSwish_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int HardSwish_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        __m256 _zero8 = (__m256)__lasx_xvreplgr2vr_w(0);
        __m256 _one8 = (__m256)__lasx_xvreplfr2vr_s(1.f);
        __m256 _alpha8 = (__m256)__lasx_xvreplfr2vr_s(alpha);
        __m256 _beta8 = (__m256)__lasx_xvreplfr2vr_s(beta);
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(ptr + 32);
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __m256 _outp = __lasx_xvfmadd_s(_alpha8, _p, _beta8);
            _outp = __lasx_xvfmax_s(_outp, _zero8);
            _outp = __lasx_xvfmin_s(_outp, _one8);
            _outp = __lasx_xvfmul_s(_outp, _p);
            __lasx_xvst(_outp, ptr, 0);

            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _zero4 = (__m128)__lsx_vreplgr2vr_w(0);
        __m128 _one4 = (__m128)__lsx_vreplfr2vr_s(1.f);
        __m128 _alpha4 = (__m128)__lsx_vreplfr2vr_s(alpha);
        __m128 _beta4 = (__m128)__lsx_vreplfr2vr_s(beta);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _outp = __lsx_vfmadd_s(_alpha4, _p, _beta4);
            _outp = __lsx_vfmax_s(_outp, _zero4);
            _outp = __lsx_vfmin_s(_outp, _one4);
            _outp = __lsx_vfmul_s(_outp, _p);
            __lsx_vst(_outp, ptr, 0);

            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            if (*ptr < lower)
                *ptr = 0.f;
            else if (*ptr > upper)
                ;
            else
                *ptr = *ptr * (*ptr * alpha + beta);
            ++ptr;
        }
    }

    return 0;
}

#if NCNN_BF16
int HardSwish_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
        unsigned short* ptr = bottom_top_blob.channel<unsigned short>(q);

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx((__m128i*)ptr);
            __m256 _outp = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(alpha), _p, (__m256)__lasx_xvreplfr2vr_s(beta));
            _outp = __lasx_xvfmax_s(_outp, (__m256)__lasx_xvreplgr2vr_w(0));
            _outp = __lasx_xvfmin_s(_outp, (__m256)__lasx_xvreplfr2vr_s(1.f));
            _outp = __lasx_xvfmul_s(_outp, _p);
            __lasx_xvst(float2bfloat_avx(_outp), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse((__m128i*)ptr);
            __m128 _outp = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(alpha), _p, (__m128)__lsx_vreplfr2vr_s(beta));
            _outp = __lsx_vfmax_s(_outp, (__m128)__lsx_vreplgr2vr_w(0));
            _outp = __lsx_vfmin_s(_outp, (__m128)__lsx_vreplfr2vr_s(1.f));
            _outp = __lsx_vfmul_s(_outp, _p);
            __lsx_vst(float2bfloat_sse(_outp), ptr, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < lower)
                v = 0.f;
            else if (v > upper)
                ;
            else
                v = v * (v * alpha + beta);
            *ptr = float32_to_bfloat16(v);
            ++ptr;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
