// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "swish_loongarch.h"

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

Swish_loongarch::Swish_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Swish_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        __m256 _one8 = (__m256)__lasx_xvreplfr2vr_s(1.f);
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(ptr + 32);
            __m256i _p = (__m256i)__lasx_xvld(ptr, 0);
            _p = (__m256i)__lasx_xvfdiv_s((__m256)_p, __lasx_xvfadd_s(_one8, exp256_ps((__m256)__lasx_xvbitrevi_w((__m256i)_p, 31))));
            __lasx_xvst(_p, ptr, 0);

            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _one4 = (__m128)__lsx_vreplfr2vr_s(1.f);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __m128i _p = __lsx_vld(ptr, 0);
            _p = (__m128i)__lsx_vfdiv_s((__m128)_p, __lsx_vfadd_s(_one4, exp_ps((__m128)__lsx_vbitrevi_w(_p, 31))));
            __lsx_vst(_p, ptr, 0);

            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr = *ptr / (1.f + expf(-*ptr));
            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int Swish_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
        __m256 _one256 = (__m256)__lasx_xvreplfr2vr_s(1.f);
        for (; i + 7 < size; i += 8)
        {
            // load 8 bf16 values and convert to fp32
            __m256 _p = bfloat2float_lasx((__m128i*)ptr);
            // swish
            _p = __lasx_xvfdiv_s(_p, __lasx_xvfadd_s(_one256, exp256_ps((__m256)__lasx_xvbitrevi_w((__m256i)_p, 31))));
            // fp32 -> bf16
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        __m128i _zero = __lsx_vreplgr2vr_w(0);
        __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
        for (; i + 3 < size; i += 4)
        {
            // load 4 bf16 values safely via 64-bit load
            int64_t v;
            memcpy(&v, ptr, 8);
            __m128i _raw = __lsx_vreplgr2vr_d(v);
            __m128i _pi = __lsx_vilvl_h(_raw, _zero);
            __m128 _p = (__m128)_pi;
            // swish
            _p = __lsx_vfdiv_s(_p, __lsx_vfadd_s(_one, exp_ps((__m128)__lsx_vbitrevi_w((__m128i)_p, 31))));
            // fp32 -> bf16
            _pi = (__m128i)_p;
            _pi = __lsx_vsrli_w(_pi, 16);
            __m128i _out = __lsx_vpickev_h(__lsx_vreplgr2vr_w(0), _pi);
            __lsx_vstelm_d(_out, ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            v = v / (1.f + expf(-v));
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
