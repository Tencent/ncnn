// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "selu_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#if __loongarch_asx
#include <lasxintrin.h>
#include "lasx_mathfun.h"
#endif // __loongarch_asx
#endif // __loongarch_sx

namespace ncnn {

SELU_loongarch::SELU_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int SELU_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
    int size = w * h * d * elempack;
    float alphaxlambda = alpha * lambda;

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

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
        __m256 _lambda8 = (__m256)__lasx_xvreplfr2vr_s(lambda);
        for (; i + 7 < size; i += 8)
        {
            __builtin_prefetch(ptr + 32);

            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __m256 _pos = __lasx_xvfmax_s(_p, _zero8);
            __m256 _neg = __lasx_xvfmin_s(_p, _zero8);
            __m256 _outp = exp256_ps(_neg);
            _outp = __lasx_xvfsub_s(_outp, _one8);
            _outp = __lasx_xvfmul_s(_outp, _alpha8);
            _outp = __lasx_xvfadd_s(_outp, _pos);
            _outp = __lasx_xvfmul_s(_outp, _lambda8);
            __lasx_xvst(_outp, ptr, 0);

            ptr += 8;
        }
#endif
        __m128 _zero4 = (__m128)__lsx_vreplgr2vr_w(0);
        __m128 _one4 = (__m128)__lsx_vreplfr2vr_s(1.f);
        __m128 _alpha4 = (__m128)__lsx_vreplfr2vr_s(alpha);
        __m128 _lambda4 = (__m128)__lsx_vreplfr2vr_s(lambda);
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);

            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _pos = __lsx_vfmax_s(_p, _zero4);
            __m128 _neg = __lsx_vfmin_s(_p, _zero4);
            __m128 _outp = exp_ps(_neg);
            _outp = __lsx_vfsub_s(_outp, _one4);
            _outp = __lsx_vfmul_s(_outp, _alpha4);
            _outp = __lsx_vfadd_s(_outp, _pos);
            _outp = __lsx_vfmul_s(_outp, _lambda4);
            __lsx_vst(_outp, ptr, 0);

            ptr += 4;
        }
#endif
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

#if NCNN_BF16
int SELU_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    Option opt_cast = opt;
    opt_cast.blob_allocator = opt.workspace_allocator;

    Mat bottom_top_blob_fp32;
    cast_bfloat16_to_float32(bottom_top_blob, bottom_top_blob_fp32, opt_cast);
    if (bottom_top_blob_fp32.empty())
        return -100;

    int ret = forward_inplace(bottom_top_blob_fp32, opt);
    if (ret != 0)
        return ret;

    cast_float32_to_bfloat16(bottom_top_blob_fp32, bottom_top_blob, opt);
    if (bottom_top_blob.empty())
        return -100;

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
