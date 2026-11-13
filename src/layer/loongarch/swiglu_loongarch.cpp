// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "swiglu_loongarch.h"

#include <float.h>

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

SwiGLU_loongarch::SwiGLU_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

#if __loongarch_sx
static inline __m128 swiglu_f32x4(__m128 _g, __m128 _u, __m128 _scale, __m128 _offset, __m128 _limit, __m128 _mlimit, __m128 _one)
{
    _g = __lsx_vfmin_s(_g, _limit);
    _u = __lsx_vfmin_s(__lsx_vfmax_s(_u, _mlimit), _limit);

    // silu(scale * g) = g / (1 + exp(-scale * g))
    __m128 _z = __lsx_vfmul_s(_g, _scale);
    _z = (__m128)__lsx_vbitrevi_w((__m128i)_z, 31);
    _g = __lsx_vfdiv_s(_g, __lsx_vfadd_s(_one, exp_ps(_z)));

    return __lsx_vfmul_s(_g, __lsx_vfadd_s(_u, _offset));
}

#if __loongarch_asx
static inline __m256 swiglu_f32x8(__m256 _g, __m256 _u, __m256 _scale, __m256 _offset, __m256 _limit, __m256 _mlimit, __m256 _one)
{
    _g = __lasx_xvfmin_s(_g, _limit);
    _u = __lasx_xvfmin_s(__lasx_xvfmax_s(_u, _mlimit), _limit);

    // silu(scale * g) = g / (1 + exp(-scale * g))
    __m256 _z = __lasx_xvfmul_s(_g, _scale);
    _z = (__m256)__lasx_xvbitrevi_w((__m256i)_z, 31);
    _g = __lasx_xvfdiv_s(_g, __lasx_xvfadd_s(_one, exp256_ps(_z)));

    return __lasx_xvfmul_s(_g, __lasx_xvfadd_s(_u, _offset));
}
#endif // __loongarch_asx
#endif // __loongarch_sx

int SwiGLU_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() != 2)
        return -100;

    const Mat& gate = bottom_blobs[0];
    const Mat& up = bottom_blobs[1];

    if (gate.dims != up.dims || gate.w != up.w || gate.h != up.h || gate.d != up.d || gate.c != up.c || gate.elempack != up.elempack || gate.elembits() != up.elembits())
        return -100;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(gate, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if NCNN_BF16
    if (opt.use_bf16_storage && gate.elembits() == 16)
        return forward_bf16s(gate, up, top_blob, opt);
#endif

    return forward_fp32(gate, up, top_blob, opt);
}

int SwiGLU_loongarch::forward_fp32(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const
{
    const int channels = gate.c;
    const int size = gate.w * gate.h * gate.d * gate.elempack;
    const float climit = activation_limit > 0.f ? activation_limit : FLT_MAX;
    const float mlimit = -climit;

    const int block_size = 1024;
    const int nn_block = (size + block_size - 1) / block_size;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < channels * nn_block; ii++)
    {
        const int q = ii / nn_block;
        const int offset = ii % nn_block * block_size;
        const int len = std::min(size - offset, block_size);

        const float* gateptr = (const float*)gate.channel(q) + offset;
        const float* upptr = (const float*)up.channel(q) + offset;
        float* outptr = (float*)top_blob.channel(q) + offset;

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _scale8 = (__m256)__lasx_xvreplfr2vr_s(sigmoid_scale);
        __m256 _offset8 = (__m256)__lasx_xvreplfr2vr_s(up_offset);
        __m256 _limit8 = (__m256)__lasx_xvreplfr2vr_s(climit);
        __m256 _mlimit8 = (__m256)__lasx_xvreplfr2vr_s(mlimit);
        __m256 _one8 = (__m256)__lasx_xvreplfr2vr_s(1.f);
        for (; i + 7 < len; i += 8)
        {
            __m256 _g = (__m256)__lasx_xvld(gateptr, 0);
            __m256 _u = (__m256)__lasx_xvld(upptr, 0);
            __lasx_xvst(swiglu_f32x8(_g, _u, _scale8, _offset8, _limit8, _mlimit8, _one8), outptr, 0);

            gateptr += 8;
            upptr += 8;
            outptr += 8;
        }
#endif // __loongarch_asx
        __m128 _scale4 = (__m128)__lsx_vreplfr2vr_s(sigmoid_scale);
        __m128 _offset4 = (__m128)__lsx_vreplfr2vr_s(up_offset);
        __m128 _limit4 = (__m128)__lsx_vreplfr2vr_s(climit);
        __m128 _mlimit4 = (__m128)__lsx_vreplfr2vr_s(mlimit);
        __m128 _one4 = (__m128)__lsx_vreplfr2vr_s(1.f);
        for (; i + 3 < len; i += 4)
        {
            __m128 _g = (__m128)__lsx_vld(gateptr, 0);
            __m128 _u = (__m128)__lsx_vld(upptr, 0);
            __lsx_vst(swiglu_f32x4(_g, _u, _scale4, _offset4, _limit4, _mlimit4, _one4), outptr, 0);

            gateptr += 4;
            upptr += 4;
            outptr += 4;
        }
#endif // __loongarch_sx
        for (; i < len; i++)
        {
            float g = gateptr[0];
            float u = upptr[0];
            g = std::min(g, climit);
            u = std::min(std::max(u, mlimit), climit);
            outptr[0] = g / (1.f + expf(-sigmoid_scale * g)) * (u + up_offset);
            gateptr++;
            upptr++;
            outptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int SwiGLU_loongarch::forward_bf16s(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const
{
    const int channels = gate.c;
    const int size = gate.w * gate.h * gate.d * gate.elempack;
    const float climit = activation_limit > 0.f ? activation_limit : FLT_MAX;
    const float mlimit = -climit;

    const int block_size = 1024;
    const int nn_block = (size + block_size - 1) / block_size;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < channels * nn_block; ii++)
    {
        const int q = ii / nn_block;
        const int offset = ii % nn_block * block_size;
        const int len = std::min(size - offset, block_size);

        const unsigned short* gateptr = (const unsigned short*)gate.channel(q) + offset;
        const unsigned short* upptr = (const unsigned short*)up.channel(q) + offset;
        unsigned short* outptr = (unsigned short*)top_blob.channel(q) + offset;

        int i = 0;
#if __loongarch_sx
        __m128 _scale4 = (__m128)__lsx_vreplfr2vr_s(sigmoid_scale);
        __m128 _offset4 = (__m128)__lsx_vreplfr2vr_s(up_offset);
        __m128 _limit4 = (__m128)__lsx_vreplfr2vr_s(climit);
        __m128 _mlimit4 = (__m128)__lsx_vreplfr2vr_s(mlimit);
        __m128 _one4 = (__m128)__lsx_vreplfr2vr_s(1.f);
#if __loongarch_asx
        __m256 _scale8 = (__m256)__lasx_xvreplfr2vr_s(sigmoid_scale);
        __m256 _offset8 = (__m256)__lasx_xvreplfr2vr_s(up_offset);
        __m256 _limit8 = (__m256)__lasx_xvreplfr2vr_s(climit);
        __m256 _mlimit8 = (__m256)__lasx_xvreplfr2vr_s(mlimit);
        __m256 _one8 = (__m256)__lasx_xvreplfr2vr_s(1.f);
        for (; i + 7 < len; i += 8)
        {
            __m256 _g = bfloat2float_lasx((const __m128i*)gateptr);
            __m256 _u = bfloat2float_lasx((const __m128i*)upptr);
            __lsx_vst(float2bfloat_lasx(swiglu_f32x8(_g, _u, _scale8, _offset8, _limit8, _mlimit8, _one8)), outptr, 0);

            gateptr += 8;
            upptr += 8;
            outptr += 8;
        }
#else  // __loongarch_asx
        __m128i _zero = __lsx_vreplgr2vr_w(0);
        for (; i + 7 < len; i += 8)
        {
            __m128i _g16 = __lsx_vld(gateptr, 0);
            __m128i _u16 = __lsx_vld(upptr, 0);
            __m128 _g0 = (__m128)__lsx_vilvl_h(_g16, _zero);
            __m128 _g1 = (__m128)__lsx_vilvh_h(_g16, _zero);
            __m128 _u0 = (__m128)__lsx_vilvl_h(_u16, _zero);
            __m128 _u1 = (__m128)__lsx_vilvh_h(_u16, _zero);
            __m128 _out0 = swiglu_f32x4(_g0, _u0, _scale4, _offset4, _limit4, _mlimit4, _one4);
            __m128 _out1 = swiglu_f32x4(_g1, _u1, _scale4, _offset4, _limit4, _mlimit4, _one4);
            __lsx_vst(float2bfloat_lsx(_out0, _out1), outptr, 0);

            gateptr += 8;
            upptr += 8;
            outptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < len; i += 4)
        {
            __m128 _g = bfloat2float_lsx(__lsx_vldrepl_d(gateptr, 0));
            __m128 _u = bfloat2float_lsx(__lsx_vldrepl_d(upptr, 0));
            __m128 _out = swiglu_f32x4(_g, _u, _scale4, _offset4, _limit4, _mlimit4, _one4);
            __lsx_vstelm_d(float2bfloat_lsx(_out), outptr, 0, 0);

            gateptr += 4;
            upptr += 4;
            outptr += 4;
        }
#endif // __loongarch_sx
        for (; i < len; i++)
        {
            float g = bfloat16_to_float32(gateptr[0]);
            float u = bfloat16_to_float32(upptr[0]);
            g = std::min(g, climit);
            u = std::min(std::max(u, mlimit), climit);
            outptr[0] = float32_to_bfloat16(g / (1.f + expf(-sigmoid_scale * g)) * (u + up_offset));
            gateptr++;
            upptr++;
            outptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
