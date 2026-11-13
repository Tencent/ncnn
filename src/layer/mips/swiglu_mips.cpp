// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "swiglu_mips.h"

#include <float.h>

#if __mips_msa
#include <msa.h>
#include "mips_usability.h"
#include "msa_mathfun.h"
#endif // __mips_msa

namespace ncnn {

SwiGLU_mips::SwiGLU_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

#if __mips_msa
static inline v4f32 swiglu_f32x4(v4f32 _g, v4f32 _u, v4f32 _scale, v4f32 _offset, v4f32 _limit, v4f32 _mlimit, v4f32 _one)
{
    _g = __msa_fmin_w(_g, _limit);
    _u = __msa_fmin_w(__msa_fmax_w(_u, _mlimit), _limit);

    // silu(scale * g) = g / (1 + exp(-scale * g))
    v4f32 _z = __msa_fmul_w(_g, _scale);
    _z = (v4f32)__msa_bnegi_w((v4u32)_z, 31);
    _g = __msa_fdiv_w(_g, __msa_fadd_w(_one, exp_ps(_z)));

    return __msa_fmul_w(_g, __msa_fadd_w(_u, _offset));
}
#endif // __mips_msa

int SwiGLU_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

int SwiGLU_mips::forward_fp32(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const
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
#if __mips_msa
        v4f32 _scale = (v4f32)__msa_fill_w_f32(sigmoid_scale);
        v4f32 _offset = (v4f32)__msa_fill_w_f32(up_offset);
        v4f32 _limit = (v4f32)__msa_fill_w_f32(climit);
        v4f32 _mlimit = (v4f32)__msa_fill_w_f32(mlimit);
        v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
        for (; i + 3 < len; i += 4)
        {
            v4f32 _g = (v4f32)__msa_ld_w((void*)gateptr, 0);
            v4f32 _u = (v4f32)__msa_ld_w((void*)upptr, 0);
            __msa_st_w((v4i32)swiglu_f32x4(_g, _u, _scale, _offset, _limit, _mlimit, _one), outptr, 0);

            gateptr += 4;
            upptr += 4;
            outptr += 4;
        }
#endif // __mips_msa
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
int SwiGLU_mips::forward_bf16s(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const
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
#if __mips_msa
        v4f32 _scale = (v4f32)__msa_fill_w_f32(sigmoid_scale);
        v4f32 _offset = (v4f32)__msa_fill_w_f32(up_offset);
        v4f32 _limit = (v4f32)__msa_fill_w_f32(climit);
        v4f32 _mlimit = (v4f32)__msa_fill_w_f32(mlimit);
        v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
        v8i16 _zero = __msa_fill_h(0);
        for (; i + 7 < len; i += 8)
        {
            v8i16 _g16 = __msa_ld_h((void*)gateptr, 0);
            v8i16 _u16 = __msa_ld_h((void*)upptr, 0);
            v4f32 _g0 = (v4f32)__msa_ilvr_h(_g16, _zero);
            v4f32 _g1 = (v4f32)__msa_ilvl_h(_g16, _zero);
            v4f32 _u0 = (v4f32)__msa_ilvr_h(_u16, _zero);
            v4f32 _u1 = (v4f32)__msa_ilvl_h(_u16, _zero);
            v4f32 _out0 = swiglu_f32x4(_g0, _u0, _scale, _offset, _limit, _mlimit, _one);
            v4f32 _out1 = swiglu_f32x4(_g1, _u1, _scale, _offset, _limit, _mlimit, _one);
            __msa_st_w(float2bfloat_msa(_out0, _out1), outptr, 0);

            gateptr += 8;
            upptr += 8;
            outptr += 8;
        }
        for (; i + 3 < len; i += 4)
        {
            v4f32 _g = bfloat2float_msa(gateptr);
            v4f32 _u = bfloat2float_msa(upptr);
            v4f32 _out = swiglu_f32x4(_g, _u, _scale, _offset, _limit, _mlimit, _one);
            *(int64_t*)outptr = __msa_copy_s_d((v2i64)float2bfloat_msa(_out), 0);

            gateptr += 4;
            upptr += 4;
            outptr += 4;
        }
#endif // __mips_msa
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
