// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "swiglu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "cpu.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "swiglu_bf16s.h"
#endif

SwiGLU_riscv::SwiGLU_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif // NCNN_ZFH
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

#if __riscv_vector
static void swiglu_vector_f32(const float* gateptr, const float* upptr, float* outptr, int size, float scale, float offset, float limit)
{
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m4(n);

        vfloat32m4_t _g = __riscv_vle32_v_f32m4(gateptr, vl);
        vfloat32m4_t _u = __riscv_vle32_v_f32m4(upptr, vl);

        if (limit > 0.f)
        {
            _g = __riscv_vfmin_vf_f32m4(_g, limit, vl);
            _u = __riscv_vfmax_vf_f32m4(_u, -limit, vl);
            _u = __riscv_vfmin_vf_f32m4(_u, limit, vl);
        }

        // silu(scale * g) = g * sigmoid(scale * g)
        _g = __riscv_vfmul_vv_f32m4(_g, sigmoid_ps(__riscv_vfmul_vf_f32m4(_g, scale, vl), vl), vl);
        _u = __riscv_vfadd_vf_f32m4(_u, offset, vl);

        __riscv_vse32_v_f32m4(outptr, __riscv_vfmul_vv_f32m4(_g, _u, vl), vl);

        gateptr += vl;
        upptr += vl;
        outptr += vl;
        n -= vl;
    }
}
#endif // __riscv_vector

int SwiGLU_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

#if NCNN_ZFH || NCNN_BF16
    const int elembits = gate.elembits();
#endif

#if NCNN_ZFH
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_fp16s(gate, up, top_blob, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(gate, up, top_blob, opt);
#endif

    return forward_fp32(gate, up, top_blob, opt);
}

int SwiGLU_riscv::forward_fp32(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const
{
    const int channels = gate.c;
    const int size = gate.w * gate.h * gate.d * gate.elempack;

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

#if __riscv_vector
        swiglu_vector_f32(gateptr, upptr, outptr, len, sigmoid_scale, up_offset, activation_limit);
#else
        for (int i = 0; i < len; i++)
        {
            float g = gateptr[i];
            float u = upptr[i];
            if (activation_limit > 0.f)
            {
                g = std::min(g, activation_limit);
                u = std::min(std::max(u, -activation_limit), activation_limit);
            }
            outptr[i] = g / (1.f + expf(-sigmoid_scale * g)) * (u + up_offset);
        }
#endif // __riscv_vector
    }

    return 0;
}

#if NCNN_BF16
int SwiGLU_riscv::forward_bf16s(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt) const
{
    swiglu_bf16s(gate, up, top_blob, opt, sigmoid_scale, up_offset, activation_limit);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
