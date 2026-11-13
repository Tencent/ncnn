// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <float.h>

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void swiglu_bf16s_zvfbfmin(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt, float scale, float offset, float limit);
#endif

static void swiglu_bf16s(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt, float scale, float offset, float limit)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (ncnn::cpu_support_riscv_zvfbfmin())
    {
        swiglu_bf16s_zvfbfmin(gate, up, top_blob, opt, scale, offset, limit);
        return;
    }
#endif

    const int channels = gate.c;
    const int size = gate.w * gate.h * gate.d * gate.elempack;
    const float climit = limit > 0.f ? limit : FLT_MAX;
    const float mlimit = -climit;

    const int block_size = 1024;
    const int nn_block = (size + block_size - 1) / block_size;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < channels * nn_block; ii++)
    {
        const int q = ii / nn_block;
        const int offseti = ii % nn_block * block_size;
        const int len = std::min(size - offseti, block_size);

        const unsigned short* gateptr = (const unsigned short*)gate.channel(q) + offseti;
        const unsigned short* upptr = (const unsigned short*)up.channel(q) + offseti;
        unsigned short* outptr = (unsigned short*)top_blob.channel(q) + offseti;

#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        int n = len;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vuint16m2_t _g16 = __riscv_vle16_v_u16m2(gateptr, vl);
            vuint16m2_t _u16 = __riscv_vle16_v_u16m2(upptr, vl);

            vfloat32m4_t _g = bfloat2float_rvv(_g16, vl);
            vfloat32m4_t _u = bfloat2float_rvv(_u16, vl);

            _g = __riscv_vfmin_vf_f32m4(_g, climit, vl);
            _u = __riscv_vfmax_vf_f32m4(_u, mlimit, vl);
            _u = __riscv_vfmin_vf_f32m4(_u, climit, vl);

            // silu(scale * g) = g * sigmoid(scale * g)
            _g = __riscv_vfmul_vv_f32m4(_g, sigmoid_ps(__riscv_vfmul_vf_f32m4(_g, scale, vl), vl), vl);
            _u = __riscv_vfadd_vf_f32m4(_u, offset, vl);

            __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(__riscv_vfmul_vv_f32m4(_g, _u, vl), vl), vl);

            gateptr += vl;
            upptr += vl;
            outptr += vl;
            n -= vl;
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
        for (int i = 0; i < len; i++)
        {
            float g = bfloat16_to_float32(gateptr[i]);
            float u = bfloat16_to_float32(upptr[i]);
            g = std::min(g, climit);
            u = std::min(std::max(u, mlimit), climit);
            outptr[i] = float32_to_bfloat16(g / (1.f + expf(-scale * g)) * (u + offset));
        }
#endif // __riscv_vector
    }
}
