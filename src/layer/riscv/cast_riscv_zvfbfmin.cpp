// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mat.h"

#if NCNN_BF16
#include <riscv_vector.h>

namespace ncnn {

void cast_fp32_to_bf16_zvfbfmin(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    const int channels = bottom_blob.c;
    const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.elempack;
    const int total_bc = bottom_blob.n * channels;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int bc = 0; bc < total_bc; bc++)
    {
        int b = bc / channels;
        int q = bc % channels;
        const float* ptr = bottom_blob.batch(b).channel(q);
        unsigned short* outptr = top_blob.batch(b).channel(q);

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t _p = __riscv_vle32_v_f32m4(ptr, vl);
            vbfloat16m2_t _outp = __riscv_vfncvtbf16_f_f_w_bf16m2_rm(_p, __RISCV_FRM_RNE, vl);
            __riscv_vse16_v_u16m2(outptr, __riscv_vreinterpret_v_bf16m2_u16m2(_outp), vl);
            ptr += vl;
            outptr += vl;
            n -= vl;
        }
    }
}

void cast_bf16_to_fp32_zvfbfmin(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    const int channels = bottom_blob.c;
    const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.elempack;
    const int total_bc = bottom_blob.n * channels;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int bc = 0; bc < total_bc; bc++)
    {
        int b = bc / channels;
        int q = bc % channels;
        const unsigned short* ptr = bottom_blob.batch(b).channel(q);
        float* outptr = top_blob.batch(b).channel(q);

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vbfloat16m2_t _p = __riscv_vreinterpret_v_u16m2_bf16m2(__riscv_vle16_v_u16m2(ptr, vl));
            __riscv_vse32_v_f32m4(outptr, __riscv_vfwcvtbf16_f_f_v_f32m4(_p, vl), vl);
            ptr += vl;
            outptr += vl;
            n -= vl;
        }
    }
}

} // namespace ncnn
#endif // NCNN_BF16
