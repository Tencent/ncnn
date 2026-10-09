// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "swish_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
#if __riscv_zvfh
// 与 fp32 路径同一多项式（见 swish_riscv.cpp），fp16 只做存取转换
static inline vfloat32m1_t vexp_f32m1_z(vfloat32m1_t x, size_t vl)
{
    const vfloat32m1_t vlog2e = __riscv_vfmv_v_f_f32m1(1.4426950408889634f, vl);
    vfloat32m1_t t = __riscv_vfmul_vv_f32m1(x, vlog2e, vl);
    vfloat32m1_t nf = __riscv_vfadd_vv_f32m1(t, __riscv_vfmv_v_f_f32m1(0.5f, vl), vl);
    // 饱和：n 超出 [-126,127] 时 (n<<23) 会溢出 int32 -> NaN；钳制后 2^n 取 0 或 +inf
    nf = __riscv_vfmin_vf_f32m1(nf, 127.f, vl);
    nf = __riscv_vfmax_vf_f32m1(nf, -126.f, vl);
    vint32m1_t ni = __riscv_vfcvt_x_f_v_i32m1_rm(nf, __RISCV_FRM_RDN, vl);
    vfloat32m1_t f = __riscv_vfsub_vv_f32m1(t, __riscv_vfcvt_f_x_v_f32m1(ni, vl), vl);
    vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(1.3392112e-4f, vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(1.3392112e-3f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(9.6181291e-3f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(5.5504109e-2f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(2.4022651e-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(6.9314718e-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(1.0f, vl), vl);
    vint32m1_t bias = __riscv_vsll_vx_i32m1(ni, 23, vl);
    vfloat32m1_t scale = __riscv_vreinterpret_v_i32m1_f32m1(__riscv_vadd_vx_i32m1(bias, 0x3f800000, vl));
    return __riscv_vfmul_vv_f32m1(p, scale, vl);
}
#endif

int Swish_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    // 并行维度取"通道 × 元素"：LLM 的激活性张量通常 c=1（如 [3072, L, 1]），
    // 只按通道并行会退化成单线程（实测单线程标量 expf 高达 44.7 ns/元素）。
    const int total = channels * size;
    #pragma omp parallel for num_threads(opt.num_threads) schedule(static)
    for (int base = 0; base < total; base += 4096)
    {
        const int cnt = (total - base) < 4096 ? (total - base) : 4096;
        __fp16* ptr = (__fp16*)bottom_top_blob.data + base;

        int i = 0;
        const int size = cnt;
#if __riscv_zvfh
        for (; i < size;)
        {
            size_t vl = __riscv_vsetvl_e16mf2(size - i);
            vfloat32m1_t x = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(ptr + i, vl), vl);
            vfloat32m1_t sig = __riscv_vfrdiv_vf_f32m1(
                                   __riscv_vfadd_vv_f32m1(__riscv_vfmv_v_f_f32m1(1.f, vl),
                                           vexp_f32m1_z(__riscv_vfsub_vv_f32m1(__riscv_vfmv_v_f_f32m1(0.f, vl), x, vl), vl), vl),
                                   1.f, vl);
            vfloat32m1_t r = __riscv_vfmul_vv_f32m1(x, sig, vl);
            __riscv_vse16_v_f16mf2(ptr + i, __riscv_vfncvt_f_f_w_f16mf2(r, vl), vl);
            i += (int)vl;
        }
#endif
        for (; i < size; i++)
        {
            float x = (float)ptr[i];
            ptr[i] = (__fp16)(x / (1.f + expf(-x)));
        }
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
