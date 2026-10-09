// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "swish_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

Swish_riscv::Swish_riscv()
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
#endif
}

#if __riscv_vector
// 向量化 exp：exp(x) = 2^(x*log2e) = 2^n * 2^f
//   n = round(x*log2e)，f = x*log2e - n ∈ [-0.5, 0.5]
//   2^f 用 6 阶多项式（相对误差 ~1e-7，与 libm expf 同量级）
//   2^n 通过整数指数位加法实现
// 逐元素运算，结果与 VLEN 无关 —— 满足跨簇逐字节一致这一硬门槛。
static inline vfloat32m1_t vexp_f32m1(vfloat32m1_t x, size_t vl)
{
    const vfloat32m1_t vlog2e = __riscv_vfmv_v_f_f32m1(1.4426950408889634f, vl);
    const vfloat32m1_t vhalf = __riscv_vfmv_v_f_f32m1(0.5f, vl);

    vfloat32m1_t t = __riscv_vfmul_vv_f32m1(x, vlog2e, vl);
    // 四舍五入到最近整数（floor(t+0.5) 对 |t|<2^22 足够，且与整数转换一致）
    vfloat32m1_t nf = __riscv_vfadd_vv_f32m1(t, vhalf, vl);
    // 饱和：n 超出 [-126,127] 时 (n<<23) 会溢出 int32 -> NaN；钳制后 2^n 取 0 或 +inf
    nf = __riscv_vfmin_vf_f32m1(nf, 127.f, vl);
    nf = __riscv_vfmax_vf_f32m1(nf, -126.f, vl);
    vint32m1_t ni = __riscv_vfcvt_x_f_v_i32m1_rm(nf, __RISCV_FRM_RDN, vl);
    vfloat32m1_t n = __riscv_vfcvt_f_x_v_f32m1(ni, vl);
    vfloat32m1_t f = __riscv_vfsub_vv_f32m1(t, n, vl);

    // 2^f 多项式（minimax，f ∈ [-0.5,0.5]）
    vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(1.3392112e-4f, vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(1.3392112e-3f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(9.6181291e-3f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(5.5504109e-2f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(2.4022651e-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(6.9314718e-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, __riscv_vfmv_v_f_f32m1(1.0f, vl), vl);

    // 2^n：把 n 加到 float 的指数位
    vint32m1_t bias = __riscv_vsll_vx_i32m1(ni, 23, vl);
    vfloat32m1_t scale = __riscv_vreinterpret_v_i32m1_f32m1(__riscv_vadd_vx_i32m1(bias, 0x3f800000, vl));
    return __riscv_vfmul_vv_f32m1(p, scale, vl);
}

static inline vfloat32m1_t vsigmoid_f32m1(vfloat32m1_t x, size_t vl)
{
    const vfloat32m1_t vone = __riscv_vfmv_v_f_f32m1(1.f, vl);
    vfloat32m1_t e = vexp_f32m1(__riscv_vfsub_vv_f32m1(__riscv_vfmv_v_f_f32m1(0.f, vl), x, vl), vl);
    return __riscv_vfrdiv_vf_f32m1(__riscv_vfadd_vv_f32m1(vone, e, vl), 1.f, vl);
}
#endif // __riscv_vector

int Swish_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_ZFH
    int elembits = bottom_top_blob.elembits();

    if (opt.use_fp16_storage && elembits == 16)
    {
        return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    // 同 fp16 路径：按"通道 × 元素"并行，避免 c=1 时退化为单线程
    const int total = channels * size;
    #pragma omp parallel for num_threads(opt.num_threads) schedule(static)
    for (int base = 0; base < total; base += 4096)
    {
        const int cnt = (total - base) < 4096 ? (total - base) : 4096;
        float* ptr = (float*)bottom_top_blob.data + base;

        int i = 0;
        const int size = cnt;
#if __riscv_vector
        for (; i < size;)
        {
            size_t vl = __riscv_vsetvl_e32m1(size - i);
            vfloat32m1_t x = __riscv_vle32_v_f32m1(ptr + i, vl);
            vfloat32m1_t r = __riscv_vfmul_vv_f32m1(x, vsigmoid_f32m1(x, vl), vl);
            __riscv_vse32_v_f32m1(ptr + i, r, vl);
            i += (int)vl;
        }
#endif
        for (; i < size; i++)
        {
            float x = ptr[i];
            ptr[i] = x / (1.f + expf(-x));
        }
    }

    return 0;
}

} // namespace ncnn
