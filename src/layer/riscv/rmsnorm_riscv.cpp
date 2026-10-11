// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rmsnorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector
#include <cstdio>
#include <cstdlib>

#include "cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstdlib>

namespace ncnn {

RMSNorm_riscv::RMSNorm_riscv()
{
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif
}

#if __riscv_vector
// 单行 RMSNorm（pack1、fp32）：平方和用 RVV 归约，缩放用向量乘。
// 与通用实现的数学等价；浮点求和顺序不同（向量成对归约），同一二进制内结果确定。
static void rmsnorm_rvv(float* ptr, const float* gamma_ptr, float eps, int size)
{
    /* 归约的 VL 固定为 8：向量求和的结合顺序与 VLEN 无关，保证 A100(VLEN=1024) 与
     * X100(VLEN=256) 逐字节一致（这是本项目的硬门槛）。缩放段与 VL 无关，可用满 VL。 */
    /* e32m2, not e32m1: at VLEN=128 e32m1 has only four lanes, so the hardware
     * clamps a requested vl of 8 to 4 while a constant advance of 8 would skip
     * four elements per iteration. e32m2 has exactly eight lanes at VLEN=128 and
     * more above, so the chunk always fits and the summation order stays
     * VLEN-independent. The advance uses the vl the hardware returns. */
    const size_t vlmax = __riscv_vsetvlmax_e32m2();
    const size_t REDCHUNK = vlmax < 8 ? vlmax : 8;
    vfloat32m2_t vsum = __riscv_vfmv_v_f_f32m2(0.f, vlmax);

    int i = 0;
    while (i < size)
    {
        const size_t want = (size_t)(size - i) < REDCHUNK ? (size_t)(size - i) : REDCHUNK;
        const size_t vl = __riscv_vsetvl_e32m2(want);
        vfloat32m2_t v = __riscv_vle32_v_f32m2(ptr + i, vl);
        /* _tu keeps vsum's tail: the unsuffixed form leaves it agnostic (possibly
         * NaN) and the reduction below reads the whole REDCHUNK. */
        vsum = __riscv_vfmacc_vv_f32m2_tu(vsum, v, v, vl);
        i += (int)vl;
    }

    /* 归约的目的 LMUL 是 m1：vfredusum_vs_f32m2_f32m1 的种子参数必须是
     * vfloat32m1_t，写成 m2 会找不到匹配的重载。 */
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.f, 1);
    float sqsum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m2_f32m1(vsum, vzero, REDCHUNK));

    float rms = sqsum / size;
    float a = 1.f / sqrtf(rms + eps);

    i = 0;
    if (gamma_ptr)
    {
        for (; i < size;)
        {
            size_t vl = __riscv_vsetvl_e32m1(size - i);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr + i, vl);
            vfloat32m1_t g = __riscv_vle32_v_f32m1(gamma_ptr + i, vl);
            v = __riscv_vfmul_vf_f32m1(v, a, vl);
            v = __riscv_vfmul_vv_f32m1(v, g, vl);
            __riscv_vse32_v_f32m1(ptr + i, v, vl);
            i += vl;
        }
    }
    else
    {
        for (; i < size;)
        {
            size_t vl = __riscv_vsetvl_e32m1(size - i);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr + i, vl);
            v = __riscv_vfmul_vf_f32m1(v, a, vl);
            __riscv_vse32_v_f32m1(ptr + i, v, vl);
            i += vl;
        }
    }
}
#endif // __riscv_vector

int RMSNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_ZFH
    /* The fp16 dispatch has to come first: the constructor advertises fp16
     * support whenever Zfh is available, so a non-vector RISC-V CPU can still be
     * handed a 16-bit blob. Falling through to the generic fp32 implementation
     * would read and write it at twice the element width. */
    int elembits = bottom_top_blob.elembits();

    if (opt.use_fp16_storage && elembits == 16)
    {
        return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif
#if !__riscv_vector
    return RMSNorm::forward_inplace(bottom_top_blob, opt);
#else

    int dims = bottom_top_blob.dims;

    /* RMSNorm normalises over w, w*h or w*h*d, so 4-D is a supported shape. None
     * of the branches below would run for it and returning 0 would report
     * success without normalising anything. */
    if (dims == 4)
    {
        return RMSNorm::forward_inplace(bottom_top_blob, opt);
    }

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;
        rmsnorm_rvv(ptr, gamma_data, eps, w);
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
#if __riscv_vector
            rmsnorm_rvv(ptr, gamma_data, eps, w);
#endif
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;

        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
#if __riscv_vector
                    rmsnorm_rvv(ptr, gamma_data, eps, w);
#else
                    rmsnorm(ptr, gamma_data, eps, w);
#endif
                }
            }
        }
        else // if (affine_size == size)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
#if __riscv_vector
                rmsnorm_rvv(ptr, gamma_data, eps, affine_size);
#endif
            }
        }
    }

    return 0;
#endif // __riscv_vector
}

} // namespace ncnn
