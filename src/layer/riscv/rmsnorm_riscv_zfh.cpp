// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rmsnorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
// 单行 RMSNorm（pack1、fp16）：向量内做 fp16<->fp32 转换，平方和用 RVV 归约。
// 该函数在 _zfh.cpp（标量 zfh）与生成的 _zfh_rvv.cpp（RVV+zvfh）里各编译一次。
static void rmsnorm_fp16s_rvv(__fp16* ptr, const float* gamma_ptr, float eps, int size)
{
#if !__riscv_zvfh
    // 标量回退：与通用 fp32 实现同一数学路径
    float sqsum = 0.f;
    for (int i = 0; i < size; i++)
    {
        float v = (float)ptr[i];
        sqsum += v * v;
    }
    float a = 1.f / sqrtf(sqsum / size + eps);
    if (gamma_ptr)
    {
        for (int i = 0; i < size; i++)
            ptr[i] = (__fp16)(((float)ptr[i] * a) * gamma_ptr[i]);
    }
    else
    {
        for (int i = 0; i < size; i++)
            ptr[i] = (__fp16)((float)ptr[i] * a);
    }
#else
    /* same correction as the fp32 path: e32m2 guarantees eight lanes at
     * VLEN=128 (e32m1 has four and the hardware would clamp the requested vl
     * while a constant advance of 8 skipped elements), the advance uses the
     * returned vl, and _tu keeps the accumulator tail defined. */
    const size_t vlmax = __riscv_vsetvlmax_e32m2();
    const size_t REDCHUNK = vlmax < 8 ? vlmax : 8;
    vfloat32m2_t vsum = __riscv_vfmv_v_f_f32m2(0.f, vlmax);

    int i = 0;
    while (i < size)
    {
        const size_t want = (size_t)(size - i) < REDCHUNK ? (size_t)(size - i) : REDCHUNK;
        const size_t vl = __riscv_vsetvl_e16m1(want);
        vfloat16m1_t hv = __riscv_vle16_v_f16m1((const __fp16*)(ptr + i), vl);
        vfloat32m2_t v = __riscv_vfwcvt_f_f_v_f32m2(hv, vl);
        vsum = __riscv_vfmacc_vv_f32m2_tu(vsum, v, v, vl);
        i += (int)vl;
    }

    /* 归约的目的 LMUL 是 m1：vfredusum_vs_f32m2_f32m1 的种子参数必须是
     * vfloat32m1_t，写成 m2 会找不到匹配的重载。 */
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.f, 1);
    float sqsum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vsum, vzero, REDCHUNK));

    float rms = sqsum / size;
    float a = 1.f / sqrtf(rms + eps);

    i = 0;
    if (gamma_ptr)
    {
        for (; i < size;)
        {
            size_t vl = __riscv_vsetvl_e16mf2(size - i);
            vfloat16mf2_t hv = __riscv_vle16_v_f16mf2((const __fp16*)(ptr + i), vl);
            vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(hv, vl);
            vfloat32m1_t g = __riscv_vle32_v_f32m1(gamma_ptr + i, vl);
            v = __riscv_vfmul_vf_f32m1(v, a, vl);
            v = __riscv_vfmul_vv_f32m1(v, g, vl);
            __riscv_vse16_v_f16mf2((__fp16*)(ptr + i), __riscv_vfncvt_f_f_w_f16mf2(v, vl), vl);
            i += vl;
        }
    }
    else
    {
        for (; i < size;)
        {
            size_t vl = __riscv_vsetvl_e16mf2(size - i);
            vfloat16mf2_t hv = __riscv_vle16_v_f16mf2((const __fp16*)(ptr + i), vl);
            vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(hv, vl);
            v = __riscv_vfmul_vf_f32m1(v, a, vl);
            __riscv_vse16_v_f16mf2((__fp16*)(ptr + i), __riscv_vfncvt_f_f_w_f16mf2(v, vl), vl);
            i += vl;
        }
    }
#endif // __riscv_vector
}

int RMSNorm_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;
        __fp16* ptr = (__fp16*)bottom_top_blob;
        rmsnorm_fp16s_rvv(ptr, gamma_data, eps, w);
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = (__fp16*)bottom_top_blob.row(i);
            rmsnorm_fp16s_rvv(ptr, gamma_data, eps, w);
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
                    __fp16* ptr = (__fp16*)bottom_top_blob.channel(q).row(i);
                    rmsnorm_fp16s_rvv(ptr, gamma_data, eps, w);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                __fp16* ptr = (__fp16*)bottom_top_blob.channel(q);
                rmsnorm_fp16s_rvv(ptr, gamma_data, eps, affine_size);
            }
        }
    }

    if (dims == 4)
    {
        /* RMSNorm normalises over w, w*h or w*h*d, so 4-D is a supported shape;
         * without this branch the helper reports success without normalising. */
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;

        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
                for (int z = 0; z < d; z++)
                    for (int y = 0; y < h; y++)
                        rmsnorm_fp16s_rvv((__fp16*)bottom_top_blob.channel(q).depth(z).row(y), gamma_data, eps, w);
        }
        else if (affine_size == w * h)
        {
            /* affine_size == w*h means one RMS over the whole contiguous plane,
             * with all w*h gammas -- same as the dims==3 else-branch and the
             * generic implementation. Normalising per row instead would compute
             * a different RMS for each row and apply only gamma[0:w]. */
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
                for (int z = 0; z < d; z++)
                {
                    __fp16* ptr = (__fp16*)bottom_top_blob.channel(q).depth(z);
                    rmsnorm_fp16s_rvv(ptr, gamma_data, eps, w * h);
                }
        }
        else // if (affine_size == w * h * d)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
                rmsnorm_fp16s_rvv((__fp16*)bottom_top_blob.channel(q), gamma_data, eps, w * h * d);
        }
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
