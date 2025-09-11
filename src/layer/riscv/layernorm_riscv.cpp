// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

LayerNorm_riscv::LayerNorm_riscv()
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
#if __riscv_xtheadvector
// FIXME inline causes illegal instruction :(
__attribute__((noinline)) static vfloat32m8_t
reset_tails(vfloat32m8_t x, size_t vl, float v)
{
    const size_t vlm8 = __riscv_vsetvlmax_e32m8();
    vbool4_t _vl_mask = __riscv_vmsgeu_vx_u32m8_b4(__riscv_vid_v_u32m8(vlm8), vl, vlm8);
    x = __riscv_vfmerge_vfm_f32m8(x, v, _vl_mask, vlm8);
    return x;
}
#endif // __riscv_xtheadvector
#endif // __riscv_vector

static void layernorm(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const size_t size = elemcount * elempack;

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;

    // reduce sum
    vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvlmax_e32m8());
    {
        const float* ptr0 = ptr;

        int n = size / __riscv_vsetvlmax_e32m8() * __riscv_vsetvlmax_e32m8();
        const size_t vl = __riscv_vsetvlmax_e32m8();
        while (n > 0)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vl);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl);
            ptr0 += vl;
            n -= vl;
        }
        int remain = size % __riscv_vsetvlmax_e32m8();
        if (remain > 0)
        {
            size_t vlr = __riscv_vsetvl_e32m8(remain);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vlr);
#if __riscv_xtheadvector
            // xtheadvector does not support tail undisturbed policy
            _p = reset_tails(_p, vlr, 0.f);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl);
#else  // __riscv_xtheadvector
            _sum = __riscv_vfadd_vv_f32m8_tu(_sum, _sum, _p, vlr);
#endif // __riscv_xtheadvector
        }
    }

    vfloat32m8_t _mean;
    if (elempack == packn)
    {
        // reduce sum n,n,n,n,n,n,n,n to n
        // broadcast n to n,n,n,n,n,n,n,n

        vfloat32m4_t _sum0 = __riscv_vfadd_vv_f32m4(__riscv_vget_v_f32m8_f32m4(_sum, 0), __riscv_vget_v_f32m8_f32m4(_sum, 1), __riscv_vsetvlmax_e32m4());
        vfloat32m2_t _sum2 = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(_sum0, 0), __riscv_vget_v_f32m4_f32m2(_sum0, 1), __riscv_vsetvlmax_e32m2());
        vfloat32m1_t _sum4 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(_sum2, 0), __riscv_vget_v_f32m2_f32m1(_sum2, 1), __riscv_vsetvlmax_e32m1());
        vfloat32m1_t _mean4 = __riscv_vfdiv_vf_f32m1(_sum4, elemcount, __riscv_vsetvlmax_e32m1());
        _mean = __riscv_vcreate_v_f32m1_f32m8(_mean4, _mean4, _mean4, _mean4, _mean4, _mean4, _mean4, _mean4);
    }
    else // if (elempack == 1)
    {
        // reduce sum n,n,n,n,n,n,n,n to 1
        // broadcast 1 to n,n,n,n,n,n,n,n

        vfloat32m1_t _sum0 = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        _sum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sum, _sum0, __riscv_vsetvlmax_e32m8());
        vfloat32m1_t _mean0 = __riscv_vfdiv_vf_f32m1(_sum0, elemcount, __riscv_vsetvlmax_e32m1());
        _mean = __riscv_vset_v_f32m1_f32m8(__riscv_vundefined_f32m8(), 0, _mean0);
        _mean = __riscv_vrgather_vx_f32m8(_mean, 0, __riscv_vsetvlmax_e32m8());
    }

    // reduce sqsum
    vfloat32m8_t _sqsum = __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvlmax_e32m8());
    {
        const float* ptr0 = ptr;

        int n = size / __riscv_vsetvlmax_e32m8() * __riscv_vsetvlmax_e32m8();
        const size_t vl = __riscv_vsetvlmax_e32m8();
        while (n > 0)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vl);
            _p = __riscv_vfsub_vv_f32m8(_p, _mean, vl);
            _sqsum = __riscv_vfmadd_vv_f32m8(_p, _p, _sqsum, vl);
            ptr0 += vl;
            n -= vl;
        }
        int remain = size % __riscv_vsetvlmax_e32m8();
        if (remain > 0)
        {
            size_t vlr = __riscv_vsetvl_e32m8(remain);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vlr);
            _p = __riscv_vfsub_vv_f32m8(_p, _mean, vlr);
#if __riscv_xtheadvector
            // xtheadvector does not support tail undisturbed policy
            _p = reset_tails(_p, vlr, 0.f);
            _sqsum = __riscv_vfmadd_vv_f32m8(_p, _p, _sqsum, vl);
#else  // __riscv_xtheadvector
            _p = __riscv_vfmul_vv_f32m8(_p, _p, vlr);
            _sqsum = __riscv_vfadd_vv_f32m8_tu(_sqsum, _sqsum, _p, vlr);
#endif // __riscv_xtheadvector
        }
    }

    vfloat32m8_t _var;
    if (elempack == packn)
    {
        // reduce sqsum n,n,n,n,n,n,n,n to n
        // broadcast n to n,n,n,n,n,n,n,n

        vfloat32m4_t _sqsum0 = __riscv_vfadd_vv_f32m4(__riscv_vget_v_f32m8_f32m4(_sqsum, 0), __riscv_vget_v_f32m8_f32m4(_sqsum, 1), __riscv_vsetvlmax_e32m4());
        vfloat32m2_t _sqsum2 = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(_sqsum0, 0), __riscv_vget_v_f32m4_f32m2(_sqsum0, 1), __riscv_vsetvlmax_e32m2());
        vfloat32m1_t _sqsum4 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(_sqsum2, 0), __riscv_vget_v_f32m2_f32m1(_sqsum2, 1), __riscv_vsetvlmax_e32m1());
        vfloat32m1_t _var4 = __riscv_vfdiv_vf_f32m1(_sqsum4, elemcount, __riscv_vsetvlmax_e32m1());
        _var = __riscv_vcreate_v_f32m1_f32m8(_var4, _var4, _var4, _var4, _var4, _var4, _var4, _var4);
    }
    else // if (elempack == 1)
    {
        // reduce sqsum n,n,n,n,n,n,n,n to 1
        // broadcast 1 to n,n,n,n,n,n,n,n

        vfloat32m1_t _sqsum0 = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        _sqsum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sqsum, _sqsum0, __riscv_vsetvlmax_e32m8());
        vfloat32m1_t _var0 = __riscv_vfdiv_vf_f32m1(_sqsum0, elemcount, __riscv_vsetvlmax_e32m1());
        _var = __riscv_vset_v_f32m1_f32m8(__riscv_vundefined_f32m8(), 0, _var0);
        _var = __riscv_vrgather_vx_f32m8(_var, 0, __riscv_vsetvlmax_e32m8());
    }

    // a = 1.f / sqrtf(var + eps);
    // b = -mean * a;
    vfloat32m8_t _a = __riscv_vfrdiv_vf_f32m8(__riscv_vfsqrt_v_f32m8(__riscv_vfadd_vf_f32m8(_var, eps, __riscv_vsetvlmax_e32m8()), __riscv_vsetvlmax_e32m8()), 1.f, __riscv_vsetvlmax_e32m8());
    vfloat32m8_t _b = __riscv_vfmul_vv_f32m8(__riscv_vfsgnjn_vv_f32m8(_mean, _mean, __riscv_vsetvlmax_e32m8()), _a, __riscv_vsetvlmax_e32m8());

    if (gamma_ptr && beta_ptr)
    {
        if (elempack == packn)
        {
            const size_t vlm8 = __riscv_vsetvlmax_e32m8();
            const size_t vlm4 = __riscv_vsetvlmax_e32m4();
            const size_t vlm2 = __riscv_vsetvlmax_e32m2();
            const size_t vlm1 = __riscv_vsetvlmax_e32m1();

            int i = 0;
            for (; i + 7 < elemcount; i += 8)
            {
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vlm8);
                _p = __riscv_vfmadd_vv_f32m8(_p, _a, _b, vlm8);
                vfloat32m1_t _gamma0 = __riscv_vfmv_v_f_f32m1(gamma_ptr[0], vlm1);
                vfloat32m1_t _gamma1 = __riscv_vfmv_v_f_f32m1(gamma_ptr[1], vlm1);
                vfloat32m1_t _gamma2 = __riscv_vfmv_v_f_f32m1(gamma_ptr[2], vlm1);
                vfloat32m1_t _gamma3 = __riscv_vfmv_v_f_f32m1(gamma_ptr[3], vlm1);
                vfloat32m1_t _gamma4 = __riscv_vfmv_v_f_f32m1(gamma_ptr[4], vlm1);
                vfloat32m1_t _gamma5 = __riscv_vfmv_v_f_f32m1(gamma_ptr[5], vlm1);
                vfloat32m1_t _gamma6 = __riscv_vfmv_v_f_f32m1(gamma_ptr[6], vlm1);
                vfloat32m1_t _gamma7 = __riscv_vfmv_v_f_f32m1(gamma_ptr[7], vlm1);
                vfloat32m8_t _gamma = __riscv_vcreate_v_f32m1_f32m8(_gamma0, _gamma1, _gamma2, _gamma3, _gamma4, _gamma5, _gamma6, _gamma7);
                vfloat32m1_t _beta0 = __riscv_vfmv_v_f_f32m1(beta_ptr[0], vlm1);
                vfloat32m1_t _beta1 = __riscv_vfmv_v_f_f32m1(beta_ptr[1], vlm1);
                vfloat32m1_t _beta2 = __riscv_vfmv_v_f_f32m1(beta_ptr[2], vlm1);
                vfloat32m1_t _beta3 = __riscv_vfmv_v_f_f32m1(beta_ptr[3], vlm1);
                vfloat32m1_t _beta4 = __riscv_vfmv_v_f_f32m1(beta_ptr[4], vlm1);
                vfloat32m1_t _beta5 = __riscv_vfmv_v_f_f32m1(beta_ptr[5], vlm1);
                vfloat32m1_t _beta6 = __riscv_vfmv_v_f_f32m1(beta_ptr[6], vlm1);
                vfloat32m1_t _beta7 = __riscv_vfmv_v_f_f32m1(beta_ptr[7], vlm1);
                vfloat32m8_t _beta = __riscv_vcreate_v_f32m1_f32m8(_beta0, _beta1, _beta2, _beta3, _beta4, _beta5, _beta6, _beta7);
                _p = __riscv_vfmadd_vv_f32m8(_p, _gamma, _beta, vlm8);
                __riscv_vse32_v_f32m8(ptr, _p, vlm8);
                ptr += vlm8;
                gamma_ptr += 8;
                beta_ptr += 8;
            }
            for (; i + 3 < elemcount; i += 4)
            {
                vfloat32m4_t _p = __riscv_vle32_v_f32m4(ptr, vlm4);
                _p = __riscv_vfmadd_vv_f32m4(_p, __riscv_vget_v_f32m8_f32m4(_a, 0), __riscv_vget_v_f32m8_f32m4(_b, 0), vlm4);
                vfloat32m1_t _gamma0 = __riscv_vfmv_v_f_f32m1(gamma_ptr[0], vlm1);
                vfloat32m1_t _gamma1 = __riscv_vfmv_v_f_f32m1(gamma_ptr[1], vlm1);
                vfloat32m1_t _gamma2 = __riscv_vfmv_v_f_f32m1(gamma_ptr[2], vlm1);
                vfloat32m1_t _gamma3 = __riscv_vfmv_v_f_f32m1(gamma_ptr[3], vlm1);
                vfloat32m4_t _gamma = __riscv_vcreate_v_f32m1_f32m4(_gamma0, _gamma1, _gamma2, _gamma3);
                vfloat32m1_t _beta0 = __riscv_vfmv_v_f_f32m1(beta_ptr[0], vlm1);
                vfloat32m1_t _beta1 = __riscv_vfmv_v_f_f32m1(beta_ptr[1], vlm1);
                vfloat32m1_t _beta2 = __riscv_vfmv_v_f_f32m1(beta_ptr[2], vlm1);
                vfloat32m1_t _beta3 = __riscv_vfmv_v_f_f32m1(beta_ptr[3], vlm1);
                vfloat32m4_t _beta = __riscv_vcreate_v_f32m1_f32m4(_beta0, _beta1, _beta2, _beta3);
                _p = __riscv_vfmadd_vv_f32m4(_p, _gamma, _beta, vlm4);
                __riscv_vse32_v_f32m4(ptr, _p, vlm4);
                ptr += vlm4;
                gamma_ptr += 4;
                beta_ptr += 4;
            }
            for (; i + 1 < elemcount; i += 2)
            {
                vfloat32m2_t _p = __riscv_vle32_v_f32m2(ptr, vlm2);
                _p = __riscv_vfmadd_vv_f32m2(_p, __riscv_vget_v_f32m8_f32m2(_a, 0), __riscv_vget_v_f32m8_f32m2(_b, 0), vlm2);
                vfloat32m1_t _gamma0 = __riscv_vfmv_v_f_f32m1(gamma_ptr[0], vlm1);
                vfloat32m1_t _gamma1 = __riscv_vfmv_v_f_f32m1(gamma_ptr[1], vlm1);
                vfloat32m2_t _gamma = __riscv_vcreate_v_f32m1_f32m2(_gamma0, _gamma1);
                vfloat32m1_t _beta0 = __riscv_vfmv_v_f_f32m1(beta_ptr[0], vlm1);
                vfloat32m1_t _beta1 = __riscv_vfmv_v_f_f32m1(beta_ptr[1], vlm1);
                vfloat32m2_t _beta = __riscv_vcreate_v_f32m1_f32m2(_beta0, _beta1);
                _p = __riscv_vfmadd_vv_f32m2(_p, _gamma, _beta, vlm2);
                __riscv_vse32_v_f32m2(ptr, _p, vlm2);
                ptr += vlm2;
                gamma_ptr += 2;
                beta_ptr += 2;
            }
            for (; i < elemcount; i++)
            {
                vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vlm1);
                _p = __riscv_vfmadd_vv_f32m1(_p, __riscv_vget_v_f32m8_f32m1(_a, 0), __riscv_vget_v_f32m8_f32m1(_b, 0), vlm1);
                _p = __riscv_vfmul_vf_f32m1(_p, gamma_ptr[0], vlm1);
                _p = __riscv_vfadd_vf_f32m1(_p, beta_ptr[0], vlm1);
                __riscv_vse32_v_f32m1(ptr, _p, vlm1);
                ptr += vlm1;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        else // if (elempack == 1)
        {
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                _p = __riscv_vfmadd_vv_f32m8(_p, _a, _b, vl);
                vfloat32m8_t _gamma = __riscv_vle32_v_f32m8(gamma_ptr, vl);
                vfloat32m8_t _beta = __riscv_vle32_v_f32m8(beta_ptr, vl);
                _p = __riscv_vfmadd_vv_f32m8(_p, _gamma, _beta, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);
                n -= vl;
                ptr += vl;
                gamma_ptr += vl;
                beta_ptr += vl;
            }
        }
    }
    else
    {
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = __riscv_vfmadd_vv_f32m8(_p, _a, _b, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            n -= vl;
            ptr += vl;
            gamma_ptr += vl;
            beta_ptr += vl;
        }
    }
#else  // __riscv_vector
    float sum = 0.f;
    for (int i = 0; i < size; i++)
    {
        sum += ptr[i];
    }

    float mean = sum / size;

    float sqsum = 0.f;
    for (int i = 0; i < size; i++)
    {
        float v = ptr[i] - mean;
        sqsum += v * v;
    }

    float var = sqsum / size;

    float a = 1.f / sqrtf(var + eps);
    float b = -mean * a;

    if (gamma_ptr && beta_ptr)
    {
        for (int i = 0; i < size; i++)
        {
            ptr[i] = (ptr[i] * a + b) * gamma_ptr[i] + beta_ptr[i];
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            ptr[i] = ptr[i] * a + b;
        }
    }
#endif // __riscv_vector
}

int LayerNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_ZFH
    const int elembits = bottom_top_blob.elembits();
    if (opt.use_fp16_storage && elembits == 16)
    {
        return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif // NCNN_ZFH

    const int elempack = bottom_top_blob.elempack;
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        layernorm(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            layernorm(ptr, gamma_data, beta_data, eps, w, elempack);
        }
    }

    if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    layernorm(ptr, gamma_data, beta_data, eps, w, elempack);
                }
            }
        }
        else // if (affine_size == size)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                layernorm(ptr, gamma_data, beta_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}

} // namespace ncnn
