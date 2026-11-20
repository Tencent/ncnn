// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void layernorm_bf16s_zvfbfmin(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack);
#endif

static void layernorm_bf16s(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        layernorm_bf16s_zvfbfmin(ptr, gamma_ptr, beta_ptr, eps, elemcount, elempack);
        return;
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;

    if (elempack == packn)
    {
        // f32m4 covers two packed pixels of packn lanes
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        const size_t vlmax4 = __riscv_vsetvlmax_e32m4();
        const size_t vlmax2 = __riscv_vsetvlmax_e32m2();

        const int npix = elemcount / 2 * 2;
        const int size2 = npix * packn;

        // reduce sum
        vfloat32m4_t _sum = __riscv_vfmv_v_f_f32m4(0.f, vlmax4);
        {
            int i = 0;
            for (; i < size2; i += vlmax4)
            {
                vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + i, vlmax4), vlmax4);
                _sum = __riscv_vfadd_vv_f32m4(_sum, _p, vlmax4);
            }

            vfloat32m2_t _sum2 = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(_sum, 0), __riscv_vget_v_f32m4_f32m2(_sum, 1), vlmax2);
            if (elemcount & 1)
            {
                vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + size2, vlmax2), vlmax2);
                _sum2 = __riscv_vfadd_vv_f32m2(_sum2, __riscv_vget_v_f32m4_f32m2(_p, 0), vlmax2);
            }

            vfloat32m2_t _mean2 = __riscv_vfdiv_vf_f32m2(_sum2, elemcount, vlmax2);
            vfloat32m4_t _mean = __riscv_vcreate_v_f32m2_f32m4(_mean2, _mean2);

            // reduce sqsum
            vfloat32m4_t _sqsum = __riscv_vfmv_v_f_f32m4(0.f, vlmax4);
            vfloat32m2_t _sqsum2 = __riscv_vfmv_v_f_f32m2(0.f, vlmax2);
            i = 0;
            for (; i < size2; i += vlmax4)
            {
                vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + i, vlmax4), vlmax4);
                _p = __riscv_vfsub_vv_f32m4(_p, _mean, vlmax4);
                _sqsum = __riscv_vfmadd_vv_f32m4(_p, _p, _sqsum, vlmax4);
            }
            if (elemcount & 1)
            {
                vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + size2, vlmax2), vlmax2);
                _p = __riscv_vfsub_vv_f32m4(_p, _mean, vlmax2);
                vfloat32m2_t _p2 = __riscv_vget_v_f32m4_f32m2(_p, 0);
                _sqsum2 = __riscv_vfmadd_vv_f32m2(_p2, _p2, _sqsum2, vlmax2);
            }

            vfloat32m2_t _sq = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(_sqsum, 0), __riscv_vget_v_f32m4_f32m2(_sqsum, 1), vlmax2);
            _sq = __riscv_vfadd_vv_f32m2(_sq, _sqsum2, vlmax2);

            // a = 1.f / sqrtf(var + eps);
            // b = -mean * a;
            vfloat32m2_t _var = __riscv_vfdiv_vf_f32m2(_sq, elemcount, vlmax2);
            vfloat32m2_t _a = __riscv_vfrdiv_vf_f32m2(__riscv_vfsqrt_v_f32m2(__riscv_vfadd_vf_f32m2(_var, eps, vlmax2), vlmax2), 1.f, vlmax2);
            vfloat32m2_t _b = __riscv_vfmul_vv_f32m2(__riscv_vfsgnjn_vv_f32m2(_mean2, _mean2, vlmax2), _a, vlmax2);

            vfloat32m4_t _a4 = __riscv_vcreate_v_f32m2_f32m4(_a, _a);
            vfloat32m4_t _b4 = __riscv_vcreate_v_f32m2_f32m4(_b, _b);

            if (gamma_ptr && beta_ptr)
            {
                i = 0;
                for (; i < size2; i += vlmax4)
                {
                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + i, vlmax4), vlmax4);
                    _p = __riscv_vfmadd_vv_f32m4(_p, _a4, _b4, vlmax4);
                    vfloat32m2_t _gamma0 = __riscv_vfmv_v_f_f32m2(gamma_ptr[0], vlmax2);
                    vfloat32m2_t _gamma1 = __riscv_vfmv_v_f_f32m2(gamma_ptr[1], vlmax2);
                    vfloat32m4_t _gamma = __riscv_vcreate_v_f32m2_f32m4(_gamma0, _gamma1);
                    vfloat32m2_t _beta0 = __riscv_vfmv_v_f_f32m2(beta_ptr[0], vlmax2);
                    vfloat32m2_t _beta1 = __riscv_vfmv_v_f_f32m2(beta_ptr[1], vlmax2);
                    vfloat32m4_t _beta = __riscv_vcreate_v_f32m2_f32m4(_beta0, _beta1);
                    _p = __riscv_vfmadd_vv_f32m4(_p, _gamma, _beta, vlmax4);
                    __riscv_vse16_v_u16m2(ptr + i, float2bfloat_rvv(_p, vlmax4), vlmax4);
                    gamma_ptr += 2;
                    beta_ptr += 2;
                }
                if (elemcount & 1)
                {
                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + size2, vlmax2), vlmax2);
                    _p = __riscv_vfmadd_vv_f32m4(_p, _a4, _b4, vlmax2);
                    _p = __riscv_vfmul_vf_f32m4(_p, gamma_ptr[0], vlmax2);
                    _p = __riscv_vfadd_vf_f32m4(_p, beta_ptr[0], vlmax2);
                    __riscv_vse16_v_u16m2(ptr + size2, float2bfloat_rvv(_p, vlmax2), vlmax2);
                }
            }
            else
            {
                i = 0;
                for (; i < size2; i += vlmax4)
                {
                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + i, vlmax4), vlmax4);
                    _p = __riscv_vfmadd_vv_f32m4(_p, _a4, _b4, vlmax4);
                    __riscv_vse16_v_u16m2(ptr + i, float2bfloat_rvv(_p, vlmax4), vlmax4);
                }
                if (elemcount & 1)
                {
                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + size2, vlmax2), vlmax2);
                    _p = __riscv_vfmadd_vv_f32m4(_p, _a4, _b4, vlmax2);
                    __riscv_vse16_v_u16m2(ptr + size2, float2bfloat_rvv(_p, vlmax2), vlmax2);
                }
            }
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif

        return;
    }

    if (elempack == 1)
    {
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        const size_t vlmax4 = __riscv_vsetvlmax_e32m4();
        const size_t vlmax = __riscv_vsetvlmax_e16m2();
        const size_t vlmax1 = __riscv_vsetvlmax_e32m1();

        const int size = elemcount;

        // reduce sum
        vfloat32m4_t _sum = __riscv_vfmv_v_f_f32m4(0.f, vlmax4);
        int nfull = size / (int)vlmax * (int)vlmax;
        {
            int i = 0;
            for (; i < nfull; i += vlmax)
            {
                vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + i, vlmax), vlmax);
                _sum = __riscv_vfadd_vv_f32m4(_sum, _p, vlmax);
            }

            vfloat32m1_t _sum1 = __riscv_vfmv_s_f_f32m1(0.f, vlmax1);
            _sum1 = __riscv_vfredusum_vs_f32m4_f32m1(_sum, _sum1, vlmax4);
            float mean = __riscv_vfmv_f_s_f32m1_f32(_sum1);
            for (; i < size; i++)
            {
                mean += bfloat16_to_float32(ptr[i]);
            }
            mean /= elemcount;

            // reduce sqsum
            vfloat32m4_t _sqsum = __riscv_vfmv_v_f_f32m4(0.f, vlmax4);
            vfloat32m4_t _mean = __riscv_vfmv_v_f_f32m4(mean, vlmax4);
            i = 0;
            for (; i < nfull; i += vlmax)
            {
                vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + i, vlmax), vlmax);
                _p = __riscv_vfsub_vv_f32m4(_p, _mean, vlmax);
                _sqsum = __riscv_vfmadd_vv_f32m4(_p, _p, _sqsum, vlmax);
            }

            vfloat32m1_t _sqsum1 = __riscv_vfmv_s_f_f32m1(0.f, vlmax1);
            _sqsum1 = __riscv_vfredusum_vs_f32m4_f32m1(_sqsum, _sqsum1, vlmax4);
            float sqsum = __riscv_vfmv_f_s_f32m1_f32(_sqsum1);
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(ptr[i]) - mean;
                sqsum += v * v;
            }

            float var = sqsum / elemcount;
            float a = 1.f / sqrtf(var + eps);
            float b = -mean * a;

            if (gamma_ptr && beta_ptr)
            {
                vfloat32m4_t _a = __riscv_vfmv_v_f_f32m4(a, vlmax4);
                vfloat32m4_t _b = __riscv_vfmv_v_f_f32m4(b, vlmax4);
                i = 0;
                for (; i < nfull; i += vlmax)
                {
                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + i, vlmax), vlmax);
                    _p = __riscv_vfmadd_vv_f32m4(_p, _a, _b, vlmax);
                    vfloat32m4_t _gamma = __riscv_vle32_v_f32m4(gamma_ptr + i, vlmax);
                    vfloat32m4_t _beta = __riscv_vle32_v_f32m4(beta_ptr + i, vlmax);
                    _p = __riscv_vfmadd_vv_f32m4(_p, _gamma, _beta, vlmax);
                    __riscv_vse16_v_u16m2(ptr + i, float2bfloat_rvv(_p, vlmax), vlmax);
                }
                for (; i < size; i++)
                {
                    float v = bfloat16_to_float32(ptr[i]) * a + b;
                    ptr[i] = float32_to_bfloat16(v * gamma_ptr[i] + beta_ptr[i]);
                }
            }
            else
            {
                vfloat32m4_t _a = __riscv_vfmv_v_f_f32m4(a, vlmax4);
                vfloat32m4_t _b = __riscv_vfmv_v_f_f32m4(b, vlmax4);
                i = 0;
                for (; i < nfull; i += vlmax)
                {
                    vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + i, vlmax), vlmax);
                    _p = __riscv_vfmadd_vv_f32m4(_p, _a, _b, vlmax);
                    __riscv_vse16_v_u16m2(ptr + i, float2bfloat_rvv(_p, vlmax), vlmax);
                }
                for (; i < size; i++)
                {
                    float v = bfloat16_to_float32(ptr[i]) * a + b;
                    ptr[i] = float32_to_bfloat16(v);
                }
            }
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif

        return;
    }
#endif // __riscv_vector

    // scalar fallback, per packed lane
    for (int p = 0; p < elempack; p++)
    {
        float sum = 0.f;
        for (int i = 0; i < elemcount; i++)
        {
            sum += bfloat16_to_float32(ptr[i * elempack + p]);
        }

        float mean = sum / elemcount;

        float sqsum = 0.f;
        for (int i = 0; i < elemcount; i++)
        {
            float v = bfloat16_to_float32(ptr[i * elempack + p]) - mean;
            sqsum += v * v;
        }

        float var = sqsum / elemcount;

        float a = 1.f / sqrtf(var + eps);
        float b = -mean * a;

        if (gamma_ptr && beta_ptr)
        {
            for (int i = 0; i < elemcount; i++)
            {
                float v = bfloat16_to_float32(ptr[i * elempack + p]) * a + b;
                ptr[i * elempack + p] = float32_to_bfloat16(v * gamma_ptr[i] + beta_ptr[i]);
            }
        }
        else
        {
            for (int i = 0; i < elemcount; i++)
            {
                float v = bfloat16_to_float32(ptr[i * elempack + p]) * a + b;
                ptr[i * elempack + p] = float32_to_bfloat16(v);
            }
        }
    }

}
