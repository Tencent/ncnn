// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void instancenorm_bf16s_zvfbfmin(unsigned short* ptr, int size, float a, float b);
void instancenorm_bf16s_compute_mean_var_zvfbfmin(const unsigned short* ptr, int size, float& mean, float& var);
void instancenorm_bf16s_packn_zvfbfmin(unsigned short* ptr, int size, int elempack, const float* gamma, const float* beta, float eps, int affine);
#endif

// xtheadvector needs FXCR.BF16 and FRM=RNE for bf16 narrowing, mirror the tanh template
#if __riscv_vector && __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
#define INSTANCENORM_BF16S_FXCR_SAVE                                 \
    unsigned long old_fxcr = csrr_fxcr();                            \
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#define INSTANCENORM_BF16S_FXCR_RESTORE \
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#else
#define INSTANCENORM_BF16S_FXCR_SAVE
#define INSTANCENORM_BF16S_FXCR_RESTORE
#endif

static void instancenorm_bf16s(unsigned short* ptr, int size, float a, float b)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        instancenorm_bf16s_zvfbfmin(ptr, size, a, b);
        return;
    }
#endif

#if __riscv_vector
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m2(n);
        vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
        _p = __riscv_vfmul_vf_f32m4(_p, a, vl);
        _p = __riscv_vfadd_vf_f32m4(_p, b, vl);
        __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);
        n -= vl;
        ptr += vl;
    }
#else
    for (int i = 0; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * a + b);
    }
#endif
}

static void instancenorm_bf16s_compute_mean_var(const unsigned short* ptr, int size, float& mean, float& var)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        instancenorm_bf16s_compute_mean_var_zvfbfmin(ptr, size, mean, var);
        return;
    }
#endif

    float sum = 0.f;
    float sqsum = 0.f;

#if __riscv_vector && !defined(C906)
    {
        vfloat32m1_t _sum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());

        const unsigned short* ptr_sum = ptr;
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr_sum, vl), vl);
            _sum = __riscv_vfredusum_vs_f32m4_f32m1(_p, _sum, vl);
            ptr_sum += vl;
            n -= vl;
        }
        sum = __riscv_vfmv_f_s_f32m1_f32(_sum);
    }
#else
    for (int i = 0; i < size; i++)
    {
        sum += bfloat16_to_float32(ptr[i]);
    }
#endif // __riscv_vector

    mean = sum / size;

#if __riscv_vector && !defined(C906)
    {
        vfloat32m1_t _sqsum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());

        const unsigned short* ptr_sqsum = ptr;
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr_sqsum, vl), vl);
            _p = __riscv_vfsub_vf_f32m4(_p, mean, vl);
            _sqsum = __riscv_vfredosum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(_p, _p, vl), _sqsum, vl);
            ptr_sqsum += vl;
            n -= vl;
        }
        sqsum = __riscv_vfmv_f_s_f32m1_f32(_sqsum);
    }
#else
    for (int i = 0; i < size; i++)
    {
        float tmp = bfloat16_to_float32(ptr[i]) - mean;
        sqsum += tmp * tmp;
    }
#endif // __riscv_vector

    var = sqsum / size;
    // the var maybe minus due to accuracy
    //float var = sqsum / size - mean * mean;
}

// whole channel of a packed bf16 blob, one mean/var/a/b per pack lane
static void instancenorm_bf16s_packn(unsigned short* ptr, int size, int elempack, const float* gamma, const float* beta, float eps, int affine)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        instancenorm_bf16s_packn_zvfbfmin(ptr, size, elempack, gamma, beta, eps, affine);
        return;
    }
#endif

#if __riscv_vector && !defined(C906)
    const size_t vl = __riscv_vsetvl_e16m2(elempack);

    INSTANCENORM_BF16S_FXCR_SAVE

    // mean
    vfloat32m4_t _sum = __riscv_vfmv_v_f_f32m4(0.f, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + (size_t)i * elempack, vl), vl);
        _sum = __riscv_vfadd_vv_f32m4(_p, _sum, vl);
    }
    vfloat32m4_t _mean = __riscv_vfdiv_vf_f32m4(_sum, size, vl);

    // var
    vfloat32m4_t _sqsum = __riscv_vfmv_v_f_f32m4(0.f, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + (size_t)i * elempack, vl), vl);
        _p = __riscv_vfsub_vv_f32m4(_p, _mean, vl);
        _sqsum = __riscv_vfmadd_vv_f32m4(_p, _p, _sqsum, vl);
    }
    vfloat32m4_t _var = __riscv_vfdiv_vf_f32m4(_sqsum, size, vl);

    vfloat32m4_t _a;
    vfloat32m4_t _b;
    if (affine)
    {
        vfloat32m4_t _gamma = __riscv_vle32_v_f32m4(gamma, vl);
        vfloat32m4_t _beta = __riscv_vle32_v_f32m4(beta, vl);
        _a = __riscv_vfdiv_vv_f32m4(_gamma, __riscv_vfsqrt_v_f32m4(__riscv_vfadd_vf_f32m4(_var, eps, vl), vl), vl);
        _b = __riscv_vfnmsub_vv_f32m4(_a, _mean, _beta, vl);
    }
    else
    {
        _a = __riscv_vfrdiv_vf_f32m4(__riscv_vfsqrt_v_f32m4(__riscv_vfadd_vf_f32m4(_var, eps, vl), vl), 1.f, vl);
        _b = __riscv_vfmul_vv_f32m4(_a, _mean, vl);
        _b = __riscv_vfsgnjn_vv_f32m4(_b, _b, vl);
    }

    for (int i = 0; i < size; i++)
    {
        vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + (size_t)i * elempack, vl), vl);
        _p = __riscv_vfmadd_vv_f32m4(_p, _a, _b, vl);
        __riscv_vse16_v_u16m2(ptr + (size_t)i * elempack, float2bfloat_rvv(_p, vl), vl);
    }

    INSTANCENORM_BF16S_FXCR_RESTORE
#else
    const int lane_count = elempack;

    for (int lane = 0; lane < lane_count; lane++)
    {
        float sum = 0.f;
        for (int i = 0; i < size; i++)
        {
            sum += bfloat16_to_float32(ptr[(size_t)i * elempack + lane]);
        }
        float mean = sum / size;

        float sqsum = 0.f;
        for (int i = 0; i < size; i++)
        {
            float tmp = bfloat16_to_float32(ptr[(size_t)i * elempack + lane]) - mean;
            sqsum += tmp * tmp;
        }
        float var = sqsum / size;

        float a;
        float b;
        if (affine)
        {
            a = gamma[lane] / (sqrtf(var + eps));
            b = -mean * a + beta[lane];
        }
        else
        {
            a = 1.f / (sqrtf(var + eps));
            b = -mean * a;
        }

        for (int i = 0; i < size; i++)
        {
            unsigned short* p = ptr + (size_t)i * elempack + lane;
            *p = float32_to_bfloat16(bfloat16_to_float32(*p) * a + b);
        }
    }
#endif // __riscv_vector
}
