// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void cast_fp32_to_bf16_zvfbfmin(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
void cast_bf16_to_fp32_zvfbfmin(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
#endif

static void cast_fp32_to_bf16(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        cast_fp32_to_bf16_zvfbfmin(bottom_blob, top_blob, opt);
        return;
    }
#endif

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int batch = bottom_blob.n;
    const int size = w * h * d * elempack;

    const int total_bc = batch * channels;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int bc = 0; bc < total_bc; bc++)
    {
        int b = bc / channels;
        int q = bc % channels;
        const float* ptr = bottom_blob.batch(b).channel(q);
        unsigned short* outptr = top_blob.batch(b).channel(q);

#if __riscv_vector
        int n = size;
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        // enable BF16 with round-to-nearest-even
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t _p = __riscv_vle32_v_f32m4(ptr, vl);
            __riscv_vse16_v_u16m2(outptr, float2bfloat_rvv(_p, vl), vl);
            ptr += vl;
            outptr += vl;
            n -= vl;
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        // restore FXCR while retaining accrued exception flags, including FE
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            *outptr++ = float32_to_bfloat16(*ptr++);
        }
#endif // __riscv_vector
    }
}

static void cast_bf16_to_fp32(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        cast_bf16_to_fp32_zvfbfmin(bottom_blob, top_blob, opt);
        return;
    }
#endif

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int batch = bottom_blob.n;
    const int size = w * h * d * elempack;

    const int total_bc = batch * channels;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int bc = 0; bc < total_bc; bc++)
    {
        int b = bc / channels;
        int q = bc % channels;
        const unsigned short* ptr = bottom_blob.batch(b).channel(q);
        float* outptr = top_blob.batch(b).channel(q);

#if __riscv_vector
        int n = size;
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        // enable BF16 with round-to-nearest-even
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vuint16m2_t _p = __riscv_vle16_v_u16m2(ptr, vl);
            __riscv_vse32_v_f32m4(outptr, bfloat2float_rvv(_p, vl), vl);
            ptr += vl;
            outptr += vl;
            n -= vl;
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        // restore FXCR while retaining accrued exception flags, including FE
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            *outptr++ = bfloat16_to_float32(*ptr++);
        }
#endif // __riscv_vector
    }
}
