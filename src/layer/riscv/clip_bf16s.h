// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void clip_bf16s_zvfbfmin(Mat& bottom_top_blob, float min, float max, const Option& opt);
#endif

static void clip_bf16s(Mat& bottom_top_blob, float min, float max, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        clip_bf16s_zvfbfmin(bottom_top_blob, min, max, opt);
        return;
    }
#endif

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        // enable BF16 with round-to-nearest-even
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat32m4_t _v = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            _v = __riscv_vfmax_vf_f32m4(_v, min, vl);
            _v = __riscv_vfmin_vf_f32m4(_v, max, vl);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_v, vl), vl);

            ptr += vl;
            n -= vl;
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        // restore FXCR while retaining accrued exception flags, including FE
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < min)
                v = min;

            if (v > max)
                v = max;

            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
#endif // __riscv_vector
    }
}
