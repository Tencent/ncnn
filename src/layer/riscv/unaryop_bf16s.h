// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
int unaryop_bf16s_zvfbfmin(Mat& bottom_top_blob, int op_type, const Option& opt);
#endif

template<typename Op>
static int unary_op_inplace_bf16s(Mat& a, const Option& opt)
{
    Op op;

    const int w = a.w;
    const int h = a.h;
    const int d = a.d;
    const int channels = a.c;
    const int size = w * h * d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = a.channel(q);

#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        // enable BF16 with round-to-nearest-even
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);

            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);
            _p = op(_p, vl);
            __riscv_vse16_v_u16m4(ptr, float2bfloat_rvv(_p, vl), vl);

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
            ptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[i])));
        }
#endif // __riscv_vector
    }

    return 0;
}
