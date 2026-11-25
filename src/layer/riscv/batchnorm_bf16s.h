// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void batchnorm_bf16s_zvfbfmin(Mat& bottom_top_blob, const Mat& a_data, const Mat& b_data, const Option& opt);
#endif

static void batchnorm_bf16s(Mat& bottom_top_blob, const Mat& a_data, const Mat& b_data, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        batchnorm_bf16s_zvfbfmin(bottom_top_blob, a_data, b_data, opt);
        return;
    }
#endif

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int channels = dims == 2 ? bottom_top_blob.h : bottom_top_blob.c;
    const int size = dims <= 2 ? bottom_top_blob.w : bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = dims == 2 ? bottom_top_blob.row<unsigned short>(q) : bottom_top_blob.channel(q);
        const float* ptr_a = a_data;
        const float* ptr_b = b_data;
        if (dims != 1)
        {
            ptr_a += q * elempack;
            ptr_b += q * elempack;
        }

#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        if (dims == 1)
        {
            int n = size * elempack;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m2(n);
                vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
                vfloat32m4_t _a = __riscv_vle32_v_f32m4(ptr_a, vl);
                vfloat32m4_t _b = __riscv_vle32_v_f32m4(ptr_b, vl);
                _p = __riscv_vfmadd_vv_f32m4(_p, _b, _a, vl);
                __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);
                ptr += vl;
                ptr_a += vl;
                ptr_b += vl;
                n -= vl;
            }
        }
        else if (elempack == 1)
        {
            const float b = ptr_b[0];
            vfloat32m4_t _a = __riscv_vfmv_v_f_f32m4(ptr_a[0], __riscv_vsetvlmax_e32m4());
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m2(n);
                vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
                _p = __riscv_vfmadd_vf_f32m4(_p, b, _a, vl);
                __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);
                ptr += vl;
                n -= vl;
            }
        }
        else
        {
            const size_t vl = __riscv_vsetvl_e16m1(elempack);
            vfloat32m2_t _a = __riscv_vle32_v_f32m2(ptr_a, vl);
            vfloat32m2_t _b = __riscv_vle32_v_f32m2(ptr_b, vl);
            for (int i = 0; i < size; i++)
            {
                vfloat32m2_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m1(ptr, vl), vl);
                _p = __riscv_vfmadd_vv_f32m2(_p, _b, _a, vl);
                __riscv_vse16_v_u16m1(ptr, float2bfloat_rvv(_p, vl), vl);
                ptr += vl;
            }
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
        if (dims == 1)
        {
            for (int i = 0; i < size; i++)
            {
                ptr[i] = float32_to_bfloat16(ptr_b[i] * bfloat16_to_float32(ptr[i]) + ptr_a[i]);
            }
        }
        else
        {
            const float a = ptr_a[0];
            const float b = ptr_b[0];
            for (int i = 0; i < size; i++)
            {
                ptr[i] = float32_to_bfloat16(b * bfloat16_to_float32(ptr[i]) + a);
            }
        }
#endif // __riscv_vector
    }
}
