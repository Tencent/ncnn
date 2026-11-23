// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void relu_bf16s_zvfbfmin(Mat& bottom_top_blob, float slope, const Option& opt);
#endif

static void relu_bf16s(Mat& bottom_top_blob, float slope, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        relu_bf16s_zvfbfmin(bottom_top_blob, slope, opt);
        return;
    }
#endif

    const int size = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d * bottom_top_blob.elempack;
    const int channels = bottom_top_blob.c;

    // ncnn assumes normal values, so x < 0 is equivalent to bf16 bits > 0x8000
    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            // negative bf16 values are negative int16 values, relu is an integer max with zero
#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m8(n);
                vint16m8_t _p = __riscv_vle16_v_i16m8((const short*)ptr, vl);
                _p = __riscv_vmax_vx_i16m8(_p, 0, vl);
                __riscv_vse16_v_i16m8((short*)ptr, _p, vl);
                ptr += vl;
                n -= vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                if (ptr[i] > 0x8000)
                    ptr[i] = 0;
            }
#endif
        }

        return;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);
            vuint16m2_t _p = __riscv_vle16_v_u16m2(ptr, vl);
            vbool8_t _lower = __riscv_vmsgtu_vx_u16m2_b8(_p, 0x8000, vl);
            vfloat32m4_t _v = bfloat2float_rvv(_p, vl);
            _v = __riscv_vfmul_vf_f32m4_mu(_lower, _v, _v, slope, vl);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_v, vl), vl);
            ptr += vl;
            n -= vl;
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else
        for (int i = 0; i < size; i++)
        {
            if (ptr[i] > 0x8000)
                ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * slope);
        }
#endif
    }
}
