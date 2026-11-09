// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void prelu_bf16s_zvfbfmin(Mat& bottom_top_blob, int num_slope, const Mat& slope_data, const Option& opt);
#endif

static void prelu_bf16s(Mat& bottom_top_blob, int num_slope, const Mat& slope_data, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        prelu_bf16s_zvfbfmin(bottom_top_blob, num_slope, slope_data, opt);
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
        const float* ptr_slope = slope_data;
        if (num_slope > 1 && dims != 1)
            ptr_slope += q * elempack;

        // ncnn assumes normal values, so x < 0 is equivalent to bf16 bits > 0x8000
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
        if (num_slope > 1 && dims == 1)
        {
            int n = size * elempack;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m2(n);
                vuint16m2_t _p = __riscv_vle16_v_u16m2(ptr, vl);
                vbool8_t _lower = __riscv_vmsgtu_vx_u16m2_b8(_p, 0x8000, vl);
                vfloat32m4_t _v = bfloat2float_rvv(_p, vl);
                vfloat32m4_t _slope = __riscv_vle32_v_f32m4(ptr_slope, vl);
                _v = __riscv_vfmul_vv_f32m4_mu(_lower, _v, _v, _slope, vl);
                __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_v, vl), vl);
                ptr += vl;
                ptr_slope += vl;
                n -= vl;
            }
        }
        else if (num_slope > 1 && elempack > 1)
        {
            const size_t vl = __riscv_vsetvl_e16m1(elempack);
            vfloat32m2_t _slope = __riscv_vle32_v_f32m2(ptr_slope, vl);
            for (int i = 0; i < size; i++)
            {
                vuint16m1_t _p = __riscv_vle16_v_u16m1(ptr, vl);
                vbool16_t _lower = __riscv_vmsgtu_vx_u16m1_b16(_p, 0x8000, vl);
                vfloat32m2_t _v = bfloat2float_rvv(_p, vl);
                _v = __riscv_vfmul_vv_f32m2_mu(_lower, _v, _v, _slope, vl);
                __riscv_vse16_v_u16m1(ptr, float2bfloat_rvv(_v, vl), vl);
                ptr += vl;
            }
        }
        else
        {
            const float slope = ptr_slope[0];
            int n = size * elempack;
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
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
        if (num_slope > 1 && dims == 1)
        {
            for (int i = 0; i < size; i++)
            {
                if (ptr[i] > 0x8000)
                    ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * ptr_slope[i]);
            }
        }
        else
        {
            const float slope = ptr_slope[0];
            for (int i = 0; i < size; i++)
            {
                if (ptr[i] > 0x8000)
                    ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * slope);
            }
        }
#endif // __riscv_vector
    }
}
