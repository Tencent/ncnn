// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_xtheadvector && !__riscv_zvfbfmin
void clip_bf16s_zvfbfmin(Mat& bottom_top_blob, float min, float max, const Option& opt);
#endif

static void clip_bf16s(Mat& bottom_top_blob, float min, float max, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_xtheadvector && !__riscv_zvfbfmin
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
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

#if NCNN_ZVFBFMIN && __riscv_zvfbfmin && !__riscv_xtheadvector
            vbfloat16m2_t _p = __riscv_vreinterpret_v_u16m2_bf16m2(__riscv_vle16_v_u16m2(ptr, vl));
            vfloat32m4_t _v = __riscv_vfwcvtbf16_f_f_v_f32m4(_p, vl);
#else
            vfloat32m4_t _v = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
#endif
            _v = __riscv_vfmax_vf_f32m4(_v, min, vl);
            _v = __riscv_vfmin_vf_f32m4(_v, max, vl);
#if NCNN_ZVFBFMIN && __riscv_zvfbfmin && !__riscv_xtheadvector
            _p = __riscv_vfncvtbf16_f_f_w_bf16m2_rm(_v, __RISCV_FRM_RNE, vl);
            __riscv_vse16_v_u16m2(ptr, __riscv_vreinterpret_v_bf16m2_u16m2(_p), vl);
#else
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_v, vl), vl);
#endif

            ptr += vl;
            n -= vl;
        }
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
