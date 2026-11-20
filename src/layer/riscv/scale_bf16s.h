// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void scale_bf16s_zvfbfmin(Mat& bottom_top_blob, const float* scale, const float* bias, bool bias_term, const Option& opt);
#endif

#if __riscv_vector
static void scale_bf16s_per_element(unsigned short* ptr, int size, const float* scale, const float* bias, bool bias_term)
{
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    // enable BF16 with round-to-nearest-even
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

    int n = size;
    if (bias_term)
    {
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            vfloat32m4_t _s = __riscv_vle32_v_f32m4(scale, vl);
            vfloat32m4_t _bias = __riscv_vle32_v_f32m4(bias, vl);
            _p = __riscv_vfmadd_vv_f32m4(_p, _s, _bias, vl);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);

            ptr += vl;
            scale += vl;
            bias += vl;
            n -= vl;
        }
    }
    else
    {
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            vfloat32m4_t _s = __riscv_vle32_v_f32m4(scale, vl);
            _p = __riscv_vfmul_vv_f32m4(_p, _s, vl);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);

            ptr += vl;
            scale += vl;
            n -= vl;
        }
    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    // restore FXCR while retaining accrued exception flags, including FE
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
}

static void scale_bf16s_pack1(unsigned short* ptr, int size, float s, float b, bool bias_term)
{
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    // enable BF16 with round-to-nearest-even
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

    int n = size;
    if (bias_term)
    {
        vfloat32m4_t _bias = __riscv_vfmv_v_f_f32m4(b, __riscv_vsetvlmax_e32m4());
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            _p = __riscv_vfmadd_vf_f32m4(_p, s, _bias, vl);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);

            ptr += vl;
            n -= vl;
        }
    }
    else
    {
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            _p = __riscv_vfmul_vf_f32m4(_p, s, vl);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);

            ptr += vl;
            n -= vl;
        }
    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    // restore FXCR while retaining accrued exception flags, including FE
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
}

// elempack values are interleaved at the beginning of every packed element
// the scale/bias array holds elempack values per element, so build one full
// f32m4 vector by duplicating the elempack wide f32m2 half
// vl is pinned to whole elements, as vsetvl may split a tail of 2*vlmax unevenly
// and shift the duplicated scale/bias pattern
static void scale_bf16s_packn(unsigned short* ptr, int size, int elempack, const float* scale, const float* bias, bool bias_term)
{
    const size_t vl = __riscv_vsetvl_e16m1(elempack);
    const size_t vl2 = vl * 2;

    vfloat32m2_t _s = __riscv_vle32_v_f32m2(scale, vl);
    vfloat32m4_t _s2 = __riscv_vcreate_v_f32m2_f32m4(_s, _s);

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    // enable BF16 with round-to-nearest-even
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

    int i = 0;
    if (bias_term)
    {
        vfloat32m2_t _bias = __riscv_vle32_v_f32m2(bias, vl);
        vfloat32m4_t _bias2 = __riscv_vcreate_v_f32m2_f32m4(_bias, _bias);

        for (; i + 1 < size; i += 2)
        {
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl2), vl2);
            _p = __riscv_vfmadd_vv_f32m4(_p, _s2, _bias2, vl2);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl2), vl2);
            ptr += vl2;
        }
        for (; i < size; i++)
        {
            vfloat32m2_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m1(ptr, vl), vl);
            _p = __riscv_vfmadd_vv_f32m2(_p, _s, _bias, vl);
            __riscv_vse16_v_u16m1(ptr, float2bfloat_rvv(_p, vl), vl);
            ptr += vl;
        }
    }
    else
    {
        for (; i + 1 < size; i += 2)
        {
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl2), vl2);
            _p = __riscv_vfmul_vv_f32m4(_p, _s2, vl2);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl2), vl2);
            ptr += vl2;
        }
        for (; i < size; i++)
        {
            vfloat32m2_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m1(ptr, vl), vl);
            _p = __riscv_vfmul_vv_f32m2(_p, _s, vl);
            __riscv_vse16_v_u16m1(ptr, float2bfloat_rvv(_p, vl), vl);
            ptr += vl;
        }
    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    // restore FXCR while retaining accrued exception flags, including FE
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
}
#endif // __riscv_vector

static void scale_bf16s(Mat& bottom_top_blob, const float* scale, const float* bias, bool bias_term, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        scale_bf16s_zvfbfmin(bottom_top_blob, scale, bias, bias_term, opt);
        return;
    }
#endif

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        unsigned short* ptr = bottom_top_blob;
        const int size = bottom_top_blob.w * elempack;

#if __riscv_vector
        scale_bf16s_per_element(ptr, size, scale, bias, bias_term);
#else // __riscv_vector
        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i] + bias[i]);
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i]);
            }
        }
#endif // __riscv_vector

        return;
    }

    if (elempack == 1)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int d = bottom_top_blob.d;
        const int channels = bottom_top_blob.c;

        if (dims == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
                const float s = scale[i];
                const float b = bias_term ? bias[i] : 0.f;

#if __riscv_vector
                scale_bf16s_pack1(ptr, w, s, b, bias_term);
#else  // __riscv_vector
                if (bias_term)
                {
                    for (int j = 0; j < w; j++)
                    {
                        ptr[j] = float32_to_bfloat16(bfloat16_to_float32(ptr[j]) * s + b);
                    }
                }
                else
                {
                    for (int j = 0; j < w; j++)
                    {
                        ptr[j] = float32_to_bfloat16(bfloat16_to_float32(ptr[j]) * s);
                    }
                }
#endif // __riscv_vector
            }
        }

        if (dims == 3 || dims == 4)
        {
            const int size = w * h * d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = bottom_top_blob.channel(q);
                const float s = scale[q];
                const float b = bias_term ? bias[q] : 0.f;

#if __riscv_vector
                scale_bf16s_pack1(ptr, size, s, b, bias_term);
#else  // __riscv_vector
                if (bias_term)
                {
                    for (int i = 0; i < size; i++)
                    {
                        ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * s + b);
                    }
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * s);
                    }
                }
#endif // __riscv_vector
            }
        }

        return;
    }

#if __riscv_vector
    // elempack == packn
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            const float* sptr = scale + i * elempack;
            const float* bptr = bias_term ? bias + i * elempack : 0;

            scale_bf16s_packn(ptr, w, elempack, sptr, bptr, bias_term);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            const float* sptr = scale + q * elempack;
            const float* bptr = bias_term ? bias + q * elempack : 0;

            scale_bf16s_packn(ptr, size, elempack, sptr, bptr, bias_term);
        }
    }
#endif // __riscv_vector
}
