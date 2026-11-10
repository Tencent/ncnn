// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void rotaryembed_bf16s_zvfbfmin(const Mat& bottom_blob, const Mat& cos_cache, const Mat& sin_cache, Mat& top_blob, int interleaved, const Option& opt);
#endif

static void rotaryembed_bf16s(const Mat& bottom_blob, const Mat& cos_cache, const Mat& sin_cache, Mat& top_blob, int interleaved, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        rotaryembed_bf16s_zvfbfmin(bottom_blob, cos_cache, sin_cache, top_blob, interleaved, opt);
        return;
    }
#endif

    const int embed_dim = bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int num_heads = bottom_blob.c;
    const int half = embed_dim / 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat head = bottom_blob.channel(q);
        Mat out_head = top_blob.channel(q);

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        // enable BF16 with round-to-nearest-even
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

        for (int i = 0; i < seqlen; i++)
        {
            const unsigned short* cos_ptr = cos_cache.row<const unsigned short>(i);
            const unsigned short* sin_ptr = sin_cache.row<const unsigned short>(i);

            if (interleaved)
            {
                const unsigned short* ptr = head.row<const unsigned short>(i);
                unsigned short* outptr = out_head.row<unsigned short>(i);

#if __riscv_vector
                int n = half;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vuint16m2x2_t _p = __riscv_vlseg2e16_v_u16m2x2(ptr, vl);
                    vfloat32m4_t _x0 = bfloat2float_rvv(__riscv_vget_v_u16m2x2_u16m2(_p, 0), vl);
                    vfloat32m4_t _x1 = bfloat2float_rvv(__riscv_vget_v_u16m2x2_u16m2(_p, 1), vl);
                    vfloat32m4_t _c = bfloat2float_rvv(__riscv_vle16_v_u16m2(cos_ptr, vl), vl);
                    vfloat32m4_t _s = bfloat2float_rvv(__riscv_vle16_v_u16m2(sin_ptr, vl), vl);

                    vfloat32m4_t _y0 = __riscv_vfnmsac_vv_f32m4(__riscv_vfmul_vv_f32m4(_x0, _c, vl), _x1, _s, vl); // x0*c - x1*s
                    vfloat32m4_t _y1 = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(_x1, _c, vl), _x0, _s, vl);  // x0*s + x1*c
                    __riscv_vsseg2e16_v_u16m2x2(outptr, __riscv_vcreate_v_u16m2x2(float2bfloat_rvv(_y0, vl), float2bfloat_rvv(_y1, vl)), vl);

                    ptr += vl * 2;
                    outptr += vl * 2;
                    cos_ptr += vl;
                    sin_ptr += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < half; j++)
                {
                    const float x0 = bfloat16_to_float32(ptr[0]);
                    const float x1 = bfloat16_to_float32(ptr[1]);
                    const float cos_val = bfloat16_to_float32(*cos_ptr++);
                    const float sin_val = bfloat16_to_float32(*sin_ptr++);
                    outptr[0] = float32_to_bfloat16(x0 * cos_val - x1 * sin_val);
                    outptr[1] = float32_to_bfloat16(x0 * sin_val + x1 * cos_val);
                    ptr += 2;
                    outptr += 2;
                }
#endif // __riscv_vector
            }
            else
            {
                // A full-width (embed_dim) cache carries independent cos/sin for the two halves
                // (2D / vision rope); a half-width (embed_dim/2) cache is the standard-rope form
                // whose halves are identical, so the second half reuses the first.
                const int cw = cos_cache.w == embed_dim ? half : 0;
                const unsigned short* ptr0 = head.row<const unsigned short>(i);
                const unsigned short* ptr1 = ptr0 + half;
                const unsigned short* cos_ptr1 = cos_ptr + cw;
                const unsigned short* sin_ptr1 = sin_ptr + cw;
                unsigned short* outptr0 = out_head.row<unsigned short>(i);
                unsigned short* outptr1 = outptr0 + half;

#if __riscv_vector
                int n = half;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vfloat32m4_t _x0 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr0, vl), vl);
                    vfloat32m4_t _x1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr1, vl), vl);
                    vfloat32m4_t _c0 = bfloat2float_rvv(__riscv_vle16_v_u16m2(cos_ptr, vl), vl);
                    vfloat32m4_t _s0 = bfloat2float_rvv(__riscv_vle16_v_u16m2(sin_ptr, vl), vl);
                    vfloat32m4_t _c1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(cos_ptr1, vl), vl);
                    vfloat32m4_t _s1 = bfloat2float_rvv(__riscv_vle16_v_u16m2(sin_ptr1, vl), vl);

                    vfloat32m4_t _y0 = __riscv_vfnmsac_vv_f32m4(__riscv_vfmul_vv_f32m4(_x0, _c0, vl), _x1, _s0, vl); // x0*c0 - x1*s0
                    vfloat32m4_t _y1 = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(_x1, _c1, vl), _x0, _s1, vl);  // x1*c1 + x0*s1
                    __riscv_vse16_v_u16m2(outptr0, float2bfloat_rvv(_y0, vl), vl);
                    __riscv_vse16_v_u16m2(outptr1, float2bfloat_rvv(_y1, vl), vl);

                    ptr0 += vl;
                    ptr1 += vl;
                    cos_ptr += vl;
                    sin_ptr += vl;
                    cos_ptr1 += vl;
                    sin_ptr1 += vl;
                    outptr0 += vl;
                    outptr1 += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < half; j++)
                {
                    const float x0 = bfloat16_to_float32(*ptr0++);
                    const float x1 = bfloat16_to_float32(*ptr1++);
                    *outptr0++ = float32_to_bfloat16(x0 * bfloat16_to_float32(*cos_ptr++) - x1 * bfloat16_to_float32(*sin_ptr++));
                    *outptr1++ = float32_to_bfloat16(x0 * bfloat16_to_float32(*sin_ptr1++) + x1 * bfloat16_to_float32(*cos_ptr1++));
                }
#endif // __riscv_vector
            }
        }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        // restore FXCR while retaining accrued exception flags, including FE
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
    }
}
