// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if __riscv_vector
static void resize_bilinear_image_packn_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m2(packn);

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)packn * 4u, packn);
    Mat rowsbuf1(w, (size_t)packn * 4u, packn);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned short* S1 = src.row<const unsigned short>(sy + 1);

            const float* alphap = alpha;
            float* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const unsigned short* S1p = S1 + sx;

                vfloat32m4_t _S10 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S1p, vl), vl);
                vfloat32m4_t _S11 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S1p + packn, vl), vl);
                vfloat32m4_t _rows1 = __riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_S10, alphap[0], vl), alphap[1], _S11, vl);

                __riscv_vse32_v_f32m4(rows1p + dx * packn, _rows1, vl);

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const unsigned short* S0 = src.row<const unsigned short>(sy);
            const unsigned short* S1 = src.row<const unsigned short>(sy + 1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;

                vfloat32m4_t _S00 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S0p, vl), vl);
                vfloat32m4_t _S01 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S0p + packn, vl), vl);
                vfloat32m4_t _S10 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S1p, vl), vl);
                vfloat32m4_t _S11 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S1p + packn, vl), vl);
                vfloat32m4_t _rows0 = __riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_S00, alphap[0], vl), alphap[1], _S01, vl);
                vfloat32m4_t _rows1 = __riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_S10, alphap[0], vl), alphap[1], _S11, vl);

                __riscv_vse32_v_f32m4(rows0p + dx * packn, _rows0, vl);
                __riscv_vse32_v_f32m4(rows1p + dx * packn, _rows1, vl);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        unsigned short* Dp = dst.row<unsigned short>(dy);

        for (int dx = 0; dx < w; dx++)
        {
            vfloat32m4_t _rows0 = __riscv_vle32_v_f32m4(rows0p, vl);
            vfloat32m4_t _rows1 = __riscv_vle32_v_f32m4(rows1p, vl);

            vfloat32m4_t _Dp = __riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_rows0, b0, vl), b1, _rows1, vl);

            __riscv_vse16_v_u16m2(Dp, float2bfloat_rvv(_Dp, vl), vl);

            Dp += packn;
            rows0p += packn;
            rows1p += packn;
        }

        beta += 2;
    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
}
#endif // __riscv_vector
