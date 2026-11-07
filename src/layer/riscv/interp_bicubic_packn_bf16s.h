// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if __riscv_vector
static void resize_bicubic_image_packn_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
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
    Mat rowsbuf2(w, (size_t)packn * 4u, packn);
    Mat rowsbuf3(w, (size_t)packn * 4u, packn);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    float* rows2 = rowsbuf2;
    float* rows3 = rowsbuf3;

    int prev_sy1 = -3;

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
            rows1 = rows2;
            rows2 = rows3;
            rows3 = rows0_old;
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const unsigned short* S3p = S3 + sx;

                vfloat32m4_t _S30 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p - packn, vl), vl);
                vfloat32m4_t _S31 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p, vl), vl);
                vfloat32m4_t _S32 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p + packn, vl), vl);
                vfloat32m4_t _S33 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p + packn * 2, vl), vl);
                vfloat32m4_t _rows3 = __riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                __riscv_vse32_v_f32m4(rows3p + dx * packn, _rows3, vl);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize two rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            rows0 = rows2;
            rows1 = rows3;
            rows2 = rows0_old;
            rows3 = rows1_old;
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                vfloat32m4_t _S20 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S2p - packn, vl), vl);
                vfloat32m4_t _S21 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S2p, vl), vl);
                vfloat32m4_t _S22 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S2p + packn, vl), vl);
                vfloat32m4_t _S23 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S2p + packn * 2, vl), vl);
                vfloat32m4_t _S30 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p - packn, vl), vl);
                vfloat32m4_t _S31 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p, vl), vl);
                vfloat32m4_t _S32 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p + packn, vl), vl);
                vfloat32m4_t _S33 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p + packn * 2, vl), vl);
                vfloat32m4_t _rows2 = __riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_S20, alphap[0], vl), alphap[1], _S21, vl), alphap[2], _S22, vl), alphap[3], _S23, vl);
                vfloat32m4_t _rows3 = __riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                __riscv_vse32_v_f32m4(rows2p + dx * packn, _rows2, vl);
                __riscv_vse32_v_f32m4(rows3p + dx * packn, _rows3, vl);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 3)
        {
            // hresize three rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            float* rows2_old = rows2;
            rows0 = rows3;
            rows1 = rows0_old;
            rows2 = rows1_old;
            rows3 = rows2_old;
            const unsigned short* S1 = src.row<const unsigned short>(sy);
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                vfloat32m4_t _S10 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S1p - packn, vl), vl);
                vfloat32m4_t _S11 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S1p, vl), vl);
                vfloat32m4_t _S12 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S1p + packn, vl), vl);
                vfloat32m4_t _S13 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S1p + packn * 2, vl), vl);
                vfloat32m4_t _S20 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S2p - packn, vl), vl);
                vfloat32m4_t _S21 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S2p, vl), vl);
                vfloat32m4_t _S22 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S2p + packn, vl), vl);
                vfloat32m4_t _S23 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S2p + packn * 2, vl), vl);
                vfloat32m4_t _S30 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p - packn, vl), vl);
                vfloat32m4_t _S31 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p, vl), vl);
                vfloat32m4_t _S32 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p + packn, vl), vl);
                vfloat32m4_t _S33 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p + packn * 2, vl), vl);
                vfloat32m4_t _rows1 = __riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_S10, alphap[0], vl), alphap[1], _S11, vl), alphap[2], _S12, vl), alphap[3], _S13, vl);
                vfloat32m4_t _rows2 = __riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_S20, alphap[0], vl), alphap[1], _S21, vl), alphap[2], _S22, vl), alphap[3], _S23, vl);
                vfloat32m4_t _rows3 = __riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                __riscv_vse32_v_f32m4(rows1p + dx * packn, _rows1, vl);
                __riscv_vse32_v_f32m4(rows2p + dx * packn, _rows2, vl);
                __riscv_vse32_v_f32m4(rows3p + dx * packn, _rows3, vl);

                alphap += 4;
            }
        }
        else
        {
            // hresize four rows
            const unsigned short* S0 = src.row<const unsigned short>(sy - 1);
            const unsigned short* S1 = src.row<const unsigned short>(sy);
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                vfloat32m4_t _S00 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S0p - packn, vl), vl);
                vfloat32m4_t _S01 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S0p, vl), vl);
                vfloat32m4_t _S02 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S0p + packn, vl), vl);
                vfloat32m4_t _S03 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S0p + packn * 2, vl), vl);
                vfloat32m4_t _S10 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S1p - packn, vl), vl);
                vfloat32m4_t _S11 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S1p, vl), vl);
                vfloat32m4_t _S12 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S1p + packn, vl), vl);
                vfloat32m4_t _S13 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S1p + packn * 2, vl), vl);
                vfloat32m4_t _S20 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S2p - packn, vl), vl);
                vfloat32m4_t _S21 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S2p, vl), vl);
                vfloat32m4_t _S22 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S2p + packn, vl), vl);
                vfloat32m4_t _S23 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S2p + packn * 2, vl), vl);
                vfloat32m4_t _S30 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p - packn, vl), vl);
                vfloat32m4_t _S31 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p, vl), vl);
                vfloat32m4_t _S32 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p + packn, vl), vl);
                vfloat32m4_t _S33 = bfloat2float_rvv(__riscv_vle16_v_u16m2(S3p + packn * 2, vl), vl);
                vfloat32m4_t _rows0 = __riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_S00, alphap[0], vl), alphap[1], _S01, vl), alphap[2], _S02, vl), alphap[3], _S03, vl);
                vfloat32m4_t _rows1 = __riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_S10, alphap[0], vl), alphap[1], _S11, vl), alphap[2], _S12, vl), alphap[3], _S13, vl);
                vfloat32m4_t _rows2 = __riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_S20, alphap[0], vl), alphap[1], _S21, vl), alphap[2], _S22, vl), alphap[3], _S23, vl);
                vfloat32m4_t _rows3 = __riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                __riscv_vse32_v_f32m4(rows0p + dx * packn, _rows0, vl);
                __riscv_vse32_v_f32m4(rows1p + dx * packn, _rows1, vl);
                __riscv_vse32_v_f32m4(rows2p + dx * packn, _rows2, vl);
                __riscv_vse32_v_f32m4(rows3p + dx * packn, _rows3, vl);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];
        float b2 = beta[2];
        float b3 = beta[3];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        unsigned short* Dp = dst.row<unsigned short>(dy);

        for (int dx = 0; dx < w; dx++)
        {
            vfloat32m4_t _rows0 = __riscv_vle32_v_f32m4(rows0p, vl);
            vfloat32m4_t _rows1 = __riscv_vle32_v_f32m4(rows1p, vl);
            vfloat32m4_t _rows2 = __riscv_vle32_v_f32m4(rows2p, vl);
            vfloat32m4_t _rows3 = __riscv_vle32_v_f32m4(rows3p, vl);

            vfloat32m4_t _Dp = __riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_rows0, b0, vl), b1, _rows1, vl), b2, _rows2, vl), b3, _rows3, vl);

            __riscv_vse16_v_u16m2(Dp, float2bfloat_rvv(_Dp, vl), vl);

            Dp += packn;
            rows0p += packn;
            rows1p += packn;
            rows2p += packn;
            rows3p += packn;
        }

        beta += 4;
    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
}
#endif // __riscv_vector
