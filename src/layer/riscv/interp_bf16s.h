// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void interp_bf16s_zvfbfmin(const Mat& bottom_blob, Mat& top_blob, int resize_type, int align_corner, float height_scale, float width_scale, int output_height, int output_width, int has_output_size, const Option& opt);
#endif

static void vresize_bilinear_bf16s(const float* rows0, const float* rows1, unsigned short* Dp, int n, float b0, float b1)
{
#if __riscv_vector
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m4(n);

        vfloat32m8_t _rows0 = __riscv_vle32_v_f32m8(rows0, vl);
        vfloat32m8_t _rows1 = __riscv_vle32_v_f32m8(rows1, vl);

        vfloat32m8_t _Dp = __riscv_vfmacc_vf_f32m8(__riscv_vfmul_vf_f32m8(_rows0, b0, vl), b1, _rows1, vl);

        __riscv_vse16_v_u16m4(Dp, float2bfloat_rvv(_Dp, vl), vl);

        Dp += vl;
        rows0 += vl;
        rows1 += vl;
        n -= vl;
    }
#else  // __riscv_vector
    for (int i = 0; i < n; i++)
    {
        *Dp++ = float32_to_bfloat16(*rows0++ * b0 + *rows1++ * b1);
    }
#endif // __riscv_vector
}

static void vresize_bicubic_bf16s(const float* rows0, const float* rows1, const float* rows2, const float* rows3, unsigned short* Dp, int n, float b0, float b1, float b2, float b3)
{
#if __riscv_vector
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m2(n);

        vfloat32m4_t _rows0 = __riscv_vle32_v_f32m4(rows0, vl);
        vfloat32m4_t _rows1 = __riscv_vle32_v_f32m4(rows1, vl);
        vfloat32m4_t _rows2 = __riscv_vle32_v_f32m4(rows2, vl);
        vfloat32m4_t _rows3 = __riscv_vle32_v_f32m4(rows3, vl);

        vfloat32m4_t _Dp = __riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmul_vf_f32m4(_rows0, b0, vl), b1, _rows1, vl), b2, _rows2, vl), b3, _rows3, vl);

        __riscv_vse16_v_u16m2(Dp, float2bfloat_rvv(_Dp, vl), vl);

        Dp += vl;
        rows0 += vl;
        rows1 += vl;
        rows2 += vl;
        rows3 += vl;
        n -= vl;
    }
#else  // __riscv_vector
    for (int i = 0; i < n; i++)
    {
        *Dp++ = float32_to_bfloat16(*rows0++ * b0 + *rows1++ * b1 + *rows2++ * b2 + *rows3++ * b3);
    }
#endif // __riscv_vector
}

#if __riscv_vector
static void resize_bilinear_image_packn_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    // enable BF16 with round-to-nearest-even
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

                vfloat32m2_t _S10 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S1p, vl), vl);
                vfloat32m2_t _S11 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S1p + packn, vl), vl);
                vfloat32m2_t _rows1 = __riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S10, alphap[0], vl), alphap[1], _S11, vl);

                __riscv_vse32_v_f32m2(rows1p + dx * packn, _rows1, vl);

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

                vfloat32m2_t _S00 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S0p, vl), vl);
                vfloat32m2_t _S01 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S0p + packn, vl), vl);
                vfloat32m2_t _S10 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S1p, vl), vl);
                vfloat32m2_t _S11 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S1p + packn, vl), vl);
                vfloat32m2_t _rows0 = __riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S00, alphap[0], vl), alphap[1], _S01, vl);
                vfloat32m2_t _rows1 = __riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S10, alphap[0], vl), alphap[1], _S11, vl);

                __riscv_vse32_v_f32m2(rows0p + dx * packn, _rows0, vl);
                __riscv_vse32_v_f32m2(rows1p + dx * packn, _rows1, vl);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bilinear_bf16s(rows0, rows1, dst.row<unsigned short>(dy), w * packn, beta[0], beta[1]);

        beta += 2;
    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    // restore FXCR while retaining accrued exception flags, including FE
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
}

static void resize_bicubic_image_packn_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    // enable BF16 with round-to-nearest-even
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

                vfloat32m2_t _S30 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p - packn, vl), vl);
                vfloat32m2_t _S31 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p, vl), vl);
                vfloat32m2_t _S32 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p + packn, vl), vl);
                vfloat32m2_t _S33 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p + packn * 2, vl), vl);
                vfloat32m2_t _rows3 = __riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S30, alphap[0], vl), alphap[1], _S31, vl), alphap[2], _S32, vl), alphap[3], _S33, vl);

                __riscv_vse32_v_f32m2(rows3p + dx * packn, _rows3, vl);

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

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];

                vfloat32m2_t _S20 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S2p - packn, vl), vl);
                vfloat32m2_t _S21 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S2p, vl), vl);
                vfloat32m2_t _S22 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S2p + packn, vl), vl);
                vfloat32m2_t _S23 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S2p + packn * 2, vl), vl);
                vfloat32m2_t _rows2 = __riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S20, a0, vl), a1, _S21, vl), a2, _S22, vl), a3, _S23, vl);
                __riscv_vse32_v_f32m2(rows2p + dx * packn, _rows2, vl);

                vfloat32m2_t _S30 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p - packn, vl), vl);
                vfloat32m2_t _S31 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p, vl), vl);
                vfloat32m2_t _S32 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p + packn, vl), vl);
                vfloat32m2_t _S33 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p + packn * 2, vl), vl);
                vfloat32m2_t _rows3 = __riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S30, a0, vl), a1, _S31, vl), a2, _S32, vl), a3, _S33, vl);
                __riscv_vse32_v_f32m2(rows3p + dx * packn, _rows3, vl);

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

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];

                vfloat32m2_t _S10 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S1p - packn, vl), vl);
                vfloat32m2_t _S11 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S1p, vl), vl);
                vfloat32m2_t _S12 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S1p + packn, vl), vl);
                vfloat32m2_t _S13 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S1p + packn * 2, vl), vl);
                vfloat32m2_t _rows1 = __riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S10, a0, vl), a1, _S11, vl), a2, _S12, vl), a3, _S13, vl);
                __riscv_vse32_v_f32m2(rows1p + dx * packn, _rows1, vl);

                vfloat32m2_t _S20 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S2p - packn, vl), vl);
                vfloat32m2_t _S21 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S2p, vl), vl);
                vfloat32m2_t _S22 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S2p + packn, vl), vl);
                vfloat32m2_t _S23 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S2p + packn * 2, vl), vl);
                vfloat32m2_t _rows2 = __riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S20, a0, vl), a1, _S21, vl), a2, _S22, vl), a3, _S23, vl);
                __riscv_vse32_v_f32m2(rows2p + dx * packn, _rows2, vl);

                vfloat32m2_t _S30 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p - packn, vl), vl);
                vfloat32m2_t _S31 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p, vl), vl);
                vfloat32m2_t _S32 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p + packn, vl), vl);
                vfloat32m2_t _S33 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p + packn * 2, vl), vl);
                vfloat32m2_t _rows3 = __riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S30, a0, vl), a1, _S31, vl), a2, _S32, vl), a3, _S33, vl);
                __riscv_vse32_v_f32m2(rows3p + dx * packn, _rows3, vl);

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

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];

                vfloat32m2_t _S00 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S0p - packn, vl), vl);
                vfloat32m2_t _S01 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S0p, vl), vl);
                vfloat32m2_t _S02 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S0p + packn, vl), vl);
                vfloat32m2_t _S03 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S0p + packn * 2, vl), vl);
                vfloat32m2_t _rows0 = __riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S00, a0, vl), a1, _S01, vl), a2, _S02, vl), a3, _S03, vl);
                __riscv_vse32_v_f32m2(rows0p + dx * packn, _rows0, vl);

                vfloat32m2_t _S10 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S1p - packn, vl), vl);
                vfloat32m2_t _S11 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S1p, vl), vl);
                vfloat32m2_t _S12 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S1p + packn, vl), vl);
                vfloat32m2_t _S13 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S1p + packn * 2, vl), vl);
                vfloat32m2_t _rows1 = __riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S10, a0, vl), a1, _S11, vl), a2, _S12, vl), a3, _S13, vl);
                __riscv_vse32_v_f32m2(rows1p + dx * packn, _rows1, vl);

                vfloat32m2_t _S20 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S2p - packn, vl), vl);
                vfloat32m2_t _S21 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S2p, vl), vl);
                vfloat32m2_t _S22 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S2p + packn, vl), vl);
                vfloat32m2_t _S23 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S2p + packn * 2, vl), vl);
                vfloat32m2_t _rows2 = __riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S20, a0, vl), a1, _S21, vl), a2, _S22, vl), a3, _S23, vl);
                __riscv_vse32_v_f32m2(rows2p + dx * packn, _rows2, vl);

                vfloat32m2_t _S30 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p - packn, vl), vl);
                vfloat32m2_t _S31 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p, vl), vl);
                vfloat32m2_t _S32 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p + packn, vl), vl);
                vfloat32m2_t _S33 = bfloat2float_rvv(__riscv_vle16_v_u16m1(S3p + packn * 2, vl), vl);
                vfloat32m2_t _rows3 = __riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S30, a0, vl), a1, _S31, vl), a2, _S32, vl), a3, _S33, vl);
                __riscv_vse32_v_f32m2(rows3p + dx * packn, _rows3, vl);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bicubic_bf16s(rows0, rows1, rows2, rows3, dst.row<unsigned short>(dy), w * packn, beta[0], beta[1], beta[2], beta[3]);

        beta += 4;
    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    // restore FXCR while retaining accrued exception flags, including FE
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
}
#endif // __riscv_vector

static void resize_bilinear_image_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    // enable BF16 with round-to-nearest-even
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w);
    Mat rowsbuf1(w);
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
                int sx = xofs[dx];
                const unsigned short* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = bfloat16_to_float32(S1p[0]) * a0 + bfloat16_to_float32(S1p[1]) * a1;

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
                int sx = xofs[dx];
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = bfloat16_to_float32(S0p[0]) * a0 + bfloat16_to_float32(S0p[1]) * a1;
                rows1p[dx] = bfloat16_to_float32(S1p[0]) * a0 + bfloat16_to_float32(S1p[1]) * a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bilinear_bf16s(rows0, rows1, dst.row<unsigned short>(dy), w, beta[0], beta[1]);

        beta += 2;
    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    // restore FXCR while retaining accrued exception flags, including FE
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
}

static void resize_bicubic_image_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    // enable BF16 with round-to-nearest-even
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w);
    Mat rowsbuf1(w);
    Mat rowsbuf2(w);
    Mat rowsbuf3(w);
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
                int sx = xofs[dx];
                const unsigned short* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows3p[dx] = bfloat16_to_float32(S3p[-1]) * a0 + bfloat16_to_float32(S3p[0]) * a1 + bfloat16_to_float32(S3p[1]) * a2 + bfloat16_to_float32(S3p[2]) * a3;

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
                int sx = xofs[dx];
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows2p[dx] = bfloat16_to_float32(S2p[-1]) * a0 + bfloat16_to_float32(S2p[0]) * a1 + bfloat16_to_float32(S2p[1]) * a2 + bfloat16_to_float32(S2p[2]) * a3;
                rows3p[dx] = bfloat16_to_float32(S3p[-1]) * a0 + bfloat16_to_float32(S3p[0]) * a1 + bfloat16_to_float32(S3p[1]) * a2 + bfloat16_to_float32(S3p[2]) * a3;

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
                int sx = xofs[dx];
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows1p[dx] = bfloat16_to_float32(S1p[-1]) * a0 + bfloat16_to_float32(S1p[0]) * a1 + bfloat16_to_float32(S1p[1]) * a2 + bfloat16_to_float32(S1p[2]) * a3;
                rows2p[dx] = bfloat16_to_float32(S2p[-1]) * a0 + bfloat16_to_float32(S2p[0]) * a1 + bfloat16_to_float32(S2p[1]) * a2 + bfloat16_to_float32(S2p[2]) * a3;
                rows3p[dx] = bfloat16_to_float32(S3p[-1]) * a0 + bfloat16_to_float32(S3p[0]) * a1 + bfloat16_to_float32(S3p[1]) * a2 + bfloat16_to_float32(S3p[2]) * a3;

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
                int sx = xofs[dx];
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows0p[dx] = bfloat16_to_float32(S0p[-1]) * a0 + bfloat16_to_float32(S0p[0]) * a1 + bfloat16_to_float32(S0p[1]) * a2 + bfloat16_to_float32(S0p[2]) * a3;
                rows1p[dx] = bfloat16_to_float32(S1p[-1]) * a0 + bfloat16_to_float32(S1p[0]) * a1 + bfloat16_to_float32(S1p[1]) * a2 + bfloat16_to_float32(S1p[2]) * a3;
                rows2p[dx] = bfloat16_to_float32(S2p[-1]) * a0 + bfloat16_to_float32(S2p[0]) * a1 + bfloat16_to_float32(S2p[1]) * a2 + bfloat16_to_float32(S2p[2]) * a3;
                rows3p[dx] = bfloat16_to_float32(S3p[-1]) * a0 + bfloat16_to_float32(S3p[0]) * a1 + bfloat16_to_float32(S3p[1]) * a2 + bfloat16_to_float32(S3p[2]) * a3;

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bicubic_bf16s(rows0, rows1, rows2, rows3, dst.row<unsigned short>(dy), w, beta[0], beta[1], beta[2], beta[3]);

        beta += 4;
    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    // restore FXCR while retaining accrued exception flags, including FE
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
}

static void interp_bf16s(const Mat& bottom_blob, Mat& top_blob, int resize_type, int align_corner, float height_scale, float width_scale, int output_height, int output_width, int has_output_size, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        interp_bf16s_zvfbfmin(bottom_blob, top_blob, resize_type, align_corner, height_scale, width_scale, output_height, output_width, has_output_size, opt);
        return;
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
#endif

    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    (void)elempack;

    if (dims == 1)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            const size_t vl = __riscv_vsetvl_e16m1(packn);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                vuint16m1_t _v = __riscv_vle16_v_u16m1((const unsigned short*)bottom_blob + q * packn, vl);
                top_blob_c.fill(_v);
            }

            return;
        }
#endif // __riscv_vector

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < w; q++)
        {
            Mat top_blob_c = top_blob.channel(q);
            const unsigned short* ptr = bottom_blob;
            top_blob_c.fill(ptr[q]);
        }

        return;
    }

    if (dims == 2)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            if (resize_type == 1) // nearest
            {
                const size_t vl = __riscv_vsetvl_e16m1(packn);

                const float ws = (output_width || has_output_size) ? w / (float)outw : 1.f / width_scale;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                    unsigned short* outptr = top_blob.row<unsigned short>(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        vuint16m1_t _p = __riscv_vle16_v_u16m1(ptr + in_x * packn, vl);
                        __riscv_vse16_v_u16m1(outptr, _p, vl);

                        outptr += packn;
                    }
                }
            }

            if (resize_type == 2) // bilinear
            {
                const size_t vl = __riscv_vsetvl_e16m1(packn);

                int* buf = new int[outw + outw * 2];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                const double ws = (output_width || has_output_size) ? (double)w / outw : 1.0 / width_scale;
                linear_coeffs(w, outw, ws, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                    unsigned long old_fxcr = csrr_fxcr();
                    // enable BF16 with round-to-nearest-even
                    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

                    const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                    unsigned short* outptr = top_blob.row<unsigned short>(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * packn;
                        const unsigned short* Sp = ptr + sx;

                        vfloat32m2_t _S0 = bfloat2float_rvv(__riscv_vle16_v_u16m1(Sp, vl), vl);
                        vfloat32m2_t _S1 = bfloat2float_rvv(__riscv_vle16_v_u16m1(Sp + packn, vl), vl);
                        vfloat32m2_t _p = __riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S0, alphap[0], vl), alphap[1], _S1, vl);

                        __riscv_vse16_v_u16m1(outptr, float2bfloat_rvv(_p, vl), vl);

                        alphap += 2;
                        outptr += packn;
                    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                    // restore FXCR while retaining accrued exception flags, including FE
                    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
                }

                delete[] buf;
            }

            if (resize_type == 3) // bicubic
            {
                const size_t vl = __riscv_vsetvl_e16m1(packn);

                int* buf = new int[outw + outw * 4];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                const double ws = (output_width || has_output_size) ? (double)w / outw : 1.0 / width_scale;
                cubic_coeffs(w, outw, ws, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                    unsigned long old_fxcr = csrr_fxcr();
                    // enable BF16 with round-to-nearest-even
                    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

                    const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                    unsigned short* outptr = top_blob.row<unsigned short>(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * packn;
                        const unsigned short* Sp = ptr + sx;

                        vfloat32m2_t _S0 = bfloat2float_rvv(__riscv_vle16_v_u16m1(Sp - packn, vl), vl);
                        vfloat32m2_t _S1 = bfloat2float_rvv(__riscv_vle16_v_u16m1(Sp, vl), vl);
                        vfloat32m2_t _S2 = bfloat2float_rvv(__riscv_vle16_v_u16m1(Sp + packn, vl), vl);
                        vfloat32m2_t _S3 = bfloat2float_rvv(__riscv_vle16_v_u16m1(Sp + packn * 2, vl), vl);
                        vfloat32m2_t _p = __riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(_S0, alphap[0], vl), alphap[1], _S1, vl), alphap[2], _S2, vl), alphap[3], _S3, vl);

                        __riscv_vse16_v_u16m1(outptr, float2bfloat_rvv(_p, vl), vl);

                        alphap += 4;
                        outptr += packn;
                    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                    // restore FXCR while retaining accrued exception flags, including FE
                    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
                }

                delete[] buf;
            }

            return;
        }
#endif // __riscv_vector

        if (resize_type == 1) // nearest
        {
            const float ws = (output_width || has_output_size) ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                unsigned short* outptr = top_blob.row<unsigned short>(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            const double ws = (output_width || has_output_size) ? (double)w / outw : 1.0 / width_scale;
            linear_coeffs(w, outw, ws, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                unsigned short* outptr = top_blob.row<unsigned short>(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const unsigned short* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    *outptr++ = float32_to_bfloat16(bfloat16_to_float32(Sp[0]) * a0 + bfloat16_to_float32(Sp[1]) * a1);
                    alphap += 2;
                }
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outw * 4];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            const double ws = (output_width || has_output_size) ? (double)w / outw : 1.0 / width_scale;
            cubic_coeffs(w, outw, ws, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                unsigned short* outptr = top_blob.row<unsigned short>(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const unsigned short* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    float a2 = alphap[2];
                    float a3 = alphap[3];
                    *outptr++ = float32_to_bfloat16(bfloat16_to_float32(Sp[-1]) * a0 + bfloat16_to_float32(Sp[0]) * a1 + bfloat16_to_float32(Sp[1]) * a2 + bfloat16_to_float32(Sp[2]) * a3);
                    alphap += 4;
                }
            }

            delete[] buf;
        }

        return;
    }

#if __riscv_vector
    if (elempack == packn)
    {
        if (resize_type == 1) // nearest
        {
            const size_t vl = __riscv_vsetvl_e16m1(packn);

            const float hs = (output_height || has_output_size) ? h / (float)outh : 1.f / height_scale;
            const float ws = (output_width || has_output_size) ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int in_y = std::min((int)(y * hs), (h - 1));

                    const unsigned short* ptr = src.row<const unsigned short>(in_y);
                    unsigned short* outptr = dst.row<unsigned short>(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        vuint16m1_t _p = __riscv_vle16_v_u16m1(ptr + in_x * packn, vl);
                        __riscv_vse16_v_u16m1(outptr, _p, vl);

                        outptr += packn;
                    }
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
            float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

            const double ws = (output_width || has_output_size) ? (double)w / outw : 1.0 / width_scale;
            linear_coeffs(w, outw, ws, xofs, alpha, align_corner);
            const double hs = (output_height || has_output_size) ? (double)h / outh : 1.0 / height_scale;
            linear_coeffs(h, outh, hs, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bilinear_image_packn_bf16s(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outh + outw * 4 + outh * 4];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
            float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

            const double ws = (output_width || has_output_size) ? (double)w / outw : 1.0 / width_scale;
            cubic_coeffs(w, outw, ws, xofs, alpha, align_corner);
            const double hs = (output_height || has_output_size) ? (double)h / outh : 1.0 / height_scale;
            cubic_coeffs(h, outh, hs, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bicubic_image_packn_bf16s(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return;
    }
#endif // __riscv_vector

    if (resize_type == 1) // nearest
    {
        const float hs = (output_height || has_output_size) ? h / (float)outh : 1.f / height_scale;
        const float ws = (output_width || has_output_size) ? w / (float)outw : 1.f / width_scale;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            for (int y = 0; y < outh; y++)
            {
                int in_y = std::min((int)(y * hs), (h - 1));

                const unsigned short* ptr = src.row<const unsigned short>(in_y);
                unsigned short* outptr = dst.row<unsigned short>(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }
    }

    if (resize_type == 2) // bilinear
    {
        int* buf = new int[outw + outh + outw * 2 + outh * 2];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
        float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

        const double ws = (output_width || has_output_size) ? (double)w / outw : 1.0 / width_scale;
        linear_coeffs(w, outw, ws, xofs, alpha, align_corner);
        const double hs = (output_height || has_output_size) ? (double)h / outh : 1.0 / height_scale;
        linear_coeffs(h, outh, hs, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image_bf16s(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    if (resize_type == 3) // bicubic
    {
        int* buf = new int[outw + outh + outw * 4 + outh * 4];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
        float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

        const double ws = (output_width || has_output_size) ? (double)w / outw : 1.0 / width_scale;
        cubic_coeffs(w, outw, ws, xofs, alpha, align_corner);
        const double hs = (output_height || has_output_size) ? (double)h / outh : 1.0 / height_scale;
        cubic_coeffs(h, outh, hs, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bicubic_image_bf16s(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }
}
