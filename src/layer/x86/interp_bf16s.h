// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void interp_forward_bf16s_sse_avx512bf16(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt, int resize_type, int align_corner, float height_scale, float width_scale, int output_height, int output_width, int has_size_expr);
#endif

static void interp_forward_bf16s_sse(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt, int resize_type, int align_corner, float height_scale, float width_scale, int output_height, int output_width, int has_size_expr);

static void vresize_bilinear_bf16s(const float* rows0, const float* rows1, unsigned short* Dp, int n, float b0, float b1)
{
    int nn = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _b0_512 = _mm512_set1_ps(b0);
    __m512 _b1_512 = _mm512_set1_ps(b1);
    for (; nn + 15 < n; nn += 16)
    {
        __m512 _rows0 = _mm512_loadu_ps(rows0 + nn);
        __m512 _rows1 = _mm512_loadu_ps(rows1 + nn);
        __m512 _Dp = _mm512_mul_ps(_rows0, _b0_512);
        _Dp = _mm512_fmadd_ps(_rows1, _b1_512, _Dp);
        _mm256_storeu_si256((__m256i*)(Dp + nn), float2bfloat_avx512(_Dp));
    }
#endif // __AVX512F__
    __m256 _b0_256 = _mm256_set1_ps(b0);
    __m256 _b1_256 = _mm256_set1_ps(b1);
    for (; nn + 7 < n; nn += 8)
    {
        __m256 _rows0 = _mm256_loadu_ps(rows0 + nn);
        __m256 _rows1 = _mm256_loadu_ps(rows1 + nn);
        __m256 _Dp = _mm256_mul_ps(_rows0, _b0_256);
        _Dp = _mm256_comp_fmadd_ps(_rows1, _b1_256, _Dp);
        _mm_storeu_si128((__m128i*)(Dp + nn), float2bfloat_avx(_Dp));
    }
#endif // __AVX__
    __m128 _b0_128 = _mm_set1_ps(b0);
    __m128 _b1_128 = _mm_set1_ps(b1);
    for (; nn + 3 < n; nn += 4)
    {
        __m128 _rows0 = _mm_loadu_ps(rows0 + nn);
        __m128 _rows1 = _mm_loadu_ps(rows1 + nn);
        __m128 _Dp = _mm_mul_ps(_rows0, _b0_128);
        _Dp = _mm_comp_fmadd_ps(_rows1, _b1_128, _Dp);
        _mm_storel_epi64((__m128i*)(Dp + nn), float2bfloat_sse(_Dp, _Dp));
    }
#endif // __SSE2__
    for (; nn < n; nn++)
    {
        Dp[nn] = float32_to_bfloat16(rows0[nn] * b0 + rows1[nn] * b1);
    }
}

static void vresize_bicubic_bf16s(const float* rows0, const float* rows1, const float* rows2, const float* rows3, unsigned short* Dp, int n, float b0, float b1, float b2, float b3)
{
    int nn = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _b0_512 = _mm512_set1_ps(b0);
    __m512 _b1_512 = _mm512_set1_ps(b1);
    __m512 _b2_512 = _mm512_set1_ps(b2);
    __m512 _b3_512 = _mm512_set1_ps(b3);
    for (; nn + 15 < n; nn += 16)
    {
        __m512 _rows0 = _mm512_loadu_ps(rows0 + nn);
        __m512 _rows1 = _mm512_loadu_ps(rows1 + nn);
        __m512 _rows2 = _mm512_loadu_ps(rows2 + nn);
        __m512 _rows3 = _mm512_loadu_ps(rows3 + nn);
        __m512 _Dp = _mm512_mul_ps(_rows0, _b0_512);
        _Dp = _mm512_fmadd_ps(_rows1, _b1_512, _Dp);
        _Dp = _mm512_fmadd_ps(_rows2, _b2_512, _Dp);
        _Dp = _mm512_fmadd_ps(_rows3, _b3_512, _Dp);
        _mm256_storeu_si256((__m256i*)(Dp + nn), float2bfloat_avx512(_Dp));
    }
#endif // __AVX512F__
    __m256 _b0_256 = _mm256_set1_ps(b0);
    __m256 _b1_256 = _mm256_set1_ps(b1);
    __m256 _b2_256 = _mm256_set1_ps(b2);
    __m256 _b3_256 = _mm256_set1_ps(b3);
    for (; nn + 7 < n; nn += 8)
    {
        __m256 _rows0 = _mm256_loadu_ps(rows0 + nn);
        __m256 _rows1 = _mm256_loadu_ps(rows1 + nn);
        __m256 _rows2 = _mm256_loadu_ps(rows2 + nn);
        __m256 _rows3 = _mm256_loadu_ps(rows3 + nn);
        __m256 _Dp = _mm256_mul_ps(_rows0, _b0_256);
        _Dp = _mm256_comp_fmadd_ps(_rows1, _b1_256, _Dp);
        _Dp = _mm256_comp_fmadd_ps(_rows2, _b2_256, _Dp);
        _Dp = _mm256_comp_fmadd_ps(_rows3, _b3_256, _Dp);
        _mm_storeu_si128((__m128i*)(Dp + nn), float2bfloat_avx(_Dp));
    }
#endif // __AVX__
    __m128 _b0_128 = _mm_set1_ps(b0);
    __m128 _b1_128 = _mm_set1_ps(b1);
    __m128 _b2_128 = _mm_set1_ps(b2);
    __m128 _b3_128 = _mm_set1_ps(b3);
    for (; nn + 3 < n; nn += 4)
    {
        __m128 _rows0 = _mm_loadu_ps(rows0 + nn);
        __m128 _rows1 = _mm_loadu_ps(rows1 + nn);
        __m128 _rows2 = _mm_loadu_ps(rows2 + nn);
        __m128 _rows3 = _mm_loadu_ps(rows3 + nn);
        __m128 _Dp = _mm_mul_ps(_rows0, _b0_128);
        _Dp = _mm_comp_fmadd_ps(_rows1, _b1_128, _Dp);
        _Dp = _mm_comp_fmadd_ps(_rows2, _b2_128, _Dp);
        _Dp = _mm_comp_fmadd_ps(_rows3, _b3_128, _Dp);
        _mm_storel_epi64((__m128i*)(Dp + nn), float2bfloat_sse(_Dp, _Dp));
    }
#endif // __SSE2__
    for (; nn < n; nn++)
    {
        Dp[nn] = float32_to_bfloat16(rows0[nn] * b0 + rows1[nn] * b1 + rows2[nn] * b2 + rows3[nn] * b3);
    }
}

#if __SSE2__

static void resize_bilinear_image_pack4_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)4 * 4u, 4);
    Mat rowsbuf1(w, (size_t)4 * 4u, 4);
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
            int dx = 0;
#if __AVX__
#if __AVX512F__
            for (; dx + 3 < w; dx += 4)
            {
                int sx0 = xofs[dx] * 4;
                int sx1 = xofs[dx + 1] * 4;
                int sx2 = xofs[dx + 2] * 4;
                int sx3 = xofs[dx + 3] * 4;

                __m512 _a0 = _mm512_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[2], alphap[2], alphap[2], alphap[2], alphap[4], alphap[4], alphap[4], alphap[4], alphap[6], alphap[6], alphap[6], alphap[6]);
                __m512 _a1 = _mm512_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[3], alphap[3], alphap[3], alphap[3], alphap[5], alphap[5], alphap[5], alphap[5], alphap[7], alphap[7], alphap[7], alphap[7]);

                __m128 _S10_0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx0)));
                __m128 _S10_1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx1)));
                __m128 _S10_2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx2)));
                __m128 _S10_3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx3)));
                __m128 _S11_0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx0 + 4)));
                __m128 _S11_1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx1 + 4)));
                __m128 _S11_2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx2 + 4)));
                __m128 _S11_3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx3 + 4)));

                __m512 _S10 = combine4x4_ps(_S10_0, _S10_1, _S10_2, _S10_3);
                __m512 _S11 = combine4x4_ps(_S11_0, _S11_1, _S11_2, _S11_3);

                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _mm512_storeu_ps(rows1p + dx * 4, _rows1);

                alphap += 8;
            }
#endif // __AVX512F__
            for (; dx + 1 < w; dx += 2)
            {
                int sx0 = xofs[dx] * 4;
                int sx1 = xofs[dx + 1] * 4;

                __m256 _a0 = _mm256_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[2], alphap[2], alphap[2], alphap[2]);
                __m256 _a1 = _mm256_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[3], alphap[3], alphap[3], alphap[3]);

                __m256 _S10 = combine4x2_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx0))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx1))));
                __m256 _S11 = combine4x2_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx0 + 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx1 + 4))));

                __m256 _rows1 = _mm256_mul_ps(_S10, _a0);
                _rows1 = _mm256_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm256_storeu_ps(rows1p + dx * 4, _rows1);

                alphap += 4;
            }
#endif // __AVX__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S1p = S1 + sx;

                __m128 _a0 = _mm_set1_ps(alphap[0]);
                __m128 _a1 = _mm_set1_ps(alphap[1]);

                __m128 _S10 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)S1p));
                __m128 _S11 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1p + 4)));
                __m128 _rows1 = _mm_mul_ps(_S10, _a0);
                _rows1 = _mm_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm_store_ps(rows1p + dx * 4, _rows1);

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
            int dx = 0;
#if __AVX__
#if __AVX512F__
            for (; dx + 3 < w; dx += 4)
            {
                int sx0 = xofs[dx] * 4;
                int sx1 = xofs[dx + 1] * 4;
                int sx2 = xofs[dx + 2] * 4;
                int sx3 = xofs[dx + 3] * 4;

                __m512 _a0 = _mm512_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[2], alphap[2], alphap[2], alphap[2], alphap[4], alphap[4], alphap[4], alphap[4], alphap[6], alphap[6], alphap[6], alphap[6]);
                __m512 _a1 = _mm512_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[3], alphap[3], alphap[3], alphap[3], alphap[5], alphap[5], alphap[5], alphap[5], alphap[7], alphap[7], alphap[7], alphap[7]);

                __m128 _S00_0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S0 + sx0)));
                __m128 _S00_1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S0 + sx1)));
                __m128 _S00_2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S0 + sx2)));
                __m128 _S00_3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S0 + sx3)));
                __m128 _S01_0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S0 + sx0 + 4)));
                __m128 _S01_1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S0 + sx1 + 4)));
                __m128 _S01_2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S0 + sx2 + 4)));
                __m128 _S01_3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S0 + sx3 + 4)));

                __m512 _S00 = combine4x4_ps(_S00_0, _S00_1, _S00_2, _S00_3);
                __m512 _S01 = combine4x4_ps(_S01_0, _S01_1, _S01_2, _S01_3);

                __m512 _rows0 = _mm512_mul_ps(_S00, _a0);
                _rows0 = _mm512_fmadd_ps(_S01, _a1, _rows0);
                _mm512_storeu_ps(rows0p + dx * 4, _rows0);

                __m128 _S10_0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx0)));
                __m128 _S10_1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx1)));
                __m128 _S10_2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx2)));
                __m128 _S10_3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx3)));
                __m128 _S11_0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx0 + 4)));
                __m128 _S11_1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx1 + 4)));
                __m128 _S11_2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx2 + 4)));
                __m128 _S11_3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx3 + 4)));

                __m512 _S10 = combine4x4_ps(_S10_0, _S10_1, _S10_2, _S10_3);
                __m512 _S11 = combine4x4_ps(_S11_0, _S11_1, _S11_2, _S11_3);

                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _mm512_storeu_ps(rows1p + dx * 4, _rows1);

                alphap += 8;
            }
#endif // __AVX512F__
            for (; dx + 1 < w; dx += 2)
            {
                int sx0 = xofs[dx] * 4;
                int sx1 = xofs[dx + 1] * 4;

                __m256 _a0 = _mm256_setr_ps(alphap[0], alphap[0], alphap[0], alphap[0], alphap[2], alphap[2], alphap[2], alphap[2]);
                __m256 _a1 = _mm256_setr_ps(alphap[1], alphap[1], alphap[1], alphap[1], alphap[3], alphap[3], alphap[3], alphap[3]);

                __m256 _S00 = combine4x2_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S0 + sx0))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S0 + sx1))));
                __m256 _S01 = combine4x2_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S0 + sx0 + 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S0 + sx1 + 4))));
                __m256 _S10 = combine4x2_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx0))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx1))));
                __m256 _S11 = combine4x2_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx0 + 4))), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1 + sx1 + 4))));

                __m256 _rows0 = _mm256_mul_ps(_S00, _a0);
                __m256 _rows1 = _mm256_mul_ps(_S10, _a0);
                _rows0 = _mm256_comp_fmadd_ps(_S01, _a1, _rows0);
                _rows1 = _mm256_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm256_storeu_ps(rows0p + dx * 4, _rows0);
                _mm256_storeu_ps(rows1p + dx * 4, _rows1);

                alphap += 4;
            }
#endif // __AVX__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;

                __m128 _a0 = _mm_set1_ps(alphap[0]);
                __m128 _a1 = _mm_set1_ps(alphap[1]);

                __m128 _S00 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)S0p));
                __m128 _S01 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S0p + 4)));
                __m128 _S10 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)S1p));
                __m128 _S11 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S1p + 4)));
                __m128 _rows0 = _mm_mul_ps(_S00, _a0);
                __m128 _rows1 = _mm_mul_ps(_S10, _a0);
                _rows0 = _mm_comp_fmadd_ps(_S01, _a1, _rows0);
                _rows1 = _mm_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm_store_ps(rows0p + dx * 4, _rows0);
                _mm_store_ps(rows1p + dx * 4, _rows1);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bilinear_bf16s(rows0, rows1, dst.row<unsigned short>(dy), w * 4, beta[0], beta[1]);

        beta += 2;
    }
}

#if __AVX__

static void resize_bilinear_image_pack8_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)8 * 4u, 8);
    Mat rowsbuf1(w, (size_t)8 * 4u, 8);
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
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const unsigned short* S1p = S1 + sx;

                __m256 _a0 = _mm256_set1_ps(alphap[0]);
                __m256 _a1 = _mm256_set1_ps(alphap[1]);

                __m256 _S10 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)S1p));
                __m256 _S11 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S1p + 8)));
                __m256 _rows1 = _mm256_mul_ps(_S10, _a0);
                _rows1 = _mm256_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm256_store_ps(rows1p + dx * 8, _rows1);

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
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;

                __m256 _a0 = _mm256_set1_ps(alphap[0]);
                __m256 _a1 = _mm256_set1_ps(alphap[1]);

                __m256 _S00 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)S0p));
                __m256 _S01 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S0p + 8)));
                __m256 _S10 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)S1p));
                __m256 _S11 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S1p + 8)));
                __m256 _rows0 = _mm256_mul_ps(_S00, _a0);
                __m256 _rows1 = _mm256_mul_ps(_S10, _a0);
                _rows0 = _mm256_comp_fmadd_ps(_S01, _a1, _rows0);
                _rows1 = _mm256_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm256_store_ps(rows0p + dx * 8, _rows0);
                _mm256_store_ps(rows1p + dx * 8, _rows1);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bilinear_bf16s(rows0, rows1, dst.row<unsigned short>(dy), w * 8, beta[0], beta[1]);

        beta += 2;
    }
}

#if __AVX512F__

static void resize_bilinear_image_pack16_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)16 * 4u, 16);
    Mat rowsbuf1(w, (size_t)16 * 4u, 16);
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
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 16;
                const unsigned short* S1p = S1 + sx;

                __m512 _a0 = _mm512_set1_ps(alphap[0]);
                __m512 _a1 = _mm512_set1_ps(alphap[1]);

                __m512 _S10 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)S1p));
                __m512 _S11 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S1p + 16)));
                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _mm512_store_ps(rows1p + dx * 16, _rows1);

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
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 16;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;

                __m512 _a0 = _mm512_set1_ps(alphap[0]);
                __m512 _a1 = _mm512_set1_ps(alphap[1]);

                __m512 _S00 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)S0p));
                __m512 _S01 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S0p + 16)));
                __m512 _S10 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)S1p));
                __m512 _S11 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S1p + 16)));
                __m512 _rows0 = _mm512_mul_ps(_S00, _a0);
                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows0 = _mm512_fmadd_ps(_S01, _a1, _rows0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _mm512_store_ps(rows0p + dx * 16, _rows0);
                _mm512_store_ps(rows1p + dx * 16, _rows1);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bilinear_bf16s(rows0, rows1, dst.row<unsigned short>(dy), w * 16, beta[0], beta[1]);

        beta += 2;
    }
}

#endif // __AVX512F__
#endif // __AVX__

static void resize_bicubic_image_pack4_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)4 * 4u, 4);
    Mat rowsbuf1(w, (size_t)4 * 4u, 4);
    Mat rowsbuf2(w, (size_t)4 * 4u, 4);
    Mat rowsbuf3(w, (size_t)4 * 4u, 4);
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
                int sx = xofs[dx] * 4;
                const unsigned short* S3p = S3 + sx;

                __m128 _a0 = _mm_set1_ps(alphap[0]);
                __m128 _a1 = _mm_set1_ps(alphap[1]);
                __m128 _a2 = _mm_set1_ps(alphap[2]);
                __m128 _a3 = _mm_set1_ps(alphap[3]);

                __m128 _S30 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S3p - 4)));
                __m128 _S31 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S3p + 0)));
                __m128 _S32 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S3p + 4)));
                __m128 _S33 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S3p + 8)));
                __m128 _rows3 = _mm_mul_ps(_S30, _a0);
                _rows3 = _mm_comp_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm_comp_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm_comp_fmadd_ps(_S33, _a3, _rows3);
                _mm_store_ps(rows3p + dx * 4, _rows3);

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
                int sx = xofs[dx] * 4;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m128 _a0 = _mm_set1_ps(alphap[0]);
                __m128 _a1 = _mm_set1_ps(alphap[1]);
                __m128 _a2 = _mm_set1_ps(alphap[2]);
                __m128 _a3 = _mm_set1_ps(alphap[3]);

                __m128 _S20 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S2p - 4)));
                __m128 _S21 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S2p + 0)));
                __m128 _S22 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S2p + 4)));
                __m128 _S23 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S2p + 8)));
                __m128 _S30 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S3p - 4)));
                __m128 _S31 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S3p + 0)));
                __m128 _S32 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S3p + 4)));
                __m128 _S33 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(S3p + 8)));
                __m128 _rows2 = _mm_mul_ps(_S20, _a0);
                __m128 _rows3 = _mm_mul_ps(_S30, _a0);
                _rows2 = _mm_comp_fmadd_ps(_S21, _a1, _rows2);
                _rows3 = _mm_comp_fmadd_ps(_S31, _a1, _rows3);
                _rows2 = _mm_comp_fmadd_ps(_S22, _a2, _rows2);
                _rows3 = _mm_comp_fmadd_ps(_S32, _a2, _rows3);
                _rows2 = _mm_comp_fmadd_ps(_S23, _a3, _rows2);
                _rows3 = _mm_comp_fmadd_ps(_S33, _a3, _rows3);
                _mm_store_ps(rows2p + dx * 4, _rows2);
                _mm_store_ps(rows3p + dx * 4, _rows3);

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
                int sx = xofs[dx] * 4;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m128 _a0 = _mm_set1_ps(alphap[0]);
                __m128 _a1 = _mm_set1_ps(alphap[1]);
                __m128 _a2 = _mm_set1_ps(alphap[2]);
                __m128 _a3 = _mm_set1_ps(alphap[3]);

                for (int r = 0; r < 3; r++)
                {
                    const unsigned short* Snp = (r == 0) ? S1p : ((r == 1) ? S2p : S3p);
                    float* rowsnp = (r == 0) ? rows1p : ((r == 1) ? rows2p : rows3p);

                    __m128 _Sn0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Snp - 4)));
                    __m128 _Sn1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Snp + 0)));
                    __m128 _Sn2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Snp + 4)));
                    __m128 _Sn3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Snp + 8)));
                    __m128 _rowsn = _mm_mul_ps(_Sn0, _a0);
                    _rowsn = _mm_comp_fmadd_ps(_Sn1, _a1, _rowsn);
                    _rowsn = _mm_comp_fmadd_ps(_Sn2, _a2, _rowsn);
                    _rowsn = _mm_comp_fmadd_ps(_Sn3, _a3, _rowsn);
                    _mm_store_ps(rowsnp + dx * 4, _rowsn);
                }

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
                int sx = xofs[dx] * 4;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m128 _a0 = _mm_set1_ps(alphap[0]);
                __m128 _a1 = _mm_set1_ps(alphap[1]);
                __m128 _a2 = _mm_set1_ps(alphap[2]);
                __m128 _a3 = _mm_set1_ps(alphap[3]);

                for (int r = 0; r < 4; r++)
                {
                    const unsigned short* Snp = (r == 0) ? S0p : ((r == 1) ? S1p : ((r == 2) ? S2p : S3p));
                    float* rowsnp = (r == 0) ? rows0p : ((r == 1) ? rows1p : ((r == 2) ? rows2p : rows3p));

                    __m128 _Sn0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Snp - 4)));
                    __m128 _Sn1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Snp + 0)));
                    __m128 _Sn2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Snp + 4)));
                    __m128 _Sn3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Snp + 8)));
                    __m128 _rowsn = _mm_mul_ps(_Sn0, _a0);
                    _rowsn = _mm_comp_fmadd_ps(_Sn1, _a1, _rowsn);
                    _rowsn = _mm_comp_fmadd_ps(_Sn2, _a2, _rowsn);
                    _rowsn = _mm_comp_fmadd_ps(_Sn3, _a3, _rowsn);
                    _mm_store_ps(rowsnp + dx * 4, _rowsn);
                }

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bicubic_bf16s(rows0, rows1, rows2, rows3, dst.row<unsigned short>(dy), w * 4, beta[0], beta[1], beta[2], beta[3]);

        beta += 4;
    }
}

#if __AVX__

static void resize_bicubic_image_pack8_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)8 * 4u, 8);
    Mat rowsbuf1(w, (size_t)8 * 4u, 8);
    Mat rowsbuf2(w, (size_t)8 * 4u, 8);
    Mat rowsbuf3(w, (size_t)8 * 4u, 8);
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
                int sx = xofs[dx] * 8;
                const unsigned short* S3p = S3 + sx;

                __m256 _a0 = _mm256_set1_ps(alphap[0]);
                __m256 _a1 = _mm256_set1_ps(alphap[1]);
                __m256 _a2 = _mm256_set1_ps(alphap[2]);
                __m256 _a3 = _mm256_set1_ps(alphap[3]);

                __m256 _S30 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S3p - 8)));
                __m256 _S31 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S3p + 0)));
                __m256 _S32 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S3p + 8)));
                __m256 _S33 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S3p + 16)));
                __m256 _rows3 = _mm256_mul_ps(_S30, _a0);
                _rows3 = _mm256_comp_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm256_comp_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm256_comp_fmadd_ps(_S33, _a3, _rows3);
                _mm256_store_ps(rows3p + dx * 8, _rows3);

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
                int sx = xofs[dx] * 8;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m256 _a0 = _mm256_set1_ps(alphap[0]);
                __m256 _a1 = _mm256_set1_ps(alphap[1]);
                __m256 _a2 = _mm256_set1_ps(alphap[2]);
                __m256 _a3 = _mm256_set1_ps(alphap[3]);

                __m256 _S20 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S2p - 8)));
                __m256 _S21 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S2p + 0)));
                __m256 _S22 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S2p + 8)));
                __m256 _S23 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S2p + 16)));
                __m256 _S30 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S3p - 8)));
                __m256 _S31 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S3p + 0)));
                __m256 _S32 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S3p + 8)));
                __m256 _S33 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(S3p + 16)));
                __m256 _rows2 = _mm256_mul_ps(_S20, _a0);
                __m256 _rows3 = _mm256_mul_ps(_S30, _a0);
                _rows2 = _mm256_comp_fmadd_ps(_S21, _a1, _rows2);
                _rows3 = _mm256_comp_fmadd_ps(_S31, _a1, _rows3);
                _rows2 = _mm256_comp_fmadd_ps(_S22, _a2, _rows2);
                _rows3 = _mm256_comp_fmadd_ps(_S32, _a2, _rows3);
                _rows2 = _mm256_comp_fmadd_ps(_S23, _a3, _rows2);
                _rows3 = _mm256_comp_fmadd_ps(_S33, _a3, _rows3);
                _mm256_store_ps(rows2p + dx * 8, _rows2);
                _mm256_store_ps(rows3p + dx * 8, _rows3);

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
                int sx = xofs[dx] * 8;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m256 _a0 = _mm256_set1_ps(alphap[0]);
                __m256 _a1 = _mm256_set1_ps(alphap[1]);
                __m256 _a2 = _mm256_set1_ps(alphap[2]);
                __m256 _a3 = _mm256_set1_ps(alphap[3]);

                for (int r = 0; r < 3; r++)
                {
                    const unsigned short* Snp = (r == 0) ? S1p : ((r == 1) ? S2p : S3p);
                    float* rowsnp = (r == 0) ? rows1p : ((r == 1) ? rows2p : rows3p);

                    __m256 _Sn0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Snp - 8)));
                    __m256 _Sn1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Snp + 0)));
                    __m256 _Sn2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Snp + 8)));
                    __m256 _Sn3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Snp + 16)));
                    __m256 _rowsn = _mm256_mul_ps(_Sn0, _a0);
                    _rowsn = _mm256_comp_fmadd_ps(_Sn1, _a1, _rowsn);
                    _rowsn = _mm256_comp_fmadd_ps(_Sn2, _a2, _rowsn);
                    _rowsn = _mm256_comp_fmadd_ps(_Sn3, _a3, _rowsn);
                    _mm256_store_ps(rowsnp + dx * 8, _rowsn);
                }

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
                int sx = xofs[dx] * 8;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m256 _a0 = _mm256_set1_ps(alphap[0]);
                __m256 _a1 = _mm256_set1_ps(alphap[1]);
                __m256 _a2 = _mm256_set1_ps(alphap[2]);
                __m256 _a3 = _mm256_set1_ps(alphap[3]);

                for (int r = 0; r < 4; r++)
                {
                    const unsigned short* Snp = (r == 0) ? S0p : ((r == 1) ? S1p : ((r == 2) ? S2p : S3p));
                    float* rowsnp = (r == 0) ? rows0p : ((r == 1) ? rows1p : ((r == 2) ? rows2p : rows3p));

                    __m256 _Sn0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Snp - 8)));
                    __m256 _Sn1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Snp + 0)));
                    __m256 _Sn2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Snp + 8)));
                    __m256 _Sn3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Snp + 16)));
                    __m256 _rowsn = _mm256_mul_ps(_Sn0, _a0);
                    _rowsn = _mm256_comp_fmadd_ps(_Sn1, _a1, _rowsn);
                    _rowsn = _mm256_comp_fmadd_ps(_Sn2, _a2, _rowsn);
                    _rowsn = _mm256_comp_fmadd_ps(_Sn3, _a3, _rowsn);
                    _mm256_store_ps(rowsnp + dx * 8, _rowsn);
                }

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bicubic_bf16s(rows0, rows1, rows2, rows3, dst.row<unsigned short>(dy), w * 8, beta[0], beta[1], beta[2], beta[3]);

        beta += 4;
    }
}

#if __AVX512F__

static void resize_bicubic_image_pack16_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)16 * 4u, 16);
    Mat rowsbuf1(w, (size_t)16 * 4u, 16);
    Mat rowsbuf2(w, (size_t)16 * 4u, 16);
    Mat rowsbuf3(w, (size_t)16 * 4u, 16);
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
                int sx = xofs[dx] * 16;
                const unsigned short* S3p = S3 + sx;

                __m512 _a0 = _mm512_set1_ps(alphap[0]);
                __m512 _a1 = _mm512_set1_ps(alphap[1]);
                __m512 _a2 = _mm512_set1_ps(alphap[2]);
                __m512 _a3 = _mm512_set1_ps(alphap[3]);

                __m512 _S30 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S3p - 16)));
                __m512 _S31 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S3p + 0)));
                __m512 _S32 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S3p + 16)));
                __m512 _S33 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S3p + 32)));
                __m512 _rows3 = _mm512_mul_ps(_S30, _a0);
                _rows3 = _mm512_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm512_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm512_fmadd_ps(_S33, _a3, _rows3);
                _mm512_store_ps(rows3p + dx * 16, _rows3);

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
                int sx = xofs[dx] * 16;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m512 _a0 = _mm512_set1_ps(alphap[0]);
                __m512 _a1 = _mm512_set1_ps(alphap[1]);
                __m512 _a2 = _mm512_set1_ps(alphap[2]);
                __m512 _a3 = _mm512_set1_ps(alphap[3]);

                __m512 _S20 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S2p - 16)));
                __m512 _S21 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S2p + 0)));
                __m512 _S22 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S2p + 16)));
                __m512 _S23 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S2p + 32)));
                __m512 _S30 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S3p - 16)));
                __m512 _S31 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S3p + 0)));
                __m512 _S32 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S3p + 16)));
                __m512 _S33 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(S3p + 32)));
                __m512 _rows2 = _mm512_mul_ps(_S20, _a0);
                __m512 _rows3 = _mm512_mul_ps(_S30, _a0);
                _rows2 = _mm512_fmadd_ps(_S21, _a1, _rows2);
                _rows3 = _mm512_fmadd_ps(_S31, _a1, _rows3);
                _rows2 = _mm512_fmadd_ps(_S22, _a2, _rows2);
                _rows3 = _mm512_fmadd_ps(_S32, _a2, _rows3);
                _rows2 = _mm512_fmadd_ps(_S23, _a3, _rows2);
                _rows3 = _mm512_fmadd_ps(_S33, _a3, _rows3);
                _mm512_store_ps(rows2p + dx * 16, _rows2);
                _mm512_store_ps(rows3p + dx * 16, _rows3);

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
                int sx = xofs[dx] * 16;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m512 _a0 = _mm512_set1_ps(alphap[0]);
                __m512 _a1 = _mm512_set1_ps(alphap[1]);
                __m512 _a2 = _mm512_set1_ps(alphap[2]);
                __m512 _a3 = _mm512_set1_ps(alphap[3]);

                for (int r = 0; r < 3; r++)
                {
                    const unsigned short* Snp = (r == 0) ? S1p : ((r == 1) ? S2p : S3p);
                    float* rowsnp = (r == 0) ? rows1p : ((r == 1) ? rows2p : rows3p);

                    __m512 _Sn0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Snp - 16)));
                    __m512 _Sn1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Snp + 0)));
                    __m512 _Sn2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Snp + 16)));
                    __m512 _Sn3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Snp + 32)));
                    __m512 _rowsn = _mm512_mul_ps(_Sn0, _a0);
                    _rowsn = _mm512_fmadd_ps(_Sn1, _a1, _rowsn);
                    _rowsn = _mm512_fmadd_ps(_Sn2, _a2, _rowsn);
                    _rowsn = _mm512_fmadd_ps(_Sn3, _a3, _rowsn);
                    _mm512_store_ps(rowsnp + dx * 16, _rowsn);
                }

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
                int sx = xofs[dx] * 16;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m512 _a0 = _mm512_set1_ps(alphap[0]);
                __m512 _a1 = _mm512_set1_ps(alphap[1]);
                __m512 _a2 = _mm512_set1_ps(alphap[2]);
                __m512 _a3 = _mm512_set1_ps(alphap[3]);

                for (int r = 0; r < 4; r++)
                {
                    const unsigned short* Snp = (r == 0) ? S0p : ((r == 1) ? S1p : ((r == 2) ? S2p : S3p));
                    float* rowsnp = (r == 0) ? rows0p : ((r == 1) ? rows1p : ((r == 2) ? rows2p : rows3p));

                    __m512 _Sn0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Snp - 16)));
                    __m512 _Sn1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Snp + 0)));
                    __m512 _Sn2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Snp + 16)));
                    __m512 _Sn3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Snp + 32)));
                    __m512 _rowsn = _mm512_mul_ps(_Sn0, _a0);
                    _rowsn = _mm512_fmadd_ps(_Sn1, _a1, _rowsn);
                    _rowsn = _mm512_fmadd_ps(_Sn2, _a2, _rowsn);
                    _rowsn = _mm512_fmadd_ps(_Sn3, _a3, _rowsn);
                    _mm512_store_ps(rowsnp + dx * 16, _rowsn);
                }

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bicubic_bf16s(rows0, rows1, rows2, rows3, dst.row<unsigned short>(dy), w * 16, beta[0], beta[1], beta[2], beta[3]);

        beta += 4;
    }
}

#endif // __AVX512F__
#endif // __AVX__

#endif // __SSE2__

static void resize_bilinear_image_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
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
            int dx = 0;
            for (; dx < w; dx++)
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
            int dx = 0;
            for (; dx < w; dx++)
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
}

static void resize_bicubic_image_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
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
            int dx = 0;
            for (; dx < w; dx++)
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
            int dx = 0;
            for (; dx < w; dx++)
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
            int dx = 0;
            for (; dx < w; dx++)
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
            int dx = 0;
            for (; dx < w; dx++)
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
}

static void interp_forward_bf16s_sse(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt, int resize_type, int align_corner, float height_scale, float width_scale, int output_height, int output_width, int has_size_expr)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        interp_forward_bf16s_sse_avx512bf16(bottom_blobs, top_blobs, opt, resize_type, align_corner, height_scale, width_scale, output_height, output_width, has_size_expr);
        return;
    }
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    (void)elemsize;

    if (dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < w; q++)
        {
            Mat top_blob_c = top_blob.channel(q);
            const unsigned short* ptr = (const unsigned short*)bottom_blob + q * elempack;
            unsigned short* outptr_c = top_blob_c;

            int size = outw * outh;
            for (int i = 0; i < size; i++)
            {
                memcpy(outptr_c + (size_t)i * elempack, ptr, elempack * sizeof(unsigned short));
            }
        }

        return;
    }

    if (dims == 2)
    {
        if (resize_type == 1) // nearest
        {
            const float ws = (output_width || has_size_expr) ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                unsigned short* outptr = top_blob.row<unsigned short>(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    const unsigned short* Sp = ptr + in_x * elempack;

                    memcpy(outptr, Sp, elempack * sizeof(unsigned short));

                    outptr += elempack;
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            linear_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                unsigned short* outptr = top_blob.row<unsigned short>(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x] * elempack;
                    const unsigned short* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];

                    int ep = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    for (; ep + 15 < elempack; ep += 16)
                    {
                        __m512 _a0 = _mm512_set1_ps(a0);
                        __m512 _a1 = _mm512_set1_ps(a1);
                        __m512 _S0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Sp + ep)));
                        __m512 _S1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Sp + ep + elempack)));
                        __m512 _p = _mm512_mul_ps(_S0, _a0);
                        _p = _mm512_fmadd_ps(_S1, _a1, _p);
                        _mm256_storeu_si256((__m256i*)(outptr + ep), float2bfloat_avx512(_p));
                    }
#endif // __AVX512F__
                    for (; ep + 7 < elempack; ep += 8)
                    {
                        __m256 _a0 = _mm256_set1_ps(a0);
                        __m256 _a1 = _mm256_set1_ps(a1);
                        __m256 _S0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Sp + ep)));
                        __m256 _S1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Sp + ep + elempack)));
                        __m256 _p = _mm256_mul_ps(_S0, _a0);
                        _p = _mm256_comp_fmadd_ps(_S1, _a1, _p);
                        _mm_storeu_si128((__m128i*)(outptr + ep), float2bfloat_avx(_p));
                    }
#endif // __AVX__
                    for (; ep + 3 < elempack; ep += 4)
                    {
                        __m128 _a0 = _mm_set1_ps(a0);
                        __m128 _a1 = _mm_set1_ps(a1);
                        __m128 _S0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Sp + ep)));
                        __m128 _S1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Sp + ep + elempack)));
                        __m128 _p = _mm_mul_ps(_S0, _a0);
                        _p = _mm_comp_fmadd_ps(_S1, _a1, _p);
                        _mm_storel_epi64((__m128i*)(outptr + ep), float2bfloat_sse(_p, _p));
                    }
#endif // __SSE2__
                    for (; ep < elempack; ep++)
                    {
                        outptr[ep] = float32_to_bfloat16(bfloat16_to_float32(Sp[ep]) * a0 + bfloat16_to_float32(Sp[ep + elempack]) * a1);
                    }

                    alphap += 2;
                    outptr += elempack;
                }
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outw * 4];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            cubic_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                unsigned short* outptr = top_blob.row<unsigned short>(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x] * elempack;
                    const unsigned short* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    float a2 = alphap[2];
                    float a3 = alphap[3];

                    int ep = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    for (; ep + 15 < elempack; ep += 16)
                    {
                        __m512 _a0 = _mm512_set1_ps(a0);
                        __m512 _a1 = _mm512_set1_ps(a1);
                        __m512 _a2 = _mm512_set1_ps(a2);
                        __m512 _a3 = _mm512_set1_ps(a3);
                        __m512 _S0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Sp + ep - elempack)));
                        __m512 _S1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Sp + ep)));
                        __m512 _S2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Sp + ep + elempack)));
                        __m512 _S3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(Sp + ep + elempack * 2)));
                        __m512 _p = _mm512_mul_ps(_S0, _a0);
                        _p = _mm512_fmadd_ps(_S1, _a1, _p);
                        _p = _mm512_fmadd_ps(_S2, _a2, _p);
                        _p = _mm512_fmadd_ps(_S3, _a3, _p);
                        _mm256_storeu_si256((__m256i*)(outptr + ep), float2bfloat_avx512(_p));
                    }
#endif // __AVX512F__
                    for (; ep + 7 < elempack; ep += 8)
                    {
                        __m256 _a0 = _mm256_set1_ps(a0);
                        __m256 _a1 = _mm256_set1_ps(a1);
                        __m256 _a2 = _mm256_set1_ps(a2);
                        __m256 _a3 = _mm256_set1_ps(a3);
                        __m256 _S0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Sp + ep - elempack)));
                        __m256 _S1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Sp + ep)));
                        __m256 _S2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Sp + ep + elempack)));
                        __m256 _S3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(Sp + ep + elempack * 2)));
                        __m256 _p = _mm256_mul_ps(_S0, _a0);
                        _p = _mm256_comp_fmadd_ps(_S1, _a1, _p);
                        _p = _mm256_comp_fmadd_ps(_S2, _a2, _p);
                        _p = _mm256_comp_fmadd_ps(_S3, _a3, _p);
                        _mm_storeu_si128((__m128i*)(outptr + ep), float2bfloat_avx(_p));
                    }
#endif // __AVX__
                    for (; ep + 3 < elempack; ep += 4)
                    {
                        __m128 _a0 = _mm_set1_ps(a0);
                        __m128 _a1 = _mm_set1_ps(a1);
                        __m128 _a2 = _mm_set1_ps(a2);
                        __m128 _a3 = _mm_set1_ps(a3);
                        __m128 _S0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Sp + ep - elempack)));
                        __m128 _S1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Sp + ep)));
                        __m128 _S2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Sp + ep + elempack)));
                        __m128 _S3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(Sp + ep + elempack * 2)));
                        __m128 _p = _mm_mul_ps(_S0, _a0);
                        _p = _mm_comp_fmadd_ps(_S1, _a1, _p);
                        _p = _mm_comp_fmadd_ps(_S2, _a2, _p);
                        _p = _mm_comp_fmadd_ps(_S3, _a3, _p);
                        _mm_storel_epi64((__m128i*)(outptr + ep), float2bfloat_sse(_p, _p));
                    }
#endif // __SSE2__
                    for (; ep < elempack; ep++)
                    {
                        outptr[ep] = float32_to_bfloat16(bfloat16_to_float32(Sp[ep - elempack]) * a0 + bfloat16_to_float32(Sp[ep]) * a1 + bfloat16_to_float32(Sp[ep + elempack]) * a2 + bfloat16_to_float32(Sp[ep + elempack * 2]) * a3);
                    }

                    alphap += 4;
                    outptr += elempack;
                }
            }

            delete[] buf;
        }

        return;
    }

    if (resize_type == 1) // nearest
    {
        const float hs = (output_height || has_size_expr) ? h / (float)outh : 1.f / height_scale;
        const float ws = (output_width || has_size_expr) ? w / (float)outw : 1.f / width_scale;

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
                    const unsigned short* Sp = ptr + in_x * elempack;

                    memcpy(outptr, Sp, elempack * sizeof(unsigned short));

                    outptr += elempack;
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

        linear_coeffs(w, outw, xofs, alpha, align_corner);
        linear_coeffs(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

#if __SSE2__
#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                resize_bilinear_image_pack16_bf16s(src, dst, alpha, xofs, beta, yofs);
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
                resize_bilinear_image_pack8_bf16s(src, dst, alpha, xofs, beta, yofs);
            }
#endif // __AVX__
            if (elempack == 4)
            {
                resize_bilinear_image_pack4_bf16s(src, dst, alpha, xofs, beta, yofs);
            }
#endif // __SSE2__
            if (elempack == 1)
            {
                resize_bilinear_image_bf16s(src, dst, alpha, xofs, beta, yofs);
            }
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

        cubic_coeffs(w, outw, xofs, alpha, align_corner);
        cubic_coeffs(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

#if __SSE2__
#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                resize_bicubic_image_pack16_bf16s(src, dst, alpha, xofs, beta, yofs);
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
                resize_bicubic_image_pack8_bf16s(src, dst, alpha, xofs, beta, yofs);
            }
#endif // __AVX__
            if (elempack == 4)
            {
                resize_bicubic_image_pack4_bf16s(src, dst, alpha, xofs, beta, yofs);
            }
#endif // __SSE2__
            if (elempack == 1)
            {
                resize_bicubic_image_bf16s(src, dst, alpha, xofs, beta, yofs);
            }
        }

        delete[] buf;
    }
}
