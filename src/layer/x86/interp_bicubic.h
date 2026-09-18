// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
void resize_bicubic_image_avx2(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs);
#endif

static inline void interpolate_cubic(float fx, float* coeffs)
{
    const float A = -0.75f;

    float fx0 = fx + 1;
    float fx1 = fx;
    float fx2 = 1 - fx;
    // float fx3 = 2 - fx;

    coeffs[0] = A * fx0 * fx0 * fx0 - 5 * A * fx0 * fx0 + 8 * A * fx0 - 4 * A;
    coeffs[1] = (A + 2) * fx1 * fx1 * fx1 - (A + 3) * fx1 * fx1 + 1;
    coeffs[2] = (A + 2) * fx2 * fx2 * fx2 - (A + 3) * fx2 * fx2 + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

static void cubic_coeffs(int w, int outw, int* xofs, float* alpha, int align_corner)
{
    double scale = (double)w / outw;
    if (align_corner)
    {
        scale = (double)(w - 1) / (outw - 1);
    }

    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        if (align_corner)
        {
            fx = (float)(dx * scale);
        }

        int sx = static_cast<int>(floor(fx));
        fx -= sx;

        interpolate_cubic(fx, alpha + dx * 4);

        if (sx <= -1)
        {
            sx = 1;
            alpha[dx * 4 + 0] = 1.f - alpha[dx * 4 + 3];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = 0.f;
            alpha[dx * 4 + 3] = 0.f;
        }
        if (sx == 0)
        {
            sx = 1;
            alpha[dx * 4 + 0] = alpha[dx * 4 + 0] + alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 2];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 3];
            alpha[dx * 4 + 3] = 0.f;
        }
        if (sx == w - 2)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = alpha[dx * 4 + 2] + alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 0];
            alpha[dx * 4 + 0] = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = 1.f - alpha[dx * 4 + 0];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 0];
            alpha[dx * 4 + 1] = 0.f;
            alpha[dx * 4 + 0] = 0.f;
        }

        xofs[dx] = sx;
    }
}

static void vresize_bicubic(const float* rows0, const float* rows1, const float* rows2, const float* rows3, float* Dp, int n, float b0, float b1, float b2, float b3)
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
        _mm512_storeu_ps(Dp + nn, _Dp);
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
        _mm256_storeu_ps(Dp + nn, _Dp);
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
        _mm_storeu_ps(Dp + nn, _Dp);
    }
#endif // __SSE2__
    for (; nn < n; nn++)
    {
        Dp[nn] = rows0[nn] * b0 + rows1[nn] * b1 + rows2[nn] * b2 + rows3[nn] * b3;
    }
}

static void resize_bicubic_image(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        resize_bicubic_image_avx2(src, dst, alpha, xofs, beta, yofs);
        return;
    }
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
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows3p = rows3;
            int dx = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; dx + 15 < w; dx += 16)
            {
                __m512i _sx = _mm512_loadu_si512(xofs + dx);
                __m512i _sxn1 = _mm512_add_epi32(_sx, _mm512_set1_epi32(-1));
                __m512i _sx1 = _mm512_add_epi32(_sx, _mm512_set1_epi32(1));
                __m512i _sx2 = _mm512_add_epi32(_sx, _mm512_set1_epi32(2));

                __m512 _S30 = _mm512_i32gather_ps(_sxn1, S3, sizeof(float));
                __m512 _S31 = _mm512_i32gather_ps(_sx, S3, sizeof(float));
                __m512 _S32 = _mm512_i32gather_ps(_sx1, S3, sizeof(float));
                __m512 _S33 = _mm512_i32gather_ps(_sx2, S3, sizeof(float));

                __m512i _alpha_idx = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60);
                __m512 _a0 = _mm512_i32gather_ps(_alpha_idx, alphap, sizeof(float));
                __m512 _a1 = _mm512_i32gather_ps(_mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(1)), alphap, sizeof(float));
                __m512 _a2 = _mm512_i32gather_ps(_mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(2)), alphap, sizeof(float));
                __m512 _a3 = _mm512_i32gather_ps(_mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(3)), alphap, sizeof(float));

                __m512 _rows3 = _mm512_mul_ps(_S30, _a0);
                _rows3 = _mm512_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm512_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm512_fmadd_ps(_S33, _a3, _rows3);
                _mm512_storeu_ps(rows3p + dx, _rows3);

                alphap += 64;
            }
#endif // __AVX512F__
            for (; dx + 7 < w; dx += 8)
            {
#if __AVX2__
                __m256i _sx = _mm256_loadu_si256((const __m256i*)(xofs + dx));
                __m256i _sxn1 = _mm256_add_epi32(_sx, _mm256_set1_epi32(-1));
                __m256i _sx1 = _mm256_add_epi32(_sx, _mm256_set1_epi32(1));
                __m256i _sx2 = _mm256_add_epi32(_sx, _mm256_set1_epi32(2));

                __m256 _S30 = _mm256_i32gather_ps(S3, _sxn1, sizeof(float));
                __m256 _S31 = _mm256_i32gather_ps(S3, _sx, sizeof(float));
                __m256 _S32 = _mm256_i32gather_ps(S3, _sx1, sizeof(float));
                __m256 _S33 = _mm256_i32gather_ps(S3, _sx2, sizeof(float));

                __m256i _alpha_idx = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
                __m256 _a0 = _mm256_i32gather_ps(alphap, _alpha_idx, sizeof(float));
                __m256 _a1 = _mm256_i32gather_ps(alphap, _mm256_add_epi32(_alpha_idx, _mm256_set1_epi32(1)), sizeof(float));
                __m256 _a2 = _mm256_i32gather_ps(alphap, _mm256_add_epi32(_alpha_idx, _mm256_set1_epi32(2)), sizeof(float));
                __m256 _a3 = _mm256_i32gather_ps(alphap, _mm256_add_epi32(_alpha_idx, _mm256_set1_epi32(3)), sizeof(float));
#else
                __m256 _S30 = _mm256_setr_ps(S3[xofs[dx] - 1], S3[xofs[dx + 1] - 1], S3[xofs[dx + 2] - 1], S3[xofs[dx + 3] - 1], S3[xofs[dx + 4] - 1], S3[xofs[dx + 5] - 1], S3[xofs[dx + 6] - 1], S3[xofs[dx + 7] - 1]);
                __m256 _S31 = _mm256_setr_ps(S3[xofs[dx]], S3[xofs[dx + 1]], S3[xofs[dx + 2]], S3[xofs[dx + 3]], S3[xofs[dx + 4]], S3[xofs[dx + 5]], S3[xofs[dx + 6]], S3[xofs[dx + 7]]);
                __m256 _S32 = _mm256_setr_ps(S3[xofs[dx] + 1], S3[xofs[dx + 1] + 1], S3[xofs[dx + 2] + 1], S3[xofs[dx + 3] + 1], S3[xofs[dx + 4] + 1], S3[xofs[dx + 5] + 1], S3[xofs[dx + 6] + 1], S3[xofs[dx + 7] + 1]);
                __m256 _S33 = _mm256_setr_ps(S3[xofs[dx] + 2], S3[xofs[dx + 1] + 2], S3[xofs[dx + 2] + 2], S3[xofs[dx + 3] + 2], S3[xofs[dx + 4] + 2], S3[xofs[dx + 5] + 2], S3[xofs[dx + 6] + 2], S3[xofs[dx + 7] + 2]);

                __m256 _a0 = _mm256_setr_ps(alphap[0], alphap[4], alphap[8], alphap[12], alphap[16], alphap[20], alphap[24], alphap[28]);
                __m256 _a1 = _mm256_setr_ps(alphap[1], alphap[5], alphap[9], alphap[13], alphap[17], alphap[21], alphap[25], alphap[29]);
                __m256 _a2 = _mm256_setr_ps(alphap[2], alphap[6], alphap[10], alphap[14], alphap[18], alphap[22], alphap[26], alphap[30]);
                __m256 _a3 = _mm256_setr_ps(alphap[3], alphap[7], alphap[11], alphap[15], alphap[19], alphap[23], alphap[27], alphap[31]);
#endif

                __m256 _rows3 = _mm256_mul_ps(_S30, _a0);
                _rows3 = _mm256_comp_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm256_comp_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm256_comp_fmadd_ps(_S33, _a3, _rows3);
                _mm256_storeu_ps(rows3p + dx, _rows3);

                alphap += 32;
            }
#endif // __AVX__
#endif // __SSE2__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

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
            const float* S2 = src.row(sy + 1);
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows2p = rows2;
            float* rows3p = rows3;
            int dx = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; dx + 15 < w; dx += 16)
            {
                __m512i _sx = _mm512_loadu_si512(xofs + dx);
                __m512i _sxn1 = _mm512_add_epi32(_sx, _mm512_set1_epi32(-1));
                __m512i _sx1 = _mm512_add_epi32(_sx, _mm512_set1_epi32(1));
                __m512i _sx2 = _mm512_add_epi32(_sx, _mm512_set1_epi32(2));

                __m512i _alpha_idx = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60);
                __m512 _a0 = _mm512_i32gather_ps(_alpha_idx, alphap, sizeof(float));
                __m512 _a1 = _mm512_i32gather_ps(_mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(1)), alphap, sizeof(float));
                __m512 _a2 = _mm512_i32gather_ps(_mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(2)), alphap, sizeof(float));
                __m512 _a3 = _mm512_i32gather_ps(_mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(3)), alphap, sizeof(float));

                __m512 _S20 = _mm512_i32gather_ps(_sxn1, S2, sizeof(float));
                __m512 _S21 = _mm512_i32gather_ps(_sx, S2, sizeof(float));
                __m512 _S22 = _mm512_i32gather_ps(_sx1, S2, sizeof(float));
                __m512 _S23 = _mm512_i32gather_ps(_sx2, S2, sizeof(float));

                __m512 _rows2 = _mm512_mul_ps(_S20, _a0);
                _rows2 = _mm512_fmadd_ps(_S21, _a1, _rows2);
                _rows2 = _mm512_fmadd_ps(_S22, _a2, _rows2);
                _rows2 = _mm512_fmadd_ps(_S23, _a3, _rows2);
                _mm512_storeu_ps(rows2p + dx, _rows2);

                __m512 _S30 = _mm512_i32gather_ps(_sxn1, S3, sizeof(float));
                __m512 _S31 = _mm512_i32gather_ps(_sx, S3, sizeof(float));
                __m512 _S32 = _mm512_i32gather_ps(_sx1, S3, sizeof(float));
                __m512 _S33 = _mm512_i32gather_ps(_sx2, S3, sizeof(float));

                __m512 _rows3 = _mm512_mul_ps(_S30, _a0);
                _rows3 = _mm512_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm512_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm512_fmadd_ps(_S33, _a3, _rows3);
                _mm512_storeu_ps(rows3p + dx, _rows3);

                alphap += 64;
            }
#endif // __AVX512F__
            for (; dx + 7 < w; dx += 8)
            {
#if __AVX2__
                __m256i _sx = _mm256_loadu_si256((const __m256i*)(xofs + dx));
                __m256i _sxn1 = _mm256_add_epi32(_sx, _mm256_set1_epi32(-1));
                __m256i _sx1 = _mm256_add_epi32(_sx, _mm256_set1_epi32(1));
                __m256i _sx2 = _mm256_add_epi32(_sx, _mm256_set1_epi32(2));

                __m256i _alpha_idx = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
                __m256 _a0 = _mm256_i32gather_ps(alphap, _alpha_idx, sizeof(float));
                __m256 _a1 = _mm256_i32gather_ps(alphap, _mm256_add_epi32(_alpha_idx, _mm256_set1_epi32(1)), sizeof(float));
                __m256 _a2 = _mm256_i32gather_ps(alphap, _mm256_add_epi32(_alpha_idx, _mm256_set1_epi32(2)), sizeof(float));
                __m256 _a3 = _mm256_i32gather_ps(alphap, _mm256_add_epi32(_alpha_idx, _mm256_set1_epi32(3)), sizeof(float));

                __m256 _S20 = _mm256_i32gather_ps(S2, _sxn1, sizeof(float));
                __m256 _S21 = _mm256_i32gather_ps(S2, _sx, sizeof(float));
                __m256 _S22 = _mm256_i32gather_ps(S2, _sx1, sizeof(float));
                __m256 _S23 = _mm256_i32gather_ps(S2, _sx2, sizeof(float));
#else
                __m256 _S20 = _mm256_setr_ps(S2[xofs[dx] - 1], S2[xofs[dx + 1] - 1], S2[xofs[dx + 2] - 1], S2[xofs[dx + 3] - 1], S2[xofs[dx + 4] - 1], S2[xofs[dx + 5] - 1], S2[xofs[dx + 6] - 1], S2[xofs[dx + 7] - 1]);
                __m256 _S21 = _mm256_setr_ps(S2[xofs[dx]], S2[xofs[dx + 1]], S2[xofs[dx + 2]], S2[xofs[dx + 3]], S2[xofs[dx + 4]], S2[xofs[dx + 5]], S2[xofs[dx + 6]], S2[xofs[dx + 7]]);
                __m256 _S22 = _mm256_setr_ps(S2[xofs[dx] + 1], S2[xofs[dx + 1] + 1], S2[xofs[dx + 2] + 1], S2[xofs[dx + 3] + 1], S2[xofs[dx + 4] + 1], S2[xofs[dx + 5] + 1], S2[xofs[dx + 6] + 1], S2[xofs[dx + 7] + 1]);
                __m256 _S23 = _mm256_setr_ps(S2[xofs[dx] + 2], S2[xofs[dx + 1] + 2], S2[xofs[dx + 2] + 2], S2[xofs[dx + 3] + 2], S2[xofs[dx + 4] + 2], S2[xofs[dx + 5] + 2], S2[xofs[dx + 6] + 2], S2[xofs[dx + 7] + 2]);

                __m256 _a0 = _mm256_setr_ps(alphap[0], alphap[4], alphap[8], alphap[12], alphap[16], alphap[20], alphap[24], alphap[28]);
                __m256 _a1 = _mm256_setr_ps(alphap[1], alphap[5], alphap[9], alphap[13], alphap[17], alphap[21], alphap[25], alphap[29]);
                __m256 _a2 = _mm256_setr_ps(alphap[2], alphap[6], alphap[10], alphap[14], alphap[18], alphap[22], alphap[26], alphap[30]);
                __m256 _a3 = _mm256_setr_ps(alphap[3], alphap[7], alphap[11], alphap[15], alphap[19], alphap[23], alphap[27], alphap[31]);
#endif

                __m256 _rows2 = _mm256_mul_ps(_S20, _a0);
                _rows2 = _mm256_comp_fmadd_ps(_S21, _a1, _rows2);
                _rows2 = _mm256_comp_fmadd_ps(_S22, _a2, _rows2);
                _rows2 = _mm256_comp_fmadd_ps(_S23, _a3, _rows2);
                _mm256_storeu_ps(rows2p + dx, _rows2);

#if __AVX2__
                __m256 _S30 = _mm256_i32gather_ps(S3, _sxn1, sizeof(float));
                __m256 _S31 = _mm256_i32gather_ps(S3, _sx, sizeof(float));
                __m256 _S32 = _mm256_i32gather_ps(S3, _sx1, sizeof(float));
                __m256 _S33 = _mm256_i32gather_ps(S3, _sx2, sizeof(float));
#else
                __m256 _S30 = _mm256_setr_ps(S3[xofs[dx] - 1], S3[xofs[dx + 1] - 1], S3[xofs[dx + 2] - 1], S3[xofs[dx + 3] - 1], S3[xofs[dx + 4] - 1], S3[xofs[dx + 5] - 1], S3[xofs[dx + 6] - 1], S3[xofs[dx + 7] - 1]);
                __m256 _S31 = _mm256_setr_ps(S3[xofs[dx]], S3[xofs[dx + 1]], S3[xofs[dx + 2]], S3[xofs[dx + 3]], S3[xofs[dx + 4]], S3[xofs[dx + 5]], S3[xofs[dx + 6]], S3[xofs[dx + 7]]);
                __m256 _S32 = _mm256_setr_ps(S3[xofs[dx] + 1], S3[xofs[dx + 1] + 1], S3[xofs[dx + 2] + 1], S3[xofs[dx + 3] + 1], S3[xofs[dx + 4] + 1], S3[xofs[dx + 5] + 1], S3[xofs[dx + 6] + 1], S3[xofs[dx + 7] + 1]);
                __m256 _S33 = _mm256_setr_ps(S3[xofs[dx] + 2], S3[xofs[dx + 1] + 2], S3[xofs[dx + 2] + 2], S3[xofs[dx + 3] + 2], S3[xofs[dx + 4] + 2], S3[xofs[dx + 5] + 2], S3[xofs[dx + 6] + 2], S3[xofs[dx + 7] + 2]);
#endif

                __m256 _rows3 = _mm256_mul_ps(_S30, _a0);
                _rows3 = _mm256_comp_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm256_comp_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm256_comp_fmadd_ps(_S33, _a3, _rows3);
                _mm256_storeu_ps(rows3p + dx, _rows3);

                alphap += 32;
            }
#endif // __AVX__
#endif // __SSE2__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

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
            const float* S1 = src.row(sy);
            const float* S2 = src.row(sy + 1);
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            int dx = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; dx + 15 < w; dx += 16)
            {
                __m512i _sx = _mm512_loadu_si512(xofs + dx);
                __m512i _sxn1 = _mm512_add_epi32(_sx, _mm512_set1_epi32(-1));
                __m512i _sx1 = _mm512_add_epi32(_sx, _mm512_set1_epi32(1));
                __m512i _sx2 = _mm512_add_epi32(_sx, _mm512_set1_epi32(2));

                __m512i _alpha_idx = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60);
                __m512 _a0 = _mm512_i32gather_ps(_alpha_idx, alphap, sizeof(float));
                __m512 _a1 = _mm512_i32gather_ps(_mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(1)), alphap, sizeof(float));
                __m512 _a2 = _mm512_i32gather_ps(_mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(2)), alphap, sizeof(float));
                __m512 _a3 = _mm512_i32gather_ps(_mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(3)), alphap, sizeof(float));

                __m512 _S10 = _mm512_i32gather_ps(_sxn1, S1, sizeof(float));
                __m512 _S11 = _mm512_i32gather_ps(_sx, S1, sizeof(float));
                __m512 _S12 = _mm512_i32gather_ps(_sx1, S1, sizeof(float));
                __m512 _S13 = _mm512_i32gather_ps(_sx2, S1, sizeof(float));
                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _rows1 = _mm512_fmadd_ps(_S12, _a2, _rows1);
                _rows1 = _mm512_fmadd_ps(_S13, _a3, _rows1);
                _mm512_storeu_ps(rows1p + dx, _rows1);

                __m512 _S20 = _mm512_i32gather_ps(_sxn1, S2, sizeof(float));
                __m512 _S21 = _mm512_i32gather_ps(_sx, S2, sizeof(float));
                __m512 _S22 = _mm512_i32gather_ps(_sx1, S2, sizeof(float));
                __m512 _S23 = _mm512_i32gather_ps(_sx2, S2, sizeof(float));
                __m512 _rows2 = _mm512_mul_ps(_S20, _a0);
                _rows2 = _mm512_fmadd_ps(_S21, _a1, _rows2);
                _rows2 = _mm512_fmadd_ps(_S22, _a2, _rows2);
                _rows2 = _mm512_fmadd_ps(_S23, _a3, _rows2);
                _mm512_storeu_ps(rows2p + dx, _rows2);

                __m512 _S30 = _mm512_i32gather_ps(_sxn1, S3, sizeof(float));
                __m512 _S31 = _mm512_i32gather_ps(_sx, S3, sizeof(float));
                __m512 _S32 = _mm512_i32gather_ps(_sx1, S3, sizeof(float));
                __m512 _S33 = _mm512_i32gather_ps(_sx2, S3, sizeof(float));
                __m512 _rows3 = _mm512_mul_ps(_S30, _a0);
                _rows3 = _mm512_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm512_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm512_fmadd_ps(_S33, _a3, _rows3);
                _mm512_storeu_ps(rows3p + dx, _rows3);

                alphap += 64;
            }
#endif // __AVX512F__
            for (; dx + 7 < w; dx += 8)
            {
#if __AVX2__
                __m256i _sx = _mm256_loadu_si256((const __m256i*)(xofs + dx));
                __m256i _sxn1 = _mm256_add_epi32(_sx, _mm256_set1_epi32(-1));
                __m256i _sx1 = _mm256_add_epi32(_sx, _mm256_set1_epi32(1));
                __m256i _sx2 = _mm256_add_epi32(_sx, _mm256_set1_epi32(2));

                __m256i _alpha_idx = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
                __m256 _a0 = _mm256_i32gather_ps(alphap, _alpha_idx, sizeof(float));
                __m256 _a1 = _mm256_i32gather_ps(alphap, _mm256_add_epi32(_alpha_idx, _mm256_set1_epi32(1)), sizeof(float));
                __m256 _a2 = _mm256_i32gather_ps(alphap, _mm256_add_epi32(_alpha_idx, _mm256_set1_epi32(2)), sizeof(float));
                __m256 _a3 = _mm256_i32gather_ps(alphap, _mm256_add_epi32(_alpha_idx, _mm256_set1_epi32(3)), sizeof(float));
#else
                __m256 _a0 = _mm256_setr_ps(alphap[0], alphap[4], alphap[8], alphap[12], alphap[16], alphap[20], alphap[24], alphap[28]);
                __m256 _a1 = _mm256_setr_ps(alphap[1], alphap[5], alphap[9], alphap[13], alphap[17], alphap[21], alphap[25], alphap[29]);
                __m256 _a2 = _mm256_setr_ps(alphap[2], alphap[6], alphap[10], alphap[14], alphap[18], alphap[22], alphap[26], alphap[30]);
                __m256 _a3 = _mm256_setr_ps(alphap[3], alphap[7], alphap[11], alphap[15], alphap[19], alphap[23], alphap[27], alphap[31]);
#endif

                for (int r = 0; r < 3; r++)
                {
                    const float* Sn = (r == 0) ? S1 : ((r == 1) ? S2 : S3);
                    float* rowsnp = (r == 0) ? rows1p : ((r == 1) ? rows2p : rows3p);

#if __AVX2__
                    __m256 _Sn0 = _mm256_i32gather_ps(Sn, _sxn1, sizeof(float));
                    __m256 _Sn1 = _mm256_i32gather_ps(Sn, _sx, sizeof(float));
                    __m256 _Sn2 = _mm256_i32gather_ps(Sn, _sx1, sizeof(float));
                    __m256 _Sn3 = _mm256_i32gather_ps(Sn, _sx2, sizeof(float));
#else
                    __m256 _Sn0 = _mm256_setr_ps(Sn[xofs[dx] - 1], Sn[xofs[dx + 1] - 1], Sn[xofs[dx + 2] - 1], Sn[xofs[dx + 3] - 1], Sn[xofs[dx + 4] - 1], Sn[xofs[dx + 5] - 1], Sn[xofs[dx + 6] - 1], Sn[xofs[dx + 7] - 1]);
                    __m256 _Sn1 = _mm256_setr_ps(Sn[xofs[dx]], Sn[xofs[dx + 1]], Sn[xofs[dx + 2]], Sn[xofs[dx + 3]], Sn[xofs[dx + 4]], Sn[xofs[dx + 5]], Sn[xofs[dx + 6]], Sn[xofs[dx + 7]]);
                    __m256 _Sn2 = _mm256_setr_ps(Sn[xofs[dx] + 1], Sn[xofs[dx + 1] + 1], Sn[xofs[dx + 2] + 1], Sn[xofs[dx + 3] + 1], Sn[xofs[dx + 4] + 1], Sn[xofs[dx + 5] + 1], Sn[xofs[dx + 6] + 1], Sn[xofs[dx + 7] + 1]);
                    __m256 _Sn3 = _mm256_setr_ps(Sn[xofs[dx] + 2], Sn[xofs[dx + 1] + 2], Sn[xofs[dx + 2] + 2], Sn[xofs[dx + 3] + 2], Sn[xofs[dx + 4] + 2], Sn[xofs[dx + 5] + 2], Sn[xofs[dx + 6] + 2], Sn[xofs[dx + 7] + 2]);
#endif

                    __m256 _rowsn = _mm256_mul_ps(_Sn0, _a0);
                    _rowsn = _mm256_comp_fmadd_ps(_Sn1, _a1, _rowsn);
                    _rowsn = _mm256_comp_fmadd_ps(_Sn2, _a2, _rowsn);
                    _rowsn = _mm256_comp_fmadd_ps(_Sn3, _a3, _rowsn);
                    _mm256_storeu_ps(rowsnp + dx, _rowsn);
                }

                alphap += 32;
            }
#endif // __AVX__
#endif // __SSE2__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows1p[dx] = S1p[-1] * a0 + S1p[0] * a1 + S1p[1] * a2 + S1p[2] * a3;
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }
        else
        {
            // hresize four rows
            const float* S0 = src.row(sy - 1);
            const float* S1 = src.row(sy);
            const float* S2 = src.row(sy + 1);
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            int dx = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; dx + 15 < w; dx += 16)
            {
                __m512i _sx = _mm512_loadu_si512(xofs + dx);
                __m512i _sxn1 = _mm512_add_epi32(_sx, _mm512_set1_epi32(-1));
                __m512i _sx1 = _mm512_add_epi32(_sx, _mm512_set1_epi32(1));
                __m512i _sx2 = _mm512_add_epi32(_sx, _mm512_set1_epi32(2));

                __m512i _alpha_idx = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60);
                __m512 _a0 = _mm512_i32gather_ps(_alpha_idx, alphap, sizeof(float));
                __m512 _a1 = _mm512_i32gather_ps(_mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(1)), alphap, sizeof(float));
                __m512 _a2 = _mm512_i32gather_ps(_mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(2)), alphap, sizeof(float));
                __m512 _a3 = _mm512_i32gather_ps(_mm512_add_epi32(_alpha_idx, _mm512_set1_epi32(3)), alphap, sizeof(float));

                __m512 _S00 = _mm512_i32gather_ps(_sxn1, S0, sizeof(float));
                __m512 _S01 = _mm512_i32gather_ps(_sx, S0, sizeof(float));
                __m512 _S02 = _mm512_i32gather_ps(_sx1, S0, sizeof(float));
                __m512 _S03 = _mm512_i32gather_ps(_sx2, S0, sizeof(float));
                __m512 _rows0 = _mm512_mul_ps(_S00, _a0);
                _rows0 = _mm512_fmadd_ps(_S01, _a1, _rows0);
                _rows0 = _mm512_fmadd_ps(_S02, _a2, _rows0);
                _rows0 = _mm512_fmadd_ps(_S03, _a3, _rows0);
                _mm512_storeu_ps(rows0p + dx, _rows0);

                __m512 _S10 = _mm512_i32gather_ps(_sxn1, S1, sizeof(float));
                __m512 _S11 = _mm512_i32gather_ps(_sx, S1, sizeof(float));
                __m512 _S12 = _mm512_i32gather_ps(_sx1, S1, sizeof(float));
                __m512 _S13 = _mm512_i32gather_ps(_sx2, S1, sizeof(float));
                __m512 _rows1 = _mm512_mul_ps(_S10, _a0);
                _rows1 = _mm512_fmadd_ps(_S11, _a1, _rows1);
                _rows1 = _mm512_fmadd_ps(_S12, _a2, _rows1);
                _rows1 = _mm512_fmadd_ps(_S13, _a3, _rows1);
                _mm512_storeu_ps(rows1p + dx, _rows1);

                __m512 _S20 = _mm512_i32gather_ps(_sxn1, S2, sizeof(float));
                __m512 _S21 = _mm512_i32gather_ps(_sx, S2, sizeof(float));
                __m512 _S22 = _mm512_i32gather_ps(_sx1, S2, sizeof(float));
                __m512 _S23 = _mm512_i32gather_ps(_sx2, S2, sizeof(float));
                __m512 _rows2 = _mm512_mul_ps(_S20, _a0);
                _rows2 = _mm512_fmadd_ps(_S21, _a1, _rows2);
                _rows2 = _mm512_fmadd_ps(_S22, _a2, _rows2);
                _rows2 = _mm512_fmadd_ps(_S23, _a3, _rows2);
                _mm512_storeu_ps(rows2p + dx, _rows2);

                __m512 _S30 = _mm512_i32gather_ps(_sxn1, S3, sizeof(float));
                __m512 _S31 = _mm512_i32gather_ps(_sx, S3, sizeof(float));
                __m512 _S32 = _mm512_i32gather_ps(_sx1, S3, sizeof(float));
                __m512 _S33 = _mm512_i32gather_ps(_sx2, S3, sizeof(float));
                __m512 _rows3 = _mm512_mul_ps(_S30, _a0);
                _rows3 = _mm512_fmadd_ps(_S31, _a1, _rows3);
                _rows3 = _mm512_fmadd_ps(_S32, _a2, _rows3);
                _rows3 = _mm512_fmadd_ps(_S33, _a3, _rows3);
                _mm512_storeu_ps(rows3p + dx, _rows3);

                alphap += 64;
            }
#endif // __AVX512F__
            for (; dx + 7 < w; dx += 8)
            {
#if __AVX2__
                __m256i _sx = _mm256_loadu_si256((const __m256i*)(xofs + dx));
                __m256i _sxn1 = _mm256_add_epi32(_sx, _mm256_set1_epi32(-1));
                __m256i _sx1 = _mm256_add_epi32(_sx, _mm256_set1_epi32(1));
                __m256i _sx2 = _mm256_add_epi32(_sx, _mm256_set1_epi32(2));

                __m256i _alpha_idx = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
                __m256 _a0 = _mm256_i32gather_ps(alphap, _alpha_idx, sizeof(float));
                __m256 _a1 = _mm256_i32gather_ps(alphap, _mm256_add_epi32(_alpha_idx, _mm256_set1_epi32(1)), sizeof(float));
                __m256 _a2 = _mm256_i32gather_ps(alphap, _mm256_add_epi32(_alpha_idx, _mm256_set1_epi32(2)), sizeof(float));
                __m256 _a3 = _mm256_i32gather_ps(alphap, _mm256_add_epi32(_alpha_idx, _mm256_set1_epi32(3)), sizeof(float));
#else
                __m256 _a0 = _mm256_setr_ps(alphap[0], alphap[4], alphap[8], alphap[12], alphap[16], alphap[20], alphap[24], alphap[28]);
                __m256 _a1 = _mm256_setr_ps(alphap[1], alphap[5], alphap[9], alphap[13], alphap[17], alphap[21], alphap[25], alphap[29]);
                __m256 _a2 = _mm256_setr_ps(alphap[2], alphap[6], alphap[10], alphap[14], alphap[18], alphap[22], alphap[26], alphap[30]);
                __m256 _a3 = _mm256_setr_ps(alphap[3], alphap[7], alphap[11], alphap[15], alphap[19], alphap[23], alphap[27], alphap[31]);
#endif

                for (int r = 0; r < 4; r++)
                {
                    const float* Sn = (r == 0) ? S0 : ((r == 1) ? S1 : ((r == 2) ? S2 : S3));
                    float* rowsnp = (r == 0) ? rows0p : ((r == 1) ? rows1p : ((r == 2) ? rows2p : rows3p));

#if __AVX2__
                    __m256 _Sn0 = _mm256_i32gather_ps(Sn, _sxn1, sizeof(float));
                    __m256 _Sn1 = _mm256_i32gather_ps(Sn, _sx, sizeof(float));
                    __m256 _Sn2 = _mm256_i32gather_ps(Sn, _sx1, sizeof(float));
                    __m256 _Sn3 = _mm256_i32gather_ps(Sn, _sx2, sizeof(float));
#else
                    __m256 _Sn0 = _mm256_setr_ps(Sn[xofs[dx] - 1], Sn[xofs[dx + 1] - 1], Sn[xofs[dx + 2] - 1], Sn[xofs[dx + 3] - 1], Sn[xofs[dx + 4] - 1], Sn[xofs[dx + 5] - 1], Sn[xofs[dx + 6] - 1], Sn[xofs[dx + 7] - 1]);
                    __m256 _Sn1 = _mm256_setr_ps(Sn[xofs[dx]], Sn[xofs[dx + 1]], Sn[xofs[dx + 2]], Sn[xofs[dx + 3]], Sn[xofs[dx + 4]], Sn[xofs[dx + 5]], Sn[xofs[dx + 6]], Sn[xofs[dx + 7]]);
                    __m256 _Sn2 = _mm256_setr_ps(Sn[xofs[dx] + 1], Sn[xofs[dx + 1] + 1], Sn[xofs[dx + 2] + 1], Sn[xofs[dx + 3] + 1], Sn[xofs[dx + 4] + 1], Sn[xofs[dx + 5] + 1], Sn[xofs[dx + 6] + 1], Sn[xofs[dx + 7] + 1]);
                    __m256 _Sn3 = _mm256_setr_ps(Sn[xofs[dx] + 2], Sn[xofs[dx + 1] + 2], Sn[xofs[dx + 2] + 2], Sn[xofs[dx + 3] + 2], Sn[xofs[dx + 4] + 2], Sn[xofs[dx + 5] + 2], Sn[xofs[dx + 6] + 2], Sn[xofs[dx + 7] + 2]);
#endif

                    __m256 _rowsn = _mm256_mul_ps(_Sn0, _a0);
                    _rowsn = _mm256_comp_fmadd_ps(_Sn1, _a1, _rowsn);
                    _rowsn = _mm256_comp_fmadd_ps(_Sn2, _a2, _rowsn);
                    _rowsn = _mm256_comp_fmadd_ps(_Sn3, _a3, _rowsn);
                    _mm256_storeu_ps(rowsnp + dx, _rowsn);
                }

                alphap += 32;
            }
#endif // __AVX__
#endif // __SSE2__
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows0p[dx] = S0p[-1] * a0 + S0p[0] * a1 + S0p[1] * a2 + S0p[2] * a3;
                rows1p[dx] = S1p[-1] * a0 + S1p[0] * a1 + S1p[1] * a2 + S1p[2] * a3;
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        vresize_bicubic(rows0, rows1, rows2, rows3, dst.row(dy), w, beta[0], beta[1], beta[2], beta[3]);

        beta += 4;
    }
}
