// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// gemm_loongarch.h
#if NCNN_RUNTIME_CPU && __loongarch_asx
namespace Gemm_loongarch_lasx_utility {
#elif NCNN_RUNTIME_CPU && __loongarch_sx
namespace Gemm_loongarch_lsx_utility {
#else
namespace Gemm_loongarch_utility {
#endif
void pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
}

static void convolution_im2col_pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // A = (pa, maxk, inch/pa), outch
#if NCNN_RUNTIME_CPU && __loongarch_asx
    Gemm_loongarch_lasx_utility::pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
#elif NCNN_RUNTIME_CPU && __loongarch_sx
    Gemm_loongarch_lsx_utility::pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
#else
    Gemm_loongarch_utility::pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
#endif
}

static void convolution_gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    // NCNN_LOGE("convolution_gemm_transB_packed_tile_int8 %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

#if NCNN_RUNTIME_CPU && __loongarch_asx
    Gemm_loongarch_lasx_utility::gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
#elif NCNN_RUNTIME_CPU && __loongarch_sx
    Gemm_loongarch_lsx_utility::gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
#else
    Gemm_loongarch_utility::gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
#endif
}

static void unpack_output_tile_int32(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = (int)top_blob.cstep;

    // NCNN_LOGE("unpack_output_tile_int32 %d %d %d %d     %d %d", i, max_ii, j, max_jj, out_elempack, out_hstep);

    const int* pp = topT;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        int* p0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m256i _f0 = __lasx_xvld(pp, 0);
            __m256i _f1 = __lasx_xvld(pp + 8, 0);
            __m256i _f2 = __lasx_xvld(pp + 16, 0);
            __m256i _f3 = __lasx_xvld(pp + 24, 0);

            if (out_elempack == 4)
            {
                __m256i _tmp0 = __lasx_xvpermi_q(_f1, _f0, _LSX_SHUFFLE(0, 2, 0, 0));
                __m256i _tmp1 = __lasx_xvpermi_q(_f3, _f2, _LSX_SHUFFLE(0, 2, 0, 0));
                __m256i _tmp2 = __lasx_xvpermi_q(_f1, _f0, _LSX_SHUFFLE(0, 3, 0, 1));
                __m256i _tmp3 = __lasx_xvpermi_q(_f3, _f2, _LSX_SHUFFLE(0, 3, 0, 1));

                __lasx_xvst(_tmp0, p0, 0);
                __lasx_xvst(_tmp1, p0 + 8, 0);
                __lasx_xvst(_tmp2, p0 + out_hstep * 4, 0);
                __lasx_xvst(_tmp3, p0 + out_hstep * 4 + 8, 0);
                p0 += 16;
            }
            if (out_elempack == 1)
            {
                int tmp[32];
                __lasx_xvst(_f0, tmp, 0);
                __lasx_xvst(_f1, tmp + 8, 0);
                __lasx_xvst(_f2, tmp + 16, 0);
                __lasx_xvst(_f3, tmp + 24, 0);

                p0[0] = tmp[0];
                p0[1] = tmp[8];
                p0[2] = tmp[16];
                p0[3] = tmp[24];
                p0[out_hstep] = tmp[1];
                p0[out_hstep + 1] = tmp[9];
                p0[out_hstep + 2] = tmp[17];
                p0[out_hstep + 3] = tmp[25];
                p0[out_hstep * 2] = tmp[2];
                p0[out_hstep * 2 + 1] = tmp[10];
                p0[out_hstep * 2 + 2] = tmp[18];
                p0[out_hstep * 2 + 3] = tmp[26];
                p0[out_hstep * 3] = tmp[3];
                p0[out_hstep * 3 + 1] = tmp[11];
                p0[out_hstep * 3 + 2] = tmp[19];
                p0[out_hstep * 3 + 3] = tmp[27];
                p0[out_hstep * 4] = tmp[4];
                p0[out_hstep * 4 + 1] = tmp[12];
                p0[out_hstep * 4 + 2] = tmp[20];
                p0[out_hstep * 4 + 3] = tmp[28];
                p0[out_hstep * 5] = tmp[5];
                p0[out_hstep * 5 + 1] = tmp[13];
                p0[out_hstep * 5 + 2] = tmp[21];
                p0[out_hstep * 5 + 3] = tmp[29];
                p0[out_hstep * 6] = tmp[6];
                p0[out_hstep * 6 + 1] = tmp[14];
                p0[out_hstep * 6 + 2] = tmp[22];
                p0[out_hstep * 6 + 3] = tmp[30];
                p0[out_hstep * 7] = tmp[7];
                p0[out_hstep * 7 + 1] = tmp[15];
                p0[out_hstep * 7 + 2] = tmp[23];
                p0[out_hstep * 7 + 3] = tmp[31];
                p0 += 4;
            }

            pp += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m256i _f0 = __lasx_xvld(pp, 0);
            __m256i _f1 = __lasx_xvld(pp + 8, 0);

            if (out_elempack == 4)
            {
                __m256i _tmp0 = __lasx_xvpermi_q(_f1, _f0, _LSX_SHUFFLE(0, 2, 0, 0));
                __m256i _tmp1 = __lasx_xvpermi_q(_f1, _f0, _LSX_SHUFFLE(0, 3, 0, 1));

                __lasx_xvst(_tmp0, p0, 0);
                __lasx_xvst(_tmp1, p0 + out_hstep * 4, 0);
                p0 += 8;
            }
            if (out_elempack == 1)
            {
                int tmp[16];
                __lasx_xvst(_f0, tmp, 0);
                __lasx_xvst(_f1, tmp + 8, 0);

                p0[0] = tmp[0];
                p0[1] = tmp[8];
                p0[out_hstep] = tmp[1];
                p0[out_hstep + 1] = tmp[9];
                p0[out_hstep * 2] = tmp[2];
                p0[out_hstep * 2 + 1] = tmp[10];
                p0[out_hstep * 3] = tmp[3];
                p0[out_hstep * 3 + 1] = tmp[11];
                p0[out_hstep * 4] = tmp[4];
                p0[out_hstep * 4 + 1] = tmp[12];
                p0[out_hstep * 5] = tmp[5];
                p0[out_hstep * 5 + 1] = tmp[13];
                p0[out_hstep * 6] = tmp[6];
                p0[out_hstep * 6 + 1] = tmp[14];
                p0[out_hstep * 7] = tmp[7];
                p0[out_hstep * 7 + 1] = tmp[15];
                p0 += 2;
            }

            pp += 16;
        }
        for (; jj < max_jj; jj++)
        {
            __m256i _f0 = __lasx_xvld(pp, 0);

            if (out_elempack == 4)
            {
                __lsx_vst(__lasx_extract_lo128(_f0), p0, 0);
                __lsx_vst(__lasx_extract_hi128(_f0), p0 + out_hstep * 4, 0);
                p0 += 4;
            }
            if (out_elempack == 1)
            {
                int tmp[8];
                __lasx_xvst(_f0, tmp, 0);

                p0[0] = tmp[0];
                p0[out_hstep] = tmp[1];
                p0[out_hstep * 2] = tmp[2];
                p0[out_hstep * 3] = tmp[3];
                p0[out_hstep * 4] = tmp[4];
                p0[out_hstep * 5] = tmp[5];
                p0[out_hstep * 6] = tmp[6];
                p0[out_hstep * 7] = tmp[7];
                p0++;
            }

            pp += 8;
        }
    }
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        int* p0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _f0 = __lsx_vld(pp, 0);
            __m128i _f1 = __lsx_vld(pp + 4, 0);
            __m128i _f2 = __lsx_vld(pp + 8, 0);
            __m128i _f3 = __lsx_vld(pp + 12, 0);

            if (out_elempack == 4)
            {
                __lsx_vst(_f0, p0, 0);
                __lsx_vst(_f1, p0 + 4, 0);
                __lsx_vst(_f2, p0 + 8, 0);
                __lsx_vst(_f3, p0 + 12, 0);
                p0 += 16;
            }
            if (out_elempack == 1)
            {
                transpose4x4_ps((__m128&)_f0, (__m128&)_f1, (__m128&)_f2, (__m128&)_f3);

                __lsx_vst(_f0, p0, 0);
                __lsx_vst(_f1, p0 + out_hstep, 0);
                __lsx_vst(_f2, p0 + out_hstep * 2, 0);
                __lsx_vst(_f3, p0 + out_hstep * 3, 0);
                p0 += 4;
            }

            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128i _f0 = __lsx_vld(pp, 0);
            __m128i _f1 = __lsx_vld(pp + 4, 0);

            if (out_elempack == 4)
            {
                __lsx_vst(_f0, p0, 0);
                __lsx_vst(_f1, p0 + 4, 0);
                p0 += 8;
            }
            if (out_elempack == 1)
            {
                int tmp[8];
                __lsx_vst(_f0, tmp, 0);
                __lsx_vst(_f1, tmp + 4, 0);

                p0[0] = tmp[0];
                p0[out_hstep] = tmp[1];
                p0[out_hstep * 2] = tmp[2];
                p0[out_hstep * 3] = tmp[3];
                p0[1] = tmp[4];
                p0[out_hstep + 1] = tmp[5];
                p0[out_hstep * 2 + 1] = tmp[6];
                p0[out_hstep * 3 + 1] = tmp[7];
                p0 += 2;
            }

            pp += 8;
        }
        for (; jj < max_jj; jj++)
        {
            __m128i _f0 = __lsx_vld(pp, 0);

            if (out_elempack == 4)
            {
                __lsx_vst(_f0, p0, 0);
                p0 += 4;
            }
            if (out_elempack == 1)
            {
                int tmp[4];
                __lsx_vst(_f0, tmp, 0);

                p0[0] = tmp[0];
                p0[out_hstep] = tmp[1];
                p0[out_hstep * 2] = tmp[2];
                p0[out_hstep * 3] = tmp[3];
                p0++;
            }

            pp += 4;
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        int* p0 = (int*)top_blob + (i + ii) * out_hstep + j;

        int jj = 0;
#if __loongarch_sx
        for (; jj + 3 < max_jj; jj += 4)
        {
            p0[0] = pp[0];
            p0[1] = pp[2];
            p0[2] = pp[4];
            p0[3] = pp[6];
            p0[out_hstep] = pp[1];
            p0[out_hstep + 1] = pp[3];
            p0[out_hstep + 2] = pp[5];
            p0[out_hstep + 3] = pp[7];
            p0 += 4;

            pp += 8;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            p0[0] = pp[0];
            p0[1] = pp[2];
            p0[out_hstep] = pp[1];
            p0[out_hstep + 1] = pp[3];
            p0 += 2;

            pp += 4;
        }
        for (; jj < max_jj; jj++)
        {
            p0[0] = pp[0];
            p0[out_hstep] = pp[1];
            p0++;

            pp += 2;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        int* p0 = (int*)top_blob + (i + ii) * out_hstep + j;

        int jj = 0;
#if __loongarch_sx
        for (; jj + 3 < max_jj; jj += 4)
        {
            p0[0] = pp[0];
            p0[1] = pp[1];
            p0[2] = pp[2];
            p0[3] = pp[3];
            p0 += 4;

            pp += 4;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            p0[0] = pp[0];
            p0[1] = pp[1];
            p0 += 2;

            pp += 2;
        }
        for (; jj < max_jj; jj++)
        {
            p0[0] = pp[0];
            p0++;

            pp += 1;
        }
    }
}

static void convolution_im2col_gemm_get_optimal_tile_mnk_int8(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_int8 = (int)(get_cpu_level2_cache_size() / sizeof(signed char));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
#if __loongarch_sx
        int tile_size = (l2_cache_size_int8 - 16) / 8;
#else
        int tile_size = (l2_cache_size_int8 - 2) / 3;
#endif

#if __loongarch_sx
        TILE_K = std::max(4, tile_size / 4 * 4);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __loongarch_sx
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    // solve M
    {
#if __loongarch_sx
#if __loongarch_asx
        int nn_M = (M + 15) / 16;
#else
        int nn_M = (M + 7) / 8;
#endif
#else
        int nn_M = (M + 3) / 4;
#endif

#if __loongarch_sx
#if __loongarch_asx
        TILE_M = std::max(8, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::max(4, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#endif
#else
        TILE_M = std::max(2, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __loongarch_sx
#if __loongarch_asx
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#endif
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif

        if (nT > 1)
        {
#if __loongarch_sx
#if __loongarch_asx
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#endif
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }
    }

    if (N > 0)
    {
        int tile_size;
        if (TILE_K >= K)
        {
            tile_size = (l2_cache_size_int8 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_int8 - TILE_M * TILE_K) / (TILE_M + TILE_K);
        }

#if __loongarch_sx
        TILE_N = std::max(4, tile_size / 4 * 4);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __loongarch_sx
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif

#if __loongarch_sx
        TILE_N = std::max(4, TILE_N);
#else
        TILE_N = std::max(1, TILE_N);
#endif
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;

    signed char* pp = B;

    int jj = 0;
#if __loongarch_sx
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                for (int n = 0; n < 8; n++)
                {
                    pp[0] = p0[n];
                    pp[1] = p0[8 + n];
                    pp[2] = p0[16 + n];
                    pp[3] = p0[24 + n];
                    pp += 4;
                }
                p0 += bottom_blob.cstep * 8;
            }
        }
        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __loongarch_sx
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __loongarch_sx
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                for (int n = 0; n < 8; n++)
                {
                    pp[0] = p0[n];
                    pp[1] = p0[8 + n];
                    pp += 2;
                }
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __loongarch_sx
        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
#if __loongarch_sx
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                for (int n = 0; n < 8; n++)
                {
                    pp[0] = p0[n];
                    pp += 1;
                }
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __loongarch_sx
        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += bottom_blob.cstep;
            }
        }
    }
}

static inline void convolution_im2col_input_tile_impl_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    const int w = bottom_blob.w;
    // const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;

    // j max_jj     outw*outh    split w and h

    // k max_kk     pa*maxk*(inch/pa)    split inch

    // k/max_kk shall be multiple of maxk

    const int maxk = kernel_w * kernel_h;

    signed char* pp = B;

    int jj = 0;
#if __loongarch_sx
    for (; jj + 3 < max_jj; jj += 4)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;

        if (dy0 == dy3)
        {
            int kk = 0;
            if (elempack == 8)
            {
                for (; kk < max_kk; kk++)
                {
                    int raw_ch = kk / maxk;
                    int kpos = kk % maxk;
                    int p = raw_ch / 8;
                    int n = raw_ch % 8;
                    int u = kpos / kernel_w;
                    int v = kpos % kernel_w;

                    const Mat img = bottom_blob.channel(p);

                    int x0 = stride_w * dx0 + dilation_w * v;
                    int y0 = stride_h * dy0 + dilation_h * u;

                    const signed char* sptr = img.row<const signed char>(y0) + x0 * 8;

                    pp[0] = sptr[n];
                    pp[1] = sptr[stride_w * 8 + n];
                    pp[2] = sptr[stride_w * 16 + n];
                    pp[3] = sptr[stride_w * 24 + n];
                    pp += 4;
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp += 4;
                }
            }
        }
        else
        {
            int kk = 0;
            if (elempack == 8)
            {
                for (; kk < max_kk; kk++)
                {
                    int raw_ch = kk / maxk;
                    int kpos = kk % maxk;
                    int p = raw_ch / 8;
                    int n = raw_ch % 8;
                    int u = kpos / kernel_w;
                    int v = kpos % kernel_w;

                    const Mat img = bottom_blob.channel(p);

                    int x0 = stride_w * dx0 + dilation_w * v;
                    int x1 = stride_w * dx1 + dilation_w * v;
                    int x2 = stride_w * dx2 + dilation_w * v;
                    int x3 = stride_w * dx3 + dilation_w * v;

                    int y0 = stride_h * dy0 + dilation_h * u;
                    int y1 = stride_h * dy1 + dilation_h * u;
                    int y2 = stride_h * dy2 + dilation_h * u;
                    int y3 = stride_h * dy3 + dilation_h * u;

                    const signed char* sptr0 = img.row<const signed char>(y0) + x0 * 8;
                    const signed char* sptr1 = img.row<const signed char>(y1) + x1 * 8;
                    const signed char* sptr2 = img.row<const signed char>(y2) + x2 * 8;
                    const signed char* sptr3 = img.row<const signed char>(y3) + x3 * 8;

                    pp[0] = sptr0[n];
                    pp[1] = sptr1[n];
                    pp[2] = sptr2[n];
                    pp[3] = sptr3[n];
                    pp += 4;
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;

                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;

                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp += 4;
                }
            }
        }
    }
#endif // __loongarch_sx
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;

        if (dy0 == dy1)
        {
            int kk = 0;
#if __loongarch_sx
            if (elempack == 8)
            {
                for (; kk < max_kk; kk++)
                {
                    int raw_ch = kk / maxk;
                    int kpos = kk % maxk;
                    int p = raw_ch / 8;
                    int n = raw_ch % 8;
                    int u = kpos / kernel_w;
                    int v = kpos % kernel_w;

                    const Mat img = bottom_blob.channel(p);

                    int x0 = stride_w * dx0 + dilation_w * v;
                    int y0 = stride_h * dy0 + dilation_h * u;

                    const signed char* sptr = img.row<const signed char>(y0) + x0 * 8;

                    pp[0] = sptr[n];
                    pp[1] = sptr[stride_w * 8 + n];
                    pp += 2;
                }
            }
#endif // __loongarch_sx
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp += 2;
                }
            }
        }
        else
        {
            int kk = 0;
#if __loongarch_sx
            if (elempack == 8)
            {
                for (; kk < max_kk; kk++)
                {
                    int raw_ch = kk / maxk;
                    int kpos = kk % maxk;
                    int p = raw_ch / 8;
                    int n = raw_ch % 8;
                    int u = kpos / kernel_w;
                    int v = kpos % kernel_w;

                    const Mat img = bottom_blob.channel(p);

                    int x0 = stride_w * dx0 + dilation_w * v;
                    int x1 = stride_w * dx1 + dilation_w * v;
                    int y0 = stride_h * dy0 + dilation_h * u;
                    int y1 = stride_h * dy1 + dilation_h * u;

                    const signed char* sptr0 = img.row<const signed char>(y0) + x0 * 8;
                    const signed char* sptr1 = img.row<const signed char>(y1) + x1 * 8;

                    pp[0] = sptr0[n];
                    pp[1] = sptr1[n];
                    pp += 2;
                }
            }
#endif // __loongarch_sx
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;

                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp += 2;
                }
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw;
        int dx = (j + jj) % outw;

        int kk = 0;
#if __loongarch_sx
        if (elempack == 8)
        {
            for (; kk < max_kk; kk++)
            {
                int raw_ch = kk / maxk;
                int kpos = kk % maxk;
                int p = raw_ch / 8;
                int n = raw_ch % 8;
                int u = kpos / kernel_w;
                int v = kpos % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x = stride_w * dx + dilation_w * v;
                int y = stride_h * dy + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y) + x * 8;

                pp[0] = sptr[n];
                pp += 1;
            }
        }
#endif // __loongarch_sx
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x = stride_w * dx + dilation_w * v;
            int y = stride_h * dy + dilation_h * u;

            const signed char* sptr = img.row<const signed char>(y) + x * elempack;

            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

template<int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h>
void convolution_im2col_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    convolution_im2col_input_tile_impl_int8(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

template void convolution_im2col_input_tile_int8<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);

static void convolution_im2col_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_int8(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 1 && kernel_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_int8<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_int8<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_int8<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_int8<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_int8<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile_int8<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    convolution_im2col_input_tile_impl_int8(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

static void convolution_im2col_gemm_transform_kernel_int8(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel_int8");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_int8(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    if (maxk == 1)
    {
        A_data = kernel.reshape(maxk * inch, outch);
    }
    else
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch, (size_t)1u, 1);

        for (int q = 0; q < outch; q += 1)
        {
            signed char* g00 = A_data.row<signed char>(q);

            for (int p = 0; p < inch; p += 1)
            {
                for (int k = 0; k < maxk; k++)
                {
                    const signed char* k00 = weight_data_r2.channel(q).row<const signed char>(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)1u, 1);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile_int8(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static int convolution_im2col_gemm_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_int8(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 1u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        // im2col
        convolution_im2col_input_tile_int8(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                convolution_gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_int32(topT_tile, top_blob, i, max_ii, j, max_jj);
        }
    }

    return 0;
}
