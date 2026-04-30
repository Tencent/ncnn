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
    for (; ii + 7 < max_ii; ii += 8)
    {
        int* p0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256i _f0 = __lasx_xvld(pp, 0);
            __m256i _f1 = __lasx_xvld(pp + 8, 0);
            __m256i _f2 = __lasx_xvld(pp + 16, 0);
            __m256i _f3 = __lasx_xvld(pp + 24, 0);
            __m256i _f4 = __lasx_xvld(pp + 32, 0);
            __m256i _f5 = __lasx_xvld(pp + 40, 0);
            __m256i _f6 = __lasx_xvld(pp + 48, 0);
            __m256i _f7 = __lasx_xvld(pp + 56, 0);
            __m256i _f8 = __lasx_xvld(pp + 64, 0);
            __m256i _f9 = __lasx_xvld(pp + 72, 0);
            __m256i _fa = __lasx_xvld(pp + 80, 0);
            __m256i _fb = __lasx_xvld(pp + 88, 0);
            __m256i _fc = __lasx_xvld(pp + 96, 0);
            __m256i _fd = __lasx_xvld(pp + 104, 0);
            __m256i _fe = __lasx_xvld(pp + 112, 0);
            __m256i _ff = __lasx_xvld(pp + 120, 0);

            _f4 = __lasx_xvpermi_q(_f4, _f4, _LSX_SHUFFLE(0, 0, 0, 1));
            _f5 = __lasx_xvpermi_q(_f5, _f5, _LSX_SHUFFLE(0, 0, 0, 1));
            _f6 = __lasx_xvpermi_q(_f6, _f6, _LSX_SHUFFLE(0, 0, 0, 1));
            _f7 = __lasx_xvpermi_q(_f7, _f7, _LSX_SHUFFLE(0, 0, 0, 1));
            _fc = __lasx_xvpermi_q(_fc, _fc, _LSX_SHUFFLE(0, 0, 0, 1));
            _fd = __lasx_xvpermi_q(_fd, _fd, _LSX_SHUFFLE(0, 0, 0, 1));
            _fe = __lasx_xvpermi_q(_fe, _fe, _LSX_SHUFFLE(0, 0, 0, 1));
            _ff = __lasx_xvpermi_q(_ff, _ff, _LSX_SHUFFLE(0, 0, 0, 1));

            {
                __m256i _tmp0 = _f0;
                __m256i _tmp1 = __lasx_xvshuf4i_w(_f1, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp2 = _f2;
                __m256i _tmp3 = __lasx_xvshuf4i_w(_f3, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp4 = _f4;
                __m256i _tmp5 = __lasx_xvshuf4i_w(_f5, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp6 = _f6;
                __m256i _tmp7 = __lasx_xvshuf4i_w(_f7, _LSX_SHUFFLE(2, 1, 0, 3));

                _f0 = __lasx_xvilvl_w(_tmp3, _tmp0);
                _f1 = __lasx_xvilvh_w(_tmp3, _tmp0);
                _f2 = __lasx_xvilvl_w(_tmp1, _tmp2);
                _f3 = __lasx_xvilvh_w(_tmp1, _tmp2);
                _f4 = __lasx_xvilvl_w(_tmp7, _tmp4);
                _f5 = __lasx_xvilvh_w(_tmp7, _tmp4);
                _f6 = __lasx_xvilvl_w(_tmp5, _tmp6);
                _f7 = __lasx_xvilvh_w(_tmp5, _tmp6);

                _tmp0 = __lasx_xvilvl_d(_f2, _f0);
                _tmp1 = __lasx_xvilvh_d(_f2, _f0);
                _tmp2 = __lasx_xvilvl_d(_f1, _f3);
                _tmp3 = __lasx_xvilvh_d(_f1, _f3);
                _tmp4 = __lasx_xvilvl_d(_f6, _f4);
                _tmp5 = __lasx_xvilvh_d(_f6, _f4);
                _tmp6 = __lasx_xvilvl_d(_f5, _f7);
                _tmp7 = __lasx_xvilvh_d(_f5, _f7);

                _tmp1 = __lasx_xvshuf4i_w(_tmp1, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp3 = __lasx_xvshuf4i_w(_tmp3, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp5 = __lasx_xvshuf4i_w(_tmp5, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp7 = __lasx_xvshuf4i_w(_tmp7, _LSX_SHUFFLE(2, 1, 0, 3));

                _f0 = __lasx_xvpermi_q(_tmp4, _tmp0, _LSX_SHUFFLE(0, 3, 0, 0));
                _f1 = __lasx_xvpermi_q(_tmp5, _tmp1, _LSX_SHUFFLE(0, 3, 0, 0));
                _f2 = __lasx_xvpermi_q(_tmp6, _tmp2, _LSX_SHUFFLE(0, 3, 0, 0));
                _f3 = __lasx_xvpermi_q(_tmp7, _tmp3, _LSX_SHUFFLE(0, 3, 0, 0));
                _f4 = __lasx_xvpermi_q(_tmp0, _tmp4, _LSX_SHUFFLE(0, 3, 0, 0));
                _f5 = __lasx_xvpermi_q(_tmp1, _tmp5, _LSX_SHUFFLE(0, 3, 0, 0));
                _f6 = __lasx_xvpermi_q(_tmp2, _tmp6, _LSX_SHUFFLE(0, 3, 0, 0));
                _f7 = __lasx_xvpermi_q(_tmp3, _tmp7, _LSX_SHUFFLE(0, 3, 0, 0));
            }
            {
                __m256i _tmp0 = _f8;
                __m256i _tmp1 = __lasx_xvshuf4i_w(_f9, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp2 = _fa;
                __m256i _tmp3 = __lasx_xvshuf4i_w(_fb, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp4 = _fc;
                __m256i _tmp5 = __lasx_xvshuf4i_w(_fd, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp6 = _fe;
                __m256i _tmp7 = __lasx_xvshuf4i_w(_ff, _LSX_SHUFFLE(2, 1, 0, 3));

                _f8 = __lasx_xvilvl_w(_tmp3, _tmp0);
                _f9 = __lasx_xvilvh_w(_tmp3, _tmp0);
                _fa = __lasx_xvilvl_w(_tmp1, _tmp2);
                _fb = __lasx_xvilvh_w(_tmp1, _tmp2);
                _fc = __lasx_xvilvl_w(_tmp7, _tmp4);
                _fd = __lasx_xvilvh_w(_tmp7, _tmp4);
                _fe = __lasx_xvilvl_w(_tmp5, _tmp6);
                _ff = __lasx_xvilvh_w(_tmp5, _tmp6);

                _tmp0 = __lasx_xvilvl_d(_fa, _f8);
                _tmp1 = __lasx_xvilvh_d(_fa, _f8);
                _tmp2 = __lasx_xvilvl_d(_f9, _fb);
                _tmp3 = __lasx_xvilvh_d(_f9, _fb);
                _tmp4 = __lasx_xvilvl_d(_fe, _fc);
                _tmp5 = __lasx_xvilvh_d(_fe, _fc);
                _tmp6 = __lasx_xvilvl_d(_fd, _ff);
                _tmp7 = __lasx_xvilvh_d(_fd, _ff);

                _tmp1 = __lasx_xvshuf4i_w(_tmp1, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp3 = __lasx_xvshuf4i_w(_tmp3, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp5 = __lasx_xvshuf4i_w(_tmp5, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp7 = __lasx_xvshuf4i_w(_tmp7, _LSX_SHUFFLE(2, 1, 0, 3));

                _f8 = __lasx_xvpermi_q(_tmp4, _tmp0, _LSX_SHUFFLE(0, 3, 0, 0));
                _f9 = __lasx_xvpermi_q(_tmp5, _tmp1, _LSX_SHUFFLE(0, 3, 0, 0));
                _fa = __lasx_xvpermi_q(_tmp6, _tmp2, _LSX_SHUFFLE(0, 3, 0, 0));
                _fb = __lasx_xvpermi_q(_tmp7, _tmp3, _LSX_SHUFFLE(0, 3, 0, 0));
                _fc = __lasx_xvpermi_q(_tmp0, _tmp4, _LSX_SHUFFLE(0, 3, 0, 0));
                _fd = __lasx_xvpermi_q(_tmp1, _tmp5, _LSX_SHUFFLE(0, 3, 0, 0));
                _fe = __lasx_xvpermi_q(_tmp2, _tmp6, _LSX_SHUFFLE(0, 3, 0, 0));
                _ff = __lasx_xvpermi_q(_tmp3, _tmp7, _LSX_SHUFFLE(0, 3, 0, 0));
            }

            if (out_elempack == 4)
            {
                int* p1 = p0 + out_hstep * 4;
                __lasx_xvst(__lasx_xvpermi_q(_f1, _f0, _LSX_SHUFFLE(0, 2, 0, 0)), p0, 0);
                __lasx_xvst(__lasx_xvpermi_q(_f3, _f2, _LSX_SHUFFLE(0, 2, 0, 0)), p0 + 8, 0);
                __lasx_xvst(__lasx_xvpermi_q(_f5, _f4, _LSX_SHUFFLE(0, 2, 0, 0)), p0 + 16, 0);
                __lasx_xvst(__lasx_xvpermi_q(_f7, _f6, _LSX_SHUFFLE(0, 2, 0, 0)), p0 + 24, 0);
                __lasx_xvst(__lasx_xvpermi_q(_f9, _f8, _LSX_SHUFFLE(0, 2, 0, 0)), p0 + 32, 0);
                __lasx_xvst(__lasx_xvpermi_q(_fb, _fa, _LSX_SHUFFLE(0, 2, 0, 0)), p0 + 40, 0);
                __lasx_xvst(__lasx_xvpermi_q(_fd, _fc, _LSX_SHUFFLE(0, 2, 0, 0)), p0 + 48, 0);
                __lasx_xvst(__lasx_xvpermi_q(_ff, _fe, _LSX_SHUFFLE(0, 2, 0, 0)), p0 + 56, 0);
                __lasx_xvst(__lasx_xvpermi_q(_f1, _f0, _LSX_SHUFFLE(0, 3, 0, 1)), p1, 0);
                __lasx_xvst(__lasx_xvpermi_q(_f3, _f2, _LSX_SHUFFLE(0, 3, 0, 1)), p1 + 8, 0);
                __lasx_xvst(__lasx_xvpermi_q(_f5, _f4, _LSX_SHUFFLE(0, 3, 0, 1)), p1 + 16, 0);
                __lasx_xvst(__lasx_xvpermi_q(_f7, _f6, _LSX_SHUFFLE(0, 3, 0, 1)), p1 + 24, 0);
                __lasx_xvst(__lasx_xvpermi_q(_f9, _f8, _LSX_SHUFFLE(0, 3, 0, 1)), p1 + 32, 0);
                __lasx_xvst(__lasx_xvpermi_q(_fb, _fa, _LSX_SHUFFLE(0, 3, 0, 1)), p1 + 40, 0);
                __lasx_xvst(__lasx_xvpermi_q(_fd, _fc, _LSX_SHUFFLE(0, 3, 0, 1)), p1 + 48, 0);
                __lasx_xvst(__lasx_xvpermi_q(_ff, _fe, _LSX_SHUFFLE(0, 3, 0, 1)), p1 + 56, 0);
                p0 += 64;
            }
            if (out_elempack == 1)
            {
                transpose8x8_epi32(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                transpose8x8_epi32(_f8, _f9, _fa, _fb, _fc, _fd, _fe, _ff);
                __lasx_xvst(_f0, p0, 0);
                __lasx_xvst(_f1, p0 + out_hstep, 0);
                __lasx_xvst(_f2, p0 + out_hstep * 2, 0);
                __lasx_xvst(_f3, p0 + out_hstep * 3, 0);
                __lasx_xvst(_f4, p0 + out_hstep * 4, 0);
                __lasx_xvst(_f5, p0 + out_hstep * 5, 0);
                __lasx_xvst(_f6, p0 + out_hstep * 6, 0);
                __lasx_xvst(_f7, p0 + out_hstep * 7, 0);
                __lasx_xvst(_f8, p0 + 8, 0);
                __lasx_xvst(_f9, p0 + out_hstep + 8, 0);
                __lasx_xvst(_fa, p0 + out_hstep * 2 + 8, 0);
                __lasx_xvst(_fb, p0 + out_hstep * 3 + 8, 0);
                __lasx_xvst(_fc, p0 + out_hstep * 4 + 8, 0);
                __lasx_xvst(_fd, p0 + out_hstep * 5 + 8, 0);
                __lasx_xvst(_fe, p0 + out_hstep * 6 + 8, 0);
                __lasx_xvst(_ff, p0 + out_hstep * 7 + 8, 0);
                p0 += 16;
            }

            pp += 128;
        }
#endif // __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
#if __loongarch_asx
            __m256i _f0 = __lasx_xvld(pp, 0);
            __m256i _f1 = __lasx_xvld(pp + 8, 0);
            __m256i _f2 = __lasx_xvld(pp + 16, 0);
            __m256i _f3 = __lasx_xvld(pp + 24, 0);
            __m256i _f4 = __lasx_xvld(pp + 32, 0);
            __m256i _f5 = __lasx_xvld(pp + 40, 0);
            __m256i _f6 = __lasx_xvld(pp + 48, 0);
            __m256i _f7 = __lasx_xvld(pp + 56, 0);

            {
                __m256i _tmp0 = _f0;
                __m256i _tmp1 = __lasx_xvshuf4i_w(_f1, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp2 = _f2;
                __m256i _tmp3 = __lasx_xvshuf4i_w(_f3, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp4 = _f4;
                __m256i _tmp5 = __lasx_xvshuf4i_w(_f5, _LSX_SHUFFLE(2, 1, 0, 3));
                __m256i _tmp6 = _f6;
                __m256i _tmp7 = __lasx_xvshuf4i_w(_f7, _LSX_SHUFFLE(2, 1, 0, 3));

                _f0 = __lasx_xvilvl_w(_tmp3, _tmp0);
                _f1 = __lasx_xvilvh_w(_tmp3, _tmp0);
                _f2 = __lasx_xvilvl_w(_tmp1, _tmp2);
                _f3 = __lasx_xvilvh_w(_tmp1, _tmp2);
                _f4 = __lasx_xvilvl_w(_tmp7, _tmp4);
                _f5 = __lasx_xvilvh_w(_tmp7, _tmp4);
                _f6 = __lasx_xvilvl_w(_tmp5, _tmp6);
                _f7 = __lasx_xvilvh_w(_tmp5, _tmp6);

                _tmp0 = __lasx_xvilvl_d(_f2, _f0);
                _tmp1 = __lasx_xvilvh_d(_f2, _f0);
                _tmp2 = __lasx_xvilvl_d(_f1, _f3);
                _tmp3 = __lasx_xvilvh_d(_f1, _f3);
                _tmp4 = __lasx_xvilvl_d(_f6, _f4);
                _tmp5 = __lasx_xvilvh_d(_f6, _f4);
                _tmp6 = __lasx_xvilvl_d(_f5, _f7);
                _tmp7 = __lasx_xvilvh_d(_f5, _f7);

                _tmp1 = __lasx_xvshuf4i_w(_tmp1, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp3 = __lasx_xvshuf4i_w(_tmp3, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp5 = __lasx_xvshuf4i_w(_tmp5, _LSX_SHUFFLE(2, 1, 0, 3));
                _tmp7 = __lasx_xvshuf4i_w(_tmp7, _LSX_SHUFFLE(2, 1, 0, 3));

                _f0 = __lasx_xvpermi_q(_tmp4, _tmp0, _LSX_SHUFFLE(0, 3, 0, 0));
                _f1 = __lasx_xvpermi_q(_tmp5, _tmp1, _LSX_SHUFFLE(0, 3, 0, 0));
                _f2 = __lasx_xvpermi_q(_tmp6, _tmp2, _LSX_SHUFFLE(0, 3, 0, 0));
                _f3 = __lasx_xvpermi_q(_tmp7, _tmp3, _LSX_SHUFFLE(0, 3, 0, 0));
                _f4 = __lasx_xvpermi_q(_tmp0, _tmp4, _LSX_SHUFFLE(0, 3, 0, 0));
                _f5 = __lasx_xvpermi_q(_tmp1, _tmp5, _LSX_SHUFFLE(0, 3, 0, 0));
                _f6 = __lasx_xvpermi_q(_tmp2, _tmp6, _LSX_SHUFFLE(0, 3, 0, 0));
                _f7 = __lasx_xvpermi_q(_tmp3, _tmp7, _LSX_SHUFFLE(0, 3, 0, 0));
            }

            if (out_elempack == 4)
            {
                int* p1 = p0 + out_hstep * 4;
                __lsx_vst(__lasx_extract_128_lo(_f0), p0, 0);
                __lsx_vst(__lasx_extract_128_lo(_f1), p0 + 4, 0);
                __lsx_vst(__lasx_extract_128_lo(_f2), p0 + 8, 0);
                __lsx_vst(__lasx_extract_128_lo(_f3), p0 + 12, 0);
                __lsx_vst(__lasx_extract_128_lo(_f4), p0 + 16, 0);
                __lsx_vst(__lasx_extract_128_lo(_f5), p0 + 20, 0);
                __lsx_vst(__lasx_extract_128_lo(_f6), p0 + 24, 0);
                __lsx_vst(__lasx_extract_128_lo(_f7), p0 + 28, 0);
                __lsx_vst(__lasx_extract_128_hi(_f0), p1, 0);
                __lsx_vst(__lasx_extract_128_hi(_f1), p1 + 4, 0);
                __lsx_vst(__lasx_extract_128_hi(_f2), p1 + 8, 0);
                __lsx_vst(__lasx_extract_128_hi(_f3), p1 + 12, 0);
                __lsx_vst(__lasx_extract_128_hi(_f4), p1 + 16, 0);
                __lsx_vst(__lasx_extract_128_hi(_f5), p1 + 20, 0);
                __lsx_vst(__lasx_extract_128_hi(_f6), p1 + 24, 0);
                __lsx_vst(__lasx_extract_128_hi(_f7), p1 + 28, 0);
                p0 += 32;
            }
            if (out_elempack == 1)
            {
                transpose8x8_epi32(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                __lasx_xvst(_f0, p0, 0);
                __lasx_xvst(_f1, p0 + out_hstep, 0);
                __lasx_xvst(_f2, p0 + out_hstep * 2, 0);
                __lasx_xvst(_f3, p0 + out_hstep * 3, 0);
                __lasx_xvst(_f4, p0 + out_hstep * 4, 0);
                __lasx_xvst(_f5, p0 + out_hstep * 5, 0);
                __lasx_xvst(_f6, p0 + out_hstep * 6, 0);
                __lasx_xvst(_f7, p0 + out_hstep * 7, 0);
                p0 += 8;
            }

            pp += 64;
#else  // __loongarch_asx
            __m128i _f00 = __lsx_vld(pp, 0);
            __m128i _f01 = __lsx_vld(pp + 4, 0);
            __m128i _f10 = __lsx_vld(pp + 8, 0);
            __m128i _f11 = __lsx_vld(pp + 12, 0);
            __m128i _f20 = __lsx_vld(pp + 16, 0);
            __m128i _f21 = __lsx_vld(pp + 20, 0);
            __m128i _f30 = __lsx_vld(pp + 24, 0);
            __m128i _f31 = __lsx_vld(pp + 28, 0);
            __m128i _f40 = __lsx_vld(pp + 32, 0);
            __m128i _f41 = __lsx_vld(pp + 36, 0);
            __m128i _f50 = __lsx_vld(pp + 40, 0);
            __m128i _f51 = __lsx_vld(pp + 44, 0);
            __m128i _f60 = __lsx_vld(pp + 48, 0);
            __m128i _f61 = __lsx_vld(pp + 52, 0);
            __m128i _f70 = __lsx_vld(pp + 56, 0);
            __m128i _f71 = __lsx_vld(pp + 60, 0);

            _f20 = __lsx_vshuf4i_w(_f20, _LSX_SHUFFLE(1, 0, 3, 2));
            _f30 = __lsx_vshuf4i_w(_f30, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_f00, _f10, _f20, _f30);
            _f10 = __lsx_vshuf4i_w(_f10, _LSX_SHUFFLE(2, 1, 0, 3));
            _f20 = __lsx_vshuf4i_w(_f20, _LSX_SHUFFLE(1, 0, 3, 2));
            _f30 = __lsx_vshuf4i_w(_f30, _LSX_SHUFFLE(0, 3, 2, 1));

            _f21 = __lsx_vshuf4i_w(_f21, _LSX_SHUFFLE(1, 0, 3, 2));
            _f31 = __lsx_vshuf4i_w(_f31, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_f01, _f11, _f21, _f31);
            _f11 = __lsx_vshuf4i_w(_f11, _LSX_SHUFFLE(2, 1, 0, 3));
            _f21 = __lsx_vshuf4i_w(_f21, _LSX_SHUFFLE(1, 0, 3, 2));
            _f31 = __lsx_vshuf4i_w(_f31, _LSX_SHUFFLE(0, 3, 2, 1));

            _f60 = __lsx_vshuf4i_w(_f60, _LSX_SHUFFLE(1, 0, 3, 2));
            _f70 = __lsx_vshuf4i_w(_f70, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_f40, _f50, _f60, _f70);
            _f50 = __lsx_vshuf4i_w(_f50, _LSX_SHUFFLE(2, 1, 0, 3));
            _f60 = __lsx_vshuf4i_w(_f60, _LSX_SHUFFLE(1, 0, 3, 2));
            _f70 = __lsx_vshuf4i_w(_f70, _LSX_SHUFFLE(0, 3, 2, 1));

            _f61 = __lsx_vshuf4i_w(_f61, _LSX_SHUFFLE(1, 0, 3, 2));
            _f71 = __lsx_vshuf4i_w(_f71, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_f41, _f51, _f61, _f71);
            _f51 = __lsx_vshuf4i_w(_f51, _LSX_SHUFFLE(2, 1, 0, 3));
            _f61 = __lsx_vshuf4i_w(_f61, _LSX_SHUFFLE(1, 0, 3, 2));
            _f71 = __lsx_vshuf4i_w(_f71, _LSX_SHUFFLE(0, 3, 2, 1));

            if (out_elempack == 4)
            {
                transpose4x4_epi32(_f00, _f10, _f20, _f30);
                transpose4x4_epi32(_f01, _f11, _f21, _f31);
                transpose4x4_epi32(_f40, _f50, _f60, _f70);
                transpose4x4_epi32(_f41, _f51, _f61, _f71);

                __lsx_vst(_f00, p0, 0);
                __lsx_vst(_f10, p0 + 4, 0);
                __lsx_vst(_f20, p0 + 8, 0);
                __lsx_vst(_f30, p0 + 12, 0);
                __lsx_vst(_f40, p0 + 16, 0);
                __lsx_vst(_f50, p0 + 20, 0);
                __lsx_vst(_f60, p0 + 24, 0);
                __lsx_vst(_f70, p0 + 28, 0);
                __lsx_vst(_f01, p0 + out_hstep * 4, 0);
                __lsx_vst(_f11, p0 + out_hstep * 4 + 4, 0);
                __lsx_vst(_f21, p0 + out_hstep * 4 + 8, 0);
                __lsx_vst(_f31, p0 + out_hstep * 4 + 12, 0);
                __lsx_vst(_f41, p0 + out_hstep * 4 + 16, 0);
                __lsx_vst(_f51, p0 + out_hstep * 4 + 20, 0);
                __lsx_vst(_f61, p0 + out_hstep * 4 + 24, 0);
                __lsx_vst(_f71, p0 + out_hstep * 4 + 28, 0);
                p0 += 32;
            }
            if (out_elempack == 1)
            {
                __lsx_vst(_f00, p0, 0);
                __lsx_vst(_f10, p0 + out_hstep, 0);
                __lsx_vst(_f20, p0 + out_hstep * 2, 0);
                __lsx_vst(_f30, p0 + out_hstep * 3, 0);
                __lsx_vst(_f01, p0 + out_hstep * 4, 0);
                __lsx_vst(_f11, p0 + out_hstep * 5, 0);
                __lsx_vst(_f21, p0 + out_hstep * 6, 0);
                __lsx_vst(_f31, p0 + out_hstep * 7, 0);
                __lsx_vst(_f40, p0 + 4, 0);
                __lsx_vst(_f50, p0 + out_hstep + 4, 0);
                __lsx_vst(_f60, p0 + out_hstep * 2 + 4, 0);
                __lsx_vst(_f70, p0 + out_hstep * 3 + 4, 0);
                __lsx_vst(_f41, p0 + out_hstep * 4 + 4, 0);
                __lsx_vst(_f51, p0 + out_hstep * 5 + 4, 0);
                __lsx_vst(_f61, p0 + out_hstep * 6 + 4, 0);
                __lsx_vst(_f71, p0 + out_hstep * 7 + 4, 0);
                p0 += 8;
            }

            pp += 64;
#endif // __loongarch_asx
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _f00 = __lsx_vld(pp, 0);
            __m128i _f01 = __lsx_vld(pp + 4, 0);
            __m128i _f10 = __lsx_vld(pp + 8, 0);
            __m128i _f11 = __lsx_vld(pp + 12, 0);
            __m128i _f20 = __lsx_vld(pp + 16, 0);
            __m128i _f21 = __lsx_vld(pp + 20, 0);
            __m128i _f30 = __lsx_vld(pp + 24, 0);
            __m128i _f31 = __lsx_vld(pp + 28, 0);

            _f20 = __lsx_vshuf4i_w(_f20, _LSX_SHUFFLE(1, 0, 3, 2));
            _f30 = __lsx_vshuf4i_w(_f30, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_f00, _f10, _f20, _f30);
            _f10 = __lsx_vshuf4i_w(_f10, _LSX_SHUFFLE(2, 1, 0, 3));
            _f20 = __lsx_vshuf4i_w(_f20, _LSX_SHUFFLE(1, 0, 3, 2));
            _f30 = __lsx_vshuf4i_w(_f30, _LSX_SHUFFLE(0, 3, 2, 1));

            _f21 = __lsx_vshuf4i_w(_f21, _LSX_SHUFFLE(1, 0, 3, 2));
            _f31 = __lsx_vshuf4i_w(_f31, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_f01, _f11, _f21, _f31);
            _f11 = __lsx_vshuf4i_w(_f11, _LSX_SHUFFLE(2, 1, 0, 3));
            _f21 = __lsx_vshuf4i_w(_f21, _LSX_SHUFFLE(1, 0, 3, 2));
            _f31 = __lsx_vshuf4i_w(_f31, _LSX_SHUFFLE(0, 3, 2, 1));

            if (out_elempack == 4)
            {
                transpose4x4_epi32(_f00, _f10, _f20, _f30);
                transpose4x4_epi32(_f01, _f11, _f21, _f31);

                __lsx_vst(_f00, p0, 0);
                __lsx_vst(_f10, p0 + 4, 0);
                __lsx_vst(_f20, p0 + 8, 0);
                __lsx_vst(_f30, p0 + 12, 0);
                __lsx_vst(_f01, p0 + out_hstep * 4, 0);
                __lsx_vst(_f11, p0 + out_hstep * 4 + 4, 0);
                __lsx_vst(_f21, p0 + out_hstep * 4 + 8, 0);
                __lsx_vst(_f31, p0 + out_hstep * 4 + 12, 0);
                p0 += 16;
            }
            if (out_elempack == 1)
            {
                __lsx_vst(_f00, p0, 0);
                __lsx_vst(_f10, p0 + out_hstep, 0);
                __lsx_vst(_f20, p0 + out_hstep * 2, 0);
                __lsx_vst(_f30, p0 + out_hstep * 3, 0);
                __lsx_vst(_f01, p0 + out_hstep * 4, 0);
                __lsx_vst(_f11, p0 + out_hstep * 5, 0);
                __lsx_vst(_f21, p0 + out_hstep * 6, 0);
                __lsx_vst(_f31, p0 + out_hstep * 7, 0);
                p0 += 4;
            }

            pp += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            if (out_elempack == 4)
            {
                p0[0] = pp[0];
                p0[1] = pp[9];
                p0[2] = pp[2];
                p0[3] = pp[11];
                p0[4] = pp[8];
                p0[5] = pp[1];
                p0[6] = pp[10];
                p0[7] = pp[3];
                p0[out_hstep * 4] = pp[4];
                p0[out_hstep * 4 + 1] = pp[13];
                p0[out_hstep * 4 + 2] = pp[6];
                p0[out_hstep * 4 + 3] = pp[15];
                p0[out_hstep * 4 + 4] = pp[12];
                p0[out_hstep * 4 + 5] = pp[5];
                p0[out_hstep * 4 + 6] = pp[14];
                p0[out_hstep * 4 + 7] = pp[7];
                p0 += 8;
            }
            if (out_elempack == 1)
            {
                p0[0] = pp[0];
                p0[1] = pp[8];
                p0[out_hstep] = pp[9];
                p0[out_hstep + 1] = pp[1];
                p0[out_hstep * 2] = pp[2];
                p0[out_hstep * 2 + 1] = pp[10];
                p0[out_hstep * 3] = pp[11];
                p0[out_hstep * 3 + 1] = pp[3];
                p0[out_hstep * 4] = pp[4];
                p0[out_hstep * 4 + 1] = pp[12];
                p0[out_hstep * 5] = pp[13];
                p0[out_hstep * 5 + 1] = pp[5];
                p0[out_hstep * 6] = pp[6];
                p0[out_hstep * 6 + 1] = pp[14];
                p0[out_hstep * 7] = pp[15];
                p0[out_hstep * 7 + 1] = pp[7];
                p0 += 2;
            }

            pp += 16;
        }
        for (; jj < max_jj; jj++)
        {
            __m128i _f00 = __lsx_vld(pp, 0);
            __m128i _f01 = __lsx_vld(pp + 4, 0);

            if (out_elempack == 4)
            {
                __lsx_vst(_f00, p0, 0);
                __lsx_vst(_f01, p0 + out_hstep * 4, 0);
                p0 += 4;
            }
            if (out_elempack == 1)
            {
                p0[0] = __lsx_vpickve2gr_w(_f00, 0);
                p0[out_hstep] = __lsx_vpickve2gr_w(_f00, 1);
                p0[out_hstep * 2] = __lsx_vpickve2gr_w(_f00, 2);
                p0[out_hstep * 3] = __lsx_vpickve2gr_w(_f00, 3);
                p0[out_hstep * 4] = __lsx_vpickve2gr_w(_f01, 0);
                p0[out_hstep * 5] = __lsx_vpickve2gr_w(_f01, 1);
                p0[out_hstep * 6] = __lsx_vpickve2gr_w(_f01, 2);
                p0[out_hstep * 7] = __lsx_vpickve2gr_w(_f01, 3);
                p0++;
            }

            pp += 8;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        int* p0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        int jj = 0;
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256i _f04 = __lasx_xvld(pp, 0);
            __m256i _f15 = __lasx_xvld(pp + 8, 0);
            __m256i _f26 = __lasx_xvld(pp + 16, 0);
            __m256i _f37 = __lasx_xvld(pp + 24, 0);
            __m256i _f8c = __lasx_xvld(pp + 32, 0);
            __m256i _f9d = __lasx_xvld(pp + 40, 0);
            __m256i _fae = __lasx_xvld(pp + 48, 0);
            __m256i _fbf = __lasx_xvld(pp + 56, 0);

            _f26 = __lasx_xvshuf4i_w(_f26, _LSX_SHUFFLE(1, 0, 3, 2));
            _f37 = __lasx_xvshuf4i_w(_f37, _LSX_SHUFFLE(1, 0, 3, 2));
            {
                __m256i _tmp0 = __lasx_xvilvl_w(_f15, _f04);
                __m256i _tmp1 = __lasx_xvilvh_w(_f15, _f04);
                __m256i _tmp2 = __lasx_xvilvl_w(_f37, _f26);
                __m256i _tmp3 = __lasx_xvilvh_w(_f37, _f26);
                _f04 = __lasx_xvilvl_d(_tmp2, _tmp0);
                _f15 = __lasx_xvilvh_d(_tmp2, _tmp0);
                _f26 = __lasx_xvilvl_d(_tmp3, _tmp1);
                _f37 = __lasx_xvilvh_d(_tmp3, _tmp1);
            }
            _f15 = __lasx_xvshuf4i_w(_f15, _LSX_SHUFFLE(2, 1, 0, 3));
            _f26 = __lasx_xvshuf4i_w(_f26, _LSX_SHUFFLE(1, 0, 3, 2));
            _f37 = __lasx_xvshuf4i_w(_f37, _LSX_SHUFFLE(0, 3, 2, 1));

            _fae = __lasx_xvshuf4i_w(_fae, _LSX_SHUFFLE(1, 0, 3, 2));
            _fbf = __lasx_xvshuf4i_w(_fbf, _LSX_SHUFFLE(1, 0, 3, 2));
            {
                __m256i _tmp0 = __lasx_xvilvl_w(_f9d, _f8c);
                __m256i _tmp1 = __lasx_xvilvh_w(_f9d, _f8c);
                __m256i _tmp2 = __lasx_xvilvl_w(_fbf, _fae);
                __m256i _tmp3 = __lasx_xvilvh_w(_fbf, _fae);
                _f8c = __lasx_xvilvl_d(_tmp2, _tmp0);
                _f9d = __lasx_xvilvh_d(_tmp2, _tmp0);
                _fae = __lasx_xvilvl_d(_tmp3, _tmp1);
                _fbf = __lasx_xvilvh_d(_tmp3, _tmp1);
            }
            _f9d = __lasx_xvshuf4i_w(_f9d, _LSX_SHUFFLE(2, 1, 0, 3));
            _fae = __lasx_xvshuf4i_w(_fae, _LSX_SHUFFLE(1, 0, 3, 2));
            _fbf = __lasx_xvshuf4i_w(_fbf, _LSX_SHUFFLE(0, 3, 2, 1));

            if (out_elempack == 4)
            {
                {
                    __m256i _tmp0 = __lasx_xvilvl_w(_f15, _f04);
                    __m256i _tmp1 = __lasx_xvilvh_w(_f15, _f04);
                    __m256i _tmp2 = __lasx_xvilvl_w(_f37, _f26);
                    __m256i _tmp3 = __lasx_xvilvh_w(_f37, _f26);
                    __m256i _r0 = __lasx_xvilvl_d(_tmp2, _tmp0);
                    __m256i _r1 = __lasx_xvilvh_d(_tmp2, _tmp0);
                    __m256i _r2 = __lasx_xvilvl_d(_tmp3, _tmp1);
                    __m256i _r3 = __lasx_xvilvh_d(_tmp3, _tmp1);
                    __lasx_xvst(__lasx_xvpermi_q(_r1, _r0, _LSX_SHUFFLE(0, 2, 0, 0)), p0, 0);
                    __lasx_xvst(__lasx_xvpermi_q(_r3, _r2, _LSX_SHUFFLE(0, 2, 0, 0)), p0 + 8, 0);
                    __lasx_xvst(__lasx_xvpermi_q(_r1, _r0, _LSX_SHUFFLE(0, 3, 0, 1)), p0 + 16, 0);
                    __lasx_xvst(__lasx_xvpermi_q(_r3, _r2, _LSX_SHUFFLE(0, 3, 0, 1)), p0 + 24, 0);
                }
                {
                    __m256i _tmp0 = __lasx_xvilvl_w(_f9d, _f8c);
                    __m256i _tmp1 = __lasx_xvilvh_w(_f9d, _f8c);
                    __m256i _tmp2 = __lasx_xvilvl_w(_fbf, _fae);
                    __m256i _tmp3 = __lasx_xvilvh_w(_fbf, _fae);
                    __m256i _r0 = __lasx_xvilvl_d(_tmp2, _tmp0);
                    __m256i _r1 = __lasx_xvilvh_d(_tmp2, _tmp0);
                    __m256i _r2 = __lasx_xvilvl_d(_tmp3, _tmp1);
                    __m256i _r3 = __lasx_xvilvh_d(_tmp3, _tmp1);
                    __lasx_xvst(__lasx_xvpermi_q(_r1, _r0, _LSX_SHUFFLE(0, 2, 0, 0)), p0 + 32, 0);
                    __lasx_xvst(__lasx_xvpermi_q(_r3, _r2, _LSX_SHUFFLE(0, 2, 0, 0)), p0 + 40, 0);
                    __lasx_xvst(__lasx_xvpermi_q(_r1, _r0, _LSX_SHUFFLE(0, 3, 0, 1)), p0 + 48, 0);
                    __lasx_xvst(__lasx_xvpermi_q(_r3, _r2, _LSX_SHUFFLE(0, 3, 0, 1)), p0 + 56, 0);
                }
                p0 += 64;
            }
            if (out_elempack == 1)
            {
                __lasx_xvst(_f04, p0, 0);
                __lasx_xvst(_f8c, p0 + 8, 0);
                __lasx_xvst(_f15, p0 + out_hstep, 0);
                __lasx_xvst(_f9d, p0 + out_hstep + 8, 0);
                __lasx_xvst(_f26, p0 + out_hstep * 2, 0);
                __lasx_xvst(_fae, p0 + out_hstep * 2 + 8, 0);
                __lasx_xvst(_f37, p0 + out_hstep * 3, 0);
                __lasx_xvst(_fbf, p0 + out_hstep * 3 + 8, 0);
                p0 += 16;
            }

            pp += 64;
        }
#endif // __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _f0 = __lsx_vld(pp, 0);
            __m128i _f1 = __lsx_vld(pp + 4, 0);
            __m128i _f2 = __lsx_vld(pp + 8, 0);
            __m128i _f3 = __lsx_vld(pp + 12, 0);
            __m128i _f4 = __lsx_vld(pp + 16, 0);
            __m128i _f5 = __lsx_vld(pp + 20, 0);
            __m128i _f6 = __lsx_vld(pp + 24, 0);
            __m128i _f7 = __lsx_vld(pp + 28, 0);

            _f2 = __lsx_vshuf4i_w(_f2, _LSX_SHUFFLE(1, 0, 3, 2));
            _f3 = __lsx_vshuf4i_w(_f3, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_f0, _f1, _f2, _f3);
            _f1 = __lsx_vshuf4i_w(_f1, _LSX_SHUFFLE(2, 1, 0, 3));
            _f2 = __lsx_vshuf4i_w(_f2, _LSX_SHUFFLE(1, 0, 3, 2));
            _f3 = __lsx_vshuf4i_w(_f3, _LSX_SHUFFLE(0, 3, 2, 1));

            _f6 = __lsx_vshuf4i_w(_f6, _LSX_SHUFFLE(1, 0, 3, 2));
            _f7 = __lsx_vshuf4i_w(_f7, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_f4, _f5, _f6, _f7);
            _f5 = __lsx_vshuf4i_w(_f5, _LSX_SHUFFLE(2, 1, 0, 3));
            _f6 = __lsx_vshuf4i_w(_f6, _LSX_SHUFFLE(1, 0, 3, 2));
            _f7 = __lsx_vshuf4i_w(_f7, _LSX_SHUFFLE(0, 3, 2, 1));

            if (out_elempack == 4)
            {
                transpose4x4_epi32(_f0, _f1, _f2, _f3);
                transpose4x4_epi32(_f4, _f5, _f6, _f7);

                __lsx_vst(_f0, p0, 0);
                __lsx_vst(_f1, p0 + 4, 0);
                __lsx_vst(_f2, p0 + 8, 0);
                __lsx_vst(_f3, p0 + 12, 0);
                __lsx_vst(_f4, p0 + 16, 0);
                __lsx_vst(_f5, p0 + 20, 0);
                __lsx_vst(_f6, p0 + 24, 0);
                __lsx_vst(_f7, p0 + 28, 0);
                p0 += 32;
            }
            if (out_elempack == 1)
            {
                __lsx_vst(_f0, p0, 0);
                __lsx_vst(_f1, p0 + out_hstep, 0);
                __lsx_vst(_f2, p0 + out_hstep * 2, 0);
                __lsx_vst(_f3, p0 + out_hstep * 3, 0);
                __lsx_vst(_f4, p0 + 4, 0);
                __lsx_vst(_f5, p0 + out_hstep + 4, 0);
                __lsx_vst(_f6, p0 + out_hstep * 2 + 4, 0);
                __lsx_vst(_f7, p0 + out_hstep * 3 + 4, 0);
                p0 += 8;
            }

            pp += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _f0 = __lsx_vld(pp, 0);
            __m128i _f1 = __lsx_vld(pp + 4, 0);
            __m128i _f2 = __lsx_vld(pp + 8, 0);
            __m128i _f3 = __lsx_vld(pp + 12, 0);

            _f2 = __lsx_vshuf4i_w(_f2, _LSX_SHUFFLE(1, 0, 3, 2));
            _f3 = __lsx_vshuf4i_w(_f3, _LSX_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_f0, _f1, _f2, _f3);
            _f1 = __lsx_vshuf4i_w(_f1, _LSX_SHUFFLE(2, 1, 0, 3));
            _f2 = __lsx_vshuf4i_w(_f2, _LSX_SHUFFLE(1, 0, 3, 2));
            _f3 = __lsx_vshuf4i_w(_f3, _LSX_SHUFFLE(0, 3, 2, 1));

            if (out_elempack == 4)
            {
                transpose4x4_epi32(_f0, _f1, _f2, _f3);

                __lsx_vst(_f0, p0, 0);
                __lsx_vst(_f1, p0 + 4, 0);
                __lsx_vst(_f2, p0 + 8, 0);
                __lsx_vst(_f3, p0 + 12, 0);
                p0 += 16;
            }
            if (out_elempack == 1)
            {
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
            if (out_elempack == 4)
            {
                p0[0] = pp[0];
                p0[1] = pp[5];
                p0[2] = pp[2];
                p0[3] = pp[7];
                p0[4] = pp[4];
                p0[5] = pp[1];
                p0[6] = pp[6];
                p0[7] = pp[3];
                p0 += 8;
            }
            if (out_elempack == 1)
            {
                p0[0] = pp[0];
                p0[1] = pp[4];
                p0[out_hstep] = pp[5];
                p0[out_hstep + 1] = pp[1];
                p0[out_hstep * 2] = pp[2];
                p0[out_hstep * 2 + 1] = pp[6];
                p0[out_hstep * 3] = pp[7];
                p0[out_hstep * 3 + 1] = pp[3];
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
                p0[0] = __lsx_vpickve2gr_w(_f0, 0);
                p0[out_hstep] = __lsx_vpickve2gr_w(_f0, 1);
                p0[out_hstep * 2] = __lsx_vpickve2gr_w(_f0, 2);
                p0[out_hstep * 3] = __lsx_vpickve2gr_w(_f0, 3);
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
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m256i _sum0 = __lasx_xvld(pp, 0);
            __m256i _sum1 = __lasx_xvld(pp + 8, 0);
            __m256i _sum2 = __lasx_xvld(pp + 16, 0);
            __m256i _sum3 = __lasx_xvld(pp + 24, 0);

            __m256i _s0 = __lasx_xvilvl_w(_sum1, _sum0);
            __m256i _s1 = __lasx_xvilvh_w(_sum1, _sum0);
            __m256i _s2 = __lasx_xvilvl_w(_sum3, _sum2);
            __m256i _s3 = __lasx_xvilvh_w(_sum3, _sum2);

            __m256i _t0 = __lasx_xvpermi_q(_s1, _s0, _LSX_SHUFFLE(0, 2, 0, 0));
            __m256i _t1 = __lasx_xvpermi_q(_s1, _s0, _LSX_SHUFFLE(0, 3, 0, 1));
            __m256i _t2 = __lasx_xvpermi_q(_s3, _s2, _LSX_SHUFFLE(0, 2, 0, 0));
            __m256i _t3 = __lasx_xvpermi_q(_s3, _s2, _LSX_SHUFFLE(0, 3, 0, 1));

            __m256i _f01 = __lasx_xvpermi_d(__lasx_xvpickev_w(_t1, _t0), _LSX_SHUFFLE(3, 1, 2, 0));
            __m256i _f23 = __lasx_xvpermi_d(__lasx_xvpickod_w(_t1, _t0), _LSX_SHUFFLE(3, 1, 2, 0));
            __m256i _f45 = __lasx_xvpermi_d(__lasx_xvpickev_w(_t3, _t2), _LSX_SHUFFLE(3, 1, 2, 0));
            __m256i _f67 = __lasx_xvpermi_d(__lasx_xvpickod_w(_t3, _t2), _LSX_SHUFFLE(3, 1, 2, 0));

            __lasx_xvst(_f01, p0, 0);
            __lasx_xvst(_f45, p0 + 8, 0);
            __lasx_xvst(_f23, p0 + out_hstep, 0);
            __lasx_xvst(_f67, p0 + out_hstep + 8, 0);
            p0 += 16;

            pp += 32;
        }
#endif // __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            p0[0] = pp[0];
            p0[1] = pp[2];
            p0[2] = pp[4];
            p0[3] = pp[6];
            p0[4] = pp[8];
            p0[5] = pp[10];
            p0[6] = pp[12];
            p0[7] = pp[14];
            p0[out_hstep] = pp[1];
            p0[out_hstep + 1] = pp[3];
            p0[out_hstep + 2] = pp[5];
            p0[out_hstep + 3] = pp[7];
            p0[out_hstep + 4] = pp[9];
            p0[out_hstep + 5] = pp[11];
            p0[out_hstep + 6] = pp[13];
            p0[out_hstep + 7] = pp[15];
            p0 += 8;

            pp += 16;
        }
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
#if __loongarch_asx
        for (; jj + 15 < max_jj; jj += 16)
        {
            __lasx_xvst(__lasx_xvld(pp, 0), p0, 0);
            __lasx_xvst(__lasx_xvld(pp + 8, 0), p0 + 8, 0);
            p0 += 16;

            pp += 16;
        }
#endif // __loongarch_asx
        for (; jj + 7 < max_jj; jj += 8)
        {
            p0[0] = pp[0];
            p0[1] = pp[1];
            p0[2] = pp[2];
            p0[3] = pp[3];
            p0[4] = pp[4];
            p0[5] = pp[5];
            p0[6] = pp[6];
            p0[7] = pp[7];
            p0 += 8;

            pp += 8;
        }
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
#if __loongarch_asx
    for (; jj + 15 < max_jj; jj += 16)
    {
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                for (int n = 0; n < 8; n += 4)
                {
                    pp[0] = p0[n];
                    pp[1] = p0[n + 1];
                    pp[2] = p0[n + 2];
                    pp[3] = p0[n + 3];
                    pp[4] = p0[8 + n];
                    pp[5] = p0[8 + n + 1];
                    pp[6] = p0[8 + n + 2];
                    pp[7] = p0[8 + n + 3];
                    pp[8] = p0[16 + n];
                    pp[9] = p0[16 + n + 1];
                    pp[10] = p0[16 + n + 2];
                    pp[11] = p0[16 + n + 3];
                    pp[12] = p0[24 + n];
                    pp[13] = p0[24 + n + 1];
                    pp[14] = p0[24 + n + 2];
                    pp[15] = p0[24 + n + 3];
                    pp[16] = p0[32 + n];
                    pp[17] = p0[32 + n + 1];
                    pp[18] = p0[32 + n + 2];
                    pp[19] = p0[32 + n + 3];
                    pp[20] = p0[40 + n];
                    pp[21] = p0[40 + n + 1];
                    pp[22] = p0[40 + n + 2];
                    pp[23] = p0[40 + n + 3];
                    pp[24] = p0[48 + n];
                    pp[25] = p0[48 + n + 1];
                    pp[26] = p0[48 + n + 2];
                    pp[27] = p0[48 + n + 3];
                    pp[28] = p0[56 + n];
                    pp[29] = p0[56 + n + 1];
                    pp[30] = p0[56 + n + 2];
                    pp[31] = p0[56 + n + 3];
                    pp[32] = p0[64 + n];
                    pp[33] = p0[64 + n + 1];
                    pp[34] = p0[64 + n + 2];
                    pp[35] = p0[64 + n + 3];
                    pp[36] = p0[72 + n];
                    pp[37] = p0[72 + n + 1];
                    pp[38] = p0[72 + n + 2];
                    pp[39] = p0[72 + n + 3];
                    pp[40] = p0[80 + n];
                    pp[41] = p0[80 + n + 1];
                    pp[42] = p0[80 + n + 2];
                    pp[43] = p0[80 + n + 3];
                    pp[44] = p0[88 + n];
                    pp[45] = p0[88 + n + 1];
                    pp[46] = p0[88 + n + 2];
                    pp[47] = p0[88 + n + 3];
                    pp[48] = p0[96 + n];
                    pp[49] = p0[96 + n + 1];
                    pp[50] = p0[96 + n + 2];
                    pp[51] = p0[96 + n + 3];
                    pp[52] = p0[104 + n];
                    pp[53] = p0[104 + n + 1];
                    pp[54] = p0[104 + n + 2];
                    pp[55] = p0[104 + n + 3];
                    pp[56] = p0[112 + n];
                    pp[57] = p0[112 + n + 1];
                    pp[58] = p0[112 + n + 2];
                    pp[59] = p0[112 + n + 3];
                    pp[60] = p0[120 + n];
                    pp[61] = p0[120 + n + 1];
                    pp[62] = p0[120 + n + 2];
                    pp[63] = p0[120 + n + 3];
                    pp += 64;
                }
                p0 += bottom_blob.cstep * 8;
            }
        }
        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const signed char* p1 = p0 + bottom_blob.cstep;
                const signed char* p2 = p0 + bottom_blob.cstep * 2;
                const signed char* p3 = p0 + bottom_blob.cstep * 3;

                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p0[1];
                pp[5] = p1[1];
                pp[6] = p2[1];
                pp[7] = p3[1];
                pp[8] = p0[2];
                pp[9] = p1[2];
                pp[10] = p2[2];
                pp[11] = p3[2];
                pp[12] = p0[3];
                pp[13] = p1[3];
                pp[14] = p2[3];
                pp[15] = p3[3];
                pp[16] = p0[4];
                pp[17] = p1[4];
                pp[18] = p2[4];
                pp[19] = p3[4];
                pp[20] = p0[5];
                pp[21] = p1[5];
                pp[22] = p2[5];
                pp[23] = p3[5];
                pp[24] = p0[6];
                pp[25] = p1[6];
                pp[26] = p2[6];
                pp[27] = p3[6];
                pp[28] = p0[7];
                pp[29] = p1[7];
                pp[30] = p2[7];
                pp[31] = p3[7];
                pp[32] = p0[8];
                pp[33] = p1[8];
                pp[34] = p2[8];
                pp[35] = p3[8];
                pp[36] = p0[9];
                pp[37] = p1[9];
                pp[38] = p2[9];
                pp[39] = p3[9];
                pp[40] = p0[10];
                pp[41] = p1[10];
                pp[42] = p2[10];
                pp[43] = p3[10];
                pp[44] = p0[11];
                pp[45] = p1[11];
                pp[46] = p2[11];
                pp[47] = p3[11];
                pp[48] = p0[12];
                pp[49] = p1[12];
                pp[50] = p2[12];
                pp[51] = p3[12];
                pp[52] = p0[13];
                pp[53] = p1[13];
                pp[54] = p2[13];
                pp[55] = p3[13];
                pp[56] = p0[14];
                pp[57] = p1[14];
                pp[58] = p2[14];
                pp[59] = p3[14];
                pp[60] = p0[15];
                pp[61] = p1[15];
                pp[62] = p2[15];
                pp[63] = p3[15];
                pp += 64;
                p0 += bottom_blob.cstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[4];
                pp[5] = p0[5];
                pp[6] = p0[6];
                pp[7] = p0[7];
                pp[8] = p0[8];
                pp[9] = p0[9];
                pp[10] = p0[10];
                pp[11] = p0[11];
                pp[12] = p0[12];
                pp[13] = p0[13];
                pp[14] = p0[14];
                pp[15] = p0[15];
                pp += 16;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __loongarch_asx
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                for (int n = 0; n < 8; n += 4)
                {
                    pp[0] = p0[n];
                    pp[1] = p0[n + 1];
                    pp[2] = p0[n + 2];
                    pp[3] = p0[n + 3];
                    pp[4] = p0[8 + n];
                    pp[5] = p0[8 + n + 1];
                    pp[6] = p0[8 + n + 2];
                    pp[7] = p0[8 + n + 3];
                    pp[8] = p0[16 + n];
                    pp[9] = p0[16 + n + 1];
                    pp[10] = p0[16 + n + 2];
                    pp[11] = p0[16 + n + 3];
                    pp[12] = p0[24 + n];
                    pp[13] = p0[24 + n + 1];
                    pp[14] = p0[24 + n + 2];
                    pp[15] = p0[24 + n + 3];
                    pp[16] = p0[32 + n];
                    pp[17] = p0[32 + n + 1];
                    pp[18] = p0[32 + n + 2];
                    pp[19] = p0[32 + n + 3];
                    pp[20] = p0[40 + n];
                    pp[21] = p0[40 + n + 1];
                    pp[22] = p0[40 + n + 2];
                    pp[23] = p0[40 + n + 3];
                    pp[24] = p0[48 + n];
                    pp[25] = p0[48 + n + 1];
                    pp[26] = p0[48 + n + 2];
                    pp[27] = p0[48 + n + 3];
                    pp[28] = p0[56 + n];
                    pp[29] = p0[56 + n + 1];
                    pp[30] = p0[56 + n + 2];
                    pp[31] = p0[56 + n + 3];
                    pp += 32;
                }
                p0 += bottom_blob.cstep * 8;
            }
        }
        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const signed char* p1 = p0 + bottom_blob.cstep;
                const signed char* p2 = p0 + bottom_blob.cstep * 2;
                const signed char* p3 = p0 + bottom_blob.cstep * 3;

                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p0[1];
                pp[5] = p1[1];
                pp[6] = p2[1];
                pp[7] = p3[1];
                pp[8] = p0[2];
                pp[9] = p1[2];
                pp[10] = p2[2];
                pp[11] = p3[2];
                pp[12] = p0[3];
                pp[13] = p1[3];
                pp[14] = p2[3];
                pp[15] = p3[3];
                pp[16] = p0[4];
                pp[17] = p1[4];
                pp[18] = p2[4];
                pp[19] = p3[4];
                pp[20] = p0[5];
                pp[21] = p1[5];
                pp[22] = p2[5];
                pp[23] = p3[5];
                pp[24] = p0[6];
                pp[25] = p1[6];
                pp[26] = p2[6];
                pp[27] = p3[6];
                pp[28] = p0[7];
                pp[29] = p1[7];
                pp[30] = p2[7];
                pp[31] = p3[7];
                pp += 32;
                p0 += bottom_blob.cstep * 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[4];
                pp[5] = p0[5];
                pp[6] = p0[6];
                pp[7] = p0[7];
                pp += 8;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                for (int n = 0; n < 8; n += 4)
                {
                    pp[0] = p0[n];
                    pp[1] = p0[n + 1];
                    pp[2] = p0[n + 2];
                    pp[3] = p0[n + 3];
                    pp[4] = p0[8 + n];
                    pp[5] = p0[8 + n + 1];
                    pp[6] = p0[8 + n + 2];
                    pp[7] = p0[8 + n + 3];
                    pp[8] = p0[16 + n];
                    pp[9] = p0[16 + n + 1];
                    pp[10] = p0[16 + n + 2];
                    pp[11] = p0[16 + n + 3];
                    pp[12] = p0[24 + n];
                    pp[13] = p0[24 + n + 1];
                    pp[14] = p0[24 + n + 2];
                    pp[15] = p0[24 + n + 3];
                    pp += 16;
                }
                p0 += bottom_blob.cstep * 8;
            }
        }
        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const signed char* p1 = p0 + bottom_blob.cstep;
                const signed char* p2 = p0 + bottom_blob.cstep * 2;
                const signed char* p3 = p0 + bottom_blob.cstep * 3;

                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p0[1];
                pp[5] = p1[1];
                pp[6] = p2[1];
                pp[7] = p3[1];
                pp[8] = p0[2];
                pp[9] = p1[2];
                pp[10] = p2[2];
                pp[11] = p3[2];
                pp[12] = p0[3];
                pp[13] = p1[3];
                pp[14] = p2[3];
                pp[15] = p3[3];
                pp += 16;
                p0 += bottom_blob.cstep * 4;
            }
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
                for (int n = 0; n < 8; n += 4)
                {
                    pp[0] = p0[n];
                    pp[1] = p0[n + 1];
                    pp[2] = p0[n + 2];
                    pp[3] = p0[n + 3];
                    pp[4] = p0[8 + n];
                    pp[5] = p0[8 + n + 1];
                    pp[6] = p0[8 + n + 2];
                    pp[7] = p0[8 + n + 3];
                    pp += 8;
                }
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __loongarch_sx
        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                const signed char* p1 = p0 + bottom_blob.cstep;
                const signed char* p2 = p0 + bottom_blob.cstep * 2;
                const signed char* p3 = p0 + bottom_blob.cstep * 3;

                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p0[1];
                pp[5] = p1[1];
                pp[6] = p2[1];
                pp[7] = p3[1];
                pp += 8;
                p0 += bottom_blob.cstep * 4;
            }
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
#if __loongarch_asx
    for (; jj + 15 < max_jj; jj += 16)
    {
        int dy[16];
        int dx[16];
        for (int c = 0; c < 16; c++)
        {
            dy[c] = (j + jj + c) / outw;
            dx[c] = (j + jj + c) % outw;
        }

        int kk = 0;
        if (elempack == 8)
        {
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int q = 0; q < 4; q++)
                {
                    const int raw_ch = (kk + q) / maxk;
                    const int kpos = (kk + q) % maxk;
                    const int p = raw_ch / 8;
                    const int n = raw_ch % 8;
                    const int u = kpos / kernel_w;
                    const int v = kpos % kernel_w;

                    const Mat img = bottom_blob.channel(p);

                    for (int c = 0; c < 16; c++)
                    {
                        const int x = stride_w * dx[c] + dilation_w * v;
                        const int y = stride_h * dy[c] + dilation_h * u;
                        const signed char* sptr = img.row<const signed char>(y) + x * 8;
                        pp[c * 4 + q] = sptr[n];
                    }
                }
                pp += 64;
            }
            for (; kk < max_kk; kk++)
            {
                const int raw_ch = kk / maxk;
                const int kpos = kk % maxk;
                const int p = raw_ch / 8;
                const int n = raw_ch % 8;
                const int u = kpos / kernel_w;
                const int v = kpos % kernel_w;

                const Mat img = bottom_blob.channel(p);

                for (int c = 0; c < 16; c++)
                {
                    const int x = stride_w * dx[c] + dilation_w * v;
                    const int y = stride_h * dy[c] + dilation_h * u;
                    const signed char* sptr = img.row<const signed char>(y) + x * 8;
                    pp[c] = sptr[n];
                }
                pp += 16;
            }
        }
        if (elempack == 1)
        {
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int q = 0; q < 4; q++)
                {
                    const int p = (k + kk + q) / maxk;
                    const int uv = (k + kk + q) % maxk;
                    const int u = uv / kernel_w;
                    const int v = uv % kernel_w;

                    const Mat img = bottom_blob.channel(p);

                    for (int c = 0; c < 16; c++)
                    {
                        const int x = stride_w * dx[c] + dilation_w * v;
                        const int y = stride_h * dy[c] + dilation_h * u;
                        const signed char* sptr = img.row<const signed char>(y) + x;
                        pp[c * 4 + q] = sptr[0];
                    }
                }
                pp += 64;
            }
            for (; kk < max_kk; kk++)
            {
                const int p = (k + kk) / maxk;
                const int uv = (k + kk) % maxk;
                const int u = uv / kernel_w;
                const int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                for (int c = 0; c < 16; c++)
                {
                    const int x = stride_w * dx[c] + dilation_w * v;
                    const int y = stride_h * dy[c] + dilation_h * u;
                    const signed char* sptr = img.row<const signed char>(y) + x;
                    pp[c] = sptr[0];
                }
                pp += 16;
            }
        }
    }
#endif // __loongarch_asx
    for (; jj + 7 < max_jj; jj += 8)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;

        if (dy0 == dy7)
        {
            int kk = 0;
            if (elempack == 8)
            {
                for (; kk + 3 < max_kk; kk += 4)
                {
                    for (int q = 0; q < 4; q++)
                    {
                        int raw_ch = (kk + q) / maxk;
                        int kpos = (kk + q) % maxk;
                        int p = raw_ch / 8;
                        int n = raw_ch % 8;
                        int u = kpos / kernel_w;
                        int v = kpos % kernel_w;

                        const Mat img = bottom_blob.channel(p);

                        int x0 = stride_w * dx0 + dilation_w * v;
                        int y0 = stride_h * dy0 + dilation_h * u;

                        const signed char* sptr = img.row<const signed char>(y0) + x0 * 8;

                        pp[q] = sptr[n];
                        pp[4 + q] = sptr[stride_w * 8 + n];
                        pp[8 + q] = sptr[stride_w * 16 + n];
                        pp[12 + q] = sptr[stride_w * 24 + n];
                        pp[16 + q] = sptr[stride_w * 32 + n];
                        pp[20 + q] = sptr[stride_w * 40 + n];
                        pp[24 + q] = sptr[stride_w * 48 + n];
                        pp[28 + q] = sptr[stride_w * 56 + n];
                    }
                    pp += 32;
                }
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
                    pp[4] = sptr[stride_w * 32 + n];
                    pp[5] = sptr[stride_w * 40 + n];
                    pp[6] = sptr[stride_w * 48 + n];
                    pp[7] = sptr[stride_w * 56 + n];
                    pp += 8;
                }
            }
            for (; kk + 3 < max_kk / elempack; kk += 4)
            {
                for (int q = 0; q < 4; q++)
                {
                    int p = (k / elempack + kk + q) / maxk;
                    int uv = (k / elempack + kk + q) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    const Mat img = bottom_blob.channel(p);

                    int x0 = stride_w * dx0 + dilation_w * v;
                    int y0 = stride_h * dy0 + dilation_h * u;

                    const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

                    if (elempack == 1)
                    {
                        pp[q] = sptr[0];
                        pp[4 + q] = sptr[stride_w];
                        pp[8 + q] = sptr[stride_w * 2];
                        pp[12 + q] = sptr[stride_w * 3];
                        pp[16 + q] = sptr[stride_w * 4];
                        pp[20 + q] = sptr[stride_w * 5];
                        pp[24 + q] = sptr[stride_w * 6];
                        pp[28 + q] = sptr[stride_w * 7];
                    }
                }
                pp += 32;
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
                    pp[4] = sptr[stride_w * 4];
                    pp[5] = sptr[stride_w * 5];
                    pp[6] = sptr[stride_w * 6];
                    pp[7] = sptr[stride_w * 7];
                    pp += 8;
                }
            }
        }
        else
        {
            int kk = 0;
            if (elempack == 8)
            {
                for (; kk + 3 < max_kk; kk += 4)
                {
                    for (int q = 0; q < 4; q++)
                    {
                        int raw_ch = (kk + q) / maxk;
                        int kpos = (kk + q) % maxk;
                        int p = raw_ch / 8;
                        int n = raw_ch % 8;
                        int u = kpos / kernel_w;
                        int v = kpos % kernel_w;

                        const Mat img = bottom_blob.channel(p);

                        int x0 = stride_w * dx0 + dilation_w * v;
                        int x1 = stride_w * dx1 + dilation_w * v;
                        int x2 = stride_w * dx2 + dilation_w * v;
                        int x3 = stride_w * dx3 + dilation_w * v;
                        int x4 = stride_w * dx4 + dilation_w * v;
                        int x5 = stride_w * dx5 + dilation_w * v;
                        int x6 = stride_w * dx6 + dilation_w * v;
                        int x7 = stride_w * dx7 + dilation_w * v;

                        int y0 = stride_h * dy0 + dilation_h * u;
                        int y1 = stride_h * dy1 + dilation_h * u;
                        int y2 = stride_h * dy2 + dilation_h * u;
                        int y3 = stride_h * dy3 + dilation_h * u;
                        int y4 = stride_h * dy4 + dilation_h * u;
                        int y5 = stride_h * dy5 + dilation_h * u;
                        int y6 = stride_h * dy6 + dilation_h * u;
                        int y7 = stride_h * dy7 + dilation_h * u;

                        const signed char* sptr0 = img.row<const signed char>(y0) + x0 * 8;
                        const signed char* sptr1 = img.row<const signed char>(y1) + x1 * 8;
                        const signed char* sptr2 = img.row<const signed char>(y2) + x2 * 8;
                        const signed char* sptr3 = img.row<const signed char>(y3) + x3 * 8;
                        const signed char* sptr4 = img.row<const signed char>(y4) + x4 * 8;
                        const signed char* sptr5 = img.row<const signed char>(y5) + x5 * 8;
                        const signed char* sptr6 = img.row<const signed char>(y6) + x6 * 8;
                        const signed char* sptr7 = img.row<const signed char>(y7) + x7 * 8;

                        pp[q] = sptr0[n];
                        pp[4 + q] = sptr1[n];
                        pp[8 + q] = sptr2[n];
                        pp[12 + q] = sptr3[n];
                        pp[16 + q] = sptr4[n];
                        pp[20 + q] = sptr5[n];
                        pp[24 + q] = sptr6[n];
                        pp[28 + q] = sptr7[n];
                    }
                    pp += 32;
                }
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
                    int x4 = stride_w * dx4 + dilation_w * v;
                    int x5 = stride_w * dx5 + dilation_w * v;
                    int x6 = stride_w * dx6 + dilation_w * v;
                    int x7 = stride_w * dx7 + dilation_w * v;

                    int y0 = stride_h * dy0 + dilation_h * u;
                    int y1 = stride_h * dy1 + dilation_h * u;
                    int y2 = stride_h * dy2 + dilation_h * u;
                    int y3 = stride_h * dy3 + dilation_h * u;
                    int y4 = stride_h * dy4 + dilation_h * u;
                    int y5 = stride_h * dy5 + dilation_h * u;
                    int y6 = stride_h * dy6 + dilation_h * u;
                    int y7 = stride_h * dy7 + dilation_h * u;

                    const signed char* sptr0 = img.row<const signed char>(y0) + x0 * 8;
                    const signed char* sptr1 = img.row<const signed char>(y1) + x1 * 8;
                    const signed char* sptr2 = img.row<const signed char>(y2) + x2 * 8;
                    const signed char* sptr3 = img.row<const signed char>(y3) + x3 * 8;
                    const signed char* sptr4 = img.row<const signed char>(y4) + x4 * 8;
                    const signed char* sptr5 = img.row<const signed char>(y5) + x5 * 8;
                    const signed char* sptr6 = img.row<const signed char>(y6) + x6 * 8;
                    const signed char* sptr7 = img.row<const signed char>(y7) + x7 * 8;

                    pp[0] = sptr0[n];
                    pp[1] = sptr1[n];
                    pp[2] = sptr2[n];
                    pp[3] = sptr3[n];
                    pp[4] = sptr4[n];
                    pp[5] = sptr5[n];
                    pp[6] = sptr6[n];
                    pp[7] = sptr7[n];
                    pp += 8;
                }
            }
            for (; kk + 3 < max_kk / elempack; kk += 4)
            {
                for (int q = 0; q < 4; q++)
                {
                    int p = (k / elempack + kk + q) / maxk;
                    int uv = (k / elempack + kk + q) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    const Mat img = bottom_blob.channel(p);

                    int x0 = stride_w * dx0 + dilation_w * v;
                    int x1 = stride_w * dx1 + dilation_w * v;
                    int x2 = stride_w * dx2 + dilation_w * v;
                    int x3 = stride_w * dx3 + dilation_w * v;
                    int x4 = stride_w * dx4 + dilation_w * v;
                    int x5 = stride_w * dx5 + dilation_w * v;
                    int x6 = stride_w * dx6 + dilation_w * v;
                    int x7 = stride_w * dx7 + dilation_w * v;

                    int y0 = stride_h * dy0 + dilation_h * u;
                    int y1 = stride_h * dy1 + dilation_h * u;
                    int y2 = stride_h * dy2 + dilation_h * u;
                    int y3 = stride_h * dy3 + dilation_h * u;
                    int y4 = stride_h * dy4 + dilation_h * u;
                    int y5 = stride_h * dy5 + dilation_h * u;
                    int y6 = stride_h * dy6 + dilation_h * u;
                    int y7 = stride_h * dy7 + dilation_h * u;

                    const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                    const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                    const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                    const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;
                    const signed char* sptr4 = img.row<const signed char>(y4) + x4 * elempack;
                    const signed char* sptr5 = img.row<const signed char>(y5) + x5 * elempack;
                    const signed char* sptr6 = img.row<const signed char>(y6) + x6 * elempack;
                    const signed char* sptr7 = img.row<const signed char>(y7) + x7 * elempack;

                    if (elempack == 1)
                    {
                        pp[q] = sptr0[0];
                        pp[4 + q] = sptr1[0];
                        pp[8 + q] = sptr2[0];
                        pp[12 + q] = sptr3[0];
                        pp[16 + q] = sptr4[0];
                        pp[20 + q] = sptr5[0];
                        pp[24 + q] = sptr6[0];
                        pp[28 + q] = sptr7[0];
                    }
                }
                pp += 32;
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
                int x4 = stride_w * dx4 + dilation_w * v;
                int x5 = stride_w * dx5 + dilation_w * v;
                int x6 = stride_w * dx6 + dilation_w * v;
                int x7 = stride_w * dx7 + dilation_w * v;

                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;
                int y4 = stride_h * dy4 + dilation_h * u;
                int y5 = stride_h * dy5 + dilation_h * u;
                int y6 = stride_h * dy6 + dilation_h * u;
                int y7 = stride_h * dy7 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;
                const signed char* sptr4 = img.row<const signed char>(y4) + x4 * elempack;
                const signed char* sptr5 = img.row<const signed char>(y5) + x5 * elempack;
                const signed char* sptr6 = img.row<const signed char>(y6) + x6 * elempack;
                const signed char* sptr7 = img.row<const signed char>(y7) + x7 * elempack;

                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr4[0];
                    pp[5] = sptr5[0];
                    pp[6] = sptr6[0];
                    pp[7] = sptr7[0];
                    pp += 8;
                }
            }
        }
    }
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
                for (; kk + 3 < max_kk; kk += 4)
                {
                    for (int q = 0; q < 4; q++)
                    {
                        int raw_ch = (kk + q) / maxk;
                        int kpos = (kk + q) % maxk;
                        int p = raw_ch / 8;
                        int n = raw_ch % 8;
                        int u = kpos / kernel_w;
                        int v = kpos % kernel_w;

                        const Mat img = bottom_blob.channel(p);

                        int x0 = stride_w * dx0 + dilation_w * v;
                        int y0 = stride_h * dy0 + dilation_h * u;

                        const signed char* sptr = img.row<const signed char>(y0) + x0 * 8;

                        pp[q] = sptr[n];
                        pp[4 + q] = sptr[stride_w * 8 + n];
                        pp[8 + q] = sptr[stride_w * 16 + n];
                        pp[12 + q] = sptr[stride_w * 24 + n];
                    }
                    pp += 16;
                }
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
            for (; kk + 3 < max_kk / elempack; kk += 4)
            {
                for (int q = 0; q < 4; q++)
                {
                    int p = (k / elempack + kk + q) / maxk;
                    int uv = (k / elempack + kk + q) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    const Mat img = bottom_blob.channel(p);

                    int x0 = stride_w * dx0 + dilation_w * v;
                    int y0 = stride_h * dy0 + dilation_h * u;

                    const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

                    if (elempack == 1)
                    {
                        pp[q] = sptr[0];
                        pp[4 + q] = sptr[stride_w];
                        pp[8 + q] = sptr[stride_w * 2];
                        pp[12 + q] = sptr[stride_w * 3];
                    }
                }
                pp += 16;
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
                for (; kk + 3 < max_kk; kk += 4)
                {
                    for (int q = 0; q < 4; q++)
                    {
                        int raw_ch = (kk + q) / maxk;
                        int kpos = (kk + q) % maxk;
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

                        pp[q] = sptr0[n];
                        pp[4 + q] = sptr1[n];
                        pp[8 + q] = sptr2[n];
                        pp[12 + q] = sptr3[n];
                    }
                    pp += 16;
                }
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
            for (; kk + 3 < max_kk / elempack; kk += 4)
            {
                for (int q = 0; q < 4; q++)
                {
                    int p = (k / elempack + kk + q) / maxk;
                    int uv = (k / elempack + kk + q) % maxk;
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
                        pp[q] = sptr0[0];
                        pp[4 + q] = sptr1[0];
                        pp[8 + q] = sptr2[0];
                        pp[12 + q] = sptr3[0];
                    }
                }
                pp += 16;
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
                for (; kk + 3 < max_kk; kk += 4)
                {
                    for (int q = 0; q < 4; q++)
                    {
                        int raw_ch = (kk + q) / maxk;
                        int kpos = (kk + q) % maxk;
                        int p = raw_ch / 8;
                        int n = raw_ch % 8;
                        int u = kpos / kernel_w;
                        int v = kpos % kernel_w;

                        const Mat img = bottom_blob.channel(p);

                        int x0 = stride_w * dx0 + dilation_w * v;
                        int y0 = stride_h * dy0 + dilation_h * u;

                        const signed char* sptr = img.row<const signed char>(y0) + x0 * 8;

                        pp[q] = sptr[n];
                        pp[4 + q] = sptr[stride_w * 8 + n];
                    }
                    pp += 8;
                }
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
            for (; kk + 3 < max_kk / elempack; kk += 4)
            {
                for (int q = 0; q < 4; q++)
                {
                    int p = (k / elempack + kk + q) / maxk;
                    int uv = (k / elempack + kk + q) % maxk;
                    int u = uv / kernel_w;
                    int v = uv % kernel_w;

                    const Mat img = bottom_blob.channel(p);

                    int x0 = stride_w * dx0 + dilation_w * v;
                    int y0 = stride_h * dy0 + dilation_h * u;

                    const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

                    if (elempack == 1)
                    {
                        pp[q] = sptr[0];
                        pp[4 + q] = sptr[stride_w];
                    }
                }
                pp += 8;
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
                for (; kk + 3 < max_kk; kk += 4)
                {
                    for (int q = 0; q < 4; q++)
                    {
                        int raw_ch = (kk + q) / maxk;
                        int kpos = (kk + q) % maxk;
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

                        pp[q] = sptr0[n];
                        pp[4 + q] = sptr1[n];
                    }
                    pp += 8;
                }
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
            for (; kk + 3 < max_kk / elempack; kk += 4)
            {
                for (int q = 0; q < 4; q++)
                {
                    int p = (k / elempack + kk + q) / maxk;
                    int uv = (k / elempack + kk + q) % maxk;
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
                        pp[q] = sptr0[0];
                        pp[4 + q] = sptr1[0];
                    }
                }
                pp += 8;
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
