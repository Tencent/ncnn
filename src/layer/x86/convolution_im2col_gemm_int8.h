// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
void convolution_im2col_gemm_int8_avx512vnni(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__
void convolution_im2col_gemm_int8_avxvnni(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
void convolution_im2col_gemm_transform_kernel_int8_avx2(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt);
void convolution_im2col_gemm_int8_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__
void convolution_im2col_gemm_int8_xop(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt);
#endif
#endif

static void convolution_im2col_pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // A = (pa, maxk, inch/pa), outch
    const int A_hstep = A.w;

    signed char* pp = AT;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;
        const signed char* p2 = (const signed char*)A + (i + ii + 2) * A_hstep + k;
        const signed char* p3 = (const signed char*)A + (i + ii + 3) * A_hstep + k;
        const signed char* p4 = (const signed char*)A + (i + ii + 4) * A_hstep + k;
        const signed char* p5 = (const signed char*)A + (i + ii + 5) * A_hstep + k;
        const signed char* p6 = (const signed char*)A + (i + ii + 6) * A_hstep + k;
        const signed char* p7 = (const signed char*)A + (i + ii + 7) * A_hstep + k;
        const signed char* p8 = (const signed char*)A + (i + ii + 8) * A_hstep + k;
        const signed char* p9 = (const signed char*)A + (i + ii + 9) * A_hstep + k;
        const signed char* pa = (const signed char*)A + (i + ii + 10) * A_hstep + k;
        const signed char* pb = (const signed char*)A + (i + ii + 11) * A_hstep + k;
        const signed char* pc = (const signed char*)A + (i + ii + 12) * A_hstep + k;
        const signed char* pd = (const signed char*)A + (i + ii + 13) * A_hstep + k;
        const signed char* pe = (const signed char*)A + (i + ii + 14) * A_hstep + k;
        const signed char* pf = (const signed char*)A + (i + ii + 15) * A_hstep + k;

        int kk = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; kk + 15 < max_kk; kk += 16)
        {
            __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
            __m128i _r1 = _mm_loadu_si128((const __m128i*)p1);
            __m128i _r2 = _mm_loadu_si128((const __m128i*)p2);
            __m128i _r3 = _mm_loadu_si128((const __m128i*)p3);
            __m128i _r4 = _mm_loadu_si128((const __m128i*)p4);
            __m128i _r5 = _mm_loadu_si128((const __m128i*)p5);
            __m128i _r6 = _mm_loadu_si128((const __m128i*)p6);
            __m128i _r7 = _mm_loadu_si128((const __m128i*)p7);
            __m128i _r8 = _mm_loadu_si128((const __m128i*)p8);
            __m128i _r9 = _mm_loadu_si128((const __m128i*)p9);
            __m128i _ra = _mm_loadu_si128((const __m128i*)pa);
            __m128i _rb = _mm_loadu_si128((const __m128i*)pb);
            __m128i _rc = _mm_loadu_si128((const __m128i*)pc);
            __m128i _rd = _mm_loadu_si128((const __m128i*)pd);
            __m128i _re = _mm_loadu_si128((const __m128i*)pe);
            __m128i _rf = _mm_loadu_si128((const __m128i*)pf);
            __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
            __m128i _t1 = _mm_unpackhi_epi16(_r0, _r1);
            __m128i _t2 = _mm_unpacklo_epi16(_r2, _r3);
            __m128i _t3 = _mm_unpackhi_epi16(_r2, _r3);
            __m128i _t4 = _mm_unpacklo_epi16(_r4, _r5);
            __m128i _t5 = _mm_unpackhi_epi16(_r4, _r5);
            __m128i _t6 = _mm_unpacklo_epi16(_r6, _r7);
            __m128i _t7 = _mm_unpackhi_epi16(_r6, _r7);
            __m128i _t8 = _mm_unpacklo_epi16(_r8, _r9);
            __m128i _t9 = _mm_unpackhi_epi16(_r8, _r9);
            __m128i _ta = _mm_unpacklo_epi16(_ra, _rb);
            __m128i _tb = _mm_unpackhi_epi16(_ra, _rb);
            __m128i _tc = _mm_unpacklo_epi16(_rc, _rd);
            __m128i _td = _mm_unpackhi_epi16(_rc, _rd);
            __m128i _te = _mm_unpacklo_epi16(_re, _rf);
            __m128i _tf = _mm_unpackhi_epi16(_re, _rf);
            _r0 = _mm_unpacklo_epi32(_t0, _t2);
            _r1 = _mm_unpackhi_epi32(_t0, _t2);
            _r2 = _mm_unpacklo_epi32(_t4, _t6);
            _r3 = _mm_unpackhi_epi32(_t4, _t6);
            _r4 = _mm_unpacklo_epi32(_t8, _ta);
            _r5 = _mm_unpackhi_epi32(_t8, _ta);
            _r6 = _mm_unpacklo_epi32(_tc, _te);
            _r7 = _mm_unpackhi_epi32(_tc, _te);
            _r8 = _mm_unpacklo_epi32(_t1, _t3);
            _r9 = _mm_unpackhi_epi32(_t1, _t3);
            _ra = _mm_unpacklo_epi32(_t5, _t7);
            _rb = _mm_unpackhi_epi32(_t5, _t7);
            _rc = _mm_unpacklo_epi32(_t9, _tb);
            _rd = _mm_unpackhi_epi32(_t9, _tb);
            _re = _mm_unpacklo_epi32(_td, _tf);
            _rf = _mm_unpackhi_epi32(_td, _tf);
            _t0 = _mm_unpacklo_epi64(_r0, _r2);
            _t1 = _mm_unpackhi_epi64(_r0, _r2);
            _t2 = _mm_unpacklo_epi64(_r1, _r3);
            _t3 = _mm_unpackhi_epi64(_r1, _r3);
            _t4 = _mm_unpacklo_epi64(_r4, _r6);
            _t5 = _mm_unpackhi_epi64(_r4, _r6);
            _t6 = _mm_unpacklo_epi64(_r5, _r7);
            _t7 = _mm_unpackhi_epi64(_r5, _r7);
            _t8 = _mm_unpacklo_epi64(_r8, _ra);
            _t9 = _mm_unpackhi_epi64(_r8, _ra);
            _ta = _mm_unpacklo_epi64(_r9, _rb);
            _tb = _mm_unpackhi_epi64(_r9, _rb);
            _tc = _mm_unpacklo_epi64(_rc, _re);
            _td = _mm_unpackhi_epi64(_rc, _re);
            _te = _mm_unpacklo_epi64(_rd, _rf);
            _tf = _mm_unpackhi_epi64(_rd, _rf);
            _mm_storeu_si128((__m128i*)pp, _t0);
            _mm_storeu_si128((__m128i*)(pp + 16), _t4);
            _mm_storeu_si128((__m128i*)(pp + 32), _t1);
            _mm_storeu_si128((__m128i*)(pp + 48), _t5);
            _mm_storeu_si128((__m128i*)(pp + 64), _t2);
            _mm_storeu_si128((__m128i*)(pp + 80), _t6);
            _mm_storeu_si128((__m128i*)(pp + 96), _t3);
            _mm_storeu_si128((__m128i*)(pp + 112), _t7);
            _mm_storeu_si128((__m128i*)(pp + 128), _t8);
            _mm_storeu_si128((__m128i*)(pp + 144), _tc);
            _mm_storeu_si128((__m128i*)(pp + 160), _t9);
            _mm_storeu_si128((__m128i*)(pp + 176), _td);
            _mm_storeu_si128((__m128i*)(pp + 192), _ta);
            _mm_storeu_si128((__m128i*)(pp + 208), _te);
            _mm_storeu_si128((__m128i*)(pp + 224), _tb);
            _mm_storeu_si128((__m128i*)(pp + 240), _tf);
            pp += 256;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
            p4 += 16;
            p5 += 16;
            p6 += 16;
            p7 += 16;
            p8 += 16;
            p9 += 16;
            pa += 16;
            pb += 16;
            pc += 16;
            pd += 16;
            pe += 16;
            pf += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
            __m128i _r1 = _mm_loadl_epi64((const __m128i*)p1);
            __m128i _r2 = _mm_loadl_epi64((const __m128i*)p2);
            __m128i _r3 = _mm_loadl_epi64((const __m128i*)p3);
            __m128i _r4 = _mm_loadl_epi64((const __m128i*)p4);
            __m128i _r5 = _mm_loadl_epi64((const __m128i*)p5);
            __m128i _r6 = _mm_loadl_epi64((const __m128i*)p6);
            __m128i _r7 = _mm_loadl_epi64((const __m128i*)p7);
            __m128i _r8 = _mm_loadl_epi64((const __m128i*)p8);
            __m128i _r9 = _mm_loadl_epi64((const __m128i*)p9);
            __m128i _ra = _mm_loadl_epi64((const __m128i*)pa);
            __m128i _rb = _mm_loadl_epi64((const __m128i*)pb);
            __m128i _rc = _mm_loadl_epi64((const __m128i*)pc);
            __m128i _rd = _mm_loadl_epi64((const __m128i*)pd);
            __m128i _re = _mm_loadl_epi64((const __m128i*)pe);
            __m128i _rf = _mm_loadl_epi64((const __m128i*)pf);
            __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
            __m128i _t1 = _mm_unpacklo_epi16(_r2, _r3);
            __m128i _t2 = _mm_unpacklo_epi16(_r4, _r5);
            __m128i _t3 = _mm_unpacklo_epi16(_r6, _r7);
            __m128i _t4 = _mm_unpacklo_epi16(_r8, _r9);
            __m128i _t5 = _mm_unpacklo_epi16(_ra, _rb);
            __m128i _t6 = _mm_unpacklo_epi16(_rc, _rd);
            __m128i _t7 = _mm_unpacklo_epi16(_re, _rf);
            _r0 = _mm_unpacklo_epi32(_t0, _t1);
            _r1 = _mm_unpackhi_epi32(_t0, _t1);
            _r2 = _mm_unpacklo_epi32(_t2, _t3);
            _r3 = _mm_unpackhi_epi32(_t2, _t3);
            _r4 = _mm_unpacklo_epi32(_t4, _t5);
            _r5 = _mm_unpackhi_epi32(_t4, _t5);
            _r6 = _mm_unpacklo_epi32(_t6, _t7);
            _r7 = _mm_unpackhi_epi32(_t6, _t7);
            _t0 = _mm_unpacklo_epi64(_r0, _r2);
            _t1 = _mm_unpackhi_epi64(_r0, _r2);
            _t2 = _mm_unpacklo_epi64(_r1, _r3);
            _t3 = _mm_unpackhi_epi64(_r1, _r3);
            _t4 = _mm_unpacklo_epi64(_r4, _r6);
            _t5 = _mm_unpackhi_epi64(_r4, _r6);
            _t6 = _mm_unpacklo_epi64(_r5, _r7);
            _t7 = _mm_unpackhi_epi64(_r5, _r7);
            _mm_storeu_si128((__m128i*)pp, _t0);
            _mm_storeu_si128((__m128i*)(pp + 16), _t4);
            _mm_storeu_si128((__m128i*)(pp + 32), _t1);
            _mm_storeu_si128((__m128i*)(pp + 48), _t5);
            _mm_storeu_si128((__m128i*)(pp + 64), _t2);
            _mm_storeu_si128((__m128i*)(pp + 80), _t6);
            _mm_storeu_si128((__m128i*)(pp + 96), _t3);
            _mm_storeu_si128((__m128i*)(pp + 112), _t7);
            pp += 128;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
            p4 += 8;
            p5 += 8;
            p6 += 8;
            p7 += 8;
            p8 += 8;
            p9 += 8;
            pa += 8;
            pb += 8;
            pc += 8;
            pd += 8;
            pe += 8;
            pf += 8;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp[8] = p4[0];
            pp[9] = p4[1];
            pp[10] = p5[0];
            pp[11] = p5[1];
            pp[12] = p6[0];
            pp[13] = p6[1];
            pp[14] = p7[0];
            pp[15] = p7[1];
            pp[16] = p8[0];
            pp[17] = p8[1];
            pp[18] = p9[0];
            pp[19] = p9[1];
            pp[20] = pa[0];
            pp[21] = pa[1];
            pp[22] = pb[0];
            pp[23] = pb[1];
            pp[24] = pc[0];
            pp[25] = pc[1];
            pp[26] = pd[0];
            pp[27] = pd[1];
            pp[28] = pe[0];
            pp[29] = pe[1];
            pp[30] = pf[0];
            pp[31] = pf[1];
            pp += 32;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
            p8 += 2;
            p9 += 2;
            pa += 2;
            pb += 2;
            pc += 2;
            pd += 2;
            pe += 2;
            pf += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p4[0];
            pp[5] = p5[0];
            pp[6] = p6[0];
            pp[7] = p7[0];
            pp[8] = p8[0];
            pp[9] = p9[0];
            pp[10] = pa[0];
            pp[11] = pb[0];
            pp[12] = pc[0];
            pp[13] = pd[0];
            pp[14] = pe[0];
            pp[15] = pf[0];
            pp += 16;
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
            p8++;
            p9++;
            pa++;
            pb++;
            pc++;
            pd++;
            pe++;
            pf++;
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;
        const signed char* p2 = (const signed char*)A + (i + ii + 2) * A_hstep + k;
        const signed char* p3 = (const signed char*)A + (i + ii + 3) * A_hstep + k;
        const signed char* p4 = (const signed char*)A + (i + ii + 4) * A_hstep + k;
        const signed char* p5 = (const signed char*)A + (i + ii + 5) * A_hstep + k;
        const signed char* p6 = (const signed char*)A + (i + ii + 6) * A_hstep + k;
        const signed char* p7 = (const signed char*)A + (i + ii + 7) * A_hstep + k;

        int kk = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; kk + 15 < max_kk; kk += 16)
        {
            __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
            __m128i _r1 = _mm_loadu_si128((const __m128i*)p1);
            __m128i _r2 = _mm_loadu_si128((const __m128i*)p2);
            __m128i _r3 = _mm_loadu_si128((const __m128i*)p3);
            __m128i _r4 = _mm_loadu_si128((const __m128i*)p4);
            __m128i _r5 = _mm_loadu_si128((const __m128i*)p5);
            __m128i _r6 = _mm_loadu_si128((const __m128i*)p6);
            __m128i _r7 = _mm_loadu_si128((const __m128i*)p7);
            __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
            __m128i _t1 = _mm_unpackhi_epi16(_r0, _r1);
            __m128i _t2 = _mm_unpacklo_epi16(_r2, _r3);
            __m128i _t3 = _mm_unpackhi_epi16(_r2, _r3);
            __m128i _t4 = _mm_unpacklo_epi16(_r4, _r5);
            __m128i _t5 = _mm_unpackhi_epi16(_r4, _r5);
            __m128i _t6 = _mm_unpacklo_epi16(_r6, _r7);
            __m128i _t7 = _mm_unpackhi_epi16(_r6, _r7);
            _r0 = _mm_unpacklo_epi32(_t0, _t2);
            _r1 = _mm_unpackhi_epi32(_t0, _t2);
            _r2 = _mm_unpacklo_epi32(_t4, _t6);
            _r3 = _mm_unpackhi_epi32(_t4, _t6);
            _r4 = _mm_unpacklo_epi32(_t1, _t3);
            _r5 = _mm_unpackhi_epi32(_t1, _t3);
            _r6 = _mm_unpacklo_epi32(_t5, _t7);
            _r7 = _mm_unpackhi_epi32(_t5, _t7);
            _t0 = _mm_unpacklo_epi64(_r0, _r2);
            _t1 = _mm_unpackhi_epi64(_r0, _r2);
            _t2 = _mm_unpacklo_epi64(_r1, _r3);
            _t3 = _mm_unpackhi_epi64(_r1, _r3);
            _t4 = _mm_unpacklo_epi64(_r4, _r6);
            _t5 = _mm_unpackhi_epi64(_r4, _r6);
            _t6 = _mm_unpacklo_epi64(_r5, _r7);
            _t7 = _mm_unpackhi_epi64(_r5, _r7);
            _mm_storeu_si128((__m128i*)pp, _t0);
            _mm_storeu_si128((__m128i*)(pp + 16), _t1);
            _mm_storeu_si128((__m128i*)(pp + 32), _t2);
            _mm_storeu_si128((__m128i*)(pp + 48), _t3);
            _mm_storeu_si128((__m128i*)(pp + 64), _t4);
            _mm_storeu_si128((__m128i*)(pp + 80), _t5);
            _mm_storeu_si128((__m128i*)(pp + 96), _t6);
            _mm_storeu_si128((__m128i*)(pp + 112), _t7);
            pp += 128;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
            p4 += 16;
            p5 += 16;
            p6 += 16;
            p7 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
            __m128i _r1 = _mm_loadl_epi64((const __m128i*)p1);
            __m128i _r2 = _mm_loadl_epi64((const __m128i*)p2);
            __m128i _r3 = _mm_loadl_epi64((const __m128i*)p3);
            __m128i _r4 = _mm_loadl_epi64((const __m128i*)p4);
            __m128i _r5 = _mm_loadl_epi64((const __m128i*)p5);
            __m128i _r6 = _mm_loadl_epi64((const __m128i*)p6);
            __m128i _r7 = _mm_loadl_epi64((const __m128i*)p7);
            __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
            __m128i _t1 = _mm_unpacklo_epi16(_r2, _r3);
            __m128i _t2 = _mm_unpacklo_epi16(_r4, _r5);
            __m128i _t3 = _mm_unpacklo_epi16(_r6, _r7);
            _r0 = _mm_unpacklo_epi32(_t0, _t1);
            _r1 = _mm_unpackhi_epi32(_t0, _t1);
            _r2 = _mm_unpacklo_epi32(_t2, _t3);
            _r3 = _mm_unpackhi_epi32(_t2, _t3);
            _r4 = _mm_unpacklo_epi64(_r0, _r2);
            _r5 = _mm_unpackhi_epi64(_r0, _r2);
            _r6 = _mm_unpacklo_epi64(_r1, _r3);
            _r7 = _mm_unpackhi_epi64(_r1, _r3);
            _mm_storeu_si128((__m128i*)pp, _r4);
            _mm_storeu_si128((__m128i*)(pp + 16), _r5);
            _mm_storeu_si128((__m128i*)(pp + 32), _r6);
            _mm_storeu_si128((__m128i*)(pp + 48), _r7);
            pp += 64;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
            p4 += 8;
            p5 += 8;
            p6 += 8;
            p7 += 8;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp[8] = p4[0];
            pp[9] = p4[1];
            pp[10] = p5[0];
            pp[11] = p5[1];
            pp[12] = p6[0];
            pp[13] = p6[1];
            pp[14] = p7[0];
            pp[15] = p7[1];
            pp += 16;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p4[0];
            pp[5] = p5[0];
            pp[6] = p6[0];
            pp[7] = p7[0];
            pp += 8;
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
        }
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;
        const signed char* p2 = (const signed char*)A + (i + ii + 2) * A_hstep + k;
        const signed char* p3 = (const signed char*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;
        for (; kk + 15 < max_kk; kk += 16)
        {
            __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
            __m128i _r1 = _mm_loadu_si128((const __m128i*)p1);
            __m128i _r2 = _mm_loadu_si128((const __m128i*)p2);
            __m128i _r3 = _mm_loadu_si128((const __m128i*)p3);
            __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
            __m128i _t1 = _mm_unpackhi_epi16(_r0, _r1);
            __m128i _t2 = _mm_unpacklo_epi16(_r2, _r3);
            __m128i _t3 = _mm_unpackhi_epi16(_r2, _r3);
            _r0 = _mm_unpacklo_epi32(_t0, _t2);
            _r1 = _mm_unpackhi_epi32(_t0, _t2);
            _r2 = _mm_unpacklo_epi32(_t1, _t3);
            _r3 = _mm_unpackhi_epi32(_t1, _t3);
            _mm_storeu_si128((__m128i*)pp, _r0);
            _mm_storeu_si128((__m128i*)(pp + 16), _r1);
            _mm_storeu_si128((__m128i*)(pp + 32), _r2);
            _mm_storeu_si128((__m128i*)(pp + 48), _r3);
            pp += 64;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
            __m128i _r1 = _mm_loadl_epi64((const __m128i*)p1);
            __m128i _r2 = _mm_loadl_epi64((const __m128i*)p2);
            __m128i _r3 = _mm_loadl_epi64((const __m128i*)p3);
            __m128i _t0 = _mm_unpacklo_epi16(_r0, _r1);
            __m128i _t1 = _mm_unpacklo_epi16(_r2, _r3);
            _r0 = _mm_unpacklo_epi32(_t0, _t1);
            _r1 = _mm_unpackhi_epi32(_t0, _t1);
            _mm_storeu_si128((__m128i*)pp, _r0);
            _mm_storeu_si128((__m128i*)(pp + 16), _r1);
            pp += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp += 8;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
#if __SSE2__
        for (; kk + 15 < max_kk; kk += 16)
        {
            __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
            __m128i _r1 = _mm_loadu_si128((const __m128i*)p1);
            __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
            __m128i _r23 = _mm_unpackhi_epi16(_r0, _r1);
            _mm_storeu_si128((__m128i*)pp, _r01);
            _mm_storeu_si128((__m128i*)(pp + 16), _r23);
            pp += 32;
            p0 += 16;
            p1 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
            __m128i _r1 = _mm_loadl_epi64((const __m128i*)p1);
            __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
            _mm_storeu_si128((__m128i*)pp, _r01);
            pp += 16;
            p0 += 8;
            p1 += 8;
        }
#endif // __SSE2__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp += 4;
            p0 += 2;
            p1 += 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;

        int kk = 0;
#if __SSE2__
        for (; kk + 15 < max_kk; kk += 15)
        {
            _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
            pp += 16;
            p0 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
            pp += 8;
            p0 += 8;
        }
#endif // __SSE2__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void convolution_gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    // NCNN_LOGE("convolution_gemm_transB_packed_tile_int8 %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const int out_elempack = top_blob.elempack;
    const int out_hstep = (int)top_blob.cstep;

    const signed char* pAT = AT_tile;
    const signed char* pBT = BT_tile;

    int* outptr = topT_tile;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        int* outptr0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const signed char* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;
            __m512i _sum2;
            __m512i _sum3;
            __m512i _sum4;
            __m512i _sum5;
            __m512i _sum6;
            __m512i _sum7;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
                _sum2 = _mm512_setzero_si512();
                _sum3 = _mm512_setzero_si512();
                _sum4 = _mm512_setzero_si512();
                _sum5 = _mm512_setzero_si512();
                _sum6 = _mm512_setzero_si512();
                _sum7 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
                _sum2 = _mm512_load_si512((const __m512i*)(outptr + 32));
                _sum3 = _mm512_load_si512((const __m512i*)(outptr + 48));
                _sum4 = _mm512_load_si512((const __m512i*)(outptr + 64));
                _sum5 = _mm512_load_si512((const __m512i*)(outptr + 80));
                _sum6 = _mm512_load_si512((const __m512i*)(outptr + 96));
                _sum7 = _mm512_load_si512((const __m512i*)(outptr + 112));
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);

                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pBl = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pBh = _mm_unpackhi_epi8(_pB, _extpB);

                __m512i _pBBBBl = _mm512_broadcast_i32x4(_pBl);
                __m512i _pBBBBh = _mm512_broadcast_i32x4(_pBh);

                // 0123012301230123 -> 00000000... 11111111... 22222222... 33333333...
                __m512i _pB0 = _mm512_shuffle_epi32(_pBBBBl, _MM_PERM_AAAA);
                __m512i _pB1 = _mm512_shuffle_epi32(_pBBBBl, _MM_PERM_BBBB);
                __m512i _pB2 = _mm512_shuffle_epi32(_pBBBBl, _MM_PERM_CCCC);
                __m512i _pB3 = _mm512_shuffle_epi32(_pBBBBl, _MM_PERM_DDDD);
                __m512i _pB4 = _mm512_shuffle_epi32(_pBBBBh, _MM_PERM_AAAA);
                __m512i _pB5 = _mm512_shuffle_epi32(_pBBBBh, _MM_PERM_BBBB);
                __m512i _pB6 = _mm512_shuffle_epi32(_pBBBBh, _MM_PERM_CCCC);
                __m512i _pB7 = _mm512_shuffle_epi32(_pBBBBh, _MM_PERM_DDDD);

#if __AVX512VNNI__
                _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm512_dpwssd_epi32(_sum2, _pA0, _pB2);
                _sum3 = _mm512_dpwssd_epi32(_sum3, _pA0, _pB3);
                _sum4 = _mm512_dpwssd_epi32(_sum4, _pA0, _pB4);
                _sum5 = _mm512_dpwssd_epi32(_sum5, _pA0, _pB5);
                _sum6 = _mm512_dpwssd_epi32(_sum6, _pA0, _pB6);
                _sum7 = _mm512_dpwssd_epi32(_sum7, _pA0, _pB7);
#else
                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA0, _pB2));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA0, _pB3));
                _sum4 = _mm512_add_epi32(_sum4, _mm512_madd_epi16(_pA0, _pB4));
                _sum5 = _mm512_add_epi32(_sum5, _mm512_madd_epi16(_pA0, _pB5));
                _sum6 = _mm512_add_epi32(_sum6, _mm512_madd_epi16(_pA0, _pB6));
                _sum7 = _mm512_add_epi32(_sum7, _mm512_madd_epi16(_pA0, _pB7));
#endif

                pA += 32;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m256i _pB0 = _mm256_set1_epi16(pB[0]);
                __m256i _pB1 = _mm256_set1_epi16(pB[1]);
                __m256i _pB2 = _mm256_set1_epi16(pB[2]);
                __m256i _pB3 = _mm256_set1_epi16(pB[3]);
                __m256i _pB4 = _mm256_set1_epi16(pB[4]);
                __m256i _pB5 = _mm256_set1_epi16(pB[5]);
                __m256i _pB6 = _mm256_set1_epi16(pB[6]);
                __m256i _pB7 = _mm256_set1_epi16(pB[7]);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));
                __m512i _s2 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB2));
                __m512i _s3 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB3));
                __m512i _s4 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB4));
                __m512i _s5 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB5));
                __m512i _s6 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB6));
                __m512i _s7 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB7));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);
                _sum2 = _mm512_add_epi32(_sum2, _s2);
                _sum3 = _mm512_add_epi32(_sum3, _s3);
                _sum4 = _mm512_add_epi32(_sum4, _s4);
                _sum5 = _mm512_add_epi32(_sum5, _s5);
                _sum6 = _mm512_add_epi32(_sum6, _s6);
                _sum7 = _mm512_add_epi32(_sum7, _s7);

                pA += 16;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16), _sum1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 32), _sum2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 48), _sum3);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 64), _sum4);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 80), _sum5);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 96), _sum6);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 112), _sum7);
                    outptr0 += 128;
                }
                if (out_elempack == 8)
                {
                    _mm256_storeu_si256((__m256i*)outptr0, _mm512_extracti32x8_epi32(_sum0, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 8), _mm512_extracti32x8_epi32(_sum1, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), _mm512_extracti32x8_epi32(_sum2, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 24), _mm512_extracti32x8_epi32(_sum3, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 32), _mm512_extracti32x8_epi32(_sum4, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 40), _mm512_extracti32x8_epi32(_sum5, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 48), _mm512_extracti32x8_epi32(_sum6, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 56), _mm512_extracti32x8_epi32(_sum7, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8), _mm512_extracti32x8_epi32(_sum0, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 8), _mm512_extracti32x8_epi32(_sum1, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 16), _mm512_extracti32x8_epi32(_sum2, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 24), _mm512_extracti32x8_epi32(_sum3, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 32), _mm512_extracti32x8_epi32(_sum4, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 40), _mm512_extracti32x8_epi32(_sum5, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 48), _mm512_extracti32x8_epi32(_sum6, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 56), _mm512_extracti32x8_epi32(_sum7, 1));
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    _mm_storeu_si128((__m128i*)outptr0, _mm512_extracti32x4_epi32(_sum0, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 4), _mm512_extracti32x4_epi32(_sum1, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8), _mm512_extracti32x4_epi32(_sum2, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 12), _mm512_extracti32x4_epi32(_sum3, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 16), _mm512_extracti32x4_epi32(_sum4, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 20), _mm512_extracti32x4_epi32(_sum5, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 24), _mm512_extracti32x4_epi32(_sum6, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 28), _mm512_extracti32x4_epi32(_sum7, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), _mm512_extracti32x4_epi32(_sum0, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 4), _mm512_extracti32x4_epi32(_sum1, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 8), _mm512_extracti32x4_epi32(_sum2, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 12), _mm512_extracti32x4_epi32(_sum3, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 16), _mm512_extracti32x4_epi32(_sum4, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 20), _mm512_extracti32x4_epi32(_sum5, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 24), _mm512_extracti32x4_epi32(_sum6, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 28), _mm512_extracti32x4_epi32(_sum7, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8), _mm512_extracti32x4_epi32(_sum0, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8 + 4), _mm512_extracti32x4_epi32(_sum1, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8 + 8), _mm512_extracti32x4_epi32(_sum2, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8 + 12), _mm512_extracti32x4_epi32(_sum3, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8 + 16), _mm512_extracti32x4_epi32(_sum4, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8 + 20), _mm512_extracti32x4_epi32(_sum5, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8 + 24), _mm512_extracti32x4_epi32(_sum6, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8 + 28), _mm512_extracti32x4_epi32(_sum7, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12), _mm512_extracti32x4_epi32(_sum0, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12 + 4), _mm512_extracti32x4_epi32(_sum1, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12 + 8), _mm512_extracti32x4_epi32(_sum2, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12 + 12), _mm512_extracti32x4_epi32(_sum3, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12 + 16), _mm512_extracti32x4_epi32(_sum4, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12 + 20), _mm512_extracti32x4_epi32(_sum5, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12 + 24), _mm512_extracti32x4_epi32(_sum6, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12 + 28), _mm512_extracti32x4_epi32(_sum7, 3));
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose16x8_epi32(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    _mm256_storeu_si256((__m256i*)outptr0, _mm512_extracti32x8_epi32(_sum0, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep), _mm512_extracti32x8_epi32(_sum0, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 2), _mm512_extracti32x8_epi32(_sum1, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 3), _mm512_extracti32x8_epi32(_sum1, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), _mm512_extracti32x8_epi32(_sum2, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 5), _mm512_extracti32x8_epi32(_sum2, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 6), _mm512_extracti32x8_epi32(_sum3, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 7), _mm512_extracti32x8_epi32(_sum3, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8), _mm512_extracti32x8_epi32(_sum4, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 9), _mm512_extracti32x8_epi32(_sum4, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 10), _mm512_extracti32x8_epi32(_sum5, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 11), _mm512_extracti32x8_epi32(_sum5, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 12), _mm512_extracti32x8_epi32(_sum6, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 13), _mm512_extracti32x8_epi32(_sum6, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 14), _mm512_extracti32x8_epi32(_sum7, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 15), _mm512_extracti32x8_epi32(_sum7, 1));
                    outptr0 += 8;
                }
            }
            else
            {
                _mm512_store_si512((__m512i*)outptr, _sum0);
                _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
                _mm512_store_si512((__m512i*)(outptr + 32), _sum2);
                _mm512_store_si512((__m512i*)(outptr + 48), _sum3);
                _mm512_store_si512((__m512i*)(outptr + 64), _sum4);
                _mm512_store_si512((__m512i*)(outptr + 80), _sum5);
                _mm512_store_si512((__m512i*)(outptr + 96), _sum6);
                _mm512_store_si512((__m512i*)(outptr + 112), _sum7);
            }

            outptr += 128;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;
            __m512i _sum2;
            __m512i _sum3;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
                _sum2 = _mm512_setzero_si512();
                _sum3 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
                _sum2 = _mm512_load_si512((const __m512i*)(outptr + 32));
                _sum3 = _mm512_load_si512((const __m512i*)(outptr + 48));
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
                __m512i _pBBBB = _mm512_broadcast_i32x4(_pB);

                // 0123012301230123 -> 00000000... 11111111... 22222222... 33333333...
                __m512i _pB0 = _mm512_shuffle_epi32(_pBBBB, _MM_PERM_AAAA);
                __m512i _pB1 = _mm512_shuffle_epi32(_pBBBB, _MM_PERM_BBBB);
                __m512i _pB2 = _mm512_shuffle_epi32(_pBBBB, _MM_PERM_CCCC);
                __m512i _pB3 = _mm512_shuffle_epi32(_pBBBB, _MM_PERM_DDDD);

#if __AVX512VNNI__
                _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm512_dpwssd_epi32(_sum2, _pA0, _pB2);
                _sum3 = _mm512_dpwssd_epi32(_sum3, _pA0, _pB3);
#else
                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA0, _pB2));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA0, _pB3));
#endif

                pA += 32;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m256i _pB0 = _mm256_set1_epi16(pB[0]);
                __m256i _pB1 = _mm256_set1_epi16(pB[1]);
                __m256i _pB2 = _mm256_set1_epi16(pB[2]);
                __m256i _pB3 = _mm256_set1_epi16(pB[3]);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));
                __m512i _s2 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB2));
                __m512i _s3 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB3));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);
                _sum2 = _mm512_add_epi32(_sum2, _s2);
                _sum3 = _mm512_add_epi32(_sum3, _s3);

                pA += 16;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16), _sum1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 32), _sum2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 48), _sum3);
                    outptr0 += 64;
                }
                if (out_elempack == 8)
                {
                    _mm256_storeu_si256((__m256i*)outptr0, _mm512_extracti32x8_epi32(_sum0, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 8), _mm512_extracti32x8_epi32(_sum1, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), _mm512_extracti32x8_epi32(_sum2, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 24), _mm512_extracti32x8_epi32(_sum3, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8), _mm512_extracti32x8_epi32(_sum0, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 8), _mm512_extracti32x8_epi32(_sum1, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 16), _mm512_extracti32x8_epi32(_sum2, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 24), _mm512_extracti32x8_epi32(_sum3, 1));
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    _mm_storeu_si128((__m128i*)outptr0, _mm512_extracti32x4_epi32(_sum0, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 4), _mm512_extracti32x4_epi32(_sum1, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8), _mm512_extracti32x4_epi32(_sum2, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 12), _mm512_extracti32x4_epi32(_sum3, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), _mm512_extracti32x4_epi32(_sum0, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 4), _mm512_extracti32x4_epi32(_sum1, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 8), _mm512_extracti32x4_epi32(_sum2, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 12), _mm512_extracti32x4_epi32(_sum3, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8), _mm512_extracti32x4_epi32(_sum0, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8 + 4), _mm512_extracti32x4_epi32(_sum1, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8 + 8), _mm512_extracti32x4_epi32(_sum2, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8 + 12), _mm512_extracti32x4_epi32(_sum3, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12), _mm512_extracti32x4_epi32(_sum0, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12 + 4), _mm512_extracti32x4_epi32(_sum1, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12 + 8), _mm512_extracti32x4_epi32(_sum2, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12 + 12), _mm512_extracti32x4_epi32(_sum3, 3));
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose16x4_epi32(_sum0, _sum1, _sum2, _sum3);

                    _mm_storeu_si128((__m128i*)outptr0, _mm512_extracti32x4_epi32(_sum0, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep), _mm512_extracti32x4_epi32(_sum0, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2), _mm512_extracti32x4_epi32(_sum0, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3), _mm512_extracti32x4_epi32(_sum0, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), _mm512_extracti32x4_epi32(_sum1, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 5), _mm512_extracti32x4_epi32(_sum1, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 6), _mm512_extracti32x4_epi32(_sum1, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 7), _mm512_extracti32x4_epi32(_sum1, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8), _mm512_extracti32x4_epi32(_sum2, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 9), _mm512_extracti32x4_epi32(_sum2, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 10), _mm512_extracti32x4_epi32(_sum2, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 11), _mm512_extracti32x4_epi32(_sum2, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12), _mm512_extracti32x4_epi32(_sum3, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 13), _mm512_extracti32x4_epi32(_sum3, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 14), _mm512_extracti32x4_epi32(_sum3, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 15), _mm512_extracti32x4_epi32(_sum3, 3));
                    outptr0 += 4;
                }
            }
            else
            {
                _mm512_store_si512((__m512i*)outptr, _sum0);
                _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
                _mm512_store_si512((__m512i*)(outptr + 32), _sum2);
                _mm512_store_si512((__m512i*)(outptr + 48), _sum3);
            }

            outptr += 64;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pB));

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pBBBB = _mm512_cvtepi8_epi16(_pB);

                // 01xx01xx01xx01xx -> 00000000... 11111111...
                __m512i _pB0 = _mm512_shuffle_epi32(_pBBBB, _MM_PERM_AAAA);
                __m512i _pB1 = _mm512_shuffle_epi32(_pBBBB, _MM_PERM_BBBB);

#if __AVX512VNNI__
                _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_dpwssd_epi32(_sum1, _pA0, _pB1);
#else
                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
#endif

                pA += 32;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m256i _pB0 = _mm256_set1_epi16(pB[0]);
                __m256i _pB1 = _mm256_set1_epi16(pB[1]);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);

                pA += 16;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16), _sum1);
                    outptr0 += 32;
                }
                if (out_elempack == 8)
                {
                    _mm256_storeu_si256((__m256i*)outptr0, _mm512_extracti32x8_epi32(_sum0, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + 8), _mm512_extracti32x8_epi32(_sum1, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8), _mm512_extracti32x8_epi32(_sum0, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8 + 8), _mm512_extracti32x8_epi32(_sum1, 1));
                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    _mm_storeu_si128((__m128i*)outptr0, _mm512_extracti32x4_epi32(_sum0, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 4), _mm512_extracti32x4_epi32(_sum1, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), _mm512_extracti32x4_epi32(_sum0, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 4), _mm512_extracti32x4_epi32(_sum1, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8), _mm512_extracti32x4_epi32(_sum0, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8 + 4), _mm512_extracti32x4_epi32(_sum1, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12), _mm512_extracti32x4_epi32(_sum0, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12 + 4), _mm512_extracti32x4_epi32(_sum1, 3));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    int sum0[16];
                    int sum1[16];
                    _mm512_storeu_si512((__m512i*)sum0, _sum0);
                    _mm512_storeu_si512((__m512i*)sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[1] = sum1[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 1 + 1] = sum1[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0[out_hstep * 8] = sum0[8];
                    outptr0[out_hstep * 8 + 1] = sum1[8];
                    outptr0[out_hstep * 9] = sum0[9];
                    outptr0[out_hstep * 9 + 1] = sum1[9];
                    outptr0[out_hstep * 10] = sum0[10];
                    outptr0[out_hstep * 10 + 1] = sum1[10];
                    outptr0[out_hstep * 11] = sum0[11];
                    outptr0[out_hstep * 11 + 1] = sum1[11];
                    outptr0[out_hstep * 12] = sum0[12];
                    outptr0[out_hstep * 12 + 1] = sum1[12];
                    outptr0[out_hstep * 13] = sum0[13];
                    outptr0[out_hstep * 13 + 1] = sum1[13];
                    outptr0[out_hstep * 14] = sum0[14];
                    outptr0[out_hstep * 14 + 1] = sum1[14];
                    outptr0[out_hstep * 15] = sum0[15];
                    outptr0[out_hstep * 15 + 1] = sum1[15];
                    outptr0 += 2;
                }
            }
            else
            {
                _mm512_store_si512((__m512i*)outptr, _sum0);
                _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
            }

            outptr += 32;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m512i _sum0;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pB));

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pBBBB = _mm512_cvtepi8_epi16(_pB);

                // 0xxx0xxx0xxx0xxx -> 00000000...
                __m512i _pB0 = _mm512_shuffle_epi32(_pBBBB, _MM_PERM_AAAA);

#if __AVX512VNNI__
                _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
#else
                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
#endif

                pA += 32;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m256i _pB = _mm256_set1_epi16(pB[0]);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB));

                _sum0 = _mm512_add_epi32(_sum0, _s0);

                pA += 16;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    outptr0 += 16;
                }
                if (out_elempack == 8)
                {
                    _mm256_storeu_si256((__m256i*)outptr0, _mm512_extracti32x8_epi32(_sum0, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8), _mm512_extracti32x8_epi32(_sum0, 1));
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_storeu_si128((__m128i*)outptr0, _mm512_extracti32x4_epi32(_sum0, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), _mm512_extracti32x4_epi32(_sum0, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8), _mm512_extracti32x4_epi32(_sum0, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12), _mm512_extracti32x4_epi32(_sum0, 3));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    int sum0[16];
                    _mm512_storeu_si512((__m512i*)sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0[out_hstep * 8] = sum0[8];
                    outptr0[out_hstep * 9] = sum0[9];
                    outptr0[out_hstep * 10] = sum0[10];
                    outptr0[out_hstep * 11] = sum0[11];
                    outptr0[out_hstep * 12] = sum0[12];
                    outptr0[out_hstep * 13] = sum0[13];
                    outptr0[out_hstep * 14] = sum0[14];
                    outptr0[out_hstep * 15] = sum0[15];
                    outptr0++;
                }
            }
            else
            {
                _mm512_store_si512((__m512i*)outptr, _sum0);
            }

            outptr += 16;
        }

        pAT += max_kk * 16;
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        int* outptr0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const signed char* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

#if __AVX512F__
            __m512i _sum01;
            __m512i _sum23;
            __m512i _sum45;
            __m512i _sum67;
#else
            __m256i _sum0;
            __m256i _sum1;
            __m256i _sum2;
            __m256i _sum3;
            __m256i _sum4;
            __m256i _sum5;
            __m256i _sum6;
            __m256i _sum7;
#endif // __AVX512F__

            if (k == 0)
            {
#if __AVX512F__
                _sum01 = _mm512_setzero_si512();
                _sum23 = _mm512_setzero_si512();
                _sum45 = _mm512_setzero_si512();
                _sum67 = _mm512_setzero_si512();
#else
                _sum0 = _mm256_setzero_si256();
                _sum1 = _mm256_setzero_si256();
                _sum2 = _mm256_setzero_si256();
                _sum3 = _mm256_setzero_si256();
                _sum4 = _mm256_setzero_si256();
                _sum5 = _mm256_setzero_si256();
                _sum6 = _mm256_setzero_si256();
                _sum7 = _mm256_setzero_si256();
#endif // __AVX512F__
            }
            else
            {
#if __AVX512F__
                _sum01 = _mm512_load_si512((const __m512i*)outptr);
                _sum23 = _mm512_load_si512((const __m512i*)(outptr + 16));
                _sum45 = _mm512_load_si512((const __m512i*)(outptr + 32));
                _sum67 = _mm512_load_si512((const __m512i*)(outptr + 48));
#else
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
                _sum1 = _mm256_load_si256((const __m256i*)(outptr + 8));
                _sum2 = _mm256_load_si256((const __m256i*)(outptr + 16));
                _sum3 = _mm256_load_si256((const __m256i*)(outptr + 24));
                _sum4 = _mm256_load_si256((const __m256i*)(outptr + 32));
                _sum5 = _mm256_load_si256((const __m256i*)(outptr + 40));
                _sum6 = _mm256_load_si256((const __m256i*)(outptr + 48));
                _sum7 = _mm256_load_si256((const __m256i*)(outptr + 56));
#endif // __AVX512F__
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pBB = _mm256_cvtepi8_epi16(_pB);

#if __AVX512F__
                __m512i _pAA = _mm512_inserti32x8(_mm512_castsi256_si512(_pA0), _pA0, 1);

                // 0123 4567 -> 0213 4657
                _pBB = _mm256_shuffle_epi32(_pBB, _MM_SHUFFLE(3, 1, 2, 0));

                // 0213 4657 -> 0246 1357
                _pBB = _mm256_permute4x64_epi64(_pBB, _MM_SHUFFLE(3, 1, 2, 0));

                // 0246 1357 -> 0246 0246 1357 1357
                __m512i _pBBBB = _mm512_shuffle_i32x4(_mm512_castsi256_si512(_pBB), _mm512_castsi256_si512(_pBB), _MM_SHUFFLE(1, 1, 0, 0));

                // 0246 0246 1357 1357 -> 00000000 11111111
                __m512i _pB01 = _mm512_shuffle_epi32(_pBBBB, _MM_PERM_AAAA);
                __m512i _pB23 = _mm512_shuffle_epi32(_pBBBB, _MM_PERM_BBBB);
                __m512i _pB45 = _mm512_shuffle_epi32(_pBBBB, _MM_PERM_CCCC);
                __m512i _pB67 = _mm512_shuffle_epi32(_pBBBB, _MM_PERM_DDDD);
#if __AVX512VNNI__
                _sum01 = _mm512_dpwssd_epi32(_sum01, _pAA, _pB01);
                _sum23 = _mm512_dpwssd_epi32(_sum23, _pAA, _pB23);
                _sum45 = _mm512_dpwssd_epi32(_sum45, _pAA, _pB45);
                _sum67 = _mm512_dpwssd_epi32(_sum67, _pAA, _pB67);
#else
                _sum01 = _mm512_add_epi32(_sum01, _mm512_madd_epi16(_pAA, _pB01));
                _sum23 = _mm512_add_epi32(_sum23, _mm512_madd_epi16(_pAA, _pB23));
                _sum45 = _mm512_add_epi32(_sum45, _mm512_madd_epi16(_pAA, _pB45));
                _sum67 = _mm512_add_epi32(_sum67, _mm512_madd_epi16(_pAA, _pB67));
#endif // __AVX512VNNI__
#else  // __AVX512F__
                __m256i _pBBl = _mm256_permute4x64_epi64(_pBB, _MM_SHUFFLE(1, 0, 1, 0));
                __m256i _pBBh = _mm256_permute4x64_epi64(_pBB, _MM_SHUFFLE(3, 2, 3, 2));

                // 01230123 -> 00000000 11111111 22222222 33333333
                __m256i _pB0 = _mm256_shuffle_epi32(_pBBl, _MM_SHUFFLE(0, 0, 0, 0));
                __m256i _pB1 = _mm256_shuffle_epi32(_pBBl, _MM_SHUFFLE(1, 1, 1, 1));
                __m256i _pB2 = _mm256_shuffle_epi32(_pBBl, _MM_SHUFFLE(2, 2, 2, 2));
                __m256i _pB3 = _mm256_shuffle_epi32(_pBBl, _MM_SHUFFLE(3, 3, 3, 3));
                __m256i _pB4 = _mm256_shuffle_epi32(_pBBh, _MM_SHUFFLE(0, 0, 0, 0));
                __m256i _pB5 = _mm256_shuffle_epi32(_pBBh, _MM_SHUFFLE(1, 1, 1, 1));
                __m256i _pB6 = _mm256_shuffle_epi32(_pBBh, _MM_SHUFFLE(2, 2, 2, 2));
                __m256i _pB7 = _mm256_shuffle_epi32(_pBBh, _MM_SHUFFLE(3, 3, 3, 3));

#if __AVXVNNI__
                _sum0 = _mm256_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm256_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm256_dpwssd_epi32(_sum2, _pA0, _pB2);
                _sum3 = _mm256_dpwssd_epi32(_sum3, _pA0, _pB3);
                _sum4 = _mm256_dpwssd_epi32(_sum4, _pA0, _pB4);
                _sum5 = _mm256_dpwssd_epi32(_sum5, _pA0, _pB5);
                _sum6 = _mm256_dpwssd_epi32(_sum6, _pA0, _pB6);
                _sum7 = _mm256_dpwssd_epi32(_sum7, _pA0, _pB7);
#else
                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
                _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_pA0, _pB2));
                _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_pA0, _pB3));
                _sum4 = _mm256_add_epi32(_sum4, _mm256_madd_epi16(_pA0, _pB4));
                _sum5 = _mm256_add_epi32(_sum5, _mm256_madd_epi16(_pA0, _pB5));
                _sum6 = _mm256_add_epi32(_sum6, _mm256_madd_epi16(_pA0, _pB6));
                _sum7 = _mm256_add_epi32(_sum7, _mm256_madd_epi16(_pA0, _pB7));
#endif // __AVXVNNI__
#endif // __AVX512F__

                pA += 16;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);

#if __AVX512F__
                __m256i _pAA = _mm256_inserti128_si256(_mm256_castsi128_si256(_pA), _pA, 1);

                // 01234567 -> 00112233 / 44556677
                __m128i _pB0123 = _mm_unpacklo_epi16(_pB, _pB);
                __m128i _pB4567 = _mm_unpackhi_epi16(_pB, _pB);
                __m256i _pBB = _mm256_inserti128_si256(_mm256_castsi128_si256(_pB0123), _pB4567, 1);

                // 00112233 / 44556677 -> 00001111 44445555 / 22223333 66667777
                __m256i _pB0145 = _mm256_unpacklo_epi32(_pBB, _pBB);
                __m256i _pB2367 = _mm256_unpackhi_epi32(_pBB, _pBB);

                // 00001111 44445555 -> 00000000 11111111 / 44444444 55555555
                __m256i _pB01 = _mm256_permute4x64_epi64(_pB0145, _MM_SHUFFLE(1, 1, 0, 0));
                __m256i _pB23 = _mm256_permute4x64_epi64(_pB2367, _MM_SHUFFLE(1, 1, 0, 0));
                __m256i _pB45 = _mm256_permute4x64_epi64(_pB0145, _MM_SHUFFLE(3, 3, 2, 2));
                __m256i _pB67 = _mm256_permute4x64_epi64(_pB2367, _MM_SHUFFLE(3, 3, 2, 2));

                __m512i _s01 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pAA, _pB01));
                __m512i _s23 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pAA, _pB23));
                __m512i _s45 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pAA, _pB45));
                __m512i _s67 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pAA, _pB67));

                _sum01 = _mm512_add_epi32(_sum01, _s01);
                _sum23 = _mm512_add_epi32(_sum23, _s23);
                _sum45 = _mm512_add_epi32(_sum45, _s45);
                _sum67 = _mm512_add_epi32(_sum67, _s67);
#else
                // 01234567 -> 00112233 / 44556677
                __m128i _pB0123 = _mm_unpacklo_epi16(_pB, _pB);
                __m128i _pB4567 = _mm_unpackhi_epi16(_pB, _pB);

                // 00112233 -> 00001111 / 22223333
                __m128i _pB01 = _mm_unpacklo_epi32(_pB0123, _pB0123);
                __m128i _pB23 = _mm_unpackhi_epi32(_pB0123, _pB0123);
                __m128i _pB45 = _mm_unpacklo_epi32(_pB4567, _pB4567);
                __m128i _pB67 = _mm_unpackhi_epi32(_pB4567, _pB4567);

                // 00001111 -> 00000000 / 11111111
                __m128i _pB0 = _mm_unpacklo_epi64(_pB01, _pB01);
                __m128i _pB1 = _mm_unpackhi_epi64(_pB01, _pB01);
                __m128i _pB2 = _mm_unpacklo_epi64(_pB23, _pB23);
                __m128i _pB3 = _mm_unpackhi_epi64(_pB23, _pB23);
                __m128i _pB4 = _mm_unpacklo_epi64(_pB45, _pB45);
                __m128i _pB5 = _mm_unpackhi_epi64(_pB45, _pB45);
                __m128i _pB6 = _mm_unpacklo_epi64(_pB67, _pB67);
                __m128i _pB7 = _mm_unpackhi_epi64(_pB67, _pB67);

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB1));
                __m256i _s2 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB2));
                __m256i _s3 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB3));
                __m256i _s4 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB4));
                __m256i _s5 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB5));
                __m256i _s6 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB6));
                __m256i _s7 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB7));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);
                _sum2 = _mm256_add_epi32(_sum2, _s2);
                _sum3 = _mm256_add_epi32(_sum3, _s3);
                _sum4 = _mm256_add_epi32(_sum4, _s4);
                _sum5 = _mm256_add_epi32(_sum5, _s5);
                _sum6 = _mm256_add_epi32(_sum6, _s6);
                _sum7 = _mm256_add_epi32(_sum7, _s7);
#endif // __AVX512F__

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
#if __AVX512F__
                    _mm512_storeu_si512((__m512i*)outptr0, _sum01);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16), _sum23);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 32), _sum45);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 48), _sum67);
#else
                    _mm256_storeu_si256((__m256i*)outptr0, _sum0);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 8), _sum1);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), _sum2);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 24), _sum3);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 32), _sum4);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 40), _sum5);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 48), _sum6);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 56), _sum7);
#endif //__AVX512F__
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
#if __AVX512F__
                    _mm_storeu_si128((__m128i*)outptr0, _mm512_extracti32x4_epi32(_sum01, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 4), _mm512_extracti32x4_epi32(_sum01, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8), _mm512_extracti32x4_epi32(_sum23, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 12), _mm512_extracti32x4_epi32(_sum23, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + 16), _mm512_extracti32x4_epi32(_sum45, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 20), _mm512_extracti32x4_epi32(_sum45, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + 24), _mm512_extracti32x4_epi32(_sum67, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 28), _mm512_extracti32x4_epi32(_sum67, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), _mm512_extracti32x4_epi32(_sum01, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 4), _mm512_extracti32x4_epi32(_sum01, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 8), _mm512_extracti32x4_epi32(_sum23, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 12), _mm512_extracti32x4_epi32(_sum23, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 16), _mm512_extracti32x4_epi32(_sum45, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 20), _mm512_extracti32x4_epi32(_sum45, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 24), _mm512_extracti32x4_epi32(_sum67, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 28), _mm512_extracti32x4_epi32(_sum67, 3));
#else
                    _mm_storeu_si128((__m128i*)outptr0, _mm256_extracti128_si256(_sum0, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 4), _mm256_extracti128_si256(_sum1, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8), _mm256_extracti128_si256(_sum2, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 12), _mm256_extracti128_si256(_sum3, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 16), _mm256_extracti128_si256(_sum4, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 20), _mm256_extracti128_si256(_sum5, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 24), _mm256_extracti128_si256(_sum6, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 28), _mm256_extracti128_si256(_sum7, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), _mm256_extracti128_si256(_sum0, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 4), _mm256_extracti128_si256(_sum1, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 8), _mm256_extracti128_si256(_sum2, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 12), _mm256_extracti128_si256(_sum3, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 16), _mm256_extracti128_si256(_sum4, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 20), _mm256_extracti128_si256(_sum5, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 24), _mm256_extracti128_si256(_sum6, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 28), _mm256_extracti128_si256(_sum7, 1));
#endif // __AVX512F__
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m256i _sum0 = _mm512_extracti32x8_epi32(_sum01, 0);
                    __m256i _sum1 = _mm512_extracti32x8_epi32(_sum01, 1);
                    __m256i _sum2 = _mm512_extracti32x8_epi32(_sum23, 0);
                    __m256i _sum3 = _mm512_extracti32x8_epi32(_sum23, 1);
                    __m256i _sum4 = _mm512_extracti32x8_epi32(_sum45, 0);
                    __m256i _sum5 = _mm512_extracti32x8_epi32(_sum45, 1);
                    __m256i _sum6 = _mm512_extracti32x8_epi32(_sum67, 0);
                    __m256i _sum7 = _mm512_extracti32x8_epi32(_sum67, 1);
#endif // __AVX512F__

                    transpose8x8_epi32(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    _mm256_storeu_si256((__m256i*)outptr0, _sum0);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep), _sum1);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 2), _sum2);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 3), _sum3);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), _sum4);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 5), _sum5);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 6), _sum6);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 7), _sum7);
                    outptr0 += 8;
                }
            }
            else
            {
#if __AVX512F__
                _mm512_store_si512((__m512i*)outptr, _sum01);
                _mm512_store_si512((__m512i*)(outptr + 16), _sum23);
                _mm512_store_si512((__m512i*)(outptr + 32), _sum45);
                _mm512_store_si512((__m512i*)(outptr + 48), _sum67);
#else
                _mm256_store_si256((__m256i*)outptr, _sum0);
                _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
                _mm256_store_si256((__m256i*)(outptr + 16), _sum2);
                _mm256_store_si256((__m256i*)(outptr + 24), _sum3);
                _mm256_store_si256((__m256i*)(outptr + 32), _sum4);
                _mm256_store_si256((__m256i*)(outptr + 40), _sum5);
                _mm256_store_si256((__m256i*)(outptr + 48), _sum6);
                _mm256_store_si256((__m256i*)(outptr + 56), _sum7);
#endif // __AVX512F__
            }

            outptr += 64;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m256i _sum0;
            __m256i _sum1;
            __m256i _sum2;
            __m256i _sum3;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_si256();
                _sum1 = _mm256_setzero_si256();
                _sum2 = _mm256_setzero_si256();
                _sum3 = _mm256_setzero_si256();
            }
            else
            {
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
                _sum1 = _mm256_load_si256((const __m256i*)(outptr + 8));
                _sum2 = _mm256_load_si256((const __m256i*)(outptr + 16));
                _sum3 = _mm256_load_si256((const __m256i*)(outptr + 24));
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
                __m256i _pBB = _mm256_inserti128_si256(_mm256_castsi128_si256(_pB), _pB, 1);

                // 01230123 -> 00000000 11111111 22222222 33333333
                __m256i _pB0 = _mm256_shuffle_epi32(_pBB, _MM_SHUFFLE(0, 0, 0, 0));
                __m256i _pB1 = _mm256_shuffle_epi32(_pBB, _MM_SHUFFLE(1, 1, 1, 1));
                __m256i _pB2 = _mm256_shuffle_epi32(_pBB, _MM_SHUFFLE(2, 2, 2, 2));
                __m256i _pB3 = _mm256_shuffle_epi32(_pBB, _MM_SHUFFLE(3, 3, 3, 3));

#if __AVXVNNI__ || __AVX512VNNI__
                _sum0 = _mm256_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm256_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm256_dpwssd_epi32(_sum2, _pA0, _pB2);
                _sum3 = _mm256_dpwssd_epi32(_sum3, _pA0, _pB3);
#else
                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
                _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_pA0, _pB2));
                _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_pA0, _pB3));
#endif

                pA += 16;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB0 = _mm_set1_epi16(pB[0]);
                __m128i _pB1 = _mm_set1_epi16(pB[1]);
                __m128i _pB2 = _mm_set1_epi16(pB[2]);
                __m128i _pB3 = _mm_set1_epi16(pB[3]);

                _pA = _mm_cvtepi8_epi16(_pA);

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB1));
                __m256i _s2 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB2));
                __m256i _s3 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB3));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);
                _sum2 = _mm256_add_epi32(_sum2, _s2);
                _sum3 = _mm256_add_epi32(_sum3, _s3);

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_storeu_si256((__m256i*)outptr0, _sum0);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 8), _sum1);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), _sum2);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 24), _sum3);
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    _mm_storeu_si128((__m128i*)outptr0, _mm256_extracti128_si256(_sum0, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 4), _mm256_extracti128_si256(_sum1, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 8), _mm256_extracti128_si256(_sum2, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 12), _mm256_extracti128_si256(_sum3, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), _mm256_extracti128_si256(_sum0, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 4), _mm256_extracti128_si256(_sum1, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 8), _mm256_extracti128_si256(_sum2, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 12), _mm256_extracti128_si256(_sum3, 1));
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose8x4_epi32(_sum0, _sum1, _sum2, _sum3);

                    _mm_storeu_si128((__m128i*)outptr0, _mm256_extracti128_si256(_sum0, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep), _mm256_extracti128_si256(_sum0, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2), _mm256_extracti128_si256(_sum1, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3), _mm256_extracti128_si256(_sum1, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), _mm256_extracti128_si256(_sum2, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 5), _mm256_extracti128_si256(_sum2, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 6), _mm256_extracti128_si256(_sum3, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 7), _mm256_extracti128_si256(_sum3, 1));
                    outptr0 += 4;
                }
            }
            else
            {
                _mm256_store_si256((__m256i*)outptr, _sum0);
                _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
                _mm256_store_si256((__m256i*)(outptr + 16), _sum2);
                _mm256_store_si256((__m256i*)(outptr + 24), _sum3);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m256i _sum0;
            __m256i _sum1;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_si256();
                _sum1 = _mm256_setzero_si256();
            }
            else
            {
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
                _sum1 = _mm256_load_si256((const __m256i*)(outptr + 8));
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pBB = _mm256_cvtepi8_epi16(_pB);

                // 01xx01xx -> 00000000 11111111
                __m256i _pB0 = _mm256_shuffle_epi32(_pBB, _MM_SHUFFLE(0, 0, 0, 0));
                __m256i _pB1 = _mm256_shuffle_epi32(_pBB, _MM_SHUFFLE(1, 1, 1, 1));

#if __AVXVNNI__ || __AVX512VNNI__
                _sum0 = _mm256_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm256_dpwssd_epi32(_sum1, _pA0, _pB1);
#else
                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
#endif

                pA += 16;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB0 = _mm_set1_epi16(pB[0]);
                __m128i _pB1 = _mm_set1_epi16(pB[1]);

                _pA = _mm_cvtepi8_epi16(_pA);

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB1));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_storeu_si256((__m256i*)outptr0, _sum0);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 8), _sum1);
                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    _mm_storeu_si128((__m128i*)outptr0, _mm256_extracti128_si256(_sum0, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + 4), _mm256_extracti128_si256(_sum1, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), _mm256_extracti128_si256(_sum0, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4 + 4), _mm256_extracti128_si256(_sum1, 1));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    int sum0[8];
                    int sum1[8];
                    _mm256_storeu_si256((__m256i*)sum0, _sum0);
                    _mm256_storeu_si256((__m256i*)sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[1] = sum1[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 1 + 1] = sum1[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0 += 2;
                }
            }
            else
            {
                _mm256_store_si256((__m256i*)outptr, _sum0);
                _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m256i _sum0;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_si256();
            }
            else
            {
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pBB = _mm256_cvtepi8_epi16(_pB);

                // 0xxx0xxx -> 00000000 11111111
                __m256i _pB0 = _mm256_shuffle_epi32(_pBB, _MM_SHUFFLE(0, 0, 0, 0));

#if __AVXVNNI__ || __AVX512VNNI__
                _sum0 = _mm256_dpwssd_epi32(_sum0, _pA0, _pB0);
#else
                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
#endif

                pA += 16;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_set1_epi16(pB[0]);

                _pA = _mm_cvtepi8_epi16(_pA);

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB));

                _sum0 = _mm256_add_epi32(_sum0, _s0);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_storeu_si256((__m256i*)outptr0, _sum0);
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_storeu_si128((__m128i*)outptr0, _mm256_extracti128_si256(_sum0, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), _mm256_extracti128_si256(_sum0, 1));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    int sum0[8];
                    _mm256_storeu_si256((__m256i*)sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0++;
                }
            }
            else
            {
                _mm256_store_si256((__m256i*)outptr, _sum0);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        int* outptr0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const signed char* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;
            __m128i _sum4;
            __m128i _sum5;
            __m128i _sum6;
            __m128i _sum7;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
                _sum2 = _mm_setzero_si128();
                _sum3 = _mm_setzero_si128();
                _sum4 = _mm_setzero_si128();
                _sum5 = _mm_setzero_si128();
                _sum6 = _mm_setzero_si128();
                _sum7 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
                _sum2 = _mm_load_si128((const __m128i*)(outptr + 8));
                _sum3 = _mm_load_si128((const __m128i*)(outptr + 12));
                _sum4 = _mm_load_si128((const __m128i*)(outptr + 16));
                _sum5 = _mm_load_si128((const __m128i*)(outptr + 20));
                _sum6 = _mm_load_si128((const __m128i*)(outptr + 24));
                _sum7 = _mm_load_si128((const __m128i*)(outptr + 28));
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif
                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pBl = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pBh = _mm_unpackhi_epi8(_pB, _extpB);

                // 0123 -> 0000 1111 2222 3333
                __m128i _pB0 = _mm_shuffle_epi32(_pBl, _MM_SHUFFLE(0, 0, 0, 0));
                __m128i _pB1 = _mm_shuffle_epi32(_pBl, _MM_SHUFFLE(1, 1, 1, 1));
                __m128i _pB2 = _mm_shuffle_epi32(_pBl, _MM_SHUFFLE(2, 2, 2, 2));
                __m128i _pB3 = _mm_shuffle_epi32(_pBl, _MM_SHUFFLE(3, 3, 3, 3));
                __m128i _pB4 = _mm_shuffle_epi32(_pBh, _MM_SHUFFLE(0, 0, 0, 0));
                __m128i _pB5 = _mm_shuffle_epi32(_pBh, _MM_SHUFFLE(1, 1, 1, 1));
                __m128i _pB6 = _mm_shuffle_epi32(_pBh, _MM_SHUFFLE(2, 2, 2, 2));
                __m128i _pB7 = _mm_shuffle_epi32(_pBh, _MM_SHUFFLE(3, 3, 3, 3));

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA, _pB1, _sum1);
                _sum2 = _mm_maddd_epi16(_pA, _pB2, _sum2);
                _sum3 = _mm_maddd_epi16(_pA, _pB3, _sum3);
                _sum4 = _mm_maddd_epi16(_pA, _pB4, _sum4);
                _sum5 = _mm_maddd_epi16(_pA, _pB5, _sum5);
                _sum6 = _mm_maddd_epi16(_pA, _pB6, _sum6);
                _sum7 = _mm_maddd_epi16(_pA, _pB7, _sum7);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA, _pB1));
                _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_pA, _pB2));
                _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_pA, _pB3));
                _sum4 = _mm_add_epi32(_sum4, _mm_madd_epi16(_pA, _pB4));
                _sum5 = _mm_add_epi32(_sum5, _mm_madd_epi16(_pA, _pB5));
                _sum6 = _mm_add_epi32(_sum6, _mm_madd_epi16(_pA, _pB6));
                _sum7 = _mm_add_epi32(_sum7, _mm_madd_epi16(_pA, _pB7));
#endif

                pA += 8;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB0 = _mm_set1_epi16(pB[0]);
                __m128i _pB1 = _mm_set1_epi16(pB[1]);
                __m128i _pB2 = _mm_set1_epi16(pB[2]);
                __m128i _pB3 = _mm_set1_epi16(pB[3]);
                __m128i _pB4 = _mm_set1_epi16(pB[4]);
                __m128i _pB5 = _mm_set1_epi16(pB[5]);
                __m128i _pB6 = _mm_set1_epi16(pB[6]);
                __m128i _pB7 = _mm_set1_epi16(pB[7]);
                __m128i _pB01 = _mm_unpacklo_epi64(_pB0, _pB1);
                __m128i _pB23 = _mm_unpacklo_epi64(_pB2, _pB3);
                __m128i _pB45 = _mm_unpacklo_epi64(_pB4, _pB5);
                __m128i _pB67 = _mm_unpacklo_epi64(_pB6, _pB7);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

                __m128i _sl0 = _mm_mullo_epi16(_pA, _pB01);
                __m128i _sh0 = _mm_mulhi_epi16(_pA, _pB01);
                __m128i _sl1 = _mm_mullo_epi16(_pA, _pB23);
                __m128i _sh1 = _mm_mulhi_epi16(_pA, _pB23);
                __m128i _sl2 = _mm_mullo_epi16(_pA, _pB45);
                __m128i _sh2 = _mm_mulhi_epi16(_pA, _pB45);
                __m128i _sl3 = _mm_mullo_epi16(_pA, _pB67);
                __m128i _sh3 = _mm_mulhi_epi16(_pA, _pB67);
                __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                __m128i _s1 = _mm_unpackhi_epi16(_sl0, _sh0);
                __m128i _s2 = _mm_unpacklo_epi16(_sl1, _sh1);
                __m128i _s3 = _mm_unpackhi_epi16(_sl1, _sh1);
                __m128i _s4 = _mm_unpacklo_epi16(_sl2, _sh2);
                __m128i _s5 = _mm_unpackhi_epi16(_sl2, _sh2);
                __m128i _s6 = _mm_unpacklo_epi16(_sl3, _sh3);
                __m128i _s7 = _mm_unpackhi_epi16(_sl3, _sh3);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
                _sum2 = _mm_add_epi32(_sum2, _s2);
                _sum3 = _mm_add_epi32(_sum3, _s3);
                _sum4 = _mm_add_epi32(_sum4, _s4);
                _sum5 = _mm_add_epi32(_sum5, _s5);
                _sum6 = _mm_add_epi32(_sum6, _s6);
                _sum7 = _mm_add_epi32(_sum7, _s7);

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_storeu_si128((__m128i*)outptr0, _sum0);
                    _mm_storeu_si128((__m128i*)(outptr0 + 4), _sum1);
                    _mm_storeu_si128((__m128i*)(outptr0 + 8), _sum2);
                    _mm_storeu_si128((__m128i*)(outptr0 + 12), _sum3);
                    _mm_storeu_si128((__m128i*)(outptr0 + 16), _sum4);
                    _mm_storeu_si128((__m128i*)(outptr0 + 20), _sum5);
                    _mm_storeu_si128((__m128i*)(outptr0 + 24), _sum6);
                    _mm_storeu_si128((__m128i*)(outptr0 + 28), _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
                    transpose4x4_epi32(_sum4, _sum5, _sum6, _sum7);

                    _mm_storeu_si128((__m128i*)outptr0, _sum0);
                    _mm_storeu_si128((__m128i*)(outptr0 + 4), _sum4);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep), _sum1);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep + 4), _sum5);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2), _sum2);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2 + 4), _sum6);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3), _sum3);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3 + 4), _sum7);
                    outptr0 += 8;
                }
            }
            else
            {
                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
                _mm_store_si128((__m128i*)(outptr + 8), _sum2);
                _mm_store_si128((__m128i*)(outptr + 12), _sum3);
                _mm_store_si128((__m128i*)(outptr + 16), _sum4);
                _mm_store_si128((__m128i*)(outptr + 20), _sum5);
                _mm_store_si128((__m128i*)(outptr + 24), _sum6);
                _mm_store_si128((__m128i*)(outptr + 28), _sum7);
            }

            outptr += 32;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
                _sum2 = _mm_setzero_si128();
                _sum3 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
                _sum2 = _mm_load_si128((const __m128i*)(outptr + 8));
                _sum3 = _mm_load_si128((const __m128i*)(outptr + 12));
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 0123 -> 0000 1111 2222 3333
                __m128i _pB0 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(0, 0, 0, 0));
                __m128i _pB1 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(1, 1, 1, 1));
                __m128i _pB2 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(2, 2, 2, 2));
                __m128i _pB3 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(3, 3, 3, 3));

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA, _pB1, _sum1);
                _sum2 = _mm_maddd_epi16(_pA, _pB2, _sum2);
                _sum3 = _mm_maddd_epi16(_pA, _pB3, _sum3);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA, _pB1));
                _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_pA, _pB2));
                _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_pA, _pB3));
#endif

                pA += 8;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB0 = _mm_set1_epi16(pB[0]);
                __m128i _pB1 = _mm_set1_epi16(pB[1]);
                __m128i _pB2 = _mm_set1_epi16(pB[2]);
                __m128i _pB3 = _mm_set1_epi16(pB[3]);
                __m128i _pB01 = _mm_unpacklo_epi64(_pB0, _pB1);
                __m128i _pB23 = _mm_unpacklo_epi64(_pB2, _pB3);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

                __m128i _sl0 = _mm_mullo_epi16(_pA, _pB01);
                __m128i _sh0 = _mm_mulhi_epi16(_pA, _pB01);
                __m128i _sl1 = _mm_mullo_epi16(_pA, _pB23);
                __m128i _sh1 = _mm_mulhi_epi16(_pA, _pB23);
                __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                __m128i _s1 = _mm_unpackhi_epi16(_sl0, _sh0);
                __m128i _s2 = _mm_unpacklo_epi16(_sl1, _sh1);
                __m128i _s3 = _mm_unpackhi_epi16(_sl1, _sh1);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
                _sum2 = _mm_add_epi32(_sum2, _s2);
                _sum3 = _mm_add_epi32(_sum3, _s3);

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_storeu_si128((__m128i*)outptr0, _sum0);
                    _mm_storeu_si128((__m128i*)(outptr0 + 4), _sum1);
                    _mm_storeu_si128((__m128i*)(outptr0 + 8), _sum2);
                    _mm_storeu_si128((__m128i*)(outptr0 + 12), _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);

                    _mm_storeu_si128((__m128i*)outptr0, _sum0);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep), _sum1);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2), _sum2);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3), _sum3);
                    outptr0 += 4;
                }
            }
            else
            {
                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
                _mm_store_si128((__m128i*)(outptr + 8), _sum2);
                _mm_store_si128((__m128i*)(outptr + 12), _sum3);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m128i _sum0;
            __m128i _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 01xx -> 0000 1111
                __m128i _pB0 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(0, 0, 0, 0));
                __m128i _pB1 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(1, 1, 1, 1));

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA, _pB1, _sum1);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA, _pB1));
#endif

                pA += 8;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB0 = _mm_set1_epi16(pB[0]);
                __m128i _pB1 = _mm_set1_epi16(pB[1]);
                __m128i _pB = _mm_unpacklo_epi64(_pB0, _pB1);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

                __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_storeu_si128((__m128i*)outptr0, _sum0);
                    _mm_storeu_si128((__m128i*)(outptr0 + 4), _sum1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    int sum0[4];
                    int sum1[4];
                    _mm_storeu_si128((__m128i*)sum0, _sum0);
                    _mm_storeu_si128((__m128i*)sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[1] = sum1[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0 += 2;
                }
            }
            else
            {
                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m128i _sum0;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 0xxx -> 0000
                __m128i _pB0 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(0, 0, 0, 0));

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB0, _sum0);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
#endif

                pA += 8;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_set1_epi16(pB[0]);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

                __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_storeu_si128((__m128i*)outptr0, _sum0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    int sum0[4];
                    _mm_storeu_si128((__m128i*)sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                _mm_store_si128((__m128i*)outptr, _sum0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        int* outptr0 = (int*)top_blob + (i + ii) * out_hstep + j;

        const signed char* pB = pBT;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
                _sum2 = _mm_setzero_si128();
                _sum3 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
                _sum2 = _mm_load_si128((const __m128i*)(outptr + 8));
                _sum3 = _mm_load_si128((const __m128i*)(outptr + 12));
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pBl = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pBh = _mm_unpackhi_epi8(_pB, _extpB);

                __m128i _pB0 = _mm_unpacklo_epi32(_pBl, _pBl);
                __m128i _pB1 = _mm_unpackhi_epi32(_pBl, _pBl);
                __m128i _pB2 = _mm_unpacklo_epi32(_pBh, _pBh);
                __m128i _pB3 = _mm_unpackhi_epi32(_pBh, _pBh);

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA, _pB1, _sum1);
                _sum2 = _mm_maddd_epi16(_pA, _pB2, _sum2);
                _sum3 = _mm_maddd_epi16(_pA, _pB3, _sum3);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA, _pB1));
                _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_pA, _pB2));
                _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_pA, _pB3));
#endif

                pA += 4;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 0xxx -> 0000
                __m128i _pA0 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(0, 0, 0, 0));

                __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);
                __m128i _pB1 = _mm_unpackhi_epi16(_pB, _pB);

                __m128i _sl0 = _mm_mullo_epi16(_pA0, _pB0);
                __m128i _sh0 = _mm_mulhi_epi16(_pA0, _pB0);
                __m128i _sl1 = _mm_mullo_epi16(_pA0, _pB1);
                __m128i _sh1 = _mm_mulhi_epi16(_pA0, _pB1);
                __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                __m128i _s1 = _mm_unpackhi_epi16(_sl0, _sh0);
                __m128i _s2 = _mm_unpacklo_epi16(_sl1, _sh1);
                __m128i _s3 = _mm_unpackhi_epi16(_sl1, _sh1);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
                _sum2 = _mm_add_epi32(_sum2, _s2);
                _sum3 = _mm_add_epi32(_sum3, _s3);

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __m128i _s0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_sum0), _mm_castsi128_ps(_sum1), _MM_SHUFFLE(2, 0, 2, 0)));
                    __m128i _s1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_sum2), _mm_castsi128_ps(_sum3), _MM_SHUFFLE(2, 0, 2, 0)));
                    __m128i _s2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_sum0), _mm_castsi128_ps(_sum1), _MM_SHUFFLE(3, 1, 3, 1)));
                    __m128i _s3 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_sum2), _mm_castsi128_ps(_sum3), _MM_SHUFFLE(3, 1, 3, 1)));

                    _mm_store_si128((__m128i*)outptr0, _s0);
                    _mm_store_si128((__m128i*)(outptr0 + 4), _s1);
                    _mm_store_si128((__m128i*)(outptr0 + out_hstep), _s2);
                    _mm_store_si128((__m128i*)(outptr0 + out_hstep + 4), _s3);
                    outptr0 += 8;
                }
            }
            else
            {
                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
                _mm_store_si128((__m128i*)(outptr + 8), _sum2);
                _mm_store_si128((__m128i*)(outptr + 12), _sum3);
            }

            outptr += 16;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _sum0;
            __m128i _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                __m128i _pB0 = _mm_unpacklo_epi32(_pB, _pB);
                __m128i _pB1 = _mm_unpackhi_epi32(_pB, _pB);

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA, _pB1, _sum1);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA, _pB1));
#endif

                pA += 4;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 0xxx -> 0000
                __m128i _pA0 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(0, 0, 0, 0));

                __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);

                __m128i _sl = _mm_mullo_epi16(_pA0, _pB0);
                __m128i _sh = _mm_mulhi_epi16(_pA0, _pB0);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __m128i _s0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_sum0), _mm_castsi128_ps(_sum1), _MM_SHUFFLE(2, 0, 2, 0)));
                    __m128i _s1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_sum0), _mm_castsi128_ps(_sum1), _MM_SHUFFLE(3, 1, 3, 1)));

                    _mm_store_si128((__m128i*)outptr0, _s0);
                    _mm_store_si128((__m128i*)(outptr0 + out_hstep), _s1);
                    outptr0 += 4;
                }
            }
            else
            {
                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
            }

            outptr += 8;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            int sum00;
            int sum10;
            int sum01;
            int sum11;

            if (k == 0)
            {
                sum00 = 0;
                sum10 = 0;
                sum01 = 0;
                sum11 = 0;
            }
            else
            {
                sum00 = outptr[0];
                sum10 = outptr[1];
                sum01 = outptr[2];
                sum11 = outptr[3];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum00 += pA[0] * pB[0];
                sum00 += pA[1] * pB[1];
                sum10 += pA[2] * pB[0];
                sum10 += pA[3] * pB[1];
                sum01 += pA[0] * pB[2];
                sum01 += pA[1] * pB[3];
                sum11 += pA[2] * pB[2];
                sum11 += pA[3] * pB[3];
                pA += 4;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum10 += pA[1] * pB[0];
                sum01 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];
                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum00;
                    outptr0[1] = sum01;
                    outptr0[out_hstep] = sum10;
                    outptr0[out_hstep + 1] = sum11;
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum10;
                outptr[2] = sum01;
                outptr[3] = sum11;
            }

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[2] * pB[0];
                sum1 += pA[3] * pB[1];
                pA += 4;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[out_hstep] = sum1;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        int* outptr0 = (int*)top_blob + (i + ii) * out_hstep + j;

        const signed char* pB = pBT;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _sum0;
            __m128i _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pB0 = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pB1 = _mm_unpackhi_epi8(_pB, _extpB);

                // 0xxx -> 0000
                __m128i _pA0 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(0, 0, 0, 0));

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA0, _pB1, _sum1);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA0, _pB1));
#endif

                pA += 2;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(pA[0]);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_store_si128((__m128i*)outptr0, _sum0);
                    _mm_store_si128((__m128i*)(outptr0 + 4), _sum1);
                    outptr0 += 8;
                }
            }
            else
            {
                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
            }

            outptr += 8;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _sum0;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 0xxx -> 0000
                __m128i _pA0 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(0, 0, 0, 0));

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA0, _pB, _sum0);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB));
#endif

                pA += 2;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(pA[0]);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_store_si128((__m128i*)outptr0, _sum0);
                    outptr0 += 4;
                }
            }
            else
            {
                _mm_store_si128((__m128i*)outptr, _sum0);
            }

            outptr += 4;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[0] * pB[2];
                sum1 += pA[1] * pB[3];
                pA += 2;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];
                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[1] = sum1;
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            int sum;

            if (k == 0)
            {
                sum = 0;
            }
            else
            {
                sum = outptr[0];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void convolution_im2col_gemm_get_optimal_tile_mnk_int8(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size_int8 = (int)(get_cpu_level2_cache_size() / sizeof(signed char));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
#if __AVX512F__
        int tile_size = (l2_cache_size_int8 - 64) / 16;
#elif __AVX2__
        int tile_size = (l2_cache_size_int8 - 32) / 8;
#elif __SSE2__
        int tile_size = (l2_cache_size_int8 - 16) / 8;
#else
        int tile_size = (l2_cache_size_int8 - 2) / 3;
#endif

#if __AVX512F__
        TILE_K = std::max(16, tile_size / 16 * 16);
#elif __AVX2__
        TILE_K = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
        TILE_K = std::max(4, tile_size / 4 * 4);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __AVX512F__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 15) / 16 * 16);
#elif __AVX2__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#elif __SSE2__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    // solve M
    {
#if __AVX512F__
        int nn_M = (M + 63) / 64;
#elif __AVX2__
        int nn_M = (M + 31) / 32;
#elif __SSE2__
        int nn_M = (M + 15) / 16;
#else
        int nn_M = (M + 7) / 8;
#endif

#if __AVX512F__
        TILE_M = std::max(16, ((M + nn_M - 1) / nn_M + 15) / 16 * 16);
#elif __AVX2__
        TILE_M = std::max(8, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __SSE2__
        TILE_M = std::max(4, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::max(2, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __AVX512F__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 15) / 16 * 16);
#elif __AVX2__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __SSE2__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif

        if (nT > 1)
        {
#if __AVX512F__
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 15) / 16 * 16);
#elif __AVX2__
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#elif __SSE2__
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
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
            tile_size = (l2_cache_size_int8 - TILE_M * TILE_K) / (TILE_M * 4 + TILE_K);
        }

#if __AVX512F__
        TILE_N = std::max(4, tile_size / 4 * 4);
#elif __AVX2__
        TILE_N = std::max(4, tile_size / 4 * 4);
#elif __SSE2__
        TILE_N = std::max(4, tile_size / 4 * 4);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __AVX512F__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#elif __AVX2__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#elif __SSE2__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;

    signed char* pp = B;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                __m128i _r01 = _mm_load_si128((const __m128i*)p0);
                __m128i _r23 = _mm_load_si128((const __m128i*)(p0 + 16));
                __m128i _r45 = _mm_load_si128((const __m128i*)(p0 + 32));
                __m128i _r67 = _mm_load_si128((const __m128i*)(p0 + 48));

                __m128i _t0 = _mm_unpacklo_epi16(_r01, _r23);
                __m128i _t1 = _mm_unpackhi_epi16(_r01, _r23);
                __m128i _t2 = _mm_unpacklo_epi16(_r45, _r67);
                __m128i _t3 = _mm_unpackhi_epi16(_r45, _r67);

                _r01 = _mm_unpacklo_epi16(_t0, _t1);
                _r23 = _mm_unpackhi_epi16(_t0, _t1);
                _r45 = _mm_unpacklo_epi16(_t2, _t3);
                _r67 = _mm_unpackhi_epi16(_t2, _t3);

                __m128i _r0 = _mm_unpacklo_epi64(_r01, _r45);
                __m128i _r1 = _mm_unpackhi_epi64(_r01, _r45);
                __m128i _r2 = _mm_unpacklo_epi64(_r23, _r67);
                __m128i _r3 = _mm_unpackhi_epi64(_r23, _r67);

                _mm_store_si128((__m128i*)pp, _r0);
                _mm_store_si128((__m128i*)(pp + 16), _r1);
                _mm_store_si128((__m128i*)(pp + 32), _r2);
                _mm_store_si128((__m128i*)(pp + 48), _r3);

                pp += 64;
                p0 += bottom_blob.cstep * 8;
            }
        }

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + bottom_blob.cstep));
                __m128i _r01 = _mm_unpacklo_epi8(_r0, _r1);
                _mm_storeu_si128((__m128i*)pp, _r01);
                pp += 16;
                p0 += bottom_blob.cstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                pp += 8;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                __m128i _r01 = _mm_load_si128((const __m128i*)p0);
                __m128i _r23 = _mm_load_si128((const __m128i*)(p0 + 16));

                __m128i _t0 = _mm_unpacklo_epi16(_r01, _r23);
                __m128i _t1 = _mm_unpackhi_epi16(_r01, _r23);

                __m128i _r0 = _mm_unpacklo_epi16(_t0, _t1);
                __m128i _r1 = _mm_unpackhi_epi16(_t0, _t1);

                _mm_store_si128((__m128i*)pp, _r0);
                _mm_store_si128((__m128i*)(pp + 16), _r1);

                pp += 32;
                p0 += bottom_blob.cstep * 8;
            }
        }

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[bottom_blob.cstep + 0];
                pp[2] = p0[1];
                pp[3] = p0[bottom_blob.cstep + 1];
                pp[4] = p0[2];
                pp[5] = p0[bottom_blob.cstep + 2];
                pp[6] = p0[3];
                pp[7] = p0[bottom_blob.cstep + 3];
                pp += 8;
                p0 += bottom_blob.cstep * 2;
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
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __SSE2__
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)p0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)(p0 + 8));
                __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
                _mm_storeu_si128((__m128i*)pp, _r01);
                pp += 16;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __SSE2__

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[bottom_blob.cstep];
                pp[2] = p0[1];
                pp[3] = p0[bottom_blob.cstep + 1];
                pp += 4;
                p0 += bottom_blob.cstep * 2;
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
#if __SSE2__
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)p0));
                pp += 8;
                p0 += bottom_blob.cstep * 8;
            }
        }
#endif // __SSE2__

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

static void convolution_im2col_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_int8(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

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
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
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

        int kk = 0;
        if (elempack == 1)
        {
            for (; kk + 1 < max_kk; kk += 2)
            {
                int p0 = (k + kk) / maxk;
                int p1 = (k + kk + 1) / maxk;
                int uv0 = (k + kk) % maxk;
                int uv1 = (k + kk + 1) % maxk;
                int u0 = uv0 / kernel_w;
                int u1 = uv1 / kernel_w;
                int v0 = uv0 % kernel_w;
                int v1 = uv1 % kernel_w;

                const Mat img0 = bottom_blob.channel(p0);
                const Mat img1 = bottom_blob.channel(p1);

                int x00 = stride_w * dx0 + dilation_w * v0;
                int x01 = stride_w * dx1 + dilation_w * v0;
                int x02 = stride_w * dx2 + dilation_w * v0;
                int x03 = stride_w * dx3 + dilation_w * v0;
                int x04 = stride_w * dx4 + dilation_w * v0;
                int x05 = stride_w * dx5 + dilation_w * v0;
                int x06 = stride_w * dx6 + dilation_w * v0;
                int x07 = stride_w * dx7 + dilation_w * v0;
                int y00 = stride_h * dy0 + dilation_h * u0;
                int y01 = stride_h * dy1 + dilation_h * u0;
                int y02 = stride_h * dy2 + dilation_h * u0;
                int y03 = stride_h * dy3 + dilation_h * u0;
                int y04 = stride_h * dy4 + dilation_h * u0;
                int y05 = stride_h * dy5 + dilation_h * u0;
                int y06 = stride_h * dy6 + dilation_h * u0;
                int y07 = stride_h * dy7 + dilation_h * u0;

                int x10 = stride_w * dx0 + dilation_w * v1;
                int x11 = stride_w * dx1 + dilation_w * v1;
                int x12 = stride_w * dx2 + dilation_w * v1;
                int x13 = stride_w * dx3 + dilation_w * v1;
                int x14 = stride_w * dx4 + dilation_w * v1;
                int x15 = stride_w * dx5 + dilation_w * v1;
                int x16 = stride_w * dx6 + dilation_w * v1;
                int x17 = stride_w * dx7 + dilation_w * v1;
                int y10 = stride_h * dy0 + dilation_h * u1;
                int y11 = stride_h * dy1 + dilation_h * u1;
                int y12 = stride_h * dy2 + dilation_h * u1;
                int y13 = stride_h * dy3 + dilation_h * u1;
                int y14 = stride_h * dy4 + dilation_h * u1;
                int y15 = stride_h * dy5 + dilation_h * u1;
                int y16 = stride_h * dy6 + dilation_h * u1;
                int y17 = stride_h * dy7 + dilation_h * u1;

                const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                const signed char* sptr03 = img0.row<const signed char>(y03) + x03;
                const signed char* sptr04 = img0.row<const signed char>(y04) + x04;
                const signed char* sptr05 = img0.row<const signed char>(y05) + x05;
                const signed char* sptr06 = img0.row<const signed char>(y06) + x06;
                const signed char* sptr07 = img0.row<const signed char>(y07) + x07;

                const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                const signed char* sptr13 = img1.row<const signed char>(y13) + x13;
                const signed char* sptr14 = img1.row<const signed char>(y14) + x14;
                const signed char* sptr15 = img1.row<const signed char>(y15) + x15;
                const signed char* sptr16 = img1.row<const signed char>(y16) + x16;
                const signed char* sptr17 = img1.row<const signed char>(y17) + x17;

                pp[0] = sptr00[0];
                pp[1] = sptr10[0];
                pp[2] = sptr01[0];
                pp[3] = sptr11[0];
                pp[4] = sptr02[0];
                pp[5] = sptr12[0];
                pp[6] = sptr03[0];
                pp[7] = sptr13[0];
                pp[8] = sptr04[0];
                pp[9] = sptr14[0];
                pp[10] = sptr05[0];
                pp[11] = sptr15[0];
                pp[12] = sptr06[0];
                pp[13] = sptr16[0];
                pp[14] = sptr07[0];
                pp[15] = sptr17[0];
                pp += 16;
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

            if (elempack == 8)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)sptr1);
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)sptr2);
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)sptr3);
                __m128i _r4 = _mm_loadl_epi64((const __m128i*)sptr4);
                __m128i _r5 = _mm_loadl_epi64((const __m128i*)sptr5);
                __m128i _r6 = _mm_loadl_epi64((const __m128i*)sptr6);
                __m128i _r7 = _mm_loadl_epi64((const __m128i*)sptr7);
                __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _r23 = _mm_unpacklo_epi16(_r2, _r3);
                __m128i _r45 = _mm_unpacklo_epi16(_r4, _r5);
                __m128i _r67 = _mm_unpacklo_epi16(_r6, _r7);
                _r0 = _mm_unpacklo_epi32(_r01, _r23);
                _r1 = _mm_unpackhi_epi32(_r01, _r23);
                _r2 = _mm_unpacklo_epi32(_r45, _r67);
                _r3 = _mm_unpackhi_epi32(_r45, _r67);
                _r4 = _mm_unpacklo_epi64(_r0, _r2);
                _r5 = _mm_unpackhi_epi64(_r0, _r2);
                _r6 = _mm_unpacklo_epi64(_r1, _r3);
                _r7 = _mm_unpackhi_epi64(_r1, _r3);
                _mm_storeu_si128((__m128i*)pp, _r4);
                _mm_storeu_si128((__m128i*)(pp + 16), _r5);
                _mm_storeu_si128((__m128i*)(pp + 32), _r6);
                _mm_storeu_si128((__m128i*)(pp + 48), _r7);
                pp += 64;
            }
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
#endif // defined(__x86_64__) || defined(_M_X64)
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

        int kk = 0;
        if (elempack == 1)
        {
            for (; kk + 1 < max_kk; kk += 2)
            {
                int p0 = (k + kk) / maxk;
                int p1 = (k + kk + 1) / maxk;
                int uv0 = (k + kk) % maxk;
                int uv1 = (k + kk + 1) % maxk;
                int u0 = uv0 / kernel_w;
                int u1 = uv1 / kernel_w;
                int v0 = uv0 % kernel_w;
                int v1 = uv1 % kernel_w;

                const Mat img0 = bottom_blob.channel(p0);
                const Mat img1 = bottom_blob.channel(p1);

                int x00 = stride_w * dx0 + dilation_w * v0;
                int x01 = stride_w * dx1 + dilation_w * v0;
                int x02 = stride_w * dx2 + dilation_w * v0;
                int x03 = stride_w * dx3 + dilation_w * v0;
                int y00 = stride_h * dy0 + dilation_h * u0;
                int y01 = stride_h * dy1 + dilation_h * u0;
                int y02 = stride_h * dy2 + dilation_h * u0;
                int y03 = stride_h * dy3 + dilation_h * u0;

                int x10 = stride_w * dx0 + dilation_w * v1;
                int x11 = stride_w * dx1 + dilation_w * v1;
                int x12 = stride_w * dx2 + dilation_w * v1;
                int x13 = stride_w * dx3 + dilation_w * v1;
                int y10 = stride_h * dy0 + dilation_h * u1;
                int y11 = stride_h * dy1 + dilation_h * u1;
                int y12 = stride_h * dy2 + dilation_h * u1;
                int y13 = stride_h * dy3 + dilation_h * u1;

                const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                const signed char* sptr03 = img0.row<const signed char>(y03) + x03;

                const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                const signed char* sptr13 = img1.row<const signed char>(y13) + x13;

                pp[0] = sptr00[0];
                pp[1] = sptr10[0];
                pp[2] = sptr01[0];
                pp[3] = sptr11[0];
                pp[4] = sptr02[0];
                pp[5] = sptr12[0];
                pp[6] = sptr03[0];
                pp[7] = sptr13[0];
                pp += 8;
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

            if (elempack == 8)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)sptr1);
                __m128i _r2 = _mm_loadl_epi64((const __m128i*)sptr2);
                __m128i _r3 = _mm_loadl_epi64((const __m128i*)sptr3);
                __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
                __m128i _r23 = _mm_unpacklo_epi16(_r2, _r3);
                _r0 = _mm_unpacklo_epi32(_r01, _r23);
                _r1 = _mm_unpackhi_epi32(_r01, _r23);
                _mm_storeu_si128((__m128i*)pp, _r0);
                _mm_storeu_si128((__m128i*)(pp + 16), _r1);
                pp += 32;
            }
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
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;

        int kk = 0;
        if (elempack == 1)
        {
            for (; kk + 1 < max_kk; kk += 2)
            {
                int p0 = (k + kk) / maxk;
                int p1 = (k + kk + 1) / maxk;
                int uv0 = (k + kk) % maxk;
                int uv1 = (k + kk + 1) % maxk;
                int u0 = uv0 / kernel_w;
                int u1 = uv1 / kernel_w;
                int v0 = uv0 % kernel_w;
                int v1 = uv1 % kernel_w;

                const Mat img0 = bottom_blob.channel(p0);
                const Mat img1 = bottom_blob.channel(p1);

                int x00 = stride_w * dx0 + dilation_w * v0;
                int x01 = stride_w * dx1 + dilation_w * v0;
                int y00 = stride_h * dy0 + dilation_h * u0;
                int y01 = stride_h * dy1 + dilation_h * u0;
                int x10 = stride_w * dx0 + dilation_w * v1;
                int x11 = stride_w * dx1 + dilation_w * v1;
                int y10 = stride_h * dy0 + dilation_h * u1;
                int y11 = stride_h * dy1 + dilation_h * u1;

                const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                const signed char* sptr11 = img1.row<const signed char>(y11) + x11;

                pp[0] = sptr00[0];
                pp[1] = sptr10[0];
                pp[2] = sptr01[0];
                pp[3] = sptr11[0];
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
            int y0 = stride_h * dy0 + dilation_h * u;
            int y1 = stride_h * dy1 + dilation_h * u;

            const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
            const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;

#if __SSE2__
            if (elempack == 8)
            {
                __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr0);
                __m128i _r1 = _mm_loadl_epi64((const __m128i*)sptr1);
                __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
                _mm_storeu_si128((__m128i*)pp, _r01);
                pp += 16;
            }
#endif // __SSE2__
            if (elempack == 1)
            {
                pp[0] = sptr0[0];
                pp[1] = sptr1[0];
                pp += 2;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw;
        int dx = (j + jj) % outw;

        int kk = 0;
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

#if __SSE2__
            if (elempack == 8)
            {
                _mm_storel_epi64((__m128i*)pp, _mm_loadl_epi64((const __m128i*)sptr));
                pp += 8;
            }
#endif // __SSE2__
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

static void convolution_im2col_gemm_transform_kernel_int8(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        convolution_im2col_gemm_transform_kernel_int8_avx2(kernel, AT, inch, outch, kernel_w, kernel_h, opt);
        return;
    }
#endif
#endif

    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_int8(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
        elempack = inch % 8 == 0 ? 8 : 1;
    }
#endif // __SSE2__

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

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const signed char* k00 = weight_data_r2.channel(q).row<const signed char>(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
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

static void convolution_im2col_gemm_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512vnni())
    {
        convolution_im2col_gemm_int8_avx512vnni(bottom_blob, top_blob, AT, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, nT, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__
    if (ncnn::cpu_support_x86_avxvnni())
    {
        convolution_im2col_gemm_int8_avxvnni(bottom_blob, top_blob, AT, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, nT, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        convolution_im2col_gemm_int8_avx2(bottom_blob, top_blob, AT, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, nT, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__
    if (ncnn::cpu_support_x86_xop())
    {
        convolution_im2col_gemm_int8_xop(bottom_blob, top_blob, AT, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, nT, opt);
        return;
    }
#endif
#endif

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

    Mat topT_tileX;
    if (K > TILE_K)
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat topT_tile;
        if (K > TILE_K)
            topT_tile = topT_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = k + TILE_K >= K;

                convolution_gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
            }
        }
    }
}
