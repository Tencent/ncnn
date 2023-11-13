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
#endif // defined(__x86_64__) || defined(_M_X64)
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
        for (; jj + 15 < max_jj; jj += 16)
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
            __m512i _sum8;
            __m512i _sum9;
            __m512i _suma;
            __m512i _sumb;
            __m512i _sumc;
            __m512i _sumd;
            __m512i _sume;
            __m512i _sumf;

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
                _sum8 = _mm512_setzero_si512();
                _sum9 = _mm512_setzero_si512();
                _suma = _mm512_setzero_si512();
                _sumb = _mm512_setzero_si512();
                _sumc = _mm512_setzero_si512();
                _sumd = _mm512_setzero_si512();
                _sume = _mm512_setzero_si512();
                _sumf = _mm512_setzero_si512();
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
                _sum8 = _mm512_load_si512((const __m512i*)(outptr + 128));
                _sum9 = _mm512_load_si512((const __m512i*)(outptr + 128 + 16));
                _suma = _mm512_load_si512((const __m512i*)(outptr + 128 + 32));
                _sumb = _mm512_load_si512((const __m512i*)(outptr + 128 + 48));
                _sumc = _mm512_load_si512((const __m512i*)(outptr + 128 + 64));
                _sumd = _mm512_load_si512((const __m512i*)(outptr + 128 + 80));
                _sume = _mm512_load_si512((const __m512i*)(outptr + 128 + 96));
                _sumf = _mm512_load_si512((const __m512i*)(outptr + 128 + 112));
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 4567 89ab cdef
                // cdef 0123 4567 89ab
                // 89ab cdef 0123 4567
                // 4567 89ab cdef 0123
                __m512i _pA1 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(2, 1, 0, 3));
                __m512i _pA2 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pA3 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(0, 3, 2, 1));

                // 0123 4567 89ab cdef
                // 1230 5674 9ab8 defc
                // 2301 6745 ab89 efcd
                // 3012 7456 b89a fcde
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_shuffle_epi32(_pB0, _MM_PERM_BADC);
                __m512i _pB3 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CBAD);

#if __AVX512VNNI__
                _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm512_dpwssd_epi32(_sum2, _pA0, _pB2);
                _sum3 = _mm512_dpwssd_epi32(_sum3, _pA0, _pB3);
                _sum4 = _mm512_dpwssd_epi32(_sum4, _pA1, _pB0);
                _sum5 = _mm512_dpwssd_epi32(_sum5, _pA1, _pB1);
                _sum6 = _mm512_dpwssd_epi32(_sum6, _pA1, _pB2);
                _sum7 = _mm512_dpwssd_epi32(_sum7, _pA1, _pB3);
                _sum8 = _mm512_dpwssd_epi32(_sum8, _pA2, _pB0);
                _sum9 = _mm512_dpwssd_epi32(_sum9, _pA2, _pB1);
                _suma = _mm512_dpwssd_epi32(_suma, _pA2, _pB2);
                _sumb = _mm512_dpwssd_epi32(_sumb, _pA2, _pB3);
                _sumc = _mm512_dpwssd_epi32(_sumc, _pA3, _pB0);
                _sumd = _mm512_dpwssd_epi32(_sumd, _pA3, _pB1);
                _sume = _mm512_dpwssd_epi32(_sume, _pA3, _pB2);
                _sumf = _mm512_dpwssd_epi32(_sumf, _pA3, _pB3);
#else
                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA0, _pB2));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA0, _pB3));
                _sum4 = _mm512_add_epi32(_sum4, _mm512_madd_epi16(_pA1, _pB0));
                _sum5 = _mm512_add_epi32(_sum5, _mm512_madd_epi16(_pA1, _pB1));
                _sum6 = _mm512_add_epi32(_sum6, _mm512_madd_epi16(_pA1, _pB2));
                _sum7 = _mm512_add_epi32(_sum7, _mm512_madd_epi16(_pA1, _pB3));
                _sum8 = _mm512_add_epi32(_sum8, _mm512_madd_epi16(_pA2, _pB0));
                _sum9 = _mm512_add_epi32(_sum9, _mm512_madd_epi16(_pA2, _pB1));
                _suma = _mm512_add_epi32(_suma, _mm512_madd_epi16(_pA2, _pB2));
                _sumb = _mm512_add_epi32(_sumb, _mm512_madd_epi16(_pA2, _pB3));
                _sumc = _mm512_add_epi32(_sumc, _mm512_madd_epi16(_pA3, _pB0));
                _sumd = _mm512_add_epi32(_sumd, _mm512_madd_epi16(_pA3, _pB1));
                _sume = _mm512_add_epi32(_sume, _mm512_madd_epi16(_pA3, _pB2));
                _sumf = _mm512_add_epi32(_sumf, _mm512_madd_epi16(_pA3, _pB3));
#endif

                pA += 32;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 01234567 89abcdef
                // cdef0123 456789ab
                // 89abcdef 01234567
                // 456789ab cdef0123
                __m256i _pA1 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(2, 1, 0, 3));
                __m256i _pA2 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pA3 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(0, 3, 2, 1));

                // 01234567 89abcdef
                // 12305674 9ab8defc
                // 23016745 ab89efcd
                // 30127456 b89afcde
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 3, 0, 1));
                __m256i _pB3 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(2, 1, 0, 3)), _MM_SHUFFLE(2, 1, 0, 3));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));
                __m512i _s2 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB2));
                __m512i _s3 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB3));
                __m512i _s4 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0));
                __m512i _s5 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1));
                __m512i _s6 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB2));
                __m512i _s7 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB3));
                __m512i _s8 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA2, _pB0));
                __m512i _s9 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA2, _pB1));
                __m512i _sa = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA2, _pB2));
                __m512i _sb = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA2, _pB3));
                __m512i _sc = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA3, _pB0));
                __m512i _sd = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA3, _pB1));
                __m512i _se = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA3, _pB2));
                __m512i _sf = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA3, _pB3));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);
                _sum2 = _mm512_add_epi32(_sum2, _s2);
                _sum3 = _mm512_add_epi32(_sum3, _s3);
                _sum4 = _mm512_add_epi32(_sum4, _s4);
                _sum5 = _mm512_add_epi32(_sum5, _s5);
                _sum6 = _mm512_add_epi32(_sum6, _s6);
                _sum7 = _mm512_add_epi32(_sum7, _s7);
                _sum8 = _mm512_add_epi32(_sum8, _s8);
                _sum9 = _mm512_add_epi32(_sum9, _s9);
                _suma = _mm512_add_epi32(_suma, _sa);
                _sumb = _mm512_add_epi32(_sumb, _sb);
                _sumc = _mm512_add_epi32(_sumc, _sc);
                _sumd = _mm512_add_epi32(_sumd, _sd);
                _sume = _mm512_add_epi32(_sume, _se);
                _sumf = _mm512_add_epi32(_sumf, _sf);

                pA += 16;
                pB += 16;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff
                    //      01 12 23 30 45 56 67 74 89 9a ab b8 cd de ef fc
                    //      02 13 20 31 46 57 64 75 8a 9b a8 b9 ce df ec fd
                    //      03 10 21 32 47 54 65 76 8b 98 a9 ba cf dc ed fe
                    //      c0 d1 e2 f3 04 15 26 37 48 59 6a 7b 8c 9d ae bf
                    //      c1 d2 e3 f0 05 16 27 34 49 5a 6b 78 8d 9e af bc
                    //      c2 d3 e0 f1 06 17 24 35 4a 5b 68 79 8e 9f ac bd
                    //      c3 d0 e1 f2 07 14 25 36 4b 58 69 7a 8f 9c ad be
                    //      80 91 a2 b3 c4 d5 e6 f7 08 19 2a 3b 4c 5d 6e 7f
                    //      81 92 a3 b0 c5 d6 e7 f4 09 1a 2b 38 4d 5e 6f 7c
                    //      82 93 a0 b1 c6 d7 e4 f5 0a 1b 28 39 4e 5f 6c 7d
                    //      83 90 a1 b2 c7 d4 e5 f6 0b 18 29 3a 4f 5c 6d 7e
                    //      40 51 62 73 84 95 a6 b7 c8 d9 ea fb 0c 1d 2e 3f
                    //      41 52 63 70 85 96 a7 b4 c9 da eb f8 0d 1e 2f 3c
                    //      42 53 60 71 86 97 a4 b5 ca db e8 f9 0e 1f 2c 3d
                    //      43 50 61 72 87 94 a5 b6 cb d8 e9 fa 0f 1c 2d 3e
                    // to
                    //      00 10 20 30 44 54 64 74 88 98 a8 b8 cc dc ec fc
                    //      01 11 21 31 45 55 65 75 89 99 a9 b9 cd dd ed fd
                    //      02 12 22 32 46 56 66 76 8a 9a aa ba ce de ee fe
                    //      03 13 23 33 47 57 67 77 8b 9b ab bb cf df ef ff
                    //      c0 d0 e0 f0 04 14 24 34 48 58 68 78 8c 9c ac bc
                    //      c1 d1 e1 f1 05 15 25 35 49 59 69 79 8d 9d ad bd
                    //      c2 d2 e2 f2 06 16 26 36 4a 5a 6a 7a 8e 9e ae be
                    //      c3 d3 e3 f3 07 17 27 37 4b 5b 6b 7b 8f 9f af bf
                    //      80 90 a0 b0 c4 d4 e4 f4 08 18 28 38 4c 5c 6c 7c
                    //      81 91 a1 b1 c5 d5 e5 f5 09 19 29 39 4d 5d 6d 7d
                    //      82 92 a2 b2 c6 d6 e6 f6 0a 1a 2a 3a 4e 5e 6e 7e
                    //      83 93 a3 b3 c7 d7 e7 f7 0b 1b 2b 3b 4f 5f 6f 7f
                    //      40 50 60 70 84 94 a4 b4 c8 d8 e8 f8 0c 1c 2c 3c
                    //      41 51 61 71 85 95 a5 b5 c9 d9 e9 f9 0d 1d 2d 3d
                    //      42 52 62 72 86 96 a6 b6 ca da ea fa 0e 1e 2e 3e
                    //      43 53 63 73 87 97 a7 b7 cb db eb fb 0f 1f 2f 3f
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum2 = _mm512_shuffle_epi32(_sum2, _MM_PERM_BADC);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_ADCB);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum6 = _mm512_shuffle_epi32(_sum6, _MM_PERM_BADC);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_ADCB);
                        _sum9 = _mm512_shuffle_epi32(_sum9, _MM_PERM_CBAD);
                        _suma = _mm512_shuffle_epi32(_suma, _MM_PERM_BADC);
                        _sumb = _mm512_shuffle_epi32(_sumb, _MM_PERM_ADCB);
                        _sumd = _mm512_shuffle_epi32(_sumd, _MM_PERM_CBAD);
                        _sume = _mm512_shuffle_epi32(_sume, _MM_PERM_BADC);
                        _sumf = _mm512_shuffle_epi32(_sumf, _MM_PERM_ADCB);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum7);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum7);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum5);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum5);
                        __m512i _tmp8 = _mm512_unpacklo_epi32(_sum8, _sumb);
                        __m512i _tmp9 = _mm512_unpackhi_epi32(_sum8, _sumb);
                        __m512i _tmpa = _mm512_unpacklo_epi32(_suma, _sum9);
                        __m512i _tmpb = _mm512_unpackhi_epi32(_suma, _sum9);
                        __m512i _tmpc = _mm512_unpacklo_epi32(_sumc, _sumf);
                        __m512i _tmpd = _mm512_unpackhi_epi32(_sumc, _sumf);
                        __m512i _tmpe = _mm512_unpacklo_epi32(_sume, _sumd);
                        __m512i _tmpf = _mm512_unpackhi_epi32(_sume, _sumd);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum8 = _mm512_unpacklo_epi64(_tmp8, _tmpa);
                        _sum9 = _mm512_unpackhi_epi64(_tmp8, _tmpa);
                        _suma = _mm512_unpacklo_epi64(_tmpb, _tmp9);
                        _sumb = _mm512_unpackhi_epi64(_tmpb, _tmp9);
                        _sumc = _mm512_unpacklo_epi64(_tmpc, _tmpe);
                        _sumd = _mm512_unpackhi_epi64(_tmpc, _tmpe);
                        _sume = _mm512_unpacklo_epi64(_tmpf, _tmpd);
                        _sumf = _mm512_unpackhi_epi64(_tmpf, _tmpd);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                        _sum9 = _mm512_shuffle_epi32(_sum9, _MM_PERM_CBAD);
                        _sumb = _mm512_shuffle_epi32(_sumb, _MM_PERM_CBAD);
                        _sumd = _mm512_shuffle_epi32(_sumd, _MM_PERM_CBAD);
                        _sumf = _mm512_shuffle_epi32(_sumf, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sumc, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum4, _sum0, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum8, _sum4, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sumc, _sum8, _MM_SHUFFLE(1, 3, 1, 3));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sum1, _sumd, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_sum5, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sum9, _sum5, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sumd, _sum9, _MM_SHUFFLE(1, 3, 1, 3));
                    __m512i _tmp8 = _mm512_shuffle_i32x4(_sum2, _sume, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp9 = _mm512_shuffle_i32x4(_sum6, _sum2, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpa = _mm512_shuffle_i32x4(_suma, _sum6, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmpb = _mm512_shuffle_i32x4(_sume, _suma, _MM_SHUFFLE(1, 3, 1, 3));
                    __m512i _tmpc = _mm512_shuffle_i32x4(_sum3, _sumf, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmpd = _mm512_shuffle_i32x4(_sum7, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpe = _mm512_shuffle_i32x4(_sumb, _sum7, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmpf = _mm512_shuffle_i32x4(_sumf, _sumb, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp4, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp8, _tmpa, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmpc, _tmpe, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum4 = _mm512_shuffle_i32x4(_tmp1, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum5 = _mm512_shuffle_i32x4(_tmp5, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum6 = _mm512_shuffle_i32x4(_tmp9, _tmpb, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum7 = _mm512_shuffle_i32x4(_tmpd, _tmpf, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum8 = _mm512_shuffle_i32x4(_tmp2, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum9 = _mm512_shuffle_i32x4(_tmp6, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                    _suma = _mm512_shuffle_i32x4(_tmpa, _tmp8, _MM_SHUFFLE(3, 1, 2, 0));
                    _sumb = _mm512_shuffle_i32x4(_tmpe, _tmpc, _MM_SHUFFLE(3, 1, 2, 0));
                    _sumc = _mm512_shuffle_i32x4(_tmp3, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                    _sumd = _mm512_shuffle_i32x4(_tmp7, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                    _sume = _mm512_shuffle_i32x4(_tmpb, _tmp9, _MM_SHUFFLE(3, 1, 2, 0));
                    _sumf = _mm512_shuffle_i32x4(_tmpf, _tmpd, _MM_SHUFFLE(3, 1, 2, 0));

                    _mm512_store_si512((__m512i*)outptr0, _sum0);
                    _mm512_store_si512((__m512i*)(outptr0 + 16), _sum1);
                    _mm512_store_si512((__m512i*)(outptr0 + 32), _sum2);
                    _mm512_store_si512((__m512i*)(outptr0 + 48), _sum3);
                    _mm512_store_si512((__m512i*)(outptr0 + 64), _sum4);
                    _mm512_store_si512((__m512i*)(outptr0 + 80), _sum5);
                    _mm512_store_si512((__m512i*)(outptr0 + 96), _sum6);
                    _mm512_store_si512((__m512i*)(outptr0 + 112), _sum7);
                    _mm512_store_si512((__m512i*)(outptr0 + 128), _sum8);
                    _mm512_store_si512((__m512i*)(outptr0 + 128 + 16), _sum9);
                    _mm512_store_si512((__m512i*)(outptr0 + 128 + 32), _suma);
                    _mm512_store_si512((__m512i*)(outptr0 + 128 + 48), _sumb);
                    _mm512_store_si512((__m512i*)(outptr0 + 128 + 64), _sumc);
                    _mm512_store_si512((__m512i*)(outptr0 + 128 + 80), _sumd);
                    _mm512_store_si512((__m512i*)(outptr0 + 128 + 96), _sume);
                    _mm512_store_si512((__m512i*)(outptr0 + 128 + 112), _sumf);
                    outptr0 += 256;
                }
                if (out_elempack == 8)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff
                    //      01 12 23 30 45 56 67 74 89 9a ab b8 cd de ef fc
                    //      02 13 20 31 46 57 64 75 8a 9b a8 b9 ce df ec fd
                    //      03 10 21 32 47 54 65 76 8b 98 a9 ba cf dc ed fe
                    //      c0 d1 e2 f3 04 15 26 37 48 59 6a 7b 8c 9d ae bf
                    //      c1 d2 e3 f0 05 16 27 34 49 5a 6b 78 8d 9e af bc
                    //      c2 d3 e0 f1 06 17 24 35 4a 5b 68 79 8e 9f ac bd
                    //      c3 d0 e1 f2 07 14 25 36 4b 58 69 7a 8f 9c ad be
                    //      80 91 a2 b3 c4 d5 e6 f7 08 19 2a 3b 4c 5d 6e 7f
                    //      81 92 a3 b0 c5 d6 e7 f4 09 1a 2b 38 4d 5e 6f 7c
                    //      82 93 a0 b1 c6 d7 e4 f5 0a 1b 28 39 4e 5f 6c 7d
                    //      83 90 a1 b2 c7 d4 e5 f6 0b 18 29 3a 4f 5c 6d 7e
                    //      40 51 62 73 84 95 a6 b7 c8 d9 ea fb 0c 1d 2e 3f
                    //      41 52 63 70 85 96 a7 b4 c9 da eb f8 0d 1e 2f 3c
                    //      42 53 60 71 86 97 a4 b5 ca db e8 f9 0e 1f 2c 3d
                    //      43 50 61 72 87 94 a5 b6 cb d8 e9 fa 0f 1c 2d 3e
                    // to
                    //      00 10 20 30 44 54 64 74 88 98 a8 b8 cc dc ec fc
                    //      01 11 21 31 45 55 65 75 89 99 a9 b9 cd dd ed fd
                    //      02 12 22 32 46 56 66 76 8a 9a aa ba ce de ee fe
                    //      03 13 23 33 47 57 67 77 8b 9b ab bb cf df ef ff
                    //      c0 d0 e0 f0 04 14 24 34 48 58 68 78 8c 9c ac bc
                    //      c1 d1 e1 f1 05 15 25 35 49 59 69 79 8d 9d ad bd
                    //      c2 d2 e2 f2 06 16 26 36 4a 5a 6a 7a 8e 9e ae be
                    //      c3 d3 e3 f3 07 17 27 37 4b 5b 6b 7b 8f 9f af bf
                    //      80 90 a0 b0 c4 d4 e4 f4 08 18 28 38 4c 5c 6c 7c
                    //      81 91 a1 b1 c5 d5 e5 f5 09 19 29 39 4d 5d 6d 7d
                    //      82 92 a2 b2 c6 d6 e6 f6 0a 1a 2a 3a 4e 5e 6e 7e
                    //      83 93 a3 b3 c7 d7 e7 f7 0b 1b 2b 3b 4f 5f 6f 7f
                    //      40 50 60 70 84 94 a4 b4 c8 d8 e8 f8 0c 1c 2c 3c
                    //      41 51 61 71 85 95 a5 b5 c9 d9 e9 f9 0d 1d 2d 3d
                    //      42 52 62 72 86 96 a6 b6 ca da ea fa 0e 1e 2e 3e
                    //      43 53 63 73 87 97 a7 b7 cb db eb fb 0f 1f 2f 3f
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum2 = _mm512_shuffle_epi32(_sum2, _MM_PERM_BADC);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_ADCB);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum6 = _mm512_shuffle_epi32(_sum6, _MM_PERM_BADC);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_ADCB);
                        _sum9 = _mm512_shuffle_epi32(_sum9, _MM_PERM_CBAD);
                        _suma = _mm512_shuffle_epi32(_suma, _MM_PERM_BADC);
                        _sumb = _mm512_shuffle_epi32(_sumb, _MM_PERM_ADCB);
                        _sumd = _mm512_shuffle_epi32(_sumd, _MM_PERM_CBAD);
                        _sume = _mm512_shuffle_epi32(_sume, _MM_PERM_BADC);
                        _sumf = _mm512_shuffle_epi32(_sumf, _MM_PERM_ADCB);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum7);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum7);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum5);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum5);
                        __m512i _tmp8 = _mm512_unpacklo_epi32(_sum8, _sumb);
                        __m512i _tmp9 = _mm512_unpackhi_epi32(_sum8, _sumb);
                        __m512i _tmpa = _mm512_unpacklo_epi32(_suma, _sum9);
                        __m512i _tmpb = _mm512_unpackhi_epi32(_suma, _sum9);
                        __m512i _tmpc = _mm512_unpacklo_epi32(_sumc, _sumf);
                        __m512i _tmpd = _mm512_unpackhi_epi32(_sumc, _sumf);
                        __m512i _tmpe = _mm512_unpacklo_epi32(_sume, _sumd);
                        __m512i _tmpf = _mm512_unpackhi_epi32(_sume, _sumd);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum8 = _mm512_unpacklo_epi64(_tmp8, _tmpa);
                        _sum9 = _mm512_unpackhi_epi64(_tmp8, _tmpa);
                        _suma = _mm512_unpacklo_epi64(_tmpb, _tmp9);
                        _sumb = _mm512_unpackhi_epi64(_tmpb, _tmp9);
                        _sumc = _mm512_unpacklo_epi64(_tmpc, _tmpe);
                        _sumd = _mm512_unpackhi_epi64(_tmpc, _tmpe);
                        _sume = _mm512_unpacklo_epi64(_tmpf, _tmpd);
                        _sumf = _mm512_unpackhi_epi64(_tmpf, _tmpd);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                        _sum9 = _mm512_shuffle_epi32(_sum9, _MM_PERM_CBAD);
                        _sumb = _mm512_shuffle_epi32(_sumb, _MM_PERM_CBAD);
                        _sumd = _mm512_shuffle_epi32(_sumd, _MM_PERM_CBAD);
                        _sumf = _mm512_shuffle_epi32(_sumf, _MM_PERM_CBAD);
                    }

                    // TODO
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sumc, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum4, _sum0, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum8, _sum4, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sumc, _sum8, _MM_SHUFFLE(1, 3, 1, 3));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sum1, _sumd, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_sum5, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sum9, _sum5, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sumd, _sum9, _MM_SHUFFLE(1, 3, 1, 3));
                    __m512i _tmp8 = _mm512_shuffle_i32x4(_sum2, _sume, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp9 = _mm512_shuffle_i32x4(_sum6, _sum2, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpa = _mm512_shuffle_i32x4(_suma, _sum6, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmpb = _mm512_shuffle_i32x4(_sume, _suma, _MM_SHUFFLE(1, 3, 1, 3));
                    __m512i _tmpc = _mm512_shuffle_i32x4(_sum3, _sumf, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmpd = _mm512_shuffle_i32x4(_sum7, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpe = _mm512_shuffle_i32x4(_sumb, _sum7, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmpf = _mm512_shuffle_i32x4(_sumf, _sumb, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp4, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp8, _tmpc, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp1, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmp9, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum4 = _mm512_shuffle_i32x4(_tmp2, _tmp6, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_i32x4(_tmpa, _tmpe, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum6 = _mm512_shuffle_i32x4(_tmp3, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum7 = _mm512_shuffle_i32x4(_tmpb, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum8 = _mm512_shuffle_i32x4(_tmp2, _tmp6, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum9 = _mm512_shuffle_i32x4(_tmpa, _tmpe, _MM_SHUFFLE(3, 1, 3, 1));
                    _suma = _mm512_shuffle_i32x4(_tmp3, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                    _sumb = _mm512_shuffle_i32x4(_tmpb, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));
                    _sumc = _mm512_shuffle_i32x4(_tmp0, _tmp4, _MM_SHUFFLE(3, 1, 3, 1));
                    _sumd = _mm512_shuffle_i32x4(_tmp8, _tmpc, _MM_SHUFFLE(3, 1, 3, 1));
                    _sume = _mm512_shuffle_i32x4(_tmp1, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sumf = _mm512_shuffle_i32x4(_tmp9, _tmpc, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16), _sum1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16 * 2), _sum2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16 * 3), _sum3);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16 * 4), _sum4);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16 * 5), _sum5);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16 * 6), _sum6);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16 * 7), _sum7);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8), _sum8);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 16), _sum9);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 16 * 2), _suma);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 16 * 3), _sumb);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 16 * 4), _sumc);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 16 * 5), _sumd);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 16 * 6), _sume);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 16 * 7), _sumf);
                    outptr0 += 128;
                }
                if (out_elempack == 4)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff
                    //      01 12 23 30 45 56 67 74 89 9a ab b8 cd de ef fc
                    //      02 13 20 31 46 57 64 75 8a 9b a8 b9 ce df ec fd
                    //      03 10 21 32 47 54 65 76 8b 98 a9 ba cf dc ed fe
                    //      c0 d1 e2 f3 04 15 26 37 48 59 6a 7b 8c 9d ae bf
                    //      c1 d2 e3 f0 05 16 27 34 49 5a 6b 78 8d 9e af bc
                    //      c2 d3 e0 f1 06 17 24 35 4a 5b 68 79 8e 9f ac bd
                    //      c3 d0 e1 f2 07 14 25 36 4b 58 69 7a 8f 9c ad be
                    //      80 91 a2 b3 c4 d5 e6 f7 08 19 2a 3b 4c 5d 6e 7f
                    //      81 92 a3 b0 c5 d6 e7 f4 09 1a 2b 38 4d 5e 6f 7c
                    //      82 93 a0 b1 c6 d7 e4 f5 0a 1b 28 39 4e 5f 6c 7d
                    //      83 90 a1 b2 c7 d4 e5 f6 0b 18 29 3a 4f 5c 6d 7e
                    //      40 51 62 73 84 95 a6 b7 c8 d9 ea fb 0c 1d 2e 3f
                    //      41 52 63 70 85 96 a7 b4 c9 da eb f8 0d 1e 2f 3c
                    //      42 53 60 71 86 97 a4 b5 ca db e8 f9 0e 1f 2c 3d
                    //      43 50 61 72 87 94 a5 b6 cb d8 e9 fa 0f 1c 2d 3e
                    // to
                    //      00 10 20 30 44 54 64 74 88 98 a8 b8 cc dc ec fc
                    //      01 11 21 31 45 55 65 75 89 99 a9 b9 cd dd ed fd
                    //      02 12 22 32 46 56 66 76 8a 9a aa ba ce de ee fe
                    //      03 13 23 33 47 57 67 77 8b 9b ab bb cf df ef ff
                    //      c0 d0 e0 f0 04 14 24 34 48 58 68 78 8c 9c ac bc
                    //      c1 d1 e1 f1 05 15 25 35 49 59 69 79 8d 9d ad bd
                    //      c2 d2 e2 f2 06 16 26 36 4a 5a 6a 7a 8e 9e ae be
                    //      c3 d3 e3 f3 07 17 27 37 4b 5b 6b 7b 8f 9f af bf
                    //      80 90 a0 b0 c4 d4 e4 f4 08 18 28 38 4c 5c 6c 7c
                    //      81 91 a1 b1 c5 d5 e5 f5 09 19 29 39 4d 5d 6d 7d
                    //      82 92 a2 b2 c6 d6 e6 f6 0a 1a 2a 3a 4e 5e 6e 7e
                    //      83 93 a3 b3 c7 d7 e7 f7 0b 1b 2b 3b 4f 5f 6f 7f
                    //      40 50 60 70 84 94 a4 b4 c8 d8 e8 f8 0c 1c 2c 3c
                    //      41 51 61 71 85 95 a5 b5 c9 d9 e9 f9 0d 1d 2d 3d
                    //      42 52 62 72 86 96 a6 b6 ca da ea fa 0e 1e 2e 3e
                    //      43 53 63 73 87 97 a7 b7 cb db eb fb 0f 1f 2f 3f
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum2 = _mm512_shuffle_epi32(_sum2, _MM_PERM_BADC);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_ADCB);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum6 = _mm512_shuffle_epi32(_sum6, _MM_PERM_BADC);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_ADCB);
                        _sum9 = _mm512_shuffle_epi32(_sum9, _MM_PERM_CBAD);
                        _suma = _mm512_shuffle_epi32(_suma, _MM_PERM_BADC);
                        _sumb = _mm512_shuffle_epi32(_sumb, _MM_PERM_ADCB);
                        _sumd = _mm512_shuffle_epi32(_sumd, _MM_PERM_CBAD);
                        _sume = _mm512_shuffle_epi32(_sume, _MM_PERM_BADC);
                        _sumf = _mm512_shuffle_epi32(_sumf, _MM_PERM_ADCB);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum7);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum7);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum5);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum5);
                        __m512i _tmp8 = _mm512_unpacklo_epi32(_sum8, _sumb);
                        __m512i _tmp9 = _mm512_unpackhi_epi32(_sum8, _sumb);
                        __m512i _tmpa = _mm512_unpacklo_epi32(_suma, _sum9);
                        __m512i _tmpb = _mm512_unpackhi_epi32(_suma, _sum9);
                        __m512i _tmpc = _mm512_unpacklo_epi32(_sumc, _sumf);
                        __m512i _tmpd = _mm512_unpackhi_epi32(_sumc, _sumf);
                        __m512i _tmpe = _mm512_unpacklo_epi32(_sume, _sumd);
                        __m512i _tmpf = _mm512_unpackhi_epi32(_sume, _sumd);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum8 = _mm512_unpacklo_epi64(_tmp8, _tmpa);
                        _sum9 = _mm512_unpackhi_epi64(_tmp8, _tmpa);
                        _suma = _mm512_unpacklo_epi64(_tmpb, _tmp9);
                        _sumb = _mm512_unpackhi_epi64(_tmpb, _tmp9);
                        _sumc = _mm512_unpacklo_epi64(_tmpc, _tmpe);
                        _sumd = _mm512_unpackhi_epi64(_tmpc, _tmpe);
                        _sume = _mm512_unpacklo_epi64(_tmpf, _tmpd);
                        _sumf = _mm512_unpackhi_epi64(_tmpf, _tmpd);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                        _sum9 = _mm512_shuffle_epi32(_sum9, _MM_PERM_CBAD);
                        _sumb = _mm512_shuffle_epi32(_sumb, _MM_PERM_CBAD);
                        _sumd = _mm512_shuffle_epi32(_sumd, _MM_PERM_CBAD);
                        _sumf = _mm512_shuffle_epi32(_sumf, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum4, _sum5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum6, _sum7, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sum8, _sum9, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_suma, _sumb, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sumc, _sumd, _MM_SHUFFLE(1, 3, 1, 3));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sume, _sumf, _MM_SHUFFLE(1, 3, 1, 3));
                    __m512i _tmp8 = _mm512_shuffle_i32x4(_sumc, _sumd, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp9 = _mm512_shuffle_i32x4(_sume, _sumf, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmpa = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpb = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmpc = _mm512_shuffle_i32x4(_sum4, _sum5, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmpd = _mm512_shuffle_i32x4(_sum6, _sum7, _MM_SHUFFLE(0, 2, 0, 2));
                    __m512i _tmpe = _mm512_shuffle_i32x4(_sum8, _sum9, _MM_SHUFFLE(1, 3, 1, 3));
                    __m512i _tmpf = _mm512_shuffle_i32x4(_suma, _sumb, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum4 = _mm512_shuffle_i32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_i32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum6 = _mm512_shuffle_i32x4(_tmpc, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum7 = _mm512_shuffle_i32x4(_tmpe, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum8 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum9 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                    _suma = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sumb = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sumc = _mm512_shuffle_i32x4(_tmpc, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
                    _sumd = _mm512_shuffle_i32x4(_tmpe, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));
                    _sume = _mm512_shuffle_i32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1));
                    _sumf = _mm512_shuffle_i32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16), _sum1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 32), _sum2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 48), _sum3);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4), _sum4);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4 + 16), _sum5);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4 + 32), _sum6);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4 + 48), _sum7);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8), _sum8);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 16), _sum9);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 32), _suma);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 48), _sumb);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 12), _sumc);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 12 + 16), _sumd);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 12 + 32), _sume);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 12 + 48), _sumf);
                    outptr0 += 64;
                }
                if (out_elempack == 1)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff
                    //      01 12 23 30 45 56 67 74 89 9a ab b8 cd de ef fc
                    //      02 13 20 31 46 57 64 75 8a 9b a8 b9 ce df ec fd
                    //      03 10 21 32 47 54 65 76 8b 98 a9 ba cf dc ed fe
                    //      c0 d1 e2 f3 04 15 26 37 48 59 6a 7b 8c 9d ae bf
                    //      c1 d2 e3 f0 05 16 27 34 49 5a 6b 78 8d 9e af bc
                    //      c2 d3 e0 f1 06 17 24 35 4a 5b 68 79 8e 9f ac bd
                    //      c3 d0 e1 f2 07 14 25 36 4b 58 69 7a 8f 9c ad be
                    //      80 91 a2 b3 c4 d5 e6 f7 08 19 2a 3b 4c 5d 6e 7f
                    //      81 92 a3 b0 c5 d6 e7 f4 09 1a 2b 38 4d 5e 6f 7c
                    //      82 93 a0 b1 c6 d7 e4 f5 0a 1b 28 39 4e 5f 6c 7d
                    //      83 90 a1 b2 c7 d4 e5 f6 0b 18 29 3a 4f 5c 6d 7e
                    //      40 51 62 73 84 95 a6 b7 c8 d9 ea fb 0c 1d 2e 3f
                    //      41 52 63 70 85 96 a7 b4 c9 da eb f8 0d 1e 2f 3c
                    //      42 53 60 71 86 97 a4 b5 ca db e8 f9 0e 1f 2c 3d
                    //      43 50 61 72 87 94 a5 b6 cb d8 e9 fa 0f 1c 2d 3e
                    // to
                    //      00 01 02 03 44 45 46 47 88 80 8a 8b cc cd ce cf
                    //      10 11 12 13 54 55 56 57 98 90 9a 9b dc dd de df
                    //      20 21 22 23 64 65 66 67 a8 a0 aa ab ec ed ee ef
                    //      30 31 32 33 74 75 76 77 b8 b0 ba bb fc fd fe ff
                    //      c0 c1 c2 c3 04 05 06 07 48 40 4a 4b 8c 8d 8e 8f
                    //      d0 d1 d2 d3 14 15 16 17 58 50 5a 5b 9c 9d 9e 9f
                    //      e0 e1 e2 e3 24 25 26 27 68 60 6a 6b ac ad ae af
                    //      f0 f1 f2 f3 34 35 36 37 78 70 7a 7b bc bd be bf
                    //      80 81 82 83 c4 c5 c6 c7 08 00 0a 0b 4c 4d 4e 4f
                    //      90 91 92 93 d4 d5 d6 d7 18 10 1a 1b 5c 5d 5e 5f
                    //      a0 a1 a2 a3 e4 e5 e6 e7 28 20 2a 2b 6c 6d 6e 6f
                    //      b0 b1 b2 b3 f4 f5 f6 f7 38 30 3a 3b 7c 7d 7e 7f
                    //      40 41 42 43 84 85 86 87 c8 c0 ca cb 0c 0d 0e 0f
                    //      50 51 52 53 94 95 96 97 d8 d0 da db 1c 1d 1e 1f
                    //      60 61 62 63 a4 a5 a6 a7 e8 e0 ea eb 2c 2d 2e 2f
                    //      70 71 72 73 b4 b5 b6 b7 f8 f0 fa fb 3c 3d 3e 3f
                    {
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum1);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum1);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum3);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum3);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum5);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum5);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum7);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum7);
                        __m512i _tmp8 = _mm512_unpacklo_epi32(_sum8, _sum9);
                        __m512i _tmp9 = _mm512_unpackhi_epi32(_sum8, _sum9);
                        __m512i _tmpa = _mm512_unpacklo_epi32(_suma, _sumb);
                        __m512i _tmpb = _mm512_unpackhi_epi32(_suma, _sumb);
                        __m512i _tmpc = _mm512_unpacklo_epi32(_sumc, _sumd);
                        __m512i _tmpd = _mm512_unpackhi_epi32(_sumc, _sumd);
                        __m512i _tmpe = _mm512_unpacklo_epi32(_sume, _sumf);
                        __m512i _tmpf = _mm512_unpackhi_epi32(_sume, _sumf);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum8 = _mm512_unpacklo_epi64(_tmp8, _tmpa);
                        _sum9 = _mm512_unpackhi_epi64(_tmp8, _tmpa);
                        _suma = _mm512_unpacklo_epi64(_tmpb, _tmp9);
                        _sumb = _mm512_unpackhi_epi64(_tmpb, _tmp9);
                        _sumc = _mm512_unpacklo_epi64(_tmpc, _tmpe);
                        _sumd = _mm512_unpackhi_epi64(_tmpc, _tmpe);
                        _sume = _mm512_unpacklo_epi64(_tmpf, _tmpd);
                        _sumf = _mm512_unpackhi_epi64(_tmpf, _tmpd);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                        _sum9 = _mm512_shuffle_epi32(_sum9, _MM_PERM_CBAD);
                        _sumb = _mm512_shuffle_epi32(_sumb, _MM_PERM_CBAD);
                        _sumd = _mm512_shuffle_epi32(_sumd, _MM_PERM_CBAD);
                        _sumf = _mm512_shuffle_epi32(_sumf, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum4, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum1, _sum5, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum2, _sum6, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum3, _sum7, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sumc, _sum0, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_sumd, _sum1, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sume, _sum2, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sumf, _sum3, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp8 = _mm512_shuffle_i32x4(_sum8, _sumc, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp9 = _mm512_shuffle_i32x4(_sum9, _sumd, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmpa = _mm512_shuffle_i32x4(_suma, _sume, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmpb = _mm512_shuffle_i32x4(_sumb, _sumf, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmpc = _mm512_shuffle_i32x4(_sum4, _sum8, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmpd = _mm512_shuffle_i32x4(_sum5, _sum9, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmpe = _mm512_shuffle_i32x4(_sum6, _suma, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmpf = _mm512_shuffle_i32x4(_sum7, _sumb, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp8, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp1, _tmp9, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp2, _tmpa, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmp3, _tmpb, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum4 = _mm512_shuffle_i32x4(_tmp4, _tmpc, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum5 = _mm512_shuffle_i32x4(_tmp5, _tmpd, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum6 = _mm512_shuffle_i32x4(_tmp6, _tmpe, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum7 = _mm512_shuffle_i32x4(_tmp7, _tmpf, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum8 = _mm512_shuffle_i32x4(_tmp8, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum9 = _mm512_shuffle_i32x4(_tmp9, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                    _suma = _mm512_shuffle_i32x4(_tmpa, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                    _sumb = _mm512_shuffle_i32x4(_tmpb, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                    _sumc = _mm512_shuffle_i32x4(_tmpc, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                    _sumd = _mm512_shuffle_i32x4(_tmpd, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                    _sume = _mm512_shuffle_i32x4(_tmpe, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                    _sumf = _mm512_shuffle_i32x4(_tmpf, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));

                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep), _sum1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 2), _sum2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 3), _sum3);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4), _sum4);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 5), _sum5);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 6), _sum6);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 7), _sum7);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8), _sum8);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 9), _sum9);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 10), _suma);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 11), _sumb);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 12), _sumc);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 13), _sumd);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 14), _sume);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 15), _sumf);
                    outptr0 += 16;
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
                _mm512_store_si512((__m512i*)(outptr + 128), _sum8);
                _mm512_store_si512((__m512i*)(outptr + 128 + 16), _sum9);
                _mm512_store_si512((__m512i*)(outptr + 128 + 32), _suma);
                _mm512_store_si512((__m512i*)(outptr + 128 + 48), _sumb);
                _mm512_store_si512((__m512i*)(outptr + 128 + 64), _sumc);
                _mm512_store_si512((__m512i*)(outptr + 128 + 80), _sumd);
                _mm512_store_si512((__m512i*)(outptr + 128 + 96), _sume);
                _mm512_store_si512((__m512i*)(outptr + 128 + 112), _sumf);
            }

            outptr += 256;
        }
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
                __m256i _pBB = _mm256_cvtepi8_epi16(_pB);

                // 0123 4567 89ab cdef
                // 4567 0123 cdef 89ab
                __m512i _pA1 = _mm512_permutex_epi64(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 4567
                // 0123 4567 0123 4567
                // 1230 5674 1230 5674
                // 2301 6745 2301 6745
                // 3012 7456 3012 7456
                __m512i _pB0 = _mm512_inserti32x8(_mm512_castsi256_si512(_pBB), _pBB, 1);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_shuffle_epi32(_pB0, _MM_PERM_BADC);
                __m512i _pB3 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CBAD);

#if __AVX512VNNI__
                _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm512_dpwssd_epi32(_sum2, _pA0, _pB2);
                _sum3 = _mm512_dpwssd_epi32(_sum3, _pA0, _pB3);
                _sum4 = _mm512_dpwssd_epi32(_sum4, _pA1, _pB0);
                _sum5 = _mm512_dpwssd_epi32(_sum5, _pA1, _pB1);
                _sum6 = _mm512_dpwssd_epi32(_sum6, _pA1, _pB2);
                _sum7 = _mm512_dpwssd_epi32(_sum7, _pA1, _pB3);
#else
                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA0, _pB2));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA0, _pB3));
                _sum4 = _mm512_add_epi32(_sum4, _mm512_madd_epi16(_pA1, _pB0));
                _sum5 = _mm512_add_epi32(_sum5, _mm512_madd_epi16(_pA1, _pB1));
                _sum6 = _mm512_add_epi32(_sum6, _mm512_madd_epi16(_pA1, _pB2));
                _sum7 = _mm512_add_epi32(_sum7, _mm512_madd_epi16(_pA1, _pB3));
#endif

                pA += 32;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);

                // 01234567 89abcdef
                // 45670123 cdef89ab
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 01234567
                // 01234567 01234567
                // 12305674 12305674
                // 23016745 23016745
                // 30127456 30127456
                __m256i _pB0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_pB), _pB, 1);
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 3, 0, 1));
                __m256i _pB3 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(2, 1, 0, 3)), _MM_SHUFFLE(2, 1, 0, 3));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));
                __m512i _s2 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB2));
                __m512i _s3 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB3));
                __m512i _s4 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0));
                __m512i _s5 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1));
                __m512i _s6 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB2));
                __m512i _s7 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB3));

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
                    // from
                    //      00 11 22 33 44 55 66 77 80 91 a2 b3 c4 d5 e6 f7
                    //      01 12 23 30 45 56 67 74 81 92 a3 b0 c5 d6 e7 f4
                    //      02 13 20 31 46 57 64 75 82 93 a0 b1 c6 d7 e4 f5
                    //      03 10 21 32 47 54 65 76 83 90 a1 b2 c7 d4 e5 f6
                    //      40 51 62 73 04 15 26 37 c0 d1 e2 f3 84 95 a6 b7
                    //      41 52 63 70 05 16 27 34 c1 d2 e3 f0 85 96 a7 b4
                    //      42 53 60 71 06 17 24 35 c2 d3 e0 f1 86 97 a4 b5
                    //      43 50 61 72 07 14 25 36 c3 d0 e1 f2 87 94 a5 b6
                    // to
                    //      00 10 20 30 44 54 64 74 80 90 a0 b0 c4 d4 e4 f4
                    //      01 11 21 31 45 55 65 75 81 91 a1 b1 c5 d5 e5 f5
                    //      02 12 22 32 46 56 66 76 82 92 a2 b2 c6 d6 e6 f6
                    //      03 13 23 33 47 57 67 77 83 93 a3 b3 c7 d7 e7 f7
                    //      40 50 60 70 04 14 24 34 c0 d0 e0 f0 84 94 a4 b4
                    //      41 51 61 71 05 15 25 35 c1 d1 e1 f1 85 95 a5 b5
                    //      42 52 62 72 06 16 26 36 c2 d2 e2 f2 86 96 a6 b6
                    //      43 53 63 73 07 17 27 37 c3 d3 e3 f3 87 97 a7 b7
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum2 = _mm512_shuffle_epi32(_sum2, _MM_PERM_BADC);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_ADCB);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum6 = _mm512_shuffle_epi32(_sum6, _MM_PERM_BADC);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_ADCB);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum7);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum7);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum5);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum5);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                    }

                    // TODO
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum4, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum1, _sum5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum2, _sum6, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum3, _sum7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sum4, _sum0, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_sum5, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sum6, _sum2, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sum7, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp1, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp2, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmp3, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum4 = _mm512_shuffle_i32x4(_tmp4, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum5 = _mm512_shuffle_i32x4(_tmp5, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum6 = _mm512_shuffle_i32x4(_tmp6, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum7 = _mm512_shuffle_i32x4(_tmp7, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));

                    _mm512_store_si512((__m512i*)outptr0, _sum0);
                    _mm512_store_si512((__m512i*)(outptr0 + 16), _sum1);
                    _mm512_store_si512((__m512i*)(outptr0 + 32), _sum2);
                    _mm512_store_si512((__m512i*)(outptr0 + 48), _sum3);
                    _mm512_store_si512((__m512i*)(outptr0 + 64), _sum4);
                    _mm512_store_si512((__m512i*)(outptr0 + 80), _sum5);
                    _mm512_store_si512((__m512i*)(outptr0 + 96), _sum6);
                    _mm512_store_si512((__m512i*)(outptr0 + 112), _sum7);
                    outptr0 += 128;
                }
                if (out_elempack == 8)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 80 91 a2 b3 c4 d5 e6 f7
                    //      01 12 23 30 45 56 67 74 81 92 a3 b0 c5 d6 e7 f4
                    //      02 13 20 31 46 57 64 75 82 93 a0 b1 c6 d7 e4 f5
                    //      03 10 21 32 47 54 65 76 83 90 a1 b2 c7 d4 e5 f6
                    //      40 51 62 73 04 15 26 37 c0 d1 e2 f3 84 95 a6 b7
                    //      41 52 63 70 05 16 27 34 c1 d2 e3 f0 85 96 a7 b4
                    //      42 53 60 71 06 17 24 35 c2 d3 e0 f1 86 97 a4 b5
                    //      43 50 61 72 07 14 25 36 c3 d0 e1 f2 87 94 a5 b6
                    // to
                    //      00 10 20 30 44 54 64 74 80 90 a0 b0 c4 d4 e4 f4
                    //      01 11 21 31 45 55 65 75 81 91 a1 b1 c5 d5 e5 f5
                    //      02 12 22 32 46 56 66 76 82 92 a2 b2 c6 d6 e6 f6
                    //      03 13 23 33 47 57 67 77 83 93 a3 b3 c7 d7 e7 f7
                    //      40 50 60 70 04 14 24 34 c0 d0 e0 f0 84 94 a4 b4
                    //      41 51 61 71 05 15 25 35 c1 d1 e1 f1 85 95 a5 b5
                    //      42 52 62 72 06 16 26 36 c2 d2 e2 f2 86 96 a6 b6
                    //      43 53 63 73 07 17 27 37 c3 d3 e3 f3 87 97 a7 b7
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum2 = _mm512_shuffle_epi32(_sum2, _MM_PERM_BADC);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_ADCB);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum6 = _mm512_shuffle_epi32(_sum6, _MM_PERM_BADC);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_ADCB);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum7);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum7);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum5);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum5);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                    }

                    // TODO
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum4, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum1, _sum5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum2, _sum6, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum3, _sum7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sum4, _sum0, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_sum5, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sum6, _sum2, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sum7, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum4 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum5 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum6 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum7 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16), _sum1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16 * 2), _sum2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16 * 3), _sum3);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8), _sum4);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 16), _sum5);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 16 * 2), _sum6);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 16 * 3), _sum7);
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 80 91 a2 b3 c4 d5 e6 f7
                    //      01 12 23 30 45 56 67 74 81 92 a3 b0 c5 d6 e7 f4
                    //      02 13 20 31 46 57 64 75 82 93 a0 b1 c6 d7 e4 f5
                    //      03 10 21 32 47 54 65 76 83 90 a1 b2 c7 d4 e5 f6
                    //      40 51 62 73 04 15 26 37 c0 d1 e2 f3 84 95 a6 b7
                    //      41 52 63 70 05 16 27 34 c1 d2 e3 f0 85 96 a7 b4
                    //      42 53 60 71 06 17 24 35 c2 d3 e0 f1 86 97 a4 b5
                    //      43 50 61 72 07 14 25 36 c3 d0 e1 f2 87 94 a5 b6
                    // to
                    //      00 10 20 30 44 54 64 74 80 90 a0 b0 c4 d4 e4 f4
                    //      01 11 21 31 45 55 65 75 81 91 a1 b1 c5 d5 e5 f5
                    //      02 12 22 32 46 56 66 76 82 92 a2 b2 c6 d6 e6 f6
                    //      03 13 23 33 47 57 67 77 83 93 a3 b3 c7 d7 e7 f7
                    //      40 50 60 70 04 14 24 34 c0 d0 e0 f0 84 94 a4 b4
                    //      41 51 61 71 05 15 25 35 c1 d1 e1 f1 85 95 a5 b5
                    //      42 52 62 72 06 16 26 36 c2 d2 e2 f2 86 96 a6 b6
                    //      43 53 63 73 07 17 27 37 c3 d3 e3 f3 87 97 a7 b7
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum2 = _mm512_shuffle_epi32(_sum2, _MM_PERM_BADC);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_ADCB);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum6 = _mm512_shuffle_epi32(_sum6, _MM_PERM_BADC);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_ADCB);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum7);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum7);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum5);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum5);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum4, _sum5, _MM_SHUFFLE(0, 1, 0, 1));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum6, _sum7, _MM_SHUFFLE(0, 1, 0, 1));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sum4, _sum5, _MM_SHUFFLE(2, 3, 2, 3));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sum6, _sum7, _MM_SHUFFLE(2, 3, 2, 3));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum3 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum4 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum6 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum7 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16), _sum1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4), _sum2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4 + 16), _sum3);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8), _sum4);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 16), _sum5);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 12), _sum6);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 12 + 16), _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 80 91 a2 b3 c4 d5 e6 f7
                    //      01 12 23 30 45 56 67 74 81 92 a3 b0 c5 d6 e7 f4
                    //      02 13 20 31 46 57 64 75 82 93 a0 b1 c6 d7 e4 f5
                    //      03 10 21 32 47 54 65 76 83 90 a1 b2 c7 d4 e5 f6
                    //      40 51 62 73 04 15 26 37 c0 d1 e2 f3 84 95 a6 b7
                    //      41 52 63 70 05 16 27 34 c1 d2 e3 f0 85 96 a7 b4
                    //      42 53 60 71 06 17 24 35 c2 d3 e0 f1 86 97 a4 b5
                    //      43 50 61 72 07 14 25 36 c3 d0 e1 f2 87 94 a5 b6
                    // to
                    //      00 01 02 03 44 45 46 47 80 81 82 83 c4 c5 c6 c7
                    //      10 11 12 13 54 55 56 57 90 91 92 93 d4 d5 d6 d7
                    //      20 21 22 23 64 65 66 67 a0 a1 a2 a3 e4 e5 e6 e7
                    //      30 31 32 33 74 75 76 77 b0 b1 b2 b3 f4 f5 f6 f7
                    //      40 41 42 43 04 05 06 07 c0 c1 c2 c3 84 85 86 87
                    //      50 51 52 53 14 15 16 17 d0 d1 d2 d3 94 95 96 97
                    //      60 61 62 63 24 25 26 27 e0 e1 e2 e3 a4 a5 a6 a7
                    //      70 71 72 73 34 35 36 37 f0 f1 f2 f3 b4 b5 b6 b7
                    {
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum1);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum1);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum3);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum3);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum5);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum5);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum7);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum7);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum4, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum1, _sum5, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum2, _sum6, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum3, _sum7, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sum4, _sum0, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_sum5, _sum1, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sum6, _sum2, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sum7, _sum3, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp1, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp2, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmp3, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum4 = _mm512_shuffle_i32x4(_tmp4, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum5 = _mm512_shuffle_i32x4(_tmp5, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum6 = _mm512_shuffle_i32x4(_tmp6, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum7 = _mm512_shuffle_i32x4(_tmp7, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));

                    _mm256_storeu_si256((__m256i*)outptr0, _mm512_extracti32x8_epi32(_sum0, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep), _mm512_extracti32x8_epi32(_sum1, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 2), _mm512_extracti32x8_epi32(_sum2, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 3), _mm512_extracti32x8_epi32(_sum3, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), _mm512_extracti32x8_epi32(_sum4, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 5), _mm512_extracti32x8_epi32(_sum5, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 6), _mm512_extracti32x8_epi32(_sum6, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 7), _mm512_extracti32x8_epi32(_sum7, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 8), _mm512_extracti32x8_epi32(_sum0, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 9), _mm512_extracti32x8_epi32(_sum1, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 10), _mm512_extracti32x8_epi32(_sum2, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 11), _mm512_extracti32x8_epi32(_sum3, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 12), _mm512_extracti32x8_epi32(_sum4, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 13), _mm512_extracti32x8_epi32(_sum5, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 14), _mm512_extracti32x8_epi32(_sum6, 1));
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
                __m256i _pB = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pB));

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 4567 89ab cdef
                // 2301 6745 ab89 efcd
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);

                // 0123 0123 0123 0123
                // 1230 1230 1230 1230
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

#if __AVX512VNNI__
                _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm512_dpwssd_epi32(_sum2, _pA1, _pB0);
                _sum3 = _mm512_dpwssd_epi32(_sum3, _pA1, _pB1);
#else
                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA1, _pB0));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA1, _pB1));
#endif

                pA += 32;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 01234567 89abcdef
                // 23016745 ab89efcd
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));

                // 01230123 01230123
                // 12301230 12301230
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));
                __m512i _s2 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0));
                __m512i _s3 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1));

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
                    // from
                    //      00 11 22 33 40 51 62 73 80 91 a2 b3 c0 d1 e2 f3
                    //      01 12 23 30 41 52 63 70 81 92 a3 b0 c1 d2 e3 f0
                    //      20 31 02 13 60 71 42 53 a0 b1 82 93 e0 f1 c2 d3
                    //      21 32 03 10 61 72 43 50 a1 b2 83 90 e1 f2 c3 d0
                    // to
                    //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
                    //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
                    //      02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
                    //      03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                    }

                    _mm512_store_si512((__m512i*)outptr0, _sum0);
                    _mm512_store_si512((__m512i*)(outptr0 + 16), _sum1);
                    _mm512_store_si512((__m512i*)(outptr0 + 32), _sum2);
                    _mm512_store_si512((__m512i*)(outptr0 + 48), _sum3);
                    outptr0 += 64;
                }
                if (out_elempack == 8)
                {
                    // from
                    //      00 11 22 33 40 51 62 73 80 91 a2 b3 c0 d1 e2 f3
                    //      01 12 23 30 41 52 63 70 81 92 a3 b0 c1 d2 e3 f0
                    //      20 31 02 13 60 71 42 53 a0 b1 82 93 e0 f1 c2 d3
                    //      21 32 03 10 61 72 43 50 a1 b2 83 90 e1 f2 c3 d0
                    // to
                    //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
                    //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
                    //      02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
                    //      03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                    }

                    // TODO
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_si512((__m512i*)outptr0, _tmp0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16), _tmp1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8), _tmp2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8 + 16), _tmp3);
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    // from
                    //      00 11 22 33 40 51 62 73 80 91 a2 b3 c0 d1 e2 f3
                    //      01 12 23 30 41 52 63 70 81 92 a3 b0 c1 d2 e3 f0
                    //      20 31 02 13 60 71 42 53 a0 b1 82 93 e0 f1 c2 d3
                    //      21 32 03 10 61 72 43 50 a1 b2 83 90 e1 f2 c3 d0
                    // to
                    //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
                    //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
                    //      02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
                    //      03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4), _sum1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8), _sum2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 12), _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    // from
                    //      00 11 22 33 40 51 62 73 80 91 a2 b3 c0 d1 e2 f3
                    //      01 12 23 30 41 52 63 70 81 92 a3 b0 c1 d2 e3 f0
                    //      20 31 02 13 60 71 42 53 a0 b1 82 93 e0 f1 c2 d3
                    //      21 32 03 10 61 72 43 50 a1 b2 83 90 e1 f2 c3 d0
                    // to
                    //      00 01 02 03 40 41 42 43 80 81 82 83 c0 c1 c2 c3
                    //      10 11 12 13 50 51 52 53 90 91 92 93 d0 d1 d2 d3
                    //      20 21 22 23 60 61 62 63 a0 a1 a2 a3 e0 e1 e2 e3
                    //      30 31 32 33 70 71 72 73 b0 b1 b2 b3 f0 f1 f2 f2
                    {
                        _sum2 = _mm512_shuffle_epi32(_sum2, _MM_PERM_BADC);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_BADC);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum1);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum1);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum3);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum3);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                    }

                    _mm_storeu_si128((__m128i*)outptr0, _mm512_extracti32x4_epi32(_sum0, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep), _mm512_extracti32x4_epi32(_sum1, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2), _mm512_extracti32x4_epi32(_sum2, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3), _mm512_extracti32x4_epi32(_sum3, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), _mm512_extracti32x4_epi32(_sum0, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 5), _mm512_extracti32x4_epi32(_sum1, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 6), _mm512_extracti32x4_epi32(_sum2, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 7), _mm512_extracti32x4_epi32(_sum3, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 8), _mm512_extracti32x4_epi32(_sum0, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 9), _mm512_extracti32x4_epi32(_sum1, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 10), _mm512_extracti32x4_epi32(_sum2, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 11), _mm512_extracti32x4_epi32(_sum3, 2));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 12), _mm512_extracti32x4_epi32(_sum0, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 13), _mm512_extracti32x4_epi32(_sum1, 3));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 14), _mm512_extracti32x4_epi32(_sum2, 3));
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
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 4567 89ab cdef

                // 0101 0101 0101 0101
                // 1010 1010 1010 1010
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CDAB);

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
                __m128i _pB = _mm_set1_epi16(((const short*)pB)[0]);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 01234567 89abcdef

                // 01010101 01010101
                // 10101010 10101010
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 1, 0, 1)), _MM_SHUFFLE(0, 1, 0, 1));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);

                pA += 16;
                pB += 2;
            }

            if (k_end)
            {
                // from
                //      00 11 20 31 40 51 60 71 80 91 a0 b1 c0 d1 e0 f1
                //      01 10 21 30 41 50 61 70 81 90 a1 b0 c1 d0 e1 f0
                // to
                //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
                //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
                {
                    __m512i _tmp0 = _mm512_shuffle_epi32(_sum0, _MM_PERM_DBCA);
                    __m512i _tmp1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_ACDB);
                    _sum0 = _mm512_unpacklo_epi32(_tmp0, _tmp1);
                    _sum1 = _mm512_unpackhi_epi32(_tmp0, _tmp1);
                    _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                }

                if (out_elempack == 16)
                {
                    _mm512_store_si512((__m512i*)outptr0, _sum0);
                    _mm512_store_si512((__m512i*)(outptr0 + 16), _sum1);
                    outptr0 += 32;
                }
                if (out_elempack == 8)
                {
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_si512((__m512i*)outptr0, _tmp0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 8), _tmp1);
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
                    __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                    _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(out_hstep));
                    _mm512_i32scatter_epi32(outptr0, _vindex, _sum0, sizeof(float));
                    _mm512_i32scatter_epi32(outptr0 + 1, _vindex, _sum1, sizeof(float));
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
                    _mm512_store_si512((__m512i*)outptr0, _sum0);
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
                    __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                    _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(out_hstep));
                    _mm512_i32scatter_epi32(outptr0, _vindex, _sum0, sizeof(float));
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
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
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
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 4567 0123 4567
                // 4567 0123 4567 0123
                __m512i _pA00 = _mm512_inserti32x8(_mm512_castsi256_si512(_pA0), _pA0, 1);
                __m512i _pA11 = _mm512_permutex_epi64(_pA00, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 4567 89ab cdef
                // 1230 5674 9ab8 defc
                // 2301 6745 ab89 efcd
                // 3012 7456 b89a fcde
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_shuffle_epi32(_pB0, _MM_PERM_BADC);
                __m512i _pB3 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CBAD);

#if __AVX512VNNI__
                _sum0 = _mm512_dpwssd_epi32(_sum0, _pA00, _pB0);
                _sum1 = _mm512_dpwssd_epi32(_sum1, _pA00, _pB1);
                _sum2 = _mm512_dpwssd_epi32(_sum2, _pA00, _pB2);
                _sum3 = _mm512_dpwssd_epi32(_sum3, _pA00, _pB3);
                _sum4 = _mm512_dpwssd_epi32(_sum4, _pA11, _pB0);
                _sum5 = _mm512_dpwssd_epi32(_sum5, _pA11, _pB1);
                _sum6 = _mm512_dpwssd_epi32(_sum6, _pA11, _pB2);
                _sum7 = _mm512_dpwssd_epi32(_sum7, _pA11, _pB3);
#else
                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA00, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA00, _pB1));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA00, _pB2));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA00, _pB3));
                _sum4 = _mm512_add_epi32(_sum4, _mm512_madd_epi16(_pA11, _pB0));
                _sum5 = _mm512_add_epi32(_sum5, _mm512_madd_epi16(_pA11, _pB1));
                _sum6 = _mm512_add_epi32(_sum6, _mm512_madd_epi16(_pA11, _pB2));
                _sum7 = _mm512_add_epi32(_sum7, _mm512_madd_epi16(_pA11, _pB3));
#endif // __AVX512VNNI__

                pA += 16;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                _pA = _mm_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 01234567 01234567
                // 45670123 45670123
                __m256i _pA00 = _mm256_inserti128_si256(_mm256_castsi128_si256(_pA), _pA, 1);
                __m256i _pA11 = _mm256_permute4x64_epi64(_pA00, _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567 89abcdef
                // 12305674 9ab8defc
                // 23016745 ab89efcd
                // 30127456 b89afcde
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 3, 0, 1));
                __m256i _pB3 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(2, 1, 0, 3)), _MM_SHUFFLE(2, 1, 0, 3));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB1));
                __m512i _s2 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB2));
                __m512i _s3 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB3));
                __m512i _s4 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB0));
                __m512i _s5 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB1));
                __m512i _s6 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB2));
                __m512i _s7 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB3));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);
                _sum2 = _mm512_add_epi32(_sum2, _s2);
                _sum3 = _mm512_add_epi32(_sum3, _s3);
                _sum4 = _mm512_add_epi32(_sum4, _s4);
                _sum5 = _mm512_add_epi32(_sum5, _s5);
                _sum6 = _mm512_add_epi32(_sum6, _s6);
                _sum7 = _mm512_add_epi32(_sum7, _s7);

                pA += 8;
                pB += 16;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 08 19 2a 3b 4c 5d 6e 7f
                    //      01 12 23 30 45 56 67 74 09 1a 2b 38 4d 5e 6f 7c
                    //      02 13 20 31 46 57 64 75 0a 1b 28 39 4e 5f 6c 7d
                    //      03 10 21 32 47 54 65 76 0b 18 29 3a 4f 5c 6d 7e
                    //      40 51 62 73 04 15 26 37 48 59 6a 7b 0c 1d 2e 3f
                    //      41 52 63 70 05 16 27 34 49 5a 6b 78 0d 1e 2f 3c
                    //      42 53 60 71 06 17 24 35 4a 5b 68 79 0e 1f 2c 3d
                    //      43 50 61 72 07 14 25 36 4b 58 69 7a 0f 1c 2d 3e
                    // to
                    //      00 10 20 30 44 54 64 74 08 18 28 38 4c 5c 6c 7c
                    //      01 11 21 31 45 55 65 75 09 19 29 39 4d 5d 6d 7d
                    //      02 12 22 32 46 56 66 76 0a 1a 2a 3a 4e 5e 6e 7e
                    //      03 13 23 33 47 57 67 77 0b 1b 2b 3b 4f 5f 6f 7f
                    //      40 50 60 70 04 14 24 34 48 58 68 78 0c 1c 2c 3c
                    //      41 51 61 71 05 15 25 35 49 59 69 79 0d 1d 2d 3d
                    //      42 52 62 72 06 16 26 36 4a 5a 6a 7a 0e 1e 2e 3e
                    //      43 53 63 73 07 17 27 37 4b 5b 6b 7b 0f 1f 2f 3f
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum2 = _mm512_shuffle_epi32(_sum2, _MM_PERM_BADC);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_ADCB);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum6 = _mm512_shuffle_epi32(_sum6, _MM_PERM_BADC);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_ADCB);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum7);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum7);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum5);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum5);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum4, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum0, _sum4, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum1, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum1, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sum2, _sum6, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_sum2, _sum6, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sum3, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sum3, _sum7, _MM_SHUFFLE(3, 2, 3, 2));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp2, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp4, _tmp6, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp0, _tmp2, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum3 = _mm512_shuffle_i32x4(_tmp4, _tmp6, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum4 = _mm512_shuffle_i32x4(_tmp1, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_i32x4(_tmp5, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum6 = _mm512_shuffle_i32x4(_tmp1, _tmp3, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum7 = _mm512_shuffle_i32x4(_tmp5, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));

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
                if (out_elempack == 4)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 08 19 2a 3b 4c 5d 6e 7f
                    //      01 12 23 30 45 56 67 74 09 1a 2b 38 4d 5e 6f 7c
                    //      02 13 20 31 46 57 64 75 0a 1b 28 39 4e 5f 6c 7d
                    //      03 10 21 32 47 54 65 76 0b 18 29 3a 4f 5c 6d 7e
                    //      40 51 62 73 04 15 26 37 48 59 6a 7b 0c 1d 2e 3f
                    //      41 52 63 70 05 16 27 34 49 5a 6b 78 0d 1e 2f 3c
                    //      42 53 60 71 06 17 24 35 4a 5b 68 79 0e 1f 2c 3d
                    //      43 50 61 72 07 14 25 36 4b 58 69 7a 0f 1c 2d 3e
                    // to
                    //      00 10 20 30 44 54 64 74 08 18 28 38 4c 5c 6c 7c
                    //      01 11 21 31 45 55 65 75 09 19 29 39 4d 5d 6d 7d
                    //      02 12 22 32 46 56 66 76 0a 1a 2a 3a 4e 5e 6e 7e
                    //      03 13 23 33 47 57 67 77 0b 1b 2b 3b 4f 5f 6f 7f
                    //      40 50 60 70 04 14 24 34 48 58 68 78 0c 1c 2c 3c
                    //      41 51 61 71 05 15 25 35 49 59 69 79 0d 1d 2d 3d
                    //      42 52 62 72 06 16 26 36 4a 5a 6a 7a 0e 1e 2e 3e
                    //      43 53 63 73 07 17 27 37 4b 5b 6b 7b 0f 1f 2f 3f
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum2 = _mm512_shuffle_epi32(_sum2, _MM_PERM_BADC);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_ADCB);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum6 = _mm512_shuffle_epi32(_sum6, _MM_PERM_BADC);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_ADCB);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum7);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum7);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum5);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum5);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum4, _sum5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum6, _sum7, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sum4, _sum5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_sum6, _sum7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum3 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum4 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum6 = _mm512_shuffle_i32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum7 = _mm512_shuffle_i32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16), _sum1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 32), _sum2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 48), _sum3);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4), _sum4);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4 + 16), _sum5);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4 + 32), _sum6);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4 + 48), _sum7);
                    outptr0 += 64;
                }
                if (out_elempack == 1)
                {
                    // from
                    //      00 11 22 33 44 55 66 77 08 19 2a 3b 4c 5d 6e 7f
                    //      01 12 23 30 45 56 67 74 09 1a 2b 38 4d 5e 6f 7c
                    //      02 13 20 31 46 57 64 75 0a 1b 28 39 4e 5f 6c 7d
                    //      03 10 21 32 47 54 65 76 0b 18 29 3a 4f 5c 6d 7e
                    //      40 51 62 73 04 15 26 37 48 59 6a 7b 0c 1d 2e 3f
                    //      41 52 63 70 05 16 27 34 49 5a 6b 78 0d 1e 2f 3c
                    //      42 53 60 71 06 17 24 35 4a 5b 68 79 0e 1f 2c 3d
                    //      43 50 61 72 07 14 25 36 4b 58 69 7a 0f 1c 2d 3e
                    // to
                    //      00 01 02 03 44 45 46 47 08 09 0a 0b 4c 4d 4e 4f
                    //      10 11 12 13 54 55 56 57 18 19 1a 1b 5c 5d 5e 5f
                    //      20 21 22 23 64 65 66 67 28 29 2a 2b 6c 6d 6e 6f
                    //      30 31 32 33 74 75 76 77 38 39 3a 3b 7c 7d 7e 7f
                    //      40 41 42 43 04 05 06 07 48 49 4a 4b 0c 0d 0e 0f
                    //      50 51 52 53 14 15 16 17 58 59 5a 5b 1c 1d 1e 1f
                    //      60 61 62 63 24 25 26 27 68 69 6a 6b 2c 2d 2e 2f
                    //      70 71 72 73 34 35 36 37 78 79 7a 7b 3c 3d 3e 3f
                    {
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum1);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum1);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum3);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum3);
                        __m512i _tmp4 = _mm512_unpacklo_epi32(_sum4, _sum5);
                        __m512i _tmp5 = _mm512_unpackhi_epi32(_sum4, _sum5);
                        __m512i _tmp6 = _mm512_unpacklo_epi32(_sum6, _sum7);
                        __m512i _tmp7 = _mm512_unpackhi_epi32(_sum6, _sum7);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm512_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm512_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm512_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm512_unpackhi_epi64(_tmp7, _tmp5);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        _sum5 = _mm512_shuffle_epi32(_sum5, _MM_PERM_CBAD);
                        _sum7 = _mm512_shuffle_epi32(_sum7, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum4, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum1, _sum5, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum2, _sum6, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum3, _sum7, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp4 = _mm512_shuffle_i32x4(_sum4, _sum0, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp5 = _mm512_shuffle_i32x4(_sum5, _sum1, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp6 = _mm512_shuffle_i32x4(_sum6, _sum2, _MM_SHUFFLE(3, 1, 2, 0));
                    __m512i _tmp7 = _mm512_shuffle_i32x4(_sum7, _sum3, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp1, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_tmp2, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmp3, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum4 = _mm512_shuffle_i32x4(_tmp4, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum5 = _mm512_shuffle_i32x4(_tmp5, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum6 = _mm512_shuffle_i32x4(_tmp6, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                    _sum7 = _mm512_shuffle_i32x4(_tmp7, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));

                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep), _sum1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 2), _sum2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 3), _sum3);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4), _sum4);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 5), _sum5);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 6), _sum6);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 7), _sum7);

                    outptr0 += 16;
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
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

#if __AVX512F__
            __m512i _sum0;
            __m512i _sum1;
            __m512i _sum2;
            __m512i _sum3;
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
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
                _sum2 = _mm512_setzero_si512();
                _sum3 = _mm512_setzero_si512();
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
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
                _sum2 = _mm512_load_si512((const __m512i*)(outptr + 32));
                _sum3 = _mm512_load_si512((const __m512i*)(outptr + 48));
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
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

#if __AVX512F__
                // 0123 4567 0123 4567
                // 4567 0123 4567 0123
                __m512i _pA00 = _mm512_inserti32x8(_mm512_castsi256_si512(_pA0), _pA0, 1);
                __m512i _pA11 = _mm512_permutex_epi64(_pA00, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 4567 1230 5674
                // 2301 6745 3012 7456
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m512i _pB01 = _mm512_inserti32x8(_mm512_castsi256_si512(_pB0), _pB1, 1);
                __m512i _pB23 = _mm512_shuffle_epi32(_pB01, _MM_PERM_BADC);

#if __AVX512VNNI__
                _sum0 = _mm512_dpwssd_epi32(_sum0, _pA00, _pB01);
                _sum1 = _mm512_dpwssd_epi32(_sum1, _pA00, _pB23);
                _sum2 = _mm512_dpwssd_epi32(_sum2, _pA11, _pB01);
                _sum3 = _mm512_dpwssd_epi32(_sum3, _pA11, _pB23);
#else
                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA00, _pB01));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA00, _pB23));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA11, _pB01));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA11, _pB23));
#endif // __AVX512VNNI__
#else  // __AVX512F__

                // 0123 4567
                // 4567 0123
                __m256i _pA1 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 4567
                // 1230 5674
                // 2301 6745
                // 3012 7456
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 1, 0, 3));

#if __AVXVNNI__
                _sum0 = _mm256_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm256_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm256_dpwssd_epi32(_sum2, _pA0, _pB2);
                _sum3 = _mm256_dpwssd_epi32(_sum3, _pA0, _pB3);
                _sum4 = _mm256_dpwssd_epi32(_sum4, _pA1, _pB0);
                _sum5 = _mm256_dpwssd_epi32(_sum5, _pA1, _pB1);
                _sum6 = _mm256_dpwssd_epi32(_sum6, _pA1, _pB2);
                _sum7 = _mm256_dpwssd_epi32(_sum7, _pA1, _pB3);
#else
                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
                _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_pA0, _pB2));
                _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_pA0, _pB3));
                _sum4 = _mm256_add_epi32(_sum4, _mm256_madd_epi16(_pA1, _pB0));
                _sum5 = _mm256_add_epi32(_sum5, _mm256_madd_epi16(_pA1, _pB1));
                _sum6 = _mm256_add_epi32(_sum6, _mm256_madd_epi16(_pA1, _pB2));
                _sum7 = _mm256_add_epi32(_sum7, _mm256_madd_epi16(_pA1, _pB3));
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
                // 01234567 01234567
                // 45670123 45670123
                __m256i _pA00 = _mm256_inserti128_si256(_mm256_castsi128_si256(_pA), _pA, 1);
                __m256i _pA11 = _mm256_permute4x64_epi64(_pA00, _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567 12305674
                // 23016745 30127456
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB01 = _mm256_inserti128_si256(_mm256_castsi128_si256(_pB), _pB1, 1);
                __m256i _pB23 = _mm256_shuffle_epi32(_pB01, _MM_SHUFFLE(2, 3, 0, 1));

                __m512i _s01 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB01));
                __m512i _s23 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB23));
                __m512i _s45 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB01));
                __m512i _s67 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB23));

                _sum0 = _mm512_add_epi32(_sum0, _s01);
                _sum1 = _mm512_add_epi32(_sum1, _s23);
                _sum2 = _mm512_add_epi32(_sum2, _s45);
                _sum3 = _mm512_add_epi32(_sum3, _s67);
#else
                // 0123 4567
                // 4567 0123
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 4567
                // 1230 5674
                // 2301 6745
                // 3012 7456
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m128i _pB2 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(2, 3, 0, 1));
                __m128i _pB3 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(2, 1, 0, 3)), _MM_SHUFFLE(2, 1, 0, 3));

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB1));
                __m256i _s2 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB2));
                __m256i _s3 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB3));
                __m256i _s4 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB0));
                __m256i _s5 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB1));
                __m256i _s6 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB2));
                __m256i _s7 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB3));

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
                    // from
                    //      00 11 22 33 44 55 66 77 01 12 23 30 45 56 67 74
                    //      02 13 20 31 46 57 64 75 03 10 21 32 47 54 65 76
                    //      40 51 62 73 04 15 26 37 41 52 63 70 05 16 27 34
                    //      42 53 60 71 06 17 24 35 43 50 61 72 07 14 25 36
                    // to
                    //      00 10 20 30 44 54 64 74 04 14 24 34 40 50 60 70
                    //      01 11 21 31 45 55 65 75 05 15 25 35 41 51 61 71
                    //      02 12 22 32 46 56 66 76 06 16 26 36 42 52 62 72
                    //      03 13 23 33 47 57 67 77 07 17 27 37 43 53 63 73
                    {
                        __m512i _s0 = _mm512_shuffle_i32x4(_sum0, _sum2, _MM_SHUFFLE(0, 1, 1, 0));
                        __m512i _s1 = _mm512_shuffle_i32x4(_sum1, _sum3, _MM_SHUFFLE(2, 3, 3, 2));
                        __m512i _s2 = _mm512_shuffle_i32x4(_sum1, _sum3, _MM_SHUFFLE(0, 1, 1, 0));
                        __m512i _s3 = _mm512_shuffle_i32x4(_sum0, _sum2, _MM_SHUFFLE(2, 3, 3, 2));
                        _s1 = _mm512_shuffle_epi32(_s1, _MM_PERM_ADCB);
                        _s2 = _mm512_shuffle_epi32(_s2, _MM_PERM_BADC);
                        _s3 = _mm512_shuffle_epi32(_s3, _MM_PERM_CBAD);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_s0, _s1);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_s0, _s1);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_s2, _s3);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_s2, _s3);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                    }

                    // TODO
                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(3, 0, 3, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(3, 0, 3, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(1, 2, 1, 2));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(1, 2, 1, 2));

                    _mm512_storeu_si512((__m512i*)outptr0, _tmp0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16), _tmp1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 32), _tmp2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 48), _tmp3);
#else
                    // from
                    //      00 11 22 33 44 55 66 77
                    //      01 12 23 30 45 56 67 74
                    //      02 13 20 31 46 57 64 75
                    //      03 10 21 32 47 54 65 76
                    //      40 51 62 73 04 15 26 37
                    //      41 52 63 70 05 16 27 34
                    //      42 53 60 71 06 17 24 35
                    //      43 50 61 72 07 14 25 36
                    // to
                    //      00 10 20 30 44 54 64 74
                    //      01 11 21 31 45 55 65 75
                    //      02 12 22 32 46 56 66 76
                    //      03 13 23 33 47 57 67 77
                    //      40 50 60 70 04 14 24 34
                    //      41 51 61 71 05 15 25 35
                    //      42 52 62 72 06 16 26 36
                    //      43 53 63 73 07 17 27 37
                    {
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum2 = _mm256_shuffle_epi32(_sum2, _MM_SHUFFLE(1, 0, 3, 2));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(0, 3, 2, 1));
                        _sum5 = _mm256_shuffle_epi32(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum6 = _mm256_shuffle_epi32(_sum6, _MM_SHUFFLE(1, 0, 3, 2));
                        _sum7 = _mm256_shuffle_epi32(_sum7, _MM_SHUFFLE(0, 3, 2, 1));
                        __m256i _tmp0 = _mm256_unpacklo_epi32(_sum0, _sum3);
                        __m256i _tmp1 = _mm256_unpackhi_epi32(_sum0, _sum3);
                        __m256i _tmp2 = _mm256_unpacklo_epi32(_sum2, _sum1);
                        __m256i _tmp3 = _mm256_unpackhi_epi32(_sum2, _sum1);
                        __m256i _tmp4 = _mm256_unpacklo_epi32(_sum4, _sum7);
                        __m256i _tmp5 = _mm256_unpackhi_epi32(_sum4, _sum7);
                        __m256i _tmp6 = _mm256_unpacklo_epi32(_sum6, _sum5);
                        __m256i _tmp7 = _mm256_unpackhi_epi32(_sum6, _sum5);
                        _sum0 = _mm256_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm256_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm256_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm256_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm256_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm256_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm256_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm256_unpackhi_epi64(_tmp7, _tmp5);
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum5 = _mm256_shuffle_epi32(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum7 = _mm256_shuffle_epi32(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
                    }

                    // TODO
                    __m256i _tmp0 = _mm256_permute2x128_si256(_sum0, _sum4, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp1 = _mm256_permute2x128_si256(_sum1, _sum5, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp2 = _mm256_permute2x128_si256(_sum2, _sum6, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp3 = _mm256_permute2x128_si256(_sum3, _sum7, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp4 = _mm256_permute2x128_si256(_sum4, _sum0, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp5 = _mm256_permute2x128_si256(_sum5, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp6 = _mm256_permute2x128_si256(_sum6, _sum2, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp7 = _mm256_permute2x128_si256(_sum7, _sum3, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_si256((__m256i*)outptr0, _tmp0);
                    _mm256_store_si256((__m256i*)(outptr0 + 8), _tmp1);
                    _mm256_store_si256((__m256i*)(outptr0 + 16), _tmp2);
                    _mm256_store_si256((__m256i*)(outptr0 + 24), _tmp3);
                    _mm256_store_si256((__m256i*)(outptr0 + 32), _tmp4);
                    _mm256_store_si256((__m256i*)(outptr0 + 40), _tmp5);
                    _mm256_store_si256((__m256i*)(outptr0 + 48), _tmp6);
                    _mm256_store_si256((__m256i*)(outptr0 + 56), _tmp7);
#endif //__AVX512F__
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
#if __AVX512F__
                    // from
                    //      00 11 22 33 44 55 66 77 01 12 23 30 45 56 67 74
                    //      02 13 20 31 46 57 64 75 03 10 21 32 47 54 65 76
                    //      40 51 62 73 04 15 26 37 41 52 63 70 05 16 27 34
                    //      42 53 60 71 06 17 24 35 43 50 61 72 07 14 25 36
                    // to
                    //      00 10 20 30 44 54 64 74 04 14 24 34 40 50 60 70
                    //      01 11 21 31 45 55 65 75 05 15 25 35 41 51 61 71
                    //      02 12 22 32 46 56 66 76 06 16 26 36 42 52 62 72
                    //      03 13 23 33 47 57 67 77 07 17 27 37 43 53 63 73
                    {
                        __m512i _s0 = _mm512_shuffle_i32x4(_sum0, _sum2, _MM_SHUFFLE(0, 1, 1, 0));
                        __m512i _s1 = _mm512_shuffle_i32x4(_sum1, _sum3, _MM_SHUFFLE(2, 3, 3, 2));
                        __m512i _s2 = _mm512_shuffle_i32x4(_sum1, _sum3, _MM_SHUFFLE(0, 1, 1, 0));
                        __m512i _s3 = _mm512_shuffle_i32x4(_sum0, _sum2, _MM_SHUFFLE(2, 3, 3, 2));
                        _s1 = _mm512_shuffle_epi32(_s1, _MM_PERM_ADCB);
                        _s2 = _mm512_shuffle_epi32(_s2, _MM_PERM_BADC);
                        _s3 = _mm512_shuffle_epi32(_s3, _MM_PERM_CBAD);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_s0, _s1);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_s0, _s1);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_s2, _s3);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_s2, _s3);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(1, 3, 1, 3));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(1, 3, 1, 3));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16), _sum1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4), _sum2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 4 + 16), _sum3);
#else
                    // from
                    //      00 11 22 33 44 55 66 77
                    //      01 12 23 30 45 56 67 74
                    //      02 13 20 31 46 57 64 75
                    //      03 10 21 32 47 54 65 76
                    //      40 51 62 73 04 15 26 37
                    //      41 52 63 70 05 16 27 34
                    //      42 53 60 71 06 17 24 35
                    //      43 50 61 72 07 14 25 36
                    // to
                    //      00 10 20 30 44 54 64 74
                    //      01 11 21 31 45 55 65 75
                    //      02 12 22 32 46 56 66 76
                    //      03 13 23 33 47 57 67 77
                    //      40 50 60 70 04 14 24 34
                    //      41 51 61 71 05 15 25 35
                    //      42 52 62 72 06 16 26 36
                    //      43 53 63 73 07 17 27 37
                    {
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum2 = _mm256_shuffle_epi32(_sum2, _MM_SHUFFLE(1, 0, 3, 2));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(0, 3, 2, 1));
                        _sum5 = _mm256_shuffle_epi32(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum6 = _mm256_shuffle_epi32(_sum6, _MM_SHUFFLE(1, 0, 3, 2));
                        _sum7 = _mm256_shuffle_epi32(_sum7, _MM_SHUFFLE(0, 3, 2, 1));
                        __m256i _tmp0 = _mm256_unpacklo_epi32(_sum0, _sum3);
                        __m256i _tmp1 = _mm256_unpackhi_epi32(_sum0, _sum3);
                        __m256i _tmp2 = _mm256_unpacklo_epi32(_sum2, _sum1);
                        __m256i _tmp3 = _mm256_unpackhi_epi32(_sum2, _sum1);
                        __m256i _tmp4 = _mm256_unpacklo_epi32(_sum4, _sum7);
                        __m256i _tmp5 = _mm256_unpackhi_epi32(_sum4, _sum7);
                        __m256i _tmp6 = _mm256_unpacklo_epi32(_sum6, _sum5);
                        __m256i _tmp7 = _mm256_unpackhi_epi32(_sum6, _sum5);
                        _sum0 = _mm256_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm256_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm256_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm256_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm256_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm256_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm256_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm256_unpackhi_epi64(_tmp7, _tmp5);
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum5 = _mm256_shuffle_epi32(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum7 = _mm256_shuffle_epi32(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
                    }

                    __m256i _tmp0 = _mm256_permute2x128_si256(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp1 = _mm256_permute2x128_si256(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp2 = _mm256_permute2x128_si256(_sum4, _sum5, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp3 = _mm256_permute2x128_si256(_sum6, _sum7, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp4 = _mm256_permute2x128_si256(_sum4, _sum5, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp5 = _mm256_permute2x128_si256(_sum6, _sum7, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp6 = _mm256_permute2x128_si256(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp7 = _mm256_permute2x128_si256(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_si256((__m256i*)outptr0, _tmp0);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 8), _tmp1);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 8 * 2), _tmp2);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 8 * 3), _tmp3);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), _tmp4);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4 + 8), _tmp5);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4 + 8 * 2), _tmp6);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4 + 8 * 3), _tmp7);
#endif // __AVX512F__
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    // from
                    //      00 11 22 33 44 55 66 77 01 12 23 30 45 56 67 74
                    //      02 13 20 31 46 57 64 75 03 10 21 32 47 54 65 76
                    //      40 51 62 73 04 15 26 37 41 52 63 70 05 16 27 34
                    //      42 53 60 71 06 17 24 35 43 50 61 72 07 14 25 36
                    // to
                    //      00 01 02 03 44 45 46 47 04 05 06 07 40 41 42 43
                    //      10 11 12 13 54 55 56 57 14 15 16 17 50 51 52 53
                    //      20 21 22 23 64 65 66 67 24 25 26 27 60 61 62 63
                    //      30 31 32 33 74 75 76 77 34 35 36 37 70 71 72 73
                    {
                        __m512i _s0 = _mm512_shuffle_i32x4(_sum0, _sum2, _MM_SHUFFLE(0, 1, 1, 0));
                        __m512i _s1 = _mm512_shuffle_i32x4(_sum0, _sum2, _MM_SHUFFLE(2, 3, 3, 2));
                        __m512i _s2 = _mm512_shuffle_i32x4(_sum1, _sum3, _MM_SHUFFLE(0, 1, 1, 0));
                        __m512i _s3 = _mm512_shuffle_i32x4(_sum1, _sum3, _MM_SHUFFLE(2, 3, 3, 2));
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_s0, _s1);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_s0, _s1);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_s2, _s3);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_s2, _s3);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                    }

                    _sum0 = _mm512_shuffle_i32x4(_sum0, _sum0, _MM_SHUFFLE(1, 3, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_sum1, _sum1, _MM_SHUFFLE(1, 3, 2, 0));
                    _sum2 = _mm512_shuffle_i32x4(_sum2, _sum2, _MM_SHUFFLE(1, 3, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_sum3, _sum3, _MM_SHUFFLE(1, 3, 2, 0));

                    _mm256_storeu_si256((__m256i*)outptr0, _mm512_extracti32x8_epi32(_sum0, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep), _mm512_extracti32x8_epi32(_sum1, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 2), _mm512_extracti32x8_epi32(_sum2, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 3), _mm512_extracti32x8_epi32(_sum3, 0));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), _mm512_extracti32x8_epi32(_sum0, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 5), _mm512_extracti32x8_epi32(_sum1, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 6), _mm512_extracti32x8_epi32(_sum2, 1));
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 7), _mm512_extracti32x8_epi32(_sum3, 1));
#else
                    // from
                    //      00 11 22 33 44 55 66 77
                    //      01 12 23 30 45 56 67 74
                    //      02 13 20 31 46 57 64 75
                    //      03 10 21 32 47 54 65 76
                    //      40 51 62 73 04 15 26 37
                    //      41 52 63 70 05 16 27 34
                    //      42 53 60 71 06 17 24 35
                    //      43 50 61 72 07 14 25 36
                    // to
                    //      00 01 02 03 44 45 46 47
                    //      10 11 12 13 54 55 56 57
                    //      20 21 22 23 64 65 66 67
                    //      30 31 32 33 74 75 76 77
                    //      40 41 42 43 04 05 06 07
                    //      50 51 52 53 14 15 16 17
                    //      60 61 62 63 24 25 26 27
                    //      70 71 72 73 34 35 36 37
                    {
                        __m256i _tmp0 = _mm256_unpacklo_epi32(_sum0, _sum1);
                        __m256i _tmp1 = _mm256_unpackhi_epi32(_sum0, _sum1);
                        __m256i _tmp2 = _mm256_unpacklo_epi32(_sum2, _sum3);
                        __m256i _tmp3 = _mm256_unpackhi_epi32(_sum2, _sum3);
                        __m256i _tmp4 = _mm256_unpacklo_epi32(_sum4, _sum5);
                        __m256i _tmp5 = _mm256_unpackhi_epi32(_sum4, _sum5);
                        __m256i _tmp6 = _mm256_unpacklo_epi32(_sum6, _sum7);
                        __m256i _tmp7 = _mm256_unpackhi_epi32(_sum6, _sum7);
                        _sum0 = _mm256_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm256_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm256_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm256_unpackhi_epi64(_tmp3, _tmp1);
                        _sum4 = _mm256_unpacklo_epi64(_tmp4, _tmp6);
                        _sum5 = _mm256_unpackhi_epi64(_tmp4, _tmp6);
                        _sum6 = _mm256_unpacklo_epi64(_tmp7, _tmp5);
                        _sum7 = _mm256_unpackhi_epi64(_tmp7, _tmp5);
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum5 = _mm256_shuffle_epi32(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum7 = _mm256_shuffle_epi32(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
                    }

                    __m256i _tmp0 = _mm256_permute2x128_si256(_sum0, _sum4, _MM_SHUFFLE(0, 3, 0, 0));
                    __m256i _tmp1 = _mm256_permute2x128_si256(_sum1, _sum5, _MM_SHUFFLE(0, 3, 0, 0));
                    __m256i _tmp2 = _mm256_permute2x128_si256(_sum2, _sum6, _MM_SHUFFLE(0, 3, 0, 0));
                    __m256i _tmp3 = _mm256_permute2x128_si256(_sum3, _sum7, _MM_SHUFFLE(0, 3, 0, 0));
                    __m256i _tmp4 = _mm256_permute2x128_si256(_sum4, _sum0, _MM_SHUFFLE(0, 3, 0, 0));
                    __m256i _tmp5 = _mm256_permute2x128_si256(_sum5, _sum1, _MM_SHUFFLE(0, 3, 0, 0));
                    __m256i _tmp6 = _mm256_permute2x128_si256(_sum6, _sum2, _MM_SHUFFLE(0, 3, 0, 0));
                    __m256i _tmp7 = _mm256_permute2x128_si256(_sum7, _sum3, _MM_SHUFFLE(0, 3, 0, 0));

                    _mm256_storeu_si256((__m256i*)outptr0, _tmp0);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep), _tmp1);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 2), _tmp2);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 3), _tmp3);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), _tmp4);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 5), _tmp5);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 6), _tmp6);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 7), _tmp7);
#endif // __AVX512F__
                    outptr0 += 8;
                }
            }
            else
            {
#if __AVX512F__
                _mm512_store_si512((__m512i*)outptr, _sum0);
                _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
                _mm512_store_si512((__m512i*)(outptr + 32), _sum2);
                _mm512_store_si512((__m512i*)(outptr + 48), _sum3);
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
                __m128i _pB = _mm_castpd_si128(_mm_load1_pd((const double*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 0123 4567
                // 2301 6745
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 0123
                // 1230 1230
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

#if __AVXVNNI__ || __AVX512VNNI__
                _sum0 = _mm256_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm256_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm256_dpwssd_epi32(_sum2, _pA1, _pB0);
                _sum3 = _mm256_dpwssd_epi32(_sum3, _pA1, _pB1);
#else
                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
                _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_pA1, _pB0));
                _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_pA1, _pB1));
#endif

                pA += 16;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);

                // 01234567
                // 23016745
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 01230123
                // 12301230
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB1));
                __m256i _s2 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB0));
                __m256i _s3 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB1));

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
                    // from
                    //      00 11 22 33 40 51 62 73
                    //      01 12 23 30 41 52 63 70
                    //      20 31 02 13 60 71 42 53
                    //      21 32 03 10 61 72 43 50
                    // to
                    //      00 10 20 30 40 50 60 70
                    //      01 11 21 31 41 51 61 71
                    //      02 12 22 32 42 52 62 72
                    //      03 13 23 33 43 53 63 73
                    {
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        __m256i _tmp0 = _mm256_unpacklo_epi32(_sum0, _sum3);
                        __m256i _tmp1 = _mm256_unpackhi_epi32(_sum0, _sum3);
                        __m256i _tmp2 = _mm256_unpacklo_epi32(_sum2, _sum1);
                        __m256i _tmp3 = _mm256_unpackhi_epi32(_sum2, _sum1);
                        _sum0 = _mm256_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm256_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm256_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm256_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                    }

                    _mm256_store_si256((__m256i*)outptr0, _sum0);
                    _mm256_store_si256((__m256i*)(outptr0 + 8), _sum1);
                    _mm256_store_si256((__m256i*)(outptr0 + 16), _sum2);
                    _mm256_store_si256((__m256i*)(outptr0 + 24), _sum3);
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    // from
                    //      00 11 22 33 40 51 62 73
                    //      01 12 23 30 41 52 63 70
                    //      20 31 02 13 60 71 42 53
                    //      21 32 03 10 61 72 43 50
                    // to
                    //      00 10 20 30 40 50 60 70
                    //      01 11 21 31 41 51 61 71
                    //      02 12 22 32 42 52 62 72
                    //      03 13 23 33 43 53 63 73
                    {
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        __m256i _tmp0 = _mm256_unpacklo_epi32(_sum0, _sum3);
                        __m256i _tmp1 = _mm256_unpackhi_epi32(_sum0, _sum3);
                        __m256i _tmp2 = _mm256_unpacklo_epi32(_sum2, _sum1);
                        __m256i _tmp3 = _mm256_unpackhi_epi32(_sum2, _sum1);
                        _sum0 = _mm256_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm256_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm256_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm256_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                    }

                    __m256i _tmp0 = _mm256_permute2x128_si256(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp1 = _mm256_permute2x128_si256(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp2 = _mm256_permute2x128_si256(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp3 = _mm256_permute2x128_si256(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_si256((__m256i*)outptr0, _tmp0);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 8), _tmp1);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), _tmp2);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4 + 8), _tmp3);

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    // from
                    //      00 11 22 33 40 51 62 73
                    //      01 12 23 30 41 52 63 70
                    //      20 31 02 13 60 71 42 53
                    //      21 32 03 10 61 72 43 50
                    // to
                    //      00 01 02 03 40 41 42 43
                    //      10 11 12 13 50 51 52 53
                    //      20 21 22 23 60 61 62 63
                    //      30 31 32 33 70 71 72 73
                    {
                        _sum2 = _mm256_shuffle_epi32(_sum2, _MM_SHUFFLE(1, 0, 3, 2));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(1, 0, 3, 2));
                        __m256i _tmp0 = _mm256_unpacklo_epi32(_sum0, _sum1);
                        __m256i _tmp1 = _mm256_unpackhi_epi32(_sum0, _sum1);
                        __m256i _tmp2 = _mm256_unpacklo_epi32(_sum2, _sum3);
                        __m256i _tmp3 = _mm256_unpackhi_epi32(_sum2, _sum3);
                        _sum0 = _mm256_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm256_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm256_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm256_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                    }

                    _mm_storeu_si128((__m128i*)outptr0, _mm256_extracti128_si256(_sum0, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep), _mm256_extracti128_si256(_sum1, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2), _mm256_extracti128_si256(_sum2, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3), _mm256_extracti128_si256(_sum3, 0));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 4), _mm256_extracti128_si256(_sum0, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 5), _mm256_extracti128_si256(_sum1, 1));
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 6), _mm256_extracti128_si256(_sum2, 1));
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
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 0123 4567

                // 0101 0101
                // 1010 1010
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 1, 0, 1));

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
                __m128i _pB = _mm_set1_epi16(((const short*)pB)[0]);

                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);

                // 01234567

                // 01010101
                // 10101010
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 1, 0, 1)), _MM_SHUFFLE(0, 1, 0, 1));

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB1));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                // from
                //      00 11 20 31 40 51 60 71
                //      01 10 21 30 41 50 61 70
                // to
                //      00 10 20 30 40 50 60 70
                //      01 11 21 31 41 51 61 71
                {
                    __m256i _tmp0 = _mm256_shuffle_epi32(_sum0, _MM_SHUFFLE(3, 1, 2, 0));
                    __m256i _tmp1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(0, 2, 3, 1));
                    _sum0 = _mm256_unpacklo_epi32(_tmp0, _tmp1);
                    _sum1 = _mm256_unpackhi_epi32(_tmp0, _tmp1);
                    _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                }

                if (out_elempack == 8)
                {
                    _mm256_store_si256((__m256i*)outptr0, _sum0);
                    _mm256_store_si256((__m256i*)(outptr0 + 8), _sum1);
                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    __m256i _tmp0 = _mm256_permute2x128_si256(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp1 = _mm256_permute2x128_si256(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_si256((__m256i*)outptr0, _tmp0);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 4), _tmp1);

                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                    _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(out_hstep));
                    _mm256_i32scatter_epi32(outptr0, _vindex, _sum0, sizeof(float));
                    _mm256_i32scatter_epi32(outptr0 + 1, _vindex, _sum1, sizeof(float));
#else
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
#endif // __AVX512F__
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
                    _mm256_store_si256((__m256i*)outptr0, _sum0);
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
#if __AVX512F__
                    __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                    _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(out_hstep));
                    _mm256_i32scatter_epi32(outptr0, _vindex, _sum0, sizeof(float));
#else
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
#endif // __AVX512F__
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
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
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
                __m256i _pA = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pA));
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 0123 0123 0123
                // 2301 2301 2301 2301
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);

                // 0123 4567 89ab cdef
                // 1230 5674 9ab8 defc
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

#if __AVX512VNNI__
                _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm512_dpwssd_epi32(_sum2, _pA1, _pB0);
                _sum3 = _mm512_dpwssd_epi32(_sum3, _pA1, _pB1);
#else
                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA1, _pB0));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA1, _pB1));
#endif

                pA += 8;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 01230123 01230123
                // 23012301 23012301
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567 89abcdef
                // 12305674 9ab8defc
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));
                __m512i _s2 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0));
                __m512i _s3 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);
                _sum2 = _mm512_add_epi32(_sum2, _s2);
                _sum3 = _mm512_add_epi32(_sum3, _s3);

                pA += 4;
                pB += 16;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    // from
                    //      00 11 22 33 04 15 26 37 08 19 2a 3b 0c 1d 2e 3f
                    //      01 12 23 30 05 16 27 34 09 1a 2b 38 0d 1e 2f 3c
                    //      20 31 02 13 24 35 06 17 28 3a 0a 1b 2c 3d 0e 1f
                    //      21 32 03 10 25 36 07 14 29 3a 0b 18 2d 3e 0f 1c
                    // to
                    //      00 10 20 30 04 14 24 34 08 18 28 38 0c 1c 2c 3c
                    //      01 11 21 31 05 15 25 35 09 19 29 39 0d 1d 2d 3d
                    //      02 12 22 32 06 16 26 36 0a 1a 2a 3a 0e 1e 2e 3e
                    //      03 13 23 33 07 17 27 37 0b 1b 2b 3b 0f 1f 2f 3f
                    {
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum3);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum3);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum1);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum1);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                    }

                    __m512i _tmp0 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp1 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i _tmp2 = _mm512_shuffle_i32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i _tmp3 = _mm512_shuffle_i32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    _sum0 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_i32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_i32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 16), _sum1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 32), _sum2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + 48), _sum3);
                    outptr0 += 64;
                }
                if (out_elempack == 1)
                {
                    // from
                    //      00 11 22 33 04 15 26 37 08 19 2a 3b 0c 1d 2e 3f
                    //      01 12 23 30 05 16 27 34 09 1a 2b 38 0d 1e 2f 3c
                    //      20 31 02 13 24 35 06 17 28 3a 0a 1b 2c 3d 0e 1f
                    //      21 32 03 10 25 36 07 14 29 3a 0b 18 2d 3e 0f 1c
                    // to
                    //      00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
                    //      10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
                    //      20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
                    //      30 31 32 33 34 35 36 37 38 39 3a 3b 3c 3d 3e 3f
                    {
                        _sum2 = _mm512_shuffle_epi32(_sum2, _MM_PERM_BADC);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_BADC);
                        __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum1);
                        __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum1);
                        __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum3);
                        __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum3);
                        _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm512_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm512_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);
                        _sum3 = _mm512_shuffle_epi32(_sum3, _MM_PERM_CBAD);
                    }

                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep), _sum1);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 2), _sum2);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep * 3), _sum3);

                    outptr0 += 16;
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
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

#if __AVX2__
            __m256i _sum0;
            __m256i _sum1;
            __m256i _sum2;
            __m256i _sum3;
#else
            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;
            __m128i _sum4;
            __m128i _sum5;
            __m128i _sum6;
            __m128i _sum7;
#endif

            if (k == 0)
            {
#if __AVX2__
                _sum0 = _mm256_setzero_si256();
                _sum1 = _mm256_setzero_si256();
                _sum2 = _mm256_setzero_si256();
                _sum3 = _mm256_setzero_si256();
#else
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
                _sum2 = _mm_setzero_si128();
                _sum3 = _mm_setzero_si128();
                _sum4 = _mm_setzero_si128();
                _sum5 = _mm_setzero_si128();
                _sum6 = _mm_setzero_si128();
                _sum7 = _mm_setzero_si128();
#endif
            }
            else
            {
#if __AVX2__
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
                _sum1 = _mm256_load_si256((const __m256i*)(outptr + 8));
                _sum2 = _mm256_load_si256((const __m256i*)(outptr + 16));
                _sum3 = _mm256_load_si256((const __m256i*)(outptr + 24));
#else
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
                _sum2 = _mm_load_si128((const __m128i*)(outptr + 8));
                _sum3 = _mm_load_si128((const __m128i*)(outptr + 12));
                _sum4 = _mm_load_si128((const __m128i*)(outptr + 16));
                _sum5 = _mm_load_si128((const __m128i*)(outptr + 20));
                _sum6 = _mm_load_si128((const __m128i*)(outptr + 24));
                _sum7 = _mm_load_si128((const __m128i*)(outptr + 28));
#endif
            }

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __AVX2__
                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 0123 0123
                // 2301 2301
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 4567
                // 1230 5674
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

#if __AVXVNNI__ || __AVX512VNNI__
                _sum0 = _mm256_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm256_dpwssd_epi32(_sum1, _pA0, _pB1);
                _sum2 = _mm256_dpwssd_epi32(_sum2, _pA1, _pB0);
                _sum3 = _mm256_dpwssd_epi32(_sum3, _pA1, _pB1);
#else
                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
                _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_pA1, _pB0));
                _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_pA1, _pB1));
#endif
#else // __AVX2__
#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif
                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pBl = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pBh = _mm_unpackhi_epi8(_pB, _extpB);

                // 0123
                // 2301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123
                // 4567
                // 1230
                // 5674
                __m128i _pB0 = _pBl;
                __m128i _pB1 = _pBh;
                __m128i _pB2 = _mm_shuffle_epi32(_pBl, _MM_SHUFFLE(0, 3, 2, 1));
                __m128i _pB3 = _mm_shuffle_epi32(_pBh, _MM_SHUFFLE(0, 3, 2, 1));

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA0, _pB1, _sum1);
                _sum2 = _mm_maddd_epi16(_pA0, _pB2, _sum2);
                _sum3 = _mm_maddd_epi16(_pA0, _pB3, _sum3);
                _sum4 = _mm_maddd_epi16(_pA1, _pB0, _sum4);
                _sum5 = _mm_maddd_epi16(_pA1, _pB1, _sum5);
                _sum6 = _mm_maddd_epi16(_pA1, _pB2, _sum6);
                _sum7 = _mm_maddd_epi16(_pA1, _pB3, _sum7);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA0, _pB1));
                _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_pA0, _pB2));
                _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_pA0, _pB3));
                _sum4 = _mm_add_epi32(_sum4, _mm_madd_epi16(_pA1, _pB0));
                _sum5 = _mm_add_epi32(_sum5, _mm_madd_epi16(_pA1, _pB1));
                _sum6 = _mm_add_epi32(_sum6, _mm_madd_epi16(_pA1, _pB2));
                _sum7 = _mm_add_epi32(_sum7, _mm_madd_epi16(_pA1, _pB3));
#endif
#endif // __AVX2__

                pA += 8;
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

#if __AVX2__
                // 01230123
                // 23012301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567
                // 12305674
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB1));
                __m256i _s2 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB0));
                __m256i _s3 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB1));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);
                _sum2 = _mm256_add_epi32(_sum2, _s2);
                _sum3 = _mm256_add_epi32(_sum3, _s3);
#else // __AVX2__
#if __XOP__
                // 00112233
                // 22330011
                __m128i _pA0 = _mm_unpacklo_epi16(_pA, _pA);
                __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 00112233
                // 44556677
                // 1.2.3.0.
                __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);
                __m128i _pB1 = _mm_unpackhi_epi16(_pB, _pB);
                __m128i _pB2 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m128i _pB3 = _mm_shuffle_epi32(_pB1, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm_maccd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maccd_epi16(_pA0, _pB1, _sum1);
                _sum2 = _mm_maccd_epi16(_pA0, _pB2, _sum2);
                _sum3 = _mm_maccd_epi16(_pA0, _pB3, _sum3);
                _sum4 = _mm_maccd_epi16(_pA1, _pB0, _sum4);
                _sum5 = _mm_maccd_epi16(_pA1, _pB1, _sum5);
                _sum6 = _mm_maccd_epi16(_pA1, _pB2, _sum6);
                _sum7 = _mm_maccd_epi16(_pA1, _pB3, _sum7);
#else
                // 01230123
                // 23012301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567
                // 12305674
                __m128i _pB01 = _pB;
                __m128i _pB23 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m128i _sl0 = _mm_mullo_epi16(_pA0, _pB01);
                __m128i _sh0 = _mm_mulhi_epi16(_pA0, _pB01);
                __m128i _sl1 = _mm_mullo_epi16(_pA0, _pB23);
                __m128i _sh1 = _mm_mulhi_epi16(_pA0, _pB23);
                __m128i _sl2 = _mm_mullo_epi16(_pA1, _pB01);
                __m128i _sh2 = _mm_mulhi_epi16(_pA1, _pB01);
                __m128i _sl3 = _mm_mullo_epi16(_pA1, _pB23);
                __m128i _sh3 = _mm_mulhi_epi16(_pA1, _pB23);
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
#endif
#endif // __AVX2__

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
#if __AVX2__
                    // from
                    //      00 11 22 33 04 15 26 37
                    //      01 12 23 30 05 16 27 34
                    //      20 31 02 13 24 35 06 17
                    //      21 32 03 10 25 36 07 14
                    // to
                    //      00 10 20 30 04 14 24 34
                    //      01 11 21 31 05 15 25 35
                    //      02 12 22 32 06 16 26 36
                    //      03 13 23 33 07 17 27 37
                    {
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        __m256i _tmp0 = _mm256_unpacklo_epi32(_sum0, _sum3);
                        __m256i _tmp1 = _mm256_unpackhi_epi32(_sum0, _sum3);
                        __m256i _tmp2 = _mm256_unpacklo_epi32(_sum2, _sum1);
                        __m256i _tmp3 = _mm256_unpackhi_epi32(_sum2, _sum1);
                        _sum0 = _mm256_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm256_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm256_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm256_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                    }

                    __m256i _tmp0 = _mm256_permute2x128_si256(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp1 = _mm256_permute2x128_si256(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256i _tmp2 = _mm256_permute2x128_si256(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256i _tmp3 = _mm256_permute2x128_si256(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_si256((__m256i*)outptr0, _tmp0);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 8), _tmp1);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 16), _tmp2);
                    _mm256_storeu_si256((__m256i*)(outptr0 + 24), _tmp3);
#else
                    // from
                    //      00 11 22 33  04 15 26 37
                    //      01 12 23 30  05 16 27 34
                    //      20 31 02 13  24 35 06 17
                    //      21 32 03 10  25 36 07 14
                    // to
                    //      00 10 20 30  04 14 24 34
                    //      01 11 21 31  05 15 25 35
                    //      02 12 22 32  06 16 26 36
                    //      03 13 23 33  07 17 27 37
                    {
                        _sum2 = _mm_shuffle_epi32(_sum2, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum6 = _mm_shuffle_epi32(_sum6, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum7 = _mm_shuffle_epi32(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
                        __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum6);
                        __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum6);
                        __m128i _tmp2 = _mm_unpacklo_epi32(_sum1, _sum7);
                        __m128i _tmp3 = _mm_unpackhi_epi32(_sum1, _sum7);
                        __m128i _tmp4 = _mm_unpacklo_epi32(_sum4, _sum2);
                        __m128i _tmp5 = _mm_unpackhi_epi32(_sum4, _sum2);
                        __m128i _tmp6 = _mm_unpacklo_epi32(_sum5, _sum3);
                        __m128i _tmp7 = _mm_unpackhi_epi32(_sum5, _sum3);
                        _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp4);
                        _sum1 = _mm_unpacklo_epi64(_tmp2, _tmp6);
                        _sum2 = _mm_unpackhi_epi64(_tmp0, _tmp4);
                        _sum3 = _mm_unpackhi_epi64(_tmp2, _tmp6);
                        _sum4 = _mm_unpacklo_epi64(_tmp5, _tmp1);
                        _sum5 = _mm_unpacklo_epi64(_tmp7, _tmp3);
                        _sum6 = _mm_unpackhi_epi64(_tmp5, _tmp1);
                        _sum7 = _mm_unpackhi_epi64(_tmp7, _tmp3);
                        _sum2 = _mm_shuffle_epi32(_sum2, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum6 = _mm_shuffle_epi32(_sum6, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum7 = _mm_shuffle_epi32(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
                    }

                    _mm_store_si128((__m128i*)outptr0, _sum0);
                    _mm_store_si128((__m128i*)(outptr0 + 4), _sum2);
                    _mm_store_si128((__m128i*)(outptr0 + 8), _sum4);
                    _mm_store_si128((__m128i*)(outptr0 + 12), _sum6);
                    _mm_store_si128((__m128i*)(outptr0 + 16), _sum1);
                    _mm_store_si128((__m128i*)(outptr0 + 20), _sum3);
                    _mm_store_si128((__m128i*)(outptr0 + 24), _sum5);
                    _mm_store_si128((__m128i*)(outptr0 + 28), _sum7);
#endif // __AVX2__
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
#if __AVX2__
                    // from
                    //      00 11 22 33 04 15 26 37
                    //      01 12 23 30 05 16 27 34
                    //      20 31 02 13 24 35 06 17
                    //      21 32 03 10 25 36 07 14
                    // to
                    //      00 01 02 03 04 05 06 07
                    //      10 11 12 13 14 15 16 17
                    //      20 21 22 23 24 25 26 27
                    //      30 31 32 33 34 35 36 37
                    {
                        _sum2 = _mm256_shuffle_epi32(_sum2, _MM_SHUFFLE(1, 0, 3, 2));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(1, 0, 3, 2));
                        __m256i _tmp0 = _mm256_unpacklo_epi32(_sum0, _sum1);
                        __m256i _tmp1 = _mm256_unpackhi_epi32(_sum0, _sum1);
                        __m256i _tmp2 = _mm256_unpacklo_epi32(_sum2, _sum3);
                        __m256i _tmp3 = _mm256_unpackhi_epi32(_sum2, _sum3);
                        _sum0 = _mm256_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm256_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm256_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm256_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm256_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                    }

                    _mm256_storeu_si256((__m256i*)outptr0, _sum0);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep), _sum1);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 2), _sum2);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep * 3), _sum3);
#else
                    // from
                    //      00 11 22 33  04 15 26 37
                    //      01 12 23 30  05 16 27 34
                    //      20 31 02 13  24 35 06 17
                    //      21 32 03 10  25 36 07 14
                    // to
                    //      00 01 02 03  04 05 06 07
                    //      10 11 12 13  14 15 16 17
                    //      20 21 22 23  24 25 26 27
                    //      30 31 32 33  34 35 36 37
                    {
                        _sum4 = _mm_shuffle_epi32(_sum4, _MM_SHUFFLE(1, 0, 3, 2));
                        _sum5 = _mm_shuffle_epi32(_sum5, _MM_SHUFFLE(1, 0, 3, 2));
                        _sum6 = _mm_shuffle_epi32(_sum6, _MM_SHUFFLE(1, 0, 3, 2));
                        _sum7 = _mm_shuffle_epi32(_sum7, _MM_SHUFFLE(1, 0, 3, 2));
                        __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum2);
                        __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum2);
                        __m128i _tmp2 = _mm_unpacklo_epi32(_sum1, _sum3);
                        __m128i _tmp3 = _mm_unpackhi_epi32(_sum1, _sum3);
                        __m128i _tmp4 = _mm_unpacklo_epi32(_sum4, _sum6);
                        __m128i _tmp5 = _mm_unpackhi_epi32(_sum4, _sum6);
                        __m128i _tmp6 = _mm_unpacklo_epi32(_sum5, _sum7);
                        __m128i _tmp7 = _mm_unpackhi_epi32(_sum5, _sum7);
                        _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp4);
                        _sum1 = _mm_unpacklo_epi64(_tmp2, _tmp6);
                        _sum2 = _mm_unpackhi_epi64(_tmp0, _tmp4);
                        _sum3 = _mm_unpackhi_epi64(_tmp2, _tmp6);
                        _sum4 = _mm_unpacklo_epi64(_tmp5, _tmp1);
                        _sum5 = _mm_unpacklo_epi64(_tmp7, _tmp3);
                        _sum6 = _mm_unpackhi_epi64(_tmp5, _tmp1);
                        _sum7 = _mm_unpackhi_epi64(_tmp7, _tmp3);
                        _sum2 = _mm_shuffle_epi32(_sum2, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum6 = _mm_shuffle_epi32(_sum6, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum7 = _mm_shuffle_epi32(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
                    }

                    _mm_storeu_si128((__m128i*)outptr0, _sum0);
                    _mm_storeu_si128((__m128i*)(outptr0 + 4), _sum1);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep), _sum2);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep + 4), _sum3);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2), _sum4);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 2 + 4), _sum5);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3), _sum6);
                    _mm_storeu_si128((__m128i*)(outptr0 + out_hstep * 3 + 4), _sum7);
#endif // __AVX2__
                    outptr0 += 8;
                }
            }
            else
            {
#if __AVX2__
                _mm256_store_si256((__m256i*)outptr, _sum0);
                _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
                _mm256_store_si256((__m256i*)(outptr + 16), _sum2);
                _mm256_store_si256((__m256i*)(outptr + 24), _sum3);
#else
                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
                _mm_store_si128((__m128i*)(outptr + 8), _sum2);
                _mm_store_si128((__m128i*)(outptr + 12), _sum3);
                _mm_store_si128((__m128i*)(outptr + 16), _sum4);
                _mm_store_si128((__m128i*)(outptr + 20), _sum5);
                _mm_store_si128((__m128i*)(outptr + 24), _sum6);
                _mm_store_si128((__m128i*)(outptr + 28), _sum7);
#endif
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

                // 0123
                // 2301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123
                // 1230
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(0, 3, 2, 1));

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA0, _pB1, _sum1);
                _sum2 = _mm_maddd_epi16(_pA1, _pB0, _sum2);
                _sum3 = _mm_maddd_epi16(_pA1, _pB1, _sum3);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA0, _pB1));
                _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_pA1, _pB0));
                _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_pA1, _pB1));
#endif

                pA += 8;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

#if __XOP__
                // 00112233
                // 22330011
                __m128i _pA0 = _mm_unpacklo_epi16(_pA, _pA);
                __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 00112233
                // 1.2.3.0.
                __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm_maccd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maccd_epi16(_pA0, _pB1, _sum1);
                _sum2 = _mm_maccd_epi16(_pA1, _pB0, _sum2);
                _sum3 = _mm_maccd_epi16(_pA1, _pB1, _sum3);
#else
                // 0123 0123
                // 2301 2301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 0123 1230
                __m128i _pB01 = _mm_shufflehi_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1));

                __m128i _sl0 = _mm_mullo_epi16(_pA0, _pB01);
                __m128i _sh0 = _mm_mulhi_epi16(_pA0, _pB01);
                __m128i _sl1 = _mm_mullo_epi16(_pA1, _pB01);
                __m128i _sh1 = _mm_mulhi_epi16(_pA1, _pB01);
                __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                __m128i _s1 = _mm_unpackhi_epi16(_sl0, _sh0);
                __m128i _s2 = _mm_unpacklo_epi16(_sl1, _sh1);
                __m128i _s3 = _mm_unpackhi_epi16(_sl1, _sh1);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
                _sum2 = _mm_add_epi32(_sum2, _s2);
                _sum3 = _mm_add_epi32(_sum3, _s3);
#endif

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    // from
                    //      00 11 22 33
                    //      01 12 23 30
                    //      20 31 02 13
                    //      21 32 03 10
                    // to
                    //      00 10 20 30
                    //      01 11 21 31
                    //      02 12 22 32
                    //      03 13 23 33
                    {
                        _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                        __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum3);
                        __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum3);
                        __m128i _tmp2 = _mm_unpacklo_epi32(_sum2, _sum1);
                        __m128i _tmp3 = _mm_unpackhi_epi32(_sum2, _sum1);
                        _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                    }

                    _mm_store_si128((__m128i*)outptr0, _sum0);
                    _mm_store_si128((__m128i*)(outptr0 + 4), _sum1);
                    _mm_store_si128((__m128i*)(outptr0 + 8), _sum2);
                    _mm_store_si128((__m128i*)(outptr0 + 12), _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    // from
                    //      00 11 22 33
                    //      01 12 23 30
                    //      20 31 02 13
                    //      21 32 03 10
                    // to
                    //      00 01 02 03
                    //      10 11 12 13
                    //      20 21 22 23
                    //      30 31 32 33
                    {
                        _sum2 = _mm_shuffle_epi32(_sum2, _MM_SHUFFLE(1, 0, 3, 2));
                        _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(1, 0, 3, 2));
                        __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum1);
                        __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum1);
                        __m128i _tmp2 = _mm_unpacklo_epi32(_sum2, _sum3);
                        __m128i _tmp3 = _mm_unpackhi_epi32(_sum2, _sum3);
                        _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp2);
                        _sum1 = _mm_unpackhi_epi64(_tmp0, _tmp2);
                        _sum2 = _mm_unpacklo_epi64(_tmp3, _tmp1);
                        _sum3 = _mm_unpackhi_epi64(_tmp3, _tmp1);
                        _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                        _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                    }

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

                // 0123

                // 0101
                // 1010
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(2, 3, 0, 1));

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
                __m128i _pB = _mm_set1_epi16(((const short*)pB)[0]);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

#if __XOP__
                // 00112233
                _pA = _mm_unpacklo_epi16(_pA, _pA);

                // 00110011
                // 1.0.1.0.
                __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 3, 0, 1));

                _sum0 = _mm_maccd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maccd_epi16(_pA, _pB1, _sum1);
#else
                // 01230123
                // 01011010
                __m128i _pB01 = _mm_shufflehi_epi16(_pB, _MM_SHUFFLE(0, 1, 0, 1));

                __m128i _sl = _mm_mullo_epi16(_pA, _pB01);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB01);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
#endif

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                // from
                //      00 11 20 31
                //      01 10 21 30
                // to
                //      00 10 20 30
                //      01 11 21 31
                {
                    __m128i _tmp0 = _mm_shuffle_epi32(_sum0, _MM_SHUFFLE(3, 1, 2, 0));
                    __m128i _tmp1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(0, 2, 3, 1));
                    _sum0 = _mm_unpacklo_epi32(_tmp0, _tmp1);
                    _sum1 = _mm_unpackhi_epi32(_tmp0, _tmp1);
                    _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                }

                if (out_elempack == 4)
                {
                    _mm_store_si128((__m128i*)outptr0, _sum0);
                    _mm_store_si128((__m128i*)(outptr0 + 4), _sum1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                    _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(out_hstep));
                    _mm_i32scatter_epi32(outptr0, _vindex, _sum0, sizeof(float));
                    _mm_i32scatter_epi32(outptr0 + 1, _vindex, _sum1, sizeof(float));
#else
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
#endif // __AVX512F__
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
                __m128i _pB = _mm_set1_epi16(((const short*)pB)[0]);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB, _sum0);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB));
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

#if __XOP__
                _pA = _mm_unpacklo_epi16(_pA, _pA);

                _sum0 = _mm_maccd_epi16(_pA, _pB, _sum0);
#else
                __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
#endif

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_store_si128((__m128i*)outptr0, _sum0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
#if __AVX512F__
                    __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                    _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(out_hstep));
                    _mm_i32scatter_epi32(outptr0, _vindex, _sum0, sizeof(float));
#else
                    int sum0[4];
                    _mm_storeu_si128((__m128i*)sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
#endif // __AVX512F__
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
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
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

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pA));
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0101 0101 0101 0101

                // 0123 4567 89ab cdef
                // 1230 5674 9ab8 defc
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

#if __AVX512VNNI__
                _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm512_dpwssd_epi32(_sum1, _pA0, _pB1);
#else
                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
#endif

                pA += 4;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(((const short*)pA)[0]);
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 01010101 01010101

                // 01234567 89abcdef
                // 12305674 9ab8defc
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);

                pA += 2;
                pB += 16;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    // 00 11 02 13  04 15 06 17  08 19 0a 1b  0c 1d 0e 1f
                    // 01 12 03 10  05 16 07 14  09 1a 0b 18  0d 1e 0f 1c

                    __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum1);
                    __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum1);

                    _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp1);
                    _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp1);

                    _sum0 = _sum0;
                    _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);

                    // 0123 4567 89ab cdef  x 0
                    // 0123 4567 89ab cdef  x 1

                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    _mm512_storeu_si512((__m512i*)(outptr0 + out_hstep), _sum1);
                    outptr0 += 16;
                }
            }
            else
            {
                _mm512_store_si512((__m512i*)outptr, _sum0);
                _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
            }

            outptr += 32;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
#if __AVX2__
            __m256i _sum0;
            __m256i _sum1;
#else
            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;
#endif

            if (k == 0)
            {
#if __AVX2__
                _sum0 = _mm256_setzero_si256();
                _sum1 = _mm256_setzero_si256();
#else
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
                _sum2 = _mm_setzero_si128();
                _sum3 = _mm_setzero_si128();
#endif
            }
            else
            {
#if __AVX2__
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
                _sum1 = _mm256_load_si256((const __m256i*)(outptr + 8));
#else
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
                _sum2 = _mm_load_si128((const __m128i*)(outptr + 8));
                _sum3 = _mm_load_si128((const __m128i*)(outptr + 12));
#endif
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __AVX2__
                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 0101 0101

                // 0123 4567
                // 1230 5674
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

#if __AVX512VNNI__ || __AVXVNNI__
                _sum0 = _mm256_dpwssd_epi32(_sum0, _pA0, _pB0);
                _sum1 = _mm256_dpwssd_epi32(_sum1, _pA0, _pB1);
#else
                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
#endif // __AVX512VNNI__ || __AVXVNNI__
#else  // __AVX2__
#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pB0 = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pB1 = _mm_unpackhi_epi8(_pB, _extpB);

                // 0101
                // 1010
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 0123
                // 4567

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA0, _pB1, _sum1);
                _sum2 = _mm_maddd_epi16(_pA1, _pB0, _sum2);
                _sum3 = _mm_maddd_epi16(_pA1, _pB1, _sum3);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA0, _pB1));
                _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_pA1, _pB0));
                _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_pA1, _pB1));
#endif
#endif // __AVX2__

                pA += 4;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(((const short*)pA)[0]);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __AVX2__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);

                // 01010101

                // 01234567
                // 12305674
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB1));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);
#else // __AVX2__
#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 01010101
                // 10101010
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pA, _MM_SHUFFLE(2, 3, 0, 1)), _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567

                __m128i _sl0 = _mm_mullo_epi16(_pA0, _pB);
                __m128i _sh0 = _mm_mulhi_epi16(_pA0, _pB);
                __m128i _sl1 = _mm_mullo_epi16(_pA1, _pB);
                __m128i _sh1 = _mm_mulhi_epi16(_pA1, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                __m128i _s1 = _mm_unpackhi_epi16(_sl0, _sh0);
                __m128i _s2 = _mm_unpacklo_epi16(_sl1, _sh1);
                __m128i _s3 = _mm_unpackhi_epi16(_sl1, _sh1);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
                _sum2 = _mm_add_epi32(_sum2, _s2);
                _sum3 = _mm_add_epi32(_sum3, _s3);
#endif // __AVX2__

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
#if __AVX2__
                    // 00 11 02 13  04 15 06 17
                    // 01 12 03 10  05 16 07 14
                    __m256i _tmp0 = _mm256_unpacklo_epi32(_sum0, _sum1);
                    __m256i _tmp1 = _mm256_unpackhi_epi32(_sum0, _sum1);

                    // 00 01 11 12  04 05 15 16
                    // 02 03 13 10  06 07 17 14
                    _sum0 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                    _sum1 = _mm256_unpackhi_epi64(_tmp0, _tmp1);

                    // 00 01 02 03  04 05 06 07
                    // 11 12 13 10  15 16 17 14
                    _sum1 = _mm256_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));

                    _mm256_storeu_si256((__m256i*)outptr0, _sum0);
                    _mm256_storeu_si256((__m256i*)(outptr0 + out_hstep), _sum1);
#else
                    // 00 11 02 13
                    // 04 15 06 17
                    // 10 01 12 03
                    // 14 05 16 07
                    _sum2 = _mm_shuffle_epi32(_sum2, _MM_SHUFFLE(2, 3, 0, 1));
                    _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 3, 0, 1));

                    // 00 11 02 13
                    // 04 15 06 17
                    // 01 10 03 12
                    // 05 14 07 16

                    __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum2);
                    __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum2);
                    __m128i _tmp2 = _mm_unpacklo_epi32(_sum1, _sum3);
                    __m128i _tmp3 = _mm_unpackhi_epi32(_sum1, _sum3);

                    // 00 01 11 10
                    // 02 03 13 12
                    // 04 05 15 14
                    // 06 07 17 16

                    _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                    _sum1 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                    _sum2 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                    _sum3 = _mm_unpackhi_epi64(_tmp2, _tmp3);

                    // 00 01 02 03
                    // 04 05 06 07
                    // 11 10 13 12
                    // 15 14 17 16
                    _sum2 = _mm_shuffle_epi32(_sum2, _MM_SHUFFLE(2, 3, 0, 1));
                    _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 3, 0, 1));

                    _mm_store_si128((__m128i*)outptr0, _sum0);
                    _mm_store_si128((__m128i*)(outptr0 + 4), _sum1);
                    _mm_store_si128((__m128i*)(outptr0 + out_hstep), _sum2);
                    _mm_store_si128((__m128i*)(outptr0 + out_hstep + 4), _sum3);
#endif // __AVX2__
                    outptr0 += 8;
                }
            }
            else
            {
#if __AVX2__
                _mm256_store_si256((__m256i*)outptr, _sum0);
                _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
#else
                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
                _mm_store_si128((__m128i*)(outptr + 8), _sum2);
                _mm_store_si128((__m128i*)(outptr + 12), _sum3);
#endif
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

                // 0101

                // 0123
                // 1230
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(0, 3, 2, 1));

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
                __m128i _pA = _mm_set1_epi16(((const short*)pA)[0]);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 01010101

                // 01231230
                __m128i _pB01 = _mm_shufflehi_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1));

                __m128i _sl = _mm_mullo_epi16(_pA, _pB01);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB01);
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
                    // 00 11 02 13
                    // 01 12 03 10
                    __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum1);
                    __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum1);

                    // 00 01 11 12
                    // 02 03 13 10
                    _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                    _sum1 = _mm_unpackhi_epi64(_tmp1, _tmp0);

                    // 00 01 02 03
                    // 13 10 11 12
                    _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(0, 3, 2, 1));

                    _mm_store_si128((__m128i*)outptr0, _sum0);
                    _mm_store_si128((__m128i*)(outptr0 + out_hstep), _sum1);
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
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512i _sum0;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_set1_epi16(((const short*)pA)[0]);
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

#if __AVX512VNNI__
                _sum0 = _mm512_dpwssd_epi32(_sum0, _pA0, _pB0);
#else
                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
#endif

                pA += 2;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m256i _pA = _mm256_set1_epi16(pA[0]);
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA, _pB0));

                _sum0 = _mm512_add_epi32(_sum0, _s0);

                pA += 1;
                pB += 16;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm512_storeu_si512((__m512i*)outptr0, _sum0);
                    outptr0 += 16;
                }
            }
            else
            {
                _mm512_store_si512((__m512i*)outptr, _sum0);
            }

            outptr += 16;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
#if __AVX2__
            __m256i _sum0;
#else
            __m128i _sum0;
            __m128i _sum1;
#endif

            if (k == 0)
            {
#if __AVX2__
                _sum0 = _mm256_setzero_si256();
#else
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
#endif
            }
            else
            {
#if __AVX2__
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
#else
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
#endif
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_set1_epi16(((const short*)pA)[0]);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __AVX2__
                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

#if __AVX512VNNI__ || __AVXVNNI__
                _sum0 = _mm256_dpwssd_epi32(_sum0, _pA0, _pB0);
#else
                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
#endif // __AVX512VNNI__ || __AVXVNNI__
#else  // __AVX2__
#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pB0 = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pB1 = _mm_unpackhi_epi8(_pB, _extpB);

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA, _pB1, _sum1);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA, _pB1));
#endif
#endif // __AVX2__

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

#if __AVX2__
                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
#else  // __AVX2__
                __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
#endif // __AVX2__

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
#if __AVX2__
                    _mm256_storeu_si256((__m256i*)outptr0, _sum0);
#else
                    _mm_store_si128((__m128i*)outptr0, _sum0);
                    _mm_store_si128((__m128i*)(outptr0 + 4), _sum1);
#endif
                    outptr0 += 8;
                }
            }
            else
            {
#if __AVX2__
                _mm256_store_si256((__m256i*)outptr, _sum0);
#else
                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
#endif
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
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
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
                __m128i _r89 = _mm_load_si128((const __m128i*)(p0 + 64));
                __m128i _rab = _mm_load_si128((const __m128i*)(p0 + 80));
                __m128i _rcd = _mm_load_si128((const __m128i*)(p0 + 96));
                __m128i _ref = _mm_load_si128((const __m128i*)(p0 + 112));

                __m128i _t0 = _mm_unpacklo_epi16(_r01, _r23);
                __m128i _t1 = _mm_unpackhi_epi16(_r01, _r23);
                __m128i _t2 = _mm_unpacklo_epi16(_r45, _r67);
                __m128i _t3 = _mm_unpackhi_epi16(_r45, _r67);
                __m128i _t4 = _mm_unpacklo_epi16(_r89, _rab);
                __m128i _t5 = _mm_unpackhi_epi16(_r89, _rab);
                __m128i _t6 = _mm_unpacklo_epi16(_rcd, _ref);
                __m128i _t7 = _mm_unpackhi_epi16(_rcd, _ref);

                _r01 = _mm_unpacklo_epi16(_t0, _t1);
                _r23 = _mm_unpackhi_epi16(_t0, _t1);
                _r45 = _mm_unpacklo_epi16(_t2, _t3);
                _r67 = _mm_unpackhi_epi16(_t2, _t3);
                _r89 = _mm_unpacklo_epi16(_t4, _t5);
                _rab = _mm_unpackhi_epi16(_t4, _t5);
                _rcd = _mm_unpacklo_epi16(_t6, _t7);
                _ref = _mm_unpackhi_epi16(_t6, _t7);

                __m128i _r0 = _mm_unpacklo_epi64(_r01, _r45);
                __m128i _r1 = _mm_unpacklo_epi64(_r89, _rcd);
                __m128i _r2 = _mm_unpackhi_epi64(_r01, _r45);
                __m128i _r3 = _mm_unpackhi_epi64(_r89, _rcd);
                __m128i _r4 = _mm_unpacklo_epi64(_r23, _r67);
                __m128i _r5 = _mm_unpacklo_epi64(_rab, _ref);
                __m128i _r6 = _mm_unpackhi_epi64(_r23, _r67);
                __m128i _r7 = _mm_unpackhi_epi64(_rab, _ref);

                _mm_store_si128((__m128i*)pp, _r0);
                _mm_store_si128((__m128i*)(pp + 16), _r1);
                _mm_store_si128((__m128i*)(pp + 32), _r2);
                _mm_store_si128((__m128i*)(pp + 48), _r3);
                _mm_store_si128((__m128i*)(pp + 64), _r4);
                _mm_store_si128((__m128i*)(pp + 80), _r5);
                _mm_store_si128((__m128i*)(pp + 96), _r6);
                _mm_store_si128((__m128i*)(pp + 112), _r7);

                pp += 128;
                p0 += bottom_blob.cstep * 8;
            }
        }

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)p0);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)(p0 + bottom_blob.cstep));
                __m128i _r01 = _mm_unpacklo_epi8(_r0, _r1);
                __m128i _r23 = _mm_unpackhi_epi8(_r0, _r1);
                _mm_storeu_si128((__m128i*)pp, _r01);
                _mm_storeu_si128((__m128i*)(pp + 16), _r23);
                pp += 32;
                p0 += bottom_blob.cstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                _mm_storeu_si128((__m128i*)pp, _mm_loadu_si128((const __m128i*)p0));
                pp += 16;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __AVX512F__
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

template<int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h>
#if __AVX512F__
void convolution_im2col_input_tile_int8_avx512(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
#else  // __AVX512F__
void convolution_im2col_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
#endif // __AVX512F__
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
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dy8 = (j + jj + 8) / outw;
        int dy9 = (j + jj + 9) / outw;
        int dya = (j + jj + 10) / outw;
        int dyb = (j + jj + 11) / outw;
        int dyc = (j + jj + 12) / outw;
        int dyd = (j + jj + 13) / outw;
        int dye = (j + jj + 14) / outw;
        int dyf = (j + jj + 15) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;
        int dx8 = (j + jj + 8) % outw;
        int dx9 = (j + jj + 9) % outw;
        int dxa = (j + jj + 10) % outw;
        int dxb = (j + jj + 11) % outw;
        int dxc = (j + jj + 12) % outw;
        int dxd = (j + jj + 13) % outw;
        int dxe = (j + jj + 14) % outw;
        int dxf = (j + jj + 15) % outw;

        if (dy0 == dyf)
        {
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
                    int y00 = stride_h * dy0 + dilation_h * u0;
                    int x10 = stride_w * dx0 + dilation_w * v1;
                    int y10 = stride_h * dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        __m128i _r0 = _mm_loadu_si128((const __m128i*)sptr0);
                        __m128i _r1 = _mm_loadu_si128((const __m128i*)sptr1);
                        __m128i _tmp0 = _mm_unpacklo_epi8(_r0, _r1);
                        __m128i _tmp1 = _mm_unpackhi_epi8(_r0, _r1);
                        _mm_store_si128((__m128i*)pp, _tmp0);
                        _mm_store_si128((__m128i*)(pp + 16), _tmp1);
                        pp += 32;
                    }
                    else if (stride_w == 2)
                    {
                        __m256i _r0 = _mm256_loadu_si256((const __m256i*)sptr0);
                        __m256i _r1 = _mm256_loadu_si256((const __m256i*)sptr1);
                        __m256i _tmp0 = _mm256_unpacklo_epi8(_r0, _r1);
                        __m256i _tmp1 = _mm256_unpackhi_epi8(_r0, _r1);
                        _tmp0 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_tmp0, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
                        _tmp1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_tmp1, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
                        _tmp0 = _mm256_shuffle_epi32(_tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                        _tmp1 = _mm256_shuffle_epi32(_tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                        __m256i _r01 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                        _mm256_storeu_si256((__m256i*)pp, _r01);
                        pp += 32;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp[8] = sptr0[stride_w * 4];
                        pp[9] = sptr1[stride_w * 4];
                        pp[10] = sptr0[stride_w * 5];
                        pp[11] = sptr1[stride_w * 5];
                        pp[12] = sptr0[stride_w * 6];
                        pp[13] = sptr1[stride_w * 6];
                        pp[14] = sptr0[stride_w * 7];
                        pp[15] = sptr1[stride_w * 7];
                        pp[16 + 0] = sptr0[stride_w * 8];
                        pp[16 + 1] = sptr1[stride_w * 8];
                        pp[16 + 2] = sptr0[stride_w * 9];
                        pp[16 + 3] = sptr1[stride_w * 9];
                        pp[16 + 4] = sptr0[stride_w * 10];
                        pp[16 + 5] = sptr1[stride_w * 10];
                        pp[16 + 6] = sptr0[stride_w * 11];
                        pp[16 + 7] = sptr1[stride_w * 11];
                        pp[16 + 8] = sptr0[stride_w * 12];
                        pp[16 + 9] = sptr1[stride_w * 12];
                        pp[16 + 10] = sptr0[stride_w * 13];
                        pp[16 + 11] = sptr1[stride_w * 13];
                        pp[16 + 12] = sptr0[stride_w * 14];
                        pp[16 + 13] = sptr1[stride_w * 14];
                        pp[16 + 14] = sptr0[stride_w * 15];
                        pp[16 + 15] = sptr1[stride_w * 15];
                        pp += 32;
                    }
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

                if (elempack == 8)
                {
                    __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr);
                    __m128i _r1 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 8));
                    __m128i _r2 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 16));
                    __m128i _r3 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 24));
                    __m128i _r4 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 32));
                    __m128i _r5 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 40));
                    __m128i _r6 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 48));
                    __m128i _r7 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 56));
                    __m128i _r8 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 64));
                    __m128i _r9 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 72));
                    __m128i _ra = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 80));
                    __m128i _rb = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 88));
                    __m128i _rc = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 96));
                    __m128i _rd = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 104));
                    __m128i _re = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 112));
                    __m128i _rf = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 120));
                    __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
                    __m128i _r23 = _mm_unpacklo_epi16(_r2, _r3);
                    __m128i _r45 = _mm_unpacklo_epi16(_r4, _r5);
                    __m128i _r67 = _mm_unpacklo_epi16(_r6, _r7);
                    __m128i _r89 = _mm_unpacklo_epi16(_r8, _r9);
                    __m128i _rab = _mm_unpacklo_epi16(_ra, _rb);
                    __m128i _rcd = _mm_unpacklo_epi16(_rc, _rd);
                    __m128i _ref = _mm_unpacklo_epi16(_re, _rf);
                    _r0 = _mm_unpacklo_epi32(_r01, _r23);
                    _r1 = _mm_unpackhi_epi32(_r01, _r23);
                    _r2 = _mm_unpacklo_epi32(_r45, _r67);
                    _r3 = _mm_unpackhi_epi32(_r45, _r67);
                    _r4 = _mm_unpacklo_epi32(_r89, _rab);
                    _r5 = _mm_unpackhi_epi32(_r89, _rab);
                    _r6 = _mm_unpacklo_epi32(_rcd, _ref);
                    _r7 = _mm_unpackhi_epi32(_rcd, _ref);
                    _r8 = _mm_unpacklo_epi64(_r0, _r2);
                    _r9 = _mm_unpacklo_epi64(_r4, _r6);
                    _ra = _mm_unpackhi_epi64(_r0, _r2);
                    _rb = _mm_unpackhi_epi64(_r4, _r6);
                    _rc = _mm_unpacklo_epi64(_r1, _r3);
                    _rd = _mm_unpacklo_epi64(_r5, _r7);
                    _re = _mm_unpackhi_epi64(_r1, _r3);
                    _rf = _mm_unpackhi_epi64(_r5, _r7);
                    _mm_store_si128((__m128i*)pp, _r8);
                    _mm_store_si128((__m128i*)(pp + 16), _r9);
                    _mm_store_si128((__m128i*)(pp + 32), _ra);
                    _mm_store_si128((__m128i*)(pp + 48), _rb);
                    _mm_store_si128((__m128i*)(pp + 64), _rc);
                    _mm_store_si128((__m128i*)(pp + 80), _rd);
                    _mm_store_si128((__m128i*)(pp + 96), _re);
                    _mm_store_si128((__m128i*)(pp + 112), _rf);
                    pp += 128;
                }
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
                    pp[8] = sptr[stride_w * 8];
                    pp[9] = sptr[stride_w * 9];
                    pp[10] = sptr[stride_w * 10];
                    pp[11] = sptr[stride_w * 11];
                    pp[12] = sptr[stride_w * 12];
                    pp[13] = sptr[stride_w * 13];
                    pp[14] = sptr[stride_w * 14];
                    pp[15] = sptr[stride_w * 15];
                    pp += 16;
                }
            }
        }
        else
        {
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
                    int x08 = stride_w * dx8 + dilation_w * v0;
                    int x09 = stride_w * dx9 + dilation_w * v0;
                    int x0a = stride_w * dxa + dilation_w * v0;
                    int x0b = stride_w * dxb + dilation_w * v0;
                    int x0c = stride_w * dxc + dilation_w * v0;
                    int x0d = stride_w * dxd + dilation_w * v0;
                    int x0e = stride_w * dxe + dilation_w * v0;
                    int x0f = stride_w * dxf + dilation_w * v0;
                    int y00 = stride_h * dy0 + dilation_h * u0;
                    int y01 = stride_h * dy1 + dilation_h * u0;
                    int y02 = stride_h * dy2 + dilation_h * u0;
                    int y03 = stride_h * dy3 + dilation_h * u0;
                    int y04 = stride_h * dy4 + dilation_h * u0;
                    int y05 = stride_h * dy5 + dilation_h * u0;
                    int y06 = stride_h * dy6 + dilation_h * u0;
                    int y07 = stride_h * dy7 + dilation_h * u0;
                    int y08 = stride_h * dy8 + dilation_h * u0;
                    int y09 = stride_h * dy9 + dilation_h * u0;
                    int y0a = stride_h * dya + dilation_h * u0;
                    int y0b = stride_h * dyb + dilation_h * u0;
                    int y0c = stride_h * dyc + dilation_h * u0;
                    int y0d = stride_h * dyd + dilation_h * u0;
                    int y0e = stride_h * dye + dilation_h * u0;
                    int y0f = stride_h * dyf + dilation_h * u0;

                    int x10 = stride_w * dx0 + dilation_w * v1;
                    int x11 = stride_w * dx1 + dilation_w * v1;
                    int x12 = stride_w * dx2 + dilation_w * v1;
                    int x13 = stride_w * dx3 + dilation_w * v1;
                    int x14 = stride_w * dx4 + dilation_w * v1;
                    int x15 = stride_w * dx5 + dilation_w * v1;
                    int x16 = stride_w * dx6 + dilation_w * v1;
                    int x17 = stride_w * dx7 + dilation_w * v1;
                    int x18 = stride_w * dx8 + dilation_w * v1;
                    int x19 = stride_w * dx9 + dilation_w * v1;
                    int x1a = stride_w * dxa + dilation_w * v1;
                    int x1b = stride_w * dxb + dilation_w * v1;
                    int x1c = stride_w * dxc + dilation_w * v1;
                    int x1d = stride_w * dxd + dilation_w * v1;
                    int x1e = stride_w * dxe + dilation_w * v1;
                    int x1f = stride_w * dxf + dilation_w * v1;
                    int y10 = stride_h * dy0 + dilation_h * u1;
                    int y11 = stride_h * dy1 + dilation_h * u1;
                    int y12 = stride_h * dy2 + dilation_h * u1;
                    int y13 = stride_h * dy3 + dilation_h * u1;
                    int y14 = stride_h * dy4 + dilation_h * u1;
                    int y15 = stride_h * dy5 + dilation_h * u1;
                    int y16 = stride_h * dy6 + dilation_h * u1;
                    int y17 = stride_h * dy7 + dilation_h * u1;
                    int y18 = stride_h * dy8 + dilation_h * u1;
                    int y19 = stride_h * dy9 + dilation_h * u1;
                    int y1a = stride_h * dya + dilation_h * u1;
                    int y1b = stride_h * dyb + dilation_h * u1;
                    int y1c = stride_h * dyc + dilation_h * u1;
                    int y1d = stride_h * dyd + dilation_h * u1;
                    int y1e = stride_h * dye + dilation_h * u1;
                    int y1f = stride_h * dyf + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;
                    const signed char* sptr04 = img0.row<const signed char>(y04) + x04;
                    const signed char* sptr05 = img0.row<const signed char>(y05) + x05;
                    const signed char* sptr06 = img0.row<const signed char>(y06) + x06;
                    const signed char* sptr07 = img0.row<const signed char>(y07) + x07;
                    const signed char* sptr08 = img0.row<const signed char>(y08) + x08;
                    const signed char* sptr09 = img0.row<const signed char>(y09) + x09;
                    const signed char* sptr0a = img0.row<const signed char>(y0a) + x0a;
                    const signed char* sptr0b = img0.row<const signed char>(y0b) + x0b;
                    const signed char* sptr0c = img0.row<const signed char>(y0c) + x0c;
                    const signed char* sptr0d = img0.row<const signed char>(y0d) + x0d;
                    const signed char* sptr0e = img0.row<const signed char>(y0e) + x0e;
                    const signed char* sptr0f = img0.row<const signed char>(y0f) + x0f;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;
                    const signed char* sptr14 = img1.row<const signed char>(y14) + x14;
                    const signed char* sptr15 = img1.row<const signed char>(y15) + x15;
                    const signed char* sptr16 = img1.row<const signed char>(y16) + x16;
                    const signed char* sptr17 = img1.row<const signed char>(y17) + x17;
                    const signed char* sptr18 = img1.row<const signed char>(y18) + x18;
                    const signed char* sptr19 = img1.row<const signed char>(y19) + x19;
                    const signed char* sptr1a = img1.row<const signed char>(y1a) + x1a;
                    const signed char* sptr1b = img1.row<const signed char>(y1b) + x1b;
                    const signed char* sptr1c = img1.row<const signed char>(y1c) + x1c;
                    const signed char* sptr1d = img1.row<const signed char>(y1d) + x1d;
                    const signed char* sptr1e = img1.row<const signed char>(y1e) + x1e;
                    const signed char* sptr1f = img1.row<const signed char>(y1f) + x1f;

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
                    pp[16 + 0] = sptr08[0];
                    pp[16 + 1] = sptr18[0];
                    pp[16 + 2] = sptr09[0];
                    pp[16 + 3] = sptr19[0];
                    pp[16 + 4] = sptr0a[0];
                    pp[16 + 5] = sptr1a[0];
                    pp[16 + 6] = sptr0b[0];
                    pp[16 + 7] = sptr1b[0];
                    pp[16 + 8] = sptr0c[0];
                    pp[16 + 9] = sptr1c[0];
                    pp[16 + 10] = sptr0d[0];
                    pp[16 + 11] = sptr1d[0];
                    pp[16 + 12] = sptr0e[0];
                    pp[16 + 13] = sptr1e[0];
                    pp[16 + 14] = sptr0f[0];
                    pp[16 + 15] = sptr1f[0];
                    pp += 32;
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
                int x8 = stride_w * dx8 + dilation_w * v;
                int x9 = stride_w * dx9 + dilation_w * v;
                int xa = stride_w * dxa + dilation_w * v;
                int xb = stride_w * dxb + dilation_w * v;
                int xc = stride_w * dxc + dilation_w * v;
                int xd = stride_w * dxd + dilation_w * v;
                int xe = stride_w * dxe + dilation_w * v;
                int xf = stride_w * dxf + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;
                int y4 = stride_h * dy4 + dilation_h * u;
                int y5 = stride_h * dy5 + dilation_h * u;
                int y6 = stride_h * dy6 + dilation_h * u;
                int y7 = stride_h * dy7 + dilation_h * u;
                int y8 = stride_h * dy8 + dilation_h * u;
                int y9 = stride_h * dy9 + dilation_h * u;
                int ya = stride_h * dya + dilation_h * u;
                int yb = stride_h * dyb + dilation_h * u;
                int yc = stride_h * dyc + dilation_h * u;
                int yd = stride_h * dyd + dilation_h * u;
                int ye = stride_h * dye + dilation_h * u;
                int yf = stride_h * dyf + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;
                const signed char* sptr4 = img.row<const signed char>(y4) + x4 * elempack;
                const signed char* sptr5 = img.row<const signed char>(y5) + x5 * elempack;
                const signed char* sptr6 = img.row<const signed char>(y6) + x6 * elempack;
                const signed char* sptr7 = img.row<const signed char>(y7) + x7 * elempack;
                const signed char* sptr8 = img.row<const signed char>(y8) + x8 * elempack;
                const signed char* sptr9 = img.row<const signed char>(y9) + x9 * elempack;
                const signed char* sptra = img.row<const signed char>(ya) + xa * elempack;
                const signed char* sptrb = img.row<const signed char>(yb) + xb * elempack;
                const signed char* sptrc = img.row<const signed char>(yc) + xc * elempack;
                const signed char* sptrd = img.row<const signed char>(yd) + xd * elempack;
                const signed char* sptre = img.row<const signed char>(ye) + xe * elempack;
                const signed char* sptrf = img.row<const signed char>(yf) + xf * elempack;

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
                    __m128i _r8 = _mm_loadl_epi64((const __m128i*)sptr8);
                    __m128i _r9 = _mm_loadl_epi64((const __m128i*)sptr9);
                    __m128i _ra = _mm_loadl_epi64((const __m128i*)sptra);
                    __m128i _rb = _mm_loadl_epi64((const __m128i*)sptrb);
                    __m128i _rc = _mm_loadl_epi64((const __m128i*)sptrc);
                    __m128i _rd = _mm_loadl_epi64((const __m128i*)sptrd);
                    __m128i _re = _mm_loadl_epi64((const __m128i*)sptre);
                    __m128i _rf = _mm_loadl_epi64((const __m128i*)sptrf);
                    __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
                    __m128i _r23 = _mm_unpacklo_epi16(_r2, _r3);
                    __m128i _r45 = _mm_unpacklo_epi16(_r4, _r5);
                    __m128i _r67 = _mm_unpacklo_epi16(_r6, _r7);
                    __m128i _r89 = _mm_unpacklo_epi16(_r8, _r9);
                    __m128i _rab = _mm_unpacklo_epi16(_ra, _rb);
                    __m128i _rcd = _mm_unpacklo_epi16(_rc, _rd);
                    __m128i _ref = _mm_unpacklo_epi16(_re, _rf);
                    _r0 = _mm_unpacklo_epi32(_r01, _r23);
                    _r1 = _mm_unpackhi_epi32(_r01, _r23);
                    _r2 = _mm_unpacklo_epi32(_r45, _r67);
                    _r3 = _mm_unpackhi_epi32(_r45, _r67);
                    _r4 = _mm_unpacklo_epi32(_r89, _rab);
                    _r5 = _mm_unpackhi_epi32(_r89, _rab);
                    _r6 = _mm_unpacklo_epi32(_rcd, _ref);
                    _r7 = _mm_unpackhi_epi32(_rcd, _ref);
                    _r8 = _mm_unpacklo_epi64(_r0, _r2);
                    _r9 = _mm_unpacklo_epi64(_r4, _r6);
                    _ra = _mm_unpackhi_epi64(_r0, _r2);
                    _rb = _mm_unpackhi_epi64(_r4, _r6);
                    _rc = _mm_unpacklo_epi64(_r1, _r3);
                    _rd = _mm_unpacklo_epi64(_r5, _r7);
                    _re = _mm_unpackhi_epi64(_r1, _r3);
                    _rf = _mm_unpackhi_epi64(_r5, _r7);
                    _mm_store_si128((__m128i*)pp, _r8);
                    _mm_store_si128((__m128i*)(pp + 16), _r9);
                    _mm_store_si128((__m128i*)(pp + 32), _ra);
                    _mm_store_si128((__m128i*)(pp + 48), _rb);
                    _mm_store_si128((__m128i*)(pp + 64), _rc);
                    _mm_store_si128((__m128i*)(pp + 80), _rd);
                    _mm_store_si128((__m128i*)(pp + 96), _re);
                    _mm_store_si128((__m128i*)(pp + 112), _rf);
                    pp += 128;
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
                    pp[8] = sptr8[0];
                    pp[9] = sptr9[0];
                    pp[10] = sptra[0];
                    pp[11] = sptrb[0];
                    pp[12] = sptrc[0];
                    pp[13] = sptrd[0];
                    pp[14] = sptre[0];
                    pp[15] = sptrf[0];
                    pp += 16;
                }
            }
        }
    }
#endif // __AVX512F__
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
                    int y00 = stride_h * dy0 + dilation_h * u0;
                    int x10 = stride_w * dx0 + dilation_w * v1;
                    int y10 = stride_h * dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr0);
                        __m128i _r1 = _mm_loadl_epi64((const __m128i*)sptr1);
                        __m128i _r01 = _mm_unpacklo_epi8(_r0, _r1);
                        _mm_storeu_si128((__m128i*)pp, _r01);
                        pp += 16;
                    }
                    else if (stride_w == 2)
                    {
                        __m128i _r0 = _mm_loadu_si128((const __m128i*)sptr0);
                        __m128i _r1 = _mm_loadu_si128((const __m128i*)sptr1);
                        __m128i _tmp0 = _mm_unpacklo_epi8(_r0, _r1);
                        __m128i _tmp1 = _mm_unpackhi_epi8(_r0, _r1);
                        _tmp0 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_tmp0, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
                        _tmp1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_tmp1, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
                        _tmp0 = _mm_shuffle_epi32(_tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                        _tmp1 = _mm_shuffle_epi32(_tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                        __m128i _r01 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_tmp0), _mm_castsi128_ps(_tmp1), _MM_SHUFFLE(1, 0, 1, 0)));
                        _mm_storeu_si128((__m128i*)pp, _r01);
                        pp += 16;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp[8] = sptr0[stride_w * 4];
                        pp[9] = sptr1[stride_w * 4];
                        pp[10] = sptr0[stride_w * 5];
                        pp[11] = sptr1[stride_w * 5];
                        pp[12] = sptr0[stride_w * 6];
                        pp[13] = sptr1[stride_w * 6];
                        pp[14] = sptr0[stride_w * 7];
                        pp[15] = sptr1[stride_w * 7];
                        pp += 16;
                    }
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

                if (elempack == 8)
                {
                    __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr);
                    __m128i _r1 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 8));
                    __m128i _r2 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 16));
                    __m128i _r3 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 24));
                    __m128i _r4 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 32));
                    __m128i _r5 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 40));
                    __m128i _r6 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 48));
                    __m128i _r7 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 56));
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

        if (dy0 == dy3)
        {
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
                    int y00 = stride_h * dy0 + dilation_h * u0;
                    int x10 = stride_w * dx0 + dilation_w * v1;
                    int y10 = stride_h * dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr0);
                        __m128i _r1 = _mm_loadl_epi64((const __m128i*)sptr1);
                        __m128i _r01 = _mm_unpacklo_epi8(_r0, _r1);
                        _mm_storel_epi64((__m128i*)pp, _r01);
                        pp += 8;
                    }
                    else if (stride_w == 2)
                    {
                        __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr0);
                        __m128i _r1 = _mm_loadl_epi64((const __m128i*)sptr1);
                        __m128i _r01 = _mm_unpacklo_epi8(_r0, _r1);
                        _r01 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_r01, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
                        _r01 = _mm_shuffle_epi32(_r01, _MM_SHUFFLE(3, 1, 2, 0));
                        _mm_storel_epi64((__m128i*)pp, _r01);
                        pp += 8;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp += 8;
                    }
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

                if (elempack == 8)
                {
                    __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr);
                    __m128i _r1 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 8));
                    __m128i _r2 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 16));
                    __m128i _r3 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 24));
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
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;

        if (dy0 == dy1)
        {
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
                    int y00 = stride_h * dy0 + dilation_h * u0;
                    int x10 = stride_w * dx0 + dilation_w * v1;
                    int y10 = stride_h * dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr0[stride_w];
                    pp[3] = sptr1[stride_w];
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

#if __SSE2__
                if (elempack == 8)
                {
                    __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr);
                    __m128i _r1 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 8));
                    __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
                    _mm_storeu_si128((__m128i*)pp, _r01);
                    pp += 16;
                }
#endif // __SSE2__
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

#if __AVX512F__
template void convolution_im2col_input_tile_int8_avx512<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_avx512<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_avx512<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_avx512<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_avx512<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_avx512<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
#else  // __AVX512F__
template void convolution_im2col_input_tile_int8<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
#endif // __AVX512F__

static void convolution_im2col_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_int8(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 1 && kernel_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __AVX512F__
        convolution_im2col_input_tile_int8_avx512<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else  // __AVX512F__
        convolution_im2col_input_tile_int8<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif // __AVX512F__
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
#if __AVX512F__
        convolution_im2col_input_tile_int8_avx512<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#else  // __AVX512F__
        convolution_im2col_input_tile_int8<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#endif // __AVX512F__
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __AVX512F__
        convolution_im2col_input_tile_int8_avx512<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else  // __AVX512F__
        convolution_im2col_input_tile_int8<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif // __AVX512F__
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
#if __AVX512F__
        convolution_im2col_input_tile_int8_avx512<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#else  // __AVX512F__
        convolution_im2col_input_tile_int8<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#endif // __AVX512F__
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __AVX512F__
        convolution_im2col_input_tile_int8_avx512<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else  // __AVX512F__
        convolution_im2col_input_tile_int8<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif // __AVX512F__
        return;
    }

    if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __AVX512F__
        convolution_im2col_input_tile_int8_avx512<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else  // __AVX512F__
        convolution_im2col_input_tile_int8<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif // __AVX512F__
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
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dy8 = (j + jj + 8) / outw;
        int dy9 = (j + jj + 9) / outw;
        int dya = (j + jj + 10) / outw;
        int dyb = (j + jj + 11) / outw;
        int dyc = (j + jj + 12) / outw;
        int dyd = (j + jj + 13) / outw;
        int dye = (j + jj + 14) / outw;
        int dyf = (j + jj + 15) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;
        int dx8 = (j + jj + 8) % outw;
        int dx9 = (j + jj + 9) % outw;
        int dxa = (j + jj + 10) % outw;
        int dxb = (j + jj + 11) % outw;
        int dxc = (j + jj + 12) % outw;
        int dxd = (j + jj + 13) % outw;
        int dxe = (j + jj + 14) % outw;
        int dxf = (j + jj + 15) % outw;

        if (dy0 == dyf)
        {
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
                    int y00 = stride_h * dy0 + dilation_h * u0;
                    int x10 = stride_w * dx0 + dilation_w * v1;
                    int y10 = stride_h * dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        __m128i _r0 = _mm_loadu_si128((const __m128i*)sptr0);
                        __m128i _r1 = _mm_loadu_si128((const __m128i*)sptr1);
                        __m128i _tmp0 = _mm_unpacklo_epi8(_r0, _r1);
                        __m128i _tmp1 = _mm_unpackhi_epi8(_r0, _r1);
                        _mm_store_si128((__m128i*)pp, _tmp0);
                        _mm_store_si128((__m128i*)(pp + 16), _tmp1);
                        pp += 32;
                    }
                    else if (stride_w == 2)
                    {
                        __m256i _r0 = _mm256_loadu_si256((const __m256i*)sptr0);
                        __m256i _r1 = _mm256_loadu_si256((const __m256i*)sptr1);
                        __m256i _tmp0 = _mm256_unpacklo_epi8(_r0, _r1);
                        __m256i _tmp1 = _mm256_unpackhi_epi8(_r0, _r1);
                        _tmp0 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_tmp0, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
                        _tmp1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_tmp1, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
                        _tmp0 = _mm256_shuffle_epi32(_tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                        _tmp1 = _mm256_shuffle_epi32(_tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                        __m256i _r01 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                        _mm256_storeu_si256((__m256i*)pp, _r01);
                        pp += 32;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp[8] = sptr0[stride_w * 4];
                        pp[9] = sptr1[stride_w * 4];
                        pp[10] = sptr0[stride_w * 5];
                        pp[11] = sptr1[stride_w * 5];
                        pp[12] = sptr0[stride_w * 6];
                        pp[13] = sptr1[stride_w * 6];
                        pp[14] = sptr0[stride_w * 7];
                        pp[15] = sptr1[stride_w * 7];
                        pp[16 + 0] = sptr0[stride_w * 8];
                        pp[16 + 1] = sptr1[stride_w * 8];
                        pp[16 + 2] = sptr0[stride_w * 9];
                        pp[16 + 3] = sptr1[stride_w * 9];
                        pp[16 + 4] = sptr0[stride_w * 10];
                        pp[16 + 5] = sptr1[stride_w * 10];
                        pp[16 + 6] = sptr0[stride_w * 11];
                        pp[16 + 7] = sptr1[stride_w * 11];
                        pp[16 + 8] = sptr0[stride_w * 12];
                        pp[16 + 9] = sptr1[stride_w * 12];
                        pp[16 + 10] = sptr0[stride_w * 13];
                        pp[16 + 11] = sptr1[stride_w * 13];
                        pp[16 + 12] = sptr0[stride_w * 14];
                        pp[16 + 13] = sptr1[stride_w * 14];
                        pp[16 + 14] = sptr0[stride_w * 15];
                        pp[16 + 15] = sptr1[stride_w * 15];
                        pp += 32;
                    }
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

                if (elempack == 8)
                {
                    __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr);
                    __m128i _r1 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 8));
                    __m128i _r2 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 16));
                    __m128i _r3 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 24));
                    __m128i _r4 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 32));
                    __m128i _r5 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 40));
                    __m128i _r6 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 48));
                    __m128i _r7 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 56));
                    __m128i _r8 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 64));
                    __m128i _r9 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 72));
                    __m128i _ra = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 80));
                    __m128i _rb = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 88));
                    __m128i _rc = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 96));
                    __m128i _rd = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 104));
                    __m128i _re = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 112));
                    __m128i _rf = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 120));
                    __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
                    __m128i _r23 = _mm_unpacklo_epi16(_r2, _r3);
                    __m128i _r45 = _mm_unpacklo_epi16(_r4, _r5);
                    __m128i _r67 = _mm_unpacklo_epi16(_r6, _r7);
                    __m128i _r89 = _mm_unpacklo_epi16(_r8, _r9);
                    __m128i _rab = _mm_unpacklo_epi16(_ra, _rb);
                    __m128i _rcd = _mm_unpacklo_epi16(_rc, _rd);
                    __m128i _ref = _mm_unpacklo_epi16(_re, _rf);
                    _r0 = _mm_unpacklo_epi32(_r01, _r23);
                    _r1 = _mm_unpackhi_epi32(_r01, _r23);
                    _r2 = _mm_unpacklo_epi32(_r45, _r67);
                    _r3 = _mm_unpackhi_epi32(_r45, _r67);
                    _r4 = _mm_unpacklo_epi32(_r89, _rab);
                    _r5 = _mm_unpackhi_epi32(_r89, _rab);
                    _r6 = _mm_unpacklo_epi32(_rcd, _ref);
                    _r7 = _mm_unpackhi_epi32(_rcd, _ref);
                    _r8 = _mm_unpacklo_epi64(_r0, _r2);
                    _r9 = _mm_unpacklo_epi64(_r4, _r6);
                    _ra = _mm_unpackhi_epi64(_r0, _r2);
                    _rb = _mm_unpackhi_epi64(_r4, _r6);
                    _rc = _mm_unpacklo_epi64(_r1, _r3);
                    _rd = _mm_unpacklo_epi64(_r5, _r7);
                    _re = _mm_unpackhi_epi64(_r1, _r3);
                    _rf = _mm_unpackhi_epi64(_r5, _r7);
                    _mm_store_si128((__m128i*)pp, _r8);
                    _mm_store_si128((__m128i*)(pp + 16), _r9);
                    _mm_store_si128((__m128i*)(pp + 32), _ra);
                    _mm_store_si128((__m128i*)(pp + 48), _rb);
                    _mm_store_si128((__m128i*)(pp + 64), _rc);
                    _mm_store_si128((__m128i*)(pp + 80), _rd);
                    _mm_store_si128((__m128i*)(pp + 96), _re);
                    _mm_store_si128((__m128i*)(pp + 112), _rf);
                    pp += 128;
                }
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
                    pp[8] = sptr[stride_w * 8];
                    pp[9] = sptr[stride_w * 9];
                    pp[10] = sptr[stride_w * 10];
                    pp[11] = sptr[stride_w * 11];
                    pp[12] = sptr[stride_w * 12];
                    pp[13] = sptr[stride_w * 13];
                    pp[14] = sptr[stride_w * 14];
                    pp[15] = sptr[stride_w * 15];
                    pp += 16;
                }
            }
        }
        else
        {
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
                    int x08 = stride_w * dx8 + dilation_w * v0;
                    int x09 = stride_w * dx9 + dilation_w * v0;
                    int x0a = stride_w * dxa + dilation_w * v0;
                    int x0b = stride_w * dxb + dilation_w * v0;
                    int x0c = stride_w * dxc + dilation_w * v0;
                    int x0d = stride_w * dxd + dilation_w * v0;
                    int x0e = stride_w * dxe + dilation_w * v0;
                    int x0f = stride_w * dxf + dilation_w * v0;
                    int y00 = stride_h * dy0 + dilation_h * u0;
                    int y01 = stride_h * dy1 + dilation_h * u0;
                    int y02 = stride_h * dy2 + dilation_h * u0;
                    int y03 = stride_h * dy3 + dilation_h * u0;
                    int y04 = stride_h * dy4 + dilation_h * u0;
                    int y05 = stride_h * dy5 + dilation_h * u0;
                    int y06 = stride_h * dy6 + dilation_h * u0;
                    int y07 = stride_h * dy7 + dilation_h * u0;
                    int y08 = stride_h * dy8 + dilation_h * u0;
                    int y09 = stride_h * dy9 + dilation_h * u0;
                    int y0a = stride_h * dya + dilation_h * u0;
                    int y0b = stride_h * dyb + dilation_h * u0;
                    int y0c = stride_h * dyc + dilation_h * u0;
                    int y0d = stride_h * dyd + dilation_h * u0;
                    int y0e = stride_h * dye + dilation_h * u0;
                    int y0f = stride_h * dyf + dilation_h * u0;

                    int x10 = stride_w * dx0 + dilation_w * v1;
                    int x11 = stride_w * dx1 + dilation_w * v1;
                    int x12 = stride_w * dx2 + dilation_w * v1;
                    int x13 = stride_w * dx3 + dilation_w * v1;
                    int x14 = stride_w * dx4 + dilation_w * v1;
                    int x15 = stride_w * dx5 + dilation_w * v1;
                    int x16 = stride_w * dx6 + dilation_w * v1;
                    int x17 = stride_w * dx7 + dilation_w * v1;
                    int x18 = stride_w * dx8 + dilation_w * v1;
                    int x19 = stride_w * dx9 + dilation_w * v1;
                    int x1a = stride_w * dxa + dilation_w * v1;
                    int x1b = stride_w * dxb + dilation_w * v1;
                    int x1c = stride_w * dxc + dilation_w * v1;
                    int x1d = stride_w * dxd + dilation_w * v1;
                    int x1e = stride_w * dxe + dilation_w * v1;
                    int x1f = stride_w * dxf + dilation_w * v1;
                    int y10 = stride_h * dy0 + dilation_h * u1;
                    int y11 = stride_h * dy1 + dilation_h * u1;
                    int y12 = stride_h * dy2 + dilation_h * u1;
                    int y13 = stride_h * dy3 + dilation_h * u1;
                    int y14 = stride_h * dy4 + dilation_h * u1;
                    int y15 = stride_h * dy5 + dilation_h * u1;
                    int y16 = stride_h * dy6 + dilation_h * u1;
                    int y17 = stride_h * dy7 + dilation_h * u1;
                    int y18 = stride_h * dy8 + dilation_h * u1;
                    int y19 = stride_h * dy9 + dilation_h * u1;
                    int y1a = stride_h * dya + dilation_h * u1;
                    int y1b = stride_h * dyb + dilation_h * u1;
                    int y1c = stride_h * dyc + dilation_h * u1;
                    int y1d = stride_h * dyd + dilation_h * u1;
                    int y1e = stride_h * dye + dilation_h * u1;
                    int y1f = stride_h * dyf + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;
                    const signed char* sptr04 = img0.row<const signed char>(y04) + x04;
                    const signed char* sptr05 = img0.row<const signed char>(y05) + x05;
                    const signed char* sptr06 = img0.row<const signed char>(y06) + x06;
                    const signed char* sptr07 = img0.row<const signed char>(y07) + x07;
                    const signed char* sptr08 = img0.row<const signed char>(y08) + x08;
                    const signed char* sptr09 = img0.row<const signed char>(y09) + x09;
                    const signed char* sptr0a = img0.row<const signed char>(y0a) + x0a;
                    const signed char* sptr0b = img0.row<const signed char>(y0b) + x0b;
                    const signed char* sptr0c = img0.row<const signed char>(y0c) + x0c;
                    const signed char* sptr0d = img0.row<const signed char>(y0d) + x0d;
                    const signed char* sptr0e = img0.row<const signed char>(y0e) + x0e;
                    const signed char* sptr0f = img0.row<const signed char>(y0f) + x0f;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;
                    const signed char* sptr14 = img1.row<const signed char>(y14) + x14;
                    const signed char* sptr15 = img1.row<const signed char>(y15) + x15;
                    const signed char* sptr16 = img1.row<const signed char>(y16) + x16;
                    const signed char* sptr17 = img1.row<const signed char>(y17) + x17;
                    const signed char* sptr18 = img1.row<const signed char>(y18) + x18;
                    const signed char* sptr19 = img1.row<const signed char>(y19) + x19;
                    const signed char* sptr1a = img1.row<const signed char>(y1a) + x1a;
                    const signed char* sptr1b = img1.row<const signed char>(y1b) + x1b;
                    const signed char* sptr1c = img1.row<const signed char>(y1c) + x1c;
                    const signed char* sptr1d = img1.row<const signed char>(y1d) + x1d;
                    const signed char* sptr1e = img1.row<const signed char>(y1e) + x1e;
                    const signed char* sptr1f = img1.row<const signed char>(y1f) + x1f;

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
                    pp[16 + 0] = sptr08[0];
                    pp[16 + 1] = sptr18[0];
                    pp[16 + 2] = sptr09[0];
                    pp[16 + 3] = sptr19[0];
                    pp[16 + 4] = sptr0a[0];
                    pp[16 + 5] = sptr1a[0];
                    pp[16 + 6] = sptr0b[0];
                    pp[16 + 7] = sptr1b[0];
                    pp[16 + 8] = sptr0c[0];
                    pp[16 + 9] = sptr1c[0];
                    pp[16 + 10] = sptr0d[0];
                    pp[16 + 11] = sptr1d[0];
                    pp[16 + 12] = sptr0e[0];
                    pp[16 + 13] = sptr1e[0];
                    pp[16 + 14] = sptr0f[0];
                    pp[16 + 15] = sptr1f[0];
                    pp += 32;
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
                int x8 = stride_w * dx8 + dilation_w * v;
                int x9 = stride_w * dx9 + dilation_w * v;
                int xa = stride_w * dxa + dilation_w * v;
                int xb = stride_w * dxb + dilation_w * v;
                int xc = stride_w * dxc + dilation_w * v;
                int xd = stride_w * dxd + dilation_w * v;
                int xe = stride_w * dxe + dilation_w * v;
                int xf = stride_w * dxf + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;
                int y4 = stride_h * dy4 + dilation_h * u;
                int y5 = stride_h * dy5 + dilation_h * u;
                int y6 = stride_h * dy6 + dilation_h * u;
                int y7 = stride_h * dy7 + dilation_h * u;
                int y8 = stride_h * dy8 + dilation_h * u;
                int y9 = stride_h * dy9 + dilation_h * u;
                int ya = stride_h * dya + dilation_h * u;
                int yb = stride_h * dyb + dilation_h * u;
                int yc = stride_h * dyc + dilation_h * u;
                int yd = stride_h * dyd + dilation_h * u;
                int ye = stride_h * dye + dilation_h * u;
                int yf = stride_h * dyf + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;
                const signed char* sptr4 = img.row<const signed char>(y4) + x4 * elempack;
                const signed char* sptr5 = img.row<const signed char>(y5) + x5 * elempack;
                const signed char* sptr6 = img.row<const signed char>(y6) + x6 * elempack;
                const signed char* sptr7 = img.row<const signed char>(y7) + x7 * elempack;
                const signed char* sptr8 = img.row<const signed char>(y8) + x8 * elempack;
                const signed char* sptr9 = img.row<const signed char>(y9) + x9 * elempack;
                const signed char* sptra = img.row<const signed char>(ya) + xa * elempack;
                const signed char* sptrb = img.row<const signed char>(yb) + xb * elempack;
                const signed char* sptrc = img.row<const signed char>(yc) + xc * elempack;
                const signed char* sptrd = img.row<const signed char>(yd) + xd * elempack;
                const signed char* sptre = img.row<const signed char>(ye) + xe * elempack;
                const signed char* sptrf = img.row<const signed char>(yf) + xf * elempack;

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
                    __m128i _r8 = _mm_loadl_epi64((const __m128i*)sptr8);
                    __m128i _r9 = _mm_loadl_epi64((const __m128i*)sptr9);
                    __m128i _ra = _mm_loadl_epi64((const __m128i*)sptra);
                    __m128i _rb = _mm_loadl_epi64((const __m128i*)sptrb);
                    __m128i _rc = _mm_loadl_epi64((const __m128i*)sptrc);
                    __m128i _rd = _mm_loadl_epi64((const __m128i*)sptrd);
                    __m128i _re = _mm_loadl_epi64((const __m128i*)sptre);
                    __m128i _rf = _mm_loadl_epi64((const __m128i*)sptrf);
                    __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
                    __m128i _r23 = _mm_unpacklo_epi16(_r2, _r3);
                    __m128i _r45 = _mm_unpacklo_epi16(_r4, _r5);
                    __m128i _r67 = _mm_unpacklo_epi16(_r6, _r7);
                    __m128i _r89 = _mm_unpacklo_epi16(_r8, _r9);
                    __m128i _rab = _mm_unpacklo_epi16(_ra, _rb);
                    __m128i _rcd = _mm_unpacklo_epi16(_rc, _rd);
                    __m128i _ref = _mm_unpacklo_epi16(_re, _rf);
                    _r0 = _mm_unpacklo_epi32(_r01, _r23);
                    _r1 = _mm_unpackhi_epi32(_r01, _r23);
                    _r2 = _mm_unpacklo_epi32(_r45, _r67);
                    _r3 = _mm_unpackhi_epi32(_r45, _r67);
                    _r4 = _mm_unpacklo_epi32(_r89, _rab);
                    _r5 = _mm_unpackhi_epi32(_r89, _rab);
                    _r6 = _mm_unpacklo_epi32(_rcd, _ref);
                    _r7 = _mm_unpackhi_epi32(_rcd, _ref);
                    _r8 = _mm_unpacklo_epi64(_r0, _r2);
                    _r9 = _mm_unpacklo_epi64(_r4, _r6);
                    _ra = _mm_unpackhi_epi64(_r0, _r2);
                    _rb = _mm_unpackhi_epi64(_r4, _r6);
                    _rc = _mm_unpacklo_epi64(_r1, _r3);
                    _rd = _mm_unpacklo_epi64(_r5, _r7);
                    _re = _mm_unpackhi_epi64(_r1, _r3);
                    _rf = _mm_unpackhi_epi64(_r5, _r7);
                    _mm_store_si128((__m128i*)pp, _r8);
                    _mm_store_si128((__m128i*)(pp + 16), _r9);
                    _mm_store_si128((__m128i*)(pp + 32), _ra);
                    _mm_store_si128((__m128i*)(pp + 48), _rb);
                    _mm_store_si128((__m128i*)(pp + 64), _rc);
                    _mm_store_si128((__m128i*)(pp + 80), _rd);
                    _mm_store_si128((__m128i*)(pp + 96), _re);
                    _mm_store_si128((__m128i*)(pp + 112), _rf);
                    pp += 128;
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
                    pp[8] = sptr8[0];
                    pp[9] = sptr9[0];
                    pp[10] = sptra[0];
                    pp[11] = sptrb[0];
                    pp[12] = sptrc[0];
                    pp[13] = sptrd[0];
                    pp[14] = sptre[0];
                    pp[15] = sptrf[0];
                    pp += 16;
                }
            }
        }
    }
#endif // __AVX512F__
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
                    int y00 = stride_h * dy0 + dilation_h * u0;
                    int x10 = stride_w * dx0 + dilation_w * v1;
                    int y10 = stride_h * dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr0);
                        __m128i _r1 = _mm_loadl_epi64((const __m128i*)sptr1);
                        __m128i _r01 = _mm_unpacklo_epi8(_r0, _r1);
                        _mm_storeu_si128((__m128i*)pp, _r01);
                        pp += 16;
                    }
                    else if (stride_w == 2)
                    {
                        __m128i _r0 = _mm_loadu_si128((const __m128i*)sptr0);
                        __m128i _r1 = _mm_loadu_si128((const __m128i*)sptr1);
                        __m128i _tmp0 = _mm_unpacklo_epi8(_r0, _r1);
                        __m128i _tmp1 = _mm_unpackhi_epi8(_r0, _r1);
                        _tmp0 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_tmp0, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
                        _tmp1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_tmp1, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
                        _tmp0 = _mm_shuffle_epi32(_tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                        _tmp1 = _mm_shuffle_epi32(_tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                        __m128i _r01 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_tmp0), _mm_castsi128_ps(_tmp1), _MM_SHUFFLE(1, 0, 1, 0)));
                        _mm_storeu_si128((__m128i*)pp, _r01);
                        pp += 16;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp[8] = sptr0[stride_w * 4];
                        pp[9] = sptr1[stride_w * 4];
                        pp[10] = sptr0[stride_w * 5];
                        pp[11] = sptr1[stride_w * 5];
                        pp[12] = sptr0[stride_w * 6];
                        pp[13] = sptr1[stride_w * 6];
                        pp[14] = sptr0[stride_w * 7];
                        pp[15] = sptr1[stride_w * 7];
                        pp += 16;
                    }
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

                if (elempack == 8)
                {
                    __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr);
                    __m128i _r1 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 8));
                    __m128i _r2 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 16));
                    __m128i _r3 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 24));
                    __m128i _r4 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 32));
                    __m128i _r5 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 40));
                    __m128i _r6 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 48));
                    __m128i _r7 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 56));
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

        if (dy0 == dy3)
        {
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
                    int y00 = stride_h * dy0 + dilation_h * u0;
                    int x10 = stride_w * dx0 + dilation_w * v1;
                    int y10 = stride_h * dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr0);
                        __m128i _r1 = _mm_loadl_epi64((const __m128i*)sptr1);
                        __m128i _r01 = _mm_unpacklo_epi8(_r0, _r1);
                        _mm_storel_epi64((__m128i*)pp, _r01);
                        pp += 8;
                    }
                    else if (stride_w == 2)
                    {
                        __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr0);
                        __m128i _r1 = _mm_loadl_epi64((const __m128i*)sptr1);
                        __m128i _r01 = _mm_unpacklo_epi8(_r0, _r1);
                        _r01 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_r01, _MM_SHUFFLE(3, 1, 2, 0)), _MM_SHUFFLE(3, 1, 2, 0));
                        _r01 = _mm_shuffle_epi32(_r01, _MM_SHUFFLE(3, 1, 2, 0));
                        _mm_storel_epi64((__m128i*)pp, _r01);
                        pp += 8;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp += 8;
                    }
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

                if (elempack == 8)
                {
                    __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr);
                    __m128i _r1 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 8));
                    __m128i _r2 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 16));
                    __m128i _r3 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 24));
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
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;

        if (dy0 == dy1)
        {
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
                    int y00 = stride_h * dy0 + dilation_h * u0;
                    int x10 = stride_w * dx0 + dilation_w * v1;
                    int y10 = stride_h * dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr0[stride_w];
                    pp[3] = sptr1[stride_w];
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

#if __SSE2__
                if (elempack == 8)
                {
                    __m128i _r0 = _mm_loadl_epi64((const __m128i*)sptr);
                    __m128i _r1 = _mm_loadl_epi64((const __m128i*)(sptr + stride_w * 8));
                    __m128i _r01 = _mm_unpacklo_epi16(_r0, _r1);
                    _mm_storeu_si128((__m128i*)pp, _r01);
                    pp += 16;
                }
#endif // __SSE2__
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
