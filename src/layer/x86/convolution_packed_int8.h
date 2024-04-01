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

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
void convolution_transform_kernel_packed_int8_avx2(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);
#endif

#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
void convolution_packed_int8_avx512vnni(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__
void convolution_packed_int8_avxvnni(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
void convolution_packed_int8_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__
void convolution_packed_int8_xop(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt);
#endif
#endif

static void convolution_transform_kernel_packed_int8(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        convolution_transform_kernel_packed_int8_avx2(kernel, kernel_tm, inch, outch, kernel_w, kernel_h);
        return;
    }
#endif

    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb

    // clang-format off
    // *INDENT-OFF*
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    if (outch >= 16)
    {
        if (inch >= 16)
            kernel_tm.create(maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 2 + inch % 2, outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)256u, 256);
        else if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)128u, 128);
        else if (inch >= 2)
            kernel_tm.create(maxk, inch / 2 + inch % 2, outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)32u, 32);
        else
            kernel_tm.create(maxk, inch, outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)16u, 16);
    }
    else
#endif // __AVX512F__
    if (outch >= 8)
    {
#if __AVX512F__
        if (inch >= 16)
            kernel_tm.create(maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)128u, 128);
        else
#endif // __AVX512F__
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)64u, 64);
        else if (inch >= 2)
            kernel_tm.create(maxk, inch / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)16u, 16);
        else
            kernel_tm.create(maxk, inch, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)8u, 8);
    }
    else
#endif // __AVX2__
    if (outch >= 4)
    {
#if __AVX512F__
        if (inch >= 16)
            kernel_tm.create(maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)64u, 64);
        else
#endif // __AVX512F__
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)32u, 32);
        else if (inch >= 2)
            kernel_tm.create(maxk, inch / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)8u, 8);
        else
            kernel_tm.create(maxk, inch, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)4u, 4);
    }
    else
#endif // __SSE2__
    if (outch >= 2)
    {
#if __SSE2__
#if __AVX512F__
        if (inch >= 16)
            kernel_tm.create(maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 2 + inch % 2, outch / 2 + outch % 2, (size_t)32u, 32);
        else
#endif // __AVX512F__
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch / 2 + outch % 2, (size_t)16u, 16);
        else
#endif // __SSE2__
        if (inch >= 2)
            kernel_tm.create(maxk, inch / 2 + inch % 2, outch / 2 + outch % 2, (size_t)4u, 4);
        else
            kernel_tm.create(maxk, inch, outch / 2 + outch % 2, (size_t)2u, 2);
    }
    else
    {
#if __SSE2__
#if __AVX512F__
        if (inch >= 16)
            kernel_tm.create(maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 2 + inch % 2, outch, (size_t)16u, 16);
        else
#endif // __AVX512F__
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch, (size_t)8u, 8);
        else
#endif // __SSE2__
        if (inch >= 2)
            kernel_tm.create(maxk, inch / 2 + inch % 2, outch, (size_t)2u, 2);
        else
            kernel_tm.create(maxk, inch, outch, (size_t)1u, 1);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; q + 15 < outch; q += 16)
    {
        const signed char* kptr0 = (const signed char*)kernel + q * inch * maxk;
        const signed char* kptr1 = (const signed char*)kernel + (q + 1) * inch * maxk;
        const signed char* kptr2 = (const signed char*)kernel + (q + 2) * inch * maxk;
        const signed char* kptr3 = (const signed char*)kernel + (q + 3) * inch * maxk;
        const signed char* kptr4 = (const signed char*)kernel + (q + 4) * inch * maxk;
        const signed char* kptr5 = (const signed char*)kernel + (q + 5) * inch * maxk;
        const signed char* kptr6 = (const signed char*)kernel + (q + 6) * inch * maxk;
        const signed char* kptr7 = (const signed char*)kernel + (q + 7) * inch * maxk;
        const signed char* kptr8 = (const signed char*)kernel + (q + 8) * inch * maxk;
        const signed char* kptr9 = (const signed char*)kernel + (q + 9) * inch * maxk;
        const signed char* kptra = (const signed char*)kernel + (q + 10) * inch * maxk;
        const signed char* kptrb = (const signed char*)kernel + (q + 11) * inch * maxk;
        const signed char* kptrc = (const signed char*)kernel + (q + 12) * inch * maxk;
        const signed char* kptrd = (const signed char*)kernel + (q + 13) * inch * maxk;
        const signed char* kptre = (const signed char*)kernel + (q + 14) * inch * maxk;
        const signed char* kptrf = (const signed char*)kernel + (q + 15) * inch * maxk;

        signed char* g00 = kernel_tm.channel(q / 16);

        int p = 0;
        for (; p + 15 < inch; p += 16)
        {
            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(maxk));

            for (int k = 0; k < maxk; k++)
            {
                __m128i _w0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr0 + k), 1));
                __m128i _w1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr1 + k), 1));
                __m128i _w2 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr2 + k), 1));
                __m128i _w3 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr3 + k), 1));
                __m128i _w4 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr4 + k), 1));
                __m128i _w5 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr5 + k), 1));
                __m128i _w6 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr6 + k), 1));
                __m128i _w7 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr7 + k), 1));
                __m128i _w8 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr8 + k), 1));
                __m128i _w9 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr9 + k), 1));
                __m128i _wa = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptra + k), 1));
                __m128i _wb = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptrb + k), 1));
                __m128i _wc = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptrc + k), 1));
                __m128i _wd = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptrd + k), 1));
                __m128i _we = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptre + k), 1));
                __m128i _wf = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptrf + k), 1));

                transpose8x16_epi16(_w0, _w1, _w2, _w3, _w4, _w5, _w6, _w7, _w8, _w9, _wa, _wb, _wc, _wd, _we, _wf);

                _mm_storeu_si128((__m128i*)g00, _w0);
                _mm_storeu_si128((__m128i*)(g00 + 16), _w1);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 2), _w2);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 3), _w3);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 4), _w4);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 5), _w5);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 6), _w6);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 7), _w7);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 8), _w8);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 9), _w9);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 10), _wa);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 11), _wb);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 12), _wc);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 13), _wd);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 14), _we);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 15), _wf);
                g00 += 256;
            }

            kptr0 += maxk * 16;
            kptr1 += maxk * 16;
            kptr2 += maxk * 16;
            kptr3 += maxk * 16;
            kptr4 += maxk * 16;
            kptr5 += maxk * 16;
            kptr6 += maxk * 16;
            kptr7 += maxk * 16;
            kptr8 += maxk * 16;
            kptr9 += maxk * 16;
            kptra += maxk * 16;
            kptrb += maxk * 16;
            kptrc += maxk * 16;
            kptrd += maxk * 16;
            kptre += maxk * 16;
            kptrf += maxk * 16;
        }
        for (; p + 7 < inch; p += 8)
        {
            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(maxk));

            for (int k = 0; k < maxk; k++)
            {
                __m128i _w0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr0 + k), _vindex, 1));
                __m128i _w1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr1 + k), _vindex, 1));
                __m128i _w2 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr2 + k), _vindex, 1));
                __m128i _w3 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr3 + k), _vindex, 1));
                __m128i _w4 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr4 + k), _vindex, 1));
                __m128i _w5 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr5 + k), _vindex, 1));
                __m128i _w6 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr6 + k), _vindex, 1));
                __m128i _w7 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr7 + k), _vindex, 1));
                __m128i _w8 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr8 + k), _vindex, 1));
                __m128i _w9 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr9 + k), _vindex, 1));
                __m128i _wa = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptra + k), _vindex, 1));
                __m128i _wb = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptrb + k), _vindex, 1));
                __m128i _wc = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptrc + k), _vindex, 1));
                __m128i _wd = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptrd + k), _vindex, 1));
                __m128i _we = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptre + k), _vindex, 1));
                __m128i _wf = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptrf + k), _vindex, 1));

                __m128i _w08 = _mm_unpacklo_epi64(_w0, _w8);
                __m128i _w19 = _mm_unpacklo_epi64(_w1, _w9);
                __m128i _w2a = _mm_unpacklo_epi64(_w2, _wa);
                __m128i _w3b = _mm_unpacklo_epi64(_w3, _wb);
                __m128i _w4c = _mm_unpacklo_epi64(_w4, _wc);
                __m128i _w5d = _mm_unpacklo_epi64(_w5, _wd);
                __m128i _w6e = _mm_unpacklo_epi64(_w6, _we);
                __m128i _w7f = _mm_unpacklo_epi64(_w7, _wf);

                transpose8x8_epi16(_w08, _w19, _w2a, _w3b, _w4c, _w5d, _w6e, _w7f);

                _mm_storeu_si128((__m128i*)g00, _w08);
                _mm_storeu_si128((__m128i*)(g00 + 16), _w4c);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 2), _w19);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 3), _w5d);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 4), _w2a);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 5), _w6e);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 6), _w3b);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 7), _w7f);
                g00 += 128;
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
            kptr2 += maxk * 8;
            kptr3 += maxk * 8;
            kptr4 += maxk * 8;
            kptr5 += maxk * 8;
            kptr6 += maxk * 8;
            kptr7 += maxk * 8;
            kptr8 += maxk * 8;
            kptr9 += maxk * 8;
            kptra += maxk * 8;
            kptrb += maxk * 8;
            kptrc += maxk * 8;
            kptrd += maxk * 8;
            kptre += maxk * 8;
            kptrf += maxk * 8;
        }
        for (; p + 1 < inch; p += 2)
        {
            __m128i _vindex0 = _mm_setr_epi32(0, maxk, inch * maxk, inch * maxk + maxk);
            __m128i _vindex1 = _mm_add_epi32(_vindex0, _mm_set1_epi32(inch * maxk * 2));
            __m256i _vindex01 = _mm256_inserti128_si256(_mm256_castsi128_si256(_vindex0), _vindex1, 1);
            __m256i _vindex23 = _mm256_add_epi32(_vindex01, _mm256_set1_epi32(inch * maxk * 4));
            __m512i _vindex = _mm512_inserti64x4(_mm512_castsi256_si512(_vindex01), _vindex23, 1);
            for (int k = 0; k < maxk; k++)
            {
                __m128i _w0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr0 + k), 1));
                __m128i _w1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr8 + k), 1));

                _mm_storeu_si128((__m128i*)g00, _w0);
                _mm_storeu_si128((__m128i*)(g00 + 16), _w1);
                g00 += 32;
            }

            kptr0 += maxk * 2;
            kptr8 += maxk * 2;
        }
        for (; p < inch; p++)
        {
            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(inch * maxk));
            for (int k = 0; k < maxk; k++)
            {
                __m128i _w0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr0 + k), 1));
                _mm_storeu_si128((__m128i*)g00, _w0);
                g00 += 16;
            }
        }
    }
#endif // __AVX512F__
    for (; q + 7 < outch; q += 8)
    {
        const signed char* kptr0 = (const signed char*)kernel + q * inch * maxk;
        const signed char* kptr1 = (const signed char*)kernel + (q + 1) * inch * maxk;
        const signed char* kptr2 = (const signed char*)kernel + (q + 2) * inch * maxk;
        const signed char* kptr3 = (const signed char*)kernel + (q + 3) * inch * maxk;
        const signed char* kptr4 = (const signed char*)kernel + (q + 4) * inch * maxk;
        const signed char* kptr5 = (const signed char*)kernel + (q + 5) * inch * maxk;
        const signed char* kptr6 = (const signed char*)kernel + (q + 6) * inch * maxk;
        const signed char* kptr7 = (const signed char*)kernel + (q + 7) * inch * maxk;

#if __AVX512F__
        signed char* g00 = kernel_tm.channel(q / 16 + (q % 16) / 8);
#else
        signed char* g00 = kernel_tm.channel(q / 8);
#endif

        int p = 0;
#if __AVX512F__
        for (; p + 15 < inch; p += 16)
        {
            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(maxk));

            for (int k = 0; k < maxk; k++)
            {
                __m128i _w0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr0 + k), 1));
                __m128i _w1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr1 + k), 1));
                __m128i _w2 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr2 + k), 1));
                __m128i _w3 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr3 + k), 1));
                __m128i _w4 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr4 + k), 1));
                __m128i _w5 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr5 + k), 1));
                __m128i _w6 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr6 + k), 1));
                __m128i _w7 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr7 + k), 1));

                transpose8x8_epi16(_w0, _w1, _w2, _w3, _w4, _w5, _w6, _w7);

                _mm_storeu_si128((__m128i*)g00, _w0);
                _mm_storeu_si128((__m128i*)(g00 + 16), _w1);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 2), _w2);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 3), _w3);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 4), _w4);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 5), _w5);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 6), _w6);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 7), _w7);
                g00 += 128;
            }

            kptr0 += maxk * 16;
            kptr1 += maxk * 16;
            kptr2 += maxk * 16;
            kptr3 += maxk * 16;
            kptr4 += maxk * 16;
            kptr5 += maxk * 16;
            kptr6 += maxk * 16;
            kptr7 += maxk * 16;
        }
#endif // __AVX512F__
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;
                const signed char* k4 = kptr4 + k;
                const signed char* k5 = kptr5 + k;
                const signed char* k6 = kptr6 + k;
                const signed char* k7 = kptr7 + k;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[maxk];
                    g00[2] = k1[0];
                    g00[3] = k1[maxk];
                    g00[4] = k2[0];
                    g00[5] = k2[maxk];
                    g00[6] = k3[0];
                    g00[7] = k3[maxk];
                    g00[8] = k4[0];
                    g00[9] = k4[maxk];
                    g00[10] = k5[0];
                    g00[11] = k5[maxk];
                    g00[12] = k6[0];
                    g00[13] = k6[maxk];
                    g00[14] = k7[0];
                    g00[15] = k7[maxk];

                    g00 += 16;
                    k0 += maxk * 2;
                    k1 += maxk * 2;
                    k2 += maxk * 2;
                    k3 += maxk * 2;
                    k4 += maxk * 2;
                    k5 += maxk * 2;
                    k6 += maxk * 2;
                    k7 += maxk * 2;
                }
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
            kptr2 += maxk * 8;
            kptr3 += maxk * 8;
            kptr4 += maxk * 8;
            kptr5 += maxk * 8;
            kptr6 += maxk * 8;
            kptr7 += maxk * 8;
        }
        for (; p + 1 < inch; p += 2)
        {
            __m128i _vindex0 = _mm_setr_epi32(0, maxk, inch * maxk, inch * maxk + maxk);
            __m128i _vindex1 = _mm_add_epi32(_vindex0, _mm_set1_epi32(inch * maxk * 2));
            __m256i _vindex01 = _mm256_inserti128_si256(_mm256_castsi128_si256(_vindex0), _vindex1, 1);
#if __AVX512F__
            __m256i _vindex23 = _mm256_add_epi32(_vindex01, _mm256_set1_epi32(inch * maxk * 4));
            __m512i _vindex = _mm512_inserti64x4(_mm512_castsi256_si512(_vindex01), _vindex23, 1);
#else
            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
#endif
            for (int k = 0; k < maxk; k++)
            {
#if __AVX512F__
                __m128i _w0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr0 + k), 1));
#else
                __m256i _w01 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr0 + k), _vindex01, 1), _sindex88);
                __m256i _w23 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr4 + k), _vindex01, 1), _sindex88);
                __m128i _w01xx = _mm_unpacklo_epi32(_mm256_extracti128_si256(_w01, 0), _mm256_extracti128_si256(_w01, 1));
                __m128i _w23xx = _mm_unpacklo_epi32(_mm256_extracti128_si256(_w23, 0), _mm256_extracti128_si256(_w23, 1));
                __m128i _w0 = _mm_unpacklo_epi64(_w01xx, _w23xx);
#endif
                _mm_storeu_si128((__m128i*)g00, _w0);
                g00 += 16;
            }

            kptr0 += maxk * 2;
            kptr4 += maxk * 2;
        }
        for (; p < inch; p++)
        {
            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(inch * maxk));
#if !__AVX512F__
            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
#endif
            for (int k = 0; k < maxk; k++)
            {
                __m256i _w32 = _mm256_i32gather_epi32((const int*)(kptr0 + k), _vindex, 1);
#if __AVX512F__
                __m128i _w0 = _mm256_cvtepi32_epi8(_w32);
#else
                __m256i _w01 = _mm256_shuffle_epi8(_w32, _sindex88);
                __m128i _w0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_w01, 0), _mm256_extracti128_si256(_w01, 1));
#endif
                _mm_storel_epi64((__m128i*)g00, _w0);
                g00 += 8;
            }
        }
    }
#endif // __AVX2__
    for (; q + 3 < outch; q += 4)
    {
        const signed char* kptr0 = (const signed char*)kernel + q * inch * maxk;
        const signed char* kptr1 = (const signed char*)kernel + (q + 1) * inch * maxk;
        const signed char* kptr2 = (const signed char*)kernel + (q + 2) * inch * maxk;
        const signed char* kptr3 = (const signed char*)kernel + (q + 3) * inch * maxk;

#if __AVX512F__
        signed char* g00 = kernel_tm.channel(q / 16 + (q % 16) / 8 + (q % 8) / 4);
#elif __AVX2__
        signed char* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        signed char* g00 = kernel_tm.channel(q / 4);
#endif

        int p = 0;
#if __AVX512F__
        for (; p + 15 < inch; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[maxk];
                    g00[2] = k1[0];
                    g00[3] = k1[maxk];
                    g00[4] = k2[0];
                    g00[5] = k2[maxk];
                    g00[6] = k3[0];
                    g00[7] = k3[maxk];

                    g00 += 8;
                    k0 += maxk * 2;
                    k1 += maxk * 2;
                    k2 += maxk * 2;
                    k3 += maxk * 2;
                }
            }

            kptr0 += maxk * 16;
            kptr1 += maxk * 16;
            kptr2 += maxk * 16;
            kptr3 += maxk * 16;
        }
#endif // __AVX512F__
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[maxk];
                    g00[2] = k1[0];
                    g00[3] = k1[maxk];
                    g00[4] = k2[0];
                    g00[5] = k2[maxk];
                    g00[6] = k3[0];
                    g00[7] = k3[maxk];

                    g00 += 8;
                    k0 += maxk * 2;
                    k1 += maxk * 2;
                    k2 += maxk * 2;
                    k3 += maxk * 2;
                }
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
            kptr2 += maxk * 8;
            kptr3 += maxk * 8;
        }
        for (; p + 1 < inch; p += 2)
        {
#if __AVX2__
            __m128i _vindex0 = _mm_setr_epi32(0, maxk, inch * maxk, inch * maxk + maxk);
            __m128i _vindex1 = _mm_add_epi32(_vindex0, _mm_set1_epi32(inch * maxk * 2));
            __m256i _vindex01 = _mm256_inserti128_si256(_mm256_castsi128_si256(_vindex0), _vindex1, 1);
#if !__AVX512F__
            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
#endif
#endif

            for (int k = 0; k < maxk; k++)
            {
#if __AVX512F__
                __m128i _w0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr0 + k), _vindex01, 1));
                _mm_storel_epi64((__m128i*)g00, _w0);
#elif __AVX2__
                __m256i _w01 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr0 + k), _vindex01, 1), _sindex88);
                __m128i _w0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_w01, 0), _mm256_extracti128_si256(_w01, 1));
                _mm_storel_epi64((__m128i*)g00, _w0);
#else
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;

                g00[0] = k0[0];
                g00[1] = k0[maxk];
                g00[2] = k1[0];
                g00[3] = k1[maxk];
                g00[4] = k2[0];
                g00[5] = k2[maxk];
                g00[6] = k3[0];
                g00[7] = k3[maxk];
#endif
                g00 += 8;
            }

            kptr0 += maxk * 2;
            kptr1 += maxk * 2;
            kptr2 += maxk * 2;
            kptr3 += maxk * 2;
        }
        for (; p < inch; p++)
        {
#if __AVX2__
            __m128i _vindex = _mm_mullo_epi32(_mm_setr_epi32(0, 1, 2, 3), _mm_set1_epi32(inch * maxk));
#if !__AVX512F__
            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
#endif
#endif

            for (int k = 0; k < maxk; k++)
            {
#if __AVX512F__
                __m128i _w0 = _mm_cvtepi32_epi8(_mm_i32gather_epi32((const int*)(kptr0 + k), _vindex, 1));
                _mm_store_ss((float*)g00, _mm_castsi128_ps(_w0));
#elif __AVX2__
                __m128i _w0 = _mm_shuffle_epi8(_mm_i32gather_epi32((const int*)(kptr0 + k), _vindex, 1), _sindex8);
                _mm_store_ss((float*)g00, _mm_castsi128_ps(_w0));
#else
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k2[0];
                g00[3] = k3[0];
#endif
                g00 += 4;
            }
        }
    }
#endif // __SSE2__
    for (; q + 1 < outch; q += 2)
    {
        const signed char* kptr0 = (const signed char*)kernel + q * inch * maxk;
        const signed char* kptr1 = (const signed char*)kernel + (q + 1) * inch * maxk;

#if __AVX512F__
        signed char* g00 = kernel_tm.channel(q / 16 + (q % 16) / 8 + (q % 8) / 4 + (q % 4) / 2);
#elif __AVX2__
        signed char* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#elif __SSE2__
        signed char* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2);
#else
        signed char* g00 = kernel_tm.channel(q / 2);
#endif

        int p = 0;
#if __SSE2__
#if __AVX512F__
        for (; p + 15 < inch; p += 16)
        {
            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(maxk));

            for (int k = 0; k < maxk; k++)
            {
                __m128i _w0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr0 + k), 1));
                __m128i _w1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr1 + k), 1));

                _mm_storeu_si128((__m128i*)g00, _w0);
                _mm_storeu_si128((__m128i*)(g00 + 16), _w1);
                g00 += 32;
            }

            kptr0 += maxk * 16;
            kptr1 += maxk * 16;
        }
#endif // __AVX512F__
        for (; p + 7 < inch; p += 8)
        {
#if __AVX2__
            __m256i _vindex0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            _vindex0 = _mm256_mullo_epi32(_vindex0, _mm256_set1_epi32(maxk));
#if __AVX512F__
            __m256i _vindex1 = _mm256_add_epi32(_vindex0, _mm256_set1_epi32(inch * maxk));
            __m512i _vindex = _mm512_inserti64x4(_mm512_castsi256_si512(_vindex0), _vindex1, 1);
#else
            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
#endif
#endif

            for (int k = 0; k < maxk; k++)
            {
#if __AVX512F__
                __m128i _w0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr0 + k), 1));
                _mm_storeu_si128((__m128i*)g00, _w0);
#elif __AVX2__
                __m256i _w00 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr0 + k), _vindex0, 1), _sindex88);
                __m256i _w11 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr1 + k), _vindex0, 1), _sindex88);
                __m128i _w0x = _mm_unpacklo_epi32(_mm256_extracti128_si256(_w00, 0), _mm256_extracti128_si256(_w00, 1));
                __m128i _w1x = _mm_unpacklo_epi32(_mm256_extracti128_si256(_w11, 0), _mm256_extracti128_si256(_w11, 1));
                __m128i _w0 = _mm_unpacklo_epi64(_w0x, _w1x);
                _mm_storeu_si128((__m128i*)g00, _w0);
#else
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k0[maxk];
                g00[2] = k0[maxk * 2];
                g00[3] = k0[maxk * 3];
                g00[4] = k0[maxk * 4];
                g00[5] = k0[maxk * 5];
                g00[6] = k0[maxk * 6];
                g00[7] = k0[maxk * 7];
                g00[8] = k1[0];
                g00[9] = k1[maxk];
                g00[10] = k1[maxk * 2];
                g00[11] = k1[maxk * 3];
                g00[12] = k1[maxk * 4];
                g00[13] = k1[maxk * 5];
                g00[14] = k1[maxk * 6];
                g00[15] = k1[maxk * 7];
#endif
                g00 += 16;
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
        }
#endif // __SSE2__
        for (; p + 1 < inch; p += 2)
        {
#if __AVX2__
            __m128i _vindex = _mm_setr_epi32(0, inch * maxk, maxk, inch * maxk + maxk);
#if !__AVX512F__
            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
#endif
#endif

            for (int k = 0; k < maxk; k++)
            {
#if __AVX512F__
                __m128i _w0 = _mm_cvtepi32_epi8(_mm_i32gather_epi32((const int*)(kptr0 + k), _vindex, 1));
                _mm_store_ss((float*)g00, _mm_castsi128_ps(_w0));
#elif __AVX2__
                __m128i _w0 = _mm_shuffle_epi8(_mm_i32gather_epi32((const int*)(kptr0 + k), _vindex, 1), _sindex8);
                _mm_store_ss((float*)g00, _mm_castsi128_ps(_w0));
#else
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k0[maxk];
                g00[3] = k1[maxk];
#endif
                g00 += 4;
            }

            kptr0 += maxk * 2;
            kptr1 += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00 += 2;
            }
        }
    }
    for (; q < outch; q++)
    {
        const signed char* kptr = (const signed char*)kernel + q * inch * maxk;

#if __AVX512F__
        signed char* g00 = kernel_tm.channel(q / 16 + (q % 16) / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#elif __AVX2__
        signed char* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#elif __SSE2__
        signed char* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#else
        signed char* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __SSE2__
#if __AVX512F__
        for (; p + 15 < inch; p += 16)
        {
            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(maxk));

            for (int k = 0; k < maxk; k++)
            {
                __m128i _w0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(kptr + k), 1));

                _mm_storeu_si128((__m128i*)g00, _w0);
                g00 += 16;
            }

            kptr += maxk * 16;
        }
#endif // __AVX512F__
        for (; p + 7 < inch; p += 8)
        {
#if __AVX2__
            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(maxk));
#if !__AVX512F__
            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
#endif
#endif
            for (int k = 0; k < maxk; k++)
            {
#if __AVX512F__
                __m128i _w0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr + k), _vindex, 1));

                _mm_storel_epi64((__m128i*)g00, _w0);
                g00 += 8;
#elif __AVX2__
                __m256i _w00 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr + k), _vindex, 1), _sindex88);
                __m128i _w0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_w00, 0), _mm256_extracti128_si256(_w00, 1));

                _mm_storel_epi64((__m128i*)g00, _w0);
                g00 += 8;
#else
                const signed char* k0 = kptr + k;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
#endif
            }

            kptr += maxk * 8;
        }
#endif // __SSE2__
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr + k;

                g00[0] = k0[0];
                g00[1] = k0[maxk];
                g00 += 2;
            }

            kptr += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr + k;

                g00[0] = k0[0];
                g00++;
            }
        }
    }
}

static void convolution_packed_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        convolution_packed_int8_avx512vnni(bottom_blob, top_blob, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        convolution_packed_int8_avxvnni(bottom_blob, top_blob, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        convolution_packed_int8_avx2(bottom_blob, top_blob, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__
    if (ncnn::cpu_support_x86_xop())
    {
        convolution_packed_int8_xop(bottom_blob, top_blob, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        return;
    }
#endif
#endif

    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int inch = bottom_blob.c * elempack;

    const int N = bottom_blob.cstep * elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c * out_elempack;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2 * elempack;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    int nn_outch = 0;
    int remain_outch_start = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    nn_outch = outch / 16;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = pp * 16;

        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int N = bottom_blob.cstep * elempack;
        const int M = top_blob.cstep * out_elempack;

        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;
        for (; ij + 3 < outw * outh; ij += 4)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int i2 = (ij + 2) / outw;
            const int i3 = (ij + 3) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;
            const int j2 = (ij + 2) % outw;
            const int j3 = (ij + 3) % outw;

            __m512i _sum0 = _mm512_setzero_si512();
            __m512i _sum1 = _mm512_setzero_si512();
            __m512i _sum2 = _mm512_setzero_si512();
            __m512i _sum3 = _mm512_setzero_si512();

            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8);

            int q = 0;
            for (; q + 15 < inch; q += 16)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;
                const signed char* r2 = bottom_blob.channel(q / elempack).row<const signed char>(i2 * stride_h) + j2 * stride_w * elempack;
                const signed char* r3 = bottom_blob.channel(q / elempack).row<const signed char>(i3 * stride_h) + j3 * stride_w * elempack;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    const signed char* r2s = r2 + space_ofs[k];
                    const signed char* r3s = r3 + space_ofs[k];

                    __m128i _r0;
                    __m128i _r1;
                    __m128i _r2;
                    __m128i _r3;
                    if (elempack == 16)
                    {
                        _r0 = _mm_load_si128((const __m128i*)r0s);
                        _r1 = _mm_load_si128((const __m128i*)r1s);
                        _r2 = _mm_load_si128((const __m128i*)r2s);
                        _r3 = _mm_load_si128((const __m128i*)r3s);
                    }
                    else if (elempack == 8)
                    {
                        __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                        __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                        __m128i _t2 = _mm_loadl_epi64((const __m128i*)r1s);
                        __m128i _t3 = _mm_loadl_epi64((const __m128i*)(r1s + N));
                        __m128i _t4 = _mm_loadl_epi64((const __m128i*)r2s);
                        __m128i _t5 = _mm_loadl_epi64((const __m128i*)(r2s + N));
                        __m128i _t6 = _mm_loadl_epi64((const __m128i*)r3s);
                        __m128i _t7 = _mm_loadl_epi64((const __m128i*)(r3s + N));
                        _r0 = _mm_unpacklo_epi64(_t0, _t1);
                        _r1 = _mm_unpacklo_epi64(_t2, _t3);
                        _r2 = _mm_unpacklo_epi64(_t4, _t5);
                        _r3 = _mm_unpacklo_epi64(_t6, _t7);
                    }
                    else // if (elempack == 1)
                    {
                        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                        _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                        _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r1s), 1));
                        _r2 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r2s), 1));
                        _r3 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r3s), 1));
                    }

                    __m256i _rr0 = _mm256_cvtepi8_epi16(_r0);
                    __m256i _rr1 = _mm256_cvtepi8_epi16(_r1);
                    __m256i _rr2 = _mm256_cvtepi8_epi16(_r2);
                    __m256i _rr3 = _mm256_cvtepi8_epi16(_r3);

                    __m512i _w01 = _mm512_load_si512((const __m512i*)kptr);
                    __m512i _w23 = _mm512_load_si512((const __m512i*)(kptr + 64));
                    __m512i _w45 = _mm512_load_si512((const __m512i*)(kptr + 128));
                    __m512i _w67 = _mm512_load_si512((const __m512i*)(kptr + 192));
                    __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                    __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                    __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                    __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));
                    __m512i _w4 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w45, 0));
                    __m512i _w5 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w45, 1));
                    __m512i _w6 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w67, 0));
                    __m512i _w7 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w67, 1));

                    // 01234567 89abcdef -> 01010101 01010101 01010101 01010101 01010101 01010101 01010101 01010101
                    __m512i _rrrr00 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr0, 0));
                    __m512i _rrrr01 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr0, 1));
                    __m512i _rrrr10 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr1, 0));
                    __m512i _rrrr11 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr1, 1));
                    __m512i _rrrr20 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr2, 0));
                    __m512i _rrrr21 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr2, 1));
                    __m512i _rrrr30 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr3, 0));
                    __m512i _rrrr31 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr3, 1));

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr00, _MM_PERM_AAAA), _w0);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr10, _MM_PERM_AAAA), _w0);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr20, _MM_PERM_AAAA), _w0);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr30, _MM_PERM_AAAA), _w0);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr00, _MM_PERM_BBBB), _w1);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr10, _MM_PERM_BBBB), _w1);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr20, _MM_PERM_BBBB), _w1);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr30, _MM_PERM_BBBB), _w1);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr00, _MM_PERM_CCCC), _w2);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr10, _MM_PERM_CCCC), _w2);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr20, _MM_PERM_CCCC), _w2);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr30, _MM_PERM_CCCC), _w2);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr00, _MM_PERM_DDDD), _w3);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr10, _MM_PERM_DDDD), _w3);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr20, _MM_PERM_DDDD), _w3);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr30, _MM_PERM_DDDD), _w3);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr01, _MM_PERM_AAAA), _w4);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr11, _MM_PERM_AAAA), _w4);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr21, _MM_PERM_AAAA), _w4);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr31, _MM_PERM_AAAA), _w4);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr01, _MM_PERM_BBBB), _w5);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr11, _MM_PERM_BBBB), _w5);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr21, _MM_PERM_BBBB), _w5);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr31, _MM_PERM_BBBB), _w5);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr01, _MM_PERM_CCCC), _w6);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr11, _MM_PERM_CCCC), _w6);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr21, _MM_PERM_CCCC), _w6);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr31, _MM_PERM_CCCC), _w6);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr01, _MM_PERM_DDDD), _w7);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr11, _MM_PERM_DDDD), _w7);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr21, _MM_PERM_DDDD), _w7);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr31, _MM_PERM_DDDD), _w7);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr00, _MM_PERM_AAAA), _w0));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr10, _MM_PERM_AAAA), _w0));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr20, _MM_PERM_AAAA), _w0));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr30, _MM_PERM_AAAA), _w0));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr00, _MM_PERM_BBBB), _w1));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr10, _MM_PERM_BBBB), _w1));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr20, _MM_PERM_BBBB), _w1));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr30, _MM_PERM_BBBB), _w1));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr00, _MM_PERM_CCCC), _w2));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr10, _MM_PERM_CCCC), _w2));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr20, _MM_PERM_CCCC), _w2));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr30, _MM_PERM_CCCC), _w2));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr00, _MM_PERM_DDDD), _w3));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr10, _MM_PERM_DDDD), _w3));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr20, _MM_PERM_DDDD), _w3));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr30, _MM_PERM_DDDD), _w3));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr01, _MM_PERM_AAAA), _w4));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr11, _MM_PERM_AAAA), _w4));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr21, _MM_PERM_AAAA), _w4));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr31, _MM_PERM_AAAA), _w4));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr01, _MM_PERM_BBBB), _w5));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr11, _MM_PERM_BBBB), _w5));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr21, _MM_PERM_BBBB), _w5));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr31, _MM_PERM_BBBB), _w5));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr01, _MM_PERM_CCCC), _w6));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr11, _MM_PERM_CCCC), _w6));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr21, _MM_PERM_CCCC), _w6));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr31, _MM_PERM_CCCC), _w6));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr01, _MM_PERM_DDDD), _w7));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr11, _MM_PERM_DDDD), _w7));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr21, _MM_PERM_DDDD), _w7));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr31, _MM_PERM_DDDD), _w7));
#endif // __AVX512VNNI__

                    kptr += 256;
                }
            }
            for (; q + 7 < inch; q += 8)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;
                const signed char* r2 = bottom_blob.channel(q / elempack).row<const signed char>(i2 * stride_h) + j2 * stride_w * elempack;
                const signed char* r3 = bottom_blob.channel(q / elempack).row<const signed char>(i3 * stride_h) + j3 * stride_w * elempack;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    const signed char* r2s = r2 + space_ofs[k];
                    const signed char* r3s = r3 + space_ofs[k];

                    __m128i _r0;
                    __m128i _r1;
                    __m128i _r2;
                    __m128i _r3;
                    if (elempack == 8)
                    {
                        _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                        _r1 = _mm_loadl_epi64((const __m128i*)r1s);
                        _r2 = _mm_loadl_epi64((const __m128i*)r2s);
                        _r3 = _mm_loadl_epi64((const __m128i*)r3s);
                    }
                    else // if (elempack == 1)
                    {
                        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
                        _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
                        _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1));
                        _r2 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r2s, _vindex, 1));
                        _r3 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r3s, _vindex, 1));
                    }

                    _r0 = _mm_cvtepi8_epi16(_r0);
                    _r1 = _mm_cvtepi8_epi16(_r1);
                    _r2 = _mm_cvtepi8_epi16(_r2);
                    _r3 = _mm_cvtepi8_epi16(_r3);

                    __m512i _w01 = _mm512_load_si512((const __m512i*)kptr);
                    __m512i _w23 = _mm512_load_si512((const __m512i*)(kptr + 64));
                    __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                    __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                    __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                    __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));

                    // 01234567 -> 01010101 01010101 01010101 01010101
                    __m512i _rrrr0 = _mm512_broadcast_i32x4(_r0);
                    __m512i _rrrr1 = _mm512_broadcast_i32x4(_r1);
                    __m512i _rrrr2 = _mm512_broadcast_i32x4(_r2);
                    __m512i _rrrr3 = _mm512_broadcast_i32x4(_r3);

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr0, _MM_PERM_AAAA), _w0);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr1, _MM_PERM_AAAA), _w0);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr2, _MM_PERM_AAAA), _w0);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr3, _MM_PERM_AAAA), _w0);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr0, _MM_PERM_BBBB), _w1);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr1, _MM_PERM_BBBB), _w1);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr2, _MM_PERM_BBBB), _w1);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr3, _MM_PERM_BBBB), _w1);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr0, _MM_PERM_CCCC), _w2);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr1, _MM_PERM_CCCC), _w2);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr2, _MM_PERM_CCCC), _w2);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr3, _MM_PERM_CCCC), _w2);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr0, _MM_PERM_DDDD), _w3);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr1, _MM_PERM_DDDD), _w3);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr2, _MM_PERM_DDDD), _w3);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr3, _MM_PERM_DDDD), _w3);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr0, _MM_PERM_AAAA), _w0));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr1, _MM_PERM_AAAA), _w0));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr2, _MM_PERM_AAAA), _w0));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr3, _MM_PERM_AAAA), _w0));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr0, _MM_PERM_BBBB), _w1));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr1, _MM_PERM_BBBB), _w1));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr2, _MM_PERM_BBBB), _w1));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr3, _MM_PERM_BBBB), _w1));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr0, _MM_PERM_CCCC), _w2));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr1, _MM_PERM_CCCC), _w2));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr2, _MM_PERM_CCCC), _w2));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr3, _MM_PERM_CCCC), _w2));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr0, _MM_PERM_DDDD), _w3));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr1, _MM_PERM_DDDD), _w3));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr2, _MM_PERM_DDDD), _w3));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr3, _MM_PERM_DDDD), _w3));
#endif // __AVX512VNNI__

                    kptr += 128;
                }
            }
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;
                const signed char* r2 = bottom_blob.channel(q).row<const signed char>(i2 * stride_h) + j2 * stride_w;
                const signed char* r3 = bottom_blob.channel(q).row<const signed char>(i3 * stride_h) + j3 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    const signed char* r2s = r2 + space_ofs[k];
                    const signed char* r3s = r3 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m512i _r0 = _mm512_broadcastd_epi32(_mm_setr_epi16(r0s[0], r0s[N], 0, 0, 0, 0, 0, 0));
                        __m512i _r1 = _mm512_broadcastd_epi32(_mm_setr_epi16(r1s[0], r1s[N], 0, 0, 0, 0, 0, 0));
                        __m512i _r2 = _mm512_broadcastd_epi32(_mm_setr_epi16(r2s[0], r2s[N], 0, 0, 0, 0, 0, 0));
                        __m512i _r3 = _mm512_broadcastd_epi32(_mm_setr_epi16(r3s[0], r3s[N], 0, 0, 0, 0, 0, 0));

                        __m512i _w = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)kptr));

#if __AVX512VNNI__
                        _sum0 = _mm512_dpwssd_epi32(_sum0, _r0, _w);
                        _sum1 = _mm512_dpwssd_epi32(_sum1, _r1, _w);
                        _sum2 = _mm512_dpwssd_epi32(_sum2, _r2, _w);
                        _sum3 = _mm512_dpwssd_epi32(_sum3, _r3, _w);
#else
                        _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_r0, _w));
                        _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_r1, _w));
                        _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_r2, _w));
                        _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_r3, _w));
#endif // __AVX512VNNI__

                        kptr += 32;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;
                const signed char* r2 = bottom_blob.channel(q).row<const signed char>(i2 * stride_h) + j2 * stride_w;
                const signed char* r3 = bottom_blob.channel(q).row<const signed char>(i3 * stride_h) + j3 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    const signed char* r2s = r2 + space_ofs[k];
                    const signed char* r3s = r3 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m256i _r0 = _mm256_set1_epi16(r0s[0]);
                        __m256i _r1 = _mm256_set1_epi16(r1s[0]);
                        __m256i _r2 = _mm256_set1_epi16(r2s[0]);
                        __m256i _r3 = _mm256_set1_epi16(r3s[0]);

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)kptr));

                        _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_r0, _w)));
                        _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_r1, _w)));
                        _sum2 = _mm512_add_epi32(_sum2, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_r2, _w)));
                        _sum3 = _mm512_add_epi32(_sum3, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_r3, _w)));

                        kptr += 16;
                    }
                }
            }

            if (out_elempack == 16)
            {
                _mm512_store_si512((__m512i*)outptr, _sum0);
                _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
                _mm512_store_si512((__m512i*)(outptr + 32), _sum2);
                _mm512_store_si512((__m512i*)(outptr + 48), _sum3);
                outptr += 64;
            }
            if (out_elempack == 8)
            {
                _mm256_store_si256((__m256i*)outptr, _mm512_extracti32x8_epi32(_sum0, 0));
                _mm256_store_si256((__m256i*)(outptr + M), _mm512_extracti32x8_epi32(_sum0, 1));
                _mm256_store_si256((__m256i*)(outptr + 8), _mm512_extracti32x8_epi32(_sum1, 0));
                _mm256_store_si256((__m256i*)(outptr + M + 8), _mm512_extracti32x8_epi32(_sum1, 1));
                _mm256_store_si256((__m256i*)(outptr + 16), _mm512_extracti32x8_epi32(_sum2, 0));
                _mm256_store_si256((__m256i*)(outptr + M + 16), _mm512_extracti32x8_epi32(_sum2, 1));
                _mm256_store_si256((__m256i*)(outptr + 24), _mm512_extracti32x8_epi32(_sum3, 0));
                _mm256_store_si256((__m256i*)(outptr + M + 24), _mm512_extracti32x8_epi32(_sum3, 1));
                outptr += 32;
            }
            if (out_elempack == 4)
            {
                _mm_store_si128((__m128i*)outptr, _mm512_extracti32x4_epi32(_sum0, 0));
                _mm_store_si128((__m128i*)(outptr + M), _mm512_extracti32x4_epi32(_sum0, 1));
                _mm_store_si128((__m128i*)(outptr + M * 2), _mm512_extracti32x4_epi32(_sum0, 2));
                _mm_store_si128((__m128i*)(outptr + M * 3), _mm512_extracti32x4_epi32(_sum0, 3));
                _mm_store_si128((__m128i*)(outptr + 4), _mm512_extracti32x4_epi32(_sum1, 0));
                _mm_store_si128((__m128i*)(outptr + M + 4), _mm512_extracti32x4_epi32(_sum1, 1));
                _mm_store_si128((__m128i*)(outptr + M * 2 + 4), _mm512_extracti32x4_epi32(_sum1, 2));
                _mm_store_si128((__m128i*)(outptr + M * 3 + 4), _mm512_extracti32x4_epi32(_sum1, 3));
                _mm_store_si128((__m128i*)(outptr + 8), _mm512_extracti32x4_epi32(_sum2, 0));
                _mm_store_si128((__m128i*)(outptr + M + 8), _mm512_extracti32x4_epi32(_sum2, 1));
                _mm_store_si128((__m128i*)(outptr + M * 2 + 8), _mm512_extracti32x4_epi32(_sum2, 2));
                _mm_store_si128((__m128i*)(outptr + M * 3 + 8), _mm512_extracti32x4_epi32(_sum2, 3));
                _mm_store_si128((__m128i*)(outptr + 12), _mm512_extracti32x4_epi32(_sum3, 0));
                _mm_store_si128((__m128i*)(outptr + M + 12), _mm512_extracti32x4_epi32(_sum3, 1));
                _mm_store_si128((__m128i*)(outptr + M * 2 + 12), _mm512_extracti32x4_epi32(_sum3, 2));
                _mm_store_si128((__m128i*)(outptr + M * 3 + 12), _mm512_extracti32x4_epi32(_sum3, 3));
                outptr += 16;
            }
            if (out_elempack == 1)
            {
                __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(M));
                _mm512_i32scatter_epi32(outptr, _vindex, _sum0, sizeof(int));
                _mm512_i32scatter_epi32(outptr + 1, _vindex, _sum1, sizeof(int));
                _mm512_i32scatter_epi32(outptr + 2, _vindex, _sum2, sizeof(int));
                _mm512_i32scatter_epi32(outptr + 3, _vindex, _sum3, sizeof(int));
                outptr += 4;
            }
        }
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            __m512i _sum0 = _mm512_setzero_si512();
            __m512i _sum1 = _mm512_setzero_si512();
            __m512i _sum2 = _mm512_setzero_si512();
            __m512i _sum3 = _mm512_setzero_si512();

            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8);

            int q = 0;
            for (; q + 15 < inch; q += 16)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    __m128i _r0;
                    __m128i _r1;
                    if (elempack == 16)
                    {
                        _r0 = _mm_load_si128((const __m128i*)r0s);
                        _r1 = _mm_load_si128((const __m128i*)r1s);
                    }
                    else if (elempack == 8)
                    {
                        __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                        __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                        __m128i _t2 = _mm_loadl_epi64((const __m128i*)r1s);
                        __m128i _t3 = _mm_loadl_epi64((const __m128i*)(r1s + N));
                        _r0 = _mm_unpacklo_epi64(_t0, _t1);
                        _r1 = _mm_unpacklo_epi64(_t2, _t3);
                    }
                    else // if (elempack == 1)
                    {
                        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                        _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                        _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r1s), 1));
                    }

                    __m256i _rr0 = _mm256_cvtepi8_epi16(_r0);
                    __m256i _rr1 = _mm256_cvtepi8_epi16(_r1);

                    __m512i _w01 = _mm512_load_si512((const __m512i*)kptr);
                    __m512i _w23 = _mm512_load_si512((const __m512i*)(kptr + 64));
                    __m512i _w45 = _mm512_load_si512((const __m512i*)(kptr + 128));
                    __m512i _w67 = _mm512_load_si512((const __m512i*)(kptr + 192));
                    __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                    __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                    __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                    __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));
                    __m512i _w4 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w45, 0));
                    __m512i _w5 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w45, 1));
                    __m512i _w6 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w67, 0));
                    __m512i _w7 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w67, 1));

                    // 01234567 89abcdef -> 01010101 01010101 01010101 01010101 01010101 01010101 01010101 01010101
                    __m512i _rrrr00 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr0, 0));
                    __m512i _rrrr01 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr0, 1));
                    __m512i _rrrr10 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr1, 0));
                    __m512i _rrrr11 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr1, 1));

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr00, _MM_PERM_AAAA), _w0);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr10, _MM_PERM_AAAA), _w0);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr00, _MM_PERM_BBBB), _w1);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr10, _MM_PERM_BBBB), _w1);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr00, _MM_PERM_CCCC), _w2);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr10, _MM_PERM_CCCC), _w2);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr00, _MM_PERM_DDDD), _w3);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr10, _MM_PERM_DDDD), _w3);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr01, _MM_PERM_AAAA), _w4);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr11, _MM_PERM_AAAA), _w4);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr01, _MM_PERM_BBBB), _w5);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr11, _MM_PERM_BBBB), _w5);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr01, _MM_PERM_CCCC), _w6);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr11, _MM_PERM_CCCC), _w6);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr01, _MM_PERM_DDDD), _w7);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr11, _MM_PERM_DDDD), _w7);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr00, _MM_PERM_AAAA), _w0));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr10, _MM_PERM_AAAA), _w0));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr00, _MM_PERM_BBBB), _w1));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr10, _MM_PERM_BBBB), _w1));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr00, _MM_PERM_CCCC), _w2));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr10, _MM_PERM_CCCC), _w2));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr00, _MM_PERM_DDDD), _w3));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr10, _MM_PERM_DDDD), _w3));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr01, _MM_PERM_AAAA), _w4));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr11, _MM_PERM_AAAA), _w4));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr01, _MM_PERM_BBBB), _w5));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr11, _MM_PERM_BBBB), _w5));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr01, _MM_PERM_CCCC), _w6));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr11, _MM_PERM_CCCC), _w6));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr01, _MM_PERM_DDDD), _w7));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr11, _MM_PERM_DDDD), _w7));
#endif // __AVX512VNNI__

                    kptr += 256;
                }
            }
            for (; q + 7 < inch; q += 8)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    __m128i _r0;
                    __m128i _r1;
                    if (elempack == 8)
                    {
                        _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                        _r1 = _mm_loadl_epi64((const __m128i*)r1s);
                    }
                    else // if (elempack == 1)
                    {
                        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
                        _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
                        _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1));
                    }

                    _r0 = _mm_cvtepi8_epi16(_r0);
                    _r1 = _mm_cvtepi8_epi16(_r1);

                    __m512i _w01 = _mm512_load_si512((const __m512i*)kptr);
                    __m512i _w23 = _mm512_load_si512((const __m512i*)(kptr + 64));
                    __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                    __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                    __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                    __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));

                    // 01234567 -> 01010101 01010101 01010101 01010101
                    __m512i _rrrr0 = _mm512_broadcast_i32x4(_r0);
                    __m512i _rrrr1 = _mm512_broadcast_i32x4(_r1);

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr0, _MM_PERM_AAAA), _w0);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr1, _MM_PERM_AAAA), _w0);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr0, _MM_PERM_BBBB), _w1);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr1, _MM_PERM_BBBB), _w1);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr0, _MM_PERM_CCCC), _w2);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr1, _MM_PERM_CCCC), _w2);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr0, _MM_PERM_DDDD), _w3);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr1, _MM_PERM_DDDD), _w3);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr0, _MM_PERM_AAAA), _w0));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr1, _MM_PERM_AAAA), _w0));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr0, _MM_PERM_BBBB), _w1));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr1, _MM_PERM_BBBB), _w1));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr0, _MM_PERM_CCCC), _w2));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr1, _MM_PERM_CCCC), _w2));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr0, _MM_PERM_DDDD), _w3));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr1, _MM_PERM_DDDD), _w3));
#endif // __AVX512VNNI__

                    kptr += 128;
                }
            }
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m512i _r0 = _mm512_broadcastd_epi32(_mm_setr_epi16(r0s[0], r0s[N], 0, 0, 0, 0, 0, 0));
                        __m512i _r1 = _mm512_broadcastd_epi32(_mm_setr_epi16(r1s[0], r1s[N], 0, 0, 0, 0, 0, 0));

                        __m512i _w = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)kptr));

#if __AVX512VNNI__
                        _sum0 = _mm512_dpwssd_epi32(_sum0, _r0, _w);
                        _sum1 = _mm512_dpwssd_epi32(_sum1, _r1, _w);
#else
                        _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_r0, _w));
                        _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_r1, _w));
#endif // __AVX512VNNI__

                        kptr += 32;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m256i _r0 = _mm256_set1_epi16(r0s[0]);
                        __m256i _r1 = _mm256_set1_epi16(r1s[0]);

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)kptr));

                        _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_r0, _w)));
                        _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_r1, _w)));

                        kptr += 16;
                    }
                }
            }

            _sum0 = _mm512_add_epi32(_sum0, _sum2);
            _sum1 = _mm512_add_epi32(_sum1, _sum3);

            if (out_elempack == 16)
            {
                _mm512_store_si512((__m512i*)outptr, _sum0);
                _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
                outptr += 32;
            }
            if (out_elempack == 8)
            {
                _mm256_store_si256((__m256i*)outptr, _mm512_extracti32x8_epi32(_sum0, 0));
                _mm256_store_si256((__m256i*)(outptr + M), _mm512_extracti32x8_epi32(_sum0, 1));
                _mm256_store_si256((__m256i*)(outptr + 8), _mm512_extracti32x8_epi32(_sum1, 0));
                _mm256_store_si256((__m256i*)(outptr + M + 8), _mm512_extracti32x8_epi32(_sum1, 1));
                outptr += 16;
            }
            if (out_elempack == 4)
            {
                _mm_store_si128((__m128i*)outptr, _mm512_extracti32x4_epi32(_sum0, 0));
                _mm_store_si128((__m128i*)(outptr + M), _mm512_extracti32x4_epi32(_sum0, 1));
                _mm_store_si128((__m128i*)(outptr + M * 2), _mm512_extracti32x4_epi32(_sum0, 2));
                _mm_store_si128((__m128i*)(outptr + M * 3), _mm512_extracti32x4_epi32(_sum0, 3));
                _mm_store_si128((__m128i*)(outptr + 4), _mm512_extracti32x4_epi32(_sum1, 0));
                _mm_store_si128((__m128i*)(outptr + M + 4), _mm512_extracti32x4_epi32(_sum1, 1));
                _mm_store_si128((__m128i*)(outptr + M * 2 + 4), _mm512_extracti32x4_epi32(_sum1, 2));
                _mm_store_si128((__m128i*)(outptr + M * 3 + 4), _mm512_extracti32x4_epi32(_sum1, 3));
                outptr += 8;
            }
            if (out_elempack == 1)
            {
                __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(M));
                _mm512_i32scatter_epi32(outptr, _vindex, _sum0, sizeof(int));
                _mm512_i32scatter_epi32(outptr + 1, _vindex, _sum1, sizeof(int));
                outptr += 2;
            }
        }
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            __m512i _sum0 = _mm512_setzero_si512();
            __m512i _sum1 = _mm512_setzero_si512();
            __m512i _sum2 = _mm512_setzero_si512();
            __m512i _sum3 = _mm512_setzero_si512();

            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8);

            int q = 0;
            for (; q + 15 < inch; q += 16)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    __m128i _r0;
                    if (elempack == 16)
                    {
                        _r0 = _mm_load_si128((const __m128i*)r0s);
                    }
                    else if (elempack == 8)
                    {
                        __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                        __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                        _r0 = _mm_unpacklo_epi64(_t0, _t1);
                    }
                    else // if (elempack == 1)
                    {
                        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                        _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                    }

                    __m256i _rr0 = _mm256_cvtepi8_epi16(_r0);

                    __m512i _w01 = _mm512_load_si512((const __m512i*)kptr);
                    __m512i _w23 = _mm512_load_si512((const __m512i*)(kptr + 64));
                    __m512i _w45 = _mm512_load_si512((const __m512i*)(kptr + 128));
                    __m512i _w67 = _mm512_load_si512((const __m512i*)(kptr + 192));
                    __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                    __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                    __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                    __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));
                    __m512i _w4 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w45, 0));
                    __m512i _w5 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w45, 1));
                    __m512i _w6 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w67, 0));
                    __m512i _w7 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w67, 1));

                    // 01234567 89abcdef -> 01010101 01010101 01010101 01010101 01010101 01010101 01010101 01010101
                    __m512i _rrrr00 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr0, 0));
                    __m512i _rrrr01 = _mm512_broadcast_i32x4(_mm256_extracti128_si256(_rr0, 1));

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr00, _MM_PERM_AAAA), _w0);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr00, _MM_PERM_BBBB), _w1);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr00, _MM_PERM_CCCC), _w2);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr00, _MM_PERM_DDDD), _w3);
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr01, _MM_PERM_AAAA), _w4);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr01, _MM_PERM_BBBB), _w5);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr01, _MM_PERM_CCCC), _w6);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr01, _MM_PERM_DDDD), _w7);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr00, _MM_PERM_AAAA), _w0));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr00, _MM_PERM_BBBB), _w1));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr00, _MM_PERM_CCCC), _w2));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr00, _MM_PERM_DDDD), _w3));
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr01, _MM_PERM_AAAA), _w4));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr01, _MM_PERM_BBBB), _w5));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr01, _MM_PERM_CCCC), _w6));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr01, _MM_PERM_DDDD), _w7));
#endif // __AVX512VNNI__

                    kptr += 256;
                }
            }
            for (; q + 7 < inch; q += 8)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    __m128i _r0;
                    if (elempack == 8)
                    {
                        _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                    }
                    else // if (elempack == 1)
                    {
                        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
                        _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
                    }

                    _r0 = _mm_cvtepi8_epi16(_r0);

                    __m512i _w01 = _mm512_load_si512((const __m512i*)kptr);
                    __m512i _w23 = _mm512_load_si512((const __m512i*)(kptr + 64));
                    __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                    __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                    __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                    __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));

                    // 01234567 -> 01010101 01010101 01010101 01010101
                    __m512i _rrrr0 = _mm512_broadcast_i32x4(_r0);

#if __AVX512VNNI__
                    _sum0 = _mm512_dpwssd_epi32(_sum0, _mm512_shuffle_epi32(_rrrr0, _MM_PERM_AAAA), _w0);
                    _sum1 = _mm512_dpwssd_epi32(_sum1, _mm512_shuffle_epi32(_rrrr0, _MM_PERM_BBBB), _w1);
                    _sum2 = _mm512_dpwssd_epi32(_sum2, _mm512_shuffle_epi32(_rrrr0, _MM_PERM_CCCC), _w2);
                    _sum3 = _mm512_dpwssd_epi32(_sum3, _mm512_shuffle_epi32(_rrrr0, _MM_PERM_DDDD), _w3);
#else
                    _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr0, _MM_PERM_AAAA), _w0));
                    _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr0, _MM_PERM_BBBB), _w1));
                    _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr0, _MM_PERM_CCCC), _w2));
                    _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_mm512_shuffle_epi32(_rrrr0, _MM_PERM_DDDD), _w3));
#endif // __AVX512VNNI__

                    kptr += 128;
                }
            }
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m512i _val = _mm512_broadcastd_epi32(_mm_setr_epi16(r0s[0], r0s[N], 0, 0, 0, 0, 0, 0));

                        __m512i _w = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)kptr));

#if __AVX512VNNI__
                        _sum0 = _mm512_dpwssd_epi32(_sum0, _val, _w);
#else
                        _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_val, _w));
#endif

                        kptr += 32;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m256i _val = _mm256_set1_epi16(r0s[0]);

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)kptr));

                        _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_val, _w)));

                        kptr += 16;
                    }
                }
            }

            _sum0 = _mm512_add_epi32(_sum0, _sum1);
            _sum2 = _mm512_add_epi32(_sum2, _sum3);
            _sum0 = _mm512_add_epi32(_sum0, _sum2);

            if (out_elempack == 16)
            {
                _mm512_store_si512((__m512i*)outptr, _sum0);
                outptr += 16;
            }
            if (out_elempack == 8)
            {
                _mm256_store_si256((__m256i*)outptr, _mm512_extracti32x8_epi32(_sum0, 0));
                _mm256_store_si256((__m256i*)(outptr + M), _mm512_extracti32x8_epi32(_sum0, 1));
                outptr += 8;
            }
            if (out_elempack == 4)
            {
                _mm_store_si128((__m128i*)outptr, _mm512_extracti32x4_epi32(_sum0, 0));
                _mm_store_si128((__m128i*)(outptr + M), _mm512_extracti32x4_epi32(_sum0, 1));
                _mm_store_si128((__m128i*)(outptr + M * 2), _mm512_extracti32x4_epi32(_sum0, 2));
                _mm_store_si128((__m128i*)(outptr + M * 3), _mm512_extracti32x4_epi32(_sum0, 3));
                outptr += 4;
            }
            if (out_elempack == 1)
            {
                __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(M));
                _mm512_i32scatter_epi32(outptr, _vindex, _sum0, sizeof(int));
                outptr += 1;
            }
        }
    }
    remain_outch_start += nn_outch * 16;
    nn_outch = (outch - remain_outch_start) / 8;
#else // __AVX512F__
    nn_outch = (outch - remain_outch_start) / 8;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __AVX512F__
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 8;

        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int N = bottom_blob.cstep * elempack;
        const int M = top_blob.cstep * out_elempack;

        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;
        for (; ij + 3 < outw * outh; ij += 4)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int i2 = (ij + 2) / outw;
            const int i3 = (ij + 3) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;
            const int j2 = (ij + 2) % outw;
            const int j3 = (ij + 3) % outw;

            __m256i _sum0 = _mm256_setzero_si256();
            __m256i _sum1 = _mm256_setzero_si256();
            __m256i _sum2 = _mm256_setzero_si256();
            __m256i _sum3 = _mm256_setzero_si256();

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8);
#else
            const signed char* kptr = weight_data_tm.channel(p / 8);
#endif

            int q = 0;
#if __AVX512F__
            {
                __m512i _sum00 = _mm512_setzero_si512();
                __m512i _sum11 = _mm512_setzero_si512();
                __m512i _sum22 = _mm512_setzero_si512();
                __m512i _sum33 = _mm512_setzero_si512();
                for (; q + 15 < inch; q += 16)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;
                    const signed char* r2 = bottom_blob.channel(q / elempack).row<const signed char>(i2 * stride_h) + j2 * stride_w * elempack;
                    const signed char* r3 = bottom_blob.channel(q / elempack).row<const signed char>(i3 * stride_h) + j3 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];
                        const signed char* r2s = r2 + space_ofs[k];
                        const signed char* r3s = r3 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        __m128i _r2;
                        __m128i _r3;
                        if (elempack == 16)
                        {
                            _r0 = _mm_load_si128((const __m128i*)r0s);
                            _r1 = _mm_load_si128((const __m128i*)r1s);
                            _r2 = _mm_load_si128((const __m128i*)r2s);
                            _r3 = _mm_load_si128((const __m128i*)r3s);
                        }
                        else if (elempack == 8)
                        {
                            __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                            __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                            __m128i _t2 = _mm_loadl_epi64((const __m128i*)r1s);
                            __m128i _t3 = _mm_loadl_epi64((const __m128i*)(r1s + N));
                            __m128i _t4 = _mm_loadl_epi64((const __m128i*)r2s);
                            __m128i _t5 = _mm_loadl_epi64((const __m128i*)(r2s + N));
                            __m128i _t6 = _mm_loadl_epi64((const __m128i*)r3s);
                            __m128i _t7 = _mm_loadl_epi64((const __m128i*)(r3s + N));
                            _r0 = _mm_unpacklo_epi64(_t0, _t1);
                            _r1 = _mm_unpacklo_epi64(_t2, _t3);
                            _r2 = _mm_unpacklo_epi64(_t4, _t5);
                            _r3 = _mm_unpacklo_epi64(_t6, _t7);
                        }
                        else // if (elempack == 1)
                        {
                            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                            _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r1s), 1));
                            _r2 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r2s), 1));
                            _r3 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r3s), 1));
                        }

                        __m256i _rr0 = _mm256_cvtepi8_epi16(_r0);
                        __m256i _rr1 = _mm256_cvtepi8_epi16(_r1);
                        __m256i _rr2 = _mm256_cvtepi8_epi16(_r2);
                        __m256i _rr3 = _mm256_cvtepi8_epi16(_r3);

                        __m512i _w01 = _mm512_load_si512((const __m512i*)kptr);
                        __m512i _w23 = _mm512_load_si512((const __m512i*)(kptr + 64));
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                        __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                        __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));

                        // 01234567 89abcdef -> 01010101 01010101 23232323 23232323
                        __m512i _rrr0 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr0), _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(1, 0, 3, 2)), 1);
                        __m512i _rrr1 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr1), _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(1, 0, 3, 2)), 1);
                        __m512i _rrr2 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr2), _mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(1, 0, 3, 2)), 1);
                        __m512i _rrr3 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr3), _mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(1, 0, 3, 2)), 1);

                        _rrr0 = _mm512_unpacklo_epi32(_rrr0, _rrr0);
                        _rrr1 = _mm512_unpacklo_epi32(_rrr1, _rrr1);
                        _rrr2 = _mm512_unpacklo_epi32(_rrr2, _rrr2);
                        _rrr3 = _mm512_unpacklo_epi32(_rrr3, _rrr3);

                        __m512i _rrr0l = _mm512_unpacklo_epi64(_rrr0, _rrr0);
                        __m512i _rrr0h = _mm512_unpackhi_epi64(_rrr0, _rrr0);
                        __m512i _rrr1l = _mm512_unpacklo_epi64(_rrr1, _rrr1);
                        __m512i _rrr1h = _mm512_unpackhi_epi64(_rrr1, _rrr1);
                        __m512i _rrr2l = _mm512_unpacklo_epi64(_rrr2, _rrr2);
                        __m512i _rrr2h = _mm512_unpackhi_epi64(_rrr2, _rrr2);
                        __m512i _rrr3l = _mm512_unpacklo_epi64(_rrr3, _rrr3);
                        __m512i _rrr3h = _mm512_unpackhi_epi64(_rrr3, _rrr3);

#if __AVX512VNNI__
                        _sum00 = _mm512_dpwssd_epi32(_sum00, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum11 = _mm512_dpwssd_epi32(_sum11, _mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum22 = _mm512_dpwssd_epi32(_sum22, _mm512_shuffle_i32x4(_rrr2l, _rrr2h, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum33 = _mm512_dpwssd_epi32(_sum33, _mm512_shuffle_i32x4(_rrr3l, _rrr3h, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum00 = _mm512_dpwssd_epi32(_sum00, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
                        _sum11 = _mm512_dpwssd_epi32(_sum11, _mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
                        _sum22 = _mm512_dpwssd_epi32(_sum22, _mm512_shuffle_i32x4(_rrr2l, _rrr2h, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
                        _sum33 = _mm512_dpwssd_epi32(_sum33, _mm512_shuffle_i32x4(_rrr3l, _rrr3h, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
                        _sum00 = _mm512_dpwssd_epi32(_sum00, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(1, 1, 1, 1)), _w2);
                        _sum11 = _mm512_dpwssd_epi32(_sum11, _mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(1, 1, 1, 1)), _w2);
                        _sum22 = _mm512_dpwssd_epi32(_sum22, _mm512_shuffle_i32x4(_rrr2l, _rrr2h, _MM_SHUFFLE(1, 1, 1, 1)), _w2);
                        _sum33 = _mm512_dpwssd_epi32(_sum33, _mm512_shuffle_i32x4(_rrr3l, _rrr3h, _MM_SHUFFLE(1, 1, 1, 1)), _w2);
                        _sum00 = _mm512_dpwssd_epi32(_sum00, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
                        _sum11 = _mm512_dpwssd_epi32(_sum11, _mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
                        _sum22 = _mm512_dpwssd_epi32(_sum22, _mm512_shuffle_i32x4(_rrr2l, _rrr2h, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
                        _sum33 = _mm512_dpwssd_epi32(_sum33, _mm512_shuffle_i32x4(_rrr3l, _rrr3h, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
#else
                        _sum00 = _mm512_add_epi32(_sum00, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum11 = _mm512_add_epi32(_sum11, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum22 = _mm512_add_epi32(_sum22, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr2l, _rrr2h, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum33 = _mm512_add_epi32(_sum33, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr3l, _rrr3h, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum00 = _mm512_add_epi32(_sum00, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
                        _sum11 = _mm512_add_epi32(_sum11, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
                        _sum22 = _mm512_add_epi32(_sum22, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr2l, _rrr2h, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
                        _sum33 = _mm512_add_epi32(_sum33, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr3l, _rrr3h, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
                        _sum00 = _mm512_add_epi32(_sum00, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(1, 1, 1, 1)), _w2));
                        _sum11 = _mm512_add_epi32(_sum11, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(1, 1, 1, 1)), _w2));
                        _sum22 = _mm512_add_epi32(_sum22, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr2l, _rrr2h, _MM_SHUFFLE(1, 1, 1, 1)), _w2));
                        _sum33 = _mm512_add_epi32(_sum33, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr3l, _rrr3h, _MM_SHUFFLE(1, 1, 1, 1)), _w2));
                        _sum00 = _mm512_add_epi32(_sum00, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
                        _sum11 = _mm512_add_epi32(_sum11, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
                        _sum22 = _mm512_add_epi32(_sum22, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr2l, _rrr2h, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
                        _sum33 = _mm512_add_epi32(_sum33, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr3l, _rrr3h, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
#endif // __AVX512VNNI__

                        kptr += 128;
                    }
                }
                _sum0 = _mm256_add_epi32(_sum0, _mm512_extracti64x4_epi64(_sum00, 0));
                _sum0 = _mm256_add_epi32(_sum0, _mm512_extracti64x4_epi64(_sum00, 1));
                _sum1 = _mm256_add_epi32(_sum1, _mm512_extracti64x4_epi64(_sum11, 0));
                _sum1 = _mm256_add_epi32(_sum1, _mm512_extracti64x4_epi64(_sum11, 1));
                _sum2 = _mm256_add_epi32(_sum2, _mm512_extracti64x4_epi64(_sum22, 0));
                _sum2 = _mm256_add_epi32(_sum2, _mm512_extracti64x4_epi64(_sum22, 1));
                _sum3 = _mm256_add_epi32(_sum3, _mm512_extracti64x4_epi64(_sum33, 0));
                _sum3 = _mm256_add_epi32(_sum3, _mm512_extracti64x4_epi64(_sum33, 1));
            }
#endif // __AVX512F__
            for (; q + 7 < inch; q += 8)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;
                const signed char* r2 = bottom_blob.channel(q / elempack).row<const signed char>(i2 * stride_h) + j2 * stride_w * elempack;
                const signed char* r3 = bottom_blob.channel(q / elempack).row<const signed char>(i3 * stride_h) + j3 * stride_w * elempack;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    const signed char* r2s = r2 + space_ofs[k];
                    const signed char* r3s = r3 + space_ofs[k];

                    __m128i _r0;
                    __m128i _r1;
                    __m128i _r2;
                    __m128i _r3;
                    if (elempack == 8)
                    {
                        _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                        _r1 = _mm_loadl_epi64((const __m128i*)r1s);
                        _r2 = _mm_loadl_epi64((const __m128i*)r2s);
                        _r3 = _mm_loadl_epi64((const __m128i*)r3s);
                    }
                    else // if (elempack == 1)
                    {
#if __AVX2__
                        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                        _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
                        _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1));
                        _r2 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r2s, _vindex, 1));
                        _r3 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r3s, _vindex, 1));
#else
                        __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                        __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                        __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1), _sindex88);
                        __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1), _sindex88);
                        __m256i _val2_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r2s, _vindex, 1), _sindex88);
                        __m256i _val3_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r3s, _vindex, 1), _sindex88);
                        _r0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val0_32, 0), _mm256_extracti128_si256(_val0_32, 1));
                        _r1 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val1_32, 0), _mm256_extracti128_si256(_val1_32, 1));
                        _r2 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val2_32, 0), _mm256_extracti128_si256(_val2_32, 1));
                        _r3 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val3_32, 0), _mm256_extracti128_si256(_val3_32, 1));
#endif // __AVX512F__
#else
                        _r0 = _mm_setr_epi8(r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                        _r1 = _mm_setr_epi8(r1s[0], r1s[N], r1s[N * 2], r1s[N * 3], r1s[N * 4], r1s[N * 5], r1s[N * 6], r1s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                        _r2 = _mm_setr_epi8(r2s[0], r2s[N], r2s[N * 2], r2s[N * 3], r2s[N * 4], r2s[N * 5], r2s[N * 6], r2s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                        _r3 = _mm_setr_epi8(r3s[0], r3s[N], r3s[N * 2], r3s[N * 3], r3s[N * 4], r3s[N * 5], r3s[N * 6], r3s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
#endif // __AVX2__
                    }

                    _r0 = _mm_cvtepi8_epi16(_r0);
                    _r1 = _mm_cvtepi8_epi16(_r1);
                    _r2 = _mm_cvtepi8_epi16(_r2);
                    _r3 = _mm_cvtepi8_epi16(_r3);

                    __m256i _w01 = _mm256_load_si256((const __m256i*)kptr);
                    __m256i _w23 = _mm256_load_si256((const __m256i*)(kptr + 32));
                    __m256i _w0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w01, 0));
                    __m256i _w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w01, 1));
                    __m256i _w2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w23, 0));
                    __m256i _w3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w23, 1));

                    // 01234567 -> 01010101 01010101
                    __m256i _rr0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r0, 1);
                    __m256i _rr1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r1), _r1, 1);
                    __m256i _rr2 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r2), _r2, 1);
                    __m256i _rr3 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r3), _r3, 1);

#if __AVXVNNI__ || __AVX512VNNI__
                    _sum0 = _mm256_dpwssd_epi32(_sum0, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                    _sum1 = _mm256_dpwssd_epi32(_sum1, _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                    _sum2 = _mm256_dpwssd_epi32(_sum2, _mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                    _sum3 = _mm256_dpwssd_epi32(_sum3, _mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                    _sum0 = _mm256_dpwssd_epi32(_sum0, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(1, 1, 1, 1)), _w1);
                    _sum1 = _mm256_dpwssd_epi32(_sum1, _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(1, 1, 1, 1)), _w1);
                    _sum2 = _mm256_dpwssd_epi32(_sum2, _mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(1, 1, 1, 1)), _w1);
                    _sum3 = _mm256_dpwssd_epi32(_sum3, _mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(1, 1, 1, 1)), _w1);
                    _sum0 = _mm256_dpwssd_epi32(_sum0, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(2, 2, 2, 2)), _w2);
                    _sum1 = _mm256_dpwssd_epi32(_sum1, _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(2, 2, 2, 2)), _w2);
                    _sum2 = _mm256_dpwssd_epi32(_sum2, _mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(2, 2, 2, 2)), _w2);
                    _sum3 = _mm256_dpwssd_epi32(_sum3, _mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(2, 2, 2, 2)), _w2);
                    _sum0 = _mm256_dpwssd_epi32(_sum0, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
                    _sum1 = _mm256_dpwssd_epi32(_sum1, _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
                    _sum2 = _mm256_dpwssd_epi32(_sum2, _mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
                    _sum3 = _mm256_dpwssd_epi32(_sum3, _mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
#else
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                    _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                    _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                    _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                    _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                    _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                    _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
                    _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
                    _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
#endif

                    kptr += 64;
                }
            }
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;
                const signed char* r2 = bottom_blob.channel(q).row<const signed char>(i2 * stride_h) + j2 * stride_w;
                const signed char* r3 = bottom_blob.channel(q).row<const signed char>(i3 * stride_h) + j3 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    const signed char* r2s = r2 + space_ofs[k];
                    const signed char* r3s = r3 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _r0 = _mm_setr_epi16(r0s[0], r0s[N], 0, 0, 0, 0, 0, 0);
                        __m128i _r1 = _mm_setr_epi16(r1s[0], r1s[N], 0, 0, 0, 0, 0, 0);
                        __m128i _r2 = _mm_setr_epi16(r2s[0], r2s[N], 0, 0, 0, 0, 0, 0);
                        __m128i _r3 = _mm_setr_epi16(r3s[0], r3s[N], 0, 0, 0, 0, 0, 0);
                        __m256i _rr0 = _mm256_broadcastd_epi32(_r0);
                        __m256i _rr1 = _mm256_broadcastd_epi32(_r1);
                        __m256i _rr2 = _mm256_broadcastd_epi32(_r2);
                        __m256i _rr3 = _mm256_broadcastd_epi32(_r3);

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)kptr));

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum0 = _mm256_dpwssd_epi32(_sum0, _rr0, _w);
                        _sum1 = _mm256_dpwssd_epi32(_sum1, _rr1, _w);
                        _sum2 = _mm256_dpwssd_epi32(_sum2, _rr2, _w);
                        _sum3 = _mm256_dpwssd_epi32(_sum3, _rr3, _w);
#else
                        _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_rr0, _w));
                        _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_rr1, _w));
                        _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_rr2, _w));
                        _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_rr3, _w));
#endif

                        kptr += 16;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;
                const signed char* r2 = bottom_blob.channel(q).row<const signed char>(i2 * stride_h) + j2 * stride_w;
                const signed char* r3 = bottom_blob.channel(q).row<const signed char>(i3 * stride_h) + j3 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    const signed char* r2s = r2 + space_ofs[k];
                    const signed char* r3s = r3 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _r0 = _mm_set1_epi16(r0s[0]);
                        __m128i _r1 = _mm_set1_epi16(r1s[0]);
                        __m128i _r2 = _mm_set1_epi16(r2s[0]);
                        __m128i _r3 = _mm_set1_epi16(r3s[0]);
                        __m128i _w = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)kptr));
                        __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_r0, _w));
                        __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_r1, _w));
                        __m256i _s2 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_r2, _w));
                        __m256i _s3 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_r3, _w));
                        _sum0 = _mm256_add_epi32(_sum0, _s0);
                        _sum1 = _mm256_add_epi32(_sum1, _s1);
                        _sum2 = _mm256_add_epi32(_sum2, _s2);
                        _sum3 = _mm256_add_epi32(_sum3, _s3);

                        kptr += 8;
                    }
                }
            }

            if (out_elempack == 8)
            {
                _mm256_store_si256((__m256i*)outptr, _sum0);
                _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
                _mm256_store_si256((__m256i*)(outptr + 16), _sum2);
                _mm256_store_si256((__m256i*)(outptr + 24), _sum3);
                outptr += 32;
            }
            if (out_elempack == 4)
            {
                _mm_store_si128((__m128i*)outptr, _mm256_extracti128_si256(_sum0, 0));
                _mm_store_si128((__m128i*)(outptr + M), _mm256_extracti128_si256(_sum0, 1));
                _mm_store_si128((__m128i*)(outptr + 4), _mm256_extracti128_si256(_sum1, 0));
                _mm_store_si128((__m128i*)(outptr + M + 4), _mm256_extracti128_si256(_sum1, 1));
                _mm_store_si128((__m128i*)(outptr + 8), _mm256_extracti128_si256(_sum2, 0));
                _mm_store_si128((__m128i*)(outptr + M + 8), _mm256_extracti128_si256(_sum2, 1));
                _mm_store_si128((__m128i*)(outptr + 12), _mm256_extracti128_si256(_sum3, 0));
                _mm_store_si128((__m128i*)(outptr + M + 12), _mm256_extracti128_si256(_sum3, 1));
                outptr += 16;
            }
            if (out_elempack == 1)
            {
#if __AVX512F__
                __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(M));
                _mm256_i32scatter_epi32(outptr, _vindex, _sum0, sizeof(int));
                _mm256_i32scatter_epi32(outptr + 1, _vindex, _sum1, sizeof(int));
                _mm256_i32scatter_epi32(outptr + 2, _vindex, _sum2, sizeof(int));
                _mm256_i32scatter_epi32(outptr + 3, _vindex, _sum3, sizeof(int));
#else
                int sum0[8];
                int sum1[8];
                int sum2[8];
                int sum3[8];
                _mm256_storeu_si256((__m256i*)sum0, _sum0);
                _mm256_storeu_si256((__m256i*)sum1, _sum1);
                _mm256_storeu_si256((__m256i*)sum2, _sum2);
                _mm256_storeu_si256((__m256i*)sum3, _sum3);

                outptr[0] = sum0[0];
                outptr[1] = sum1[0];
                outptr[2] = sum2[0];
                outptr[3] = sum3[0];
                outptr[M] = sum0[1];
                outptr[M + 1] = sum1[1];
                outptr[M + 2] = sum2[1];
                outptr[M + 3] = sum3[1];
                outptr[M * 2] = sum0[2];
                outptr[M * 2 + 1] = sum1[2];
                outptr[M * 2 + 2] = sum2[2];
                outptr[M * 2 + 3] = sum3[2];
                outptr[M * 3] = sum0[3];
                outptr[M * 3 + 1] = sum1[3];
                outptr[M * 3 + 2] = sum2[3];
                outptr[M * 3 + 3] = sum3[3];
                outptr[M * 4] = sum0[4];
                outptr[M * 4 + 1] = sum1[4];
                outptr[M * 4 + 2] = sum2[4];
                outptr[M * 4 + 3] = sum3[4];
                outptr[M * 5] = sum0[5];
                outptr[M * 5 + 1] = sum1[5];
                outptr[M * 5 + 2] = sum2[5];
                outptr[M * 5 + 3] = sum3[5];
                outptr[M * 6] = sum0[6];
                outptr[M * 6 + 1] = sum1[6];
                outptr[M * 6 + 2] = sum2[6];
                outptr[M * 6 + 3] = sum3[6];
                outptr[M * 7] = sum0[7];
                outptr[M * 7 + 1] = sum1[7];
                outptr[M * 7 + 2] = sum2[7];
                outptr[M * 7 + 3] = sum3[7];
#endif // __AVX512F__
                outptr += 4;
            }
        }
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            __m256i _sum0 = _mm256_setzero_si256();
            __m256i _sum1 = _mm256_setzero_si256();
            __m256i _sum2 = _mm256_setzero_si256();
            __m256i _sum3 = _mm256_setzero_si256();

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8);
#else
            const signed char* kptr = weight_data_tm.channel(p / 8);
#endif

            int q = 0;
#if __AVX512F__
            {
                __m512i _sum00 = _mm512_setzero_si512();
                __m512i _sum11 = _mm512_setzero_si512();
                __m512i _sum22 = _mm512_setzero_si512();
                __m512i _sum33 = _mm512_setzero_si512();
                for (; q + 15 < inch; q += 16)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        if (elempack == 16)
                        {
                            _r0 = _mm_load_si128((const __m128i*)r0s);
                            _r1 = _mm_load_si128((const __m128i*)r1s);
                        }
                        else if (elempack == 8)
                        {
                            __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                            __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                            __m128i _t2 = _mm_loadl_epi64((const __m128i*)r1s);
                            __m128i _t3 = _mm_loadl_epi64((const __m128i*)(r1s + N));
                            _r0 = _mm_unpacklo_epi64(_t0, _t1);
                            _r1 = _mm_unpacklo_epi64(_t2, _t3);
                        }
                        else // if (elempack == 1)
                        {
                            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                            _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r1s), 1));
                        }

                        __m256i _rr0 = _mm256_cvtepi8_epi16(_r0);
                        __m256i _rr1 = _mm256_cvtepi8_epi16(_r1);

                        __m512i _w01 = _mm512_load_si512((const __m512i*)kptr);
                        __m512i _w23 = _mm512_load_si512((const __m512i*)(kptr + 64));
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                        __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                        __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));

                        // 01234567 89abcdef -> 01010101 01010101 23232323 23232323
                        __m512i _rrr0 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr0), _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(1, 0, 3, 2)), 1);
                        __m512i _rrr1 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr1), _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(1, 0, 3, 2)), 1);

                        _rrr0 = _mm512_unpacklo_epi32(_rrr0, _rrr0);
                        _rrr1 = _mm512_unpacklo_epi32(_rrr1, _rrr1);

                        __m512i _rrr0l = _mm512_unpacklo_epi64(_rrr0, _rrr0);
                        __m512i _rrr0h = _mm512_unpackhi_epi64(_rrr0, _rrr0);
                        __m512i _rrr1l = _mm512_unpacklo_epi64(_rrr1, _rrr1);
                        __m512i _rrr1h = _mm512_unpackhi_epi64(_rrr1, _rrr1);

#if __AVX512VNNI__
                        _sum00 = _mm512_dpwssd_epi32(_sum00, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum11 = _mm512_dpwssd_epi32(_sum11, _mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum22 = _mm512_dpwssd_epi32(_sum22, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
                        _sum33 = _mm512_dpwssd_epi32(_sum33, _mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
                        _sum00 = _mm512_dpwssd_epi32(_sum00, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(1, 1, 1, 1)), _w2);
                        _sum11 = _mm512_dpwssd_epi32(_sum11, _mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(1, 1, 1, 1)), _w2);
                        _sum22 = _mm512_dpwssd_epi32(_sum22, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
                        _sum33 = _mm512_dpwssd_epi32(_sum33, _mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
#else
                        _sum00 = _mm512_add_epi32(_sum00, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum11 = _mm512_add_epi32(_sum11, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum22 = _mm512_add_epi32(_sum22, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
                        _sum33 = _mm512_add_epi32(_sum33, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
                        _sum00 = _mm512_add_epi32(_sum00, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(1, 1, 1, 1)), _w2));
                        _sum11 = _mm512_add_epi32(_sum11, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(1, 1, 1, 1)), _w2));
                        _sum22 = _mm512_add_epi32(_sum22, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
                        _sum33 = _mm512_add_epi32(_sum33, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
#endif // __AVX512VNNI__

                        kptr += 128;
                    }
                }
                _sum0 = _mm256_add_epi32(_sum0, _mm512_extracti64x4_epi64(_sum00, 0));
                _sum0 = _mm256_add_epi32(_sum0, _mm512_extracti64x4_epi64(_sum00, 1));
                _sum1 = _mm256_add_epi32(_sum1, _mm512_extracti64x4_epi64(_sum11, 0));
                _sum1 = _mm256_add_epi32(_sum1, _mm512_extracti64x4_epi64(_sum11, 1));
                _sum2 = _mm256_add_epi32(_sum2, _mm512_extracti64x4_epi64(_sum22, 0));
                _sum2 = _mm256_add_epi32(_sum2, _mm512_extracti64x4_epi64(_sum22, 1));
                _sum3 = _mm256_add_epi32(_sum3, _mm512_extracti64x4_epi64(_sum33, 0));
                _sum3 = _mm256_add_epi32(_sum3, _mm512_extracti64x4_epi64(_sum33, 1));
            }
#endif // __AVX512F__
            for (; q + 7 < inch; q += 8)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    __m128i _r0;
                    __m128i _r1;
                    if (elempack == 8)
                    {
                        _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                        _r1 = _mm_loadl_epi64((const __m128i*)r1s);
                    }
                    else // if (elempack == 1)
                    {
#if __AVX2__
                        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                        _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
                        _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1));
#else
                        __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                        __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                        __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1), _sindex88);
                        __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1), _sindex88);
                        _r0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val0_32, 0), _mm256_extracti128_si256(_val0_32, 1));
                        _r1 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val1_32, 0), _mm256_extracti128_si256(_val1_32, 1));
#endif // __AVX512F__
#else
                        _r0 = _mm_setr_epi8(r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                        _r1 = _mm_setr_epi8(r1s[0], r1s[N], r1s[N * 2], r1s[N * 3], r1s[N * 4], r1s[N * 5], r1s[N * 6], r1s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
#endif // __AVX2__
                    }

                    _r0 = _mm_cvtepi8_epi16(_r0);
                    _r1 = _mm_cvtepi8_epi16(_r1);

                    __m256i _w01 = _mm256_load_si256((const __m256i*)kptr);
                    __m256i _w23 = _mm256_load_si256((const __m256i*)(kptr + 32));
                    __m256i _w0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w01, 0));
                    __m256i _w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w01, 1));
                    __m256i _w2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w23, 0));
                    __m256i _w3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w23, 1));

                    // 01234567 -> 01010101 01010101
                    __m256i _rr0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r0, 1);
                    __m256i _rr1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r1), _r1, 1);

#if __AVXVNNI__ || __AVX512VNNI__
                    _sum0 = _mm256_dpwssd_epi32(_sum0, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                    _sum1 = _mm256_dpwssd_epi32(_sum1, _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                    _sum2 = _mm256_dpwssd_epi32(_sum2, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(1, 1, 1, 1)), _w1);
                    _sum3 = _mm256_dpwssd_epi32(_sum3, _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(1, 1, 1, 1)), _w1);
                    _sum0 = _mm256_dpwssd_epi32(_sum0, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(2, 2, 2, 2)), _w2);
                    _sum1 = _mm256_dpwssd_epi32(_sum1, _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(2, 2, 2, 2)), _w2);
                    _sum2 = _mm256_dpwssd_epi32(_sum2, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
                    _sum3 = _mm256_dpwssd_epi32(_sum3, _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
#else
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                    _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                    _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                    _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                    _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                    _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
                    _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
#endif

                    kptr += 64;
                }
            }
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _r0 = _mm_setr_epi16(r0s[0], r0s[N], 0, 0, 0, 0, 0, 0);
                        __m128i _r1 = _mm_setr_epi16(r1s[0], r1s[N], 0, 0, 0, 0, 0, 0);
                        __m256i _rr0 = _mm256_broadcastd_epi32(_r0);
                        __m256i _rr1 = _mm256_broadcastd_epi32(_r1);

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)kptr));

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum0 = _mm256_dpwssd_epi32(_sum0, _rr0, _w);
                        _sum1 = _mm256_dpwssd_epi32(_sum1, _rr1, _w);
#else
                        _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_rr0, _w));
                        _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_rr1, _w));
#endif

                        kptr += 16;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _r0 = _mm_set1_epi16(r0s[0]);
                        __m128i _r1 = _mm_set1_epi16(r1s[0]);
                        __m128i _w = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)kptr));
                        __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_r0, _w));
                        __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_r1, _w));
                        _sum0 = _mm256_add_epi32(_sum0, _s0);
                        _sum1 = _mm256_add_epi32(_sum1, _s1);

                        kptr += 8;
                    }
                }
            }

            _sum0 = _mm256_add_epi32(_sum0, _sum2);
            _sum1 = _mm256_add_epi32(_sum1, _sum3);

            if (out_elempack == 8)
            {
                _mm256_store_si256((__m256i*)outptr, _sum0);
                _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
                outptr += 16;
            }
            if (out_elempack == 4)
            {
                _mm_store_si128((__m128i*)outptr, _mm256_extracti128_si256(_sum0, 0));
                _mm_store_si128((__m128i*)(outptr + M), _mm256_extracti128_si256(_sum0, 1));
                _mm_store_si128((__m128i*)(outptr + 4), _mm256_extracti128_si256(_sum1, 0));
                _mm_store_si128((__m128i*)(outptr + M + 4), _mm256_extracti128_si256(_sum1, 1));
                outptr += 8;
            }
            if (out_elempack == 1)
            {
#if __AVX512F__
                __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(M));
                _mm256_i32scatter_epi32(outptr, _vindex, _sum0, sizeof(int));
                _mm256_i32scatter_epi32(outptr + 1, _vindex, _sum1, sizeof(int));
#else
                int sum0[8];
                int sum1[8];
                _mm256_storeu_si256((__m256i*)sum0, _sum0);
                _mm256_storeu_si256((__m256i*)sum1, _sum1);

                outptr[0] = sum0[0];
                outptr[1] = sum1[0];
                outptr[M] = sum0[1];
                outptr[M + 1] = sum1[1];
                outptr[M * 2] = sum0[2];
                outptr[M * 2 + 1] = sum1[2];
                outptr[M * 3] = sum0[3];
                outptr[M * 3 + 1] = sum1[3];
                outptr[M * 4] = sum0[4];
                outptr[M * 4 + 1] = sum1[4];
                outptr[M * 5] = sum0[5];
                outptr[M * 5 + 1] = sum1[5];
                outptr[M * 6] = sum0[6];
                outptr[M * 6 + 1] = sum1[6];
                outptr[M * 7] = sum0[7];
                outptr[M * 7 + 1] = sum1[7];
#endif // __AVX512F__
                outptr += 2;
            }
        }
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            __m256i _sum0 = _mm256_setzero_si256();

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8);
#else
            const signed char* kptr = weight_data_tm.channel(p / 8);
#endif

            int q = 0;
#if __AVX512F__
            {
                __m512i _sum00 = _mm512_setzero_si512();
                __m512i _sum11 = _mm512_setzero_si512();
                __m512i _sum22 = _mm512_setzero_si512();
                __m512i _sum33 = _mm512_setzero_si512();
                for (; q + 15 < inch; q += 16)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];

                        __m128i _r0;
                        if (elempack == 16)
                        {
                            _r0 = _mm_load_si128((const __m128i*)r0s);
                        }
                        else if (elempack == 8)
                        {
                            __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                            __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                            _r0 = _mm_unpacklo_epi64(_t0, _t1);
                        }
                        else // if (elempack == 1)
                        {
                            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                        }

                        __m256i _rr0 = _mm256_cvtepi8_epi16(_r0);

                        __m512i _w01 = _mm512_load_si512((const __m512i*)kptr);
                        __m512i _w23 = _mm512_load_si512((const __m512i*)(kptr + 64));
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                        __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                        __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));

                        // 01234567 89abcdef -> 01010101 01010101 23232323 23232323
                        __m512i _rrr0 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr0), _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(1, 0, 3, 2)), 1);

                        _rrr0 = _mm512_unpacklo_epi32(_rrr0, _rrr0);

                        __m512i _rrr0l = _mm512_unpacklo_epi64(_rrr0, _rrr0);
                        __m512i _rrr0h = _mm512_unpackhi_epi64(_rrr0, _rrr0);

#if __AVX512VNNI__
                        _sum00 = _mm512_dpwssd_epi32(_sum00, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum11 = _mm512_dpwssd_epi32(_sum11, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
                        _sum22 = _mm512_dpwssd_epi32(_sum22, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(1, 1, 1, 1)), _w2);
                        _sum33 = _mm512_dpwssd_epi32(_sum33, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
#else
                        _sum00 = _mm512_add_epi32(_sum00, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum11 = _mm512_add_epi32(_sum11, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
                        _sum22 = _mm512_add_epi32(_sum22, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(1, 1, 1, 1)), _w2));
                        _sum33 = _mm512_add_epi32(_sum33, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
#endif // __AVX512VNNI__

                        kptr += 128;
                    }
                }
                _sum00 = _mm512_add_epi32(_sum00, _sum11);
                _sum22 = _mm512_add_epi32(_sum22, _sum33);
                _sum00 = _mm512_add_epi32(_sum00, _sum22);
                _sum0 = _mm256_add_epi32(_sum0, _mm512_extracti64x4_epi64(_sum00, 0));
                _sum0 = _mm256_add_epi32(_sum0, _mm512_extracti64x4_epi64(_sum00, 1));
            }
#endif // __AVX512F__
            {
                __m256i _sum1 = _mm256_setzero_si256();
                __m256i _sum2 = _mm256_setzero_si256();
                __m256i _sum3 = _mm256_setzero_si256();
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];

                        __m128i _r0;
                        if (elempack == 8)
                        {
                            _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                        }
                        else // if (elempack == 1)
                        {
#if __AVX2__
                            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1), _sindex88);
                            _r0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val32, 0), _mm256_extracti128_si256(_val32, 1));
#endif // __AVX512F__
#else
                            _r0 = _mm_setr_epi8(r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
#endif // __AVX2__
                        }

                        _r0 = _mm_cvtepi8_epi16(_r0);

                        __m256i _w01 = _mm256_load_si256((const __m256i*)kptr);
                        __m256i _w23 = _mm256_load_si256((const __m256i*)(kptr + 32));
                        __m256i _w0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w01, 0));
                        __m256i _w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w01, 1));
                        __m256i _w2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w23, 0));
                        __m256i _w3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w23, 1));

                        // 01234567 -> 01010101 01010101
                        __m256i _rr0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r0, 1);

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum0 = _mm256_dpwssd_epi32(_sum0, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum1 = _mm256_dpwssd_epi32(_sum1, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(1, 1, 1, 1)), _w1);
                        _sum2 = _mm256_dpwssd_epi32(_sum2, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(2, 2, 2, 2)), _w2);
                        _sum3 = _mm256_dpwssd_epi32(_sum3, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(3, 3, 3, 3)), _w3);
#else
                        _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                        _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                        _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
#endif

                        kptr += 64;
                    }
                }
                _sum0 = _mm256_add_epi32(_sum0, _sum1);
                _sum2 = _mm256_add_epi32(_sum2, _sum3);
                _sum0 = _mm256_add_epi32(_sum0, _sum2);
            }
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _val0 = _mm_setr_epi16(r0s[0], r0s[N], 0, 0, 0, 0, 0, 0);
                        __m256i _val = _mm256_broadcastd_epi32(_val0);

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)kptr));

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum0 = _mm256_dpwssd_epi32(_sum0, _val, _w);
#else
                        _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_val, _w));
#endif

                        kptr += 16;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _val = _mm_set1_epi16(r0s[0]);
                        __m128i _w = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)kptr));
                        __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_val, _w));
                        _sum0 = _mm256_add_epi32(_sum0, _s0);

                        kptr += 8;
                    }
                }
            }

            if (out_elempack == 8)
            {
                _mm256_store_si256((__m256i*)outptr, _sum0);
                outptr += 8;
            }
            if (out_elempack == 4)
            {
                _mm_store_si128((__m128i*)outptr, _mm256_extracti128_si256(_sum0, 0));
                _mm_store_si128((__m128i*)(outptr + M), _mm256_extracti128_si256(_sum0, 1));
                outptr += 4;
            }
            if (out_elempack == 1)
            {
#if __AVX512F__
                __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(M));
                _mm256_i32scatter_epi32(outptr, _vindex, _sum0, sizeof(int));
#else
                int sum[8];
                _mm256_storeu_si256((__m256i*)sum, _sum0);

                outptr[0] = sum[0];
                outptr[M] = sum[1];
                outptr[M * 2] = sum[2];
                outptr[M * 3] = sum[3];
                outptr[M * 4] = sum[4];
                outptr[M * 5] = sum[5];
                outptr[M * 6] = sum[6];
                outptr[M * 7] = sum[7];
#endif // __AVX512F__
                outptr += 1;
            }
        }
    }
    remain_outch_start += nn_outch * 8;
    nn_outch = (outch - remain_outch_start) / 4;
#else // __AVX2__
    nn_outch = (outch - remain_outch_start) / 4;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __AVX2__
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 4;

        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int N = bottom_blob.cstep * elempack;
        const int M = top_blob.cstep * out_elempack;

        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;
        for (; ij + 3 < outw * outh; ij += 4)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int i2 = (ij + 2) / outw;
            const int i3 = (ij + 3) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;
            const int j2 = (ij + 2) % outw;
            const int j3 = (ij + 3) % outw;

            __m128i _sum0 = _mm_setzero_si128();
            __m128i _sum1 = _mm_setzero_si128();
            __m128i _sum2 = _mm_setzero_si128();
            __m128i _sum3 = _mm_setzero_si128();

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4);
#elif __AVX2__
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);
#else
            const signed char* kptr = weight_data_tm.channel(p / 4);
#endif

            int q = 0;
#if __AVX512F__
            {
                __m512i _sum00 = _mm512_setzero_si512();
                __m512i _sum11 = _mm512_setzero_si512();
                __m512i _sum22 = _mm512_setzero_si512();
                __m512i _sum33 = _mm512_setzero_si512();
                for (; q + 15 < inch; q += 16)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;
                    const signed char* r2 = bottom_blob.channel(q / elempack).row<const signed char>(i2 * stride_h) + j2 * stride_w * elempack;
                    const signed char* r3 = bottom_blob.channel(q / elempack).row<const signed char>(i3 * stride_h) + j3 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];
                        const signed char* r2s = r2 + space_ofs[k];
                        const signed char* r3s = r3 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        __m128i _r2;
                        __m128i _r3;
                        if (elempack == 16)
                        {
                            _r0 = _mm_load_si128((const __m128i*)r0s);
                            _r1 = _mm_load_si128((const __m128i*)r1s);
                            _r2 = _mm_load_si128((const __m128i*)r2s);
                            _r3 = _mm_load_si128((const __m128i*)r3s);
                        }
                        else if (elempack == 8)
                        {
                            __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                            __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                            __m128i _t2 = _mm_loadl_epi64((const __m128i*)r1s);
                            __m128i _t3 = _mm_loadl_epi64((const __m128i*)(r1s + N));
                            __m128i _t4 = _mm_loadl_epi64((const __m128i*)r2s);
                            __m128i _t5 = _mm_loadl_epi64((const __m128i*)(r2s + N));
                            __m128i _t6 = _mm_loadl_epi64((const __m128i*)r3s);
                            __m128i _t7 = _mm_loadl_epi64((const __m128i*)(r3s + N));
                            _r0 = _mm_unpacklo_epi64(_t0, _t1);
                            _r1 = _mm_unpacklo_epi64(_t2, _t3);
                            _r2 = _mm_unpacklo_epi64(_t4, _t5);
                            _r3 = _mm_unpacklo_epi64(_t6, _t7);
                        }
                        else // if (elempack == 1)
                        {
                            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                            _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r1s), 1));
                            _r2 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r2s), 1));
                            _r3 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r3s), 1));
                        }

                        __m256i _rr0 = _mm256_cvtepi8_epi16(_r0);
                        __m256i _rr1 = _mm256_cvtepi8_epi16(_r1);
                        __m256i _rr2 = _mm256_cvtepi8_epi16(_r2);
                        __m256i _rr3 = _mm256_cvtepi8_epi16(_r3);

                        __m512i _w = _mm512_load_si512((const __m512i*)kptr);
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w, 1));

                        // 01234567 89abcdef -> 01010101 23232323 45454545 67676767
                        _rr0 = _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(3, 1, 2, 0));
                        _rr1 = _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(3, 1, 2, 0));
                        _rr2 = _mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(3, 1, 2, 0));
                        _rr3 = _mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(3, 1, 2, 0));

                        __m512i _rrr0 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr0), _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(1, 0, 3, 2)), 1);
                        __m512i _rrr1 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr1), _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(1, 0, 3, 2)), 1);
                        __m512i _rrr2 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr2), _mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(1, 0, 3, 2)), 1);
                        __m512i _rrr3 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr3), _mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(1, 0, 3, 2)), 1);

                        _rrr0 = _mm512_unpacklo_epi32(_rrr0, _rrr0);
                        _rrr1 = _mm512_unpacklo_epi32(_rrr1, _rrr1);
                        _rrr2 = _mm512_unpacklo_epi32(_rrr2, _rrr2);
                        _rrr3 = _mm512_unpacklo_epi32(_rrr3, _rrr3);

                        __m512i _rrr0l = _mm512_unpacklo_epi64(_rrr0, _rrr0);
                        __m512i _rrr0h = _mm512_unpackhi_epi64(_rrr0, _rrr0);
                        __m512i _rrr1l = _mm512_unpacklo_epi64(_rrr1, _rrr1);
                        __m512i _rrr1h = _mm512_unpackhi_epi64(_rrr1, _rrr1);
                        __m512i _rrr2l = _mm512_unpacklo_epi64(_rrr2, _rrr2);
                        __m512i _rrr2h = _mm512_unpackhi_epi64(_rrr2, _rrr2);
                        __m512i _rrr3l = _mm512_unpacklo_epi64(_rrr3, _rrr3);
                        __m512i _rrr3h = _mm512_unpackhi_epi64(_rrr3, _rrr3);

#if __AVX512VNNI__
                        _sum00 = _mm512_dpwssd_epi32(_sum00, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(2, 0, 2, 0)), _w0);
                        _sum11 = _mm512_dpwssd_epi32(_sum11, _mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(2, 0, 2, 0)), _w0);
                        _sum22 = _mm512_dpwssd_epi32(_sum22, _mm512_shuffle_i32x4(_rrr2l, _rrr2h, _MM_SHUFFLE(2, 0, 2, 0)), _w0);
                        _sum33 = _mm512_dpwssd_epi32(_sum33, _mm512_shuffle_i32x4(_rrr3l, _rrr3h, _MM_SHUFFLE(2, 0, 2, 0)), _w0);
                        _sum00 = _mm512_dpwssd_epi32(_sum00, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(3, 1, 3, 1)), _w1);
                        _sum11 = _mm512_dpwssd_epi32(_sum11, _mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(3, 1, 3, 1)), _w1);
                        _sum22 = _mm512_dpwssd_epi32(_sum22, _mm512_shuffle_i32x4(_rrr2l, _rrr2h, _MM_SHUFFLE(3, 1, 3, 1)), _w1);
                        _sum33 = _mm512_dpwssd_epi32(_sum33, _mm512_shuffle_i32x4(_rrr3l, _rrr3h, _MM_SHUFFLE(3, 1, 3, 1)), _w1);
#else
                        _sum00 = _mm512_add_epi32(_sum00, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(2, 0, 2, 0)), _w0));
                        _sum11 = _mm512_add_epi32(_sum11, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(2, 0, 2, 0)), _w0));
                        _sum22 = _mm512_add_epi32(_sum22, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr2l, _rrr2h, _MM_SHUFFLE(2, 0, 2, 0)), _w0));
                        _sum33 = _mm512_add_epi32(_sum33, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr3l, _rrr3h, _MM_SHUFFLE(2, 0, 2, 0)), _w0));
                        _sum00 = _mm512_add_epi32(_sum00, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(3, 1, 3, 1)), _w1));
                        _sum11 = _mm512_add_epi32(_sum11, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(3, 1, 3, 1)), _w1));
                        _sum22 = _mm512_add_epi32(_sum22, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr2l, _rrr2h, _MM_SHUFFLE(3, 1, 3, 1)), _w1));
                        _sum33 = _mm512_add_epi32(_sum33, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr3l, _rrr3h, _MM_SHUFFLE(3, 1, 3, 1)), _w1));
#endif // __AVX512VNNI__

                        kptr += 64;
                    }
                }
                __m256i _ss0 = _mm256_add_epi32(_mm512_extracti64x4_epi64(_sum00, 0), _mm512_extracti64x4_epi64(_sum00, 1));
                __m256i _ss1 = _mm256_add_epi32(_mm512_extracti64x4_epi64(_sum11, 0), _mm512_extracti64x4_epi64(_sum11, 1));
                __m256i _ss2 = _mm256_add_epi32(_mm512_extracti64x4_epi64(_sum22, 0), _mm512_extracti64x4_epi64(_sum22, 1));
                __m256i _ss3 = _mm256_add_epi32(_mm512_extracti64x4_epi64(_sum33, 0), _mm512_extracti64x4_epi64(_sum33, 1));
                _sum0 = _mm_add_epi32(_sum0, _mm256_extracti128_si256(_ss0, 0));
                _sum0 = _mm_add_epi32(_sum0, _mm256_extracti128_si256(_ss0, 1));
                _sum1 = _mm_add_epi32(_sum1, _mm256_extracti128_si256(_ss1, 0));
                _sum1 = _mm_add_epi32(_sum1, _mm256_extracti128_si256(_ss1, 1));
                _sum2 = _mm_add_epi32(_sum2, _mm256_extracti128_si256(_ss2, 0));
                _sum2 = _mm_add_epi32(_sum2, _mm256_extracti128_si256(_ss2, 1));
                _sum3 = _mm_add_epi32(_sum3, _mm256_extracti128_si256(_ss3, 0));
                _sum3 = _mm_add_epi32(_sum3, _mm256_extracti128_si256(_ss3, 1));
            }
#endif // __AVX512F__
            {
#if __AVX2__
                __m256i _sum00 = _mm256_setzero_si256();
                __m256i _sum11 = _mm256_setzero_si256();
                __m256i _sum22 = _mm256_setzero_si256();
                __m256i _sum33 = _mm256_setzero_si256();
#endif
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;
                    const signed char* r2 = bottom_blob.channel(q / elempack).row<const signed char>(i2 * stride_h) + j2 * stride_w * elempack;
                    const signed char* r3 = bottom_blob.channel(q / elempack).row<const signed char>(i3 * stride_h) + j3 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];
                        const signed char* r2s = r2 + space_ofs[k];
                        const signed char* r3s = r3 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        __m128i _r2;
                        __m128i _r3;
                        if (elempack == 8)
                        {
                            _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                            _r1 = _mm_loadl_epi64((const __m128i*)r1s);
                            _r2 = _mm_loadl_epi64((const __m128i*)r2s);
                            _r3 = _mm_loadl_epi64((const __m128i*)r3s);
                        }
                        else // if (elempack == 1)
                        {
#if __AVX2__
                            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
                            _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1));
                            _r2 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r2s, _vindex, 1));
                            _r3 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r3s, _vindex, 1));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1), _sindex88);
                            __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1), _sindex88);
                            __m256i _val2_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r2s, _vindex, 1), _sindex88);
                            __m256i _val3_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r3s, _vindex, 1), _sindex88);
                            _r0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val0_32, 0), _mm256_extracti128_si256(_val0_32, 1));
                            _r1 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val1_32, 0), _mm256_extracti128_si256(_val1_32, 1));
                            _r2 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val2_32, 0), _mm256_extracti128_si256(_val2_32, 1));
                            _r3 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val3_32, 0), _mm256_extracti128_si256(_val3_32, 1));
#endif // __AVX512F__
#else
                            _r0 = _mm_setr_epi8(r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                            _r1 = _mm_setr_epi8(r1s[0], r1s[N], r1s[N * 2], r1s[N * 3], r1s[N * 4], r1s[N * 5], r1s[N * 6], r1s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                            _r2 = _mm_setr_epi8(r2s[0], r2s[N], r2s[N * 2], r2s[N * 3], r2s[N * 4], r2s[N * 5], r2s[N * 6], r2s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                            _r3 = _mm_setr_epi8(r3s[0], r3s[N], r3s[N * 2], r3s[N * 3], r3s[N * 4], r3s[N * 5], r3s[N * 6], r3s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
#endif // __AVX2__
                        }

#if __SSE4_1__
                        _r0 = _mm_cvtepi8_epi16(_r0);
                        _r1 = _mm_cvtepi8_epi16(_r1);
                        _r2 = _mm_cvtepi8_epi16(_r2);
                        _r3 = _mm_cvtepi8_epi16(_r3);
#else
                        _r0 = _mm_unpacklo_epi8(_r0, _mm_cmpgt_epi8(_mm_setzero_si128(), _r0));
                        _r1 = _mm_unpacklo_epi8(_r1, _mm_cmpgt_epi8(_mm_setzero_si128(), _r1));
                        _r2 = _mm_unpacklo_epi8(_r2, _mm_cmpgt_epi8(_mm_setzero_si128(), _r2));
                        _r3 = _mm_unpacklo_epi8(_r3, _mm_cmpgt_epi8(_mm_setzero_si128(), _r3));
#endif

#if __AVX2__
                        __m256i _w = _mm256_load_si256((const __m256i*)kptr);
                        __m256i _w0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w, 0));
                        __m256i _w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w, 1));

                        // 01234567 -> 01010101 23232323
                        __m256i _rr0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _mm_shuffle_epi32(_r0, _MM_SHUFFLE(2, 3, 0, 1)), 1);
                        __m256i _rr1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r1), _mm_shuffle_epi32(_r1, _MM_SHUFFLE(2, 3, 0, 1)), 1);
                        __m256i _rr2 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r2), _mm_shuffle_epi32(_r2, _MM_SHUFFLE(2, 3, 0, 1)), 1);
                        __m256i _rr3 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r3), _mm_shuffle_epi32(_r3, _MM_SHUFFLE(2, 3, 0, 1)), 1);

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum00 = _mm256_dpwssd_epi32(_sum00, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum11 = _mm256_dpwssd_epi32(_sum11, _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum22 = _mm256_dpwssd_epi32(_sum22, _mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum33 = _mm256_dpwssd_epi32(_sum33, _mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum00 = _mm256_dpwssd_epi32(_sum00, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
                        _sum11 = _mm256_dpwssd_epi32(_sum11, _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
                        _sum22 = _mm256_dpwssd_epi32(_sum22, _mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
                        _sum33 = _mm256_dpwssd_epi32(_sum33, _mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
#else
                        _sum00 = _mm256_add_epi32(_sum00, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum11 = _mm256_add_epi32(_sum11, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum22 = _mm256_add_epi32(_sum22, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum33 = _mm256_add_epi32(_sum33, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum00 = _mm256_add_epi32(_sum00, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
                        _sum11 = _mm256_add_epi32(_sum11, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
                        _sum22 = _mm256_add_epi32(_sum22, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr2, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
                        _sum33 = _mm256_add_epi32(_sum33, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr3, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
#endif
#else // __AVX2__
                        __m128i _w01 = _mm_load_si128((const __m128i*)kptr);
                        __m128i _w23 = _mm_load_si128((const __m128i*)(kptr + 16));
                        __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                        __m128i _extw23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w23);
                        __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                        __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);
                        __m128i _w2 = _mm_unpacklo_epi8(_w23, _extw23);
                        __m128i _w3 = _mm_unpackhi_epi8(_w23, _extw23);

                        // 01234567 -> 01010101
#if __XOP__
                        _sum0 = _mm_maddd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(0, 0, 0, 0)), _w0, _sum0);
                        _sum1 = _mm_maddd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(0, 0, 0, 0)), _w0, _sum1);
                        _sum2 = _mm_maddd_epi16(_mm_shuffle_epi32(_r2, _MM_SHUFFLE(0, 0, 0, 0)), _w0, _sum2);
                        _sum3 = _mm_maddd_epi16(_mm_shuffle_epi32(_r3, _MM_SHUFFLE(0, 0, 0, 0)), _w0, _sum3);
                        _sum0 = _mm_maddd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(1, 1, 1, 1)), _w1, _sum0);
                        _sum1 = _mm_maddd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(1, 1, 1, 1)), _w1, _sum1);
                        _sum2 = _mm_maddd_epi16(_mm_shuffle_epi32(_r2, _MM_SHUFFLE(1, 1, 1, 1)), _w1, _sum2);
                        _sum3 = _mm_maddd_epi16(_mm_shuffle_epi32(_r3, _MM_SHUFFLE(1, 1, 1, 1)), _w1, _sum3);
                        _sum0 = _mm_maddd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(2, 2, 2, 2)), _w2, _sum0);
                        _sum1 = _mm_maddd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(2, 2, 2, 2)), _w2, _sum1);
                        _sum2 = _mm_maddd_epi16(_mm_shuffle_epi32(_r2, _MM_SHUFFLE(2, 2, 2, 2)), _w2, _sum2);
                        _sum3 = _mm_maddd_epi16(_mm_shuffle_epi32(_r3, _MM_SHUFFLE(2, 2, 2, 2)), _w2, _sum3);
                        _sum0 = _mm_maddd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(3, 3, 3, 3)), _w3, _sum0);
                        _sum1 = _mm_maddd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(3, 3, 3, 3)), _w3, _sum1);
                        _sum2 = _mm_maddd_epi16(_mm_shuffle_epi32(_r2, _MM_SHUFFLE(3, 3, 3, 3)), _w3, _sum2);
                        _sum3 = _mm_maddd_epi16(_mm_shuffle_epi32(_r3, _MM_SHUFFLE(3, 3, 3, 3)), _w3, _sum3);
#else
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_mm_shuffle_epi32(_r2, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_mm_shuffle_epi32(_r3, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                        _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_mm_shuffle_epi32(_r2, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                        _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_mm_shuffle_epi32(_r3, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                        _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_mm_shuffle_epi32(_r2, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                        _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_mm_shuffle_epi32(_r3, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
                        _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_mm_shuffle_epi32(_r2, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
                        _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_mm_shuffle_epi32(_r3, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
#endif // __XOP__
#endif // __AVX2__

                        kptr += 32;
                    }
                }
#if __AVX2__
                __m128i _ss0 = _mm_add_epi32(_mm256_extracti128_si256(_sum00, 0), _mm256_extracti128_si256(_sum00, 1));
                __m128i _ss1 = _mm_add_epi32(_mm256_extracti128_si256(_sum11, 0), _mm256_extracti128_si256(_sum11, 1));
                __m128i _ss2 = _mm_add_epi32(_mm256_extracti128_si256(_sum22, 0), _mm256_extracti128_si256(_sum22, 1));
                __m128i _ss3 = _mm_add_epi32(_mm256_extracti128_si256(_sum33, 0), _mm256_extracti128_si256(_sum33, 1));
                _sum0 = _mm_add_epi32(_sum0, _ss0);
                _sum1 = _mm_add_epi32(_sum1, _ss1);
                _sum2 = _mm_add_epi32(_sum2, _ss2);
                _sum3 = _mm_add_epi32(_sum3, _ss3);
#endif // __AVX2__
            }
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;
                const signed char* r2 = bottom_blob.channel(q).row<const signed char>(i2 * stride_h) + j2 * stride_w;
                const signed char* r3 = bottom_blob.channel(q).row<const signed char>(i3 * stride_h) + j3 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    const signed char* r2s = r2 + space_ofs[k];
                    const signed char* r3s = r3 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _r0 = _mm_setr_epi16(r0s[0], r0s[N], r0s[0], r0s[N], r0s[0], r0s[N], r0s[0], r0s[N]);
                        __m128i _r1 = _mm_setr_epi16(r1s[0], r1s[N], r1s[0], r1s[N], r1s[0], r1s[N], r1s[0], r1s[N]);
                        __m128i _r2 = _mm_setr_epi16(r2s[0], r2s[N], r2s[0], r2s[N], r2s[0], r2s[N], r2s[0], r2s[N]);
                        __m128i _r3 = _mm_setr_epi16(r3s[0], r3s[N], r3s[0], r3s[N], r3s[0], r3s[N], r3s[0], r3s[N]);

                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
#if __SSE4_1__
                        _w = _mm_cvtepi8_epi16(_w);
#else
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
#endif

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum0 = _mm_dpwssd_epi32(_sum0, _r0, _w);
                        _sum1 = _mm_dpwssd_epi32(_sum1, _r1, _w);
                        _sum2 = _mm_dpwssd_epi32(_sum2, _r2, _w);
                        _sum3 = _mm_dpwssd_epi32(_sum3, _r3, _w);
#elif __XOP__
                        _sum0 = _mm_maddd_epi16(_r0, _w, _sum0);
                        _sum1 = _mm_maddd_epi16(_r1, _w, _sum1);
                        _sum2 = _mm_maddd_epi16(_r2, _w, _sum2);
                        _sum3 = _mm_maddd_epi16(_r3, _w, _sum3);
#else
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_r0, _w));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_r1, _w));
                        _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_r2, _w));
                        _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_r3, _w));
#endif

                        kptr += 8;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;
                const signed char* r2 = bottom_blob.channel(q).row<const signed char>(i2 * stride_h) + j2 * stride_w;
                const signed char* r3 = bottom_blob.channel(q).row<const signed char>(i3 * stride_h) + j3 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    const signed char* r2s = r2 + space_ofs[k];
                    const signed char* r3s = r3 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _r0 = _mm_set1_epi16(r0s[0]);
                        __m128i _r1 = _mm_set1_epi16(r1s[0]);
                        __m128i _r2 = _mm_set1_epi16(r2s[0]);
                        __m128i _r3 = _mm_set1_epi16(r3s[0]);

                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
#if __SSE4_1__
                        _w = _mm_cvtepi8_epi16(_w);
#else
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
#endif

#if __XOP__
                        _w = _mm_unpacklo_epi16(_w, _mm_setzero_si128());
                        _sum0 = _mm_maccd_epi16(_r0, _w, _sum0);
                        _sum1 = _mm_maccd_epi16(_r1, _w, _sum1);
                        _sum2 = _mm_maccd_epi16(_r2, _w, _sum2);
                        _sum3 = _mm_maccd_epi16(_r3, _w, _sum3);
#else
                        __m128i _sl0 = _mm_mullo_epi16(_r0, _w);
                        __m128i _sh0 = _mm_mulhi_epi16(_r0, _w);
                        __m128i _sl1 = _mm_mullo_epi16(_r1, _w);
                        __m128i _sh1 = _mm_mulhi_epi16(_r1, _w);
                        __m128i _sl2 = _mm_mullo_epi16(_r2, _w);
                        __m128i _sh2 = _mm_mulhi_epi16(_r2, _w);
                        __m128i _sl3 = _mm_mullo_epi16(_r3, _w);
                        __m128i _sh3 = _mm_mulhi_epi16(_r3, _w);
                        __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                        __m128i _s1 = _mm_unpacklo_epi16(_sl1, _sh1);
                        __m128i _s2 = _mm_unpacklo_epi16(_sl2, _sh2);
                        __m128i _s3 = _mm_unpacklo_epi16(_sl3, _sh3);

                        _sum0 = _mm_add_epi32(_sum0, _s0);
                        _sum1 = _mm_add_epi32(_sum1, _s1);
                        _sum2 = _mm_add_epi32(_sum2, _s2);
                        _sum3 = _mm_add_epi32(_sum3, _s3);
#endif // __XOP__

                        kptr += 4;
                    }
                }
            }

            if (out_elempack == 4)
            {
                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
                _mm_store_si128((__m128i*)(outptr + 8), _sum2);
                _mm_store_si128((__m128i*)(outptr + 12), _sum3);
                outptr += 16;
            }
            if (out_elempack == 1)
            {
#if __AVX512F__
                __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(M));
                _mm_i32scatter_epi32(outptr, _vindex, _sum0, sizeof(int));
                _mm_i32scatter_epi32(outptr + 1, _vindex, _sum1, sizeof(int));
                _mm_i32scatter_epi32(outptr + 2, _vindex, _sum2, sizeof(int));
                _mm_i32scatter_epi32(outptr + 3, _vindex, _sum3, sizeof(int));
#else
                int sum0[4];
                int sum1[4];
                int sum2[4];
                int sum3[4];
                _mm_storeu_si128((__m128i*)sum0, _sum0);
                _mm_storeu_si128((__m128i*)sum1, _sum1);
                _mm_storeu_si128((__m128i*)sum2, _sum2);
                _mm_storeu_si128((__m128i*)sum3, _sum3);

                outptr[0] = sum0[0];
                outptr[1] = sum1[0];
                outptr[2] = sum2[0];
                outptr[3] = sum3[0];
                outptr[M] = sum0[1];
                outptr[M + 1] = sum1[1];
                outptr[M + 2] = sum2[1];
                outptr[M + 3] = sum3[1];
                outptr[M * 2] = sum0[2];
                outptr[M * 2 + 1] = sum1[2];
                outptr[M * 2 + 2] = sum2[2];
                outptr[M * 2 + 3] = sum3[2];
                outptr[M * 3] = sum0[3];
                outptr[M * 3 + 1] = sum1[3];
                outptr[M * 3 + 2] = sum2[3];
                outptr[M * 3 + 3] = sum3[3];
#endif // __AVX512F__
                outptr += 4;
            }
        }
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            __m128i _sum0 = _mm_setzero_si128();
            __m128i _sum1 = _mm_setzero_si128();
            __m128i _sum2 = _mm_setzero_si128();
            __m128i _sum3 = _mm_setzero_si128();

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4);
#elif __AVX2__
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);
#else
            const signed char* kptr = weight_data_tm.channel(p / 4);
#endif

            int q = 0;
#if __AVX512F__
            {
                __m512i _sum00 = _mm512_setzero_si512();
                __m512i _sum11 = _mm512_setzero_si512();
                __m512i _sum22 = _mm512_setzero_si512();
                __m512i _sum33 = _mm512_setzero_si512();
                for (; q + 15 < inch; q += 16)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        if (elempack == 16)
                        {
                            _r0 = _mm_load_si128((const __m128i*)r0s);
                            _r1 = _mm_load_si128((const __m128i*)r1s);
                        }
                        else if (elempack == 8)
                        {
                            __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                            __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                            __m128i _t2 = _mm_loadl_epi64((const __m128i*)r1s);
                            __m128i _t3 = _mm_loadl_epi64((const __m128i*)(r1s + N));
                            _r0 = _mm_unpacklo_epi64(_t0, _t1);
                            _r1 = _mm_unpacklo_epi64(_t2, _t3);
                        }
                        else // if (elempack == 1)
                        {
                            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                            _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r1s), 1));
                        }

                        __m256i _rr0 = _mm256_cvtepi8_epi16(_r0);
                        __m256i _rr1 = _mm256_cvtepi8_epi16(_r1);

                        __m512i _w = _mm512_load_si512((const __m512i*)kptr);
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w, 1));

                        // 01234567 89abcdef -> 01010101 23232323 45454545 67676767
                        _rr0 = _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(3, 1, 2, 0));
                        _rr1 = _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(3, 1, 2, 0));

                        __m512i _rrr0 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr0), _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(1, 0, 3, 2)), 1);
                        __m512i _rrr1 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr1), _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(1, 0, 3, 2)), 1);

                        _rrr0 = _mm512_unpacklo_epi32(_rrr0, _rrr0);
                        _rrr1 = _mm512_unpacklo_epi32(_rrr1, _rrr1);

                        __m512i _rrr0l = _mm512_unpacklo_epi64(_rrr0, _rrr0);
                        __m512i _rrr0h = _mm512_unpackhi_epi64(_rrr0, _rrr0);
                        __m512i _rrr1l = _mm512_unpacklo_epi64(_rrr1, _rrr1);
                        __m512i _rrr1h = _mm512_unpackhi_epi64(_rrr1, _rrr1);

#if __AVX512VNNI__
                        _sum00 = _mm512_dpwssd_epi32(_sum00, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(2, 0, 2, 0)), _w0);
                        _sum11 = _mm512_dpwssd_epi32(_sum11, _mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(2, 0, 2, 0)), _w0);
                        _sum22 = _mm512_dpwssd_epi32(_sum22, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(3, 1, 3, 1)), _w1);
                        _sum33 = _mm512_dpwssd_epi32(_sum33, _mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(3, 1, 3, 1)), _w1);
#else
                        _sum00 = _mm512_add_epi32(_sum00, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(2, 0, 2, 0)), _w0));
                        _sum11 = _mm512_add_epi32(_sum11, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(2, 0, 2, 0)), _w0));
                        _sum22 = _mm512_add_epi32(_sum22, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(3, 1, 3, 1)), _w1));
                        _sum33 = _mm512_add_epi32(_sum33, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr1l, _rrr1h, _MM_SHUFFLE(3, 1, 3, 1)), _w1));
#endif // __AVX512VNNI__

                        kptr += 64;
                    }
                }
                __m256i _ss0 = _mm256_add_epi32(_mm512_extracti64x4_epi64(_sum00, 0), _mm512_extracti64x4_epi64(_sum00, 1));
                __m256i _ss1 = _mm256_add_epi32(_mm512_extracti64x4_epi64(_sum11, 0), _mm512_extracti64x4_epi64(_sum11, 1));
                __m256i _ss2 = _mm256_add_epi32(_mm512_extracti64x4_epi64(_sum22, 0), _mm512_extracti64x4_epi64(_sum22, 1));
                __m256i _ss3 = _mm256_add_epi32(_mm512_extracti64x4_epi64(_sum33, 0), _mm512_extracti64x4_epi64(_sum33, 1));
                _sum0 = _mm_add_epi32(_sum0, _mm256_extracti128_si256(_ss0, 0));
                _sum0 = _mm_add_epi32(_sum0, _mm256_extracti128_si256(_ss0, 1));
                _sum1 = _mm_add_epi32(_sum1, _mm256_extracti128_si256(_ss1, 0));
                _sum1 = _mm_add_epi32(_sum1, _mm256_extracti128_si256(_ss1, 1));
                _sum2 = _mm_add_epi32(_sum2, _mm256_extracti128_si256(_ss2, 0));
                _sum2 = _mm_add_epi32(_sum2, _mm256_extracti128_si256(_ss2, 1));
                _sum3 = _mm_add_epi32(_sum3, _mm256_extracti128_si256(_ss3, 0));
                _sum3 = _mm_add_epi32(_sum3, _mm256_extracti128_si256(_ss3, 1));
            }
#endif // __AVX512F__
            {
#if __AVX2__
                __m256i _sum00 = _mm256_setzero_si256();
                __m256i _sum11 = _mm256_setzero_si256();
                __m256i _sum22 = _mm256_setzero_si256();
                __m256i _sum33 = _mm256_setzero_si256();
#endif
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        if (elempack == 8)
                        {
                            _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                            _r1 = _mm_loadl_epi64((const __m128i*)r1s);
                        }
                        else // if (elempack == 1)
                        {
#if __AVX2__
                            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
                            _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1), _sindex88);
                            __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1), _sindex88);
                            _r0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val0_32, 0), _mm256_extracti128_si256(_val0_32, 1));
                            _r1 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val1_32, 0), _mm256_extracti128_si256(_val1_32, 1));
#endif // __AVX512F__
#else
                            _r0 = _mm_setr_epi8(r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                            _r1 = _mm_setr_epi8(r1s[0], r1s[N], r1s[N * 2], r1s[N * 3], r1s[N * 4], r1s[N * 5], r1s[N * 6], r1s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
#endif // __AVX2__
                        }

#if __SSE4_1__
                        _r0 = _mm_cvtepi8_epi16(_r0);
                        _r1 = _mm_cvtepi8_epi16(_r1);
#else
                        _r0 = _mm_unpacklo_epi8(_r0, _mm_cmpgt_epi8(_mm_setzero_si128(), _r0));
                        _r1 = _mm_unpacklo_epi8(_r1, _mm_cmpgt_epi8(_mm_setzero_si128(), _r1));
#endif

#if __AVX2__
                        __m256i _w = _mm256_load_si256((const __m256i*)kptr);
                        __m256i _w0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w, 0));
                        __m256i _w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w, 1));

                        // 01234567 -> 01010101 23232323
                        __m256i _rr0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _mm_shuffle_epi32(_r0, _MM_SHUFFLE(2, 3, 0, 1)), 1);
                        __m256i _rr1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r1), _mm_shuffle_epi32(_r1, _MM_SHUFFLE(2, 3, 0, 1)), 1);

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum00 = _mm256_dpwssd_epi32(_sum00, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum11 = _mm256_dpwssd_epi32(_sum11, _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum22 = _mm256_dpwssd_epi32(_sum22, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
                        _sum33 = _mm256_dpwssd_epi32(_sum33, _mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
#else
                        _sum00 = _mm256_add_epi32(_sum00, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum11 = _mm256_add_epi32(_sum11, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum22 = _mm256_add_epi32(_sum22, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
                        _sum33 = _mm256_add_epi32(_sum33, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr1, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
#endif
#else // __AVX2__
                        __m128i _w01 = _mm_load_si128((const __m128i*)kptr);
                        __m128i _w23 = _mm_load_si128((const __m128i*)(kptr + 16));
                        __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                        __m128i _extw23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w23);
                        __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                        __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);
                        __m128i _w2 = _mm_unpacklo_epi8(_w23, _extw23);
                        __m128i _w3 = _mm_unpackhi_epi8(_w23, _extw23);

                        // 01234567 -> 01010101
#if __XOP__
                        _sum0 = _mm_maddd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(0, 0, 0, 0)), _w0, _sum0);
                        _sum1 = _mm_maddd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(0, 0, 0, 0)), _w0, _sum1);
                        _sum2 = _mm_maddd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(1, 1, 1, 1)), _w1, _sum2);
                        _sum3 = _mm_maddd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(1, 1, 1, 1)), _w1, _sum3);
                        _sum0 = _mm_maddd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(2, 2, 2, 2)), _w2, _sum0);
                        _sum1 = _mm_maddd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(2, 2, 2, 2)), _w2, _sum1);
                        _sum2 = _mm_maddd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(3, 3, 3, 3)), _w3, _sum2);
                        _sum3 = _mm_maddd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(3, 3, 3, 3)), _w3, _sum3);
#else
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                        _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                        _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
                        _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_mm_shuffle_epi32(_r1, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
#endif // __XOP__
#endif // __AVX2__

                        kptr += 32;
                    }
                }
#if __AVX2__
                __m128i _ss0 = _mm_add_epi32(_mm256_extracti128_si256(_sum00, 0), _mm256_extracti128_si256(_sum00, 1));
                __m128i _ss1 = _mm_add_epi32(_mm256_extracti128_si256(_sum11, 0), _mm256_extracti128_si256(_sum11, 1));
                __m128i _ss2 = _mm_add_epi32(_mm256_extracti128_si256(_sum22, 0), _mm256_extracti128_si256(_sum22, 1));
                __m128i _ss3 = _mm_add_epi32(_mm256_extracti128_si256(_sum33, 0), _mm256_extracti128_si256(_sum33, 1));
                _sum0 = _mm_add_epi32(_sum0, _ss0);
                _sum1 = _mm_add_epi32(_sum1, _ss1);
                _sum2 = _mm_add_epi32(_sum2, _ss2);
                _sum3 = _mm_add_epi32(_sum3, _ss3);
#endif // __AVX2__
            }
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _r0 = _mm_setr_epi16(r0s[0], r0s[N], r0s[0], r0s[N], r0s[0], r0s[N], r0s[0], r0s[N]);
                        __m128i _r1 = _mm_setr_epi16(r1s[0], r1s[N], r1s[0], r1s[N], r1s[0], r1s[N], r1s[0], r1s[N]);

                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
#if __SSE4_1__
                        _w = _mm_cvtepi8_epi16(_w);
#else
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
#endif

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum0 = _mm_dpwssd_epi32(_sum0, _r0, _w);
                        _sum1 = _mm_dpwssd_epi32(_sum1, _r1, _w);
#elif __XOP__
                        _sum0 = _mm_maddd_epi16(_r0, _w, _sum0);
                        _sum1 = _mm_maddd_epi16(_r1, _w, _sum1);
#else
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_r0, _w));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_r1, _w));
#endif

                        kptr += 8;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _r0 = _mm_set1_epi16(r0s[0]);
                        __m128i _r1 = _mm_set1_epi16(r1s[0]);

                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
#if __SSE4_1__
                        _w = _mm_cvtepi8_epi16(_w);
#else
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
#endif

#if __XOP__
                        _w = _mm_unpacklo_epi16(_w, _mm_setzero_si128());
                        _sum0 = _mm_maccd_epi16(_r0, _w, _sum0);
                        _sum1 = _mm_maccd_epi16(_r1, _w, _sum1);
#else
                        __m128i _sl0 = _mm_mullo_epi16(_r0, _w);
                        __m128i _sh0 = _mm_mulhi_epi16(_r0, _w);
                        __m128i _sl1 = _mm_mullo_epi16(_r1, _w);
                        __m128i _sh1 = _mm_mulhi_epi16(_r1, _w);
                        __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                        __m128i _s1 = _mm_unpacklo_epi16(_sl1, _sh1);

                        _sum0 = _mm_add_epi32(_sum0, _s0);
                        _sum1 = _mm_add_epi32(_sum1, _s1);
#endif // __XOP__

                        kptr += 4;
                    }
                }
            }

            _sum0 = _mm_add_epi32(_sum0, _sum2);
            _sum1 = _mm_add_epi32(_sum1, _sum3);

            if (out_elempack == 4)
            {
                _mm_store_si128((__m128i*)outptr, _sum0);
                _mm_store_si128((__m128i*)(outptr + 4), _sum1);
                outptr += 8;
            }
            if (out_elempack == 1)
            {
#if __AVX512F__
                __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(M));
                _mm_i32scatter_epi32(outptr, _vindex, _sum0, sizeof(int));
                _mm_i32scatter_epi32(outptr + 1, _vindex, _sum1, sizeof(int));
#else
                int sum0[4];
                int sum1[4];
                _mm_storeu_si128((__m128i*)sum0, _sum0);
                _mm_storeu_si128((__m128i*)sum1, _sum1);

                outptr[0] = sum0[0];
                outptr[1] = sum1[0];
                outptr[M] = sum0[1];
                outptr[M + 1] = sum1[1];
                outptr[M * 2] = sum0[2];
                outptr[M * 2 + 1] = sum1[2];
                outptr[M * 3] = sum0[3];
                outptr[M * 3 + 1] = sum1[3];
#endif // __AVX512F__
                outptr += 2;
            }
        }
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            __m128i _sum0 = _mm_setzero_si128();

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4);
#elif __AVX2__
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);
#else
            const signed char* kptr = weight_data_tm.channel(p / 4);
#endif

            int q = 0;
#if __AVX512F__
            {
                __m512i _sum00 = _mm512_setzero_si512();
                __m512i _sum11 = _mm512_setzero_si512();
                for (; q + 15 < inch; q += 16)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];

                        __m128i _r0;
                        if (elempack == 16)
                        {
                            _r0 = _mm_load_si128((const __m128i*)r0s);
                        }
                        else if (elempack == 8)
                        {
                            __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                            __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                            _r0 = _mm_unpacklo_epi64(_t0, _t1);
                        }
                        else // if (elempack == 1)
                        {
                            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                        }

                        __m256i _rr0 = _mm256_cvtepi8_epi16(_r0);

                        __m512i _w = _mm512_load_si512((const __m512i*)kptr);
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w, 1));

                        // 01234567 89abcdef -> 01010101 23232323 45454545 67676767
                        _rr0 = _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(3, 1, 2, 0));

                        __m512i _rrr0 = _mm512_inserti64x4(_mm512_castsi256_si512(_rr0), _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(1, 0, 3, 2)), 1);

                        _rrr0 = _mm512_unpacklo_epi32(_rrr0, _rrr0);

                        __m512i _rrr0l = _mm512_unpacklo_epi64(_rrr0, _rrr0);
                        __m512i _rrr0h = _mm512_unpackhi_epi64(_rrr0, _rrr0);

#if __AVX512VNNI__
                        _sum00 = _mm512_dpwssd_epi32(_sum00, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(2, 0, 2, 0)), _w0);
                        _sum11 = _mm512_dpwssd_epi32(_sum11, _mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(3, 1, 3, 1)), _w1);
#else
                        _sum00 = _mm512_add_epi32(_sum00, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(2, 0, 2, 0)), _w0));
                        _sum11 = _mm512_add_epi32(_sum11, _mm512_madd_epi16(_mm512_shuffle_i32x4(_rrr0l, _rrr0h, _MM_SHUFFLE(3, 1, 3, 1)), _w1));
#endif // __AVX512VNNI__

                        kptr += 64;
                    }
                }
                _sum00 = _mm512_add_epi32(_sum00, _sum11);
                __m256i _ss0 = _mm256_add_epi32(_mm512_extracti64x4_epi64(_sum00, 0), _mm512_extracti64x4_epi64(_sum00, 1));
                _sum0 = _mm_add_epi32(_sum0, _mm256_extracti128_si256(_ss0, 0));
                _sum0 = _mm_add_epi32(_sum0, _mm256_extracti128_si256(_ss0, 1));
            }
#endif // __AVX512F__
            {
#if __AVX2__
                __m256i _sum00 = _mm256_setzero_si256();
                __m256i _sum11 = _mm256_setzero_si256();
#else
                __m128i _sum1 = _mm_setzero_si128();
                __m128i _sum2 = _mm_setzero_si128();
                __m128i _sum3 = _mm_setzero_si128();
#endif
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];

                        __m128i _r0;
                        if (elempack == 8)
                        {
                            _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                        }
                        else // if (elempack == 1)
                        {
#if __AVX2__
                            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1), _sindex88);
                            _r0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val32, 0), _mm256_extracti128_si256(_val32, 1));
#endif // __AVX512F__
#else
                            _r0 = _mm_setr_epi8(r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
#endif // __AVX2__
                        }

#if __SSE4_1__
                        _r0 = _mm_cvtepi8_epi16(_r0);
#else
                        _r0 = _mm_unpacklo_epi8(_r0, _mm_cmpgt_epi8(_mm_setzero_si128(), _r0));
#endif

#if __AVX2__
                        __m256i _w = _mm256_load_si256((const __m256i*)kptr);
                        __m256i _w0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w, 0));
                        __m256i _w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w, 1));

                        // 01234567 -> 01010101 23232323
                        __m256i _rr0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _mm_shuffle_epi32(_r0, _MM_SHUFFLE(2, 3, 0, 1)), 1);

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum00 = _mm256_dpwssd_epi32(_sum00, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(0, 0, 0, 0)), _w0);
                        _sum11 = _mm256_dpwssd_epi32(_sum11, _mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(2, 2, 2, 2)), _w1);
#else
                        _sum00 = _mm256_add_epi32(_sum00, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum11 = _mm256_add_epi32(_sum11, _mm256_madd_epi16(_mm256_shuffle_epi32(_rr0, _MM_SHUFFLE(2, 2, 2, 2)), _w1));
#endif
#else // __AVX2__
                        __m128i _w01 = _mm_load_si128((const __m128i*)kptr);
                        __m128i _w23 = _mm_load_si128((const __m128i*)(kptr + 16));
                        __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                        __m128i _extw23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w23);
                        __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                        __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);
                        __m128i _w2 = _mm_unpacklo_epi8(_w23, _extw23);
                        __m128i _w3 = _mm_unpackhi_epi8(_w23, _extw23);

                        // 01234567 -> 01010101
#if __XOP__
                        _sum0 = _mm_maddd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(0, 0, 0, 0)), _w0, _sum0);
                        _sum1 = _mm_maddd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(1, 1, 1, 1)), _w1, _sum1);
                        _sum2 = _mm_maddd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(2, 2, 2, 2)), _w2, _sum2);
                        _sum3 = _mm_maddd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(3, 3, 3, 3)), _w3, _sum3);
#else
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(0, 0, 0, 0)), _w0));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(1, 1, 1, 1)), _w1));
                        _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(2, 2, 2, 2)), _w2));
                        _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_mm_shuffle_epi32(_r0, _MM_SHUFFLE(3, 3, 3, 3)), _w3));
#endif // __XOP__
#endif // __AVX2__

                        kptr += 32;
                    }
                }
#if __AVX2__
                _sum00 = _mm256_add_epi32(_sum00, _sum11);
                __m128i _ss = _mm_add_epi32(_mm256_extracti128_si256(_sum00, 0), _mm256_extracti128_si256(_sum00, 1));
                _sum0 = _mm_add_epi32(_sum0, _ss);
#else
                _sum0 = _mm_add_epi32(_sum0, _sum1);
                _sum2 = _mm_add_epi32(_sum2, _sum3);
                _sum0 = _mm_add_epi32(_sum0, _sum2);
#endif // __AVX2__
            }
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _r0 = _mm_setr_epi16(r0s[0], r0s[N], r0s[0], r0s[N], r0s[0], r0s[N], r0s[0], r0s[N]);

                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
#if __SSE4_1__
                        _w = _mm_cvtepi8_epi16(_w);
#else
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
#endif

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum0 = _mm_dpwssd_epi32(_sum0, _r0, _w);
#elif __XOP__
                        _sum0 = _mm_maddd_epi16(_r0, _w, _sum0);
#else
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_r0, _w));
#endif

                        kptr += 8;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _r0 = _mm_set1_epi16(r0s[0]);

                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
#if __SSE4_1__
                        _w = _mm_cvtepi8_epi16(_w);
#else
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
#endif

#if __XOP__
                        _w = _mm_unpacklo_epi16(_w, _mm_setzero_si128());
                        _sum0 = _mm_maccd_epi16(_r0, _w, _sum0);
#else
                        __m128i _sl = _mm_mullo_epi16(_r0, _w);
                        __m128i _sh = _mm_mulhi_epi16(_r0, _w);
                        __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);

                        _sum0 = _mm_add_epi32(_sum0, _s0);
#endif // __XOP__

                        kptr += 4;
                    }
                }
            }

            if (out_elempack == 4)
            {
                _mm_store_si128((__m128i*)outptr, _sum0);
                outptr += 4;
            }
            if (out_elempack == 1)
            {
#if __AVX512F__
                __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(M));
                _mm_i32scatter_epi32(outptr, _vindex, _sum0, sizeof(int));
#else
                int sum[4];
                _mm_storeu_si128((__m128i*)sum, _sum0);

                outptr[0] = sum[0];
                outptr[M] = sum[1];
                outptr[M * 2] = sum[2];
                outptr[M * 3] = sum[3];
#endif // __AVX512F__
                outptr += 1;
            }
        }
    }
    remain_outch_start += nn_outch * 4;
    nn_outch = (outch - remain_outch_start) / 2;
#else // __SSE2__
    nn_outch = (outch - remain_outch_start) / 2;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __SSE2__
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 2;

        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int N = bottom_blob.cstep * elempack;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);

        int ij = 0;
#if __SSE2__
        for (; ij + 3 < outw * outh; ij += 4)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int i2 = (ij + 2) / outw;
            const int i3 = (ij + 3) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;
            const int j2 = (ij + 2) % outw;
            const int j3 = (ij + 3) % outw;

            __m128i _sum0 = _mm_setzero_si128();
            __m128i _sum1 = _mm_setzero_si128();

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4 + (p % 4) / 2);
#elif __AVX2__
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#else
            const signed char* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2);
#endif

            int q = 0;
#if __AVX512F__
            {
                __m512i _sum00 = _mm512_setzero_si512();
                __m512i _sum11 = _mm512_setzero_si512();
                __m512i _sum22 = _mm512_setzero_si512();
                __m512i _sum33 = _mm512_setzero_si512();
                for (; q + 15 < inch; q += 16)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;
                    const signed char* r2 = bottom_blob.channel(q / elempack).row<const signed char>(i2 * stride_h) + j2 * stride_w * elempack;
                    const signed char* r3 = bottom_blob.channel(q / elempack).row<const signed char>(i3 * stride_h) + j3 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];
                        const signed char* r2s = r2 + space_ofs[k];
                        const signed char* r3s = r3 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        __m128i _r2;
                        __m128i _r3;
                        if (elempack == 16)
                        {
                            _r0 = _mm_load_si128((const __m128i*)r0s);
                            _r1 = _mm_load_si128((const __m128i*)r1s);
                            _r2 = _mm_load_si128((const __m128i*)r2s);
                            _r3 = _mm_load_si128((const __m128i*)r3s);
                        }
                        else if (elempack == 8)
                        {
                            __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                            __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                            __m128i _t2 = _mm_loadl_epi64((const __m128i*)r1s);
                            __m128i _t3 = _mm_loadl_epi64((const __m128i*)(r1s + N));
                            __m128i _t4 = _mm_loadl_epi64((const __m128i*)r2s);
                            __m128i _t5 = _mm_loadl_epi64((const __m128i*)(r2s + N));
                            __m128i _t6 = _mm_loadl_epi64((const __m128i*)r3s);
                            __m128i _t7 = _mm_loadl_epi64((const __m128i*)(r3s + N));
                            _r0 = _mm_unpacklo_epi64(_t0, _t1);
                            _r1 = _mm_unpacklo_epi64(_t2, _t3);
                            _r2 = _mm_unpacklo_epi64(_t4, _t5);
                            _r3 = _mm_unpacklo_epi64(_t6, _t7);
                        }
                        else // if (elempack == 1)
                        {
                            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                            _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r1s), 1));
                            _r2 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r2s), 1));
                            _r3 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r3s), 1));
                        }

                        __m256i _val0 = _mm256_cvtepi8_epi16(_r0);
                        __m256i _val1 = _mm256_cvtepi8_epi16(_r1);
                        __m256i _val2 = _mm256_cvtepi8_epi16(_r2);
                        __m256i _val3 = _mm256_cvtepi8_epi16(_r3);
                        __m512i _valval0 = _mm512_inserti64x4(_mm512_castsi256_si512(_val0), _val0, 1);
                        __m512i _valval1 = _mm512_inserti64x4(_mm512_castsi256_si512(_val1), _val1, 1);
                        __m512i _valval2 = _mm512_inserti64x4(_mm512_castsi256_si512(_val2), _val2, 1);
                        __m512i _valval3 = _mm512_inserti64x4(_mm512_castsi256_si512(_val3), _val3, 1);

                        __m512i _w = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)kptr));

#if __AVX512VNNI__
                        _sum00 = _mm512_dpwssd_epi32(_sum00, _valval0, _w);
                        _sum11 = _mm512_dpwssd_epi32(_sum11, _valval1, _w);
                        _sum22 = _mm512_dpwssd_epi32(_sum22, _valval2, _w);
                        _sum33 = _mm512_dpwssd_epi32(_sum33, _valval3, _w);
#else
                        _sum00 = _mm512_add_epi32(_sum00, _mm512_madd_epi16(_valval0, _w));
                        _sum11 = _mm512_add_epi32(_sum11, _mm512_madd_epi16(_valval1, _w));
                        _sum22 = _mm512_add_epi32(_sum22, _mm512_madd_epi16(_valval2, _w));
                        _sum33 = _mm512_add_epi32(_sum33, _mm512_madd_epi16(_valval3, _w));
#endif // __AVX512VNNI__

                        kptr += 32;
                    }
                }
                __m256i _sum010 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum00, 0), _mm512_extracti64x4_epi64(_sum11, 0));
                __m256i _sum230 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum22, 0), _mm512_extracti64x4_epi64(_sum33, 0));
                __m256i _sum011 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum00, 1), _mm512_extracti64x4_epi64(_sum11, 1));
                __m256i _sum231 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum22, 1), _mm512_extracti64x4_epi64(_sum33, 1));
                __m256i _ss0 = _mm256_hadd_epi32(_sum010, _sum230);
                __m256i _ss1 = _mm256_hadd_epi32(_sum011, _sum231);
                _sum0 = _mm_add_epi32(_sum0, _mm256_extracti128_si256(_ss0, 0));
                _sum0 = _mm_add_epi32(_sum0, _mm256_extracti128_si256(_ss0, 1));
                _sum1 = _mm_add_epi32(_sum1, _mm256_extracti128_si256(_ss1, 0));
                _sum1 = _mm_add_epi32(_sum1, _mm256_extracti128_si256(_ss1, 1));
            }
#endif // __AVX512F__
            {
#if __AVX2__
                __m256i _sum00 = _mm256_setzero_si256();
                __m256i _sum11 = _mm256_setzero_si256();
                __m256i _sum22 = _mm256_setzero_si256();
                __m256i _sum33 = _mm256_setzero_si256();
#else
                __m128i _sum00 = _mm_setzero_si128();
                __m128i _sum10 = _mm_setzero_si128();
                __m128i _sum01 = _mm_setzero_si128();
                __m128i _sum11 = _mm_setzero_si128();
                __m128i _sum02 = _mm_setzero_si128();
                __m128i _sum12 = _mm_setzero_si128();
                __m128i _sum03 = _mm_setzero_si128();
                __m128i _sum13 = _mm_setzero_si128();
#endif
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;
                    const signed char* r2 = bottom_blob.channel(q / elempack).row<const signed char>(i2 * stride_h) + j2 * stride_w * elempack;
                    const signed char* r3 = bottom_blob.channel(q / elempack).row<const signed char>(i3 * stride_h) + j3 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];
                        const signed char* r2s = r2 + space_ofs[k];
                        const signed char* r3s = r3 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        __m128i _r2;
                        __m128i _r3;
                        if (elempack == 8)
                        {
                            _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                            _r1 = _mm_loadl_epi64((const __m128i*)r1s);
                            _r2 = _mm_loadl_epi64((const __m128i*)r2s);
                            _r3 = _mm_loadl_epi64((const __m128i*)r3s);
                        }
                        else // if (elempack == 1)
                        {
#if __AVX2__
                            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
                            _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1));
                            _r2 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r2s, _vindex, 1));
                            _r3 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r3s, _vindex, 1));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1), _sindex88);
                            __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1), _sindex88);
                            __m256i _val2_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r2s, _vindex, 1), _sindex88);
                            __m256i _val3_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r3s, _vindex, 1), _sindex88);
                            _r0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val0_32, 0), _mm256_extracti128_si256(_val0_32, 1));
                            _r1 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val1_32, 0), _mm256_extracti128_si256(_val1_32, 1));
                            _r2 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val2_32, 0), _mm256_extracti128_si256(_val2_32, 1));
                            _r3 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val3_32, 0), _mm256_extracti128_si256(_val3_32, 1));
#endif // __AVX512F__
#else
                            _r0 = _mm_setr_epi8(r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                            _r1 = _mm_setr_epi8(r1s[0], r1s[N], r1s[N * 2], r1s[N * 3], r1s[N * 4], r1s[N * 5], r1s[N * 6], r1s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                            _r2 = _mm_setr_epi8(r2s[0], r2s[N], r2s[N * 2], r2s[N * 3], r2s[N * 4], r2s[N * 5], r2s[N * 6], r2s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                            _r3 = _mm_setr_epi8(r3s[0], r3s[N], r3s[N * 2], r3s[N * 3], r3s[N * 4], r3s[N * 5], r3s[N * 6], r3s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
#endif // __AVX2__
                        }

#if __SSE4_1__
                        _r0 = _mm_cvtepi8_epi16(_r0);
                        _r1 = _mm_cvtepi8_epi16(_r1);
                        _r2 = _mm_cvtepi8_epi16(_r2);
                        _r3 = _mm_cvtepi8_epi16(_r3);
#else
                        _r0 = _mm_unpacklo_epi8(_r0, _mm_cmpgt_epi8(_mm_setzero_si128(), _r0));
                        _r1 = _mm_unpacklo_epi8(_r1, _mm_cmpgt_epi8(_mm_setzero_si128(), _r1));
                        _r2 = _mm_unpacklo_epi8(_r2, _mm_cmpgt_epi8(_mm_setzero_si128(), _r2));
                        _r3 = _mm_unpacklo_epi8(_r3, _mm_cmpgt_epi8(_mm_setzero_si128(), _r3));
#endif

                        __m128i _w01 = _mm_load_si128((const __m128i*)kptr);
#if __AVX2__
                        __m256i _valval0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r0, 1);
                        __m256i _valval1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r1), _r1, 1);
                        __m256i _valval2 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r2), _r2, 1);
                        __m256i _valval3 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r3), _r3, 1);

                        __m256i _w = _mm256_cvtepi8_epi16(_w01);

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum00 = _mm256_dpwssd_epi32(_sum00, _valval0, _w);
                        _sum11 = _mm256_dpwssd_epi32(_sum11, _valval1, _w);
                        _sum22 = _mm256_dpwssd_epi32(_sum22, _valval2, _w);
                        _sum33 = _mm256_dpwssd_epi32(_sum33, _valval3, _w);
#else
                        _sum00 = _mm256_add_epi32(_sum00, _mm256_madd_epi16(_valval0, _w));
                        _sum11 = _mm256_add_epi32(_sum11, _mm256_madd_epi16(_valval1, _w));
                        _sum22 = _mm256_add_epi32(_sum22, _mm256_madd_epi16(_valval2, _w));
                        _sum33 = _mm256_add_epi32(_sum33, _mm256_madd_epi16(_valval3, _w));
#endif
#else // __AVX2__
                        __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                        __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                        __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);

#if __XOP__
                        _sum00 = _mm_maddd_epi16(_r0, _w0, _sum00);
                        _sum10 = _mm_maddd_epi16(_r0, _w1, _sum10);
                        _sum01 = _mm_maddd_epi16(_r1, _w0, _sum01);
                        _sum11 = _mm_maddd_epi16(_r1, _w1, _sum11);
                        _sum02 = _mm_maddd_epi16(_r2, _w0, _sum02);
                        _sum12 = _mm_maddd_epi16(_r2, _w1, _sum12);
                        _sum03 = _mm_maddd_epi16(_r3, _w0, _sum03);
                        _sum13 = _mm_maddd_epi16(_r3, _w1, _sum13);
#else
                        _sum00 = _mm_add_epi32(_sum00, _mm_madd_epi16(_r0, _w0));
                        _sum10 = _mm_add_epi32(_sum10, _mm_madd_epi16(_r0, _w1));
                        _sum01 = _mm_add_epi32(_sum01, _mm_madd_epi16(_r1, _w0));
                        _sum11 = _mm_add_epi32(_sum11, _mm_madd_epi16(_r1, _w1));
                        _sum02 = _mm_add_epi32(_sum02, _mm_madd_epi16(_r2, _w0));
                        _sum12 = _mm_add_epi32(_sum12, _mm_madd_epi16(_r2, _w1));
                        _sum03 = _mm_add_epi32(_sum03, _mm_madd_epi16(_r3, _w0));
                        _sum13 = _mm_add_epi32(_sum13, _mm_madd_epi16(_r3, _w1));
#endif // __XOP__
#endif // __AVX2__

                        kptr += 16;
                    }
                }
#if __AVX2__
                __m256i _sum01 = _mm256_hadd_epi32(_sum00, _sum11);
                __m256i _sum23 = _mm256_hadd_epi32(_sum22, _sum33);
                __m256i _ss = _mm256_hadd_epi32(_sum01, _sum23);
                _sum0 = _mm_add_epi32(_sum0, _mm256_extracti128_si256(_ss, 0));
                _sum1 = _mm_add_epi32(_sum1, _mm256_extracti128_si256(_ss, 1));
#else
                // transpose 4x4
                __m128i _tmp00 = _mm_unpacklo_epi32(_sum00, _sum01);
                __m128i _tmp01 = _mm_unpacklo_epi32(_sum02, _sum03);
                __m128i _tmp02 = _mm_unpackhi_epi32(_sum00, _sum01);
                __m128i _tmp03 = _mm_unpackhi_epi32(_sum02, _sum03);
                __m128i _tmp10 = _mm_unpacklo_epi32(_sum10, _sum11);
                __m128i _tmp11 = _mm_unpacklo_epi32(_sum12, _sum13);
                __m128i _tmp12 = _mm_unpackhi_epi32(_sum10, _sum11);
                __m128i _tmp13 = _mm_unpackhi_epi32(_sum12, _sum13);
                _sum00 = _mm_unpacklo_epi64(_tmp00, _tmp01);
                _sum01 = _mm_unpackhi_epi64(_tmp00, _tmp01);
                _sum02 = _mm_unpacklo_epi64(_tmp02, _tmp03);
                _sum03 = _mm_unpackhi_epi64(_tmp02, _tmp03);
                _sum10 = _mm_unpacklo_epi64(_tmp10, _tmp11);
                _sum11 = _mm_unpackhi_epi64(_tmp10, _tmp11);
                _sum12 = _mm_unpacklo_epi64(_tmp12, _tmp13);
                _sum13 = _mm_unpackhi_epi64(_tmp12, _tmp13);
                _sum00 = _mm_add_epi32(_sum00, _sum01);
                _sum02 = _mm_add_epi32(_sum02, _sum03);
                _sum10 = _mm_add_epi32(_sum10, _sum11);
                _sum12 = _mm_add_epi32(_sum12, _sum13);
                _sum0 = _mm_add_epi32(_sum0, _sum00);
                _sum0 = _mm_add_epi32(_sum0, _sum02);
                _sum1 = _mm_add_epi32(_sum1, _sum10);
                _sum1 = _mm_add_epi32(_sum1, _sum12);
#endif // __AVX2__
            }
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;
                const signed char* r2 = bottom_blob.channel(q).row<const signed char>(i2 * stride_h) + j2 * stride_w;
                const signed char* r3 = bottom_blob.channel(q).row<const signed char>(i3 * stride_h) + j3 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    const signed char* r2s = r2 + space_ofs[k];
                    const signed char* r3s = r3 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _r = _mm_setr_epi16(r0s[0], r0s[N], r1s[0], r1s[N], r2s[0], r2s[N], r3s[0], r3s[N]);
                        __m128i _w0 = _mm_setr_epi16(kptr[0], kptr[2], kptr[0], kptr[2], kptr[0], kptr[2], kptr[0], kptr[2]);
                        __m128i _w1 = _mm_setr_epi16(kptr[1], kptr[3], kptr[1], kptr[3], kptr[1], kptr[3], kptr[1], kptr[3]);

#if __XOP__
                        _sum0 = _mm_maddd_epi16(_r, _w0, _sum0);
                        _sum1 = _mm_maddd_epi16(_r, _w1, _sum1);
#else
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_r, _w0));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_r, _w1));
#endif // __XOP__

                        kptr += 4;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;
                const signed char* r2 = bottom_blob.channel(q).row<const signed char>(i2 * stride_h) + j2 * stride_w;
                const signed char* r3 = bottom_blob.channel(q).row<const signed char>(i3 * stride_h) + j3 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    const signed char* r2s = r2 + space_ofs[k];
                    const signed char* r3s = r3 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _r = _mm_setr_epi16(r0s[0], r1s[0], r2s[0], r3s[0], r0s[0], r1s[0], r2s[0], r3s[0]);
                        __m128i _w = _mm_setr_epi16(kptr[0], kptr[0], kptr[0], kptr[0], kptr[1], kptr[1], kptr[1], kptr[1]);

                        __m128i _sl = _mm_mullo_epi16(_r, _w);
                        __m128i _sh = _mm_mulhi_epi16(_r, _w);
                        __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                        __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                        _sum0 = _mm_add_epi32(_sum0, _s0);
                        _sum1 = _mm_add_epi32(_sum1, _s1);

                        kptr += 2;
                    }
                }
            }

            _mm_store_si128((__m128i*)outptr0, _sum0);
            _mm_store_si128((__m128i*)outptr1, _sum1);
            outptr0 += 4;
            outptr1 += 4;
        }
#endif // __SSE2__
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            int sum00 = 0;
            int sum01 = 0;
            int sum10 = 0;
            int sum11 = 0;

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4 + (p % 4) / 2);
#elif __AVX2__
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#elif __SSE2__
            const signed char* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2);
#else
            const signed char* kptr = weight_data_tm.channel(p / 2);
#endif

            int q = 0;
#if __SSE2__
#if __AVX512F__
            {
                __m512i _sum0 = _mm512_setzero_si512();
                __m512i _sum1 = _mm512_setzero_si512();
                for (; q + 15 < inch; q += 16)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        if (elempack == 16)
                        {
                            _r0 = _mm_load_si128((const __m128i*)r0s);
                            _r1 = _mm_load_si128((const __m128i*)r1s);
                        }
                        else if (elempack == 8)
                        {
                            __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                            __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                            __m128i _t2 = _mm_loadl_epi64((const __m128i*)r1s);
                            __m128i _t3 = _mm_loadl_epi64((const __m128i*)(r1s + N));
                            _r0 = _mm_unpacklo_epi64(_t0, _t1);
                            _r1 = _mm_unpacklo_epi64(_t2, _t3);
                        }
                        else // if (elempack == 1)
                        {
                            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                            _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r1s), 1));
                        }

                        __m256i _val0 = _mm256_cvtepi8_epi16(_r0);
                        __m256i _val1 = _mm256_cvtepi8_epi16(_r1);
                        __m512i _valval0 = _mm512_inserti64x4(_mm512_castsi256_si512(_val0), _val0, 1);
                        __m512i _valval1 = _mm512_inserti64x4(_mm512_castsi256_si512(_val1), _val1, 1);

                        __m512i _w = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)kptr));

#if __AVX512VNNI__
                        _sum0 = _mm512_dpwssd_epi32(_sum0, _valval0, _w);
                        _sum1 = _mm512_dpwssd_epi32(_sum1, _valval1, _w);
#else
                        _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_valval0, _w));
                        _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_valval1, _w));
#endif // __AVX512VNNI__

                        kptr += 32;
                    }
                }
                __m256i _sum010 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum0, 0), _mm512_extracti64x4_epi64(_sum0, 1));
                __m256i _sum011 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum1, 0), _mm512_extracti64x4_epi64(_sum1, 1));
                __m128i _ss0 = _mm_add_epi32(_mm256_extracti128_si256(_sum010, 0), _mm256_extracti128_si256(_sum010, 1));
                __m128i _ss1 = _mm_add_epi32(_mm256_extracti128_si256(_sum011, 0), _mm256_extracti128_si256(_sum011, 1));

                sum00 += _mm_extract_epi32(_ss0, 0) + _mm_extract_epi32(_ss0, 1);
                sum10 += _mm_extract_epi32(_ss0, 2) + _mm_extract_epi32(_ss0, 3);
                sum01 += _mm_extract_epi32(_ss1, 0) + _mm_extract_epi32(_ss1, 1);
                sum11 += _mm_extract_epi32(_ss1, 2) + _mm_extract_epi32(_ss1, 3);
            }
#endif // __AVX512F__
            {
#if __AVX2__
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
#else
                __m128i _sum00 = _mm_setzero_si128();
                __m128i _sum10 = _mm_setzero_si128();
                __m128i _sum01 = _mm_setzero_si128();
                __m128i _sum11 = _mm_setzero_si128();
#endif
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        if (elempack == 8)
                        {
                            _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                            _r1 = _mm_loadl_epi64((const __m128i*)r1s);
                        }
                        else // if (elempack == 1)
                        {
#if __AVX2__
                            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
                            _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1), _sindex88);
                            __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1), _sindex88);
                            _r0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val0_32, 0), _mm256_extracti128_si256(_val0_32, 1));
                            _r1 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val1_32, 0), _mm256_extracti128_si256(_val1_32, 1));
#endif // __AVX512F__
#else
                            _r0 = _mm_setr_epi8(r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                            _r1 = _mm_setr_epi8(r1s[0], r1s[N], r1s[N * 2], r1s[N * 3], r1s[N * 4], r1s[N * 5], r1s[N * 6], r1s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
#endif // __AVX2__
                        }

#if __SSE4_1__
                        _r0 = _mm_cvtepi8_epi16(_r0);
                        _r1 = _mm_cvtepi8_epi16(_r1);
#else
                        _r0 = _mm_unpacklo_epi8(_r0, _mm_cmpgt_epi8(_mm_setzero_si128(), _r0));
                        _r1 = _mm_unpacklo_epi8(_r1, _mm_cmpgt_epi8(_mm_setzero_si128(), _r1));
#endif

                        __m128i _w01 = _mm_load_si128((const __m128i*)kptr);
#if __AVX2__
                        __m256i _w = _mm256_cvtepi8_epi16(_w01);

                        __m256i _valval0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r0, 1);
                        __m256i _valval1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r1), _r1, 1);

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum0 = _mm256_dpwssd_epi32(_sum0, _valval0, _w);
                        _sum1 = _mm256_dpwssd_epi32(_sum1, _valval1, _w);
#else
                        _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_valval0, _w));
                        _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_valval1, _w));
#endif
#else // __AVX2__
                        __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                        __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                        __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);

#if __XOP__
                        _sum00 = _mm_maddd_epi16(_r0, _w0, _sum00);
                        _sum10 = _mm_maddd_epi16(_r0, _w1, _sum10);
                        _sum01 = _mm_maddd_epi16(_r1, _w0, _sum01);
                        _sum11 = _mm_maddd_epi16(_r1, _w1, _sum11);
#else
                        _sum00 = _mm_add_epi32(_sum00, _mm_madd_epi16(_r0, _w0));
                        _sum10 = _mm_add_epi32(_sum10, _mm_madd_epi16(_r0, _w1));
                        _sum01 = _mm_add_epi32(_sum01, _mm_madd_epi16(_r1, _w0));
                        _sum11 = _mm_add_epi32(_sum11, _mm_madd_epi16(_r1, _w1));
#endif // __XOP__
#endif // __AVX2__

                        kptr += 16;
                    }
                }
#if __AVX2__
                sum00 += _mm_reduce_add_epi32(_mm256_extracti128_si256(_sum0, 0));
                sum10 += _mm_reduce_add_epi32(_mm256_extracti128_si256(_sum0, 1));
                sum01 += _mm_reduce_add_epi32(_mm256_extracti128_si256(_sum1, 0));
                sum11 += _mm_reduce_add_epi32(_mm256_extracti128_si256(_sum1, 1));
#else
                sum00 += _mm_reduce_add_epi32(_sum00);
                sum10 += _mm_reduce_add_epi32(_sum10);
                sum01 += _mm_reduce_add_epi32(_sum01);
                sum11 += _mm_reduce_add_epi32(_sum11);
#endif // __AVX2__
            }
#endif // __SSE2__
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum00 += r0s[0] * kptr[0];
                        sum10 += r0s[0] * kptr[1];
                        sum00 += r0s[N] * kptr[2];
                        sum10 += r0s[N] * kptr[3];
                        sum01 += r1s[0] * kptr[0];
                        sum11 += r1s[0] * kptr[1];
                        sum01 += r1s[N] * kptr[2];
                        sum11 += r1s[N] * kptr[3];

                        kptr += 4;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum00 += r0s[0] * kptr[0];
                        sum10 += r0s[0] * kptr[1];
                        sum01 += r1s[0] * kptr[0];
                        sum11 += r1s[0] * kptr[1];

                        kptr += 2;
                    }
                }
            }

            outptr0[0] = sum00;
            outptr0[1] = sum01;
            outptr1[0] = sum10;
            outptr1[1] = sum11;
            outptr0 += 2;
            outptr1 += 2;
        }
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int sum0 = 0;
            int sum1 = 0;

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4 + (p % 4) / 2);
#elif __AVX2__
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#elif __SSE2__
            const signed char* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2);
#else
            const signed char* kptr = weight_data_tm.channel(p / 2);
#endif

            int q = 0;
#if __SSE2__
#if __AVX512F__
            {
                __m512i _sum01 = _mm512_setzero_si512();
                for (; q + 15 < inch; q += 16)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];

                        __m128i _r0;
                        if (elempack == 16)
                        {
                            _r0 = _mm_load_si128((const __m128i*)r0s);
                        }
                        else if (elempack == 8)
                        {
                            __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                            __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                            _r0 = _mm_unpacklo_epi64(_t0, _t1);
                        }
                        else // if (elempack == 1)
                        {
                            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                        }

                        __m256i _val = _mm256_cvtepi8_epi16(_r0);
                        __m512i _valval = _mm512_inserti64x4(_mm512_castsi256_si512(_val), _val, 1);

                        __m512i _w = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)kptr));

#if __AVX512VNNI__
                        _sum01 = _mm512_dpwssd_epi32(_sum01, _valval, _w);
#else
                        _sum01 = _mm512_add_epi32(_sum01, _mm512_madd_epi16(_valval, _w));
#endif

                        kptr += 32;
                    }
                }
                __m256i _sum0101 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum01, 0), _mm512_extracti64x4_epi64(_sum01, 1));
                __m128i _ss = _mm_add_epi32(_mm256_extracti128_si256(_sum0101, 0), _mm256_extracti128_si256(_sum0101, 1));

                sum0 += _mm_extract_epi32(_ss, 0) + _mm_extract_epi32(_ss, 1);
                sum1 += _mm_extract_epi32(_ss, 2) + _mm_extract_epi32(_ss, 3);
            }
#endif // __AVX512F__
            {
#if __AVX2__
                __m256i _sum = _mm256_setzero_si256();
#else
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();
#endif
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];

                        __m128i _r0;
                        if (elempack == 8)
                        {
                            _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                        }
                        else // if (elempack == 1)
                        {
#if __AVX2__
                            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1), _sindex88);
                            _r0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val32, 0), _mm256_extracti128_si256(_val32, 1));
#endif // __AVX512F__
#else
                            _r0 = _mm_setr_epi8(r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
#endif // __AVX2__
                        }

#if __SSE4_1__
                        _r0 = _mm_cvtepi8_epi16(_r0);
#else
                        _r0 = _mm_unpacklo_epi8(_r0, _mm_cmpgt_epi8(_mm_setzero_si128(), _r0));
#endif

                        __m128i _w01 = _mm_load_si128((const __m128i*)kptr);
#if __AVX2__
                        __m256i _w = _mm256_cvtepi8_epi16(_w01);

                        __m256i _rr0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r0, 1);

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum = _mm256_dpwssd_epi32(_sum, _rr0, _w);
#else
                        _sum = _mm256_add_epi32(_sum, _mm256_madd_epi16(_rr0, _w));
#endif
#else // __AVX2__
                        __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                        __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                        __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);

#if __XOP__
                        _sum0 = _mm_maddd_epi16(_r0, _w0, _sum0);
                        _sum1 = _mm_maddd_epi16(_r0, _w1, _sum1);
#else
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_r0, _w0));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_r0, _w1));
#endif // __XOP__
#endif // __AVX2__

                        kptr += 16;
                    }
                }
#if __AVX2__
                sum0 += _mm_reduce_add_epi32(_mm256_extracti128_si256(_sum, 0));
                sum1 += _mm_reduce_add_epi32(_mm256_extracti128_si256(_sum, 1));
#else
                sum0 += _mm_reduce_add_epi32(_sum0);
                sum1 += _mm_reduce_add_epi32(_sum1);
#endif // __AVX2__
            }
#endif // __SSE2__
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum0 += r0s[0] * kptr[0];
                        sum1 += r0s[0] * kptr[1];
                        sum0 += r0s[N] * kptr[2];
                        sum1 += r0s[N] * kptr[3];

                        kptr += 4;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum0 += r0s[0] * kptr[0];
                        sum1 += r0s[0] * kptr[1];

                        kptr += 2;
                    }
                }
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr0 += 1;
            outptr1 += 1;
        }
    }
    remain_outch_start += nn_outch * 2;
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* outptr = top_blob.channel(p);

        int ij = 0;
#if __SSE2__
        for (; ij + 3 < outw * outh; ij += 4)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int i2 = (ij + 2) / outw;
            const int i3 = (ij + 3) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;
            const int j2 = (ij + 2) % outw;
            const int j3 = (ij + 3) % outw;

            __m128i _sum = _mm_setzero_si128();

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#elif __AVX2__
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#else
            const signed char* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2 + p % 2);
#endif

            int q = 0;
#if __AVX512F__
            {
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
                __m256i _sum2 = _mm256_setzero_si256();
                __m256i _sum3 = _mm256_setzero_si256();
                for (; q + 15 < inch; q += 16)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;
                    const signed char* r2 = bottom_blob.channel(q / elempack).row<const signed char>(i2 * stride_h) + j2 * stride_w * elempack;
                    const signed char* r3 = bottom_blob.channel(q / elempack).row<const signed char>(i3 * stride_h) + j3 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];
                        const signed char* r2s = r2 + space_ofs[k];
                        const signed char* r3s = r3 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        __m128i _r2;
                        __m128i _r3;
                        if (elempack == 16)
                        {
                            _r0 = _mm_load_si128((const __m128i*)r0s);
                            _r1 = _mm_load_si128((const __m128i*)r1s);
                            _r2 = _mm_load_si128((const __m128i*)r2s);
                            _r3 = _mm_load_si128((const __m128i*)r3s);
                        }
                        else if (elempack == 8)
                        {
                            __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                            __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                            __m128i _t2 = _mm_loadl_epi64((const __m128i*)r1s);
                            __m128i _t3 = _mm_loadl_epi64((const __m128i*)(r1s + N));
                            __m128i _t4 = _mm_loadl_epi64((const __m128i*)r2s);
                            __m128i _t5 = _mm_loadl_epi64((const __m128i*)(r2s + N));
                            __m128i _t6 = _mm_loadl_epi64((const __m128i*)r3s);
                            __m128i _t7 = _mm_loadl_epi64((const __m128i*)(r3s + N));
                            _r0 = _mm_unpacklo_epi64(_t0, _t1);
                            _r1 = _mm_unpacklo_epi64(_t2, _t3);
                            _r2 = _mm_unpacklo_epi64(_t4, _t5);
                            _r3 = _mm_unpacklo_epi64(_t6, _t7);
                        }
                        else // if (elempack == 1)
                        {
                            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));

                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                            _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r1s), 1));
                            _r2 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r2s), 1));
                            _r3 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r3s), 1));
                        }

                        __m256i _val0 = _mm256_cvtepi8_epi16(_r0);
                        __m256i _val1 = _mm256_cvtepi8_epi16(_r1);
                        __m256i _val2 = _mm256_cvtepi8_epi16(_r2);
                        __m256i _val3 = _mm256_cvtepi8_epi16(_r3);

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)kptr));

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum0 = _mm256_dpwssd_epi32(_sum0, _val0, _w);
                        _sum1 = _mm256_dpwssd_epi32(_sum1, _val1, _w);
                        _sum2 = _mm256_dpwssd_epi32(_sum2, _val2, _w);
                        _sum3 = _mm256_dpwssd_epi32(_sum3, _val3, _w);
#else
                        _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_val0, _w));
                        _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_val1, _w));
                        _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_val2, _w));
                        _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_val3, _w));
#endif

                        kptr += 16;
                    }
                }
                _sum0 = _mm256_hadd_epi32(_sum0, _sum1);
                _sum2 = _mm256_hadd_epi32(_sum2, _sum3);
                _sum0 = _mm256_hadd_epi32(_sum0, _sum2);
                _sum = _mm_add_epi32(_sum, _mm256_extracti128_si256(_sum0, 0));
                _sum = _mm_add_epi32(_sum, _mm256_extracti128_si256(_sum0, 1));
            }
#endif // __AVX512F__
            {
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();
                __m128i _sum2 = _mm_setzero_si128();
                __m128i _sum3 = _mm_setzero_si128();
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;
                    const signed char* r2 = bottom_blob.channel(q / elempack).row<const signed char>(i2 * stride_h) + j2 * stride_w * elempack;
                    const signed char* r3 = bottom_blob.channel(q / elempack).row<const signed char>(i3 * stride_h) + j3 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];
                        const signed char* r2s = r2 + space_ofs[k];
                        const signed char* r3s = r3 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        __m128i _r2;
                        __m128i _r3;
                        if (elempack == 8)
                        {
                            _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                            _r1 = _mm_loadl_epi64((const __m128i*)r1s);
                            _r2 = _mm_loadl_epi64((const __m128i*)r2s);
                            _r3 = _mm_loadl_epi64((const __m128i*)r3s);
                        }
                        else // if (elempack == 1)
                        {
#if __AVX2__
                            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
                            _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1));
                            _r2 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r2s, _vindex, 1));
                            _r3 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r3s, _vindex, 1));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);

                            __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1), _sindex88);
                            __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1), _sindex88);
                            __m256i _val2_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r2s, _vindex, 1), _sindex88);
                            __m256i _val3_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r3s, _vindex, 1), _sindex88);
                            _r0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val0_32, 0), _mm256_extracti128_si256(_val0_32, 1));
                            _r1 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val1_32, 0), _mm256_extracti128_si256(_val1_32, 1));
                            _r2 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val2_32, 0), _mm256_extracti128_si256(_val2_32, 1));
                            _r3 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val3_32, 0), _mm256_extracti128_si256(_val3_32, 1));
#endif
#else
                            _r0 = _mm_setr_epi8(r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                            _r1 = _mm_setr_epi8(r1s[0], r1s[N], r1s[N * 2], r1s[N * 3], r1s[N * 4], r1s[N * 5], r1s[N * 6], r1s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                            _r2 = _mm_setr_epi8(r2s[0], r2s[N], r2s[N * 2], r2s[N * 3], r2s[N * 4], r2s[N * 5], r2s[N * 6], r2s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                            _r3 = _mm_setr_epi8(r3s[0], r3s[N], r3s[N * 2], r3s[N * 3], r3s[N * 4], r3s[N * 5], r3s[N * 6], r3s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
#endif
                        }

#if __SSE4_1__
                        _r0 = _mm_cvtepi8_epi16(_r0);
                        _r1 = _mm_cvtepi8_epi16(_r1);
                        _r2 = _mm_cvtepi8_epi16(_r2);
                        _r3 = _mm_cvtepi8_epi16(_r3);
#else
                        _r0 = _mm_unpacklo_epi8(_r0, _mm_cmpgt_epi8(_mm_setzero_si128(), _r0));
                        _r1 = _mm_unpacklo_epi8(_r1, _mm_cmpgt_epi8(_mm_setzero_si128(), _r1));
                        _r2 = _mm_unpacklo_epi8(_r2, _mm_cmpgt_epi8(_mm_setzero_si128(), _r2));
                        _r3 = _mm_unpacklo_epi8(_r3, _mm_cmpgt_epi8(_mm_setzero_si128(), _r3));
#endif

                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
#if __SSE4_1__
                        _w = _mm_cvtepi8_epi16(_w);
#else
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
#endif

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum0 = _mm_dpwssd_epi32(_sum0, _r0, _w);
                        _sum1 = _mm_dpwssd_epi32(_sum1, _r1, _w);
                        _sum2 = _mm_dpwssd_epi32(_sum2, _r2, _w);
                        _sum3 = _mm_dpwssd_epi32(_sum3, _r3, _w);
#elif __XOP__
                        _sum0 = _mm_maddd_epi16(_r0, _w, _sum0);
                        _sum1 = _mm_maddd_epi16(_r1, _w, _sum1);
                        _sum2 = _mm_maddd_epi16(_r2, _w, _sum2);
                        _sum3 = _mm_maddd_epi16(_r3, _w, _sum3);
#else
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_r0, _w));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_r1, _w));
                        _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_r2, _w));
                        _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_r3, _w));
#endif

                        kptr += 8;
                    }
                }
#if __SSSE3__
                __m128i _ss = _mm_hadd_epi32(_mm_hadd_epi32(_sum0, _sum1), _mm_hadd_epi32(_sum2, _sum3));
                _sum = _mm_add_epi32(_sum, _ss);
#else
                // transpose 4x4
                __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum1);
                __m128i _tmp1 = _mm_unpacklo_epi32(_sum2, _sum3);
                __m128i _tmp2 = _mm_unpackhi_epi32(_sum0, _sum1);
                __m128i _tmp3 = _mm_unpackhi_epi32(_sum2, _sum3);
                _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                _sum1 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                _sum2 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                _sum3 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                _sum0 = _mm_add_epi32(_sum0, _sum1);
                _sum2 = _mm_add_epi32(_sum2, _sum3);
                _sum = _mm_add_epi32(_sum, _sum0);
                _sum = _mm_add_epi32(_sum, _sum2);
#endif // __SSSE3__
            }
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;
                const signed char* r2 = bottom_blob.channel(q).row<const signed char>(i2 * stride_h) + j2 * stride_w;
                const signed char* r3 = bottom_blob.channel(q).row<const signed char>(i3 * stride_h) + j3 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    const signed char* r2s = r2 + space_ofs[k];
                    const signed char* r3s = r3 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _r = _mm_setr_epi16(r0s[0], r0s[N], r1s[0], r1s[N], r2s[0], r2s[N], r3s[0], r3s[N]);
                        __m128i _w = _mm_setr_epi16(kptr[0], kptr[1], kptr[0], kptr[1], kptr[0], kptr[1], kptr[0], kptr[1]);

#if __XOP__
                        _sum = _mm_maddd_epi16(_r, _w, _sum);
#else
                        _sum = _mm_add_epi32(_sum, _mm_madd_epi16(_r, _w));
#endif // __XOP__

                        kptr += 2;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;
                const signed char* r2 = bottom_blob.channel(q).row<const signed char>(i2 * stride_h) + j2 * stride_w;
                const signed char* r3 = bottom_blob.channel(q).row<const signed char>(i3 * stride_h) + j3 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    const signed char* r2s = r2 + space_ofs[k];
                    const signed char* r3s = r3 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _w = _mm_set1_epi16(kptr[0]);

#if __XOP__
                        __m128i _r = _mm_setr_epi16(r0s[0], 0, r1s[0], 0, r2s[0], 0, r3s[0], 0);

                        _sum = _mm_maccd_epi16(_r, _w, _sum);
#else
                        __m128i _r = _mm_setr_epi16(r0s[0], r1s[0], r2s[0], r3s[0], 0, 0, 0, 0);

                        __m128i _sl = _mm_mullo_epi16(_r, _w);
                        __m128i _sh = _mm_mulhi_epi16(_r, _w);
                        __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);

                        _sum = _mm_add_epi32(_sum, _s0);
#endif // __XOP__

                        kptr += 1;
                    }
                }
            }

            _mm_store_si128((__m128i*)outptr, _sum);
            outptr += 4;
        }
#endif // __SSE2__
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            int sum0 = 0;
            int sum1 = 0;

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#elif __AVX2__
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#elif __SSE2__
            const signed char* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2 + p % 2);
#else
            const signed char* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

            int q = 0;
#if __SSE2__
#if __AVX512F__
            {
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
                for (; q + 15 < inch; q += 16)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        if (elempack == 16)
                        {
                            _r0 = _mm_load_si128((const __m128i*)r0s);
                            _r1 = _mm_load_si128((const __m128i*)r1s);
                        }
                        else if (elempack == 8)
                        {
                            __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                            __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                            __m128i _t2 = _mm_loadl_epi64((const __m128i*)r1s);
                            __m128i _t3 = _mm_loadl_epi64((const __m128i*)(r1s + N));
                            _r0 = _mm_unpacklo_epi64(_t0, _t1);
                            _r1 = _mm_unpacklo_epi64(_t2, _t3);
                        }
                        else // if (elempack == 1)
                        {
                            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));

                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                            _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r1s), 1));
                        }

                        __m256i _val0 = _mm256_cvtepi8_epi16(_r0);
                        __m256i _val1 = _mm256_cvtepi8_epi16(_r1);

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)kptr));

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum0 = _mm256_dpwssd_epi32(_sum0, _val0, _w);
                        _sum1 = _mm256_dpwssd_epi32(_sum1, _val1, _w);
#else
                        _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_val0, _w));
                        _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_val1, _w));
#endif

                        kptr += 16;
                    }
                }

                __m128i _ss0 = _mm_add_epi32(_mm256_extracti128_si256(_sum0, 0), _mm256_extracti128_si256(_sum0, 1));
                __m128i _ss1 = _mm_add_epi32(_mm256_extracti128_si256(_sum1, 0), _mm256_extracti128_si256(_sum1, 1));
                sum0 += _mm_reduce_add_epi32(_ss0);
                sum1 += _mm_reduce_add_epi32(_ss1);
            }
#endif // __AVX512F__
            {
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                    const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];

                        __m128i _r0;
                        __m128i _r1;
                        if (elempack == 8)
                        {
                            _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                            _r1 = _mm_loadl_epi64((const __m128i*)r1s);
                        }
                        else // if (elempack == 1)
                        {
#if __AVX2__
                            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
                            _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);

                            __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1), _sindex88);
                            __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, 1), _sindex88);
                            _r0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val0_32, 0), _mm256_extracti128_si256(_val0_32, 1));
                            _r1 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val1_32, 0), _mm256_extracti128_si256(_val1_32, 1));
#endif
#else
                            _r0 = _mm_setr_epi8(r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
                            _r1 = _mm_setr_epi8(r1s[0], r1s[N], r1s[N * 2], r1s[N * 3], r1s[N * 4], r1s[N * 5], r1s[N * 6], r1s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
#endif
                        }

#if __SSE4_1__
                        _r0 = _mm_cvtepi8_epi16(_r0);
                        _r1 = _mm_cvtepi8_epi16(_r1);
#else
                        _r0 = _mm_unpacklo_epi8(_r0, _mm_cmpgt_epi8(_mm_setzero_si128(), _r0));
                        _r1 = _mm_unpacklo_epi8(_r1, _mm_cmpgt_epi8(_mm_setzero_si128(), _r1));
#endif

                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
#if __SSE4_1__
                        _w = _mm_cvtepi8_epi16(_w);
#else
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
#endif

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum0 = _mm_dpwssd_epi32(_sum0, _r0, _w);
                        _sum1 = _mm_dpwssd_epi32(_sum1, _r1, _w);
#elif __XOP__
                        _sum0 = _mm_maddd_epi16(_r0, _w, _sum0);
                        _sum1 = _mm_maddd_epi16(_r1, _w, _sum1);
#else
                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_r0, _w));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_r1, _w));
#endif

                        kptr += 8;
                    }
                }
                sum0 += _mm_reduce_add_epi32(_sum0);
                sum1 += _mm_reduce_add_epi32(_sum1);
            }
#endif // __SSE2__
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum0 += r0s[0] * kptr[0];
                        sum0 += r0s[N] * kptr[1];
                        sum1 += r1s[0] * kptr[0];
                        sum1 += r1s[N] * kptr[1];

                        kptr += 2;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum0 += r0s[0] * kptr[0];
                        sum1 += r1s[0] * kptr[0];

                        kptr += 1;
                    }
                }
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
        }
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int sum = 0;

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#elif __AVX2__
            const signed char* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#elif __SSE2__
            const signed char* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2 + p % 2);
#else
            const signed char* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

            int q = 0;
#if __SSE2__
#if __AVX512F__
            {
                __m256i _sum = _mm256_setzero_si256();
                for (; q + 15 < inch; q += 16)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];

                        __m128i _r0;
                        if (elempack == 16)
                        {
                            _r0 = _mm_loadu_si128((const __m128i*)r0s);
                        }
                        else if (elempack == 8)
                        {
                            __m128i _t0 = _mm_loadl_epi64((const __m128i*)r0s);
                            __m128i _t1 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                            _r0 = _mm_unpacklo_epi64(_t0, _t1);
                        }
                        else // if (elempack == 1)
                        {
                            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, (const int*)(r0s), 1));
                        }

                        __m256i _val = _mm256_cvtepi8_epi16(_r0);

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)kptr));

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum = _mm256_dpwssd_epi32(_sum, _val, _w);
#else
                        _sum = _mm256_add_epi32(_sum, _mm256_madd_epi16(_val, _w));
#endif

                        kptr += 16;
                    }
                }

                __m128i _ss = _mm_add_epi32(_mm256_extracti128_si256(_sum, 0), _mm256_extracti128_si256(_sum, 1));
                sum += _mm_reduce_add_epi32(_ss);
            }
#endif // __AVX512F__
            {
                __m128i _sum = _mm_setzero_si128();
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];

                        __m128i _r0;
                        if (elempack == 8)
                        {
                            _r0 = _mm_loadl_epi64((const __m128i*)r0s);
                        }
                        else // if (elempack == 1)
                        {
#if __AVX2__
                            __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                            _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(N));
#if __AVX512F__
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, 1), _sindex88);
                            _r0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val32, 0), _mm256_extracti128_si256(_val32, 1));
#endif // __AVX512F__
#else
                            _r0 = _mm_setr_epi8(r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
#endif // __AVX2__
                        }

#if __SSE4_1__
                        _r0 = _mm_cvtepi8_epi16(_r0);
#else
                        _r0 = _mm_unpacklo_epi8(_r0, _mm_cmpgt_epi8(_mm_setzero_si128(), _r0));
#endif

                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
#if __SSE4_1__
                        _w = _mm_cvtepi8_epi16(_w);
#else
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
#endif

#if __AVXVNNI__ || __AVX512VNNI__
                        _sum = _mm_dpwssd_epi32(_sum, _r0, _w);
#elif __XOP__
                        _sum = _mm_maddd_epi16(_r0, _w, _sum);
#else
                        _sum = _mm_add_epi32(_sum, _mm_madd_epi16(_r0, _w));
#endif

                        kptr += 8;
                    }
                }
                sum += _mm_reduce_add_epi32(_sum);
            }
#endif // __SSE2__
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum += r0s[0] * kptr[0];
                        sum += r0s[N] * kptr[1];

                        kptr += 2;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum += r0s[0] * kptr[0];

                        kptr += 1;
                    }
                }
            }

            outptr[0] = sum;
            outptr += 1;
        }
    }
}
