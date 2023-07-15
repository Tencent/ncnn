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

static void convolution_transform_kernel_packed_int8(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
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
            kernel_tm.create(16 * 16 * maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 2 + inch % 2, outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else if (inch >= 8)
            kernel_tm.create(16 * 8 * maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else if (inch >= 2)
            kernel_tm.create(16 * 2 * maxk, inch / 2 + inch % 2, outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else
            kernel_tm.create(16 * maxk, inch, outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)1u);
    }
    else
#endif // __AVX512F__
    if (outch >= 8)
    {
#if __AVX512F__
        if (inch >= 16)
            kernel_tm.create(8 * 16 * maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else
#endif // __AVX512F__
        if (inch >= 8)
            kernel_tm.create(8 * 8 * maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else if (inch >= 2)
            kernel_tm.create(8 * 2 * maxk, inch / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else
            kernel_tm.create(8 * maxk, inch, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2, (size_t)1u);
    }
    else
#endif // __AVX2__
    if (outch >= 4)
    {
#if __AVX512F__
        if (inch >= 16)
            kernel_tm.create(4 * 16 * maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2);
        else
#endif // __AVX512F__
        if (inch >= 8)
            kernel_tm.create(4 * 8 * maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2);
        else if (inch >= 2)
            kernel_tm.create(4 * 2 * maxk, inch / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2);
        else
            kernel_tm.create(4 * maxk, inch, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)1u);
    }
    else
#endif // __SSE2__
    if (outch >= 2)
    {
#if __SSE2__
#if __AVX512F__
        if (inch >= 16)
            kernel_tm.create(2 * 16 * maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 2 + inch % 2, outch / 2 + outch % 2);
        else
#endif // __AVX512F__
        if (inch >= 8)
            kernel_tm.create(2 * 8 * maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch / 2 + outch % 2);
        else
#endif // __SSE2__
        if (inch >= 2)
            kernel_tm.create(2 * 2 * maxk, inch / 2 + inch % 2, outch / 2 + outch % 2);
        else
            kernel_tm.create(2 * maxk, inch, outch / 2 + outch % 2, (size_t)1u);
    }
    else
    {
#if __SSE2__
#if __AVX512F__
        if (inch >= 16)
            kernel_tm.create(16 * maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 2 + inch % 2, outch);
        else
#endif // __AVX512F__
        if (inch >= 8)
            kernel_tm.create(8 * maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch);
        else
#endif // __SSE2__
        if (inch >= 2)
            kernel_tm.create(2 * maxk, inch / 2 + inch % 2, outch);
        else
            kernel_tm.create(maxk, inch, outch, (size_t)1u);
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
        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(maxk));

        for (; p + 15 < inch; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                __m128i _w0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr0 + k, sizeof(signed char)));
                __m128i _w1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr1 + k, sizeof(signed char)));
                __m128i _w2 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr2 + k, sizeof(signed char)));
                __m128i _w3 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr3 + k, sizeof(signed char)));
                __m128i _w4 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr4 + k, sizeof(signed char)));
                __m128i _w5 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr5 + k, sizeof(signed char)));
                __m128i _w6 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr6 + k, sizeof(signed char)));
                __m128i _w7 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr7 + k, sizeof(signed char)));
                __m128i _w8 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr8 + k, sizeof(signed char)));
                __m128i _w9 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr9 + k, sizeof(signed char)));
                __m128i _wa = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptra + k, sizeof(signed char)));
                __m128i _wb = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptrb + k, sizeof(signed char)));
                __m128i _wc = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptrc + k, sizeof(signed char)));
                __m128i _wd = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptrd + k, sizeof(signed char)));
                __m128i _we = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptre + k, sizeof(signed char)));
                __m128i _wf = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptrf + k, sizeof(signed char)));

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
        __m256i _vindex256 = _mm512_extracti64x4_epi64(_vindex, 0);
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                __m128i _w0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr0 + k), _vindex256, sizeof(signed char)));
                __m128i _w1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr1 + k), _vindex256, sizeof(signed char)));
                __m128i _w2 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr2 + k), _vindex256, sizeof(signed char)));
                __m128i _w3 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr3 + k), _vindex256, sizeof(signed char)));
                __m128i _w4 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr4 + k), _vindex256, sizeof(signed char)));
                __m128i _w5 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr5 + k), _vindex256, sizeof(signed char)));
                __m128i _w6 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr6 + k), _vindex256, sizeof(signed char)));
                __m128i _w7 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr7 + k), _vindex256, sizeof(signed char)));
                __m128i _w8 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr8 + k), _vindex256, sizeof(signed char)));
                __m128i _w9 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr9 + k), _vindex256, sizeof(signed char)));
                __m128i _wa = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptra + k), _vindex256, sizeof(signed char)));
                __m128i _wb = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptrb + k), _vindex256, sizeof(signed char)));
                __m128i _wc = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptrc + k), _vindex256, sizeof(signed char)));
                __m128i _wd = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptrd + k), _vindex256, sizeof(signed char)));
                __m128i _we = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptre + k), _vindex256, sizeof(signed char)));
                __m128i _wf = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptrf + k), _vindex256, sizeof(signed char)));

                __m128i _w01 = _mm_unpacklo_epi64(_w0, _w1);
                __m128i _w23 = _mm_unpacklo_epi64(_w2, _w3);
                __m128i _w45 = _mm_unpacklo_epi64(_w4, _w5);
                __m128i _w67 = _mm_unpacklo_epi64(_w6, _w7);
                __m128i _w89 = _mm_unpacklo_epi64(_w8, _w9);
                __m128i _wab = _mm_unpacklo_epi64(_wa, _wb);
                __m128i _wcd = _mm_unpacklo_epi64(_wc, _wd);
                __m128i _wef = _mm_unpacklo_epi64(_we, _wf);

                _mm_storeu_si128((__m128i*)g00, _w01);
                _mm_storeu_si128((__m128i*)(g00 + 16), _w23);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 2), _w45);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 3), _w67);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 4), _w89);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 5), _wab);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 6), _wcd);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 7), _wef);
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
                const signed char* k8 = kptr8 + k;
                const signed char* k9 = kptr9 + k;
                const signed char* ka = kptra + k;
                const signed char* kb = kptrb + k;
                const signed char* kc = kptrc + k;
                const signed char* kd = kptrd + k;
                const signed char* ke = kptre + k;
                const signed char* kf = kptrf + k;

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
                g00[16] = k8[0];
                g00[17] = k8[maxk];
                g00[18] = k9[0];
                g00[19] = k9[maxk];
                g00[20] = ka[0];
                g00[21] = ka[maxk];
                g00[22] = kb[0];
                g00[23] = kb[maxk];
                g00[24] = kc[0];
                g00[25] = kc[maxk];
                g00[26] = kd[0];
                g00[27] = kd[maxk];
                g00[28] = ke[0];
                g00[29] = ke[maxk];
                g00[30] = kf[0];
                g00[31] = kf[maxk];
                g00 += 32;
            }

            kptr0 += maxk * 2;
            kptr1 += maxk * 2;
            kptr2 += maxk * 2;
            kptr3 += maxk * 2;
            kptr4 += maxk * 2;
            kptr5 += maxk * 2;
            kptr6 += maxk * 2;
            kptr7 += maxk * 2;
            kptr8 += maxk * 2;
            kptr9 += maxk * 2;
            kptra += maxk * 2;
            kptrb += maxk * 2;
            kptrc += maxk * 2;
            kptrd += maxk * 2;
            kptre += maxk * 2;
            kptrf += maxk * 2;
        }
        for (; p < inch; p++)
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
                const signed char* k8 = kptr8 + k;
                const signed char* k9 = kptr9 + k;
                const signed char* ka = kptra + k;
                const signed char* kb = kptrb + k;
                const signed char* kc = kptrc + k;
                const signed char* kd = kptrd + k;
                const signed char* ke = kptre + k;
                const signed char* kf = kptrf + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k2[0];
                g00[3] = k3[0];
                g00[4] = k4[0];
                g00[5] = k5[0];
                g00[6] = k6[0];
                g00[7] = k7[0];
                g00[8] = k8[0];
                g00[9] = k9[0];
                g00[10] = ka[0];
                g00[11] = kb[0];
                g00[12] = kc[0];
                g00[13] = kd[0];
                g00[14] = ke[0];
                g00[15] = kf[0];
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
        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(maxk));

        for (; p + 15 < inch; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                __m128i _w0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr0 + k, sizeof(signed char)));
                __m128i _w1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr1 + k, sizeof(signed char)));
                __m128i _w2 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr2 + k, sizeof(signed char)));
                __m128i _w3 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr3 + k, sizeof(signed char)));
                __m128i _w4 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr4 + k, sizeof(signed char)));
                __m128i _w5 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr5 + k, sizeof(signed char)));
                __m128i _w6 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr6 + k, sizeof(signed char)));
                __m128i _w7 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr7 + k, sizeof(signed char)));

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
        __m256i _vindex256 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _vindex256 = _mm256_mullo_epi32(_vindex256, _mm256_set1_epi32(maxk));
#if !__AVX512F__
        __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
        __m256i _pidx8 = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
#endif // !__AVX512F__
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
#if __AVX512F__
                __m128i _w0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr0 + k), _vindex256, sizeof(signed char)));
                __m128i _w1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr1 + k), _vindex256, sizeof(signed char)));
                __m128i _w2 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr2 + k), _vindex256, sizeof(signed char)));
                __m128i _w3 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr3 + k), _vindex256, sizeof(signed char)));
                __m128i _w4 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr4 + k), _vindex256, sizeof(signed char)));
                __m128i _w5 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr5 + k), _vindex256, sizeof(signed char)));
                __m128i _w6 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr6 + k), _vindex256, sizeof(signed char)));
                __m128i _w7 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)(kptr7 + k), _vindex256, sizeof(signed char)));

                __m128i _w01 = _mm_unpacklo_epi64(_w0, _w1);
                __m128i _w23 = _mm_unpacklo_epi64(_w2, _w3);
                __m128i _w45 = _mm_unpacklo_epi64(_w4, _w5);
                __m128i _w67 = _mm_unpacklo_epi64(_w6, _w7);

                _mm_storeu_si128((__m128i*)g00, _w01);
                _mm_storeu_si128((__m128i*)(g00 + 16), _w23);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 2), _w45);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 3), _w67);
#else
                __m256i _w0 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr0 + k), _vindex256, sizeof(signed char)), _sindex88);
                __m256i _w1 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr1 + k), _vindex256, sizeof(signed char)), _sindex88);
                __m256i _w2 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr2 + k), _vindex256, sizeof(signed char)), _sindex88);
                __m256i _w3 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr3 + k), _vindex256, sizeof(signed char)), _sindex88);
                __m256i _w4 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr4 + k), _vindex256, sizeof(signed char)), _sindex88);
                __m256i _w5 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr5 + k), _vindex256, sizeof(signed char)), _sindex88);
                __m256i _w6 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr6 + k), _vindex256, sizeof(signed char)), _sindex88);
                __m256i _w7 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)(kptr7 + k), _vindex256, sizeof(signed char)), _sindex88);

                __m256i _w01 = _mm256_unpacklo_epi32(_w0, _w1);
                __m256i _w23 = _mm256_unpacklo_epi32(_w2, _w3);
                __m256i _w45 = _mm256_unpacklo_epi32(_w4, _w5);
                __m256i _w67 = _mm256_unpacklo_epi32(_w6, _w7);
                __m256i _w0123 = _mm256_unpacklo_epi64(_w01, _w23);
                __m256i _w4567 = _mm256_unpacklo_epi64(_w45, _w67);

                _w0123 = _mm256_permutevar8x32_epi32(_w0123, _pidx8);
                _w4567 = _mm256_permutevar8x32_epi32(_w4567, _pidx8);
                _mm256_storeu_si256((__m256i*)g00, _w0123);
                _mm256_storeu_si256((__m256i*)(g00 + 32), _w4567);
#endif
                g00 += 64;
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
            }

            kptr0 += maxk * 2;
            kptr1 += maxk * 2;
            kptr2 += maxk * 2;
            kptr3 += maxk * 2;
            kptr4 += maxk * 2;
            kptr5 += maxk * 2;
            kptr6 += maxk * 2;
            kptr7 += maxk * 2;
        }
        for (; p < inch; p++)
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

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k2[0];
                g00[3] = k3[0];
                g00[4] = k4[0];
                g00[5] = k5[0];
                g00[6] = k6[0];
                g00[7] = k7[0];
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
        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(maxk));

        for (; p + 15 < inch; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                __m128i _w0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr0 + k, sizeof(signed char)));
                __m128i _w1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr1 + k, sizeof(signed char)));
                __m128i _w2 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr2 + k, sizeof(signed char)));
                __m128i _w3 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr3 + k, sizeof(signed char)));

                _mm_storeu_si128((__m128i*)g00, _w0);
                _mm_storeu_si128((__m128i*)(g00 + 16), _w1);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 2), _w2);
                _mm_storeu_si128((__m128i*)(g00 + 16 * 3), _w3);
                g00 += 64;
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

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k1[0];
                    k1 += maxk;
                    g00 += 1;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k2[0];
                    k2 += maxk;
                    g00 += 1;
                }
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k3[0];
                    k3 += maxk;
                    g00 += 1;
                }
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
            kptr2 += maxk * 8;
            kptr3 += maxk * 8;
        }
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
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
                g00 += 8;
            }

            kptr0 += maxk * 2;
            kptr1 += maxk * 2;
            kptr2 += maxk * 2;
            kptr3 += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;
                const signed char* k2 = kptr2 + k;
                const signed char* k3 = kptr3 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k2[0];
                g00[3] = k3[0];
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
        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(maxk));

        for (; p + 15 < inch; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                __m128i _w0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr0 + k, sizeof(signed char)));
                __m128i _w1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr1 + k, sizeof(signed char)));

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
            for (int k = 0; k < maxk; k++)
            {
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
                g00 += 16;
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
        }
#endif // __SSE2__
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    k0 += maxk;
                    k1 += maxk;
                    g00 += 2;
                }
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
        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(maxk));

        for (; p + 15 < inch; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                __m128i _w0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, kptr + k, sizeof(signed char)));

                _mm_storeu_si128((__m128i*)g00, _w0);
                g00 += 16;
            }

            kptr += maxk * 16;
        }
#endif // __AVX512F__
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr + k;

                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
            }

            kptr += maxk * 8;
        }
#endif // __SSE2__
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
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
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int inch = bottom_blob.c * elempack;

    const int N = bottom_blob.cstep * elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c * out_elempack;

    const int M = top_blob.cstep * out_elempack;

    NCNN_LOGE("convolution_packed_int8  %d @ %d  ->  %d @ %d", inch, elempack, outch, out_elempack);

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

        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            __m512i _sum0 = _mm512_setzero_si512();
            __m512i _sum1 = _mm512_setzero_si512();

            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8);

            int q = 0;
            {
                __m512i _sum010 = _mm512_setzero_si512();
                __m512i _sum230 = _mm512_setzero_si512();
                __m512i _sum450 = _mm512_setzero_si512();
                __m512i _sum670 = _mm512_setzero_si512();
                __m512i _sum890 = _mm512_setzero_si512();
                __m512i _sumab0 = _mm512_setzero_si512();
                __m512i _sumcd0 = _mm512_setzero_si512();
                __m512i _sumef0 = _mm512_setzero_si512();
                __m512i _sum011 = _mm512_setzero_si512();
                __m512i _sum231 = _mm512_setzero_si512();
                __m512i _sum451 = _mm512_setzero_si512();
                __m512i _sum671 = _mm512_setzero_si512();
                __m512i _sum891 = _mm512_setzero_si512();
                __m512i _sumab1 = _mm512_setzero_si512();
                __m512i _sumcd1 = _mm512_setzero_si512();
                __m512i _sumef1 = _mm512_setzero_si512();
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
                            _r0 = _mm_loadu_si128((const __m128i*)r0s);
                            _r1 = _mm_loadu_si128((const __m128i*)r1s);
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
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r0s, sizeof(signed char)));
                            _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r1s, sizeof(signed char)));
                        }

                        __m256i _val0 = _mm256_cvtepi8_epi16(_r0);
                        __m256i _val1 = _mm256_cvtepi8_epi16(_r1);
                        __m512i _valval0 = _mm512_inserti64x4(_mm512_castsi256_si512(_val0), _val0, 1);
                        __m512i _valval1 = _mm512_inserti64x4(_mm512_castsi256_si512(_val1), _val1, 1);

                        __m512i _w01 = _mm512_loadu_si512((const __m512i*)kptr);
                        __m512i _w23 = _mm512_loadu_si512((const __m512i*)(kptr + 64));
                        __m512i _w45 = _mm512_loadu_si512((const __m512i*)(kptr + 128));
                        __m512i _w67 = _mm512_loadu_si512((const __m512i*)(kptr + 192));
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                        __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                        __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));
                        __m512i _w4 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w45, 0));
                        __m512i _w5 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w45, 1));
                        __m512i _w6 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w67, 0));
                        __m512i _w7 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w67, 1));

                        _sum010 = _mm512_add_epi32(_sum010, _mm512_madd_epi16(_valval0, _w0));
                        _sum230 = _mm512_add_epi32(_sum230, _mm512_madd_epi16(_valval0, _w1));
                        _sum450 = _mm512_add_epi32(_sum450, _mm512_madd_epi16(_valval0, _w2));
                        _sum670 = _mm512_add_epi32(_sum670, _mm512_madd_epi16(_valval0, _w3));
                        _sum890 = _mm512_add_epi32(_sum890, _mm512_madd_epi16(_valval0, _w4));
                        _sumab0 = _mm512_add_epi32(_sumab0, _mm512_madd_epi16(_valval0, _w5));
                        _sumcd0 = _mm512_add_epi32(_sumcd0, _mm512_madd_epi16(_valval0, _w6));
                        _sumef0 = _mm512_add_epi32(_sumef0, _mm512_madd_epi16(_valval0, _w7));
                        _sum011 = _mm512_add_epi32(_sum011, _mm512_madd_epi16(_valval1, _w0));
                        _sum231 = _mm512_add_epi32(_sum231, _mm512_madd_epi16(_valval1, _w1));
                        _sum451 = _mm512_add_epi32(_sum451, _mm512_madd_epi16(_valval1, _w2));
                        _sum671 = _mm512_add_epi32(_sum671, _mm512_madd_epi16(_valval1, _w3));
                        _sum891 = _mm512_add_epi32(_sum891, _mm512_madd_epi16(_valval1, _w4));
                        _sumab1 = _mm512_add_epi32(_sumab1, _mm512_madd_epi16(_valval1, _w5));
                        _sumcd1 = _mm512_add_epi32(_sumcd1, _mm512_madd_epi16(_valval1, _w6));
                        _sumef1 = _mm512_add_epi32(_sumef1, _mm512_madd_epi16(_valval1, _w7));

                        kptr += 256;
                    }
                }
                __m256i _sum01010 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum010, 0), _mm512_extracti64x4_epi64(_sum010, 1));
                __m256i _sum23230 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum230, 0), _mm512_extracti64x4_epi64(_sum230, 1));
                __m256i _sum45450 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum450, 0), _mm512_extracti64x4_epi64(_sum450, 1));
                __m256i _sum67670 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum670, 0), _mm512_extracti64x4_epi64(_sum670, 1));
                __m256i _sum89890 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum890, 0), _mm512_extracti64x4_epi64(_sum890, 1));
                __m256i _sumabab0 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sumab0, 0), _mm512_extracti64x4_epi64(_sumab0, 1));
                __m256i _sumcdcd0 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sumcd0, 0), _mm512_extracti64x4_epi64(_sumcd0, 1));
                __m256i _sumefef0 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sumef0, 0), _mm512_extracti64x4_epi64(_sumef0, 1));
                __m256i _sum01011 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum011, 0), _mm512_extracti64x4_epi64(_sum011, 1));
                __m256i _sum23231 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum231, 0), _mm512_extracti64x4_epi64(_sum231, 1));
                __m256i _sum45451 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum451, 0), _mm512_extracti64x4_epi64(_sum451, 1));
                __m256i _sum67671 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum671, 0), _mm512_extracti64x4_epi64(_sum671, 1));
                __m256i _sum89891 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum891, 0), _mm512_extracti64x4_epi64(_sum891, 1));
                __m256i _sumabab1 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sumab1, 0), _mm512_extracti64x4_epi64(_sumab1, 1));
                __m256i _sumcdcd1 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sumcd1, 0), _mm512_extracti64x4_epi64(_sumcd1, 1));
                __m256i _sumefef1 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sumef1, 0), _mm512_extracti64x4_epi64(_sumef1, 1));
                __m256i _ss00 = _mm256_hadd_epi32(_sum01010, _sum23230);
                __m256i _ss10 = _mm256_hadd_epi32(_sum45450, _sum67670);
                __m256i _ss20 = _mm256_hadd_epi32(_sum89890, _sumabab0);
                __m256i _ss30 = _mm256_hadd_epi32(_sumcdcd0, _sumefef0);
                __m256i _ss01 = _mm256_hadd_epi32(_sum01011, _sum23231);
                __m256i _ss11 = _mm256_hadd_epi32(_sum45451, _sum67671);
                __m256i _ss21 = _mm256_hadd_epi32(_sum89891, _sumabab1);
                __m256i _ss31 = _mm256_hadd_epi32(_sumcdcd1, _sumefef1);
                __m256i _sss00 = _mm256_add_epi32(_mm256_permute2x128_si256(_ss00, _ss10, _MM_SHUFFLE(0, 2, 0, 0)), _mm256_permute2x128_si256(_ss00, _ss10, _MM_SHUFFLE(0, 3, 0, 1)));
                __m256i _sss10 = _mm256_add_epi32(_mm256_permute2x128_si256(_ss20, _ss30, _MM_SHUFFLE(0, 2, 0, 0)), _mm256_permute2x128_si256(_ss20, _ss30, _MM_SHUFFLE(0, 3, 0, 1)));
                __m256i _sss01 = _mm256_add_epi32(_mm256_permute2x128_si256(_ss01, _ss11, _MM_SHUFFLE(0, 2, 0, 0)), _mm256_permute2x128_si256(_ss01, _ss11, _MM_SHUFFLE(0, 3, 0, 1)));
                __m256i _sss11 = _mm256_add_epi32(_mm256_permute2x128_si256(_ss21, _ss31, _MM_SHUFFLE(0, 2, 0, 0)), _mm256_permute2x128_si256(_ss21, _ss31, _MM_SHUFFLE(0, 3, 0, 1)));
                _sum0 = _mm512_add_epi32(_sum0, _mm512_inserti64x4(_mm512_castsi256_si512(_sss00), _sss10, 1));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_inserti64x4(_mm512_castsi256_si512(_sss01), _sss11, 1));
            }
            {
                __m512i _sum01230 = _mm512_setzero_si512();
                __m512i _sum45670 = _mm512_setzero_si512();
                __m512i _sum89ab0 = _mm512_setzero_si512();
                __m512i _sumcdef0 = _mm512_setzero_si512();
                __m512i _sum01231 = _mm512_setzero_si512();
                __m512i _sum45671 = _mm512_setzero_si512();
                __m512i _sum89ab1 = _mm512_setzero_si512();
                __m512i _sumcdef1 = _mm512_setzero_si512();
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
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)));
                            _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, sizeof(signed char)));
                        }

                        _r0 = _mm_cvtepi8_epi16(_r0);
                        _r1 = _mm_cvtepi8_epi16(_r1);
                        __m512i _val0 = _mm512_broadcast_i32x4(_r0);
                        __m512i _val1 = _mm512_broadcast_i32x4(_r1);

                        __m512i _w01 = _mm512_loadu_si512((const __m512i*)kptr);
                        __m512i _w23 = _mm512_loadu_si512((const __m512i*)(kptr + 64));
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                        __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                        __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));

                        _sum01230 = _mm512_add_epi32(_sum01230, _mm512_madd_epi16(_val0, _w0));
                        _sum45670 = _mm512_add_epi32(_sum45670, _mm512_madd_epi16(_val0, _w1));
                        _sum89ab0 = _mm512_add_epi32(_sum89ab0, _mm512_madd_epi16(_val0, _w2));
                        _sumcdef0 = _mm512_add_epi32(_sumcdef0, _mm512_madd_epi16(_val0, _w3));
                        _sum01231 = _mm512_add_epi32(_sum01231, _mm512_madd_epi16(_val1, _w0));
                        _sum45671 = _mm512_add_epi32(_sum45671, _mm512_madd_epi16(_val1, _w1));
                        _sum89ab1 = _mm512_add_epi32(_sum89ab1, _mm512_madd_epi16(_val1, _w2));
                        _sumcdef1 = _mm512_add_epi32(_sumcdef1, _mm512_madd_epi16(_val1, _w3));

                        kptr += 128;
                    }
                }
                __m256i _sum02130 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum01230, 0), _mm512_extracti64x4_epi64(_sum01230, 1));
                __m256i _sum46570 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum45670, 0), _mm512_extracti64x4_epi64(_sum45670, 1));
                __m256i _sum8a9b0 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum89ab0, 0), _mm512_extracti64x4_epi64(_sum89ab0, 1));
                __m256i _sumcedf0 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sumcdef0, 0), _mm512_extracti64x4_epi64(_sumcdef0, 1));
                __m256i _sum02131 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum01231, 0), _mm512_extracti64x4_epi64(_sum01231, 1));
                __m256i _sum46571 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum45671, 0), _mm512_extracti64x4_epi64(_sum45671, 1));
                __m256i _sum8a9b1 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum89ab1, 0), _mm512_extracti64x4_epi64(_sum89ab1, 1));
                __m256i _sumcedf1 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sumcdef1, 0), _mm512_extracti64x4_epi64(_sumcdef1, 1));
                __m256i _sum024613570 = _mm256_hadd_epi32(_sum02130, _sum46570);
                __m256i _sum8ace9bdf0 = _mm256_hadd_epi32(_sum8a9b0, _sumcedf0);
                __m256i _sum024613571 = _mm256_hadd_epi32(_sum02131, _sum46571);
                __m256i _sum8ace9bdf1 = _mm256_hadd_epi32(_sum8a9b1, _sumcedf1);
                __m512i _ss0 = _mm512_inserti64x4(_mm512_castsi256_si512(_sum024613570), _sum8ace9bdf0, 1);
                __m512i _ss1 = _mm512_inserti64x4(_mm512_castsi256_si512(_sum024613571), _sum8ace9bdf1, 1);
                __m512i _ssi = _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
                _sum0 = _mm512_add_epi32(_sum0, _mm512_permutexvar_epi32(_ssi, _ss0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_permutexvar_epi32(_ssi, _ss1));
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

                        __m512i _w = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)kptr));

                        _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_r0, _w));
                        _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_r1, _w));

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

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)kptr));

                        _sum0 = _mm512_add_epi32(_sum0, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_r0, _w)));
                        _sum1 = _mm512_add_epi32(_sum1, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_r1, _w)));

                        kptr += 16;
                    }
                }
            }

            if (out_elempack == 16)
            {
                _mm512_storeu_si512((__m512i*)outptr, _sum0);
                _mm512_storeu_si512((__m512i*)(outptr + 16), _sum1);
                outptr += 32;
            }
            if (out_elempack == 8)
            {
                _mm256_storeu_si256((__m256i*)outptr, _mm512_extracti32x8_epi32(_sum0, 0));
                _mm256_storeu_si256((__m256i*)(outptr + M), _mm512_extracti32x8_epi32(_sum0, 1));
                _mm256_storeu_si256((__m256i*)(outptr + 8), _mm512_extracti32x8_epi32(_sum1, 0));
                _mm256_storeu_si256((__m256i*)(outptr + M + 8), _mm512_extracti32x8_epi32(_sum1, 1));
                outptr += 16;
            }
            if (out_elempack == 4)
            {
                _mm_storeu_si128((__m128i*)outptr, _mm512_extracti32x4_epi32(_sum0, 0));
                _mm_storeu_si128((__m128i*)(outptr + M), _mm512_extracti32x4_epi32(_sum0, 1));
                _mm_storeu_si128((__m128i*)(outptr + M * 2), _mm512_extracti32x4_epi32(_sum0, 2));
                _mm_storeu_si128((__m128i*)(outptr + M * 3), _mm512_extracti32x4_epi32(_sum0, 3));
                _mm_storeu_si128((__m128i*)(outptr + 4), _mm512_extracti32x4_epi32(_sum1, 0));
                _mm_storeu_si128((__m128i*)(outptr + M + 4), _mm512_extracti32x4_epi32(_sum1, 1));
                _mm_storeu_si128((__m128i*)(outptr + M * 2 + 4), _mm512_extracti32x4_epi32(_sum1, 2));
                _mm_storeu_si128((__m128i*)(outptr + M * 3 + 4), _mm512_extracti32x4_epi32(_sum1, 3));
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

            __m512i _sum = _mm512_setzero_si512();

            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8);

            int q = 0;
            {
                __m512i _sum01 = _mm512_setzero_si512();
                __m512i _sum23 = _mm512_setzero_si512();
                __m512i _sum45 = _mm512_setzero_si512();
                __m512i _sum67 = _mm512_setzero_si512();
                __m512i _sum89 = _mm512_setzero_si512();
                __m512i _sumab = _mm512_setzero_si512();
                __m512i _sumcd = _mm512_setzero_si512();
                __m512i _sumef = _mm512_setzero_si512();
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
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r0s, sizeof(signed char)));
                        }

                        __m256i _val = _mm256_cvtepi8_epi16(_r0);
                        __m512i _valval = _mm512_inserti64x4(_mm512_castsi256_si512(_val), _val, 1);

                        __m512i _w01 = _mm512_loadu_si512((const __m512i*)kptr);
                        __m512i _w23 = _mm512_loadu_si512((const __m512i*)(kptr + 64));
                        __m512i _w45 = _mm512_loadu_si512((const __m512i*)(kptr + 128));
                        __m512i _w67 = _mm512_loadu_si512((const __m512i*)(kptr + 192));
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                        __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                        __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));
                        __m512i _w4 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w45, 0));
                        __m512i _w5 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w45, 1));
                        __m512i _w6 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w67, 0));
                        __m512i _w7 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w67, 1));

                        _sum01 = _mm512_add_epi32(_sum01, _mm512_madd_epi16(_valval, _w0));
                        _sum23 = _mm512_add_epi32(_sum23, _mm512_madd_epi16(_valval, _w1));
                        _sum45 = _mm512_add_epi32(_sum45, _mm512_madd_epi16(_valval, _w2));
                        _sum67 = _mm512_add_epi32(_sum67, _mm512_madd_epi16(_valval, _w3));
                        _sum89 = _mm512_add_epi32(_sum89, _mm512_madd_epi16(_valval, _w4));
                        _sumab = _mm512_add_epi32(_sumab, _mm512_madd_epi16(_valval, _w5));
                        _sumcd = _mm512_add_epi32(_sumcd, _mm512_madd_epi16(_valval, _w6));
                        _sumef = _mm512_add_epi32(_sumef, _mm512_madd_epi16(_valval, _w7));

                        kptr += 256;
                    }
                }
                __m256i _sum0101 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum01, 0), _mm512_extracti64x4_epi64(_sum01, 1));
                __m256i _sum2323 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum23, 0), _mm512_extracti64x4_epi64(_sum23, 1));
                __m256i _sum4545 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum45, 0), _mm512_extracti64x4_epi64(_sum45, 1));
                __m256i _sum6767 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum67, 0), _mm512_extracti64x4_epi64(_sum67, 1));
                __m256i _sum8989 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum89, 0), _mm512_extracti64x4_epi64(_sum89, 1));
                __m256i _sumabab = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sumab, 0), _mm512_extracti64x4_epi64(_sumab, 1));
                __m256i _sumcdcd = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sumcd, 0), _mm512_extracti64x4_epi64(_sumcd, 1));
                __m256i _sumefef = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sumef, 0), _mm512_extracti64x4_epi64(_sumef, 1));
                __m256i _ss = _mm256_hadd_epi32(_sum0101, _sum2323);
                __m256i _ss2 = _mm256_hadd_epi32(_sum4545, _sum6767);
                __m256i _ss3 = _mm256_hadd_epi32(_sum8989, _sumabab);
                __m256i _ss4 = _mm256_hadd_epi32(_sumcdcd, _sumefef);
                __m256i _sss = _mm256_add_epi32(_mm256_permute2x128_si256(_ss, _ss2, _MM_SHUFFLE(0, 2, 0, 0)), _mm256_permute2x128_si256(_ss, _ss2, _MM_SHUFFLE(0, 3, 0, 1)));
                __m256i _sss2 = _mm256_add_epi32(_mm256_permute2x128_si256(_ss3, _ss4, _MM_SHUFFLE(0, 2, 0, 0)), _mm256_permute2x128_si256(_ss3, _ss4, _MM_SHUFFLE(0, 3, 0, 1)));
                _sum = _mm512_add_epi32(_sum, _mm512_inserti64x4(_mm512_castsi256_si512(_sss), _sss2, 1));
            }
            {
                __m512i _sum0123 = _mm512_setzero_si512();
                __m512i _sum4567 = _mm512_setzero_si512();
                __m512i _sum89ab = _mm512_setzero_si512();
                __m512i _sumcdef = _mm512_setzero_si512();
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
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)));
                        }

                        __m128i _val = _mm_cvtepi8_epi16(_r0);
                        __m512i _val4 = _mm512_broadcast_i32x4(_val);

                        __m512i _w01 = _mm512_loadu_si512((const __m512i*)kptr);
                        __m512i _w23 = _mm512_loadu_si512((const __m512i*)(kptr + 64));
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                        __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                        __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));

                        _sum0123 = _mm512_add_epi32(_sum0123, _mm512_madd_epi16(_val4, _w0));
                        _sum4567 = _mm512_add_epi32(_sum4567, _mm512_madd_epi16(_val4, _w1));
                        _sum89ab = _mm512_add_epi32(_sum89ab, _mm512_madd_epi16(_val4, _w2));
                        _sumcdef = _mm512_add_epi32(_sumcdef, _mm512_madd_epi16(_val4, _w3));

                        kptr += 128;
                    }
                }
                __m256i _sum0213 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum0123, 0), _mm512_extracti64x4_epi64(_sum0123, 1));
                __m256i _sum4657 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum4567, 0), _mm512_extracti64x4_epi64(_sum4567, 1));
                __m256i _sum8a9b = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum89ab, 0), _mm512_extracti64x4_epi64(_sum89ab, 1));
                __m256i _sumcedf = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sumcdef, 0), _mm512_extracti64x4_epi64(_sumcdef, 1));
                __m256i _sum02461357 = _mm256_hadd_epi32(_sum0213, _sum4657);
                __m256i _sum8ace9bdf = _mm256_hadd_epi32(_sum8a9b, _sumcedf);
                __m512i _ss0 = _mm512_inserti64x4(_mm512_castsi256_si512(_sum02461357), _sum8ace9bdf, 1);
                __m512i _ssi = _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
                _sum = _mm512_add_epi32(_sum, _mm512_permutexvar_epi32(_ssi, _ss0));
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

                        __m512i _w = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)kptr));

                        _sum = _mm512_add_epi32(_sum, _mm512_madd_epi16(_val, _w));

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

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)kptr));

                        _sum = _mm512_add_epi32(_sum, _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_val, _w)));

                        kptr += 16;
                    }
                }
            }

            if (out_elempack == 16)
            {
                _mm512_storeu_si512((__m512i*)outptr, _sum);
                outptr += 16;
            }
            if (out_elempack == 8)
            {
                _mm256_storeu_si256((__m256i*)outptr, _mm512_extracti32x8_epi32(_sum, 0));
                _mm256_storeu_si256((__m256i*)(outptr + M), _mm512_extracti32x8_epi32(_sum, 1));
                outptr += 8;
            }
            if (out_elempack == 4)
            {
                _mm_storeu_si128((__m128i*)outptr, _mm512_extracti32x4_epi32(_sum, 0));
                _mm_storeu_si128((__m128i*)(outptr + M), _mm512_extracti32x4_epi32(_sum, 1));
                _mm_storeu_si128((__m128i*)(outptr + M * 2), _mm512_extracti32x4_epi32(_sum, 2));
                _mm_storeu_si128((__m128i*)(outptr + M * 3), _mm512_extracti32x4_epi32(_sum, 3));
                outptr += 4;
            }
            if (out_elempack == 1)
            {
                __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(M));
                _mm512_i32scatter_epi32(outptr, _vindex, _sum, sizeof(int));
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

        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            __m256i _sum0 = _mm256_setzero_si256();
            __m256i _sum1 = _mm256_setzero_si256();

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8);
#else
            const signed char* kptr = weight_data_tm.channel(p / 8);
#endif

            int q = 0;
#if __AVX512F__
            {
                __m512i _sum010 = _mm512_setzero_si512();
                __m512i _sum230 = _mm512_setzero_si512();
                __m512i _sum450 = _mm512_setzero_si512();
                __m512i _sum670 = _mm512_setzero_si512();
                __m512i _sum011 = _mm512_setzero_si512();
                __m512i _sum231 = _mm512_setzero_si512();
                __m512i _sum451 = _mm512_setzero_si512();
                __m512i _sum671 = _mm512_setzero_si512();
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
                            _r0 = _mm_loadu_si128((const __m128i*)r0s);
                            _r1 = _mm_loadu_si128((const __m128i*)r1s);
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
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r0s, sizeof(signed char)));
                            _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r1s, sizeof(signed char)));
                        }

                        __m256i _val0 = _mm256_cvtepi8_epi16(_r0);
                        __m256i _val1 = _mm256_cvtepi8_epi16(_r1);
                        __m512i _valval0 = _mm512_inserti64x4(_mm512_castsi256_si512(_val0), _val0, 1);
                        __m512i _valval1 = _mm512_inserti64x4(_mm512_castsi256_si512(_val1), _val1, 1);

                        __m512i _w01 = _mm512_loadu_si512((const __m512i*)kptr);
                        __m512i _w23 = _mm512_loadu_si512((const __m512i*)(kptr + 64));
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                        __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                        __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));

                        _sum010 = _mm512_add_epi32(_sum010, _mm512_madd_epi16(_valval0, _w0));
                        _sum230 = _mm512_add_epi32(_sum230, _mm512_madd_epi16(_valval0, _w1));
                        _sum450 = _mm512_add_epi32(_sum450, _mm512_madd_epi16(_valval0, _w2));
                        _sum670 = _mm512_add_epi32(_sum670, _mm512_madd_epi16(_valval0, _w3));
                        _sum011 = _mm512_add_epi32(_sum011, _mm512_madd_epi16(_valval1, _w0));
                        _sum231 = _mm512_add_epi32(_sum231, _mm512_madd_epi16(_valval1, _w1));
                        _sum451 = _mm512_add_epi32(_sum451, _mm512_madd_epi16(_valval1, _w2));
                        _sum671 = _mm512_add_epi32(_sum671, _mm512_madd_epi16(_valval1, _w3));

                        kptr += 128;
                    }
                }
                __m256i _sum01010 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum010, 0), _mm512_extracti64x4_epi64(_sum010, 1));
                __m256i _sum23230 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum230, 0), _mm512_extracti64x4_epi64(_sum230, 1));
                __m256i _sum45450 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum450, 0), _mm512_extracti64x4_epi64(_sum450, 1));
                __m256i _sum67670 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum670, 0), _mm512_extracti64x4_epi64(_sum670, 1));
                __m256i _sum01011 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum011, 0), _mm512_extracti64x4_epi64(_sum011, 1));
                __m256i _sum23231 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum231, 0), _mm512_extracti64x4_epi64(_sum231, 1));
                __m256i _sum45451 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum451, 0), _mm512_extracti64x4_epi64(_sum451, 1));
                __m256i _sum67671 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum671, 0), _mm512_extracti64x4_epi64(_sum671, 1));
                __m256i _ss0 = _mm256_hadd_epi32(_sum01010, _sum23230);
                __m256i _ss1 = _mm256_hadd_epi32(_sum45450, _sum67670);
                __m256i _ss2 = _mm256_hadd_epi32(_sum01011, _sum23231);
                __m256i _ss3 = _mm256_hadd_epi32(_sum45451, _sum67671);
                _sum0 = _mm256_add_epi32(_sum0, _mm256_permute2x128_si256(_ss0, _ss1, _MM_SHUFFLE(0, 2, 0, 0)));
                _sum0 = _mm256_add_epi32(_sum0, _mm256_permute2x128_si256(_ss0, _ss1, _MM_SHUFFLE(0, 3, 0, 1)));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_permute2x128_si256(_ss2, _ss3, _MM_SHUFFLE(0, 2, 0, 0)));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_permute2x128_si256(_ss2, _ss3, _MM_SHUFFLE(0, 3, 0, 1)));
            }
#endif // __AVX512F__
            {
                __m256i _sum010 = _mm256_setzero_si256();
                __m256i _sum230 = _mm256_setzero_si256();
                __m256i _sum450 = _mm256_setzero_si256();
                __m256i _sum670 = _mm256_setzero_si256();
                __m256i _sum011 = _mm256_setzero_si256();
                __m256i _sum231 = _mm256_setzero_si256();
                __m256i _sum451 = _mm256_setzero_si256();
                __m256i _sum671 = _mm256_setzero_si256();
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
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)));
                            _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, sizeof(signed char)));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)), _sindex88);
                            __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, sizeof(signed char)), _sindex88);
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
                        __m256i _rr0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r0, 1);
                        __m256i _rr1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r1), _r1, 1);

                        __m256i _w01 = _mm256_loadu_si256((const __m256i*)kptr);
                        __m256i _w23 = _mm256_loadu_si256((const __m256i*)(kptr + 32));
                        __m256i _w0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w01, 0));
                        __m256i _w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w01, 1));
                        __m256i _w2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w23, 0));
                        __m256i _w3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w23, 1));

                        _sum010 = _mm256_add_epi32(_sum010, _mm256_madd_epi16(_rr0, _w0));
                        _sum230 = _mm256_add_epi32(_sum230, _mm256_madd_epi16(_rr0, _w1));
                        _sum450 = _mm256_add_epi32(_sum450, _mm256_madd_epi16(_rr0, _w2));
                        _sum670 = _mm256_add_epi32(_sum670, _mm256_madd_epi16(_rr0, _w3));
                        _sum011 = _mm256_add_epi32(_sum011, _mm256_madd_epi16(_rr1, _w0));
                        _sum231 = _mm256_add_epi32(_sum231, _mm256_madd_epi16(_rr1, _w1));
                        _sum451 = _mm256_add_epi32(_sum451, _mm256_madd_epi16(_rr1, _w2));
                        _sum671 = _mm256_add_epi32(_sum671, _mm256_madd_epi16(_rr1, _w3));

                        kptr += 64;
                    }
                }
                __m256i _sum02130 = _mm256_hadd_epi32(_sum010, _sum230);
                __m256i _sum46570 = _mm256_hadd_epi32(_sum450, _sum670);
                __m256i _sum02131 = _mm256_hadd_epi32(_sum011, _sum231);
                __m256i _sum46571 = _mm256_hadd_epi32(_sum451, _sum671);
                __m256i _sum024613570 = _mm256_hadd_epi32(_sum02130, _sum46570);
                __m256i _sum024613571 = _mm256_hadd_epi32(_sum02131, _sum46571);
                __m256i _ssi = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
                _sum0 = _mm256_add_epi32(_sum0, _mm256_permutevar8x32_epi32(_sum024613570, _ssi));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_permutevar8x32_epi32(_sum024613571, _ssi));
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

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)kptr));

                        _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_rr0, _w));
                        _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_rr1, _w));

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

            if (out_elempack == 8)
            {
                _mm256_storeu_si256((__m256i*)outptr, _sum0);
                _mm256_storeu_si256((__m256i*)(outptr + 8), _sum1);
                outptr += 16;
            }
            if (out_elempack == 4)
            {
                _mm_storeu_si128((__m128i*)outptr, _mm256_extracti128_si256(_sum0, 0));
                _mm_storeu_si128((__m128i*)(outptr + M), _mm256_extracti128_si256(_sum0, 1));
                _mm_storeu_si128((__m128i*)(outptr + 4), _mm256_extracti128_si256(_sum1, 0));
                _mm_storeu_si128((__m128i*)(outptr + M + 4), _mm256_extracti128_si256(_sum1, 1));
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

            __m256i _sum = _mm256_setzero_si256();

#if __AVX512F__
            const signed char* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8);
#else
            const signed char* kptr = weight_data_tm.channel(p / 8);
#endif

            int q = 0;
#if __AVX512F__
            {
                __m512i _sum01 = _mm512_setzero_si512();
                __m512i _sum23 = _mm512_setzero_si512();
                __m512i _sum45 = _mm512_setzero_si512();
                __m512i _sum67 = _mm512_setzero_si512();
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
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r0s, sizeof(signed char)));
                        }

                        __m256i _val = _mm256_cvtepi8_epi16(_r0);
                        __m512i _valval = _mm512_inserti64x4(_mm512_castsi256_si512(_val), _val, 1);

                        __m512i _w01 = _mm512_loadu_si512((const __m512i*)kptr);
                        __m512i _w23 = _mm512_loadu_si512((const __m512i*)(kptr + 64));
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w01, 1));
                        __m512i _w2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 0));
                        __m512i _w3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w23, 1));

                        _sum01 = _mm512_add_epi32(_sum01, _mm512_madd_epi16(_valval, _w0));
                        _sum23 = _mm512_add_epi32(_sum23, _mm512_madd_epi16(_valval, _w1));
                        _sum45 = _mm512_add_epi32(_sum45, _mm512_madd_epi16(_valval, _w2));
                        _sum67 = _mm512_add_epi32(_sum67, _mm512_madd_epi16(_valval, _w3));

                        kptr += 128;
                    }
                }
                __m256i _sum0101 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum01, 0), _mm512_extracti64x4_epi64(_sum01, 1));
                __m256i _sum2323 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum23, 0), _mm512_extracti64x4_epi64(_sum23, 1));
                __m256i _sum4545 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum45, 0), _mm512_extracti64x4_epi64(_sum45, 1));
                __m256i _sum6767 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum67, 0), _mm512_extracti64x4_epi64(_sum67, 1));
                __m256i _ss = _mm256_hadd_epi32(_sum0101, _sum2323);
                __m256i _ss2 = _mm256_hadd_epi32(_sum4545, _sum6767);
                _sum = _mm256_add_epi32(_sum, _mm256_permute2x128_si256(_ss, _ss2, _MM_SHUFFLE(0, 2, 0, 0)));
                _sum = _mm256_add_epi32(_sum, _mm256_permute2x128_si256(_ss, _ss2, _MM_SHUFFLE(0, 3, 0, 1)));
            }
#endif // __AVX512F__
            {
                __m256i _sum01 = _mm256_setzero_si256();
                __m256i _sum23 = _mm256_setzero_si256();
                __m256i _sum45 = _mm256_setzero_si256();
                __m256i _sum67 = _mm256_setzero_si256();
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
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)), _sindex88);
                            _r0 = _mm_unpacklo_epi32(_mm256_extracti128_si256(_val32, 0), _mm256_extracti128_si256(_val32, 1));
#endif // __AVX512F__
#else
                            _r0 = _mm_setr_epi8(r0s[0], r0s[N], r0s[N * 2], r0s[N * 3], r0s[N * 4], r0s[N * 5], r0s[N * 6], r0s[N * 7], 0, 0, 0, 0, 0, 0, 0, 0);
#endif // __AVX2__
                        }

                        __m128i _val = _mm_cvtepi8_epi16(_r0);
                        __m256i _valval = _mm256_inserti128_si256(_mm256_castsi128_si256(_val), _val, 1);

                        __m256i _w01 = _mm256_loadu_si256((const __m256i*)kptr);
                        __m256i _w23 = _mm256_loadu_si256((const __m256i*)(kptr + 32));
                        __m256i _w0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w01, 0));
                        __m256i _w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w01, 1));
                        __m256i _w2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w23, 0));
                        __m256i _w3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w23, 1));

                        _sum01 = _mm256_add_epi32(_sum01, _mm256_madd_epi16(_valval, _w0));
                        _sum23 = _mm256_add_epi32(_sum23, _mm256_madd_epi16(_valval, _w1));
                        _sum45 = _mm256_add_epi32(_sum45, _mm256_madd_epi16(_valval, _w2));
                        _sum67 = _mm256_add_epi32(_sum67, _mm256_madd_epi16(_valval, _w3));

                        kptr += 64;
                    }
                }
                __m256i _sum0213 = _mm256_hadd_epi32(_sum01, _sum23);
                __m256i _sum4657 = _mm256_hadd_epi32(_sum45, _sum67);
                __m256i _sum02461357 = _mm256_hadd_epi32(_sum0213, _sum4657);
                __m256i _ssi = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
                _sum = _mm256_add_epi32(_sum, _mm256_permutevar8x32_epi32(_sum02461357, _ssi));
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

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)kptr));

                        _sum = _mm256_add_epi32(_sum, _mm256_madd_epi16(_val, _w));

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
                        _sum = _mm256_add_epi32(_sum, _s0);

                        kptr += 8;
                    }
                }
            }

            if (out_elempack == 8)
            {
                _mm256_storeu_si256((__m256i*)outptr, _sum);
                outptr += 8;
            }
            if (out_elempack == 4)
            {
                _mm_storeu_si128((__m128i*)outptr, _mm256_extracti128_si256(_sum, 0));
                _mm_storeu_si128((__m128i*)(outptr + M), _mm256_extracti128_si256(_sum, 1));
                outptr += 4;
            }
            if (out_elempack == 1)
            {
#if __AVX512F__
                __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(M));
                _mm256_i32scatter_epi32(outptr, _vindex, _sum, sizeof(int));
#else
                int sum[8];
                _mm256_storeu_si256((__m256i*)sum, _sum);

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

        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            __m128i _sum0 = _mm_setzero_si128();
            __m128i _sum1 = _mm_setzero_si128();

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
                __m512i _sum10 = _mm512_setzero_si512();
                __m512i _sum01 = _mm512_setzero_si512();
                __m512i _sum11 = _mm512_setzero_si512();
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
                            _r0 = _mm_loadu_si128((const __m128i*)r0s);
                            _r1 = _mm_loadu_si128((const __m128i*)r1s);
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
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r0s, sizeof(signed char)));
                            _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r1s, sizeof(signed char)));
                        }

                        __m256i _val0 = _mm256_cvtepi8_epi16(_r0);
                        __m256i _val1 = _mm256_cvtepi8_epi16(_r1);
                        __m512i _valval0 = _mm512_inserti64x4(_mm512_castsi256_si512(_val0), _val0, 1);
                        __m512i _valval1 = _mm512_inserti64x4(_mm512_castsi256_si512(_val1), _val1, 1);

                        __m512i _w = _mm512_loadu_si512((const __m512i*)kptr);
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w, 1));

                        _sum00 = _mm512_add_epi32(_sum00, _mm512_madd_epi16(_valval0, _w0));
                        _sum10 = _mm512_add_epi32(_sum10, _mm512_madd_epi16(_valval0, _w1));
                        _sum01 = _mm512_add_epi32(_sum01, _mm512_madd_epi16(_valval1, _w0));
                        _sum11 = _mm512_add_epi32(_sum11, _mm512_madd_epi16(_valval1, _w1));

                        kptr += 64;
                    }
                }
                __m256i _ss00 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum00, 0), _mm512_extracti64x4_epi64(_sum00, 1));
                __m256i _ss10 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum10, 0), _mm512_extracti64x4_epi64(_sum10, 1));
                __m256i _ss01 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum01, 0), _mm512_extracti64x4_epi64(_sum01, 1));
                __m256i _ss11 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum11, 0), _mm512_extracti64x4_epi64(_sum11, 1));
                __m256i _ss0 = _mm256_hadd_epi32(_ss00, _ss10);
                __m256i _ss1 = _mm256_hadd_epi32(_ss01, _ss11);
                _sum0 = _mm_add_epi32(_sum0, _mm256_extracti128_si256(_ss0, 0));
                _sum0 = _mm_add_epi32(_sum0, _mm256_extracti128_si256(_ss0, 1));
                _sum1 = _mm_add_epi32(_sum1, _mm256_extracti128_si256(_ss1, 0));
                _sum1 = _mm_add_epi32(_sum1, _mm256_extracti128_si256(_ss1, 1));
            }
#endif // __AVX512F__
            {
#if __AVX2__
                __m256i _sum00 = _mm256_setzero_si256();
                __m256i _sum10 = _mm256_setzero_si256();
                __m256i _sum01 = _mm256_setzero_si256();
                __m256i _sum11 = _mm256_setzero_si256();
#else
                __m128i _sum00 = _mm_setzero_si128();
                __m128i _sum10 = _mm_setzero_si128();
                __m128i _sum20 = _mm_setzero_si128();
                __m128i _sum30 = _mm_setzero_si128();
                __m128i _sum01 = _mm_setzero_si128();
                __m128i _sum11 = _mm_setzero_si128();
                __m128i _sum21 = _mm_setzero_si128();
                __m128i _sum31 = _mm_setzero_si128();
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
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)));
                            _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, sizeof(signed char)));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)), _sindex88);
                            __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, sizeof(signed char)), _sindex88);
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
                        __m256i _valval0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r0, 1);
                        __m256i _valval1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r1), _r1, 1);

                        __m256i _w = _mm256_loadu_si256((const __m256i*)kptr);
                        __m256i _w0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w, 0));
                        __m256i _w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w, 1));

                        _sum00 = _mm256_add_epi32(_sum00, _mm256_madd_epi16(_valval0, _w0));
                        _sum10 = _mm256_add_epi32(_sum10, _mm256_madd_epi16(_valval0, _w1));
                        _sum01 = _mm256_add_epi32(_sum01, _mm256_madd_epi16(_valval1, _w0));
                        _sum11 = _mm256_add_epi32(_sum11, _mm256_madd_epi16(_valval1, _w1));
#else // __AVX2__
                        __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr);
                        __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr + 16));
                        __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                        __m128i _extw23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w23);
                        __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                        __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);
                        __m128i _w2 = _mm_unpacklo_epi8(_w23, _extw23);
                        __m128i _w3 = _mm_unpackhi_epi8(_w23, _extw23);

                        _sum00 = _mm_add_epi32(_sum00, _mm_madd_epi16(_r0, _w0));
                        _sum10 = _mm_add_epi32(_sum10, _mm_madd_epi16(_r0, _w1));
                        _sum20 = _mm_add_epi32(_sum20, _mm_madd_epi16(_r0, _w2));
                        _sum30 = _mm_add_epi32(_sum30, _mm_madd_epi16(_r0, _w3));
                        _sum01 = _mm_add_epi32(_sum01, _mm_madd_epi16(_r1, _w0));
                        _sum11 = _mm_add_epi32(_sum11, _mm_madd_epi16(_r1, _w1));
                        _sum21 = _mm_add_epi32(_sum21, _mm_madd_epi16(_r1, _w2));
                        _sum31 = _mm_add_epi32(_sum31, _mm_madd_epi16(_r1, _w3));
#endif // __AVX2__

                        kptr += 32;
                    }
                }
#if __AVX2__
                __m256i _sum0213_0 = _mm256_hadd_epi32(_sum00, _sum10);
                __m256i _sum0213_1 = _mm256_hadd_epi32(_sum01, _sum11);
                __m128i _ss0 = _mm_hadd_epi32(_mm256_extracti128_si256(_sum0213_0, 0), _mm256_extracti128_si256(_sum0213_0, 1));
                __m128i _ss1 = _mm_hadd_epi32(_mm256_extracti128_si256(_sum0213_1, 0), _mm256_extracti128_si256(_sum0213_1, 1));
                _ss0 = _mm_shuffle_epi32(_ss0, _MM_SHUFFLE(3, 1, 2, 0));
                _ss1 = _mm_shuffle_epi32(_ss1, _MM_SHUFFLE(3, 1, 2, 0));
#else
                __m128i _sum02_0 = _mm_add_epi32(_mm_unpacklo_epi32(_sum00, _sum20), _mm_unpackhi_epi32(_sum00, _sum20));
                __m128i _sum13_0 = _mm_add_epi32(_mm_unpacklo_epi32(_sum10, _sum30), _mm_unpackhi_epi32(_sum10, _sum30));
                __m128i _sum02_1 = _mm_add_epi32(_mm_unpacklo_epi32(_sum01, _sum21), _mm_unpackhi_epi32(_sum01, _sum21));
                __m128i _sum13_1 = _mm_add_epi32(_mm_unpacklo_epi32(_sum11, _sum31), _mm_unpackhi_epi32(_sum11, _sum31));
                __m128i _ss0 = _mm_add_epi32(_mm_unpacklo_epi32(_sum02_0, _sum13_0), _mm_unpackhi_epi32(_sum02_0, _sum13_0));
                __m128i _ss1 = _mm_add_epi32(_mm_unpacklo_epi32(_sum02_1, _sum13_1), _mm_unpackhi_epi32(_sum02_1, _sum13_1));
#endif // __AVX2__
                _sum0 = _mm_add_epi32(_sum0, _ss0);
                _sum1 = _mm_add_epi32(_sum1, _ss1);
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

                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_r0, _w));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_r1, _w));

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

                        __m128i _sl0 = _mm_mullo_epi16(_r0, _w);
                        __m128i _sh0 = _mm_mulhi_epi16(_r0, _w);
                        __m128i _sl1 = _mm_mullo_epi16(_r1, _w);
                        __m128i _sh1 = _mm_mulhi_epi16(_r1, _w);
                        __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                        __m128i _s1 = _mm_unpacklo_epi16(_sl1, _sh1);

                        _sum0 = _mm_add_epi32(_sum0, _s0);
                        _sum1 = _mm_add_epi32(_sum1, _s1);

                        kptr += 4;
                    }
                }
            }

            if (out_elempack == 4)
            {
                _mm_storeu_si128((__m128i*)outptr, _sum0);
                _mm_storeu_si128((__m128i*)(outptr + 4), _sum1);
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

            __m128i _sum = _mm_setzero_si128();

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
                __m512i _sum01 = _mm512_setzero_si512();
                __m512i _sum23 = _mm512_setzero_si512();
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
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r0s, sizeof(signed char)));
                        }

                        __m256i _val = _mm256_cvtepi8_epi16(_r0);
                        __m512i _valval = _mm512_inserti64x4(_mm512_castsi256_si512(_val), _val, 1);

                        __m512i _w = _mm512_loadu_si512((const __m512i*)kptr);
                        __m512i _w0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w, 0));
                        __m512i _w1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(_w, 1));

                        _sum01 = _mm512_add_epi32(_sum01, _mm512_madd_epi16(_valval, _w0));
                        _sum23 = _mm512_add_epi32(_sum23, _mm512_madd_epi16(_valval, _w1));

                        kptr += 64;
                    }
                }
                __m256i _sum0101 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum01, 0), _mm512_extracti64x4_epi64(_sum01, 1));
                __m256i _sum2323 = _mm256_hadd_epi32(_mm512_extracti64x4_epi64(_sum23, 0), _mm512_extracti64x4_epi64(_sum23, 1));
                __m256i _ss = _mm256_hadd_epi32(_sum0101, _sum2323);
                _sum = _mm_add_epi32(_sum, _mm256_extracti128_si256(_ss, 0));
                _sum = _mm_add_epi32(_sum, _mm256_extracti128_si256(_ss, 1));
            }
#endif // __AVX512F__
            {
#if __AVX2__
                __m256i _sum0 = _mm256_setzero_si256();
                __m256i _sum1 = _mm256_setzero_si256();
#else
                __m128i _sum0 = _mm_setzero_si128();
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
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)), _sindex88);
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
                        __m256i _rr0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r0, 1);

                        __m256i _w = _mm256_loadu_si256((const __m256i*)kptr);
                        __m256i _w0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w, 0));
                        __m256i _w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(_w, 1));

                        _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_rr0, _w0));
                        _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_rr0, _w1));
#else // __AVX2__
                        __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr);
                        __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr + 16));
                        __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                        __m128i _extw23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w23);
                        __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                        __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);
                        __m128i _w2 = _mm_unpacklo_epi8(_w23, _extw23);
                        __m128i _w3 = _mm_unpackhi_epi8(_w23, _extw23);

                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_r0, _w0));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_r0, _w1));
                        _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_r0, _w2));
                        _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_r0, _w3));
#endif // __AVX2__

                        kptr += 32;
                    }
                }
#if __AVX2__
                __m256i _sum0213 = _mm256_hadd_epi32(_sum0, _sum1);
                __m128i _ss = _mm_hadd_epi32(_mm256_extracti128_si256(_sum0213, 0), _mm256_extracti128_si256(_sum0213, 1));
                _ss = _mm_shuffle_epi32(_ss, _MM_SHUFFLE(3, 1, 2, 0));
#else
                __m128i _sum02 = _mm_add_epi32(_mm_unpacklo_epi32(_sum0, _sum2), _mm_unpackhi_epi32(_sum0, _sum2));
                __m128i _sum13 = _mm_add_epi32(_mm_unpacklo_epi32(_sum1, _sum3), _mm_unpackhi_epi32(_sum1, _sum3));
                __m128i _ss = _mm_add_epi32(_mm_unpacklo_epi32(_sum02, _sum13), _mm_unpackhi_epi32(_sum02, _sum13));
#endif // __AVX2__
                _sum = _mm_add_epi32(_sum, _ss);
            }
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        __m128i _val = _mm_setr_epi16(r0s[0], r0s[N], r0s[0], r0s[N], r0s[0], r0s[N], r0s[0], r0s[N]);

                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
#if __SSE4_1__
                        _w = _mm_cvtepi8_epi16(_w);
#else
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
#endif

                        _sum = _mm_add_epi32(_sum, _mm_madd_epi16(_val, _w));

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
                        __m128i _val = _mm_set1_epi16(r0s[0]);

                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
#if __SSE4_1__
                        _w = _mm_cvtepi8_epi16(_w);
#else
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
#endif

                        __m128i _sl = _mm_mullo_epi16(_val, _w);
                        __m128i _sh = _mm_mulhi_epi16(_val, _w);
                        __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);

                        _sum = _mm_add_epi32(_sum, _s0);

                        kptr += 4;
                    }
                }
            }

            if (out_elempack == 4)
            {
                _mm_storeu_si128((__m128i*)outptr, _sum);
                outptr += 4;
            }
            if (out_elempack == 1)
            {
#if __AVX512F__
                __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
                _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(M));
                _mm_i32scatter_epi32(outptr, _vindex, _sum, sizeof(int));
#else
                int sum[4];
                _mm_storeu_si128((__m128i*)sum, _sum);

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

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);

        int ij = 0;
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
                            _r0 = _mm_loadu_si128((const __m128i*)r0s);
                            _r1 = _mm_loadu_si128((const __m128i*)r1s);
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
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r0s, sizeof(signed char)));
                            _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r1s, sizeof(signed char)));
                        }

                        __m256i _val0 = _mm256_cvtepi8_epi16(_r0);
                        __m256i _val1 = _mm256_cvtepi8_epi16(_r1);
                        __m512i _valval0 = _mm512_inserti64x4(_mm512_castsi256_si512(_val0), _val0, 1);
                        __m512i _valval1 = _mm512_inserti64x4(_mm512_castsi256_si512(_val1), _val1, 1);

                        __m512i _w = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)kptr));

                        _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_valval0, _w));
                        _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_valval1, _w));

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
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)));
                            _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, sizeof(signed char)));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)), _sindex88);
                            __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, sizeof(signed char)), _sindex88);
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
                        __m256i _valval0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r0, 1);
                        __m256i _valval1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r1), _r1, 1);

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)kptr));

                        _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_valval0, _w));
                        _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_valval1, _w));
#else // __AVX2__
                        __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr);
                        __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                        __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                        __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);

                        _sum00 = _mm_add_epi32(_sum00, _mm_madd_epi16(_r0, _w0));
                        _sum10 = _mm_add_epi32(_sum10, _mm_madd_epi16(_r0, _w1));
                        _sum01 = _mm_add_epi32(_sum01, _mm_madd_epi16(_r1, _w0));
                        _sum11 = _mm_add_epi32(_sum11, _mm_madd_epi16(_r1, _w1));
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
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r0s, sizeof(signed char)));
                        }

                        __m256i _val = _mm256_cvtepi8_epi16(_r0);
                        __m512i _valval = _mm512_inserti64x4(_mm512_castsi256_si512(_val), _val, 1);

                        __m512i _w = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)kptr));

                        _sum01 = _mm512_add_epi32(_sum01, _mm512_madd_epi16(_valval, _w));

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
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)), _sindex88);
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
                        __m256i _rr0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r0, 1);

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)kptr));

                        _sum = _mm256_add_epi32(_sum, _mm256_madd_epi16(_rr0, _w));
#else // __AVX2__
                        __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr);
                        __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                        __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                        __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);

                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_r0, _w0));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_r0, _w1));
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
                            _r0 = _mm_loadu_si128((const __m128i*)r0s);
                            _r1 = _mm_loadu_si128((const __m128i*)r1s);
                        }
                        else if (elempack == 8)
                        {
                            __m128i _r00 = _mm_loadl_epi64((const __m128i*)r0s);
                            __m128i _r01 = _mm_loadl_epi64((const __m128i*)(r0s + N));
                            __m128i _r10 = _mm_loadl_epi64((const __m128i*)r1s);
                            __m128i _r11 = _mm_loadl_epi64((const __m128i*)(r1s + N));
                            _r0 = _mm_unpacklo_epi64(_r00, _r01);
                            _r1 = _mm_unpacklo_epi64(_r10, _r11);
                        }
                        else // if (elempack == 1)
                        {
                            __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                            _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(N));

                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r0s, sizeof(signed char)));
                            _r1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r1s, sizeof(signed char)));
                        }

                        __m256i _val0 = _mm256_cvtepi8_epi16(_r0);
                        __m256i _val1 = _mm256_cvtepi8_epi16(_r1);

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)kptr));

                        _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_val0, _w));
                        _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_val1, _w));

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
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)));
                            _r1 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, sizeof(signed char)));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);

                            __m256i _val0_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)), _sindex88);
                            __m256i _val1_32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r1s, _vindex, sizeof(signed char)), _sindex88);
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

                        _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_r0, _w));
                        _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_r1, _w));

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
                            _r0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_vindex, r0s, sizeof(signed char)));
                        }

                        __m256i _val = _mm256_cvtepi8_epi16(_r0);

                        __m256i _w = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)kptr));

                        _sum = _mm256_add_epi32(_sum, _mm256_madd_epi16(_val, _w));

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
                            _r0 = _mm256_cvtepi32_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)));
#else
                            __m128i _sindex8 = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m256i _sindex88 = _mm256_inserti128_si256(_mm256_castsi128_si256(_sindex8), _sindex8, 1);
                            __m256i _val32 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((const int*)r0s, _vindex, sizeof(signed char)), _sindex88);
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

                        _sum = _mm_add_epi32(_sum, _mm_madd_epi16(_r0, _w));

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
