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

static void convolution_transform_kernel_packed(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb

    // clang-format off
    // *INDENT-OFF*
#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (outch >= 16)
    {
        if (inch >= 16)
            kernel_tm.create(16 * 16 * maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else if (inch >= 8)
            kernel_tm.create(16 * 8 * maxk, inch / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else if (inch >= 4)
            kernel_tm.create(16 * 4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else if (inch >= 2)
            kernel_tm.create(16 * 2 * maxk, inch / 2 + inch % 2, outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else
            kernel_tm.create(16 * maxk, inch, outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
    }
    else
#endif // __AVX512F__
    if (outch >= 8)
    {
#if __AVX512F__
        if (inch >= 16)
            kernel_tm.create(8 * 16 * maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else
#endif // __AVX512F__
        if (inch >= 8)
            kernel_tm.create(8 * 8 * maxk, inch / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else if (inch >= 4)
            kernel_tm.create(8 * 4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else if (inch >= 2)
            kernel_tm.create(8 * 2 * maxk, inch / 2 + inch % 2, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
        else
            kernel_tm.create(8 * maxk, inch, outch / 8 + (outch % 8) / 4 + (outch % 4) / 2 + outch % 2);
    }
    else
#endif // __AVX__
    if (outch >= 4)
    {
#if __AVX__
#if __AVX512F__
        if (inch >= 16)
            kernel_tm.create(4 * 16 * maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2);
        else
#endif // __AVX512F__
        if (inch >= 8)
            kernel_tm.create(4 * 8 * maxk, inch / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2);
        else
#endif // __AVX__
        if (inch >= 4)
            kernel_tm.create(4 * 4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2);
        else if (inch >= 2)
            kernel_tm.create(4 * 2 * maxk, inch / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2);
        else
            kernel_tm.create(4 * maxk, inch, outch / 4 + (outch % 4) / 2 + outch % 2);
    }
    else
#endif // __SSE2__
    if (outch >= 2)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (inch >= 16)
            kernel_tm.create(2 * 16 * maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch / 2 + outch % 2);
        else
#endif // __AVX512F__
        if (inch >= 8)
            kernel_tm.create(2 * 8 * maxk, inch / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch / 2 + outch % 2);
        else
#endif // __AVX__
        if (inch >= 4)
            kernel_tm.create(2 * 4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch / 2 + outch % 2);
        else
#endif // __SSE2__
        if (inch >= 2)
            kernel_tm.create(2 * 2 * maxk, inch / 2 + inch % 2, outch / 2 + outch % 2);
        else
            kernel_tm.create(2 * maxk, inch, outch / 2 + outch % 2);
    }
    else
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (inch >= 16)
            kernel_tm.create(16 * maxk, inch / 16 + (inch % 16) / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch);
        else
#endif // __AVX512F__
        if (inch >= 8)
            kernel_tm.create(8 * maxk, inch / 8 + (inch % 8) / 4 + (inch % 4) / 2 + inch % 2, outch);
        else
#endif // __AVX__
        if (inch >= 4)
            kernel_tm.create(4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch);
        else
#endif // __SSE2__
        if (inch >= 2)
            kernel_tm.create(2 * maxk, inch / 2 + inch % 2, outch);
        else
            kernel_tm.create(maxk, inch, outch);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; q + 15 < outch; q += 16)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;
        const float* kptr2 = (const float*)kernel + (q + 2) * inch * maxk;
        const float* kptr3 = (const float*)kernel + (q + 3) * inch * maxk;
        const float* kptr4 = (const float*)kernel + (q + 4) * inch * maxk;
        const float* kptr5 = (const float*)kernel + (q + 5) * inch * maxk;
        const float* kptr6 = (const float*)kernel + (q + 6) * inch * maxk;
        const float* kptr7 = (const float*)kernel + (q + 7) * inch * maxk;
        const float* kptr8 = (const float*)kernel + (q + 8) * inch * maxk;
        const float* kptr9 = (const float*)kernel + (q + 9) * inch * maxk;
        const float* kptra = (const float*)kernel + (q + 10) * inch * maxk;
        const float* kptrb = (const float*)kernel + (q + 11) * inch * maxk;
        const float* kptrc = (const float*)kernel + (q + 12) * inch * maxk;
        const float* kptrd = (const float*)kernel + (q + 13) * inch * maxk;
        const float* kptre = (const float*)kernel + (q + 14) * inch * maxk;
        const float* kptrf = (const float*)kernel + (q + 15) * inch * maxk;

        float* g00 = kernel_tm.channel(q / 16);

        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(maxk));

        int p = 0;
        for (; p + 15 < inch; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;
                const float* k4 = kptr4 + k;
                const float* k5 = kptr5 + k;
                const float* k6 = kptr6 + k;
                const float* k7 = kptr7 + k;
                const float* k8 = kptr8 + k;
                const float* k9 = kptr9 + k;
                const float* ka = kptra + k;
                const float* kb = kptrb + k;
                const float* kc = kptrc + k;
                const float* kd = kptrd + k;
                const float* ke = kptre + k;
                const float* kf = kptrf + k;

                __m512 _k0 = _mm512_i32gather_ps(_vindex, k0, sizeof(float));
                __m512 _k1 = _mm512_i32gather_ps(_vindex, k1, sizeof(float));
                __m512 _k2 = _mm512_i32gather_ps(_vindex, k2, sizeof(float));
                __m512 _k3 = _mm512_i32gather_ps(_vindex, k3, sizeof(float));
                __m512 _k4 = _mm512_i32gather_ps(_vindex, k4, sizeof(float));
                __m512 _k5 = _mm512_i32gather_ps(_vindex, k5, sizeof(float));
                __m512 _k6 = _mm512_i32gather_ps(_vindex, k6, sizeof(float));
                __m512 _k7 = _mm512_i32gather_ps(_vindex, k7, sizeof(float));
                __m512 _k8 = _mm512_i32gather_ps(_vindex, k8, sizeof(float));
                __m512 _k9 = _mm512_i32gather_ps(_vindex, k9, sizeof(float));
                __m512 _ka = _mm512_i32gather_ps(_vindex, ka, sizeof(float));
                __m512 _kb = _mm512_i32gather_ps(_vindex, kb, sizeof(float));
                __m512 _kc = _mm512_i32gather_ps(_vindex, kc, sizeof(float));
                __m512 _kd = _mm512_i32gather_ps(_vindex, kd, sizeof(float));
                __m512 _ke = _mm512_i32gather_ps(_vindex, ke, sizeof(float));
                __m512 _kf = _mm512_i32gather_ps(_vindex, kf, sizeof(float));

                transpose16x16_ps(_k0, _k1, _k2, _k3, _k4, _k5, _k6, _k7, _k8, _k9, _ka, _kb, _kc, _kd, _ke, _kf);

                _mm512_store_ps(g00, _k0);
                _mm512_store_ps(g00 + 16, _k1);
                _mm512_store_ps(g00 + 16 * 2, _k2);
                _mm512_store_ps(g00 + 16 * 3, _k3);
                _mm512_store_ps(g00 + 16 * 4, _k4);
                _mm512_store_ps(g00 + 16 * 5, _k5);
                _mm512_store_ps(g00 + 16 * 6, _k6);
                _mm512_store_ps(g00 + 16 * 7, _k7);
                _mm512_store_ps(g00 + 16 * 8, _k8);
                _mm512_store_ps(g00 + 16 * 9, _k9);
                _mm512_store_ps(g00 + 16 * 10, _ka);
                _mm512_store_ps(g00 + 16 * 11, _kb);
                _mm512_store_ps(g00 + 16 * 12, _kc);
                _mm512_store_ps(g00 + 16 * 13, _kd);
                _mm512_store_ps(g00 + 16 * 14, _ke);
                _mm512_store_ps(g00 + 16 * 15, _kf);

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

        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(inch));

        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;

                for (int i = 0; i < 8; i++)
                {
                    __m512 _k0 = _mm512_i32gather_ps(_vindex, k0, sizeof(float));
                    _mm512_store_ps(g00, _k0);
                    k0 += maxk;
                    g00 += 16;
                }
            }

            kptr0 += maxk * 8;
        }
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;

                for (int i = 0; i < 4; i++)
                {
                    __m512 _k0 = _mm512_i32gather_ps(_vindex, k0, sizeof(float));
                    _mm512_store_ps(g00, _k0);
                    k0 += maxk;
                    g00 += 16;
                }
            }

            kptr0 += maxk * 4;
        }
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;

                for (int i = 0; i < 2; i++)
                {
                    __m512 _k0 = _mm512_i32gather_ps(_vindex, k0, sizeof(float));
                    _mm512_store_ps(g00, _k0);
                    k0 += maxk;
                    g00 += 16;
                }
            }

            kptr0 += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;

                __m512 _k0 = _mm512_i32gather_ps(_vindex, k0, sizeof(float));
                _mm512_store_ps(g00, _k0);
                g00 += 16;
            }
        }
    }
#endif // __AVX512F__
    for (; q + 7 < outch; q += 8)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;
        const float* kptr2 = (const float*)kernel + (q + 2) * inch * maxk;
        const float* kptr3 = (const float*)kernel + (q + 3) * inch * maxk;
        const float* kptr4 = (const float*)kernel + (q + 4) * inch * maxk;
        const float* kptr5 = (const float*)kernel + (q + 5) * inch * maxk;
        const float* kptr6 = (const float*)kernel + (q + 6) * inch * maxk;
        const float* kptr7 = (const float*)kernel + (q + 7) * inch * maxk;

#if __AVX512F__
        float* g00 = kernel_tm.channel(q / 16 + (q % 16) / 8);
#else
        float* g00 = kernel_tm.channel(q / 8);
#endif

#if __AVX2__
        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(maxk));
#if __AVX512F__
        __m512i _vindex_512 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex_512 = _mm512_mullo_epi32(_vindex_512, _mm512_set1_epi32(maxk));
#endif // __AVX512F__
#endif // __AVX2__

        int p = 0;
#if __AVX512F__
        for (; p + 15 < inch; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;
                const float* k4 = kptr4 + k;
                const float* k5 = kptr5 + k;
                const float* k6 = kptr6 + k;
                const float* k7 = kptr7 + k;

                __m512 _k0 = _mm512_i32gather_ps(_vindex_512, k0, sizeof(float));
                __m512 _k1 = _mm512_i32gather_ps(_vindex_512, k1, sizeof(float));
                __m512 _k2 = _mm512_i32gather_ps(_vindex_512, k2, sizeof(float));
                __m512 _k3 = _mm512_i32gather_ps(_vindex_512, k3, sizeof(float));
                __m512 _k4 = _mm512_i32gather_ps(_vindex_512, k4, sizeof(float));
                __m512 _k5 = _mm512_i32gather_ps(_vindex_512, k5, sizeof(float));
                __m512 _k6 = _mm512_i32gather_ps(_vindex_512, k6, sizeof(float));
                __m512 _k7 = _mm512_i32gather_ps(_vindex_512, k7, sizeof(float));

                transpose16x8_ps(_k0, _k1, _k2, _k3, _k4, _k5, _k6, _k7);

                _mm512_storeu_ps(g00, _k0);
                _mm512_storeu_ps(g00 + 16, _k1);
                _mm512_storeu_ps(g00 + 16 * 2, _k2);
                _mm512_storeu_ps(g00 + 16 * 3, _k3);
                _mm512_storeu_ps(g00 + 16 * 4, _k4);
                _mm512_storeu_ps(g00 + 16 * 5, _k5);
                _mm512_storeu_ps(g00 + 16 * 6, _k6);
                _mm512_storeu_ps(g00 + 16 * 7, _k7);

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
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;
                const float* k4 = kptr4 + k;
                const float* k5 = kptr5 + k;
                const float* k6 = kptr6 + k;
                const float* k7 = kptr7 + k;

#if __AVX2__
                __m256 _k0 = _mm256_i32gather_ps(k0, _vindex, sizeof(float));
                __m256 _k1 = _mm256_i32gather_ps(k1, _vindex, sizeof(float));
                __m256 _k2 = _mm256_i32gather_ps(k2, _vindex, sizeof(float));
                __m256 _k3 = _mm256_i32gather_ps(k3, _vindex, sizeof(float));
                __m256 _k4 = _mm256_i32gather_ps(k4, _vindex, sizeof(float));
                __m256 _k5 = _mm256_i32gather_ps(k5, _vindex, sizeof(float));
                __m256 _k6 = _mm256_i32gather_ps(k6, _vindex, sizeof(float));
                __m256 _k7 = _mm256_i32gather_ps(k7, _vindex, sizeof(float));

                transpose8x8_ps(_k0, _k1, _k2, _k3, _k4, _k5, _k6, _k7);

                _mm256_store_ps(g00, _k0);
                _mm256_store_ps(g00 + 8, _k1);
                _mm256_store_ps(g00 + 8 * 2, _k2);
                _mm256_store_ps(g00 + 8 * 3, _k3);
                _mm256_store_ps(g00 + 8 * 4, _k4);
                _mm256_store_ps(g00 + 8 * 5, _k5);
                _mm256_store_ps(g00 + 8 * 6, _k6);
                _mm256_store_ps(g00 + 8 * 7, _k7);

                g00 += 64;
#else  // __AVX2__
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
                    g00[4] = k4[0];
                    g00[5] = k5[0];
                    g00[6] = k6[0];
                    g00[7] = k7[0];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    k4 += maxk;
                    k5 += maxk;
                    k6 += maxk;
                    k7 += maxk;
                    g00 += 8;
                }
#endif // __AVX2__
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

#if __AVX2__
        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(inch));
#endif // __AVX2__

        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
#if !__AVX2__
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;
                const float* k4 = kptr4 + k;
                const float* k5 = kptr5 + k;
                const float* k6 = kptr6 + k;
                const float* k7 = kptr7 + k;
#endif // !__AVX2__

                for (int i = 0; i < 4; i++)
                {
#if __AVX2__
                    __m256 _k0 = _mm256_i32gather_ps(k0, _vindex, sizeof(float));
                    _mm256_store_ps(g00, _k0);
                    k0 += maxk;
                    g00 += 8;
#else  // __AVX2__
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
                    g00[4] = k4[0];
                    g00[5] = k5[0];
                    g00[6] = k6[0];
                    g00[7] = k7[0];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    k4 += maxk;
                    k5 += maxk;
                    k6 += maxk;
                    k7 += maxk;
                    g00 += 8;
#endif // __AVX2__
                }
            }

            kptr0 += maxk * 4;
#if !__AVX2__
            kptr1 += maxk * 4;
            kptr2 += maxk * 4;
            kptr3 += maxk * 4;
            kptr4 += maxk * 4;
            kptr5 += maxk * 4;
            kptr6 += maxk * 4;
            kptr7 += maxk * 4;
#endif // !__AVX2__
        }
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
#if !__AVX2__
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;
                const float* k4 = kptr4 + k;
                const float* k5 = kptr5 + k;
                const float* k6 = kptr6 + k;
                const float* k7 = kptr7 + k;
#endif // !__AVX2__

                for (int i = 0; i < 2; i++)
                {
#if __AVX2__
                    __m256 _k0 = _mm256_i32gather_ps(k0, _vindex, sizeof(float));
                    _mm256_store_ps(g00, _k0);
                    k0 += maxk;
                    g00 += 8;
#else  // __AVX2__
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
                    g00[4] = k4[0];
                    g00[5] = k5[0];
                    g00[6] = k6[0];
                    g00[7] = k7[0];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    k4 += maxk;
                    k5 += maxk;
                    k6 += maxk;
                    k7 += maxk;
                    g00 += 8;
#endif // __AVX2__
                }
            }

            kptr0 += maxk * 2;
#if !__AVX2__
            kptr1 += maxk * 2;
            kptr2 += maxk * 2;
            kptr3 += maxk * 2;
            kptr4 += maxk * 2;
            kptr5 += maxk * 2;
            kptr6 += maxk * 2;
            kptr7 += maxk * 2;
#endif // !__AVX2__
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
#if __AVX2__
                __m256 _k0 = _mm256_i32gather_ps(k0, _vindex, sizeof(float));
                _mm256_store_ps(g00, _k0);
                g00 += 8;
#else  // __AVX2__
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;
                const float* k4 = kptr4 + k;
                const float* k5 = kptr5 + k;
                const float* k6 = kptr6 + k;
                const float* k7 = kptr7 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k2[0];
                g00[3] = k3[0];
                g00[4] = k4[0];
                g00[5] = k5[0];
                g00[6] = k6[0];
                g00[7] = k7[0];
                g00 += 8;
#endif // __AVX2__
            }
        }
    }
#endif // __AVX__
    for (; q + 3 < outch; q += 4)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;
        const float* kptr2 = (const float*)kernel + (q + 2) * inch * maxk;
        const float* kptr3 = (const float*)kernel + (q + 3) * inch * maxk;

#if __AVX512F__
        float* g00 = kernel_tm.channel(q / 16 + (q % 16) / 8 + (q % 8) / 4);
#elif __AVX__
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        float* g00 = kernel_tm.channel(q / 4);
#endif

#if __AVX2__
        __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
        _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(maxk));
        __m256i _vindex_256 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _vindex_256 = _mm256_mullo_epi32(_vindex_256, _mm256_set1_epi32(maxk));
#if __AVX512F__
        __m512i _vindex_512 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex_512 = _mm512_mullo_epi32(_vindex_512, _mm512_set1_epi32(maxk));
#endif // __AVX512F__
#endif // __AVX2__

        int p = 0;
#if __AVX__
#if __AVX512F__
        for (; p + 15 < inch; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;

                __m512 _k0 = _mm512_i32gather_ps(_vindex_512, k0, sizeof(float));
                __m512 _k1 = _mm512_i32gather_ps(_vindex_512, k1, sizeof(float));
                __m512 _k2 = _mm512_i32gather_ps(_vindex_512, k2, sizeof(float));
                __m512 _k3 = _mm512_i32gather_ps(_vindex_512, k3, sizeof(float));

                transpose16x4_ps(_k0, _k1, _k2, _k3);

                _mm512_storeu_ps(g00, _k0);
                _mm512_storeu_ps(g00 + 16, _k1);
                _mm512_storeu_ps(g00 + 16 * 2, _k2);
                _mm512_storeu_ps(g00 + 16 * 3, _k3);

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
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;

#if __AVX2__
                __m256 _k0 = _mm256_i32gather_ps(k0, _vindex_256, sizeof(float));
                __m256 _k1 = _mm256_i32gather_ps(k1, _vindex_256, sizeof(float));
                __m256 _k2 = _mm256_i32gather_ps(k2, _vindex_256, sizeof(float));
                __m256 _k3 = _mm256_i32gather_ps(k3, _vindex_256, sizeof(float));

                transpose8x4_ps(_k0, _k1, _k2, _k3);

                _mm256_storeu_ps(g00, _k0);
                _mm256_storeu_ps(g00 + 8, _k1);
                _mm256_storeu_ps(g00 + 8 * 2, _k2);
                _mm256_storeu_ps(g00 + 8 * 3, _k3);

                g00 += 32;
#else  // __AVX2__
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
                }
#endif // __AVX2__
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
            kptr2 += maxk * 8;
            kptr3 += maxk * 8;
        }
#endif // __AVX__
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;

#if __AVX2__
                __m128 _k0 = _mm_i32gather_ps(k0, _vindex, sizeof(float));
                __m128 _k1 = _mm_i32gather_ps(k1, _vindex, sizeof(float));
                __m128 _k2 = _mm_i32gather_ps(k2, _vindex, sizeof(float));
                __m128 _k3 = _mm_i32gather_ps(k3, _vindex, sizeof(float));

                _MM_TRANSPOSE4_PS(_k0, _k1, _k2, _k3);

                _mm_store_ps(g00, _k0);
                _mm_store_ps(g00 + 4, _k1);
                _mm_store_ps(g00 + 4 * 2, _k2);
                _mm_store_ps(g00 + 4 * 3, _k3);

                g00 += 16;
#else  // __AVX2__
                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
                }
#endif // __AVX2__
            }

            kptr0 += maxk * 4;
            kptr1 += maxk * 4;
            kptr2 += maxk * 4;
            kptr3 += maxk * 4;
        }

#if __AVX2__
        _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(inch));
#endif // __AVX2__

        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
#if !__AVX2__
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;
#endif // !__AVX2__

                for (int i = 0; i < 2; i++)
                {
#if __AVX2__
                    __m128 _k0 = _mm_i32gather_ps(k0, _vindex, sizeof(float));
                    _mm_store_ps(g00, _k0);
                    k0 += maxk;
                    g00 += 4;
#else  // __AVX2__
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
#endif // __AVX2__
                }
            }

            kptr0 += maxk * 2;
#if !__AVX2__
            kptr1 += maxk * 2;
            kptr2 += maxk * 2;
            kptr3 += maxk * 2;
#endif // !__AVX2__
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
#if __AVX2__
                __m128 _k0 = _mm_i32gather_ps(k0, _vindex, sizeof(float));
                _mm_store_ps(g00, _k0);
                g00 += 4;
#else  // __AVX2__
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k2[0];
                g00[3] = k3[0];
                g00 += 4;
#endif // __AVX2__
            }
        }
    }
#endif // __SSE2__
    for (; q + 1 < outch; q += 2)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;

#if __AVX512F__
        float* g00 = kernel_tm.channel(q / 16 + (q % 16) / 8 + (q % 8) / 4 + (q % 4) / 2);
#elif __AVX__
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
#elif __SSE2__
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2);
#else
        float* g00 = kernel_tm.channel(q / 2);
#endif

#if __AVX2__
        __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
        _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(maxk));
        __m256i _vindex_256 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _vindex_256 = _mm256_mullo_epi32(_vindex_256, _mm256_set1_epi32(maxk));
#if __AVX512F__
        __m512i _vindex_512 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex_512 = _mm512_mullo_epi32(_vindex_512, _mm512_set1_epi32(maxk));
#endif // __AVX512F__
#endif // __AVX2__

        int p = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; p + 15 < inch; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                __m512 _k0 = _mm512_i32gather_ps(_vindex_512, k0, sizeof(float));
                __m512 _k1 = _mm512_i32gather_ps(_vindex_512, k1, sizeof(float));
                _mm512_storeu_ps(g00, _k0);
                _mm512_storeu_ps(g00 + 16, _k1);
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
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

#if __AVX2__
                __m256 _k0 = _mm256_i32gather_ps(k0, _vindex_256, sizeof(float));
                __m256 _k1 = _mm256_i32gather_ps(k1, _vindex_256, sizeof(float));
                _mm256_storeu_ps(g00, _k0);
                _mm256_storeu_ps(g00 + 8, _k1);
                g00 += 16;
#else  // __AVX2__
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
#endif // __AVX2__
            }

            kptr0 += maxk * 8;
            kptr1 += maxk * 8;
        }
#endif // __AVX__
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

#if __AVX2__
                __m128 _k0 = _mm_i32gather_ps(k0, _vindex, sizeof(float));
                __m128 _k1 = _mm_i32gather_ps(k1, _vindex, sizeof(float));
                _mm_storeu_ps(g00, _k0);
                _mm_storeu_ps(g00 + 4, _k1);
                g00 += 8;
#else  // __AVX2__
                g00[0] = k0[0];
                g00[1] = k0[maxk];
                g00[2] = k0[maxk * 2];
                g00[3] = k0[maxk * 3];
                g00[4] = k1[0];
                g00[5] = k1[maxk];
                g00[6] = k1[maxk * 2];
                g00[7] = k1[maxk * 3];
                g00 += 8;
#endif // __AVX2__
            }

            kptr0 += maxk * 4;
            kptr1 += maxk * 4;
        }
#endif // __SSE2__
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

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
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00 += 2;
            }
        }
    }
    for (; q < outch; q++)
    {
        const float* kptr = (const float*)kernel + q * inch * maxk;

#if __AVX512F__
        float* g00 = kernel_tm.channel(q / 16 + (q % 16) / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#elif __AVX__
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
#elif __SSE2__
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#else
        float* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

#if __AVX2__
        __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
        _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(maxk));
        __m256i _vindex_256 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _vindex_256 = _mm256_mullo_epi32(_vindex_256, _mm256_set1_epi32(maxk));
#if __AVX512F__
        __m512i _vindex_512 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex_512 = _mm512_mullo_epi32(_vindex_512, _mm512_set1_epi32(maxk));
#endif // __AVX512F__
#endif // __AVX2__

        int p = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; p + 15 < inch; p += 16)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + k;

                __m512 _k0 = _mm512_i32gather_ps(_vindex_512, k0, sizeof(float));
                _mm512_storeu_ps(g00, _k0);
                g00 += 16;
            }

            kptr += maxk * 16;
        }
#endif // __AVX512F__
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + k;

#if __AVX2__
                __m256 _k0 = _mm256_i32gather_ps(k0, _vindex_256, sizeof(float));
                _mm256_storeu_ps(g00, _k0);
                g00 += 8;
#else  // __AVX2__
                for (int i = 0; i < 8; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
#endif // __AVX2__
            }

            kptr += maxk * 8;
        }
#endif // __AVX__
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + k;

#if __AVX2__
                __m128 _k0 = _mm_i32gather_ps(k0, _vindex, sizeof(float));
                _mm_storeu_ps(g00, _k0);
                g00 += 4;
#else  // __AVX2__
                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
#endif // __AVX2__
            }

            kptr += maxk * 4;
        }
#endif // __SSE2__
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + k;

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
                const float* k0 = kptr + k;

                g00[0] = k0[0];
                g00++;
            }
        }
    }
}

static void convolution_packed(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
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

    const float* bias_data_ptr = bias_data;

    int nn_outch = 0;
    int remain_outch_start = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    nn_outch = outch / 16;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = pp * 16;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.channel(p / out_elempack);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm512_loadu_ps(bias_data_ptr + p);
                }

                const float* kptr = weight_data_tm.channel(p / 16);

                int q = 0;
                for (; q + 15 < inch; q += 16)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 16)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];

                            __m512 _w0 = _mm512_load_ps(kptr + 16 * 0);
                            __m512 _w1 = _mm512_load_ps(kptr + 16 * 1);
                            __m512 _w2 = _mm512_load_ps(kptr + 16 * 2);
                            __m512 _w3 = _mm512_load_ps(kptr + 16 * 3);
                            __m512 _w4 = _mm512_load_ps(kptr + 16 * 4);
                            __m512 _w5 = _mm512_load_ps(kptr + 16 * 5);
                            __m512 _w6 = _mm512_load_ps(kptr + 16 * 6);
                            __m512 _w7 = _mm512_load_ps(kptr + 16 * 7);
                            __m512 _w8 = _mm512_load_ps(kptr + 16 * 8);
                            __m512 _w9 = _mm512_load_ps(kptr + 16 * 9);
                            __m512 _wa = _mm512_load_ps(kptr + 16 * 10);
                            __m512 _wb = _mm512_load_ps(kptr + 16 * 11);
                            __m512 _wc = _mm512_load_ps(kptr + 16 * 12);
                            __m512 _wd = _mm512_load_ps(kptr + 16 * 13);
                            __m512 _we = _mm512_load_ps(kptr + 16 * 14);
                            __m512 _wf = _mm512_load_ps(kptr + 16 * 15);

                            _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r0s[4]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r0s[5]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r0s[6]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r0s[7]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_w8, _mm512_set1_ps(r0s[8]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w9, _mm512_set1_ps(r0s[9]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_wa, _mm512_set1_ps(r0s[10]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_wb, _mm512_set1_ps(r0s[11]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_wc, _mm512_set1_ps(r0s[12]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_wd, _mm512_set1_ps(r0s[13]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_we, _mm512_set1_ps(r0s[14]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_wf, _mm512_set1_ps(r0s[15]), _sum3);

                            kptr += 256;
                        }
                    }
                    if (elempack == 8)
                    {
                        const float* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];
                            const float* r1s = r1 + space_ofs[k];

                            __m512 _w0 = _mm512_load_ps(kptr + 16 * 0);
                            __m512 _w1 = _mm512_load_ps(kptr + 16 * 1);
                            __m512 _w2 = _mm512_load_ps(kptr + 16 * 2);
                            __m512 _w3 = _mm512_load_ps(kptr + 16 * 3);
                            __m512 _w4 = _mm512_load_ps(kptr + 16 * 4);
                            __m512 _w5 = _mm512_load_ps(kptr + 16 * 5);
                            __m512 _w6 = _mm512_load_ps(kptr + 16 * 6);
                            __m512 _w7 = _mm512_load_ps(kptr + 16 * 7);
                            __m512 _w8 = _mm512_load_ps(kptr + 16 * 8);
                            __m512 _w9 = _mm512_load_ps(kptr + 16 * 9);
                            __m512 _wa = _mm512_load_ps(kptr + 16 * 10);
                            __m512 _wb = _mm512_load_ps(kptr + 16 * 11);
                            __m512 _wc = _mm512_load_ps(kptr + 16 * 12);
                            __m512 _wd = _mm512_load_ps(kptr + 16 * 13);
                            __m512 _we = _mm512_load_ps(kptr + 16 * 14);
                            __m512 _wf = _mm512_load_ps(kptr + 16 * 15);

                            _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r0s[4]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r0s[5]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r0s[6]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r0s[7]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_w8, _mm512_set1_ps(r1s[0]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w9, _mm512_set1_ps(r1s[1]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_wa, _mm512_set1_ps(r1s[2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_wb, _mm512_set1_ps(r1s[3]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_wc, _mm512_set1_ps(r1s[4]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_wd, _mm512_set1_ps(r1s[5]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_we, _mm512_set1_ps(r1s[6]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_wf, _mm512_set1_ps(r1s[7]), _sum3);

                            kptr += 256;
                        }
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];
                            const float* r1s = r1 + space_ofs[k];
                            const float* r2s = r2 + space_ofs[k];
                            const float* r3s = r3 + space_ofs[k];

                            __m512 _w0 = _mm512_load_ps(kptr + 16 * 0);
                            __m512 _w1 = _mm512_load_ps(kptr + 16 * 1);
                            __m512 _w2 = _mm512_load_ps(kptr + 16 * 2);
                            __m512 _w3 = _mm512_load_ps(kptr + 16 * 3);
                            __m512 _w4 = _mm512_load_ps(kptr + 16 * 4);
                            __m512 _w5 = _mm512_load_ps(kptr + 16 * 5);
                            __m512 _w6 = _mm512_load_ps(kptr + 16 * 6);
                            __m512 _w7 = _mm512_load_ps(kptr + 16 * 7);
                            __m512 _w8 = _mm512_load_ps(kptr + 16 * 8);
                            __m512 _w9 = _mm512_load_ps(kptr + 16 * 9);
                            __m512 _wa = _mm512_load_ps(kptr + 16 * 10);
                            __m512 _wb = _mm512_load_ps(kptr + 16 * 11);
                            __m512 _wc = _mm512_load_ps(kptr + 16 * 12);
                            __m512 _wd = _mm512_load_ps(kptr + 16 * 13);
                            __m512 _we = _mm512_load_ps(kptr + 16 * 14);
                            __m512 _wf = _mm512_load_ps(kptr + 16 * 15);

                            _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r1s[0]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r1s[1]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r1s[2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r1s[3]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_w8, _mm512_set1_ps(r2s[0]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w9, _mm512_set1_ps(r2s[1]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_wa, _mm512_set1_ps(r2s[2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_wb, _mm512_set1_ps(r2s[3]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_wc, _mm512_set1_ps(r3s[0]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_wd, _mm512_set1_ps(r3s[1]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_we, _mm512_set1_ps(r3s[2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_wf, _mm512_set1_ps(r3s[3]), _sum3);

                            kptr += 256;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m512 _w0 = _mm512_load_ps(kptr + 16 * 0);
                            __m512 _w1 = _mm512_load_ps(kptr + 16 * 1);
                            __m512 _w2 = _mm512_load_ps(kptr + 16 * 2);
                            __m512 _w3 = _mm512_load_ps(kptr + 16 * 3);
                            __m512 _w4 = _mm512_load_ps(kptr + 16 * 4);
                            __m512 _w5 = _mm512_load_ps(kptr + 16 * 5);
                            __m512 _w6 = _mm512_load_ps(kptr + 16 * 6);
                            __m512 _w7 = _mm512_load_ps(kptr + 16 * 7);
                            __m512 _w8 = _mm512_load_ps(kptr + 16 * 8);
                            __m512 _w9 = _mm512_load_ps(kptr + 16 * 9);
                            __m512 _wa = _mm512_load_ps(kptr + 16 * 10);
                            __m512 _wb = _mm512_load_ps(kptr + 16 * 11);
                            __m512 _wc = _mm512_load_ps(kptr + 16 * 12);
                            __m512 _wd = _mm512_load_ps(kptr + 16 * 13);
                            __m512 _we = _mm512_load_ps(kptr + 16 * 14);
                            __m512 _wf = _mm512_load_ps(kptr + 16 * 15);

                            _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[sok]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[sok + N]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0[sok + N * 2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0[sok + N * 3]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r0[sok + N * 4]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r0[sok + N * 5]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r0[sok + N * 6]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r0[sok + N * 7]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_w8, _mm512_set1_ps(r0[sok + N * 8]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w9, _mm512_set1_ps(r0[sok + N * 9]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_wa, _mm512_set1_ps(r0[sok + N * 10]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_wb, _mm512_set1_ps(r0[sok + N * 11]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_wc, _mm512_set1_ps(r0[sok + N * 12]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_wd, _mm512_set1_ps(r0[sok + N * 13]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_we, _mm512_set1_ps(r0[sok + N * 14]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_wf, _mm512_set1_ps(r0[sok + N * 15]), _sum3);

                            kptr += 256;
                        }
                    }
                }
                for (; q + 7 < inch; q += 8)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 8)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];

                            __m512 _w0 = _mm512_load_ps(kptr + 16 * 0);
                            __m512 _w1 = _mm512_load_ps(kptr + 16 * 1);
                            __m512 _w2 = _mm512_load_ps(kptr + 16 * 2);
                            __m512 _w3 = _mm512_load_ps(kptr + 16 * 3);
                            __m512 _w4 = _mm512_load_ps(kptr + 16 * 4);
                            __m512 _w5 = _mm512_load_ps(kptr + 16 * 5);
                            __m512 _w6 = _mm512_load_ps(kptr + 16 * 6);
                            __m512 _w7 = _mm512_load_ps(kptr + 16 * 7);

                            _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r0s[4]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r0s[5]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r0s[6]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r0s[7]), _sum3);

                            kptr += 128;
                        }
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];
                            const float* r1s = r1 + space_ofs[k];

                            __m512 _w0 = _mm512_load_ps(kptr + 16 * 0);
                            __m512 _w1 = _mm512_load_ps(kptr + 16 * 1);
                            __m512 _w2 = _mm512_load_ps(kptr + 16 * 2);
                            __m512 _w3 = _mm512_load_ps(kptr + 16 * 3);
                            __m512 _w4 = _mm512_load_ps(kptr + 16 * 4);
                            __m512 _w5 = _mm512_load_ps(kptr + 16 * 5);
                            __m512 _w6 = _mm512_load_ps(kptr + 16 * 6);
                            __m512 _w7 = _mm512_load_ps(kptr + 16 * 7);

                            _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r1s[0]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r1s[1]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r1s[2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r1s[3]), _sum3);

                            kptr += 128;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m512 _w0 = _mm512_load_ps(kptr + 16 * 0);
                            __m512 _w1 = _mm512_load_ps(kptr + 16 * 1);
                            __m512 _w2 = _mm512_load_ps(kptr + 16 * 2);
                            __m512 _w3 = _mm512_load_ps(kptr + 16 * 3);
                            __m512 _w4 = _mm512_load_ps(kptr + 16 * 4);
                            __m512 _w5 = _mm512_load_ps(kptr + 16 * 5);
                            __m512 _w6 = _mm512_load_ps(kptr + 16 * 6);
                            __m512 _w7 = _mm512_load_ps(kptr + 16 * 7);

                            _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[sok]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[sok + N]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0[sok + N * 2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0[sok + N * 3]), _sum3);
                            _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r0[sok + N * 4]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r0[sok + N * 5]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r0[sok + N * 6]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r0[sok + N * 7]), _sum3);

                            kptr += 128;
                        }
                    }
                }
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];

                            __m512 _w0 = _mm512_load_ps(kptr);
                            __m512 _w1 = _mm512_load_ps(kptr + 16);
                            __m512 _w2 = _mm512_load_ps(kptr + 32);
                            __m512 _w3 = _mm512_load_ps(kptr + 48);

                            _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0s[3]), _sum3);

                            kptr += 64;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m512 _w0 = _mm512_load_ps(kptr);
                            __m512 _w1 = _mm512_load_ps(kptr + 16);
                            __m512 _w2 = _mm512_load_ps(kptr + 32);
                            __m512 _w3 = _mm512_load_ps(kptr + 48);

                            _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[sok]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[sok + N]), _sum1);
                            _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0[sok + N * 2]), _sum2);
                            _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0[sok + N * 3]), _sum3);

                            kptr += 64;
                        }
                    }
                }
                for (; q + 1 < inch; q += 2)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m512 _w0 = _mm512_load_ps(kptr);
                            __m512 _w1 = _mm512_load_ps(kptr + 16);

                            _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[sok]), _sum0);
                            _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[sok + N]), _sum1);

                            kptr += 32;
                        }
                    }
                }
                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            __m512 _val = _mm512_set1_ps(r0[space_ofs[k]]);
                            __m512 _w = _mm512_load_ps(kptr);
                            _sum0 = _mm512_fmadd_ps(_val, _w, _sum0);

                            kptr += 16;
                        }
                    }
                }

                _sum0 = _mm512_add_ps(_sum0, _sum1);
                _sum2 = _mm512_add_ps(_sum2, _sum3);
                _sum0 = _mm512_add_ps(_sum0, _sum2);

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);

                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr, _sum0);
                    outptr += 16;
                }
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr, _mm512_extractf32x8_ps(_sum0, 0));
                    _mm256_store_ps(outptr + M, _mm512_extractf32x8_ps(_sum0, 1));
                    outptr += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr, _mm512_extractf32x4_ps(_sum0, 0));
                    _mm_store_ps(outptr + M, _mm512_extractf32x4_ps(_sum0, 1));
                    _mm_store_ps(outptr + M * 2, _mm512_extractf32x4_ps(_sum0, 2));
                    _mm_store_ps(outptr + M * 3, _mm512_extractf32x4_ps(_sum0, 3));
                    outptr += 4;
                }
                if (out_elempack == 1)
                {
                    float sum[16];
                    _mm512_storeu_ps(sum, _sum0);

                    outptr[0] = sum[0];
                    outptr[M] = sum[1];
                    outptr[M * 2] = sum[2];
                    outptr[M * 3] = sum[3];
                    outptr[M * 4] = sum[4];
                    outptr[M * 5] = sum[5];
                    outptr[M * 6] = sum[6];
                    outptr[M * 7] = sum[7];
                    outptr[M * 8] = sum[8];
                    outptr[M * 9] = sum[9];
                    outptr[M * 10] = sum[10];
                    outptr[M * 11] = sum[11];
                    outptr[M * 12] = sum[12];
                    outptr[M * 13] = sum[13];
                    outptr[M * 14] = sum[14];
                    outptr[M * 15] = sum[15];
                    outptr += 1;
                }
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
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.channel(p / out_elempack);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm256_loadu_ps(bias_data_ptr + p);
                }

#if __AVX512F__
                const float* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8);
#else
                const float* kptr = weight_data_tm.channel(p / 8);
#endif

                int q = 0;
#if __AVX512F__
                for (; q + 15 < inch; q += 16)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 16)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];

                            __m256 _w0 = _mm256_load_ps(kptr + 8 * 0);
                            __m256 _w1 = _mm256_load_ps(kptr + 8 * 1);
                            __m256 _w2 = _mm256_load_ps(kptr + 8 * 2);
                            __m256 _w3 = _mm256_load_ps(kptr + 8 * 3);
                            __m256 _w4 = _mm256_load_ps(kptr + 8 * 4);
                            __m256 _w5 = _mm256_load_ps(kptr + 8 * 5);
                            __m256 _w6 = _mm256_load_ps(kptr + 8 * 6);
                            __m256 _w7 = _mm256_load_ps(kptr + 8 * 7);
                            __m256 _w8 = _mm256_load_ps(kptr + 8 * 8);
                            __m256 _w9 = _mm256_load_ps(kptr + 8 * 9);
                            __m256 _wa = _mm256_load_ps(kptr + 8 * 10);
                            __m256 _wb = _mm256_load_ps(kptr + 8 * 11);
                            __m256 _wc = _mm256_load_ps(kptr + 8 * 12);
                            __m256 _wd = _mm256_load_ps(kptr + 8 * 13);
                            __m256 _we = _mm256_load_ps(kptr + 8 * 14);
                            __m256 _wf = _mm256_load_ps(kptr + 8 * 15);

                            _sum0 = _mm256_fmadd_ps(_w0, _mm256_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_w1, _mm256_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_w2, _mm256_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_w3, _mm256_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm256_fmadd_ps(_w4, _mm256_set1_ps(r0s[4]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_w5, _mm256_set1_ps(r0s[5]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_w6, _mm256_set1_ps(r0s[6]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_w7, _mm256_set1_ps(r0s[7]), _sum3);
                            _sum0 = _mm256_fmadd_ps(_w8, _mm256_set1_ps(r0s[8]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_w9, _mm256_set1_ps(r0s[9]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_wa, _mm256_set1_ps(r0s[10]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_wb, _mm256_set1_ps(r0s[11]), _sum3);
                            _sum0 = _mm256_fmadd_ps(_wc, _mm256_set1_ps(r0s[12]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_wd, _mm256_set1_ps(r0s[13]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_we, _mm256_set1_ps(r0s[14]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_wf, _mm256_set1_ps(r0s[15]), _sum3);

                            kptr += 128;
                        }
                    }
                    if (elempack == 8)
                    {
                        const float* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];
                            const float* r1s = r1 + space_ofs[k];

                            __m256 _w0 = _mm256_load_ps(kptr + 8 * 0);
                            __m256 _w1 = _mm256_load_ps(kptr + 8 * 1);
                            __m256 _w2 = _mm256_load_ps(kptr + 8 * 2);
                            __m256 _w3 = _mm256_load_ps(kptr + 8 * 3);
                            __m256 _w4 = _mm256_load_ps(kptr + 8 * 4);
                            __m256 _w5 = _mm256_load_ps(kptr + 8 * 5);
                            __m256 _w6 = _mm256_load_ps(kptr + 8 * 6);
                            __m256 _w7 = _mm256_load_ps(kptr + 8 * 7);
                            __m256 _w8 = _mm256_load_ps(kptr + 8 * 8);
                            __m256 _w9 = _mm256_load_ps(kptr + 8 * 9);
                            __m256 _wa = _mm256_load_ps(kptr + 8 * 10);
                            __m256 _wb = _mm256_load_ps(kptr + 8 * 11);
                            __m256 _wc = _mm256_load_ps(kptr + 8 * 12);
                            __m256 _wd = _mm256_load_ps(kptr + 8 * 13);
                            __m256 _we = _mm256_load_ps(kptr + 8 * 14);
                            __m256 _wf = _mm256_load_ps(kptr + 8 * 15);

                            _sum0 = _mm256_fmadd_ps(_w0, _mm256_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_w1, _mm256_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_w2, _mm256_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_w3, _mm256_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm256_fmadd_ps(_w4, _mm256_set1_ps(r0s[4]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_w5, _mm256_set1_ps(r0s[5]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_w6, _mm256_set1_ps(r0s[6]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_w7, _mm256_set1_ps(r0s[7]), _sum3);
                            _sum0 = _mm256_fmadd_ps(_w8, _mm256_set1_ps(r1s[0]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_w9, _mm256_set1_ps(r1s[1]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_wa, _mm256_set1_ps(r1s[2]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_wb, _mm256_set1_ps(r1s[3]), _sum3);
                            _sum0 = _mm256_fmadd_ps(_wc, _mm256_set1_ps(r1s[4]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_wd, _mm256_set1_ps(r1s[5]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_we, _mm256_set1_ps(r1s[6]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_wf, _mm256_set1_ps(r1s[7]), _sum3);

                            kptr += 128;
                        }
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];
                            const float* r1s = r1 + space_ofs[k];
                            const float* r2s = r2 + space_ofs[k];
                            const float* r3s = r3 + space_ofs[k];

                            __m256 _w0 = _mm256_load_ps(kptr + 8 * 0);
                            __m256 _w1 = _mm256_load_ps(kptr + 8 * 1);
                            __m256 _w2 = _mm256_load_ps(kptr + 8 * 2);
                            __m256 _w3 = _mm256_load_ps(kptr + 8 * 3);
                            __m256 _w4 = _mm256_load_ps(kptr + 8 * 4);
                            __m256 _w5 = _mm256_load_ps(kptr + 8 * 5);
                            __m256 _w6 = _mm256_load_ps(kptr + 8 * 6);
                            __m256 _w7 = _mm256_load_ps(kptr + 8 * 7);
                            __m256 _w8 = _mm256_load_ps(kptr + 8 * 8);
                            __m256 _w9 = _mm256_load_ps(kptr + 8 * 9);
                            __m256 _wa = _mm256_load_ps(kptr + 8 * 10);
                            __m256 _wb = _mm256_load_ps(kptr + 8 * 11);
                            __m256 _wc = _mm256_load_ps(kptr + 8 * 12);
                            __m256 _wd = _mm256_load_ps(kptr + 8 * 13);
                            __m256 _we = _mm256_load_ps(kptr + 8 * 14);
                            __m256 _wf = _mm256_load_ps(kptr + 8 * 15);

                            _sum0 = _mm256_fmadd_ps(_w0, _mm256_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_w1, _mm256_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_w2, _mm256_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_w3, _mm256_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm256_fmadd_ps(_w4, _mm256_set1_ps(r1s[0]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_w5, _mm256_set1_ps(r1s[1]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_w6, _mm256_set1_ps(r1s[2]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_w7, _mm256_set1_ps(r1s[3]), _sum3);
                            _sum0 = _mm256_fmadd_ps(_w8, _mm256_set1_ps(r2s[0]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_w9, _mm256_set1_ps(r2s[1]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_wa, _mm256_set1_ps(r2s[2]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_wb, _mm256_set1_ps(r2s[3]), _sum3);
                            _sum0 = _mm256_fmadd_ps(_wc, _mm256_set1_ps(r3s[0]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_wd, _mm256_set1_ps(r3s[1]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_we, _mm256_set1_ps(r3s[2]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_wf, _mm256_set1_ps(r3s[3]), _sum3);

                            kptr += 128;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m256 _w0 = _mm256_load_ps(kptr + 8 * 0);
                            __m256 _w1 = _mm256_load_ps(kptr + 8 * 1);
                            __m256 _w2 = _mm256_load_ps(kptr + 8 * 2);
                            __m256 _w3 = _mm256_load_ps(kptr + 8 * 3);
                            __m256 _w4 = _mm256_load_ps(kptr + 8 * 4);
                            __m256 _w5 = _mm256_load_ps(kptr + 8 * 5);
                            __m256 _w6 = _mm256_load_ps(kptr + 8 * 6);
                            __m256 _w7 = _mm256_load_ps(kptr + 8 * 7);
                            __m256 _w8 = _mm256_load_ps(kptr + 8 * 8);
                            __m256 _w9 = _mm256_load_ps(kptr + 8 * 9);
                            __m256 _wa = _mm256_load_ps(kptr + 8 * 10);
                            __m256 _wb = _mm256_load_ps(kptr + 8 * 11);
                            __m256 _wc = _mm256_load_ps(kptr + 8 * 12);
                            __m256 _wd = _mm256_load_ps(kptr + 8 * 13);
                            __m256 _we = _mm256_load_ps(kptr + 8 * 14);
                            __m256 _wf = _mm256_load_ps(kptr + 8 * 15);

                            _sum0 = _mm256_fmadd_ps(_w0, _mm256_set1_ps(r0[sok]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_w1, _mm256_set1_ps(r0[sok + N]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_w2, _mm256_set1_ps(r0[sok + N * 2]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_w3, _mm256_set1_ps(r0[sok + N * 3]), _sum3);
                            _sum0 = _mm256_fmadd_ps(_w4, _mm256_set1_ps(r0[sok + N * 4]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_w5, _mm256_set1_ps(r0[sok + N * 5]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_w6, _mm256_set1_ps(r0[sok + N * 6]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_w7, _mm256_set1_ps(r0[sok + N * 7]), _sum3);
                            _sum0 = _mm256_fmadd_ps(_w8, _mm256_set1_ps(r0[sok + N * 8]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_w9, _mm256_set1_ps(r0[sok + N * 9]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_wa, _mm256_set1_ps(r0[sok + N * 10]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_wb, _mm256_set1_ps(r0[sok + N * 11]), _sum3);
                            _sum0 = _mm256_fmadd_ps(_wc, _mm256_set1_ps(r0[sok + N * 12]), _sum0);
                            _sum1 = _mm256_fmadd_ps(_wd, _mm256_set1_ps(r0[sok + N * 13]), _sum1);
                            _sum2 = _mm256_fmadd_ps(_we, _mm256_set1_ps(r0[sok + N * 14]), _sum2);
                            _sum3 = _mm256_fmadd_ps(_wf, _mm256_set1_ps(r0[sok + N * 15]), _sum3);

                            kptr += 128;
                        }
                    }
                }
#endif // __AVX512F__
                for (; q + 7 < inch; q += 8)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 8)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];

                            __m256 _w0 = _mm256_load_ps(kptr);
                            __m256 _w1 = _mm256_load_ps(kptr + 8);
                            __m256 _w2 = _mm256_load_ps(kptr + 16);
                            __m256 _w3 = _mm256_load_ps(kptr + 24);
                            __m256 _w4 = _mm256_load_ps(kptr + 32);
                            __m256 _w5 = _mm256_load_ps(kptr + 40);
                            __m256 _w6 = _mm256_load_ps(kptr + 48);
                            __m256 _w7 = _mm256_load_ps(kptr + 56);

                            _sum0 = _mm256_comp_fmadd_ps(_w0, _mm256_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_w1, _mm256_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_w2, _mm256_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_w3, _mm256_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm256_comp_fmadd_ps(_w4, _mm256_set1_ps(r0s[4]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_w5, _mm256_set1_ps(r0s[5]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_w6, _mm256_set1_ps(r0s[6]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_w7, _mm256_set1_ps(r0s[7]), _sum3);

                            kptr += 64;
                        }
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];
                            const float* r1s = r1 + space_ofs[k];

                            __m256 _w0 = _mm256_load_ps(kptr);
                            __m256 _w1 = _mm256_load_ps(kptr + 8);
                            __m256 _w2 = _mm256_load_ps(kptr + 16);
                            __m256 _w3 = _mm256_load_ps(kptr + 24);
                            __m256 _w4 = _mm256_load_ps(kptr + 32);
                            __m256 _w5 = _mm256_load_ps(kptr + 40);
                            __m256 _w6 = _mm256_load_ps(kptr + 48);
                            __m256 _w7 = _mm256_load_ps(kptr + 56);

                            _sum0 = _mm256_comp_fmadd_ps(_w0, _mm256_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_w1, _mm256_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_w2, _mm256_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_w3, _mm256_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm256_comp_fmadd_ps(_w4, _mm256_set1_ps(r1s[0]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_w5, _mm256_set1_ps(r1s[1]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_w6, _mm256_set1_ps(r1s[2]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_w7, _mm256_set1_ps(r1s[3]), _sum3);

                            kptr += 64;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m256 _w0 = _mm256_load_ps(kptr);
                            __m256 _w1 = _mm256_load_ps(kptr + 8);
                            __m256 _w2 = _mm256_load_ps(kptr + 16);
                            __m256 _w3 = _mm256_load_ps(kptr + 24);
                            __m256 _w4 = _mm256_load_ps(kptr + 32);
                            __m256 _w5 = _mm256_load_ps(kptr + 40);
                            __m256 _w6 = _mm256_load_ps(kptr + 48);
                            __m256 _w7 = _mm256_load_ps(kptr + 56);

                            _sum0 = _mm256_comp_fmadd_ps(_w0, _mm256_set1_ps(r0[sok]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_w1, _mm256_set1_ps(r0[sok + N]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_w2, _mm256_set1_ps(r0[sok + N * 2]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_w3, _mm256_set1_ps(r0[sok + N * 3]), _sum3);
                            _sum0 = _mm256_comp_fmadd_ps(_w4, _mm256_set1_ps(r0[sok + N * 4]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_w5, _mm256_set1_ps(r0[sok + N * 5]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_w6, _mm256_set1_ps(r0[sok + N * 6]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_w7, _mm256_set1_ps(r0[sok + N * 7]), _sum3);

                            kptr += 64;
                        }
                    }
                }
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];

                            __m256 _w0 = _mm256_load_ps(kptr);
                            __m256 _w1 = _mm256_load_ps(kptr + 8);
                            __m256 _w2 = _mm256_load_ps(kptr + 16);
                            __m256 _w3 = _mm256_load_ps(kptr + 24);

                            _sum0 = _mm256_comp_fmadd_ps(_w0, _mm256_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_w1, _mm256_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_w2, _mm256_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_w3, _mm256_set1_ps(r0s[3]), _sum3);

                            kptr += 32;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m256 _w0 = _mm256_load_ps(kptr);
                            __m256 _w1 = _mm256_load_ps(kptr + 8);
                            __m256 _w2 = _mm256_load_ps(kptr + 16);
                            __m256 _w3 = _mm256_load_ps(kptr + 24);

                            _sum0 = _mm256_comp_fmadd_ps(_w0, _mm256_set1_ps(r0[sok]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_w1, _mm256_set1_ps(r0[sok + N]), _sum1);
                            _sum2 = _mm256_comp_fmadd_ps(_w2, _mm256_set1_ps(r0[sok + N * 2]), _sum2);
                            _sum3 = _mm256_comp_fmadd_ps(_w3, _mm256_set1_ps(r0[sok + N * 3]), _sum3);

                            kptr += 32;
                        }
                    }
                }
                for (; q + 1 < inch; q += 2)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m256 _w0 = _mm256_load_ps(kptr);
                            __m256 _w1 = _mm256_load_ps(kptr + 8);

                            _sum0 = _mm256_comp_fmadd_ps(_w0, _mm256_set1_ps(r0[sok]), _sum0);
                            _sum1 = _mm256_comp_fmadd_ps(_w1, _mm256_set1_ps(r0[sok + N]), _sum1);

                            kptr += 16;
                        }
                    }
                }
                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            __m256 _val = _mm256_set1_ps(r0[space_ofs[k]]);
                            __m256 _w = _mm256_load_ps(kptr);
                            _sum0 = _mm256_comp_fmadd_ps(_val, _w, _sum0);

                            kptr += 8;
                        }
                    }
                }

                _sum0 = _mm256_add_ps(_sum0, _sum1);
                _sum2 = _mm256_add_ps(_sum2, _sum3);
                _sum0 = _mm256_add_ps(_sum0, _sum2);

                _sum0 = activation_avx(_sum0, activation_type, activation_params);

                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr, _sum0);
                    outptr += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr, _mm256_extractf128_ps(_sum0, 0));
                    _mm_store_ps(outptr + M, _mm256_extractf128_ps(_sum0, 1));
                    outptr += 4;
                }
                if (out_elempack == 1)
                {
                    float sum[8];
                    _mm256_storeu_ps(sum, _sum0);

                    outptr[0] = sum[0];
                    outptr[M] = sum[1];
                    outptr[M * 2] = sum[2];
                    outptr[M * 3] = sum[3];
                    outptr[M * 4] = sum[4];
                    outptr[M * 5] = sum[5];
                    outptr[M * 6] = sum[6];
                    outptr[M * 7] = sum[7];
                    outptr += 1;
                }
            }
        }
    }
    remain_outch_start += nn_outch * 8;
    nn_outch = (outch - remain_outch_start) / 4;
#else // __AVX__
    nn_outch = (outch - remain_outch_start) / 4;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __AVX__
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 4;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.channel(p / out_elempack);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();
                __m128 _sum2 = _mm_setzero_ps();
                __m128 _sum3 = _mm_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm_loadu_ps(bias_data_ptr + p);
                }

#if __AVX512F__
                const float* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4);
#elif __AVX__
                const float* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4);
#else
                const float* kptr = weight_data_tm.channel(p / 4);
#endif

                int q = 0;
#if __AVX__
#if __AVX512F__
                for (; q + 15 < inch; q += 16)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 16)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];

                            __m128 _w0 = _mm_load_ps(kptr + 4 * 0);
                            __m128 _w1 = _mm_load_ps(kptr + 4 * 1);
                            __m128 _w2 = _mm_load_ps(kptr + 4 * 2);
                            __m128 _w3 = _mm_load_ps(kptr + 4 * 3);
                            __m128 _w4 = _mm_load_ps(kptr + 4 * 4);
                            __m128 _w5 = _mm_load_ps(kptr + 4 * 5);
                            __m128 _w6 = _mm_load_ps(kptr + 4 * 6);
                            __m128 _w7 = _mm_load_ps(kptr + 4 * 7);
                            __m128 _w8 = _mm_load_ps(kptr + 4 * 8);
                            __m128 _w9 = _mm_load_ps(kptr + 4 * 9);
                            __m128 _wa = _mm_load_ps(kptr + 4 * 10);
                            __m128 _wb = _mm_load_ps(kptr + 4 * 11);
                            __m128 _wc = _mm_load_ps(kptr + 4 * 12);
                            __m128 _wd = _mm_load_ps(kptr + 4 * 13);
                            __m128 _we = _mm_load_ps(kptr + 4 * 14);
                            __m128 _wf = _mm_load_ps(kptr + 4 * 15);

                            _sum0 = _mm_fmadd_ps(_w0, _mm_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm_fmadd_ps(_w1, _mm_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm_fmadd_ps(_w2, _mm_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm_fmadd_ps(_w3, _mm_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm_fmadd_ps(_w4, _mm_set1_ps(r0s[4]), _sum0);
                            _sum1 = _mm_fmadd_ps(_w5, _mm_set1_ps(r0s[5]), _sum1);
                            _sum2 = _mm_fmadd_ps(_w6, _mm_set1_ps(r0s[6]), _sum2);
                            _sum3 = _mm_fmadd_ps(_w7, _mm_set1_ps(r0s[7]), _sum3);
                            _sum0 = _mm_fmadd_ps(_w8, _mm_set1_ps(r0s[8]), _sum0);
                            _sum1 = _mm_fmadd_ps(_w9, _mm_set1_ps(r0s[9]), _sum1);
                            _sum2 = _mm_fmadd_ps(_wa, _mm_set1_ps(r0s[10]), _sum2);
                            _sum3 = _mm_fmadd_ps(_wb, _mm_set1_ps(r0s[11]), _sum3);
                            _sum0 = _mm_fmadd_ps(_wc, _mm_set1_ps(r0s[12]), _sum0);
                            _sum1 = _mm_fmadd_ps(_wd, _mm_set1_ps(r0s[13]), _sum1);
                            _sum2 = _mm_fmadd_ps(_we, _mm_set1_ps(r0s[14]), _sum2);
                            _sum3 = _mm_fmadd_ps(_wf, _mm_set1_ps(r0s[15]), _sum3);

                            kptr += 64;
                        }
                    }
                    if (elempack == 8)
                    {
                        const float* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];
                            const float* r1s = r1 + space_ofs[k];

                            __m128 _w0 = _mm_load_ps(kptr + 4 * 0);
                            __m128 _w1 = _mm_load_ps(kptr + 4 * 1);
                            __m128 _w2 = _mm_load_ps(kptr + 4 * 2);
                            __m128 _w3 = _mm_load_ps(kptr + 4 * 3);
                            __m128 _w4 = _mm_load_ps(kptr + 4 * 4);
                            __m128 _w5 = _mm_load_ps(kptr + 4 * 5);
                            __m128 _w6 = _mm_load_ps(kptr + 4 * 6);
                            __m128 _w7 = _mm_load_ps(kptr + 4 * 7);
                            __m128 _w8 = _mm_load_ps(kptr + 4 * 8);
                            __m128 _w9 = _mm_load_ps(kptr + 4 * 9);
                            __m128 _wa = _mm_load_ps(kptr + 4 * 10);
                            __m128 _wb = _mm_load_ps(kptr + 4 * 11);
                            __m128 _wc = _mm_load_ps(kptr + 4 * 12);
                            __m128 _wd = _mm_load_ps(kptr + 4 * 13);
                            __m128 _we = _mm_load_ps(kptr + 4 * 14);
                            __m128 _wf = _mm_load_ps(kptr + 4 * 15);

                            _sum0 = _mm_fmadd_ps(_w0, _mm_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm_fmadd_ps(_w1, _mm_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm_fmadd_ps(_w2, _mm_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm_fmadd_ps(_w3, _mm_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm_fmadd_ps(_w4, _mm_set1_ps(r0s[4]), _sum0);
                            _sum1 = _mm_fmadd_ps(_w5, _mm_set1_ps(r0s[5]), _sum1);
                            _sum2 = _mm_fmadd_ps(_w6, _mm_set1_ps(r0s[6]), _sum2);
                            _sum3 = _mm_fmadd_ps(_w7, _mm_set1_ps(r0s[7]), _sum3);
                            _sum0 = _mm_fmadd_ps(_w8, _mm_set1_ps(r1s[0]), _sum0);
                            _sum1 = _mm_fmadd_ps(_w9, _mm_set1_ps(r1s[1]), _sum1);
                            _sum2 = _mm_fmadd_ps(_wa, _mm_set1_ps(r1s[2]), _sum2);
                            _sum3 = _mm_fmadd_ps(_wb, _mm_set1_ps(r1s[3]), _sum3);
                            _sum0 = _mm_fmadd_ps(_wc, _mm_set1_ps(r1s[4]), _sum0);
                            _sum1 = _mm_fmadd_ps(_wd, _mm_set1_ps(r1s[5]), _sum1);
                            _sum2 = _mm_fmadd_ps(_we, _mm_set1_ps(r1s[6]), _sum2);
                            _sum3 = _mm_fmadd_ps(_wf, _mm_set1_ps(r1s[7]), _sum3);

                            kptr += 64;
                        }
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];
                            const float* r1s = r1 + space_ofs[k];
                            const float* r2s = r2 + space_ofs[k];
                            const float* r3s = r3 + space_ofs[k];

                            __m128 _w0 = _mm_load_ps(kptr + 4 * 0);
                            __m128 _w1 = _mm_load_ps(kptr + 4 * 1);
                            __m128 _w2 = _mm_load_ps(kptr + 4 * 2);
                            __m128 _w3 = _mm_load_ps(kptr + 4 * 3);
                            __m128 _w4 = _mm_load_ps(kptr + 4 * 4);
                            __m128 _w5 = _mm_load_ps(kptr + 4 * 5);
                            __m128 _w6 = _mm_load_ps(kptr + 4 * 6);
                            __m128 _w7 = _mm_load_ps(kptr + 4 * 7);
                            __m128 _w8 = _mm_load_ps(kptr + 4 * 8);
                            __m128 _w9 = _mm_load_ps(kptr + 4 * 9);
                            __m128 _wa = _mm_load_ps(kptr + 4 * 10);
                            __m128 _wb = _mm_load_ps(kptr + 4 * 11);
                            __m128 _wc = _mm_load_ps(kptr + 4 * 12);
                            __m128 _wd = _mm_load_ps(kptr + 4 * 13);
                            __m128 _we = _mm_load_ps(kptr + 4 * 14);
                            __m128 _wf = _mm_load_ps(kptr + 4 * 15);

                            _sum0 = _mm_fmadd_ps(_w0, _mm_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm_fmadd_ps(_w1, _mm_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm_fmadd_ps(_w2, _mm_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm_fmadd_ps(_w3, _mm_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm_fmadd_ps(_w4, _mm_set1_ps(r1s[0]), _sum0);
                            _sum1 = _mm_fmadd_ps(_w5, _mm_set1_ps(r1s[1]), _sum1);
                            _sum2 = _mm_fmadd_ps(_w6, _mm_set1_ps(r1s[2]), _sum2);
                            _sum3 = _mm_fmadd_ps(_w7, _mm_set1_ps(r1s[3]), _sum3);
                            _sum0 = _mm_fmadd_ps(_w8, _mm_set1_ps(r2s[0]), _sum0);
                            _sum1 = _mm_fmadd_ps(_w9, _mm_set1_ps(r2s[1]), _sum1);
                            _sum2 = _mm_fmadd_ps(_wa, _mm_set1_ps(r2s[2]), _sum2);
                            _sum3 = _mm_fmadd_ps(_wb, _mm_set1_ps(r2s[3]), _sum3);
                            _sum0 = _mm_fmadd_ps(_wc, _mm_set1_ps(r3s[0]), _sum0);
                            _sum1 = _mm_fmadd_ps(_wd, _mm_set1_ps(r3s[1]), _sum1);
                            _sum2 = _mm_fmadd_ps(_we, _mm_set1_ps(r3s[2]), _sum2);
                            _sum3 = _mm_fmadd_ps(_wf, _mm_set1_ps(r3s[3]), _sum3);

                            kptr += 64;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m128 _w0 = _mm_load_ps(kptr + 4 * 0);
                            __m128 _w1 = _mm_load_ps(kptr + 4 * 1);
                            __m128 _w2 = _mm_load_ps(kptr + 4 * 2);
                            __m128 _w3 = _mm_load_ps(kptr + 4 * 3);
                            __m128 _w4 = _mm_load_ps(kptr + 4 * 4);
                            __m128 _w5 = _mm_load_ps(kptr + 4 * 5);
                            __m128 _w6 = _mm_load_ps(kptr + 4 * 6);
                            __m128 _w7 = _mm_load_ps(kptr + 4 * 7);
                            __m128 _w8 = _mm_load_ps(kptr + 4 * 8);
                            __m128 _w9 = _mm_load_ps(kptr + 4 * 9);
                            __m128 _wa = _mm_load_ps(kptr + 4 * 10);
                            __m128 _wb = _mm_load_ps(kptr + 4 * 11);
                            __m128 _wc = _mm_load_ps(kptr + 4 * 12);
                            __m128 _wd = _mm_load_ps(kptr + 4 * 13);
                            __m128 _we = _mm_load_ps(kptr + 4 * 14);
                            __m128 _wf = _mm_load_ps(kptr + 4 * 15);

                            _sum0 = _mm_fmadd_ps(_w0, _mm_set1_ps(r0[sok]), _sum0);
                            _sum1 = _mm_fmadd_ps(_w1, _mm_set1_ps(r0[sok + N]), _sum1);
                            _sum2 = _mm_fmadd_ps(_w2, _mm_set1_ps(r0[sok + N * 2]), _sum2);
                            _sum3 = _mm_fmadd_ps(_w3, _mm_set1_ps(r0[sok + N * 3]), _sum3);
                            _sum0 = _mm_fmadd_ps(_w4, _mm_set1_ps(r0[sok + N * 4]), _sum0);
                            _sum1 = _mm_fmadd_ps(_w5, _mm_set1_ps(r0[sok + N * 5]), _sum1);
                            _sum2 = _mm_fmadd_ps(_w6, _mm_set1_ps(r0[sok + N * 6]), _sum2);
                            _sum3 = _mm_fmadd_ps(_w7, _mm_set1_ps(r0[sok + N * 7]), _sum3);
                            _sum0 = _mm_fmadd_ps(_w8, _mm_set1_ps(r0[sok + N * 8]), _sum0);
                            _sum1 = _mm_fmadd_ps(_w9, _mm_set1_ps(r0[sok + N * 9]), _sum1);
                            _sum2 = _mm_fmadd_ps(_wa, _mm_set1_ps(r0[sok + N * 10]), _sum2);
                            _sum3 = _mm_fmadd_ps(_wb, _mm_set1_ps(r0[sok + N * 11]), _sum3);
                            _sum0 = _mm_fmadd_ps(_wc, _mm_set1_ps(r0[sok + N * 12]), _sum0);
                            _sum1 = _mm_fmadd_ps(_wd, _mm_set1_ps(r0[sok + N * 13]), _sum1);
                            _sum2 = _mm_fmadd_ps(_we, _mm_set1_ps(r0[sok + N * 14]), _sum2);
                            _sum3 = _mm_fmadd_ps(_wf, _mm_set1_ps(r0[sok + N * 15]), _sum3);

                            kptr += 64;
                        }
                    }
                }
#endif // __AVX512F__
                for (; q + 7 < inch; q += 8)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 8)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];

                            __m128 _w0 = _mm_load_ps(kptr);
                            __m128 _w1 = _mm_load_ps(kptr + 4);
                            __m128 _w2 = _mm_load_ps(kptr + 8);
                            __m128 _w3 = _mm_load_ps(kptr + 12);
                            __m128 _w4 = _mm_load_ps(kptr + 16);
                            __m128 _w5 = _mm_load_ps(kptr + 20);
                            __m128 _w6 = _mm_load_ps(kptr + 24);
                            __m128 _w7 = _mm_load_ps(kptr + 28);

                            _sum0 = _mm_comp_fmadd_ps(_w0, _mm_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_w1, _mm_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm_comp_fmadd_ps(_w2, _mm_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm_comp_fmadd_ps(_w3, _mm_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm_comp_fmadd_ps(_w4, _mm_set1_ps(r0s[4]), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_w5, _mm_set1_ps(r0s[5]), _sum1);
                            _sum2 = _mm_comp_fmadd_ps(_w6, _mm_set1_ps(r0s[6]), _sum2);
                            _sum3 = _mm_comp_fmadd_ps(_w7, _mm_set1_ps(r0s[7]), _sum3);

                            kptr += 32;
                        }
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];
                            const float* r1s = r1 + space_ofs[k];

                            __m128 _w0 = _mm_load_ps(kptr);
                            __m128 _w1 = _mm_load_ps(kptr + 4);
                            __m128 _w2 = _mm_load_ps(kptr + 8);
                            __m128 _w3 = _mm_load_ps(kptr + 12);
                            __m128 _w4 = _mm_load_ps(kptr + 16);
                            __m128 _w5 = _mm_load_ps(kptr + 20);
                            __m128 _w6 = _mm_load_ps(kptr + 24);
                            __m128 _w7 = _mm_load_ps(kptr + 28);

                            _sum0 = _mm_comp_fmadd_ps(_w0, _mm_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_w1, _mm_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm_comp_fmadd_ps(_w2, _mm_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm_comp_fmadd_ps(_w3, _mm_set1_ps(r0s[3]), _sum3);
                            _sum0 = _mm_comp_fmadd_ps(_w4, _mm_set1_ps(r1s[0]), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_w5, _mm_set1_ps(r1s[1]), _sum1);
                            _sum2 = _mm_comp_fmadd_ps(_w6, _mm_set1_ps(r1s[2]), _sum2);
                            _sum3 = _mm_comp_fmadd_ps(_w7, _mm_set1_ps(r1s[3]), _sum3);

                            kptr += 32;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m128 _w0 = _mm_load_ps(kptr);
                            __m128 _w1 = _mm_load_ps(kptr + 4);
                            __m128 _w2 = _mm_load_ps(kptr + 8);
                            __m128 _w3 = _mm_load_ps(kptr + 12);
                            __m128 _w4 = _mm_load_ps(kptr + 16);
                            __m128 _w5 = _mm_load_ps(kptr + 20);
                            __m128 _w6 = _mm_load_ps(kptr + 24);
                            __m128 _w7 = _mm_load_ps(kptr + 28);

                            _sum0 = _mm_comp_fmadd_ps(_w0, _mm_set1_ps(r0[sok]), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_w1, _mm_set1_ps(r0[sok + N]), _sum1);
                            _sum2 = _mm_comp_fmadd_ps(_w2, _mm_set1_ps(r0[sok + N * 2]), _sum2);
                            _sum3 = _mm_comp_fmadd_ps(_w3, _mm_set1_ps(r0[sok + N * 3]), _sum3);
                            _sum0 = _mm_comp_fmadd_ps(_w4, _mm_set1_ps(r0[sok + N * 4]), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_w5, _mm_set1_ps(r0[sok + N * 5]), _sum1);
                            _sum2 = _mm_comp_fmadd_ps(_w6, _mm_set1_ps(r0[sok + N * 6]), _sum2);
                            _sum3 = _mm_comp_fmadd_ps(_w7, _mm_set1_ps(r0[sok + N * 7]), _sum3);

                            kptr += 32;
                        }
                    }
                }
#endif // __AVX__
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];

                            __m128 _w0 = _mm_load_ps(kptr);
                            __m128 _w1 = _mm_load_ps(kptr + 4);
                            __m128 _w2 = _mm_load_ps(kptr + 8);
                            __m128 _w3 = _mm_load_ps(kptr + 12);

                            _sum0 = _mm_comp_fmadd_ps(_w0, _mm_set1_ps(r0s[0]), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_w1, _mm_set1_ps(r0s[1]), _sum1);
                            _sum2 = _mm_comp_fmadd_ps(_w2, _mm_set1_ps(r0s[2]), _sum2);
                            _sum3 = _mm_comp_fmadd_ps(_w3, _mm_set1_ps(r0s[3]), _sum3);

                            kptr += 16;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m128 _w0 = _mm_load_ps(kptr);
                            __m128 _w1 = _mm_load_ps(kptr + 4);
                            __m128 _w2 = _mm_load_ps(kptr + 8);
                            __m128 _w3 = _mm_load_ps(kptr + 12);

                            _sum0 = _mm_comp_fmadd_ps(_w0, _mm_set1_ps(r0[sok]), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_w1, _mm_set1_ps(r0[sok + N]), _sum1);
                            _sum2 = _mm_comp_fmadd_ps(_w2, _mm_set1_ps(r0[sok + N * 2]), _sum2);
                            _sum3 = _mm_comp_fmadd_ps(_w3, _mm_set1_ps(r0[sok + N * 3]), _sum3);

                            kptr += 16;
                        }
                    }
                }
                for (; q + 1 < inch; q += 2)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            __m128 _w0 = _mm_load_ps(kptr);
                            __m128 _w1 = _mm_load_ps(kptr + 4);

                            _sum0 = _mm_comp_fmadd_ps(_w0, _mm_set1_ps(r0[sok]), _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_w1, _mm_set1_ps(r0[sok + N]), _sum1);

                            kptr += 8;
                        }
                    }
                }
                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            __m128 _val = _mm_set1_ps(r0[space_ofs[k]]);
                            __m128 _w = _mm_load_ps(kptr);
                            _sum0 = _mm_comp_fmadd_ps(_val, _w, _sum0);

                            kptr += 4;
                        }
                    }
                }

                _sum0 = _mm_add_ps(_sum0, _sum1);
                _sum2 = _mm_add_ps(_sum2, _sum3);
                _sum0 = _mm_add_ps(_sum0, _sum2);

                _sum0 = activation_sse(_sum0, activation_type, activation_params);

                if (out_elempack == 4)
                {
                    _mm_storeu_ps(outptr, _sum0);
                    outptr += 4;
                }
                if (out_elempack == 1)
                {
                    float sum[4];
                    _mm_storeu_ps(sum, _sum0);

                    outptr[0] = sum[0];
                    outptr[M] = sum[1];
                    outptr[M * 2] = sum[2];
                    outptr[M * 3] = sum[3];
                    outptr += 1;
                }
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
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int outw = top_blob.w;
        const int outh = top_blob.h;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum0 = 0.f;
                float sum1 = 0.f;

                if (bias_data_ptr)
                {
                    sum0 = bias_data_ptr[p];
                    sum1 = bias_data_ptr[p + 1];
                }

#if __AVX512F__
                const float* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4 + (p % 4) / 2);
#elif __AVX__
                const float* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2);
#elif __SSE2__
                const float* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2);
#else
                const float* kptr = weight_data_tm.channel(p / 2);
#endif

                int q = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum0_avx512 = _mm512_setzero_ps();
                __m512 _sum1_avx512 = _mm512_setzero_ps();
                for (; q + 15 < inch; q += 16)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 16)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m512 _r0 = _mm512_load_ps(r0 + sok);
                            __m512 _w0 = _mm512_load_ps(kptr);
                            __m512 _w1 = _mm512_load_ps(kptr + 16);
                            _sum0_avx512 = _mm512_fmadd_ps(_r0, _w0, _sum0_avx512);
                            _sum1_avx512 = _mm512_fmadd_ps(_r0, _w1, _sum1_avx512);

                            kptr += 32;
                        }
                    }
                    if (elempack == 8)
                    {
                        const float* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m512 _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + sok)), _mm256_load_ps(r1 + sok), 1);
                            __m512 _w0 = _mm512_load_ps(kptr);
                            __m512 _w1 = _mm512_load_ps(kptr + 16);
                            _sum0_avx512 = _mm512_fmadd_ps(_r0, _w0, _sum0_avx512);
                            _sum1_avx512 = _mm512_fmadd_ps(_r0, _w1, _sum1_avx512);

                            kptr += 32;
                        }
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m512 _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + sok)), _mm_load_ps(r1 + sok), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + sok)), _mm_load_ps(r3 + sok), 1), 1);
                            __m512 _w0 = _mm512_load_ps(kptr);
                            __m512 _w1 = _mm512_load_ps(kptr + 16);
                            _sum0_avx512 = _mm512_fmadd_ps(_r0, _w0, _sum0_avx512);
                            _sum1_avx512 = _mm512_fmadd_ps(_r0, _w1, _sum1_avx512);

                            kptr += 32;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m512 _r0 = _mm512_set_ps(r0[sok + N * 15], r0[sok + N * 14], r0[sok + N * 13], r0[sok + N * 12], r0[sok + N * 11], r0[sok + N * 10], r0[sok + N * 9], r0[sok + N * 8], r0[sok + N * 7], r0[sok + N * 6], r0[sok + N * 5], r0[sok + N * 4], r0[sok + N * 3], r0[sok + N * 2], r0[sok + N], r0[sok]);
                            __m512 _w0 = _mm512_load_ps(kptr);
                            __m512 _w1 = _mm512_load_ps(kptr + 16);
                            _sum0_avx512 = _mm512_fmadd_ps(_r0, _w0, _sum0_avx512);
                            _sum1_avx512 = _mm512_fmadd_ps(_r0, _w1, _sum1_avx512);

                            kptr += 32;
                        }
                    }
                }
                sum0 += _mm512_comp_reduce_add_ps(_sum0_avx512);
                sum1 += _mm512_comp_reduce_add_ps(_sum1_avx512);
#endif // __AVX512F__
                __m256 _sum0_avx = _mm256_setzero_ps();
                __m256 _sum1_avx = _mm256_setzero_ps();
                for (; q + 7 < inch; q += 8)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 8)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m256 _r0 = _mm256_load_ps(r0 + sok);
                            __m256 _w0 = _mm256_load_ps(kptr);
                            __m256 _w1 = _mm256_load_ps(kptr + 8);
                            _sum0_avx = _mm256_comp_fmadd_ps(_r0, _w0, _sum0_avx);
                            _sum1_avx = _mm256_comp_fmadd_ps(_r0, _w1, _sum1_avx);

                            kptr += 16;
                        }
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m256 _r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + sok)), _mm_load_ps(r1 + sok), 1);
                            __m256 _w0 = _mm256_load_ps(kptr);
                            __m256 _w1 = _mm256_load_ps(kptr + 8);
                            _sum0_avx = _mm256_comp_fmadd_ps(_r0, _w0, _sum0_avx);
                            _sum1_avx = _mm256_comp_fmadd_ps(_r0, _w1, _sum1_avx);

                            kptr += 16;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m256 _r0 = _mm256_set_ps(r0[sok + N * 7], r0[sok + N * 6], r0[sok + N * 5], r0[sok + N * 4], r0[sok + N * 3], r0[sok + N * 2], r0[sok + N], r0[sok]);
                            __m256 _w0 = _mm256_load_ps(kptr);
                            __m256 _w1 = _mm256_load_ps(kptr + 8);
                            _sum0_avx = _mm256_comp_fmadd_ps(_r0, _w0, _sum0_avx);
                            _sum1_avx = _mm256_comp_fmadd_ps(_r0, _w1, _sum1_avx);

                            kptr += 16;
                        }
                    }
                }
                sum0 += _mm256_reduce_add_ps(_sum0_avx);
                sum1 += _mm256_reduce_add_ps(_sum1_avx);
#endif // __AVX__
                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m128 _r0 = _mm_load_ps(r0 + sok);
                            __m128 _w0 = _mm_load_ps(kptr);
                            __m128 _w1 = _mm_load_ps(kptr + 4);
                            _sum0 = _mm_comp_fmadd_ps(_r0, _w0, _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_r0, _w1, _sum1);

                            kptr += 8;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m128 _r0 = _mm_set_ps(r0[sok + N * 3], r0[sok + N * 2], r0[sok + N], r0[sok]);
                            __m128 _w0 = _mm_load_ps(kptr);
                            __m128 _w1 = _mm_load_ps(kptr + 4);
                            _sum0 = _mm_comp_fmadd_ps(_r0, _w0, _sum0);
                            _sum1 = _mm_comp_fmadd_ps(_r0, _w1, _sum1);

                            kptr += 8;
                        }
                    }
                }
                sum0 += _mm_reduce_add_ps(_sum0);
                sum1 += _mm_reduce_add_ps(_sum1);
#endif // __SSE2__
                for (; q + 1 < inch; q += 2)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            sum0 += r0[sok] * kptr[0];
                            sum1 += r0[sok] * kptr[1];
                            sum0 += r0[sok + N] * kptr[2];
                            sum1 += r0[sok + N] * kptr[3];

                            kptr += 4;
                        }
                    }
                }
                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            float val = r0[space_ofs[k]];
                            sum0 += val * kptr[0];
                            sum1 += val * kptr[1];

                            kptr += 2;
                        }
                    }
                }

                sum0 = activation_ss(sum0, activation_type, activation_params);
                sum1 = activation_ss(sum1, activation_type, activation_params);

                outptr0[0] = sum0;
                outptr1[0] = sum1;
                outptr0 += 1;
                outptr1 += 1;
            }
        }
    }
    remain_outch_start += nn_outch * 2;
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_data_ptr)
                {
                    sum = bias_data_ptr[p];
                }

#if __AVX512F__
                const float* kptr = weight_data_tm.channel(p / 16 + (p % 16) / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#elif __AVX__
                const float* kptr = weight_data_tm.channel(p / 8 + (p % 8) / 4 + (p % 4) / 2 + p % 2);
#elif __SSE2__
                const float* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2 + p % 2);
#else
                const float* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

                int q = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum_avx512 = _mm512_setzero_ps();
                for (; q + 15 < inch; q += 16)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 16)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m512 _r0 = _mm512_load_ps(r0 + sok);
                            __m512 _w = _mm512_load_ps(kptr);
                            _sum_avx512 = _mm512_fmadd_ps(_r0, _w, _sum_avx512);

                            kptr += 16;
                        }
                    }
                    if (elempack == 8)
                    {
                        const float* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m512 _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0 + sok)), _mm256_load_ps(r1 + sok), 1);
                            __m512 _w = _mm512_load_ps(kptr);
                            _sum_avx512 = _mm512_fmadd_ps(_r0, _w, _sum_avx512);

                            kptr += 16;
                        }
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;
                        const float* r2 = r0 + N * 2;
                        const float* r3 = r0 + N * 3;

                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m512 _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + sok)), _mm_load_ps(r1 + sok), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2 + sok)), _mm_load_ps(r3 + sok), 1), 1);
                            __m512 _w = _mm512_load_ps(kptr);
                            _sum_avx512 = _mm512_fmadd_ps(_r0, _w, _sum_avx512);

                            kptr += 16;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m512 _r0 = _mm512_set_ps(r0[sok + N * 15], r0[sok + N * 14], r0[sok + N * 13], r0[sok + N * 12], r0[sok + N * 11], r0[sok + N * 10], r0[sok + N * 9], r0[sok + N * 8], r0[sok + N * 7], r0[sok + N * 6], r0[sok + N * 5], r0[sok + N * 4], r0[sok + N * 3], r0[sok + N * 2], r0[sok + N], r0[sok]);
                            __m512 _w = _mm512_load_ps(kptr);
                            _sum_avx512 = _mm512_fmadd_ps(_r0, _w, _sum_avx512);

                            kptr += 16;
                        }
                    }
                }
                sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
                __m256 _sum_avx = _mm256_setzero_ps();
                for (; q + 7 < inch; q += 8)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 8)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m256 _r0 = _mm256_load_ps(r0 + sok);
                            __m256 _w = _mm256_load_ps(kptr);
                            _sum_avx = _mm256_comp_fmadd_ps(_r0, _w, _sum_avx);

                            kptr += 8;
                        }
                    }
                    if (elempack == 4)
                    {
                        const float* r1 = r0 + N;

                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m256 _r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0 + sok)), _mm_load_ps(r1 + sok), 1);
                            __m256 _w = _mm256_load_ps(kptr);
                            _sum_avx = _mm256_comp_fmadd_ps(_r0, _w, _sum_avx);

                            kptr += 8;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m256 _r0 = _mm256_set_ps(r0[sok + N * 7], r0[sok + N * 6], r0[sok + N * 5], r0[sok + N * 4], r0[sok + N * 3], r0[sok + N * 2], r0[sok + N], r0[sok]);
                            __m256 _w = _mm256_load_ps(kptr);
                            _sum_avx = _mm256_comp_fmadd_ps(_r0, _w, _sum_avx);

                            kptr += 8;
                        }
                    }
                }
                sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
                __m128 _sum = _mm_setzero_ps();
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m128 _r0 = _mm_load_ps(r0 + sok);
                            __m128 _w = _mm_load_ps(kptr);
                            _sum = _mm_comp_fmadd_ps(_r0, _w, _sum);

                            kptr += 4;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            __m128 _r0 = _mm_set_ps(r0[sok + N * 3], r0[sok + N * 2], r0[sok + N], r0[sok]);
                            __m128 _w = _mm_load_ps(kptr);
                            _sum = _mm_comp_fmadd_ps(_r0, _w, _sum);

                            kptr += 4;
                        }
                    }
                }
                sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
                for (; q + 1 < inch; q += 2)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            sum += r0[sok] * kptr[0];
                            sum += r0[sok + N] * kptr[1];

                            kptr += 2;
                        }
                    }
                }
                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            float val = r0[space_ofs[k]];
                            sum += val * kptr[0];

                            kptr += 1;
                        }
                    }
                }

                sum = activation_ss(sum, activation_type, activation_params);

                outptr[0] = sum;
                outptr += 1;
            }
        }
    }
}
