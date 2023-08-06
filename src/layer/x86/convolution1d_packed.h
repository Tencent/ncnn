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

static void convolution1d_transform_kernel_packed(const Mat& kernel, Mat& kernel_tm, int inh, int outh, int kernel_w)
{
    // src = kw-inh-outh
    // dst = pb-pa-kw-inh/pa-outh/pb

    // clang-format off
    // *INDENT-OFF*
#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (outh >= 16)
    {
        if (inh >= 16)
            kernel_tm.create(16 * 16 * kernel_w, inh / 16 + (inh % 16) / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 16 + (outh % 16) / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 8)
            kernel_tm.create(16 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 16 + (outh % 16) / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 4)
            kernel_tm.create(16 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 16 + (outh % 16) / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 2)
            kernel_tm.create(16 * 2 * kernel_w, inh / 2 + inh % 2, outh / 16 + (outh % 16) / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else
            kernel_tm.create(16 * kernel_w, inh, outh / 16 + (outh % 16) / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
    }
    else
#endif // __AVX512F__
    if (outh >= 8)
    {
#if __AVX512F__
        if (inh >= 16)
            kernel_tm.create(8 * 16 * kernel_w, inh / 16 + (inh % 16) / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else
#endif // __AVX512F__
        if (inh >= 8)
            kernel_tm.create(8 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 4)
            kernel_tm.create(8 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 2)
            kernel_tm.create(8 * 2 * kernel_w, inh / 2 + inh % 2, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else
            kernel_tm.create(8 * kernel_w, inh, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
    }
    else
#endif // __AVX__
    if (outh >= 4)
    {
#if __AVX__
#if __AVX512F__
        if (inh >= 16)
            kernel_tm.create(4 * 16 * kernel_w, inh / 16 + (inh % 16) / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else
#endif // __AVX512F__
        if (inh >= 8)
            kernel_tm.create(4 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else
#endif // __AVX__
        if (inh >= 4)
            kernel_tm.create(4 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 2)
            kernel_tm.create(4 * 2 * kernel_w, inh / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else
            kernel_tm.create(4 * kernel_w, inh, outh / 4 + (outh % 4) / 2 + outh % 2);
    }
    else
#endif // __SSE2__
    if (outh >= 2)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (inh >= 16)
            kernel_tm.create(2 * 16 * kernel_w, inh / 16 + (inh % 16) / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2);
        else
#endif // __AVX512F__
        if (inh >= 8)
            kernel_tm.create(2 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2);
        else
#endif // __AVX__
        if (inh >= 4)
            kernel_tm.create(2 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2);
        else
#endif // __SSE2__
        if (inh >= 2)
            kernel_tm.create(2 * 2 * kernel_w, inh / 2 + inh % 2, outh / 2 + outh % 2);
        else
            kernel_tm.create(2 * kernel_w, inh, outh / 2 + outh % 2);
    }
    else
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (inh >= 16)
            kernel_tm.create(16 * kernel_w, inh / 16 + (inh % 16) / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh);
        else
#endif // __AVX512F__
        if (inh >= 8)
            kernel_tm.create(8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh);
        else
#endif // __AVX__
        if (inh >= 4)
            kernel_tm.create(4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh);
        else
#endif // __SSE2__
        if (inh >= 2)
            kernel_tm.create(2 * kernel_w, inh / 2 + inh % 2, outh);
        else
            kernel_tm.create(kernel_w, inh, outh);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; q + 15 < outh; q += 16)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;
        const float* kptr2 = (const float*)kernel + (q + 2) * inh * kernel_w;
        const float* kptr3 = (const float*)kernel + (q + 3) * inh * kernel_w;
        const float* kptr4 = (const float*)kernel + (q + 4) * inh * kernel_w;
        const float* kptr5 = (const float*)kernel + (q + 5) * inh * kernel_w;
        const float* kptr6 = (const float*)kernel + (q + 6) * inh * kernel_w;
        const float* kptr7 = (const float*)kernel + (q + 7) * inh * kernel_w;
        const float* kptr8 = (const float*)kernel + (q + 8) * inh * kernel_w;
        const float* kptr9 = (const float*)kernel + (q + 9) * inh * kernel_w;
        const float* kptra = (const float*)kernel + (q + 10) * inh * kernel_w;
        const float* kptrb = (const float*)kernel + (q + 11) * inh * kernel_w;
        const float* kptrc = (const float*)kernel + (q + 12) * inh * kernel_w;
        const float* kptrd = (const float*)kernel + (q + 13) * inh * kernel_w;
        const float* kptre = (const float*)kernel + (q + 14) * inh * kernel_w;
        const float* kptrf = (const float*)kernel + (q + 15) * inh * kernel_w;

        float* g00 = kernel_tm.channel(q / 16);

        __m512i _vindex = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(kernel_w));

        int p = 0;
        for (; p + 15 < inh; p += 16)
        {
            for (int k = 0; k < kernel_w; k++)
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

            kptr0 += kernel_w * 16;
            kptr1 += kernel_w * 16;
            kptr2 += kernel_w * 16;
            kptr3 += kernel_w * 16;
            kptr4 += kernel_w * 16;
            kptr5 += kernel_w * 16;
            kptr6 += kernel_w * 16;
            kptr7 += kernel_w * 16;
            kptr8 += kernel_w * 16;
            kptr9 += kernel_w * 16;
            kptra += kernel_w * 16;
            kptrb += kernel_w * 16;
            kptrc += kernel_w * 16;
            kptrd += kernel_w * 16;
            kptre += kernel_w * 16;
            kptrf += kernel_w * 16;
        }

        _vindex = _mm512_mullo_epi32(_vindex, _mm512_set1_epi32(inh));

        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;

                for (int i = 0; i < 8; i++)
                {
                    __m512 _k0 = _mm512_i32gather_ps(_vindex, k0, sizeof(float));
                    _mm512_store_ps(g00, _k0);
                    k0 += kernel_w;
                    g00 += 16;
                }
            }

            kptr0 += kernel_w * 8;
        }
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;

                for (int i = 0; i < 4; i++)
                {
                    __m512 _k0 = _mm512_i32gather_ps(_vindex, k0, sizeof(float));
                    _mm512_store_ps(g00, _k0);
                    k0 += kernel_w;
                    g00 += 16;
                }
            }

            kptr0 += kernel_w * 4;
        }
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;

                for (int i = 0; i < 2; i++)
                {
                    __m512 _k0 = _mm512_i32gather_ps(_vindex, k0, sizeof(float));
                    _mm512_store_ps(g00, _k0);
                    k0 += kernel_w;
                    g00 += 16;
                }
            }

            kptr0 += kernel_w * 2;
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;

                __m512 _k0 = _mm512_i32gather_ps(_vindex, k0, sizeof(float));
                _mm512_store_ps(g00, _k0);
                g00 += 16;
            }
        }
    }
#endif // __AVX512F__
    for (; q + 7 < outh; q += 8)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;
        const float* kptr2 = (const float*)kernel + (q + 2) * inh * kernel_w;
        const float* kptr3 = (const float*)kernel + (q + 3) * inh * kernel_w;
        const float* kptr4 = (const float*)kernel + (q + 4) * inh * kernel_w;
        const float* kptr5 = (const float*)kernel + (q + 5) * inh * kernel_w;
        const float* kptr6 = (const float*)kernel + (q + 6) * inh * kernel_w;
        const float* kptr7 = (const float*)kernel + (q + 7) * inh * kernel_w;

#if __AVX512F__
        float* g00 = kernel_tm.channel(q / 16 + (q % 16) / 8);
#else
        float* g00 = kernel_tm.channel(q / 8);
#endif

#if __AVX2__
        __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(kernel_w));
#if __AVX512F__
        __m512i _vindex_512 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex_512 = _mm512_mullo_epi32(_vindex_512, _mm512_set1_epi32(kernel_w));
#endif // __AVX512F__
#endif // __AVX2__

        int p = 0;
#if __AVX512F__
        for (; p + 15 < inh; p += 16)
        {
            for (int k = 0; k < kernel_w; k++)
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

            kptr0 += kernel_w * 16;
            kptr1 += kernel_w * 16;
            kptr2 += kernel_w * 16;
            kptr3 += kernel_w * 16;
            kptr4 += kernel_w * 16;
            kptr5 += kernel_w * 16;
            kptr6 += kernel_w * 16;
            kptr7 += kernel_w * 16;
        }
#endif // __AVX512F__
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
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
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    k4 += kernel_w;
                    k5 += kernel_w;
                    k6 += kernel_w;
                    k7 += kernel_w;
                    g00 += 8;
                }
#endif // __AVX2__
            }

            kptr0 += kernel_w * 8;
            kptr1 += kernel_w * 8;
            kptr2 += kernel_w * 8;
            kptr3 += kernel_w * 8;
            kptr4 += kernel_w * 8;
            kptr5 += kernel_w * 8;
            kptr6 += kernel_w * 8;
            kptr7 += kernel_w * 8;
        }

#if __AVX2__
        _vindex = _mm256_mullo_epi32(_vindex, _mm256_set1_epi32(inh));
#endif // __AVX2__

        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
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
                    k0 += kernel_w;
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
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    k4 += kernel_w;
                    k5 += kernel_w;
                    k6 += kernel_w;
                    k7 += kernel_w;
                    g00 += 8;
#endif // __AVX2__
                }
            }

            kptr0 += kernel_w * 4;
#if !__AVX2__
            kptr1 += kernel_w * 4;
            kptr2 += kernel_w * 4;
            kptr3 += kernel_w * 4;
            kptr4 += kernel_w * 4;
            kptr5 += kernel_w * 4;
            kptr6 += kernel_w * 4;
            kptr7 += kernel_w * 4;
#endif // !__AVX2__
        }
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
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
                    k0 += kernel_w;
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
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    k4 += kernel_w;
                    k5 += kernel_w;
                    k6 += kernel_w;
                    k7 += kernel_w;
                    g00 += 8;
#endif // __AVX2__
                }
            }

            kptr0 += kernel_w * 2;
#if !__AVX2__
            kptr1 += kernel_w * 2;
            kptr2 += kernel_w * 2;
            kptr3 += kernel_w * 2;
            kptr4 += kernel_w * 2;
            kptr5 += kernel_w * 2;
            kptr6 += kernel_w * 2;
            kptr7 += kernel_w * 2;
#endif // !__AVX2__
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
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
    for (; q + 3 < outh; q += 4)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;
        const float* kptr2 = (const float*)kernel + (q + 2) * inh * kernel_w;
        const float* kptr3 = (const float*)kernel + (q + 3) * inh * kernel_w;

#if __AVX512F__
        float* g00 = kernel_tm.channel(q / 16 + (q % 16) / 8 + (q % 8) / 4);
#elif __AVX__
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        float* g00 = kernel_tm.channel(q / 4);
#endif

#if __AVX2__
        __m128i _vindex = _mm_setr_epi32(0, 1, 2, 3);
        _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(kernel_w));
        __m256i _vindex_256 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _vindex_256 = _mm256_mullo_epi32(_vindex_256, _mm256_set1_epi32(kernel_w));
#if __AVX512F__
        __m512i _vindex_512 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex_512 = _mm512_mullo_epi32(_vindex_512, _mm512_set1_epi32(kernel_w));
#endif // __AVX512F__
#endif // __AVX2__

        int p = 0;
#if __AVX__
#if __AVX512F__
        for (; p + 15 < inh; p += 16)
        {
            for (int k = 0; k < kernel_w; k++)
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

            kptr0 += kernel_w * 16;
            kptr1 += kernel_w * 16;
            kptr2 += kernel_w * 16;
            kptr3 += kernel_w * 16;
        }
#endif // __AVX512F__
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
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
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    g00 += 4;
                }
#endif // __AVX2__
            }

            kptr0 += kernel_w * 8;
            kptr1 += kernel_w * 8;
            kptr2 += kernel_w * 8;
            kptr3 += kernel_w * 8;
        }
#endif // __AVX__
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
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
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    g00 += 4;
                }
#endif // __AVX2__
            }

            kptr0 += kernel_w * 4;
            kptr1 += kernel_w * 4;
            kptr2 += kernel_w * 4;
            kptr3 += kernel_w * 4;
        }

#if __AVX2__
        _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(inh));
#endif // __AVX2__

        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
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
                    k0 += kernel_w;
                    g00 += 4;
#else  // __AVX2__
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    g00 += 4;
#endif // __AVX2__
                }
            }

            kptr0 += kernel_w * 2;
#if !__AVX2__
            kptr1 += kernel_w * 2;
            kptr2 += kernel_w * 2;
            kptr3 += kernel_w * 2;
#endif // !__AVX2__
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
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
    for (; q + 1 < outh; q += 2)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;

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
        _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(kernel_w));
        __m256i _vindex_256 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _vindex_256 = _mm256_mullo_epi32(_vindex_256, _mm256_set1_epi32(kernel_w));
#if __AVX512F__
        __m512i _vindex_512 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex_512 = _mm512_mullo_epi32(_vindex_512, _mm512_set1_epi32(kernel_w));
#endif // __AVX512F__
#endif // __AVX2__

        int p = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; p + 15 < inh; p += 16)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                __m512 _k0 = _mm512_i32gather_ps(_vindex_512, k0, sizeof(float));
                __m512 _k1 = _mm512_i32gather_ps(_vindex_512, k1, sizeof(float));
                _mm512_storeu_ps(g00, _k0);
                _mm512_storeu_ps(g00 + 16, _k1);
                g00 += 32;
            }

            kptr0 += kernel_w * 16;
            kptr1 += kernel_w * 16;
        }
#endif // __AVX512F__
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
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
                g00[1] = k0[kernel_w];
                g00[2] = k0[kernel_w * 2];
                g00[3] = k0[kernel_w * 3];
                g00[4] = k0[kernel_w * 4];
                g00[5] = k0[kernel_w * 5];
                g00[6] = k0[kernel_w * 6];
                g00[7] = k0[kernel_w * 7];
                g00[8] = k1[0];
                g00[9] = k1[kernel_w];
                g00[10] = k1[kernel_w * 2];
                g00[11] = k1[kernel_w * 3];
                g00[12] = k1[kernel_w * 4];
                g00[13] = k1[kernel_w * 5];
                g00[14] = k1[kernel_w * 6];
                g00[15] = k1[kernel_w * 7];
                g00 += 16;
#endif // __AVX2__
            }

            kptr0 += kernel_w * 8;
            kptr1 += kernel_w * 8;
        }
#endif // __AVX__
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
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
                g00[1] = k0[kernel_w];
                g00[2] = k0[kernel_w * 2];
                g00[3] = k0[kernel_w * 3];
                g00[4] = k1[0];
                g00[5] = k1[kernel_w];
                g00[6] = k1[kernel_w * 2];
                g00[7] = k1[kernel_w * 3];
                g00 += 8;
#endif // __AVX2__
            }

            kptr0 += kernel_w * 4;
            kptr1 += kernel_w * 4;
        }
#endif // __SSE2__
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    k0 += kernel_w;
                    k1 += kernel_w;
                    g00 += 2;
                }
            }

            kptr0 += kernel_w * 2;
            kptr1 += kernel_w * 2;
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00 += 2;
            }
        }
    }
    for (; q < outh; q++)
    {
        const float* kptr = (const float*)kernel + q * inh * kernel_w;

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
        _vindex = _mm_mullo_epi32(_vindex, _mm_set1_epi32(kernel_w));
        __m256i _vindex_256 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        _vindex_256 = _mm256_mullo_epi32(_vindex_256, _mm256_set1_epi32(kernel_w));
#if __AVX512F__
        __m512i _vindex_512 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        _vindex_512 = _mm512_mullo_epi32(_vindex_512, _mm512_set1_epi32(kernel_w));
#endif // __AVX512F__
#endif // __AVX2__

        int p = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; p + 15 < inh; p += 16)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr + k;

                __m512 _k0 = _mm512_i32gather_ps(_vindex_512, k0, sizeof(float));
                _mm512_storeu_ps(g00, _k0);
                g00 += 16;
            }

            kptr += kernel_w * 16;
        }
#endif // __AVX512F__
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
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
                    k0 += kernel_w;
                    g00 += 1;
                }
#endif // __AVX2__
            }

            kptr += kernel_w * 8;
        }
#endif // __AVX__
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
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
                    k0 += kernel_w;
                    g00 += 1;
                }
#endif // __AVX2__
            }

            kptr += kernel_w * 4;
        }
#endif // __SSE2__
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[0];
                    k0 += kernel_w;
                    g00 += 1;
                }
            }

            kptr += kernel_w * 2;
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr + k;
                g00[0] = k0[0];
                g00++;
            }
        }
    }
}

static void convolution1d_packed(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int dilation_w, int stride_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int elempack = bottom_blob.elempack;
    const int inh = bottom_blob.h * elempack;

    const int N = bottom_blob.w * elempack;

    const int outw = top_blob.w;
    const int out_elempack = top_blob.elempack;
    const int outh = top_blob.h * out_elempack;

    const int M = top_blob.w * out_elempack;

    const float* bias_data_ptr = bias_data;

    int nn_outh = 0;
    int remain_outh_start = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    nn_outh = outh / 16;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outh; pp++)
    {
        const int p = pp * 16;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.row(p / out_elempack);

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
            for (; q + 15 < inh; q += 16)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 16)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
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

                        _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r0[4]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r0[5]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r0[6]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r0[7]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_w8, _mm512_set1_ps(r0[8]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w9, _mm512_set1_ps(r0[9]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_wa, _mm512_set1_ps(r0[10]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_wb, _mm512_set1_ps(r0[11]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_wc, _mm512_set1_ps(r0[12]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_wd, _mm512_set1_ps(r0[13]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_we, _mm512_set1_ps(r0[14]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_wf, _mm512_set1_ps(r0[15]), _sum3);

                        r0 += dilation_w * 16;
                        kptr += 256;
                    }
                }
                if (elempack == 8)
                {
                    const float* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
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

                        _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r0[4]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r0[5]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r0[6]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r0[7]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_w8, _mm512_set1_ps(r1[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w9, _mm512_set1_ps(r1[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_wa, _mm512_set1_ps(r1[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_wb, _mm512_set1_ps(r1[3]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_wc, _mm512_set1_ps(r1[4]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_wd, _mm512_set1_ps(r1[5]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_we, _mm512_set1_ps(r1[6]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_wf, _mm512_set1_ps(r1[7]), _sum3);

                        r0 += dilation_w * 8;
                        r1 += dilation_w * 8;
                        kptr += 256;
                    }
                }
                if (elempack == 4)
                {
                    const float* r1 = r0 + N;
                    const float* r2 = r0 + N * 2;
                    const float* r3 = r0 + N * 3;

                    for (int k = 0; k < kernel_w; k++)
                    {
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

                        _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r1[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r1[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r1[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r1[3]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_w8, _mm512_set1_ps(r2[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w9, _mm512_set1_ps(r2[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_wa, _mm512_set1_ps(r2[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_wb, _mm512_set1_ps(r2[3]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_wc, _mm512_set1_ps(r3[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_wd, _mm512_set1_ps(r3[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_we, _mm512_set1_ps(r3[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_wf, _mm512_set1_ps(r3[3]), _sum3);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        r2 += dilation_w * 4;
                        r3 += dilation_w * 4;
                        kptr += 256;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
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

                        _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[N]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0[N * 2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0[N * 3]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r0[N * 4]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r0[N * 5]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r0[N * 6]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r0[N * 7]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_w8, _mm512_set1_ps(r0[N * 8]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w9, _mm512_set1_ps(r0[N * 9]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_wa, _mm512_set1_ps(r0[N * 10]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_wb, _mm512_set1_ps(r0[N * 11]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_wc, _mm512_set1_ps(r0[N * 12]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_wd, _mm512_set1_ps(r0[N * 13]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_we, _mm512_set1_ps(r0[N * 14]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_wf, _mm512_set1_ps(r0[N * 15]), _sum3);

                        r0 += dilation_w;
                        kptr += 256;
                    }
                }
            }
            for (; q + 7 < inh; q += 8)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 8)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _w0 = _mm512_load_ps(kptr + 16 * 0);
                        __m512 _w1 = _mm512_load_ps(kptr + 16 * 1);
                        __m512 _w2 = _mm512_load_ps(kptr + 16 * 2);
                        __m512 _w3 = _mm512_load_ps(kptr + 16 * 3);
                        __m512 _w4 = _mm512_load_ps(kptr + 16 * 4);
                        __m512 _w5 = _mm512_load_ps(kptr + 16 * 5);
                        __m512 _w6 = _mm512_load_ps(kptr + 16 * 6);
                        __m512 _w7 = _mm512_load_ps(kptr + 16 * 7);

                        _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r0[4]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r0[5]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r0[6]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r0[7]), _sum3);

                        r0 += dilation_w * 8;
                        kptr += 128;
                    }
                }
                if (elempack == 4)
                {
                    const float* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _w0 = _mm512_load_ps(kptr + 16 * 0);
                        __m512 _w1 = _mm512_load_ps(kptr + 16 * 1);
                        __m512 _w2 = _mm512_load_ps(kptr + 16 * 2);
                        __m512 _w3 = _mm512_load_ps(kptr + 16 * 3);
                        __m512 _w4 = _mm512_load_ps(kptr + 16 * 4);
                        __m512 _w5 = _mm512_load_ps(kptr + 16 * 5);
                        __m512 _w6 = _mm512_load_ps(kptr + 16 * 6);
                        __m512 _w7 = _mm512_load_ps(kptr + 16 * 7);

                        _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r1[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r1[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r1[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r1[3]), _sum3);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 128;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _w0 = _mm512_load_ps(kptr + 16 * 0);
                        __m512 _w1 = _mm512_load_ps(kptr + 16 * 1);
                        __m512 _w2 = _mm512_load_ps(kptr + 16 * 2);
                        __m512 _w3 = _mm512_load_ps(kptr + 16 * 3);
                        __m512 _w4 = _mm512_load_ps(kptr + 16 * 4);
                        __m512 _w5 = _mm512_load_ps(kptr + 16 * 5);
                        __m512 _w6 = _mm512_load_ps(kptr + 16 * 6);
                        __m512 _w7 = _mm512_load_ps(kptr + 16 * 7);

                        _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[N]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0[N * 2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0[N * 3]), _sum3);
                        _sum0 = _mm512_fmadd_ps(_w4, _mm512_set1_ps(r0[N * 4]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w5, _mm512_set1_ps(r0[N * 5]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w6, _mm512_set1_ps(r0[N * 6]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w7, _mm512_set1_ps(r0[N * 7]), _sum3);

                        r0 += dilation_w;
                        kptr += 128;
                    }
                }
            }
            for (; q + 3 < inh; q += 4)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _w0 = _mm512_load_ps(kptr);
                        __m512 _w1 = _mm512_load_ps(kptr + 16);
                        __m512 _w2 = _mm512_load_ps(kptr + 32);
                        __m512 _w3 = _mm512_load_ps(kptr + 48);

                        _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0[3]), _sum3);

                        r0 += dilation_w * 4;
                        kptr += 64;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _w0 = _mm512_load_ps(kptr);
                        __m512 _w1 = _mm512_load_ps(kptr + 16);
                        __m512 _w2 = _mm512_load_ps(kptr + 32);
                        __m512 _w3 = _mm512_load_ps(kptr + 48);

                        _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[N]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_w2, _mm512_set1_ps(r0[N * 2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_w3, _mm512_set1_ps(r0[N * 3]), _sum3);

                        r0 += dilation_w;
                        kptr += 64;
                    }
                }
            }
            for (; q + 1 < inh; q += 2)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _w0 = _mm512_load_ps(kptr);
                        __m512 _w1 = _mm512_load_ps(kptr + 16);

                        _sum0 = _mm512_fmadd_ps(_w0, _mm512_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_w1, _mm512_set1_ps(r0[N]), _sum1);

                        r0 += dilation_w;
                        kptr += 32;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _val = _mm512_set1_ps(r0[0]);
                        __m512 _w = _mm512_load_ps(kptr);
                        _sum0 = _mm512_fmadd_ps(_val, _w, _sum0);

                        r0 += dilation_w;
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
    remain_outh_start += nn_outh * 16;
    nn_outh = (outh - remain_outh_start) / 8;
#else // __AVX512F__
    nn_outh = (outh - remain_outh_start) / 8;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __AVX512F__
    for (int pp = 0; pp < nn_outh; pp++)
    {
        const int p = remain_outh_start + pp * 8;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.row(p / out_elempack);

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
            for (; q + 15 < inh; q += 16)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 16)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
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

                        _sum0 = _mm256_fmadd_ps(_w0, _mm256_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_w1, _mm256_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_w2, _mm256_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_w3, _mm256_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm256_fmadd_ps(_w4, _mm256_set1_ps(r0[4]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_w5, _mm256_set1_ps(r0[5]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_w6, _mm256_set1_ps(r0[6]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_w7, _mm256_set1_ps(r0[7]), _sum3);
                        _sum0 = _mm256_fmadd_ps(_w8, _mm256_set1_ps(r0[8]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_w9, _mm256_set1_ps(r0[9]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_wa, _mm256_set1_ps(r0[10]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_wb, _mm256_set1_ps(r0[11]), _sum3);
                        _sum0 = _mm256_fmadd_ps(_wc, _mm256_set1_ps(r0[12]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_wd, _mm256_set1_ps(r0[13]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_we, _mm256_set1_ps(r0[14]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_wf, _mm256_set1_ps(r0[15]), _sum3);

                        r0 += dilation_w * 16;
                        kptr += 128;
                    }
                }
                if (elempack == 8)
                {
                    const float* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
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

                        _sum0 = _mm256_fmadd_ps(_w0, _mm256_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_w1, _mm256_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_w2, _mm256_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_w3, _mm256_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm256_fmadd_ps(_w4, _mm256_set1_ps(r0[4]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_w5, _mm256_set1_ps(r0[5]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_w6, _mm256_set1_ps(r0[6]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_w7, _mm256_set1_ps(r0[7]), _sum3);
                        _sum0 = _mm256_fmadd_ps(_w8, _mm256_set1_ps(r1[0]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_w9, _mm256_set1_ps(r1[1]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_wa, _mm256_set1_ps(r1[2]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_wb, _mm256_set1_ps(r1[3]), _sum3);
                        _sum0 = _mm256_fmadd_ps(_wc, _mm256_set1_ps(r1[4]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_wd, _mm256_set1_ps(r1[5]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_we, _mm256_set1_ps(r1[6]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_wf, _mm256_set1_ps(r1[7]), _sum3);

                        r0 += dilation_w * 8;
                        r1 += dilation_w * 8;
                        kptr += 128;
                    }
                }
                if (elempack == 4)
                {
                    const float* r1 = r0 + N;
                    const float* r2 = r0 + N * 2;
                    const float* r3 = r0 + N * 3;

                    for (int k = 0; k < kernel_w; k++)
                    {
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

                        _sum0 = _mm256_fmadd_ps(_w0, _mm256_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_w1, _mm256_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_w2, _mm256_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_w3, _mm256_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm256_fmadd_ps(_w4, _mm256_set1_ps(r1[0]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_w5, _mm256_set1_ps(r1[1]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_w6, _mm256_set1_ps(r1[2]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_w7, _mm256_set1_ps(r1[3]), _sum3);
                        _sum0 = _mm256_fmadd_ps(_w8, _mm256_set1_ps(r2[0]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_w9, _mm256_set1_ps(r2[1]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_wa, _mm256_set1_ps(r2[2]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_wb, _mm256_set1_ps(r2[3]), _sum3);
                        _sum0 = _mm256_fmadd_ps(_wc, _mm256_set1_ps(r3[0]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_wd, _mm256_set1_ps(r3[1]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_we, _mm256_set1_ps(r3[2]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_wf, _mm256_set1_ps(r3[3]), _sum3);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        r2 += dilation_w * 4;
                        r3 += dilation_w * 4;
                        kptr += 128;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
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

                        _sum0 = _mm256_fmadd_ps(_w0, _mm256_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_w1, _mm256_set1_ps(r0[N]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_w2, _mm256_set1_ps(r0[N * 2]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_w3, _mm256_set1_ps(r0[N * 3]), _sum3);
                        _sum0 = _mm256_fmadd_ps(_w4, _mm256_set1_ps(r0[N * 4]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_w5, _mm256_set1_ps(r0[N * 5]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_w6, _mm256_set1_ps(r0[N * 6]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_w7, _mm256_set1_ps(r0[N * 7]), _sum3);
                        _sum0 = _mm256_fmadd_ps(_w8, _mm256_set1_ps(r0[N * 8]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_w9, _mm256_set1_ps(r0[N * 9]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_wa, _mm256_set1_ps(r0[N * 10]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_wb, _mm256_set1_ps(r0[N * 11]), _sum3);
                        _sum0 = _mm256_fmadd_ps(_wc, _mm256_set1_ps(r0[N * 12]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_wd, _mm256_set1_ps(r0[N * 13]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_we, _mm256_set1_ps(r0[N * 14]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_wf, _mm256_set1_ps(r0[N * 15]), _sum3);

                        r0 += dilation_w;
                        kptr += 128;
                    }
                }
            }
#endif // __AVX512F__
            for (; q + 7 < inh; q += 8)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 8)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = _mm256_load_ps(kptr);
                        __m256 _w1 = _mm256_load_ps(kptr + 8);
                        __m256 _w2 = _mm256_load_ps(kptr + 16);
                        __m256 _w3 = _mm256_load_ps(kptr + 24);
                        __m256 _w4 = _mm256_load_ps(kptr + 32);
                        __m256 _w5 = _mm256_load_ps(kptr + 40);
                        __m256 _w6 = _mm256_load_ps(kptr + 48);
                        __m256 _w7 = _mm256_load_ps(kptr + 56);

                        _sum0 = _mm256_comp_fmadd_ps(_w0, _mm256_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_w1, _mm256_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_w2, _mm256_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_w3, _mm256_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm256_comp_fmadd_ps(_w4, _mm256_set1_ps(r0[4]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_w5, _mm256_set1_ps(r0[5]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_w6, _mm256_set1_ps(r0[6]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_w7, _mm256_set1_ps(r0[7]), _sum3);

                        r0 += dilation_w * 8;
                        kptr += 64;
                    }
                }
                if (elempack == 4)
                {
                    const float* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = _mm256_load_ps(kptr);
                        __m256 _w1 = _mm256_load_ps(kptr + 8);
                        __m256 _w2 = _mm256_load_ps(kptr + 16);
                        __m256 _w3 = _mm256_load_ps(kptr + 24);
                        __m256 _w4 = _mm256_load_ps(kptr + 32);
                        __m256 _w5 = _mm256_load_ps(kptr + 40);
                        __m256 _w6 = _mm256_load_ps(kptr + 48);
                        __m256 _w7 = _mm256_load_ps(kptr + 56);

                        _sum0 = _mm256_comp_fmadd_ps(_w0, _mm256_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_w1, _mm256_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_w2, _mm256_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_w3, _mm256_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm256_comp_fmadd_ps(_w4, _mm256_set1_ps(r1[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_w5, _mm256_set1_ps(r1[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_w6, _mm256_set1_ps(r1[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_w7, _mm256_set1_ps(r1[3]), _sum3);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 64;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = _mm256_load_ps(kptr);
                        __m256 _w1 = _mm256_load_ps(kptr + 8);
                        __m256 _w2 = _mm256_load_ps(kptr + 16);
                        __m256 _w3 = _mm256_load_ps(kptr + 24);
                        __m256 _w4 = _mm256_load_ps(kptr + 32);
                        __m256 _w5 = _mm256_load_ps(kptr + 40);
                        __m256 _w6 = _mm256_load_ps(kptr + 48);
                        __m256 _w7 = _mm256_load_ps(kptr + 56);

                        _sum0 = _mm256_comp_fmadd_ps(_w0, _mm256_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_w1, _mm256_set1_ps(r0[N]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_w2, _mm256_set1_ps(r0[N * 2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_w3, _mm256_set1_ps(r0[N * 3]), _sum3);
                        _sum0 = _mm256_comp_fmadd_ps(_w4, _mm256_set1_ps(r0[N * 4]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_w5, _mm256_set1_ps(r0[N * 5]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_w6, _mm256_set1_ps(r0[N * 6]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_w7, _mm256_set1_ps(r0[N * 7]), _sum3);

                        r0 += dilation_w;
                        kptr += 64;
                    }
                }
            }
            for (; q + 3 < inh; q += 4)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = _mm256_load_ps(kptr);
                        __m256 _w1 = _mm256_load_ps(kptr + 8);
                        __m256 _w2 = _mm256_load_ps(kptr + 16);
                        __m256 _w3 = _mm256_load_ps(kptr + 24);

                        _sum0 = _mm256_comp_fmadd_ps(_w0, _mm256_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_w1, _mm256_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_w2, _mm256_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_w3, _mm256_set1_ps(r0[3]), _sum3);

                        r0 += dilation_w * 4;
                        kptr += 32;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = _mm256_load_ps(kptr);
                        __m256 _w1 = _mm256_load_ps(kptr + 8);
                        __m256 _w2 = _mm256_load_ps(kptr + 16);
                        __m256 _w3 = _mm256_load_ps(kptr + 24);

                        _sum0 = _mm256_comp_fmadd_ps(_w0, _mm256_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_w1, _mm256_set1_ps(r0[N]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_w2, _mm256_set1_ps(r0[N * 2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_w3, _mm256_set1_ps(r0[N * 3]), _sum3);

                        r0 += dilation_w;
                        kptr += 32;
                    }
                }
            }
            for (; q + 1 < inh; q += 2)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _w0 = _mm256_load_ps(kptr);
                        __m256 _w1 = _mm256_load_ps(kptr + 8);

                        _sum0 = _mm256_comp_fmadd_ps(_w0, _mm256_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_w1, _mm256_set1_ps(r0[N]), _sum1);

                        r0 += dilation_w;
                        kptr += 16;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _val = _mm256_set1_ps(r0[0]);
                        __m256 _w = _mm256_load_ps(kptr);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w, _sum0);

                        r0 += dilation_w;
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
    remain_outh_start += nn_outh * 8;
    nn_outh = (outh - remain_outh_start) / 4;
#else // __AVX__
    nn_outh = (outh - remain_outh_start) / 4;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __AVX__
    for (int pp = 0; pp < nn_outh; pp++)
    {
        const int p = remain_outh_start + pp * 4;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.row(p / out_elempack);

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
            for (; q + 15 < inh; q += 16)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 16)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
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

                        _sum0 = _mm_fmadd_ps(_w0, _mm_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm_fmadd_ps(_w1, _mm_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm_fmadd_ps(_w2, _mm_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm_fmadd_ps(_w3, _mm_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm_fmadd_ps(_w4, _mm_set1_ps(r0[4]), _sum0);
                        _sum1 = _mm_fmadd_ps(_w5, _mm_set1_ps(r0[5]), _sum1);
                        _sum2 = _mm_fmadd_ps(_w6, _mm_set1_ps(r0[6]), _sum2);
                        _sum3 = _mm_fmadd_ps(_w7, _mm_set1_ps(r0[7]), _sum3);
                        _sum0 = _mm_fmadd_ps(_w8, _mm_set1_ps(r0[8]), _sum0);
                        _sum1 = _mm_fmadd_ps(_w9, _mm_set1_ps(r0[9]), _sum1);
                        _sum2 = _mm_fmadd_ps(_wa, _mm_set1_ps(r0[10]), _sum2);
                        _sum3 = _mm_fmadd_ps(_wb, _mm_set1_ps(r0[11]), _sum3);
                        _sum0 = _mm_fmadd_ps(_wc, _mm_set1_ps(r0[12]), _sum0);
                        _sum1 = _mm_fmadd_ps(_wd, _mm_set1_ps(r0[13]), _sum1);
                        _sum2 = _mm_fmadd_ps(_we, _mm_set1_ps(r0[14]), _sum2);
                        _sum3 = _mm_fmadd_ps(_wf, _mm_set1_ps(r0[15]), _sum3);

                        r0 += dilation_w * 16;
                        kptr += 64;
                    }
                }
                if (elempack == 8)
                {
                    const float* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
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

                        _sum0 = _mm_fmadd_ps(_w0, _mm_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm_fmadd_ps(_w1, _mm_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm_fmadd_ps(_w2, _mm_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm_fmadd_ps(_w3, _mm_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm_fmadd_ps(_w4, _mm_set1_ps(r0[4]), _sum0);
                        _sum1 = _mm_fmadd_ps(_w5, _mm_set1_ps(r0[5]), _sum1);
                        _sum2 = _mm_fmadd_ps(_w6, _mm_set1_ps(r0[6]), _sum2);
                        _sum3 = _mm_fmadd_ps(_w7, _mm_set1_ps(r0[7]), _sum3);
                        _sum0 = _mm_fmadd_ps(_w8, _mm_set1_ps(r1[0]), _sum0);
                        _sum1 = _mm_fmadd_ps(_w9, _mm_set1_ps(r1[1]), _sum1);
                        _sum2 = _mm_fmadd_ps(_wa, _mm_set1_ps(r1[2]), _sum2);
                        _sum3 = _mm_fmadd_ps(_wb, _mm_set1_ps(r1[3]), _sum3);
                        _sum0 = _mm_fmadd_ps(_wc, _mm_set1_ps(r1[4]), _sum0);
                        _sum1 = _mm_fmadd_ps(_wd, _mm_set1_ps(r1[5]), _sum1);
                        _sum2 = _mm_fmadd_ps(_we, _mm_set1_ps(r1[6]), _sum2);
                        _sum3 = _mm_fmadd_ps(_wf, _mm_set1_ps(r1[7]), _sum3);

                        r0 += dilation_w * 8;
                        r1 += dilation_w * 8;
                        kptr += 64;
                    }
                }
                if (elempack == 4)
                {
                    const float* r1 = r0 + N;
                    const float* r2 = r0 + N * 2;
                    const float* r3 = r0 + N * 3;

                    for (int k = 0; k < kernel_w; k++)
                    {
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

                        _sum0 = _mm_fmadd_ps(_w0, _mm_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm_fmadd_ps(_w1, _mm_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm_fmadd_ps(_w2, _mm_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm_fmadd_ps(_w3, _mm_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm_fmadd_ps(_w4, _mm_set1_ps(r1[0]), _sum0);
                        _sum1 = _mm_fmadd_ps(_w5, _mm_set1_ps(r1[1]), _sum1);
                        _sum2 = _mm_fmadd_ps(_w6, _mm_set1_ps(r1[2]), _sum2);
                        _sum3 = _mm_fmadd_ps(_w7, _mm_set1_ps(r1[3]), _sum3);
                        _sum0 = _mm_fmadd_ps(_w8, _mm_set1_ps(r2[0]), _sum0);
                        _sum1 = _mm_fmadd_ps(_w9, _mm_set1_ps(r2[1]), _sum1);
                        _sum2 = _mm_fmadd_ps(_wa, _mm_set1_ps(r2[2]), _sum2);
                        _sum3 = _mm_fmadd_ps(_wb, _mm_set1_ps(r2[3]), _sum3);
                        _sum0 = _mm_fmadd_ps(_wc, _mm_set1_ps(r3[0]), _sum0);
                        _sum1 = _mm_fmadd_ps(_wd, _mm_set1_ps(r3[1]), _sum1);
                        _sum2 = _mm_fmadd_ps(_we, _mm_set1_ps(r3[2]), _sum2);
                        _sum3 = _mm_fmadd_ps(_wf, _mm_set1_ps(r3[3]), _sum3);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        r2 += dilation_w * 4;
                        r3 += dilation_w * 4;
                        kptr += 64;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
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

                        _sum0 = _mm_fmadd_ps(_w0, _mm_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm_fmadd_ps(_w1, _mm_set1_ps(r0[N]), _sum1);
                        _sum2 = _mm_fmadd_ps(_w2, _mm_set1_ps(r0[N * 2]), _sum2);
                        _sum3 = _mm_fmadd_ps(_w3, _mm_set1_ps(r0[N * 3]), _sum3);
                        _sum0 = _mm_fmadd_ps(_w4, _mm_set1_ps(r0[N * 4]), _sum0);
                        _sum1 = _mm_fmadd_ps(_w5, _mm_set1_ps(r0[N * 5]), _sum1);
                        _sum2 = _mm_fmadd_ps(_w6, _mm_set1_ps(r0[N * 6]), _sum2);
                        _sum3 = _mm_fmadd_ps(_w7, _mm_set1_ps(r0[N * 7]), _sum3);
                        _sum0 = _mm_fmadd_ps(_w8, _mm_set1_ps(r0[N * 8]), _sum0);
                        _sum1 = _mm_fmadd_ps(_w9, _mm_set1_ps(r0[N * 9]), _sum1);
                        _sum2 = _mm_fmadd_ps(_wa, _mm_set1_ps(r0[N * 10]), _sum2);
                        _sum3 = _mm_fmadd_ps(_wb, _mm_set1_ps(r0[N * 11]), _sum3);
                        _sum0 = _mm_fmadd_ps(_wc, _mm_set1_ps(r0[N * 12]), _sum0);
                        _sum1 = _mm_fmadd_ps(_wd, _mm_set1_ps(r0[N * 13]), _sum1);
                        _sum2 = _mm_fmadd_ps(_we, _mm_set1_ps(r0[N * 14]), _sum2);
                        _sum3 = _mm_fmadd_ps(_wf, _mm_set1_ps(r0[N * 15]), _sum3);

                        r0 += dilation_w;
                        kptr += 64;
                    }
                }
            }
#endif // __AVX512F__
            for (; q + 7 < inh; q += 8)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 8)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = _mm_load_ps(kptr);
                        __m128 _w1 = _mm_load_ps(kptr + 4);
                        __m128 _w2 = _mm_load_ps(kptr + 8);
                        __m128 _w3 = _mm_load_ps(kptr + 12);
                        __m128 _w4 = _mm_load_ps(kptr + 16);
                        __m128 _w5 = _mm_load_ps(kptr + 20);
                        __m128 _w6 = _mm_load_ps(kptr + 24);
                        __m128 _w7 = _mm_load_ps(kptr + 28);

                        _sum0 = _mm_comp_fmadd_ps(_w0, _mm_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_w1, _mm_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_w2, _mm_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_w3, _mm_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm_comp_fmadd_ps(_w4, _mm_set1_ps(r0[4]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_w5, _mm_set1_ps(r0[5]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_w6, _mm_set1_ps(r0[6]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_w7, _mm_set1_ps(r0[7]), _sum3);

                        r0 += dilation_w * 8;
                        kptr += 32;
                    }
                }
                if (elempack == 4)
                {
                    const float* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = _mm_load_ps(kptr);
                        __m128 _w1 = _mm_load_ps(kptr + 4);
                        __m128 _w2 = _mm_load_ps(kptr + 8);
                        __m128 _w3 = _mm_load_ps(kptr + 12);
                        __m128 _w4 = _mm_load_ps(kptr + 16);
                        __m128 _w5 = _mm_load_ps(kptr + 20);
                        __m128 _w6 = _mm_load_ps(kptr + 24);
                        __m128 _w7 = _mm_load_ps(kptr + 28);

                        _sum0 = _mm_comp_fmadd_ps(_w0, _mm_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_w1, _mm_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_w2, _mm_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_w3, _mm_set1_ps(r0[3]), _sum3);
                        _sum0 = _mm_comp_fmadd_ps(_w4, _mm_set1_ps(r1[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_w5, _mm_set1_ps(r1[1]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_w6, _mm_set1_ps(r1[2]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_w7, _mm_set1_ps(r1[3]), _sum3);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 32;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = _mm_load_ps(kptr);
                        __m128 _w1 = _mm_load_ps(kptr + 4);
                        __m128 _w2 = _mm_load_ps(kptr + 8);
                        __m128 _w3 = _mm_load_ps(kptr + 12);
                        __m128 _w4 = _mm_load_ps(kptr + 16);
                        __m128 _w5 = _mm_load_ps(kptr + 20);
                        __m128 _w6 = _mm_load_ps(kptr + 24);
                        __m128 _w7 = _mm_load_ps(kptr + 28);

                        _sum0 = _mm_comp_fmadd_ps(_w0, _mm_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_w1, _mm_set1_ps(r0[N]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_w2, _mm_set1_ps(r0[N * 2]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_w3, _mm_set1_ps(r0[N * 3]), _sum3);
                        _sum0 = _mm_comp_fmadd_ps(_w4, _mm_set1_ps(r0[N * 4]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_w5, _mm_set1_ps(r0[N * 5]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_w6, _mm_set1_ps(r0[N * 6]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_w7, _mm_set1_ps(r0[N * 7]), _sum3);

                        r0 += dilation_w;
                        kptr += 32;
                    }
                }
            }
#endif // __AVX__
            for (; q + 3 < inh; q += 4)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = _mm_load_ps(kptr);
                        __m128 _w1 = _mm_load_ps(kptr + 4);
                        __m128 _w2 = _mm_load_ps(kptr + 8);
                        __m128 _w3 = _mm_load_ps(kptr + 12);

                        _sum0 = _mm_comp_fmadd_ps(_w0, _mm_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_w1, _mm_set1_ps(r0[1]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_w2, _mm_set1_ps(r0[2]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_w3, _mm_set1_ps(r0[3]), _sum3);

                        r0 += dilation_w * 4;
                        kptr += 16;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = _mm_load_ps(kptr);
                        __m128 _w1 = _mm_load_ps(kptr + 4);
                        __m128 _w2 = _mm_load_ps(kptr + 8);
                        __m128 _w3 = _mm_load_ps(kptr + 12);

                        _sum0 = _mm_comp_fmadd_ps(_w0, _mm_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_w1, _mm_set1_ps(r0[N]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_w2, _mm_set1_ps(r0[N * 2]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_w3, _mm_set1_ps(r0[N * 3]), _sum3);

                        r0 += dilation_w;
                        kptr += 16;
                    }
                }
            }
            for (; q + 1 < inh; q += 2)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _w0 = _mm_load_ps(kptr);
                        __m128 _w1 = _mm_load_ps(kptr + 4);

                        _sum0 = _mm_comp_fmadd_ps(_w0, _mm_set1_ps(r0[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_w1, _mm_set1_ps(r0[N]), _sum1);

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _val = _mm_set1_ps(r0[0]);
                        __m128 _w = _mm_load_ps(kptr);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w, _sum0);

                        r0 += dilation_w;
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
    remain_outh_start += nn_outh * 4;
    nn_outh = (outh - remain_outh_start) / 2;
#else // __SSE2__
    nn_outh = (outh - remain_outh_start) / 2;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __SSE2__
    for (int pp = 0; pp < nn_outh; pp++)
    {
        const int p = remain_outh_start + pp * 2;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;

        float* outptr0 = top_blob.row(p);
        float* outptr1 = top_blob.row(p + 1);

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
            for (; q + 15 < inh; q += 16)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 16)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _r0 = _mm512_load_ps(r0);
                        __m512 _w0 = _mm512_load_ps(kptr);
                        __m512 _w1 = _mm512_load_ps(kptr + 16);
                        _sum0_avx512 = _mm512_fmadd_ps(_r0, _w0, _sum0_avx512);
                        _sum1_avx512 = _mm512_fmadd_ps(_r0, _w1, _sum1_avx512);

                        r0 += dilation_w * 16;
                        kptr += 32;
                    }
                }
                if (elempack == 8)
                {
                    const float* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0)), _mm256_load_ps(r1), 1);
                        __m512 _w0 = _mm512_load_ps(kptr);
                        __m512 _w1 = _mm512_load_ps(kptr + 16);
                        _sum0_avx512 = _mm512_fmadd_ps(_r0, _w0, _sum0_avx512);
                        _sum1_avx512 = _mm512_fmadd_ps(_r0, _w1, _sum1_avx512);

                        r0 += dilation_w * 8;
                        r1 += dilation_w * 8;
                        kptr += 32;
                    }
                }
                if (elempack == 4)
                {
                    const float* r1 = r0 + N;
                    const float* r2 = r0 + N * 2;
                    const float* r3 = r0 + N * 3;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2)), _mm_load_ps(r3), 1), 1);
                        __m512 _w0 = _mm512_load_ps(kptr);
                        __m512 _w1 = _mm512_load_ps(kptr + 16);
                        _sum0_avx512 = _mm512_fmadd_ps(_r0, _w0, _sum0_avx512);
                        _sum1_avx512 = _mm512_fmadd_ps(_r0, _w1, _sum1_avx512);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        r2 += dilation_w * 4;
                        r3 += dilation_w * 4;
                        kptr += 32;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _r0 = _mm512_set_ps(r0[N * 15], r0[N * 14], r0[N * 13], r0[N * 12], r0[N * 11], r0[N * 10], r0[N * 9], r0[N * 8], r0[N * 7], r0[N * 6], r0[N * 5], r0[N * 4], r0[N * 3], r0[N * 2], r0[N], r0[0]);
                        __m512 _w0 = _mm512_load_ps(kptr);
                        __m512 _w1 = _mm512_load_ps(kptr + 16);
                        _sum0_avx512 = _mm512_fmadd_ps(_r0, _w0, _sum0_avx512);
                        _sum1_avx512 = _mm512_fmadd_ps(_r0, _w1, _sum1_avx512);

                        r0 += dilation_w;
                        kptr += 32;
                    }
                }
            }
            sum0 += _mm512_comp_reduce_add_ps(_sum0_avx512);
            sum1 += _mm512_comp_reduce_add_ps(_sum1_avx512);
#endif // __AVX512F__
            __m256 _sum0_avx = _mm256_setzero_ps();
            __m256 _sum1_avx = _mm256_setzero_ps();
            for (; q + 7 < inh; q += 8)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 8)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _r0 = _mm256_load_ps(r0);
                        __m256 _w0 = _mm256_load_ps(kptr);
                        __m256 _w1 = _mm256_load_ps(kptr + 8);
                        _sum0_avx = _mm256_comp_fmadd_ps(_r0, _w0, _sum0_avx);
                        _sum1_avx = _mm256_comp_fmadd_ps(_r0, _w1, _sum1_avx);

                        r0 += dilation_w * 8;
                        kptr += 16;
                    }
                }
                if (elempack == 4)
                {
                    const float* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1);
                        __m256 _w0 = _mm256_load_ps(kptr);
                        __m256 _w1 = _mm256_load_ps(kptr + 8);
                        _sum0_avx = _mm256_comp_fmadd_ps(_r0, _w0, _sum0_avx);
                        _sum1_avx = _mm256_comp_fmadd_ps(_r0, _w1, _sum1_avx);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 16;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _r0 = _mm256_set_ps(r0[N * 7], r0[N * 6], r0[N * 5], r0[N * 4], r0[N * 3], r0[N * 2], r0[N], r0[0]);
                        __m256 _w0 = _mm256_load_ps(kptr);
                        __m256 _w1 = _mm256_load_ps(kptr + 8);
                        _sum0_avx = _mm256_comp_fmadd_ps(_r0, _w0, _sum0_avx);
                        _sum1_avx = _mm256_comp_fmadd_ps(_r0, _w1, _sum1_avx);

                        r0 += dilation_w;
                        kptr += 16;
                    }
                }
            }
            sum0 += _mm256_reduce_add_ps(_sum0_avx);
            sum1 += _mm256_reduce_add_ps(_sum1_avx);
#endif // __AVX__
            __m128 _sum0 = _mm_setzero_ps();
            __m128 _sum1 = _mm_setzero_ps();
            for (; q + 3 < inh; q += 4)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _r0 = _mm_load_ps(r0);
                        __m128 _w0 = _mm_load_ps(kptr);
                        __m128 _w1 = _mm_load_ps(kptr + 4);
                        _sum0 = _mm_comp_fmadd_ps(_r0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_r0, _w1, _sum1);

                        r0 += dilation_w * 4;
                        kptr += 8;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _r0 = _mm_set_ps(r0[N * 3], r0[N * 2], r0[N], r0[0]);
                        __m128 _w0 = _mm_load_ps(kptr);
                        __m128 _w1 = _mm_load_ps(kptr + 4);
                        _sum0 = _mm_comp_fmadd_ps(_r0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_r0, _w1, _sum1);

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
            }
            sum0 += _mm_reduce_add_ps(_sum0);
            sum1 += _mm_reduce_add_ps(_sum1);
#endif // __SSE2__
            for (; q + 1 < inh; q += 2)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        sum0 += r0[0] * kptr[0];
                        sum1 += r0[0] * kptr[1];
                        sum0 += r0[N] * kptr[2];
                        sum1 += r0[N] * kptr[3];

                        r0 += dilation_w;
                        kptr += 4;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        float val = r0[0];
                        sum0 += val * kptr[0];
                        sum1 += val * kptr[1];

                        r0 += dilation_w;
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
    remain_outh_start += nn_outh * 2;
    for (int p = remain_outh_start; p < outh; p++)
    {
        float* outptr = top_blob.row(p);

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
            for (; q + 15 < inh; q += 16)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 16)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _r0 = _mm512_load_ps(r0);
                        __m512 _w = _mm512_load_ps(kptr);
                        _sum_avx512 = _mm512_fmadd_ps(_r0, _w, _sum_avx512);

                        r0 += dilation_w * 16;
                        kptr += 16;
                    }
                }
                if (elempack == 8)
                {
                    const float* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_load_ps(r0)), _mm256_load_ps(r1), 1);
                        __m512 _w = _mm512_load_ps(kptr);
                        _sum_avx512 = _mm512_fmadd_ps(_r0, _w, _sum_avx512);

                        r0 += dilation_w * 8;
                        r1 += dilation_w * 8;
                        kptr += 16;
                    }
                }
                if (elempack == 4)
                {
                    const float* r1 = r0 + N;
                    const float* r2 = r0 + N * 2;
                    const float* r3 = r0 + N * 3;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _r0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1)), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r2)), _mm_load_ps(r3), 1), 1);
                        __m512 _w = _mm512_load_ps(kptr);
                        _sum_avx512 = _mm512_fmadd_ps(_r0, _w, _sum_avx512);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        r2 += dilation_w * 4;
                        r3 += dilation_w * 4;
                        kptr += 16;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m512 _r0 = _mm512_set_ps(r0[N * 15], r0[N * 14], r0[N * 13], r0[N * 12], r0[N * 11], r0[N * 10], r0[N * 9], r0[N * 8], r0[N * 7], r0[N * 6], r0[N * 5], r0[N * 4], r0[N * 3], r0[N * 2], r0[N], r0[0]);
                        __m512 _w = _mm512_load_ps(kptr);
                        _sum_avx512 = _mm512_fmadd_ps(_r0, _w, _sum_avx512);

                        r0 += dilation_w;
                        kptr += 16;
                    }
                }
            }
            sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
            __m256 _sum_avx = _mm256_setzero_ps();
            for (; q + 7 < inh; q += 8)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 8)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _r0 = _mm256_load_ps(r0);
                        __m256 _w = _mm256_load_ps(kptr);
                        _sum_avx = _mm256_comp_fmadd_ps(_r0, _w, _sum_avx);

                        r0 += dilation_w * 8;
                        kptr += 8;
                    }
                }
                if (elempack == 4)
                {
                    const float* r1 = r0 + N;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_load_ps(r0)), _mm_load_ps(r1), 1);
                        __m256 _w = _mm256_load_ps(kptr);
                        _sum_avx = _mm256_comp_fmadd_ps(_r0, _w, _sum_avx);

                        r0 += dilation_w * 4;
                        r1 += dilation_w * 4;
                        kptr += 8;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m256 _r0 = _mm256_set_ps(r0[N * 7], r0[N * 6], r0[N * 5], r0[N * 4], r0[N * 3], r0[N * 2], r0[N], r0[0]);
                        __m256 _w = _mm256_load_ps(kptr);
                        _sum_avx = _mm256_comp_fmadd_ps(_r0, _w, _sum_avx);

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
            }
            sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
            __m128 _sum = _mm_setzero_ps();
            for (; q + 3 < inh; q += 4)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _r0 = _mm_load_ps(r0);
                        __m128 _w = _mm_load_ps(kptr);
                        _sum = _mm_comp_fmadd_ps(_r0, _w, _sum);

                        r0 += dilation_w * 4;
                        kptr += 4;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __m128 _r0 = _mm_set_ps(r0[N * 3], r0[N * 2], r0[N], r0[0]);
                        __m128 _w = _mm_load_ps(kptr);
                        _sum = _mm_comp_fmadd_ps(_r0, _w, _sum);

                        r0 += dilation_w;
                        kptr += 4;
                    }
                }
            }
            sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
            for (; q + 1 < inh; q += 2)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        sum += r0[0] * kptr[0];
                        sum += r0[N] * kptr[1];

                        r0 += dilation_w;
                        kptr += 2;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        float val = r0[0];
                        sum += val * kptr[0];

                        r0 += dilation_w;
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
