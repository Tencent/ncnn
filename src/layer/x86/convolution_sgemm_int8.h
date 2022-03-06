// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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
void im2col_sgemm_int8_sse_avx512vnni(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__
void im2col_sgemm_int8_sse_avxvnni(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
void im2col_sgemm_int8_sse_avx2(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__
void im2col_sgemm_int8_sse_xop(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt);
#endif
#endif

static void im2col_sgemm_int8_sse(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
#if !(__AVX512VNNI__ || __AVXVNNI__ || __AVX2__ || __XOP__)
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        im2col_sgemm_int8_sse_avx512vnni(bottom_im2col, top_blob, kernel, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        im2col_sgemm_int8_sse_avxvnni(bottom_im2col, top_blob, kernel, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        im2col_sgemm_int8_sse_avx2(bottom_im2col, top_blob, kernel, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__
    if (ncnn::cpu_support_x86_xop())
    {
        im2col_sgemm_int8_sse_xop(bottom_im2col, top_blob, kernel, opt);
        return;
    }
#endif
#endif

    // Mat bottom_im2col(size, maxk, inch, 8u, 8, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    // permute
    Mat tmp;
#if __SSE2__
    if (inch >= 4)
    {
#if __AVX2__
        if (size >= 4)
            tmp.create(4 * maxk, inch / 4 + inch % 4, size / 4 + (size % 4) / 2 + size % 2, 4u, 4, opt.workspace_allocator);
        else if (size >= 2)
            tmp.create(2 * maxk, inch / 4 + inch % 4, size / 2 + size % 2, 4u, 4, opt.workspace_allocator);
        else
            tmp.create(maxk, inch / 4 + inch % 4, size, 4u, 4, opt.workspace_allocator);
#else
        if (size >= 2)
            tmp.create(2 * maxk, inch / 4 + inch % 4, size / 2 + size % 2, 4u, 4, opt.workspace_allocator);
        else
            tmp.create(maxk, inch / 4 + inch % 4, size, 4u, 4, opt.workspace_allocator);
#endif
    }
    else
    {
#if __AVX2__
        if (size >= 4)
            tmp.create(4 * maxk, inch, size / 4 + (size % 4) / 2 + size % 2, 1u, 1, opt.workspace_allocator);
        else if (size >= 2)
            tmp.create(2 * maxk, inch, size / 2 + size % 2, 1u, 1, opt.workspace_allocator);
        else
            tmp.create(maxk, inch, size, 1u, 1, opt.workspace_allocator);
#else
        if (size >= 2)
            tmp.create(2 * maxk, inch, size / 2 + size % 2, 1u, 1, opt.workspace_allocator);
        else
            tmp.create(maxk, inch, size, 1u, 1, opt.workspace_allocator);
#endif
    }
    {
#if __AVX2__
        int remain_size_start = 0;
        int nn_size = size >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            signed char* tmpptr = tmp.channel(i / 4);

            int q = 0;
            for (; q + 3 < inch; q += 4)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img0[1];
                    tmpptr[5] = img1[1];
                    tmpptr[6] = img2[1];
                    tmpptr[7] = img3[1];
                    tmpptr[8] = img0[2];
                    tmpptr[9] = img1[2];
                    tmpptr[10] = img2[2];
                    tmpptr[11] = img3[2];
                    tmpptr[12] = img0[3];
                    tmpptr[13] = img1[3];
                    tmpptr[14] = img2[3];
                    tmpptr[15] = img3[3];
                    tmpptr += 16;

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                }
            }
            for (; q < inch; q++)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img0[1];
                    tmpptr[2] = img0[2];
                    tmpptr[3] = img0[3];

                    tmpptr += 4;

                    img0 += size;
                }
            }
        }

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;
#else
        int remain_size_start = 0;
        int nn_size = (size - remain_size_start) >> 1;
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

#if __AVX2__
            signed char* tmpptr = tmp.channel(i / 4 + (i % 4) / 2);
#else
            signed char* tmpptr = tmp.channel(i / 2);
#endif

            int q = 0;
            for (; q + 3 < inch; q += 4)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img0[1];
                    tmpptr[5] = img1[1];
                    tmpptr[6] = img2[1];
                    tmpptr[7] = img3[1];
                    tmpptr += 8;

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                }
            }
            for (; q < inch; q++)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img0[1];

                    tmpptr += 2;

                    img0 += size;
                }
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
#if __AVX2__
            signed char* tmpptr = tmp.channel(i / 4 + (i % 4) / 2 + i % 2);
#else
            signed char* tmpptr = tmp.channel(i / 2 + i % 2);
#endif

            int q = 0;
            for (; q + 3 < inch; q += 4)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr += 4;

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                }
            }
            for (; q < inch; q++)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];

                    tmpptr += 1;

                    img0 += size;
                }
            }
        }
    }
#else // __SSE2__
    tmp.create(maxk, inch, size, 1u, 1, opt.workspace_allocator);
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++)
        {
            signed char* tmpptr = tmp.channel(i);

            int q = 0;
            for (; q < inch; q++)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];

                    tmpptr += 1;

                    img0 += size;
                }
            }
        }
    }
#endif // __SSE2__

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __SSE2__
    nn_outch = outch >> 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);
        int* outptr2 = top_blob.channel(p + 2);
        int* outptr3 = top_blob.channel(p + 3);

        int i = 0;
#if __AVX2__
        for (; i + 3 < size; i += 4)
        {
            const signed char* tmpptr = tmp.channel(i / 4);
            const signed char* kptr0 = kernel.channel(p / 4);

            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            __m256i _sum00_12 = _mm256_setzero_si256();
            __m256i _sum20_32 = _mm256_setzero_si256();

            if (nn4 > 0)
            {
#if __AVXVNNI__ || __AVX512VNNI__
                __m256i _sum10_02 = _mm256_setzero_si256();
                __m256i _sum30_22 = _mm256_setzero_si256();
#else
                __m256i _sum10_02 = _mm256_setzero_si256();
                __m256i _sum01_13 = _mm256_setzero_si256();
                __m256i _sum11_03 = _mm256_setzero_si256();
                __m256i _sum30_22 = _mm256_setzero_si256();
                __m256i _sum21_33 = _mm256_setzero_si256();
                __m256i _sum31_23 = _mm256_setzero_si256();
#endif

                int j = 0;
                for (; j < nn4; j++)
                {
                    __m128i _val0123 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m256i _val0123_16 = _mm256_cvtepi8_epi16(_val0123);

                    __m256i _val01_16 = _mm256_permute4x64_epi64(_val0123_16, _MM_SHUFFLE(1, 1, 0, 0));
                    __m256i _val23_16 = _mm256_permute4x64_epi64(_val0123_16, _MM_SHUFFLE(3, 3, 2, 2));

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m256i _w01_16 = _mm256_cvtepi8_epi16(_w01);

                    __m256i _val10_16 = _mm256_permute4x64_epi64(_val01_16, 78);
                    __m256i _val32_16 = _mm256_permute4x64_epi64(_val23_16, 78);

#if __AVXVNNI__ || __AVX512VNNI__
                    _sum00_12 = _mm256_dpwssd_epi32(_sum00_12, _val01_16, _w01_16);
                    _sum10_02 = _mm256_dpwssd_epi32(_sum10_02, _val10_16, _w01_16);
                    _sum20_32 = _mm256_dpwssd_epi32(_sum20_32, _val23_16, _w01_16);
                    _sum30_22 = _mm256_dpwssd_epi32(_sum30_22, _val32_16, _w01_16);
#else
                    __m256i _sl00_11 = _mm256_mullo_epi16(_val01_16, _w01_16);
                    __m256i _sh00_11 = _mm256_mulhi_epi16(_val01_16, _w01_16);
                    __m256i _sl10_01 = _mm256_mullo_epi16(_val10_16, _w01_16);
                    __m256i _sh10_01 = _mm256_mulhi_epi16(_val10_16, _w01_16);
                    __m256i _sl20_31 = _mm256_mullo_epi16(_val23_16, _w01_16);
                    __m256i _sh20_31 = _mm256_mulhi_epi16(_val23_16, _w01_16);
                    __m256i _sl30_21 = _mm256_mullo_epi16(_val32_16, _w01_16);
                    __m256i _sh30_21 = _mm256_mulhi_epi16(_val32_16, _w01_16);

                    _sum00_12 = _mm256_add_epi32(_sum00_12, _mm256_unpacklo_epi16(_sl00_11, _sh00_11));
                    _sum10_02 = _mm256_add_epi32(_sum10_02, _mm256_unpacklo_epi16(_sl10_01, _sh10_01));
                    _sum01_13 = _mm256_add_epi32(_sum01_13, _mm256_unpackhi_epi16(_sl00_11, _sh00_11));
                    _sum11_03 = _mm256_add_epi32(_sum11_03, _mm256_unpackhi_epi16(_sl10_01, _sh10_01));
                    _sum20_32 = _mm256_add_epi32(_sum20_32, _mm256_unpacklo_epi16(_sl20_31, _sh20_31));
                    _sum30_22 = _mm256_add_epi32(_sum30_22, _mm256_unpacklo_epi16(_sl30_21, _sh30_21));
                    _sum21_33 = _mm256_add_epi32(_sum21_33, _mm256_unpackhi_epi16(_sl20_31, _sh20_31));
                    _sum31_23 = _mm256_add_epi32(_sum31_23, _mm256_unpackhi_epi16(_sl30_21, _sh30_21));
#endif

                    tmpptr += 16;
                    kptr0 += 16;
                }

#if __AVXVNNI__ || __AVX512VNNI__
                _sum00_12 = _mm256_hadd_epi32(_sum00_12, _sum10_02);
                _sum20_32 = _mm256_hadd_epi32(_sum20_32, _sum30_22);

                _sum00_12 = _mm256_permute4x64_epi64(_sum00_12, _MM_SHUFFLE(2, 1, 3, 0));
                _sum20_32 = _mm256_permute4x64_epi64(_sum20_32, _MM_SHUFFLE(2, 1, 3, 0));
#else
                // transpose 4x8
                {
                    __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm256_unpacklo_epi32(_sum00_12, _sum10_02);
                    _tmp1 = _mm256_unpacklo_epi32(_sum01_13, _sum11_03);
                    _tmp2 = _mm256_unpackhi_epi32(_sum00_12, _sum10_02);
                    _tmp3 = _mm256_unpackhi_epi32(_sum01_13, _sum11_03);
                    _sum00_12 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                    _sum10_02 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                    _sum01_13 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                    _sum11_03 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                }
                {
                    __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm256_unpacklo_epi32(_sum20_32, _sum30_22);
                    _tmp1 = _mm256_unpacklo_epi32(_sum21_33, _sum31_23);
                    _tmp2 = _mm256_unpackhi_epi32(_sum20_32, _sum30_22);
                    _tmp3 = _mm256_unpackhi_epi32(_sum21_33, _sum31_23);
                    _sum20_32 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                    _sum30_22 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                    _sum21_33 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                    _sum31_23 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                }

                _sum00_12 = _mm256_add_epi32(_sum00_12, _sum10_02);
                _sum01_13 = _mm256_add_epi32(_sum01_13, _sum11_03);
                _sum00_12 = _mm256_add_epi32(_sum00_12, _sum01_13);

                _sum20_32 = _mm256_add_epi32(_sum20_32, _sum30_22);
                _sum21_33 = _mm256_add_epi32(_sum21_33, _sum31_23);
                _sum20_32 = _mm256_add_epi32(_sum20_32, _sum21_33);

                __m256i _perm_mask = _mm256_set_epi32(6, 4, 3, 1, 7, 5, 2, 0);
                _sum00_12 = _mm256_permutevar8x32_epi32(_sum00_12, _perm_mask);
                _sum20_32 = _mm256_permutevar8x32_epi32(_sum20_32, _perm_mask);
#endif
            }

            __m128i _sum00 = _mm256_extracti128_si256(_sum00_12, 0);
            __m128i _sum10 = _mm256_extracti128_si256(_sum00_12, 1);
            __m128i _sum20 = _mm256_extracti128_si256(_sum20_32, 0);
            __m128i _sum30 = _mm256_extracti128_si256(_sum20_32, 1);

            int j = 0;
            for (; j < nn1; j++)
            {
                __m128i _val0123 = _mm_loadl_epi64((const __m128i*)tmpptr);
#if __SSE4_1__
                _val0123 = _mm_cvtepi8_epi16(_val0123);
#else
                __m128i _extval0123 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val0123);
                _val0123 = _mm_unpacklo_epi8(_val0123, _extval0123);
#endif

                __m128i _val01 = _mm_shufflelo_epi16(_val0123, _MM_SHUFFLE(1, 1, 0, 0));

                _val01 = _mm_shuffle_epi32(_val01, _MM_SHUFFLE(1, 1, 0, 0));

                __m128i _val23 = _mm_shufflelo_epi16(_val0123, _MM_SHUFFLE(3, 3, 2, 2));

                _val23 = _mm_shuffle_epi32(_val23, _MM_SHUFFLE(1, 1, 0, 0));

                __m128i _w0123 = _mm_loadl_epi64((const __m128i*)kptr0);
#if __SSE4_1__
                _w0123 = _mm_cvtepi8_epi16(_w0123);
#else
                __m128i _extw0123 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w0123);
                _w0123 = _mm_unpacklo_epi8(_w0123, _extw0123);
#endif

                _w0123 = _mm_shuffle_epi32(_w0123, _MM_SHUFFLE(1, 0, 1, 0));

                __m128i _sl00 = _mm_mullo_epi16(_val01, _w0123);
                __m128i _sh00 = _mm_mulhi_epi16(_val01, _w0123);
                __m128i _sl10 = _mm_mullo_epi16(_val23, _w0123);
                __m128i _sh10 = _mm_mulhi_epi16(_val23, _w0123);

                _sum00 = _mm_add_epi32(_sum00, _mm_unpacklo_epi16(_sl00, _sh00));
                _sum10 = _mm_add_epi32(_sum10, _mm_unpackhi_epi16(_sl00, _sh00));
                _sum20 = _mm_add_epi32(_sum20, _mm_unpacklo_epi16(_sl10, _sh10));
                _sum30 = _mm_add_epi32(_sum30, _mm_unpackhi_epi16(_sl10, _sh10));

                tmpptr += 4;
                kptr0 += 4;
            }

            // transpose 4x4
            {
                __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                _tmp0 = _mm_unpacklo_epi32(_sum00, _sum10);
                _tmp1 = _mm_unpacklo_epi32(_sum20, _sum30);
                _tmp2 = _mm_unpackhi_epi32(_sum00, _sum10);
                _tmp3 = _mm_unpackhi_epi32(_sum20, _sum30);
                _sum00 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                _sum10 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                _sum20 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                _sum30 = _mm_unpackhi_epi64(_tmp2, _tmp3);
            }

            _mm_storeu_si128((__m128i*)outptr0, _sum00);
            _mm_storeu_si128((__m128i*)outptr1, _sum10);
            _mm_storeu_si128((__m128i*)outptr2, _sum20);
            _mm_storeu_si128((__m128i*)outptr3, _sum30);
            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
        }
#endif
        for (; i + 1 < size; i += 2)
        {
#if __AVX2__
            const signed char* tmpptr = tmp.channel(i / 4 + (i % 4) / 2);
#else
            const signed char* tmpptr = tmp.channel(i / 2);
#endif
            const signed char* kptr0 = kernel.channel(p / 4);

            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

#if __AVX2__
            __m256i _sum00_12 = _mm256_setzero_si256();
#else
            __m128i _sum00 = _mm_setzero_si128();
            __m128i _sum10 = _mm_setzero_si128();
#endif

            if (nn4 > 0)
            {
#if __AVX2__
#if __AVXVNNI__ || __AVX512VNNI__
                __m256i _sum10_02 = _mm256_setzero_si256();
#else
                __m256i _sum10_02 = _mm256_setzero_si256();
                __m256i _sum01_13 = _mm256_setzero_si256();
                __m256i _sum11_03 = _mm256_setzero_si256();
#endif
#else
#if __XOP__
                __m128i _sum01 = _mm_setzero_si128();
                __m128i _sum11 = _mm_setzero_si128();
#else
                __m128i _sum01 = _mm_setzero_si128();
                __m128i _sum02 = _mm_setzero_si128();
                __m128i _sum03 = _mm_setzero_si128();
                __m128i _sum11 = _mm_setzero_si128();
                __m128i _sum12 = _mm_setzero_si128();
                __m128i _sum13 = _mm_setzero_si128();
#endif
#endif

                int j = 0;
                for (; j < nn4; j++)
                {
#if __AVX2__
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m256i _val01_16 = _mm256_cvtepi8_epi16(_val01);

                    _val01_16 = _mm256_permute4x64_epi64(_val01_16, _MM_SHUFFLE(1, 1, 0, 0));

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m256i _w01_16 = _mm256_cvtepi8_epi16(_w01);

                    __m256i _val10_16 = _mm256_permute4x64_epi64(_val01_16, 78);

#if __AVXVNNI__ || __AVX512VNNI__
                    _sum00_12 = _mm256_dpwssd_epi32(_sum00_12, _val01_16, _w01_16);
                    _sum10_02 = _mm256_dpwssd_epi32(_sum10_02, _val10_16, _w01_16);
#else
                    __m256i _sl00_11 = _mm256_mullo_epi16(_val01_16, _w01_16);
                    __m256i _sh00_11 = _mm256_mulhi_epi16(_val01_16, _w01_16);
                    __m256i _sl10_01 = _mm256_mullo_epi16(_val10_16, _w01_16);
                    __m256i _sh10_01 = _mm256_mulhi_epi16(_val10_16, _w01_16);

                    _sum00_12 = _mm256_add_epi32(_sum00_12, _mm256_unpacklo_epi16(_sl00_11, _sh00_11));
                    _sum10_02 = _mm256_add_epi32(_sum10_02, _mm256_unpacklo_epi16(_sl10_01, _sh10_01));
                    _sum01_13 = _mm256_add_epi32(_sum01_13, _mm256_unpackhi_epi16(_sl00_11, _sh00_11));
                    _sum11_03 = _mm256_add_epi32(_sum11_03, _mm256_unpackhi_epi16(_sl10_01, _sh10_01));
#endif
#else
                    __m128i _val01 = _mm_loadl_epi64((const __m128i*)tmpptr);
#if __SSE4_1__
                    _val01 = _mm_cvtepi8_epi16(_val01);
#else
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    _val01 = _mm_unpacklo_epi8(_val01, _extval01);
#endif

                    __m128i _val0 = _mm_shuffle_epi32(_val01, _MM_SHUFFLE(1, 0, 1, 0));
                    __m128i _val1 = _mm_shuffle_epi32(_val01, _MM_SHUFFLE(3, 2, 3, 2));

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                    __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);

#if __XOP__
                    _sum00 = _mm_maddd_epi16(_val0, _w0, _sum00);
                    _sum01 = _mm_maddd_epi16(_val0, _w1, _sum01);
                    _sum10 = _mm_maddd_epi16(_val1, _w0, _sum10);
                    _sum11 = _mm_maddd_epi16(_val1, _w1, _sum11);
#else
                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                    __m128i _sl01 = _mm_mullo_epi16(_val0, _w1);
                    __m128i _sh01 = _mm_mulhi_epi16(_val0, _w1);
                    __m128i _sl10 = _mm_mullo_epi16(_val1, _w0);
                    __m128i _sh10 = _mm_mulhi_epi16(_val1, _w0);
                    __m128i _sl11 = _mm_mullo_epi16(_val1, _w1);
                    __m128i _sh11 = _mm_mulhi_epi16(_val1, _w1);

                    _sum00 = _mm_add_epi32(_sum00, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum01 = _mm_add_epi32(_sum01, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum02 = _mm_add_epi32(_sum02, _mm_unpacklo_epi16(_sl01, _sh01));
                    _sum03 = _mm_add_epi32(_sum03, _mm_unpackhi_epi16(_sl01, _sh01));
                    _sum10 = _mm_add_epi32(_sum10, _mm_unpacklo_epi16(_sl10, _sh10));
                    _sum11 = _mm_add_epi32(_sum11, _mm_unpackhi_epi16(_sl10, _sh10));
                    _sum12 = _mm_add_epi32(_sum12, _mm_unpacklo_epi16(_sl11, _sh11));
                    _sum13 = _mm_add_epi32(_sum13, _mm_unpackhi_epi16(_sl11, _sh11));
#endif
#endif

                    tmpptr += 8;
                    kptr0 += 16;
                }

#if __AVX2__
#if __AVXVNNI__ || __AVX512VNNI__
                _sum00_12 = _mm256_hadd_epi32(_sum00_12, _sum10_02);

                _sum00_12 = _mm256_permute4x64_epi64(_sum00_12, _MM_SHUFFLE(2, 1, 3, 0));
#else
                // transpose 4x8
                {
                    __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm256_unpacklo_epi32(_sum00_12, _sum10_02);
                    _tmp1 = _mm256_unpacklo_epi32(_sum01_13, _sum11_03);
                    _tmp2 = _mm256_unpackhi_epi32(_sum00_12, _sum10_02);
                    _tmp3 = _mm256_unpackhi_epi32(_sum01_13, _sum11_03);
                    _sum00_12 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                    _sum10_02 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                    _sum01_13 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                    _sum11_03 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                }

                _sum00_12 = _mm256_add_epi32(_sum00_12, _sum10_02);
                _sum01_13 = _mm256_add_epi32(_sum01_13, _sum11_03);
                _sum00_12 = _mm256_add_epi32(_sum00_12, _sum01_13);

                __m256i _perm_mask = _mm256_set_epi32(6, 4, 3, 1, 7, 5, 2, 0);
                _sum00_12 = _mm256_permutevar8x32_epi32(_sum00_12, _perm_mask);
#endif
#else
#if __XOP__
                _sum00 = _mm_hadd_epi32(_sum00, _sum01);
                _sum10 = _mm_hadd_epi32(_sum10, _sum11);
#else
                // transpose 4x4
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm_unpacklo_epi32(_sum00, _sum01);
                    _tmp1 = _mm_unpacklo_epi32(_sum02, _sum03);
                    _tmp2 = _mm_unpackhi_epi32(_sum00, _sum01);
                    _tmp3 = _mm_unpackhi_epi32(_sum02, _sum03);
                    _sum00 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                    _sum01 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                    _sum02 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                    _sum03 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                }
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm_unpacklo_epi32(_sum10, _sum11);
                    _tmp1 = _mm_unpacklo_epi32(_sum12, _sum13);
                    _tmp2 = _mm_unpackhi_epi32(_sum10, _sum11);
                    _tmp3 = _mm_unpackhi_epi32(_sum12, _sum13);
                    _sum10 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                    _sum11 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                    _sum12 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                    _sum13 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                }

                _sum00 = _mm_add_epi32(_sum00, _sum01);
                _sum02 = _mm_add_epi32(_sum02, _sum03);
                _sum10 = _mm_add_epi32(_sum10, _sum11);
                _sum12 = _mm_add_epi32(_sum12, _sum13);

                _sum00 = _mm_add_epi32(_sum00, _sum02);
                _sum10 = _mm_add_epi32(_sum10, _sum12);
#endif
#endif
            }

#if __AVX2__
            __m128i _sum00 = _mm256_extracti128_si256(_sum00_12, 0);
            __m128i _sum10 = _mm256_extracti128_si256(_sum00_12, 1);
#endif

            int j = 0;
            for (; j < nn1; j++)
            {
                __m128i _val = _mm_set_epi16(tmpptr[1], tmpptr[1], tmpptr[1], tmpptr[1], tmpptr[0], tmpptr[0], tmpptr[0], tmpptr[0]);

                // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=99754
                // gcc incorrectly put 32bit to tail with _mm_loadu_si32  :(
                // 0 1 2 3 x x x x x x x x x x x x
                // x x x x x x x x x x x x 0 1 2 3
                // __m128i _w0123 = _mm_loadu_si32(kptr0);
                __m128i _w0123 = _mm_loadl_epi64((const __m128i*)kptr0);
#if __SSE4_1__
                _w0123 = _mm_cvtepi8_epi16(_w0123);
#else
                __m128i _extw0123 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w0123);
                _w0123 = _mm_unpacklo_epi8(_w0123, _extw0123);
#endif

                _w0123 = _mm_shuffle_epi32(_w0123, _MM_SHUFFLE(1, 0, 1, 0));

                __m128i _sl00 = _mm_mullo_epi16(_val, _w0123);
                __m128i _sh00 = _mm_mulhi_epi16(_val, _w0123);

                _sum00 = _mm_add_epi32(_sum00, _mm_unpacklo_epi16(_sl00, _sh00));
                _sum10 = _mm_add_epi32(_sum10, _mm_unpackhi_epi16(_sl00, _sh00));

                tmpptr += 2;
                kptr0 += 4;
            }

            int sum[8];
            _mm_storeu_si128((__m128i*)sum, _sum00);
            _mm_storeu_si128((__m128i*)(sum + 4), _sum10);

            outptr0[0] = sum[0];
            outptr1[0] = sum[1];
            outptr2[0] = sum[2];
            outptr3[0] = sum[3];
            outptr0[1] = sum[4];
            outptr1[1] = sum[5];
            outptr2[1] = sum[6];
            outptr3[1] = sum[7];
            outptr0 += 2;
            outptr1 += 2;
            outptr2 += 2;
            outptr3 += 2;
        }
        for (; i < size; i++)
        {
#if __AVX2__
            const signed char* tmpptr = tmp.channel(i / 4 + (i % 4) / 2 + i % 2);
#else
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
#endif
            const signed char* kptr0 = kernel.channel(p / 4);

            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            __m128i _sum0 = _mm_setzero_si128();

            if (nn4 > 0)
            {
                __m128i _sum1 = _mm_setzero_si128();
                __m128i _sum2 = _mm_setzero_si128();
                __m128i _sum3 = _mm_setzero_si128();

                int j = 0;
                for (; j < nn4; j++)
                {
                    __m128i _val01 = _mm_loadl_epi64((const __m128i*)tmpptr);
#if __SSE4_1__
                    __m128i _val0 = _mm_cvtepi8_epi16(_val01);
#else
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);
#endif

                    _val0 = _mm_shuffle_epi32(_val0, _MM_SHUFFLE(1, 0, 1, 0));

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                    __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);

                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                    __m128i _sl01 = _mm_mullo_epi16(_val0, _w1);
                    __m128i _sh01 = _mm_mulhi_epi16(_val0, _w1);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpacklo_epi16(_sl01, _sh01));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpackhi_epi16(_sl01, _sh01));

                    tmpptr += 4;
                    kptr0 += 16;
                }

                // transpose 4x4
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm_unpacklo_epi32(_sum0, _sum1);
                    _tmp1 = _mm_unpacklo_epi32(_sum2, _sum3);
                    _tmp2 = _mm_unpackhi_epi32(_sum0, _sum1);
                    _tmp3 = _mm_unpackhi_epi32(_sum2, _sum3);
                    _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                    _sum1 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                    _sum2 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                    _sum3 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                }

                _sum0 = _mm_add_epi32(_sum0, _sum1);
                _sum2 = _mm_add_epi32(_sum2, _sum3);
                _sum0 = _mm_add_epi32(_sum0, _sum2);
            }

            int j = 0;
            for (; j < nn1; j++)
            {
                __m128i _val = _mm_set1_epi16(tmpptr[0]);

                __m128i _w0123 = _mm_loadl_epi64((const __m128i*)kptr0);
#if __SSE4_1__
                _w0123 = _mm_cvtepi8_epi16(_w0123);
#else
                __m128i _extw0123 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w0123);
                _w0123 = _mm_unpacklo_epi8(_w0123, _extw0123);
#endif

                __m128i _sl00 = _mm_mullo_epi16(_val, _w0123);
                __m128i _sh00 = _mm_mulhi_epi16(_val, _w0123);

                _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl00, _sh00));

                tmpptr += 1;
                kptr0 += 4;
            }

            int sum[4];
            _mm_storeu_si128((__m128i*)sum, _sum0);

            outptr0[0] = sum[0];
            outptr1[0] = sum[1];
            outptr2[0] = sum[2];
            outptr3[0] = sum[3];
            outptr0 += 1;
            outptr1 += 1;
            outptr2 += 1;
            outptr3 += 1;
        }
    }

    remain_outch_start += nn_outch << 2;
#endif // __SSE2__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* outptr0 = top_blob.channel(p);

        int i = 0;
#if __SSE2__
#if __AVX2__
        for (; i + 3 < size; i += 4)
        {
            const signed char* tmpptr = tmp.channel(i / 4);
            const signed char* kptr0 = kernel.channel(p / 4 + p % 4);

            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;

            if (nn4 > 0)
            {
                __m256i _sum0_2 = _mm256_setzero_si256();
                __m256i _sum1_3 = _mm256_setzero_si256();

                int j = 0;
                for (; j < nn4; j++)
                {
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m256i _val01_16 = _mm256_cvtepi8_epi16(_val01);

                    __m128i _w0123 = _mm_loadl_epi64((const __m128i*)kptr0);
                    __m128i _w = _mm_cvtepi8_epi16(_w0123);
                    _w = _mm_unpacklo_epi64(_w, _w);
                    __m256i _ww = _mm256_inserti128_si256(_mm256_castsi128_si256(_w), _w, 1);

                    __m256i _sl0_1 = _mm256_mullo_epi16(_val01_16, _ww);
                    __m256i _sh0_1 = _mm256_mulhi_epi16(_val01_16, _ww);

                    _sum0_2 = _mm256_add_epi32(_sum0_2, _mm256_unpacklo_epi16(_sl0_1, _sh0_1));
                    _sum1_3 = _mm256_add_epi32(_sum1_3, _mm256_unpackhi_epi16(_sl0_1, _sh0_1));

                    tmpptr += 16;
                    kptr0 += 4;
                }

                __m128i _sum0 = _mm256_extracti128_si256(_sum0_2, 0);
                __m128i _sum1 = _mm256_extracti128_si256(_sum1_3, 0);
                __m128i _sum2 = _mm256_extracti128_si256(_sum0_2, 1);
                __m128i _sum3 = _mm256_extracti128_si256(_sum1_3, 1);

                sum0 = _mm_reduce_add_epi32(_sum0);
                sum1 = _mm_reduce_add_epi32(_sum1);
                sum2 = _mm_reduce_add_epi32(_sum2);
                sum3 = _mm_reduce_add_epi32(_sum3);
            }

            int j = 0;
            for (; j < nn1; j++)
            {
                signed char val0 = tmpptr[0];
                signed char val1 = tmpptr[1];
                signed char val2 = tmpptr[2];
                signed char val3 = tmpptr[3];
                signed char w = kptr0[0];

                sum0 += val0 * w;
                sum1 += val1 * w;
                sum2 += val2 * w;
                sum3 += val3 * w;

                tmpptr += 4;
                kptr0 += 1;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;
            outptr0 += 4;
        }
#endif
        for (; i + 1 < size; i += 2)
        {
#if __AVX2__
            const signed char* tmpptr = tmp.channel(i / 4 + (i % 4) / 2);
#else
            const signed char* tmpptr = tmp.channel(i / 2);
#endif
            const signed char* kptr0 = kernel.channel(p / 4 + p % 4);

            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            int sum0 = 0;
            int sum1 = 0;

            if (nn4 > 0)
            {
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();

                int j = 0;
                for (; j < nn4; j++)
                {
                    __m128i _val = _mm_loadl_epi64((const __m128i*)tmpptr);
                    __m128i _extval = _mm_cmpgt_epi8(_mm_setzero_si128(), _val);
                    __m128i _val01 = _mm_unpacklo_epi8(_val, _extval);

                    __m128i _w0123 = _mm_loadl_epi64((const __m128i*)kptr0);
#if __SSE4_1__
                    __m128i _w = _mm_cvtepi8_epi16(_w0123);
#else
                    __m128i _extw = _mm_cmpgt_epi8(_mm_setzero_si128(), _w0123);
                    __m128i _w = _mm_unpacklo_epi8(_w0123, _extw);
#endif
                    _w = _mm_shuffle_epi32(_w, _MM_SHUFFLE(1, 0, 1, 0));

                    __m128i _sl01 = _mm_mullo_epi16(_val01, _w);
                    __m128i _sh01 = _mm_mulhi_epi16(_val01, _w);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl01, _sh01));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl01, _sh01));

                    tmpptr += 8;
                    kptr0 += 4;
                }

                sum0 = _mm_reduce_add_epi32(_sum0);
                sum1 = _mm_reduce_add_epi32(_sum1);
            }

            int j = 0;
            for (; j < nn1; j++)
            {
                signed char val0 = tmpptr[0];
                signed char val1 = tmpptr[1];
                signed char w = kptr0[0];

                sum0 += val0 * w;
                sum1 += val1 * w;

                tmpptr += 2;
                kptr0 += 1;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0 += 2;
        }
        for (; i < size; i++)
        {
#if __AVX2__
            const signed char* tmpptr = tmp.channel(i / 4 + (i % 4) / 2 + i % 2);
#else
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
#endif
            const signed char* kptr0 = kernel.channel(p / 4 + p % 4);

            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            int sum = 0;

            if (nn4 > 0)
            {
                __m128i _sum = _mm_setzero_si128();

                int j = 0;
                for (; j < nn4; j++)
                {
                    __m128i _val0123 = _mm_loadl_epi64((const __m128i*)tmpptr);
#if __SSE4_1__
                    __m128i _val = _mm_cvtepi8_epi16(_val0123);
#else
                    __m128i _extval = _mm_cmpgt_epi8(_mm_setzero_si128(), _val0123);
                    __m128i _val = _mm_unpacklo_epi8(_val0123, _extval);
#endif

                    __m128i _w0123 = _mm_loadl_epi64((const __m128i*)kptr0);
#if __SSE4_1__
                    __m128i _w = _mm_cvtepi8_epi16(_w0123);
#else
                    __m128i _extw = _mm_cmpgt_epi8(_mm_setzero_si128(), _w0123);
                    __m128i _w = _mm_unpacklo_epi8(_w0123, _extw);
#endif

                    __m128i _sl = _mm_mullo_epi16(_val, _w);
                    __m128i _sh = _mm_mulhi_epi16(_val, _w);

                    _sum = _mm_add_epi32(_sum, _mm_unpacklo_epi16(_sl, _sh));

                    tmpptr += 4;
                    kptr0 += 4;
                }

                sum = _mm_reduce_add_epi32(_sum);
            }

            int j = 0;
            for (; j < nn1; j++)
            {
                signed char val = tmpptr[0];
                signed char w = kptr0[0];

                sum += val * w;

                tmpptr += 1;
                kptr0 += 1;
            }

            outptr0[0] = sum;
            outptr0 += 1;
        }
#else  // __SSE2__
        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i);
            const signed char* kptr0 = kernel.channel(p);

            int nn1 = inch * maxk;

            int sum = 0;
            int j = 0;
            for (; j < nn1; j++)
            {
                signed char val = tmpptr[0];
                signed char w = kptr0[0];

                sum += val * w;

                tmpptr += 1;
                kptr0 += 1;
            }

            outptr0[0] = sum;
            outptr0 += 1;
        }
#endif // __SSE2__
    }
}

static void convolution_im2col_sgemm_transform_kernel_int8_sse(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

#if __SSE2__
    // interleave
    // src = maxk-inch-outch
    // dst = 4a-4b-maxk-inch/4a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    if (outch >= 4)
    {
        if (inch >= 4)
            kernel_tm.create(16 * maxk, inch / 4 + inch % 4, outch / 4 + outch % 4, (size_t)1u);
        else
            kernel_tm.create(4 * maxk, inch, outch / 4 + outch % 4, (size_t)1u);
    }
    else
    {
        if (inch >= 4)
            kernel_tm.create(4 * maxk, inch / 4 + inch % 4, outch, (size_t)1u);
        else
            kernel_tm.create(1 * maxk, inch, outch, (size_t)1u);
    }

    int q = 0;
    for (; q + 3 < outch; q += 4)
    {
        signed char* g00 = kernel_tm.channel(q / 4);

        int p = 0;
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const signed char* k00 = kernel.channel(q + i).row<const signed char>(p);

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
    // TODO unroll 2
    for (; q < outch; q++)
    {
        signed char* g00 = kernel_tm.channel(q / 4 + q % 4);

        int p = 0;
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 4; j++)
                {
                    const signed char* k00 = kernel.channel(q).row<const signed char>(p + j);

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k00 = kernel.channel(q).row<const signed char>(p);

                g00[0] = k00[k];

                g00++;
            }
        }
    }
#else  // __SSE2__
    kernel_tm = _kernel.reshape(maxk, inch, outch);
#endif // __SSE2__
}

static void convolution_im2col_sgemm_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 1u, 1, opt.workspace_allocator);
    {
        const int gap = w * stride_h - outw * stride_w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            signed char* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const signed char* sptr = img.row<const signed char>(dilation_h * u) + dilation_w * v;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j + 3 < outw; j += 4)
                        {
                            ptr[0] = sptr[0];
                            ptr[1] = sptr[stride_w];
                            ptr[2] = sptr[stride_w * 2];
                            ptr[3] = sptr[stride_w * 3];

                            sptr += stride_w * 4;
                            ptr += 4;
                        }
                        for (; j + 1 < outw; j += 2)
                        {
                            ptr[0] = sptr[0];
                            ptr[1] = sptr[stride_w];

                            sptr += stride_w * 2;
                            ptr += 2;
                        }
                        for (; j < outw; j++)
                        {
                            ptr[0] = sptr[0];

                            sptr += stride_w;
                            ptr += 1;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_int8_sse(bottom_im2col, top_blob, kernel, opt);
}
