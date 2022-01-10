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

static void im2col_sgemm_int8_sse(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 8u, 8, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    // permute
    Mat tmp;
#if __SSE2__
    if (inch >= 8)
    {
        if (size >= 2)
            tmp.create(2 * maxk, inch / 8 + (inch % 8) / 4 + inch % 4, size / 2 + size % 2, 8u, 8, opt.workspace_allocator);
        else
            tmp.create(maxk, inch / 8 + (inch % 8) / 4 + inch % 4, size, 8u, 8, opt.workspace_allocator);
    }
    else if (inch >= 4)
    {
        if (size >= 2)
            tmp.create(2 * maxk, inch / 4 + inch % 4, size / 2 + size % 2, 4u, 4, opt.workspace_allocator);
        else
            tmp.create(maxk, inch / 4 + inch % 4, size, 4u, 4, opt.workspace_allocator);
    }
    else
    {
        if (size >= 2)
            tmp.create(2 * maxk, inch, size / 2 + size % 2, 1u, 1, opt.workspace_allocator);
        else
            tmp.create(maxk, inch, size, 1u, 1, opt.workspace_allocator);
    }
    {
        int remain_size_start = 0;
        int nn_size = (size - remain_size_start) >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            signed char* tmpptr = tmp.channel(i / 2);

            int q = 0;
            for (; q + 7 < inch; q += 8)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;
                const signed char* img4 = (const signed char*)bottom_im2col.channel(q + 4) + i;
                const signed char* img5 = (const signed char*)bottom_im2col.channel(q + 5) + i;
                const signed char* img6 = (const signed char*)bottom_im2col.channel(q + 6) + i;
                const signed char* img7 = (const signed char*)bottom_im2col.channel(q + 7) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img4[0];
                    tmpptr[5] = img5[0];
                    tmpptr[6] = img6[0];
                    tmpptr[7] = img7[0];
                    tmpptr += 8;

                    tmpptr[0] = img0[1];
                    tmpptr[1] = img1[1];
                    tmpptr[2] = img2[1];
                    tmpptr[3] = img3[1];
                    tmpptr[4] = img4[1];
                    tmpptr[5] = img5[1];
                    tmpptr[6] = img6[1];
                    tmpptr[7] = img7[1];
                    tmpptr += 8;

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                    img4 += size;
                    img5 += size;
                    img6 += size;
                    img7 += size;
                }
            }
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
            signed char* tmpptr = tmp.channel(i / 2 + i % 2);

            int q = 0;
            for (; q + 7 < inch; q += 8)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;
                const signed char* img4 = (const signed char*)bottom_im2col.channel(q + 4) + i;
                const signed char* img5 = (const signed char*)bottom_im2col.channel(q + 5) + i;
                const signed char* img6 = (const signed char*)bottom_im2col.channel(q + 6) + i;
                const signed char* img7 = (const signed char*)bottom_im2col.channel(q + 7) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img4[0];
                    tmpptr[5] = img5[0];
                    tmpptr[6] = img6[0];
                    tmpptr[7] = img7[0];
                    tmpptr += 8;

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                    img4 += size;
                    img5 += size;
                    img6 += size;
                    img7 += size;
                }
            }
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
        for (; i + 1 < size; i += 2)
        {
            const signed char* tmpptr = tmp.channel(i / 2);
            const signed char* kptr0 = kernel.channel(p / 4);

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            __m128i _sum00 = _mm_setzero_si128();
            __m128i _sum10 = _mm_setzero_si128();
            if (nn > 0)
            {
                __m128i _sum01 = _mm_setzero_si128();
                __m128i _sum02 = _mm_setzero_si128();
                __m128i _sum03 = _mm_setzero_si128();
                __m128i _sum11 = _mm_setzero_si128();
                __m128i _sum12 = _mm_setzero_si128();
                __m128i _sum13 = _mm_setzero_si128();

                int j = 0;
                for (; j < nn; j++)
                {
                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);
                    __m128i _val1 = _mm_unpackhi_epi8(_val01, _extval01);

                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _extw23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w23);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                    __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);
                    __m128i _w2 = _mm_unpacklo_epi8(_w23, _extw23);
                    __m128i _w3 = _mm_unpackhi_epi8(_w23, _extw23);

                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                    __m128i _sl01 = _mm_mullo_epi16(_val0, _w1);
                    __m128i _sh01 = _mm_mulhi_epi16(_val0, _w1);
                    __m128i _sl02 = _mm_mullo_epi16(_val0, _w2);
                    __m128i _sh02 = _mm_mulhi_epi16(_val0, _w2);
                    __m128i _sl03 = _mm_mullo_epi16(_val0, _w3);
                    __m128i _sh03 = _mm_mulhi_epi16(_val0, _w3);
                    __m128i _sl10 = _mm_mullo_epi16(_val1, _w0);
                    __m128i _sh10 = _mm_mulhi_epi16(_val1, _w0);
                    __m128i _sl11 = _mm_mullo_epi16(_val1, _w1);
                    __m128i _sh11 = _mm_mulhi_epi16(_val1, _w1);
                    __m128i _sl12 = _mm_mullo_epi16(_val1, _w2);
                    __m128i _sh12 = _mm_mulhi_epi16(_val1, _w2);
                    __m128i _sl13 = _mm_mullo_epi16(_val1, _w3);
                    __m128i _sh13 = _mm_mulhi_epi16(_val1, _w3);

                    _sum00 = _mm_add_epi32(_sum00, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum00 = _mm_add_epi32(_sum00, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum01 = _mm_add_epi32(_sum01, _mm_unpacklo_epi16(_sl01, _sh01));
                    _sum01 = _mm_add_epi32(_sum01, _mm_unpackhi_epi16(_sl01, _sh01));
                    _sum02 = _mm_add_epi32(_sum02, _mm_unpacklo_epi16(_sl02, _sh02));
                    _sum02 = _mm_add_epi32(_sum02, _mm_unpackhi_epi16(_sl02, _sh02));
                    _sum03 = _mm_add_epi32(_sum03, _mm_unpacklo_epi16(_sl03, _sh03));
                    _sum03 = _mm_add_epi32(_sum03, _mm_unpackhi_epi16(_sl03, _sh03));
                    _sum10 = _mm_add_epi32(_sum10, _mm_unpacklo_epi16(_sl10, _sh10));
                    _sum10 = _mm_add_epi32(_sum10, _mm_unpackhi_epi16(_sl10, _sh10));
                    _sum11 = _mm_add_epi32(_sum11, _mm_unpacklo_epi16(_sl11, _sh11));
                    _sum11 = _mm_add_epi32(_sum11, _mm_unpackhi_epi16(_sl11, _sh11));
                    _sum12 = _mm_add_epi32(_sum12, _mm_unpacklo_epi16(_sl12, _sh12));
                    _sum12 = _mm_add_epi32(_sum12, _mm_unpackhi_epi16(_sl12, _sh12));
                    _sum13 = _mm_add_epi32(_sum13, _mm_unpacklo_epi16(_sl13, _sh13));
                    _sum13 = _mm_add_epi32(_sum13, _mm_unpackhi_epi16(_sl13, _sh13));

                    tmpptr += 16;
                    kptr0 += 32;
                }

                _sum00 = _mm_add_epi32(_sum00, _sum01);
                _sum02 = _mm_add_epi32(_sum02, _sum03);
                _sum10 = _mm_add_epi32(_sum10, _sum11);
                _sum12 = _mm_add_epi32(_sum12, _sum13);

                _sum00 = _mm_add_epi32(_sum00, _sum02);
                _sum10 = _mm_add_epi32(_sum10, _sum12);
            }

            if (nn4 > 0)
            {
                __m128i _sum100 = _mm_setzero_si128();
                __m128i _sum101 = _mm_setzero_si128();
                __m128i _sum110 = _mm_setzero_si128();
                __m128i _sum111 = _mm_setzero_si128();

                int j = 0;
                for (; j < nn4; j++)
                {
                    __m128i _val0 = _mm_set_epi16(tmpptr[0], tmpptr[0], tmpptr[1], tmpptr[1], tmpptr[2], tmpptr[2], tmpptr[3], tmpptr[3]);
                    __m128i _val1 = _mm_set_epi16(tmpptr[4], tmpptr[4], tmpptr[5], tmpptr[5], tmpptr[6], tmpptr[6], tmpptr[7], tmpptr[7]);

                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                    __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);

                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                    __m128i _sl01 = _mm_mullo_epi16(_val0, _w1);
                    __m128i _sh01 = _mm_mulhi_epi16(_val0, _w1);
                    __m128i _sl10 = _mm_mullo_epi16(_val1, _w0);
                    __m128i _sh10 = _mm_mulhi_epi16(_val1, _w0);
                    __m128i _sl11 = _mm_mullo_epi16(_val1, _w1);
                    __m128i _sh11 = _mm_mulhi_epi16(_val1, _w1);

                    _sum100 = _mm_add_epi32(_sum100, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum100 = _mm_add_epi32(_sum100, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum101 = _mm_add_epi32(_sum101, _mm_unpacklo_epi16(_sl01, _sh01));
                    _sum101 = _mm_add_epi32(_sum101, _mm_unpackhi_epi16(_sl01, _sh01));
                    _sum110 = _mm_add_epi32(_sum110, _mm_unpacklo_epi16(_sl10, _sh10));
                    _sum110 = _mm_add_epi32(_sum110, _mm_unpackhi_epi16(_sl10, _sh10));
                    _sum111 = _mm_add_epi32(_sum111, _mm_unpacklo_epi16(_sl11, _sh11));
                    _sum111 = _mm_add_epi32(_sum111, _mm_unpackhi_epi16(_sl11, _sh11));

                    tmpptr += 8;
                    kptr0 += 16;
                }

                _sum100 = _mm_add_epi32(_sum100, _sum101);
                _sum110 = _mm_add_epi32(_sum110, _sum111);
                _sum00 = _mm_add_epi32(_sum00, _sum100);
                _sum10 = _mm_add_epi32(_sum10, _sum110);
            }

            int j = 0;
            for (; j < nn1; j++)
            {
                __m128i _val = _mm_set_epi16(tmpptr[0], tmpptr[0], tmpptr[0], tmpptr[0], tmpptr[1], tmpptr[1], tmpptr[1], tmpptr[1]);

                __m128i _w0123 = _mm_set_epi16(kptr0[0], kptr0[1], kptr0[2], kptr0[3], kptr0[0], kptr0[1], kptr0[2], kptr0[3]);

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
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
            const signed char* kptr0 = kernel.channel(p / 4);

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            __m128i _sum0 = _mm_setzero_si128();
            if (nn > 0)
            {
                __m128i _sum1 = _mm_setzero_si128();
                __m128i _sum2 = _mm_setzero_si128();
                __m128i _sum3 = _mm_setzero_si128();

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
//                     int8x16_t _val = vld1q_s8(tmpptr);

//                     int8x16_t _w01 = vld1q_s8(kptr0);
//                     int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);
                    __m128i _val1 = _mm_unpackhi_epi8(_val01, _extval01);

                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _extw23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w23);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                    __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);
                    __m128i _w2 = _mm_unpacklo_epi8(_w23, _extw23);
                    __m128i _w3 = _mm_unpackhi_epi8(_w23, _extw23);

                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                    __m128i _sl01 = _mm_mullo_epi16(_val0, _w1);
                    __m128i _sh01 = _mm_mulhi_epi16(_val0, _w1);
                    __m128i _sl02 = _mm_mullo_epi16(_val0, _w2);
                    __m128i _sh02 = _mm_mulhi_epi16(_val0, _w2);
                    __m128i _sl03 = _mm_mullo_epi16(_val0, _w3);
                    __m128i _sh03 = _mm_mulhi_epi16(_val0, _w3);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum0 = _mm_add_epi32(_sum0, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpacklo_epi16(_sl01, _sh01));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl01, _sh01));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpacklo_epi16(_sl02, _sh02));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpackhi_epi16(_sl02, _sh02));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpacklo_epi16(_sl03, _sh03));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpackhi_epi16(_sl03, _sh03));

                    __m128i _sl10 = _mm_mullo_epi16(_val1, _w0);
                    __m128i _sh10 = _mm_mulhi_epi16(_val1, _w0);
                    __m128i _sl11 = _mm_mullo_epi16(_val1, _w1);
                    __m128i _sh11 = _mm_mulhi_epi16(_val1, _w1);
                    __m128i _sl12 = _mm_mullo_epi16(_val1, _w2);
                    __m128i _sh12 = _mm_mulhi_epi16(_val1, _w2);
                    __m128i _sl13 = _mm_mullo_epi16(_val1, _w3);
                    __m128i _sh13 = _mm_mulhi_epi16(_val1, _w3);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl10, _sh10));
                    _sum0 = _mm_add_epi32(_sum0, _mm_unpackhi_epi16(_sl10, _sh10));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpacklo_epi16(_sl11, _sh11));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl11, _sh11));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpacklo_epi16(_sl12, _sh12));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpackhi_epi16(_sl12, _sh12));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpacklo_epi16(_sl13, _sh13));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpackhi_epi16(_sl13, _sh13));

//                     int16x8_t _wv0 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w01));
//                     int16x8_t _wv1 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w01));
//                     int16x8_t _wv2 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w23));
//                     int16x8_t _wv3 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w23));
//
//                     int8x16_t _w45 = vld1q_s8(kptr0 + 32);
//                     int8x16_t _w67 = vld1q_s8(kptr0 + 48);
//
//                     _wv0 = vmlal_s8(_wv0, vget_high_s8(_val), vget_low_s8(_w45));
//                     _wv1 = vmlal_s8(_wv1, vget_high_s8(_val), vget_high_s8(_w45));
//                     _wv2 = vmlal_s8(_wv2, vget_high_s8(_val), vget_low_s8(_w67));
//                     _wv3 = vmlal_s8(_wv3, vget_high_s8(_val), vget_high_s8(_w67));
//
//                     _sum0 = vpadalq_s16(_sum0, _wv0);
//                     _sum1 = vpadalq_s16(_sum1, _wv1);
//                     _sum2 = vpadalq_s16(_sum2, _wv2);
//                     _sum3 = vpadalq_s16(_sum3, _wv3);

                    tmpptr += 16;
                    kptr0 += 64;
                }
                for (; j < nn; j++)
                {
                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);

                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _extw23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w23);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                    __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);
                    __m128i _w2 = _mm_unpacklo_epi8(_w23, _extw23);
                    __m128i _w3 = _mm_unpackhi_epi8(_w23, _extw23);

                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                    __m128i _sl01 = _mm_mullo_epi16(_val0, _w1);
                    __m128i _sh01 = _mm_mulhi_epi16(_val0, _w1);
                    __m128i _sl02 = _mm_mullo_epi16(_val0, _w2);
                    __m128i _sh02 = _mm_mulhi_epi16(_val0, _w2);
                    __m128i _sl03 = _mm_mullo_epi16(_val0, _w3);
                    __m128i _sh03 = _mm_mulhi_epi16(_val0, _w3);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum0 = _mm_add_epi32(_sum0, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpacklo_epi16(_sl01, _sh01));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl01, _sh01));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpacklo_epi16(_sl02, _sh02));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpackhi_epi16(_sl02, _sh02));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpacklo_epi16(_sl03, _sh03));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpackhi_epi16(_sl03, _sh03));

                    tmpptr += 8;
                    kptr0 += 32;
                }

                _sum0 = _mm_add_epi32(_sum0, _sum1);
                _sum2 = _mm_add_epi32(_sum2, _sum3);
                _sum0 = _mm_add_epi32(_sum0, _sum2);
            }

            if (nn4 > 0)
            {
                __m128i _sum10 = _mm_setzero_si128();
                __m128i _sum11 = _mm_setzero_si128();

                int j = 0;
                for (; j < nn4; j++)
                {
                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);

                    _val0 = _mm_shuffle_epi32(_val0, _MM_SHUFFLE(0, 1, 0, 1));

                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                    __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);

                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                    __m128i _sl01 = _mm_mullo_epi16(_val0, _w1);
                    __m128i _sh01 = _mm_mulhi_epi16(_val0, _w1);

                    _sum10 = _mm_add_epi32(_sum10, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum10 = _mm_add_epi32(_sum10, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum11 = _mm_add_epi32(_sum11, _mm_unpacklo_epi16(_sl01, _sh01));
                    _sum11 = _mm_add_epi32(_sum11, _mm_unpackhi_epi16(_sl01, _sh01));

                    tmpptr += 4;
                    kptr0 += 16;
                }

                _sum10 = _mm_add_epi32(_sum10, _sum11);
                _sum0 = _mm_add_epi32(_sum0, _sum10);
            }

            int j = 0;
            for (; j < nn1; j++)
            {
                __m128i _val = _mm_set1_epi16(tmpptr[0]);

                __m128i _w0123 = _mm_set_epi16(kptr0[0], kptr0[1], kptr0[2], kptr0[3], 0, 0, 0, 0);

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
        for (; i + 1 < size; i += 2)
        {
            const signed char* tmpptr = tmp.channel(i / 2);
            const signed char* kptr0 = kernel.channel(p / 4 + p % 4);

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            int sum0 = 0;
            int sum1 = 0;
            if (nn > 0)
            {
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();
                __m128i _sum2 = _mm_setzero_si128();
                __m128i _sum3 = _mm_setzero_si128();

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _val23 = _mm_loadu_si128((const __m128i*)(tmpptr + 16));
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _extval23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val23);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);
                    __m128i _val1 = _mm_unpackhi_epi8(_val01, _extval01);
                    __m128i _val2 = _mm_unpacklo_epi8(_val23, _extval23);
                    __m128i _val3 = _mm_unpackhi_epi8(_val23, _extval23);

                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                    __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);

                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                    __m128i _sl10 = _mm_mullo_epi16(_val1, _w0);
                    __m128i _sh10 = _mm_mulhi_epi16(_val1, _w0);
                    __m128i _sl21 = _mm_mullo_epi16(_val2, _w1);
                    __m128i _sh21 = _mm_mulhi_epi16(_val2, _w1);
                    __m128i _sl31 = _mm_mullo_epi16(_val3, _w1);
                    __m128i _sh31 = _mm_mulhi_epi16(_val3, _w1);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpacklo_epi16(_sl10, _sh10));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpackhi_epi16(_sl10, _sh10));
                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl21, _sh21));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl21, _sh21));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpacklo_epi16(_sl31, _sh31));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpackhi_epi16(_sl31, _sh31));

                    tmpptr += 32;
                    kptr0 += 16;
                }
                for (; j < nn; j++)
                {
                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);
                    __m128i _val1 = _mm_unpackhi_epi8(_val01, _extval01);

                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);

                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                    __m128i _sl10 = _mm_mullo_epi16(_val1, _w0);
                    __m128i _sh10 = _mm_mulhi_epi16(_val1, _w0);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpacklo_epi16(_sl10, _sh10));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpackhi_epi16(_sl10, _sh10));

                    tmpptr += 16;
                    kptr0 += 8;
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

                sum0 = _mm_reduce_add_epi32(_sum0);
                sum1 = _mm_reduce_add_epi32(_sum2);
            }

            if (nn4 > 0)
            {
                int j = 0;
                for (; j < nn4; j++)
                {
                    signed char val0 = tmpptr[0];
                    signed char val1 = tmpptr[1];
                    signed char val2 = tmpptr[2];
                    signed char val3 = tmpptr[3];
                    signed char val4 = tmpptr[4];
                    signed char val5 = tmpptr[5];
                    signed char val6 = tmpptr[6];
                    signed char val7 = tmpptr[7];

                    signed char w0 = kptr0[0];
                    signed char w1 = kptr0[1];
                    signed char w2 = kptr0[2];
                    signed char w3 = kptr0[3];

                    sum0 += val0 * w0;
                    sum0 += val1 * w1;
                    sum0 += val2 * w2;
                    sum0 += val3 * w3;
                    sum1 += val4 * w0;
                    sum1 += val5 * w1;
                    sum1 += val6 * w2;
                    sum1 += val7 * w3;

                    tmpptr += 8;
                    kptr0 += 4;
                }
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
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
            const signed char* kptr0 = kernel.channel(p / 4 + p % 4);

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            int sum = 0;
            if (nn > 0)
            {
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);
                    __m128i _val1 = _mm_unpackhi_epi8(_val01, _extval01);

                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                    __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);

                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                    __m128i _sl11 = _mm_mullo_epi16(_val1, _w1);
                    __m128i _sh11 = _mm_mulhi_epi16(_val1, _w1);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl11, _sh11));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl11, _sh11));

                    tmpptr += 16;
                    kptr0 += 16;
                }
                for (; j < nn; j++)
                {
                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);

                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);

                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl00, _sh00));

                    tmpptr += 8;
                    kptr0 += 8;
                }

                _sum0 = _mm_add_epi32(_sum0, _sum1);

                sum = _mm_reduce_add_epi32(_sum0);
            }

            if (nn4 > 0)
            {
                int j = 0;
                for (; j < nn4; j++)
                {
                    signed char val0 = tmpptr[0];
                    signed char val1 = tmpptr[1];
                    signed char val2 = tmpptr[2];
                    signed char val3 = tmpptr[3];

                    signed char w0 = kptr0[0];
                    signed char w1 = kptr0[1];
                    signed char w2 = kptr0[2];
                    signed char w3 = kptr0[3];

                    sum += val0 * w0;
                    sum += val1 * w1;
                    sum += val2 * w2;
                    sum += val3 * w3;

                    tmpptr += 4;
                    kptr0 += 4;
                }
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
    // dst = 8a-4b-maxk-inch/8a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    if (outch >= 4)
    {
        if (inch >= 8)
            kernel_tm.create(32 * maxk, inch / 8 + (inch % 8) / 4 + inch % 4, outch / 4 + outch % 4, (size_t)1u);
        else if (inch >= 4)
            kernel_tm.create(16 * maxk, inch / 4 + inch % 4, outch / 4 + outch % 4, (size_t)1u);
        else
            kernel_tm.create(4 * maxk, inch, outch / 4 + outch % 4, (size_t)1u);
    }
    else
    {
        if (inch >= 8)
            kernel_tm.create(8 * maxk, inch / 8 + (inch % 8) / 4 + inch % 4, outch, (size_t)1u);
        else if (inch >= 4)
            kernel_tm.create(4 * maxk, inch / 4 + inch % 4, outch, (size_t)1u);
        else
            kernel_tm.create(1 * maxk, inch, outch, (size_t)1u);
    }

    int q = 0;
    for (; q + 3 < outch; q += 4)
    {
        signed char* g00 = kernel_tm.channel(q / 4);

        int p = 0;
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
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
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 8; j++)
                {
                    const signed char* k00 = kernel.channel(q).row<const signed char>(p + j);

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
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
