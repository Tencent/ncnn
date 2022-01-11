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

static void im2col_sgemm_pack1to4_int8_sse(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 8u, 8, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    // permute
    Mat tmp;
    if (inch >= 4)
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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        int* outptr0 = top_blob.channel(p);

        int i = 0;
        for (; i + 1 < size; i += 2)
        {
            const signed char* tmpptr = tmp.channel(i / 2);
            const signed char* kptr0 = kernel.channel(p);

            int nn4 = (inch / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            __m128i _sum00 = _mm_setzero_si128();
            __m128i _sum10 = _mm_setzero_si128();

            if (nn4 > 0)
            {
                __m128i _sum01 = _mm_setzero_si128();
                __m128i _sum02 = _mm_setzero_si128();
                __m128i _sum03 = _mm_setzero_si128();
                __m128i _sum11 = _mm_setzero_si128();
                __m128i _sum12 = _mm_setzero_si128();
                __m128i _sum13 = _mm_setzero_si128();

                int j = 0;
                for (; j < nn4; j++)
                {
                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    _val01 = _mm_unpacklo_epi8(_val01, _extval01);

                    __m128i _val0 = _mm_shuffle_epi32(_val01, _MM_SHUFFLE(1, 0, 1, 0));
                    __m128i _val1 = _mm_shuffle_epi32(_val01, _MM_SHUFFLE(3, 2, 3, 2));

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

                    _sum00 = _mm_add_epi32(_sum00, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum01 = _mm_add_epi32(_sum01, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum02 = _mm_add_epi32(_sum02, _mm_unpacklo_epi16(_sl01, _sh01));
                    _sum03 = _mm_add_epi32(_sum03, _mm_unpackhi_epi16(_sl01, _sh01));
                    _sum10 = _mm_add_epi32(_sum10, _mm_unpacklo_epi16(_sl10, _sh10));
                    _sum11 = _mm_add_epi32(_sum11, _mm_unpackhi_epi16(_sl10, _sh10));
                    _sum12 = _mm_add_epi32(_sum12, _mm_unpacklo_epi16(_sl11, _sh11));
                    _sum13 = _mm_add_epi32(_sum13, _mm_unpackhi_epi16(_sl11, _sh11));

                    tmpptr += 8;
                    kptr0 += 16;
                }

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
            }

            int j = 0;
            for (; j < nn1; j++)
            {
                __m128i _val = _mm_set_epi16(tmpptr[1], tmpptr[1], tmpptr[1], tmpptr[1], tmpptr[0], tmpptr[0], tmpptr[0], tmpptr[0]);

                __m128i _w0123 = _mm_set_epi16(kptr0[3], kptr0[2], kptr0[1], kptr0[0], kptr0[3], kptr0[2], kptr0[1], kptr0[0]);

                __m128i _sl00 = _mm_mullo_epi16(_val, _w0123);
                __m128i _sh00 = _mm_mulhi_epi16(_val, _w0123);

                _sum00 = _mm_add_epi32(_sum00, _mm_unpacklo_epi16(_sl00, _sh00));
                _sum10 = _mm_add_epi32(_sum10, _mm_unpackhi_epi16(_sl00, _sh00));

                tmpptr += 2;
                kptr0 += 4;
            }

            _mm_storeu_si128((__m128i*)outptr0, _sum00);
            _mm_storeu_si128((__m128i*)(outptr0 + 4), _sum10);
            outptr0 += 8;
        }
        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
            const signed char* kptr0 = kernel.channel(p);

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
                    // TODO use _mm_cvtepi8_epi16 on sse4.1
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);

                    _val0 = _mm_shuffle_epi32(_val0, _MM_SHUFFLE(1, 0, 1, 0));

                    // TODO use _mm_cvtepi8_epi16 on sse4.1
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

                __m128i _w0123 = _mm_set_epi16(0, 0, 0, 0, kptr0[3], kptr0[2], kptr0[1], kptr0[0]);

                __m128i _sl00 = _mm_mullo_epi16(_val, _w0123);
                __m128i _sh00 = _mm_mulhi_epi16(_val, _w0123);

                _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl00, _sh00));

                tmpptr += 1;
                kptr0 += 4;
            }

            _mm_storeu_si128((__m128i*)outptr0, _sum0);
            outptr0 += 4;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack1to4_int8_sse(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4a-4b-maxk-inch/4a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    if (inch >= 4)
        kernel_tm.create(16 * maxk, inch / 4 + inch % 4, outch / 4, (size_t)1u);
    else
        kernel_tm.create(4 * maxk, inch, outch / 4, (size_t)1u);

    for (int q = 0; q + 3 < outch; q += 4)
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
}

static void convolution_im2col_sgemm_pack1to4_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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

    im2col_sgemm_pack1to4_int8_sse(bottom_im2col, top_blob, kernel, opt);
}
