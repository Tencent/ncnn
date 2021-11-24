// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void im2col_sgemm_pack8to4_int8_sse(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 8u, 8, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    // permute
    Mat tmp;
    if (size >= 2)
        tmp.create(2 * maxk, inch, size / 2 + size % 2, 8u, 8, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 8u, 8, opt.workspace_allocator);
    {
        int remain_size_start = 0;
        int nn_size = size >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            int64_t* tmpptr = tmp.channel(i / 2);

            for (int q = 0; q < inch; q++)
            {
                const int64_t* img0 = (const int64_t*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    __m128i _v = _mm_loadu_si128((const __m128i*)img0);
                    _mm_storeu_si128((__m128i*)tmpptr, _v);
                    tmpptr += 2;
                    img0 += size;
                }
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            int64_t* tmpptr = tmp.channel(i / 2 + i % 2);

            for (int q = 0; q < inch; q++)
            {
                const int64_t* img0 = (const int64_t*)bottom_im2col.channel(q) + i;

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

            int nn = inch * maxk; // inch always > 0

            __m128i _sum00 = _mm_setzero_si128();
            __m128i _sum01 = _mm_setzero_si128();
            __m128i _sum02 = _mm_setzero_si128();
            __m128i _sum03 = _mm_setzero_si128();
            __m128i _sum10 = _mm_setzero_si128();
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
                _sum01 = _mm_add_epi32(_sum01, _mm_unpacklo_epi16(_sl01, _sh01));
                _sum02 = _mm_add_epi32(_sum02, _mm_unpacklo_epi16(_sl02, _sh02));
                _sum03 = _mm_add_epi32(_sum03, _mm_unpacklo_epi16(_sl03, _sh03));
                _sum00 = _mm_add_epi32(_sum00, _mm_unpackhi_epi16(_sl00, _sh00));
                _sum01 = _mm_add_epi32(_sum01, _mm_unpackhi_epi16(_sl01, _sh01));
                _sum02 = _mm_add_epi32(_sum02, _mm_unpackhi_epi16(_sl02, _sh02));
                _sum03 = _mm_add_epi32(_sum03, _mm_unpackhi_epi16(_sl03, _sh03));
                _sum10 = _mm_add_epi32(_sum10, _mm_unpacklo_epi16(_sl10, _sh10));
                _sum11 = _mm_add_epi32(_sum11, _mm_unpacklo_epi16(_sl11, _sh11));
                _sum12 = _mm_add_epi32(_sum12, _mm_unpacklo_epi16(_sl12, _sh12));
                _sum13 = _mm_add_epi32(_sum13, _mm_unpacklo_epi16(_sl13, _sh13));
                _sum10 = _mm_add_epi32(_sum10, _mm_unpackhi_epi16(_sl10, _sh10));
                _sum11 = _mm_add_epi32(_sum11, _mm_unpackhi_epi16(_sl11, _sh11));
                _sum12 = _mm_add_epi32(_sum12, _mm_unpackhi_epi16(_sl12, _sh12));
                _sum13 = _mm_add_epi32(_sum13, _mm_unpackhi_epi16(_sl13, _sh13));

                tmpptr += 16;
                kptr0 += 32;
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

            _mm_storeu_si128((__m128i*)outptr0, _sum00);
            _mm_storeu_si128((__m128i*)(outptr0 + 4), _sum10);
            outptr0 += 8;
        }
        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
            const signed char* kptr0 = kernel.channel(p);

            int nn = inch * maxk; // inch always > 0

            __m128i _sum0 = _mm_setzero_si128();
            __m128i _sum1 = _mm_setzero_si128();
            __m128i _sum2 = _mm_setzero_si128();
            __m128i _sum3 = _mm_setzero_si128();

            int j = 0;
            for (; j < nn; j++)
            {
                // TODO use _mm_cvtepi8_epi16 on sse4.1
                __m128i _val = _mm_loadl_epi64((const __m128i*)tmpptr);
                _val = _mm_unpacklo_epi8(_val, _mm_cmpgt_epi8(_mm_setzero_si128(), _val));

                // TODO use _mm_cvtepi8_epi16 on sse4.1
                __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                __m128i _extw23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w23);
                __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);
                __m128i _w2 = _mm_unpacklo_epi8(_w23, _extw23);
                __m128i _w3 = _mm_unpackhi_epi8(_w23, _extw23);

                __m128i _sl0 = _mm_mullo_epi16(_val, _w0);
                __m128i _sh0 = _mm_mulhi_epi16(_val, _w0);
                __m128i _sl1 = _mm_mullo_epi16(_val, _w1);
                __m128i _sh1 = _mm_mulhi_epi16(_val, _w1);
                __m128i _sl2 = _mm_mullo_epi16(_val, _w2);
                __m128i _sh2 = _mm_mulhi_epi16(_val, _w2);
                __m128i _sl3 = _mm_mullo_epi16(_val, _w3);
                __m128i _sh3 = _mm_mulhi_epi16(_val, _w3);

                _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl0, _sh0));
                _sum1 = _mm_add_epi32(_sum1, _mm_unpacklo_epi16(_sl1, _sh1));
                _sum2 = _mm_add_epi32(_sum2, _mm_unpacklo_epi16(_sl2, _sh2));
                _sum3 = _mm_add_epi32(_sum3, _mm_unpacklo_epi16(_sl3, _sh3));
                _sum0 = _mm_add_epi32(_sum0, _mm_unpackhi_epi16(_sl0, _sh0));
                _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl1, _sh1));
                _sum2 = _mm_add_epi32(_sum2, _mm_unpackhi_epi16(_sl2, _sh2));
                _sum3 = _mm_add_epi32(_sum3, _mm_unpackhi_epi16(_sl3, _sh3));

                tmpptr += 8;
                kptr0 += 32;
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

            _mm_storeu_si128((__m128i*)outptr0, _sum0);
            outptr0 += 4;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack8to4_int8_sse(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8a-4b-maxk-inch/8a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(32 * maxk, inch / 8, outch / 4, 1u, nullptr);

    for (int q = 0; q + 3 < outch; q += 4)
    {
        signed char* g00 = kernel_tm.channel(q / 4);

        for (int p = 0; p + 7 < inch; p += 8)
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
    }
}

static void convolution_im2col_sgemm_pack8to4_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 8u, 8, opt.workspace_allocator);
    {
        const int gap = w * stride_h - outw * stride_w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            int64_t* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const int64_t* sptr = img.row<const int64_t>(dilation_h * u) + dilation_w * v;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
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

    im2col_sgemm_pack8to4_int8_sse(bottom_im2col, top_blob, kernel, opt);
}
