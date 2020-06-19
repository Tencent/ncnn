// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv1x1s1_sgemm_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int size = w * h;

    const float* bias = _bias;

    // interleave
    Mat tmp(8, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, elemsize, elempack, opt.workspace_allocator);
    {
        int nn_size = size >> 3;
        int remain_size_start = nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 8;

            const float* img0 = bottom_blob.channel(0);
            img0 += i * 8;
            float* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                __m256 _r0 = _mm256_loadu_ps(img0);
                __m256 _r1 = _mm256_loadu_ps(img0 + 8);
                __m256 _r2 = _mm256_loadu_ps(img0 + 16);
                __m256 _r3 = _mm256_loadu_ps(img0 + 24);
                __m256 _r4 = _mm256_loadu_ps(img0 + 32);
                __m256 _r5 = _mm256_loadu_ps(img0 + 40);
                __m256 _r6 = _mm256_loadu_ps(img0 + 48);
                __m256 _r7 = _mm256_loadu_ps(img0 + 56);
                _mm256_storeu_ps(tmpptr, _r0);
                _mm256_storeu_ps(tmpptr + 8, _r1);
                _mm256_storeu_ps(tmpptr + 16, _r2);
                _mm256_storeu_ps(tmpptr + 24, _r3);
                _mm256_storeu_ps(tmpptr + 32, _r4);
                _mm256_storeu_ps(tmpptr + 40, _r5);
                _mm256_storeu_ps(tmpptr + 48, _r6);
                _mm256_storeu_ps(tmpptr + 56, _r7);

                tmpptr += 64;
                img0 += bottom_blob.cstep * 8;
            }
        }
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            const float* img0 = bottom_blob.channel(0);
            img0 += i * 8;
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                __m256 _r0 = _mm256_loadu_ps(img0);
                __m256 _r1 = _mm256_loadu_ps(img0 + 8);
                __m256 _r2 = _mm256_loadu_ps(img0 + 16);
                __m256 _r3 = _mm256_loadu_ps(img0 + 24);
                _mm256_storeu_ps(tmpptr, _r0);
                _mm256_storeu_ps(tmpptr + 8, _r1);
                _mm256_storeu_ps(tmpptr + 16, _r2);
                _mm256_storeu_ps(tmpptr + 24, _r3);

                tmpptr += 32;
                img0 += bottom_blob.cstep * 8;
            }
        }

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            const float* img0 = bottom_blob.channel(0);
            img0 += i * 8;
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);

            for (int q = 0; q < inch; q++)
            {
                __m256 _r0 = _mm256_loadu_ps(img0);
                __m256 _r1 = _mm256_loadu_ps(img0 + 8);
                _mm256_storeu_ps(tmpptr, _r0);
                _mm256_storeu_ps(tmpptr + 8, _r1);

                tmpptr += 16;
                img0 += bottom_blob.cstep * 8;
            }
        }

        remain_size_start += nn_size << 1;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            const float* img0 = bottom_blob.channel(0);
            img0 += i * 8;
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            for (int q = 0; q < inch; q++)
            {
                __m256 _r0 = _mm256_loadu_ps(img0);
                _mm256_storeu_ps(tmpptr, _r0);

                tmpptr += 8;
                img0 += bottom_blob.cstep * 8;
            }
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + p * 8) : _mm256_set1_ps(0.f);

        float* outptr = out;
        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 8);

            __m256 _sum0 = _bias0;
            __m256 _sum1 = _bias0;
            __m256 _sum2 = _bias0;
            __m256 _sum3 = _bias0;
            __m256 _sum4 = _bias0;
            __m256 _sum5 = _bias0;
            __m256 _sum6 = _bias0;
            __m256 _sum7 = _bias0;

            const float* kptr = (const float*)kernel + p * inch * 64;
            for (int q = 0; q < inch; q++)
            {
                __m256 _w0 = _mm256_loadu_ps(kptr);
                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                __m256 _w7 = _mm256_loadu_ps(kptr + 56);

                __m256 _val00 = _mm256_broadcast_ss(tmpptr);
                __m256 _val01 = _mm256_broadcast_ss(tmpptr + 1);
                __m256 _val02 = _mm256_broadcast_ss(tmpptr + 2);
                __m256 _val03 = _mm256_broadcast_ss(tmpptr + 3);
                __m256 _val04 = _mm256_broadcast_ss(tmpptr + 4);
                __m256 _val05 = _mm256_broadcast_ss(tmpptr + 5);
                __m256 _val06 = _mm256_broadcast_ss(tmpptr + 6);
                __m256 _val07 = _mm256_broadcast_ss(tmpptr + 7);
                __m256 _val10 = _mm256_broadcast_ss(tmpptr + 8);
                __m256 _val11 = _mm256_broadcast_ss(tmpptr + 9);
                __m256 _val12 = _mm256_broadcast_ss(tmpptr + 10);
                __m256 _val13 = _mm256_broadcast_ss(tmpptr + 11);
                __m256 _val14 = _mm256_broadcast_ss(tmpptr + 12);
                __m256 _val15 = _mm256_broadcast_ss(tmpptr + 13);
                __m256 _val16 = _mm256_broadcast_ss(tmpptr + 14);
                __m256 _val17 = _mm256_broadcast_ss(tmpptr + 15);

                _sum0 = _mm256_fmadd_ps(_w0, _val00, _sum0);
                _sum0 = _mm256_fmadd_ps(_w1, _val01, _sum0);
                _sum0 = _mm256_fmadd_ps(_w2, _val02, _sum0);
                _sum0 = _mm256_fmadd_ps(_w3, _val03, _sum0);
                _sum0 = _mm256_fmadd_ps(_w4, _val04, _sum0);
                _sum0 = _mm256_fmadd_ps(_w5, _val05, _sum0);
                _sum0 = _mm256_fmadd_ps(_w6, _val06, _sum0);
                _sum0 = _mm256_fmadd_ps(_w7, _val07, _sum0);
                _sum1 = _mm256_fmadd_ps(_w0, _val10, _sum1);
                _sum1 = _mm256_fmadd_ps(_w1, _val11, _sum1);
                _sum1 = _mm256_fmadd_ps(_w2, _val12, _sum1);
                _sum1 = _mm256_fmadd_ps(_w3, _val13, _sum1);
                _sum1 = _mm256_fmadd_ps(_w4, _val14, _sum1);
                _sum1 = _mm256_fmadd_ps(_w5, _val15, _sum1);
                _sum1 = _mm256_fmadd_ps(_w6, _val16, _sum1);
                _sum1 = _mm256_fmadd_ps(_w7, _val17, _sum1);

                __m256 _val20 = _mm256_broadcast_ss(tmpptr + 16);
                __m256 _val21 = _mm256_broadcast_ss(tmpptr + 17);
                __m256 _val22 = _mm256_broadcast_ss(tmpptr + 18);
                __m256 _val23 = _mm256_broadcast_ss(tmpptr + 19);
                __m256 _val24 = _mm256_broadcast_ss(tmpptr + 20);
                __m256 _val25 = _mm256_broadcast_ss(tmpptr + 21);
                __m256 _val26 = _mm256_broadcast_ss(tmpptr + 22);
                __m256 _val27 = _mm256_broadcast_ss(tmpptr + 23);
                __m256 _val30 = _mm256_broadcast_ss(tmpptr + 24);
                __m256 _val31 = _mm256_broadcast_ss(tmpptr + 25);
                __m256 _val32 = _mm256_broadcast_ss(tmpptr + 26);
                __m256 _val33 = _mm256_broadcast_ss(tmpptr + 27);
                __m256 _val34 = _mm256_broadcast_ss(tmpptr + 28);
                __m256 _val35 = _mm256_broadcast_ss(tmpptr + 29);
                __m256 _val36 = _mm256_broadcast_ss(tmpptr + 30);
                __m256 _val37 = _mm256_broadcast_ss(tmpptr + 31);

                _sum2 = _mm256_fmadd_ps(_w0, _val20, _sum2);
                _sum2 = _mm256_fmadd_ps(_w1, _val21, _sum2);
                _sum2 = _mm256_fmadd_ps(_w2, _val22, _sum2);
                _sum2 = _mm256_fmadd_ps(_w3, _val23, _sum2);
                _sum2 = _mm256_fmadd_ps(_w4, _val24, _sum2);
                _sum2 = _mm256_fmadd_ps(_w5, _val25, _sum2);
                _sum2 = _mm256_fmadd_ps(_w6, _val26, _sum2);
                _sum2 = _mm256_fmadd_ps(_w7, _val27, _sum2);
                _sum3 = _mm256_fmadd_ps(_w0, _val30, _sum3);
                _sum3 = _mm256_fmadd_ps(_w1, _val31, _sum3);
                _sum3 = _mm256_fmadd_ps(_w2, _val32, _sum3);
                _sum3 = _mm256_fmadd_ps(_w3, _val33, _sum3);
                _sum3 = _mm256_fmadd_ps(_w4, _val34, _sum3);
                _sum3 = _mm256_fmadd_ps(_w5, _val35, _sum3);
                _sum3 = _mm256_fmadd_ps(_w6, _val36, _sum3);
                _sum3 = _mm256_fmadd_ps(_w7, _val37, _sum3);

                __m256 _val40 = _mm256_broadcast_ss(tmpptr + 32);
                __m256 _val41 = _mm256_broadcast_ss(tmpptr + 33);
                __m256 _val42 = _mm256_broadcast_ss(tmpptr + 34);
                __m256 _val43 = _mm256_broadcast_ss(tmpptr + 35);
                __m256 _val44 = _mm256_broadcast_ss(tmpptr + 36);
                __m256 _val45 = _mm256_broadcast_ss(tmpptr + 37);
                __m256 _val46 = _mm256_broadcast_ss(tmpptr + 38);
                __m256 _val47 = _mm256_broadcast_ss(tmpptr + 39);
                __m256 _val50 = _mm256_broadcast_ss(tmpptr + 40);
                __m256 _val51 = _mm256_broadcast_ss(tmpptr + 41);
                __m256 _val52 = _mm256_broadcast_ss(tmpptr + 42);
                __m256 _val53 = _mm256_broadcast_ss(tmpptr + 43);
                __m256 _val54 = _mm256_broadcast_ss(tmpptr + 44);
                __m256 _val55 = _mm256_broadcast_ss(tmpptr + 45);
                __m256 _val56 = _mm256_broadcast_ss(tmpptr + 46);
                __m256 _val57 = _mm256_broadcast_ss(tmpptr + 47);

                _sum4 = _mm256_fmadd_ps(_w0, _val40, _sum4);
                _sum4 = _mm256_fmadd_ps(_w1, _val41, _sum4);
                _sum4 = _mm256_fmadd_ps(_w2, _val42, _sum4);
                _sum4 = _mm256_fmadd_ps(_w3, _val43, _sum4);
                _sum4 = _mm256_fmadd_ps(_w4, _val44, _sum4);
                _sum4 = _mm256_fmadd_ps(_w5, _val45, _sum4);
                _sum4 = _mm256_fmadd_ps(_w6, _val46, _sum4);
                _sum4 = _mm256_fmadd_ps(_w7, _val47, _sum4);
                _sum5 = _mm256_fmadd_ps(_w0, _val50, _sum5);
                _sum5 = _mm256_fmadd_ps(_w1, _val51, _sum5);
                _sum5 = _mm256_fmadd_ps(_w2, _val52, _sum5);
                _sum5 = _mm256_fmadd_ps(_w3, _val53, _sum5);
                _sum5 = _mm256_fmadd_ps(_w4, _val54, _sum5);
                _sum5 = _mm256_fmadd_ps(_w5, _val55, _sum5);
                _sum5 = _mm256_fmadd_ps(_w6, _val56, _sum5);
                _sum5 = _mm256_fmadd_ps(_w7, _val57, _sum5);

                __m256 _val60 = _mm256_broadcast_ss(tmpptr + 48);
                __m256 _val61 = _mm256_broadcast_ss(tmpptr + 49);
                __m256 _val62 = _mm256_broadcast_ss(tmpptr + 50);
                __m256 _val63 = _mm256_broadcast_ss(tmpptr + 51);
                __m256 _val64 = _mm256_broadcast_ss(tmpptr + 52);
                __m256 _val65 = _mm256_broadcast_ss(tmpptr + 53);
                __m256 _val66 = _mm256_broadcast_ss(tmpptr + 54);
                __m256 _val67 = _mm256_broadcast_ss(tmpptr + 55);
                __m256 _val70 = _mm256_broadcast_ss(tmpptr + 56);
                __m256 _val71 = _mm256_broadcast_ss(tmpptr + 57);
                __m256 _val72 = _mm256_broadcast_ss(tmpptr + 58);
                __m256 _val73 = _mm256_broadcast_ss(tmpptr + 59);
                __m256 _val74 = _mm256_broadcast_ss(tmpptr + 60);
                __m256 _val75 = _mm256_broadcast_ss(tmpptr + 61);
                __m256 _val76 = _mm256_broadcast_ss(tmpptr + 62);
                __m256 _val77 = _mm256_broadcast_ss(tmpptr + 63);

                _sum6 = _mm256_fmadd_ps(_w0, _val60, _sum6);
                _sum6 = _mm256_fmadd_ps(_w1, _val61, _sum6);
                _sum6 = _mm256_fmadd_ps(_w2, _val62, _sum6);
                _sum6 = _mm256_fmadd_ps(_w3, _val63, _sum6);
                _sum6 = _mm256_fmadd_ps(_w4, _val64, _sum6);
                _sum6 = _mm256_fmadd_ps(_w5, _val65, _sum6);
                _sum6 = _mm256_fmadd_ps(_w6, _val66, _sum6);
                _sum6 = _mm256_fmadd_ps(_w7, _val67, _sum6);
                _sum7 = _mm256_fmadd_ps(_w0, _val70, _sum7);
                _sum7 = _mm256_fmadd_ps(_w1, _val71, _sum7);
                _sum7 = _mm256_fmadd_ps(_w2, _val72, _sum7);
                _sum7 = _mm256_fmadd_ps(_w3, _val73, _sum7);
                _sum7 = _mm256_fmadd_ps(_w4, _val74, _sum7);
                _sum7 = _mm256_fmadd_ps(_w5, _val75, _sum7);
                _sum7 = _mm256_fmadd_ps(_w6, _val76, _sum7);
                _sum7 = _mm256_fmadd_ps(_w7, _val77, _sum7);

                tmpptr += 64;

                kptr += 64;
            }
            _mm256_storeu_ps(outptr, _sum0);
            _mm256_storeu_ps(outptr + 8, _sum1);
            _mm256_storeu_ps(outptr + 16, _sum2);
            _mm256_storeu_ps(outptr + 24, _sum3);
            _mm256_storeu_ps(outptr + 32, _sum4);
            _mm256_storeu_ps(outptr + 40, _sum5);
            _mm256_storeu_ps(outptr + 48, _sum6);
            _mm256_storeu_ps(outptr + 56, _sum7);

            outptr += 64;
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            __m256 _sum0 = _bias0;
            __m256 _sum1 = _bias0;
            __m256 _sum2 = _bias0;
            __m256 _sum3 = _bias0;

            const float* kptr = (const float*)kernel + p * inch * 64;
            for (int q = 0; q < inch; q++)
            {
                __m256 _w0 = _mm256_loadu_ps(kptr);
                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                __m256 _w7 = _mm256_loadu_ps(kptr + 56);

                __m256 _val00 = _mm256_broadcast_ss(tmpptr);
                __m256 _val01 = _mm256_broadcast_ss(tmpptr + 1);
                __m256 _val02 = _mm256_broadcast_ss(tmpptr + 2);
                __m256 _val03 = _mm256_broadcast_ss(tmpptr + 3);
                __m256 _val04 = _mm256_broadcast_ss(tmpptr + 4);
                __m256 _val05 = _mm256_broadcast_ss(tmpptr + 5);
                __m256 _val06 = _mm256_broadcast_ss(tmpptr + 6);
                __m256 _val07 = _mm256_broadcast_ss(tmpptr + 7);
                __m256 _val10 = _mm256_broadcast_ss(tmpptr + 8);
                __m256 _val11 = _mm256_broadcast_ss(tmpptr + 9);
                __m256 _val12 = _mm256_broadcast_ss(tmpptr + 10);
                __m256 _val13 = _mm256_broadcast_ss(tmpptr + 11);
                __m256 _val14 = _mm256_broadcast_ss(tmpptr + 12);
                __m256 _val15 = _mm256_broadcast_ss(tmpptr + 13);
                __m256 _val16 = _mm256_broadcast_ss(tmpptr + 14);
                __m256 _val17 = _mm256_broadcast_ss(tmpptr + 15);

                _sum0 = _mm256_fmadd_ps(_w0, _val00, _sum0);
                _sum0 = _mm256_fmadd_ps(_w1, _val01, _sum0);
                _sum0 = _mm256_fmadd_ps(_w2, _val02, _sum0);
                _sum0 = _mm256_fmadd_ps(_w3, _val03, _sum0);
                _sum0 = _mm256_fmadd_ps(_w4, _val04, _sum0);
                _sum0 = _mm256_fmadd_ps(_w5, _val05, _sum0);
                _sum0 = _mm256_fmadd_ps(_w6, _val06, _sum0);
                _sum0 = _mm256_fmadd_ps(_w7, _val07, _sum0);
                _sum1 = _mm256_fmadd_ps(_w0, _val10, _sum1);
                _sum1 = _mm256_fmadd_ps(_w1, _val11, _sum1);
                _sum1 = _mm256_fmadd_ps(_w2, _val12, _sum1);
                _sum1 = _mm256_fmadd_ps(_w3, _val13, _sum1);
                _sum1 = _mm256_fmadd_ps(_w4, _val14, _sum1);
                _sum1 = _mm256_fmadd_ps(_w5, _val15, _sum1);
                _sum1 = _mm256_fmadd_ps(_w6, _val16, _sum1);
                _sum1 = _mm256_fmadd_ps(_w7, _val17, _sum1);

                __m256 _val20 = _mm256_broadcast_ss(tmpptr + 16);
                __m256 _val21 = _mm256_broadcast_ss(tmpptr + 17);
                __m256 _val22 = _mm256_broadcast_ss(tmpptr + 18);
                __m256 _val23 = _mm256_broadcast_ss(tmpptr + 19);
                __m256 _val24 = _mm256_broadcast_ss(tmpptr + 20);
                __m256 _val25 = _mm256_broadcast_ss(tmpptr + 21);
                __m256 _val26 = _mm256_broadcast_ss(tmpptr + 22);
                __m256 _val27 = _mm256_broadcast_ss(tmpptr + 23);
                __m256 _val30 = _mm256_broadcast_ss(tmpptr + 24);
                __m256 _val31 = _mm256_broadcast_ss(tmpptr + 25);
                __m256 _val32 = _mm256_broadcast_ss(tmpptr + 26);
                __m256 _val33 = _mm256_broadcast_ss(tmpptr + 27);
                __m256 _val34 = _mm256_broadcast_ss(tmpptr + 28);
                __m256 _val35 = _mm256_broadcast_ss(tmpptr + 29);
                __m256 _val36 = _mm256_broadcast_ss(tmpptr + 30);
                __m256 _val37 = _mm256_broadcast_ss(tmpptr + 31);

                _sum2 = _mm256_fmadd_ps(_w0, _val20, _sum2);
                _sum2 = _mm256_fmadd_ps(_w1, _val21, _sum2);
                _sum2 = _mm256_fmadd_ps(_w2, _val22, _sum2);
                _sum2 = _mm256_fmadd_ps(_w3, _val23, _sum2);
                _sum2 = _mm256_fmadd_ps(_w4, _val24, _sum2);
                _sum2 = _mm256_fmadd_ps(_w5, _val25, _sum2);
                _sum2 = _mm256_fmadd_ps(_w6, _val26, _sum2);
                _sum2 = _mm256_fmadd_ps(_w7, _val27, _sum2);
                _sum3 = _mm256_fmadd_ps(_w0, _val30, _sum3);
                _sum3 = _mm256_fmadd_ps(_w1, _val31, _sum3);
                _sum3 = _mm256_fmadd_ps(_w2, _val32, _sum3);
                _sum3 = _mm256_fmadd_ps(_w3, _val33, _sum3);
                _sum3 = _mm256_fmadd_ps(_w4, _val34, _sum3);
                _sum3 = _mm256_fmadd_ps(_w5, _val35, _sum3);
                _sum3 = _mm256_fmadd_ps(_w6, _val36, _sum3);
                _sum3 = _mm256_fmadd_ps(_w7, _val37, _sum3);

                tmpptr += 32;

                kptr += 64;
            }
            _mm256_storeu_ps(outptr, _sum0);
            _mm256_storeu_ps(outptr + 8, _sum1);
            _mm256_storeu_ps(outptr + 16, _sum2);
            _mm256_storeu_ps(outptr + 24, _sum3);

            outptr += 32;
        }
        for (; i + 1 < size; i += 2)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);

            __m256 _sum0 = _bias0;
            __m256 _sum1 = _bias0;

            const float* kptr = (const float*)kernel + p * inch * 64;
            for (int q = 0; q < inch; q++)
            {
                __m256 _val00 = _mm256_broadcast_ss(tmpptr);
                __m256 _val01 = _mm256_broadcast_ss(tmpptr + 1);
                __m256 _val02 = _mm256_broadcast_ss(tmpptr + 2);
                __m256 _val03 = _mm256_broadcast_ss(tmpptr + 3);
                __m256 _val04 = _mm256_broadcast_ss(tmpptr + 4);
                __m256 _val05 = _mm256_broadcast_ss(tmpptr + 5);
                __m256 _val06 = _mm256_broadcast_ss(tmpptr + 6);
                __m256 _val07 = _mm256_broadcast_ss(tmpptr + 7);
                __m256 _val10 = _mm256_broadcast_ss(tmpptr + 8);
                __m256 _val11 = _mm256_broadcast_ss(tmpptr + 9);
                __m256 _val12 = _mm256_broadcast_ss(tmpptr + 10);
                __m256 _val13 = _mm256_broadcast_ss(tmpptr + 11);
                __m256 _val14 = _mm256_broadcast_ss(tmpptr + 12);
                __m256 _val15 = _mm256_broadcast_ss(tmpptr + 13);
                __m256 _val16 = _mm256_broadcast_ss(tmpptr + 14);
                __m256 _val17 = _mm256_broadcast_ss(tmpptr + 15);

                __m256 _w0 = _mm256_loadu_ps(kptr);
                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                __m256 _w7 = _mm256_loadu_ps(kptr + 56);

                _sum0 = _mm256_fmadd_ps(_w0, _val00, _sum0);
                _sum0 = _mm256_fmadd_ps(_w1, _val01, _sum0);
                _sum0 = _mm256_fmadd_ps(_w2, _val02, _sum0);
                _sum0 = _mm256_fmadd_ps(_w3, _val03, _sum0);
                _sum0 = _mm256_fmadd_ps(_w4, _val04, _sum0);
                _sum0 = _mm256_fmadd_ps(_w5, _val05, _sum0);
                _sum0 = _mm256_fmadd_ps(_w6, _val06, _sum0);
                _sum0 = _mm256_fmadd_ps(_w7, _val07, _sum0);
                _sum1 = _mm256_fmadd_ps(_w0, _val10, _sum1);
                _sum1 = _mm256_fmadd_ps(_w1, _val11, _sum1);
                _sum1 = _mm256_fmadd_ps(_w2, _val12, _sum1);
                _sum1 = _mm256_fmadd_ps(_w3, _val13, _sum1);
                _sum1 = _mm256_fmadd_ps(_w4, _val14, _sum1);
                _sum1 = _mm256_fmadd_ps(_w5, _val15, _sum1);
                _sum1 = _mm256_fmadd_ps(_w6, _val16, _sum1);
                _sum1 = _mm256_fmadd_ps(_w7, _val17, _sum1);

                tmpptr += 16;

                kptr += 64;
            }
            _mm256_storeu_ps(outptr, _sum0);
            _mm256_storeu_ps(outptr + 8, _sum1);

            outptr += 16;
        }

        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            __m256 _sum = _bias0;
            const float* kptr = (const float*)kernel + p * inch * 64;
            for (int q = 0; q < inch; q++)
            {
                __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);
                __m256 _val4 = _mm256_broadcast_ss(tmpptr + 4);
                __m256 _val5 = _mm256_broadcast_ss(tmpptr + 5);
                __m256 _val6 = _mm256_broadcast_ss(tmpptr + 6);
                __m256 _val7 = _mm256_broadcast_ss(tmpptr + 7);

                __m256 _w0 = _mm256_loadu_ps(kptr);
                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                __m256 _w7 = _mm256_loadu_ps(kptr + 56);

                _sum = _mm256_fmadd_ps(_w0, _val0, _sum);
                _sum = _mm256_fmadd_ps(_w1, _val1, _sum);
                _sum = _mm256_fmadd_ps(_w2, _val2, _sum);
                _sum = _mm256_fmadd_ps(_w3, _val3, _sum);
                _sum = _mm256_fmadd_ps(_w4, _val4, _sum);
                _sum = _mm256_fmadd_ps(_w5, _val5, _sum);
                _sum = _mm256_fmadd_ps(_w6, _val6, _sum);
                _sum = _mm256_fmadd_ps(_w7, _val7, _sum);

                tmpptr += 8;

                kptr += 64;
            }
            _mm256_storeu_ps(outptr, _sum);

            outptr += 8;
        }
    }
}

static void conv1x1s2_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * 8;

    Mat bottom_blob_shrinked;
    bottom_blob_shrinked.create(outw, outh, channels, elemsize, elempack, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < channels; p++)
    {
        const float* r0 = bottom_blob.channel(p);
        float* outptr = bottom_blob_shrinked.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                __m256 _v = _mm256_loadu_ps(r0);
                _mm256_storeu_ps(outptr, _v);

                r0 += 16;
                outptr += 8;
            }

            r0 += tailstep;
        }
    }

    conv1x1s1_sgemm_pack8_avx(bottom_blob_shrinked, top_blob, kernel, _bias, opt);
}