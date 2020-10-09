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

static void conv1x1s1_sgemm_transform_kernel_pack8_avx(const Mat& kernel, Mat& weight_data_pack8, int num_input, int num_output)
{
    // src = kw-kh-inch-outch
    // dst = 8b-8a-kw-kh-inch/8a-outch/8b
    Mat weight_data_r2 = kernel.reshape(1, num_input, num_output);

    weight_data_pack8.create(1, num_input / 8, num_output / 8, (size_t)4 * 64, 64);

    for (int q = 0; q + 7 < num_output; q += 8)
    {
        const Mat k0 = weight_data_r2.channel(q);
        const Mat k1 = weight_data_r2.channel(q + 1);
        const Mat k2 = weight_data_r2.channel(q + 2);
        const Mat k3 = weight_data_r2.channel(q + 3);
        const Mat k4 = weight_data_r2.channel(q + 4);
        const Mat k5 = weight_data_r2.channel(q + 5);
        const Mat k6 = weight_data_r2.channel(q + 6);
        const Mat k7 = weight_data_r2.channel(q + 7);

        Mat g0 = weight_data_pack8.channel(q / 8);

        for (int p = 0; p + 7 < num_input; p += 8)
        {
            const float* k00 = k0.row(p);
            const float* k01 = k0.row(p + 1);
            const float* k02 = k0.row(p + 2);
            const float* k03 = k0.row(p + 3);
            const float* k04 = k0.row(p + 4);
            const float* k05 = k0.row(p + 5);
            const float* k06 = k0.row(p + 6);
            const float* k07 = k0.row(p + 7);

            const float* k10 = k1.row(p);
            const float* k11 = k1.row(p + 1);
            const float* k12 = k1.row(p + 2);
            const float* k13 = k1.row(p + 3);
            const float* k14 = k1.row(p + 4);
            const float* k15 = k1.row(p + 5);
            const float* k16 = k1.row(p + 6);
            const float* k17 = k1.row(p + 7);

            const float* k20 = k2.row(p);
            const float* k21 = k2.row(p + 1);
            const float* k22 = k2.row(p + 2);
            const float* k23 = k2.row(p + 3);
            const float* k24 = k2.row(p + 4);
            const float* k25 = k2.row(p + 5);
            const float* k26 = k2.row(p + 6);
            const float* k27 = k2.row(p + 7);

            const float* k30 = k3.row(p);
            const float* k31 = k3.row(p + 1);
            const float* k32 = k3.row(p + 2);
            const float* k33 = k3.row(p + 3);
            const float* k34 = k3.row(p + 4);
            const float* k35 = k3.row(p + 5);
            const float* k36 = k3.row(p + 6);
            const float* k37 = k3.row(p + 7);

            const float* k40 = k4.row(p);
            const float* k41 = k4.row(p + 1);
            const float* k42 = k4.row(p + 2);
            const float* k43 = k4.row(p + 3);
            const float* k44 = k4.row(p + 4);
            const float* k45 = k4.row(p + 5);
            const float* k46 = k4.row(p + 6);
            const float* k47 = k4.row(p + 7);

            const float* k50 = k5.row(p);
            const float* k51 = k5.row(p + 1);
            const float* k52 = k5.row(p + 2);
            const float* k53 = k5.row(p + 3);
            const float* k54 = k5.row(p + 4);
            const float* k55 = k5.row(p + 5);
            const float* k56 = k5.row(p + 6);
            const float* k57 = k5.row(p + 7);

            const float* k60 = k6.row(p);
            const float* k61 = k6.row(p + 1);
            const float* k62 = k6.row(p + 2);
            const float* k63 = k6.row(p + 3);
            const float* k64 = k6.row(p + 4);
            const float* k65 = k6.row(p + 5);
            const float* k66 = k6.row(p + 6);
            const float* k67 = k6.row(p + 7);

            const float* k70 = k7.row(p);
            const float* k71 = k7.row(p + 1);
            const float* k72 = k7.row(p + 2);
            const float* k73 = k7.row(p + 3);
            const float* k74 = k7.row(p + 4);
            const float* k75 = k7.row(p + 5);
            const float* k76 = k7.row(p + 6);
            const float* k77 = k7.row(p + 7);

            float* g00 = g0.row(p / 8);
            g00[0] = k00[0];
            g00[1] = k10[0];
            g00[2] = k20[0];
            g00[3] = k30[0];
            g00[4] = k40[0];
            g00[5] = k50[0];
            g00[6] = k60[0];
            g00[7] = k70[0];
            g00 += 8;
            g00[0] = k01[0];
            g00[1] = k11[0];
            g00[2] = k21[0];
            g00[3] = k31[0];
            g00[4] = k41[0];
            g00[5] = k51[0];
            g00[6] = k61[0];
            g00[7] = k71[0];

            g00 += 8;
            g00[0] = k02[0];
            g00[1] = k12[0];
            g00[2] = k22[0];
            g00[3] = k32[0];
            g00[4] = k42[0];
            g00[5] = k52[0];
            g00[6] = k62[0];
            g00[7] = k72[0];

            g00 += 8;
            g00[0] = k03[0];
            g00[1] = k13[0];
            g00[2] = k23[0];
            g00[3] = k33[0];
            g00[4] = k43[0];
            g00[5] = k53[0];
            g00[6] = k63[0];
            g00[7] = k73[0];

            g00 += 8;
            g00[0] = k04[0];
            g00[1] = k14[0];
            g00[2] = k24[0];
            g00[3] = k34[0];
            g00[4] = k44[0];
            g00[5] = k54[0];
            g00[6] = k64[0];
            g00[7] = k74[0];

            g00 += 8;
            g00[0] = k05[0];
            g00[1] = k15[0];
            g00[2] = k25[0];
            g00[3] = k35[0];
            g00[4] = k45[0];
            g00[5] = k55[0];
            g00[6] = k65[0];
            g00[7] = k75[0];

            g00 += 8;
            g00[0] = k06[0];
            g00[1] = k16[0];
            g00[2] = k26[0];
            g00[3] = k36[0];
            g00[4] = k46[0];
            g00[5] = k56[0];
            g00[6] = k66[0];
            g00[7] = k76[0];

            g00 += 8;
            g00[0] = k07[0];
            g00[1] = k17[0];
            g00[2] = k27[0];
            g00[3] = k37[0];
            g00[4] = k47[0];
            g00[5] = k57[0];
            g00[6] = k67[0];
            g00[7] = k77[0];

            g00 += 8;
        }
    }
}

static void conv1x1s1_sgemm_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int size = w * h;

    const float* bias = _bias;
    // interleave
    Mat tmp(12, inch, size / 12 + (size % 12) / 8 + (size % 12 % 8) / 4 + (size % 12 % 4) / 2 + size % 12 % 2, elemsize, elempack, opt.workspace_allocator);
    {
        int nn_size = size / 12;
        int remain_size_start = nn_size * 12;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 12;
            const float* img0 = bottom_blob.channel(0);
            img0 += i * 8;

            float* tmpptr = tmp.channel(i / 12);

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
                __m256 _r8 = _mm256_loadu_ps(img0 + 64);
                __m256 _r9 = _mm256_loadu_ps(img0 + 72);
                __m256 _r10 = _mm256_loadu_ps(img0 + 80);
                __m256 _r11 = _mm256_loadu_ps(img0 + 88);
                _mm256_storeu_ps(tmpptr, _r0);
                _mm256_storeu_ps(tmpptr + 8, _r1);
                _mm256_storeu_ps(tmpptr + 16, _r2);
                _mm256_storeu_ps(tmpptr + 24, _r3);
                _mm256_storeu_ps(tmpptr + 32, _r4);
                _mm256_storeu_ps(tmpptr + 40, _r5);
                _mm256_storeu_ps(tmpptr + 48, _r6);
                _mm256_storeu_ps(tmpptr + 56, _r7);
                _mm256_storeu_ps(tmpptr + 64, _r8);
                _mm256_storeu_ps(tmpptr + 72, _r9);
                _mm256_storeu_ps(tmpptr + 80, _r10);
                _mm256_storeu_ps(tmpptr + 88, _r11);

                tmpptr += 96;
                img0 += bottom_blob.cstep * 8;
            }
        }
        nn_size = (size - remain_size_start) >> 3;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            const float* img0 = bottom_blob.channel(0);
            img0 += i * 8;

            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);

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

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            const float* img0 = bottom_blob.channel(0);
            img0 += i * 8;
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

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
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

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
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
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
        for (; i + 11 < size; i += 12)
        {
            const float* tmpptr = tmp.channel(i / 12);

            __m256 _sum0 = _bias0;
            __m256 _sum1 = _bias0;
            __m256 _sum2 = _bias0;
            __m256 _sum3 = _bias0;
            __m256 _sum4 = _bias0;
            __m256 _sum5 = _bias0;
            __m256 _sum6 = _bias0;
            __m256 _sum7 = _bias0;
            __m256 _sum8 = _bias0;
            __m256 _sum9 = _bias0;
            __m256 _sum10 = _bias0;
            __m256 _sum11 = _bias0;

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

                __m256 _val80 = _mm256_broadcast_ss(tmpptr + 64);
                __m256 _val81 = _mm256_broadcast_ss(tmpptr + 65);
                __m256 _val82 = _mm256_broadcast_ss(tmpptr + 66);
                __m256 _val83 = _mm256_broadcast_ss(tmpptr + 67);
                __m256 _val84 = _mm256_broadcast_ss(tmpptr + 68);
                __m256 _val85 = _mm256_broadcast_ss(tmpptr + 69);
                __m256 _val86 = _mm256_broadcast_ss(tmpptr + 70);
                __m256 _val87 = _mm256_broadcast_ss(tmpptr + 71);
                __m256 _val90 = _mm256_broadcast_ss(tmpptr + 72);
                __m256 _val91 = _mm256_broadcast_ss(tmpptr + 73);
                __m256 _val92 = _mm256_broadcast_ss(tmpptr + 74);
                __m256 _val93 = _mm256_broadcast_ss(tmpptr + 75);
                __m256 _val94 = _mm256_broadcast_ss(tmpptr + 76);
                __m256 _val95 = _mm256_broadcast_ss(tmpptr + 77);
                __m256 _val96 = _mm256_broadcast_ss(tmpptr + 78);
                __m256 _val97 = _mm256_broadcast_ss(tmpptr + 79);

                _sum8 = _mm256_fmadd_ps(_w0, _val80, _sum8);
                _sum8 = _mm256_fmadd_ps(_w1, _val81, _sum8);
                _sum8 = _mm256_fmadd_ps(_w2, _val82, _sum8);
                _sum8 = _mm256_fmadd_ps(_w3, _val83, _sum8);
                _sum8 = _mm256_fmadd_ps(_w4, _val84, _sum8);
                _sum8 = _mm256_fmadd_ps(_w5, _val85, _sum8);
                _sum8 = _mm256_fmadd_ps(_w6, _val86, _sum8);
                _sum8 = _mm256_fmadd_ps(_w7, _val87, _sum8);
                _sum9 = _mm256_fmadd_ps(_w0, _val90, _sum9);
                _sum9 = _mm256_fmadd_ps(_w1, _val91, _sum9);
                _sum9 = _mm256_fmadd_ps(_w2, _val92, _sum9);
                _sum9 = _mm256_fmadd_ps(_w3, _val93, _sum9);
                _sum9 = _mm256_fmadd_ps(_w4, _val94, _sum9);
                _sum9 = _mm256_fmadd_ps(_w5, _val95, _sum9);
                _sum9 = _mm256_fmadd_ps(_w6, _val96, _sum9);
                _sum9 = _mm256_fmadd_ps(_w7, _val97, _sum9);

                __m256 _val100 = _mm256_broadcast_ss(tmpptr + 80);
                __m256 _val101 = _mm256_broadcast_ss(tmpptr + 81);
                __m256 _val102 = _mm256_broadcast_ss(tmpptr + 82);
                __m256 _val103 = _mm256_broadcast_ss(tmpptr + 83);
                __m256 _val104 = _mm256_broadcast_ss(tmpptr + 84);
                __m256 _val105 = _mm256_broadcast_ss(tmpptr + 85);
                __m256 _val106 = _mm256_broadcast_ss(tmpptr + 86);
                __m256 _val107 = _mm256_broadcast_ss(tmpptr + 87);
                __m256 _val110 = _mm256_broadcast_ss(tmpptr + 88);
                __m256 _val111 = _mm256_broadcast_ss(tmpptr + 89);
                __m256 _val112 = _mm256_broadcast_ss(tmpptr + 90);
                __m256 _val113 = _mm256_broadcast_ss(tmpptr + 91);
                __m256 _val114 = _mm256_broadcast_ss(tmpptr + 92);
                __m256 _val115 = _mm256_broadcast_ss(tmpptr + 93);
                __m256 _val116 = _mm256_broadcast_ss(tmpptr + 94);
                __m256 _val117 = _mm256_broadcast_ss(tmpptr + 95);

                _sum10 = _mm256_fmadd_ps(_w0, _val100, _sum10);
                _sum10 = _mm256_fmadd_ps(_w1, _val101, _sum10);
                _sum10 = _mm256_fmadd_ps(_w2, _val102, _sum10);
                _sum10 = _mm256_fmadd_ps(_w3, _val103, _sum10);
                _sum10 = _mm256_fmadd_ps(_w4, _val104, _sum10);
                _sum10 = _mm256_fmadd_ps(_w5, _val105, _sum10);
                _sum10 = _mm256_fmadd_ps(_w6, _val106, _sum10);
                _sum10 = _mm256_fmadd_ps(_w7, _val107, _sum10);
                _sum11 = _mm256_fmadd_ps(_w0, _val110, _sum11);
                _sum11 = _mm256_fmadd_ps(_w1, _val111, _sum11);
                _sum11 = _mm256_fmadd_ps(_w2, _val112, _sum11);
                _sum11 = _mm256_fmadd_ps(_w3, _val113, _sum11);
                _sum11 = _mm256_fmadd_ps(_w4, _val114, _sum11);
                _sum11 = _mm256_fmadd_ps(_w5, _val115, _sum11);
                _sum11 = _mm256_fmadd_ps(_w6, _val116, _sum11);
                _sum11 = _mm256_fmadd_ps(_w7, _val117, _sum11);

                tmpptr += 96;

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
            _mm256_storeu_ps(outptr + 64, _sum8);
            _mm256_storeu_ps(outptr + 72, _sum9);
            _mm256_storeu_ps(outptr + 80, _sum10);
            _mm256_storeu_ps(outptr + 88, _sum11);

            outptr += 96;
        }
        for (; i + 7 < size; i += 8)
        {
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);

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
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

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
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

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
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
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
