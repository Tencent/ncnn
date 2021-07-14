// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv1x1s1_sgemm_transform_kernel_pack4_sse(const Mat& kernel, Mat& kernel_pack4, int inch, int outch)
{
    // interleave
    // src = inch-outch
    // dst = 4b-4a-inch/4a-outch/4b
    kernel_pack4.create(1, inch / 4, outch / 4, (size_t)4u * 16, 16);

    int q = 0;
    for (; q + 3 < outch; q += 4)
    {
        const float* k0 = (const float*)kernel + (q + 0) * inch;
        const float* k1 = (const float*)kernel + (q + 1) * inch;
        const float* k2 = (const float*)kernel + (q + 2) * inch;
        const float* k3 = (const float*)kernel + (q + 3) * inch;

        float* g0 = kernel_pack4.channel(q / 4);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            g0[0] = k0[0];
            g0[1] = k1[0];
            g0[2] = k2[0];
            g0[3] = k3[0];

            g0[4] = k0[1];
            g0[5] = k1[1];
            g0[6] = k2[1];
            g0[7] = k3[1];

            g0[8] = k0[2];
            g0[9] = k1[2];
            g0[10] = k2[2];
            g0[11] = k3[2];

            g0[12] = k0[3];
            g0[13] = k1[3];
            g0[14] = k2[3];
            g0[15] = k3[3];

            k0 += 4;
            k1 += 4;
            k2 += 4;
            k3 += 4;
            g0 += 16;
        }
    }
}

static void conv1x1s1_sgemm_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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
    Mat tmp(4, inch, size / 4 + (size % 4) / 2 + size % 2, elemsize, elempack, opt.workspace_allocator);
    {
        int nn_size;
        int remain_size_start;

        remain_size_start = 0;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            const float* img0 = bottom_blob.channel(0);
            img0 += i * 4;

            float* tmpptr = tmp.channel(i / 4);

            for (int q = 0; q < inch; q++)
            {
                __m128 _r0 = _mm_loadu_ps(img0);
                __m128 _r1 = _mm_loadu_ps(img0 + 4);
                __m128 _r2 = _mm_loadu_ps(img0 + 8);
                __m128 _r3 = _mm_loadu_ps(img0 + 12);
                _mm_storeu_ps(tmpptr, _r0);
                _mm_storeu_ps(tmpptr + 4, _r1);
                _mm_storeu_ps(tmpptr + 8, _r2);
                _mm_storeu_ps(tmpptr + 12, _r3);

                tmpptr += 16;
                img0 += bottom_blob.cstep * 4;
            }
        }

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            const float* img0 = bottom_blob.channel(0);
            img0 += i * 4;

            float* tmpptr = tmp.channel(i / 4 + (i % 4) / 2);

            for (int q = 0; q < inch; q++)
            {
                __m128 _r0 = _mm_loadu_ps(img0);
                __m128 _r1 = _mm_loadu_ps(img0 + 4);
                _mm_storeu_ps(tmpptr, _r0);
                _mm_storeu_ps(tmpptr + 4, _r1);

                tmpptr += 8;
                img0 += bottom_blob.cstep * 4;
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            const float* img0 = bottom_blob.channel(0);
            img0 += i * 4;

            float* tmpptr = tmp.channel(i / 4 + (i % 4) / 2 + i % 2);

            for (int q = 0; q < inch; q++)
            {
                __m128 _r0 = _mm_loadu_ps(img0);
                _mm_storeu_ps(tmpptr, _r0);

                tmpptr += 4;
                img0 += bottom_blob.cstep * 4;
            }
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p * 4 : zeros;

        int i = 0;
        for (; i + 3 < size; i += 4)
        {
            float* tmpptr = tmp.channel(i / 4);
            const float* kptr0 = (const float*)kernel.channel(p);

            __m128 _sum0 = _mm_loadu_ps(biasptr);
            __m128 _sum1 = _mm_loadu_ps(biasptr);
            __m128 _sum2 = _mm_loadu_ps(biasptr);
            __m128 _sum3 = _mm_loadu_ps(biasptr);

            for (int q = 0; q < inch; q++)
            {
                __m128 _val00 = _mm_load1_ps(tmpptr);
                __m128 _val01 = _mm_load1_ps(tmpptr + 1);
                __m128 _val02 = _mm_load1_ps(tmpptr + 2);
                __m128 _val03 = _mm_load1_ps(tmpptr + 3);
                __m128 _val10 = _mm_load1_ps(tmpptr + 4);
                __m128 _val11 = _mm_load1_ps(tmpptr + 5);
                __m128 _val12 = _mm_load1_ps(tmpptr + 6);
                __m128 _val13 = _mm_load1_ps(tmpptr + 7);
                __m128 _val20 = _mm_load1_ps(tmpptr + 8);
                __m128 _val21 = _mm_load1_ps(tmpptr + 9);
                __m128 _val22 = _mm_load1_ps(tmpptr + 10);
                __m128 _val23 = _mm_load1_ps(tmpptr + 11);
                __m128 _val30 = _mm_load1_ps(tmpptr + 12);
                __m128 _val31 = _mm_load1_ps(tmpptr + 13);
                __m128 _val32 = _mm_load1_ps(tmpptr + 14);
                __m128 _val33 = _mm_load1_ps(tmpptr + 15);

                __m128 _w0 = _mm_load_ps(kptr0);
                __m128 _w1 = _mm_load_ps(kptr0 + 4);
                __m128 _w2 = _mm_load_ps(kptr0 + 8);
                __m128 _w3 = _mm_load_ps(kptr0 + 12);

#if __AVX__
                _sum0 = _mm_comp_fmadd_ps(_w0, _val00, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_w1, _val01, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_w2, _val02, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_w3, _val03, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_w0, _val10, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_w1, _val11, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_w2, _val12, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_w3, _val13, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_w0, _val20, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_w1, _val21, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_w2, _val22, _sum2);
                _sum2 = _mm_comp_fmadd_ps(_w3, _val23, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_w0, _val30, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_w1, _val31, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_w2, _val32, _sum3);
                _sum3 = _mm_comp_fmadd_ps(_w3, _val33, _sum3);
#else
                _sum0 = _mm_add_ps(_mm_mul_ps(_w0, _val00), _sum0);
                _sum0 = _mm_add_ps(_mm_mul_ps(_w1, _val01), _sum0);
                _sum0 = _mm_add_ps(_mm_mul_ps(_w2, _val02), _sum0);
                _sum0 = _mm_add_ps(_mm_mul_ps(_w3, _val03), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_w0, _val10), _sum1);
                _sum1 = _mm_add_ps(_mm_mul_ps(_w1, _val11), _sum1);
                _sum1 = _mm_add_ps(_mm_mul_ps(_w2, _val12), _sum1);
                _sum1 = _mm_add_ps(_mm_mul_ps(_w3, _val13), _sum1);
                _sum2 = _mm_add_ps(_mm_mul_ps(_w0, _val20), _sum2);
                _sum2 = _mm_add_ps(_mm_mul_ps(_w1, _val21), _sum2);
                _sum2 = _mm_add_ps(_mm_mul_ps(_w2, _val22), _sum2);
                _sum2 = _mm_add_ps(_mm_mul_ps(_w3, _val23), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_w0, _val30), _sum3);
                _sum3 = _mm_add_ps(_mm_mul_ps(_w1, _val31), _sum3);
                _sum3 = _mm_add_ps(_mm_mul_ps(_w2, _val32), _sum3);
                _sum3 = _mm_add_ps(_mm_mul_ps(_w3, _val33), _sum3);
#endif

                tmpptr += 16;
                kptr0 += 16;
            }

            _mm_store_ps(outptr0, _sum0);
            _mm_store_ps(outptr0 + 4, _sum1);
            _mm_store_ps(outptr0 + 8, _sum2);
            _mm_store_ps(outptr0 + 12, _sum3);
            outptr0 += 16;
        }
        for (; i + 1 < size; i += 2)
        {
            float* tmpptr = tmp.channel(i / 4 + (i % 4) / 2);
            const float* kptr0 = (const float*)kernel.channel(p);

            __m128 _sum0 = _mm_loadu_ps(biasptr);
            __m128 _sum1 = _mm_loadu_ps(biasptr);

            for (int q = 0; q < inch; q++)
            {
                __m128 _val00 = _mm_load1_ps(tmpptr);
                __m128 _val01 = _mm_load1_ps(tmpptr + 1);
                __m128 _val02 = _mm_load1_ps(tmpptr + 2);
                __m128 _val03 = _mm_load1_ps(tmpptr + 3);
                __m128 _val10 = _mm_load1_ps(tmpptr + 4);
                __m128 _val11 = _mm_load1_ps(tmpptr + 5);
                __m128 _val12 = _mm_load1_ps(tmpptr + 6);
                __m128 _val13 = _mm_load1_ps(tmpptr + 7);

                __m128 _w0 = _mm_load_ps(kptr0);
                __m128 _w1 = _mm_load_ps(kptr0 + 4);
                __m128 _w2 = _mm_load_ps(kptr0 + 8);
                __m128 _w3 = _mm_load_ps(kptr0 + 12);

#if __AVX__
                _sum0 = _mm_comp_fmadd_ps(_w0, _val00, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_w1, _val01, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_w2, _val02, _sum0);
                _sum0 = _mm_comp_fmadd_ps(_w3, _val03, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_w0, _val10, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_w1, _val11, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_w2, _val12, _sum1);
                _sum1 = _mm_comp_fmadd_ps(_w3, _val13, _sum1);
#else
                _sum0 = _mm_add_ps(_mm_mul_ps(_w0, _val00), _sum0);
                _sum0 = _mm_add_ps(_mm_mul_ps(_w1, _val01), _sum0);
                _sum0 = _mm_add_ps(_mm_mul_ps(_w2, _val02), _sum0);
                _sum0 = _mm_add_ps(_mm_mul_ps(_w3, _val03), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_w0, _val10), _sum1);
                _sum1 = _mm_add_ps(_mm_mul_ps(_w1, _val11), _sum1);
                _sum1 = _mm_add_ps(_mm_mul_ps(_w2, _val12), _sum1);
                _sum1 = _mm_add_ps(_mm_mul_ps(_w3, _val13), _sum1);
#endif

                tmpptr += 8;
                kptr0 += 16;
            }

            _mm_store_ps(outptr0, _sum0);
            _mm_store_ps(outptr0 + 4, _sum1);
            outptr0 += 8;
        }
        for (; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 4 + (i % 4) / 2 + i % 2);
            const float* kptr0 = (const float*)kernel.channel(p);

            __m128 _sum = _mm_loadu_ps(biasptr);

            for (int q = 0; q < inch; q++)
            {
                __m128 _val0 = _mm_load1_ps(tmpptr);
                __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                __m128 _val3 = _mm_load1_ps(tmpptr + 3);

                __m128 _w0 = _mm_load_ps(kptr0);
                __m128 _w1 = _mm_load_ps(kptr0 + 4);
                __m128 _w2 = _mm_load_ps(kptr0 + 8);
                __m128 _w3 = _mm_load_ps(kptr0 + 12);

#if __AVX__
                _sum = _mm_comp_fmadd_ps(_w0, _val0, _sum);
                _sum = _mm_comp_fmadd_ps(_w1, _val1, _sum);
                _sum = _mm_comp_fmadd_ps(_w2, _val2, _sum);
                _sum = _mm_comp_fmadd_ps(_w3, _val3, _sum);
#else
                _sum = _mm_add_ps(_mm_mul_ps(_w0, _val0), _sum);
                _sum = _mm_add_ps(_mm_mul_ps(_w1, _val1), _sum);
                _sum = _mm_add_ps(_mm_mul_ps(_w2, _val2), _sum);
                _sum = _mm_add_ps(_mm_mul_ps(_w3, _val3), _sum);
#endif

                tmpptr += 4;
                kptr0 += 16;
            }

            _mm_store_ps(outptr0, _sum);
            outptr0 += 4;
        }
    }

    //     // NOTE sgemm
    //     for (; p<outch; p++)
    //     {
    //         Mat out0 = top_blob.channel(p);
    //
    //         const float bias0 = bias ? bias[p] : 0.f;
    //
    //         float* outptr0 = out0;
    //
    //         for (int i=0; i<size; i++)
    //         {
    //             float sum = bias0;
    //
    //             const float* kptr = _kernel.channel(p);
    //
    //             for (int q=0; q<inch; q++)
    //             {
    //                 const float* img0 = bottom_blob.channel(q);
    //
    //                 sum += img0[i] * kptr[0];
    //                 kptr ++;
    //             }
    //
    //             outptr0[i] = sum;
    //         }
    //     }
}

static void conv1x1s2_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * 4;

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
                __m128 _v = _mm_load_ps(r0);
                _mm_store_ps(outptr, _v);

                r0 += 8;
                outptr += 4;
            }

            r0 += tailstep;
        }
    }

    conv1x1s1_sgemm_pack4_sse(bottom_blob_shrinked, top_blob, kernel, _bias, opt);
}
