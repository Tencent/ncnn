// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
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

#if __AVX__
static void conv_im2col_sgemm_transform_kernel_sse(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_size)
{
    const float* kernel = _kernel;

    // kernel memory packed 8 x 8
    kernel_tm.create(8 * kernel_size, inch, outch / 8 + (outch % 8) / 4 + outch % 4);

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 3;
    remain_outch_start = nn_outch << 3;

    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        const float* k0 = kernel + (p + 0) * inch * kernel_size;
        const float* k1 = kernel + (p + 1) * inch * kernel_size;
        const float* k2 = kernel + (p + 2) * inch * kernel_size;
        const float* k3 = kernel + (p + 3) * inch * kernel_size;
        const float* k4 = kernel + (p + 4) * inch * kernel_size;
        const float* k5 = kernel + (p + 5) * inch * kernel_size;
        const float* k6 = kernel + (p + 6) * inch * kernel_size;
        const float* k7 = kernel + (p + 7) * inch * kernel_size;

        float* ktmp = kernel_tm.channel(p / 8);

        for (int q = 0; q < inch * kernel_size; q++)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];
            ktmp[4] = k4[0];
            ktmp[5] = k5[0];
            ktmp[6] = k6[0];
            ktmp[7] = k7[0];
            ktmp += 8;

            k0 += 1;
            k1 += 1;
            k2 += 1;
            k3 += 1;
            k4 += 1;
            k5 += 1;
            k6 += 1;
            k7 += 1;
        }
    }

    nn_outch = (outch - remain_outch_start) >> 2;

    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        const float* k0 = kernel + (p + 0) * inch * kernel_size;
        const float* k1 = kernel + (p + 1) * inch * kernel_size;
        const float* k2 = kernel + (p + 2) * inch * kernel_size;
        const float* k3 = kernel + (p + 3) * inch * kernel_size;

        float* ktmp = kernel_tm.channel(p / 8 + (p % 8) / 4);

        for (int q = 0; q < inch * kernel_size; q++)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];
            ktmp += 4;

            k0 += 1;
            k1 += 1;
            k2 += 1;
            k3 += 1;
        }
    }

    remain_outch_start += nn_outch << 2;

    for (int p = remain_outch_start; p < outch; p++)
    {
        const float* k0 = kernel + (p + 0) * inch * kernel_size;

        float* ktmp = kernel_tm.channel(p / 8 + (p % 8) / 4 + p % 4);

        for (int q = 0; q < inch * kernel_size; q++)
        {
            ktmp[0] = k0[0];
            ktmp++;
            k0++;
        }
    }
}

static void conv_im2col_sgemm_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias,
                                  const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

    // im2col
    Mat bottom_im2col(outw * outh, kernel_h * kernel_w * inch, elemsize, opt.workspace_allocator);
    {
        const int stride = kernel_h * kernel_w * outw * outh;
        float* ret = (float*)bottom_im2col;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const float* input = bottom_blob.channel(p);
            int retID = stride * p;
            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            int row = u + i * stride_h;
                            int col = v + j * stride_w;
                            int index = row * w + col;
                            ret[retID] = input[index];
                            retID++;
                        }
                    }
                }
            }
        }
    }

    int kernel_size = kernel_w * kernel_h;
    int out_size = outw * outh;

    // bottom_im2col memory packed 8 x 8
    Mat bottom_tm(8 * kernel_size, inch, out_size / 8 + out_size % 8, elemsize, opt.workspace_allocator);
    {
        int nn_size = out_size >> 3;
        int remain_size_start = nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 8;

            const float* img0 = bottom_im2col.channel(0);
            img0 += i;

            float* tmpptr = bottom_tm.channel(i / 8);

            for (int q = 0; q < inch * kernel_size; q++)
            {
#if __AVX__
                _mm256_storeu_ps(tmpptr, _mm256_loadu_ps(img0));
#else
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];
                tmpptr[4] = img0[4];
                tmpptr[5] = img0[5];
                tmpptr[6] = img0[6];
                tmpptr[7] = img0[7];
#endif // __SSE__
                tmpptr += 8;
                img0 += out_size;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < out_size; i++)
        {
            const float* img0 = bottom_im2col.channel(0);
            img0 += i;

            float* tmpptr = bottom_tm.channel(i / 8 + i % 8);

            for (int q = 0; q < inch * kernel_size; q++)
            {
                tmpptr[0] = img0[0];

                tmpptr += 1;
                img0 += out_size;
            }
        }
    }

    // sgemm(int M, int N, int L, float* A, float* B, float* C)
    {
        //int M = outch;                    // outch
        int N = outw * outh;                // outsize or out stride
        int L = kernel_w * kernel_h * inch; // ksize * inch

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 3;
        remain_outch_start = nn_outch << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int i = pp * 8;

            float* output0 = top_blob.channel(i);
            float* output1 = top_blob.channel(i + 1);
            float* output2 = top_blob.channel(i + 2);
            float* output3 = top_blob.channel(i + 3);
            float* output4 = top_blob.channel(i + 4);
            float* output5 = top_blob.channel(i + 5);
            float* output6 = top_blob.channel(i + 6);
            float* output7 = top_blob.channel(i + 7);

            const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + i : zeros;

            int j = 0;
            for (; j + 7 < N; j = j + 8)
            {
                const float* vb = bottom_tm.channel(j / 8);
                const float* va = kernel_tm.channel(i / 8);
#if __AVX__
                __m256 _sum0 = _mm256_broadcast_ss(biasptr);
                __m256 _sum1 = _mm256_broadcast_ss(biasptr + 1);
                __m256 _sum2 = _mm256_broadcast_ss(biasptr + 2);
                __m256 _sum3 = _mm256_broadcast_ss(biasptr + 3);
                __m256 _sum4 = _mm256_broadcast_ss(biasptr + 4);
                __m256 _sum5 = _mm256_broadcast_ss(biasptr + 5);
                __m256 _sum6 = _mm256_broadcast_ss(biasptr + 6);
                __m256 _sum7 = _mm256_broadcast_ss(biasptr + 7);

                int k = 0;
                for (; k + 3 < L; k = k + 4)
                {
                    // k0
                    __m256 _va0 = _mm256_broadcast_ss(va);
                    __m256 _va1 = _mm256_broadcast_ss(va + 1);
                    __m256 _va2 = _mm256_broadcast_ss(va + 2);
                    __m256 _va3 = _mm256_broadcast_ss(va + 3);
                    __m256 _vb0 = _mm256_loadu_ps(vb);
                    __m256 _vb1 = _mm256_loadu_ps(vb + 8);
                    __m256 _vb2 = _mm256_loadu_ps(vb + 16);
                    __m256 _vb3 = _mm256_loadu_ps(vb + 24);
                    _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0); // sum0 = (a00-a07) * k00
                    _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1); // sum1 = (a00-a07) * k10
                    _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2); // sum2 = (a00-a07) * k20
                    _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3); // sum3 = (a00-a07) * k30
                    _va0 = _mm256_broadcast_ss(va + 4);
                    _va1 = _mm256_broadcast_ss(va + 5);
                    _va2 = _mm256_broadcast_ss(va + 6);
                    _va3 = _mm256_broadcast_ss(va + 7);
                    _sum4 = _mm256_fmadd_ps(_vb0, _va0, _sum4); // sum4 = (a00-a07) * k40
                    _sum5 = _mm256_fmadd_ps(_vb0, _va1, _sum5); // sum5 = (a00-a07) * k50
                    _sum6 = _mm256_fmadd_ps(_vb0, _va2, _sum6); // sum6 = (a00-a07) * k60
                    _sum7 = _mm256_fmadd_ps(_vb0, _va3, _sum7); // sum7 = (a00-a07) * k70

                    va += 8;

                    // k1
                    _va0 = _mm256_broadcast_ss(va);
                    _va1 = _mm256_broadcast_ss(va + 1);
                    _va2 = _mm256_broadcast_ss(va + 2);
                    _va3 = _mm256_broadcast_ss(va + 3);
                    _sum0 = _mm256_fmadd_ps(_vb1, _va0, _sum0); // sum0 += (a10-a17) * k01
                    _sum1 = _mm256_fmadd_ps(_vb1, _va1, _sum1); // sum1 += (a10-a17) * k11
                    _sum2 = _mm256_fmadd_ps(_vb1, _va2, _sum2); // sum2 += (a10-a17) * k21
                    _sum3 = _mm256_fmadd_ps(_vb1, _va3, _sum3); // sum3 += (a10-a17) * k31
                    _va0 = _mm256_broadcast_ss(va + 4);
                    _va1 = _mm256_broadcast_ss(va + 5);
                    _va2 = _mm256_broadcast_ss(va + 6);
                    _va3 = _mm256_broadcast_ss(va + 7);
                    _sum4 = _mm256_fmadd_ps(_vb1, _va0, _sum4); // sum4 += (a10-a17) * k41
                    _sum5 = _mm256_fmadd_ps(_vb1, _va1, _sum5); // sum5 += (a10-a17) * k51
                    _sum6 = _mm256_fmadd_ps(_vb1, _va2, _sum6); // sum6 += (a10-a17) * k61
                    _sum7 = _mm256_fmadd_ps(_vb1, _va3, _sum7); // sum7 += (a10-a17) * k71

                    va += 8;

                    // k2
                    _va0 = _mm256_broadcast_ss(va);
                    _va1 = _mm256_broadcast_ss(va + 1);
                    _va2 = _mm256_broadcast_ss(va + 2);
                    _va3 = _mm256_broadcast_ss(va + 3);
                    _sum0 = _mm256_fmadd_ps(_vb2, _va0, _sum0); // sum0 += (a20-a27) * k02
                    _sum1 = _mm256_fmadd_ps(_vb2, _va1, _sum1); // sum1 += (a20-a27) * k12
                    _sum2 = _mm256_fmadd_ps(_vb2, _va2, _sum2); // sum2 += (a20-a27) * k22
                    _sum3 = _mm256_fmadd_ps(_vb2, _va3, _sum3); // sum3 += (a20-a27) * k32
                    _va0 = _mm256_broadcast_ss(va + 4);
                    _va1 = _mm256_broadcast_ss(va + 5);
                    _va2 = _mm256_broadcast_ss(va + 6);
                    _va3 = _mm256_broadcast_ss(va + 7);
                    _sum4 = _mm256_fmadd_ps(_vb2, _va0, _sum4); // sum4 += (a20-a27) * k42
                    _sum5 = _mm256_fmadd_ps(_vb2, _va1, _sum5); // sum5 += (a20-a27) * k52
                    _sum6 = _mm256_fmadd_ps(_vb2, _va2, _sum6); // sum6 += (a20-a27) * k62
                    _sum7 = _mm256_fmadd_ps(_vb2, _va3, _sum7); // sum7 += (a20-a27) * k72

                    va += 8;

                    // k3
                    _va0 = _mm256_broadcast_ss(va);
                    _va1 = _mm256_broadcast_ss(va + 1);
                    _va2 = _mm256_broadcast_ss(va + 2);
                    _va3 = _mm256_broadcast_ss(va + 3);
                    _sum0 = _mm256_fmadd_ps(_vb3, _va0, _sum0); // sum0 += (a30-a37) * k03
                    _sum1 = _mm256_fmadd_ps(_vb3, _va1, _sum1); // sum1 += (a30-a37) * k13
                    _sum2 = _mm256_fmadd_ps(_vb3, _va2, _sum2); // sum2 += (a30-a37) * k23
                    _sum3 = _mm256_fmadd_ps(_vb3, _va3, _sum3); // sum3 += (a30-a37) * k33
                    _va0 = _mm256_broadcast_ss(va + 4);
                    _va1 = _mm256_broadcast_ss(va + 5);
                    _va2 = _mm256_broadcast_ss(va + 6);
                    _va3 = _mm256_broadcast_ss(va + 7);
                    _sum4 = _mm256_fmadd_ps(_vb3, _va0, _sum4); // sum4 += (a30-a37) * k43
                    _sum5 = _mm256_fmadd_ps(_vb3, _va1, _sum5); // sum5 += (a30-a37) * k53
                    _sum6 = _mm256_fmadd_ps(_vb3, _va2, _sum6); // sum6 += (a30-a37) * k63
                    _sum7 = _mm256_fmadd_ps(_vb3, _va3, _sum7); // sum7 += (a30-a37) * k73

                    va += 8;
                    vb += 32;
                }

                for (; k < L; k++)
                {
                    // k0
                    __m256 _va0 = _mm256_broadcast_ss(va);
                    __m256 _va1 = _mm256_broadcast_ss(va + 1);
                    __m256 _va2 = _mm256_broadcast_ss(va + 2);
                    __m256 _va3 = _mm256_broadcast_ss(va + 3);
                    __m256 _va4 = _mm256_broadcast_ss(va + 4);
                    __m256 _va5 = _mm256_broadcast_ss(va + 5);
                    __m256 _va6 = _mm256_broadcast_ss(va + 6);
                    __m256 _va7 = _mm256_broadcast_ss(va + 7);
                    __m256 _vb0 = _mm256_loadu_ps(vb);
                    _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0); // sum0 = (a00-a07) * k00
                    _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1); // sum1 = (a00-a07) * k10
                    _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2); // sum2 = (a00-a07) * k20
                    _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3); // sum3 = (a00-a07) * k30
                    _sum4 = _mm256_fmadd_ps(_vb0, _va4, _sum4); // sum4 = (a00-a07) * k40
                    _sum5 = _mm256_fmadd_ps(_vb0, _va5, _sum5); // sum5 = (a00-a07) * k50
                    _sum6 = _mm256_fmadd_ps(_vb0, _va6, _sum6); // sum6 = (a00-a07) * k60
                    _sum7 = _mm256_fmadd_ps(_vb0, _va7, _sum7); // sum7 = (a00-a07) * k70

                    va += 8;
                    vb += 8;
                }

                _mm256_storeu_ps(output0, _sum0);
                _mm256_storeu_ps(output1, _sum1);
                _mm256_storeu_ps(output2, _sum2);
                _mm256_storeu_ps(output3, _sum3);
                _mm256_storeu_ps(output4, _sum4);
                _mm256_storeu_ps(output5, _sum5);
                _mm256_storeu_ps(output6, _sum6);
                _mm256_storeu_ps(output7, _sum7);
#else
                float sum0[8] = {0};
                float sum1[8] = {0};
                float sum2[8] = {0};
                float sum3[8] = {0};
                float sum4[8] = {0};
                float sum5[8] = {0};
                float sum6[8] = {0};
                float sum7[8] = {0};

                int k = 0;
                for (; k + 7 < L; k = k + 8)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        sum0[n] += va[0] * vb[n];
                        sum1[n] += va[1] * vb[n];
                        sum2[n] += va[2] * vb[n];
                        sum3[n] += va[3] * vb[n];
                        sum4[n] += va[4] * vb[n];
                        sum5[n] += va[5] * vb[n];
                        sum6[n] += va[6] * vb[n];
                        sum7[n] += va[7] * vb[n];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 8];
                        sum1[n] += va[1] * vb[n + 8];
                        sum2[n] += va[2] * vb[n + 8];
                        sum3[n] += va[3] * vb[n + 8];
                        sum4[n] += va[4] * vb[n + 8];
                        sum5[n] += va[5] * vb[n + 8];
                        sum6[n] += va[6] * vb[n + 8];
                        sum7[n] += va[7] * vb[n + 8];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 16];
                        sum1[n] += va[1] * vb[n + 16];
                        sum2[n] += va[2] * vb[n + 16];
                        sum3[n] += va[3] * vb[n + 16];
                        sum4[n] += va[4] * vb[n + 16];
                        sum5[n] += va[5] * vb[n + 16];
                        sum6[n] += va[6] * vb[n + 16];
                        sum7[n] += va[7] * vb[n + 16];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 24];
                        sum1[n] += va[1] * vb[n + 24];
                        sum2[n] += va[2] * vb[n + 24];
                        sum3[n] += va[3] * vb[n + 24];
                        sum4[n] += va[4] * vb[n + 24];
                        sum5[n] += va[5] * vb[n + 24];
                        sum6[n] += va[6] * vb[n + 24];
                        sum7[n] += va[7] * vb[n + 24];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 32];
                        sum1[n] += va[1] * vb[n + 32];
                        sum2[n] += va[2] * vb[n + 32];
                        sum3[n] += va[3] * vb[n + 32];
                        sum4[n] += va[4] * vb[n + 32];
                        sum5[n] += va[5] * vb[n + 32];
                        sum6[n] += va[6] * vb[n + 32];
                        sum7[n] += va[7] * vb[n + 32];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 40];
                        sum1[n] += va[1] * vb[n + 40];
                        sum2[n] += va[2] * vb[n + 40];
                        sum3[n] += va[3] * vb[n + 40];
                        sum4[n] += va[4] * vb[n + 40];
                        sum5[n] += va[5] * vb[n + 40];
                        sum6[n] += va[6] * vb[n + 40];
                        sum7[n] += va[7] * vb[n + 40];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 48];
                        sum1[n] += va[1] * vb[n + 48];
                        sum2[n] += va[2] * vb[n + 48];
                        sum3[n] += va[3] * vb[n + 48];
                        sum4[n] += va[4] * vb[n + 48];
                        sum5[n] += va[5] * vb[n + 48];
                        sum6[n] += va[6] * vb[n + 48];
                        sum7[n] += va[7] * vb[n + 48];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 56];
                        sum1[n] += va[1] * vb[n + 56];
                        sum2[n] += va[2] * vb[n + 56];
                        sum3[n] += va[3] * vb[n + 56];
                        sum4[n] += va[4] * vb[n + 56];
                        sum5[n] += va[5] * vb[n + 56];
                        sum6[n] += va[6] * vb[n + 56];
                        sum7[n] += va[7] * vb[n + 56];
                        va -= 56;
                    }

                    va += 64;
                    vb += 64;
                }

                for (; k < L; k++)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        sum0[n] += va[0] * vb[n];
                        sum1[n] += va[1] * vb[n];
                        sum2[n] += va[2] * vb[n];
                        sum3[n] += va[3] * vb[n];
                        sum4[n] += va[4] * vb[n];
                        sum5[n] += va[5] * vb[n];
                        sum6[n] += va[6] * vb[n];
                        sum7[n] += va[7] * vb[n];
                    }

                    va += 8;
                    vb += 8;
                }

                for (int n = 0; n < 8; n++)
                {
                    output0[n] = sum0[n] + biasptr[0];
                    output1[n] = sum1[n] + biasptr[1];
                    output2[n] = sum2[n] + biasptr[2];
                    output3[n] = sum3[n] + biasptr[3];
                    output4[n] = sum4[n] + biasptr[4];
                    output5[n] = sum5[n] + biasptr[5];
                    output6[n] = sum6[n] + biasptr[6];
                    output7[n] = sum7[n] + biasptr[7];
                }
#endif // __AVX__
                output0 += 8;
                output1 += 8;
                output2 += 8;
                output3 += 8;
                output4 += 8;
                output5 += 8;
                output6 += 8;
                output7 += 8;
            }

            for (; j < N; j++)
            {
                const float* vb = bottom_tm.channel(j / 8 + j % 8);
                const float* va = kernel_tm.channel(i / 8);

#if __AVX__
                __m256 _sum0_7 = _mm256_loadu_ps(biasptr);
                __m256 _sum0 = _mm256_set1_ps(0.0);
                __m256 _sum1 = _mm256_set1_ps(0.0);
                __m256 _sum2 = _mm256_set1_ps(0.0);
                __m256 _sum3 = _mm256_set1_ps(0.0);

                int k = 0;
                for (; k + 3 < L; k = k + 4)
                {
                    __m256 _vb0 = _mm256_broadcast_ss(vb);
                    __m256 _vb1 = _mm256_broadcast_ss(vb + 1);
                    __m256 _vb2 = _mm256_broadcast_ss(vb + 2);
                    __m256 _vb3 = _mm256_broadcast_ss(vb + 3);
                    __m256 _va0 = _mm256_loadu_ps(va);
                    __m256 _va1 = _mm256_loadu_ps(va + 8);
                    __m256 _va2 = _mm256_loadu_ps(va + 16);
                    __m256 _va3 = _mm256_loadu_ps(va + 24);

                    _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0); // sum0 += (k00-k70) * a00
                    _sum1 = _mm256_fmadd_ps(_va1, _vb1, _sum1); // sum1 += (k01-k71) * a10
                    _sum2 = _mm256_fmadd_ps(_va2, _vb2, _sum2); // sum2 += (k02-k72) * a20
                    _sum3 = _mm256_fmadd_ps(_va3, _vb3, _sum3); // sum3 += (k03-k73) * a30

                    va += 32;
                    vb += 4;
                }

                _sum0 = _mm256_add_ps(_sum0, _sum1);
                _sum2 = _mm256_add_ps(_sum2, _sum3);
                _sum0_7 = _mm256_add_ps(_sum0_7, _sum0);
                _sum0_7 = _mm256_add_ps(_sum0_7, _sum2);

                for (; k < L; k++)
                {
                    __m256 _vb0 = _mm256_broadcast_ss(vb);
                    __m256 _va = _mm256_loadu_ps(va);

                    _sum0_7 = _mm256_fmadd_ps(_va, _vb0, _sum0_7); // sum0 += (k00-k70) * a00

                    va += 8;
                    vb += 1;
                }

                float output_sum0_7[8] = {0.f};
                _mm256_storeu_ps(output_sum0_7, _sum0_7);

                output0[0] = output_sum0_7[0];
                output1[0] = output_sum0_7[1];
                output2[0] = output_sum0_7[2];
                output3[0] = output_sum0_7[3];
                output4[0] = output_sum0_7[4];
                output5[0] = output_sum0_7[5];
                output6[0] = output_sum0_7[6];
                output7[0] = output_sum0_7[7];
#else
                float sum0 = biasptr[0];
                float sum1 = biasptr[1];
                float sum2 = biasptr[2];
                float sum3 = biasptr[3];
                float sum4 = biasptr[4];
                float sum5 = biasptr[5];
                float sum6 = biasptr[6];
                float sum7 = biasptr[7];

                for (int k = 0; k < L; k++)
                {
                    sum0 += va[0] * vb[0];
                    sum1 += va[1] * vb[0];
                    sum2 += va[2] * vb[0];
                    sum3 += va[3] * vb[0];
                    sum4 += va[4] * vb[0];
                    sum5 += va[5] * vb[0];
                    sum6 += va[6] * vb[0];
                    sum7 += va[7] * vb[0];

                    va += 8;
                    vb += 1;
                }

                output0[0] = sum0;
                output1[0] = sum1;
                output2[0] = sum2;
                output3[0] = sum3;
                output4[0] = sum4;
                output5[0] = sum5;
                output6[0] = sum6;
                output7[0] = sum7;
#endif // __AVX__
                output0++;
                output1++;
                output2++;
                output3++;
                output4++;
                output5++;
                output6++;
                output7++;
            }
        }

        nn_outch = (outch - remain_outch_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int i = remain_outch_start + pp * 4;

            float* output0 = top_blob.channel(i);
            float* output1 = top_blob.channel(i + 1);
            float* output2 = top_blob.channel(i + 2);
            float* output3 = top_blob.channel(i + 3);

            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + i : zeros;

            int j = 0;
            for (; j + 7 < N; j = j + 8)
            {
                const float* vb = bottom_tm.channel(j / 8);
                const float* va = kernel_tm.channel(i / 8 + (i % 8) / 4);
#if __AVX__
                __m256 _sum0 = _mm256_broadcast_ss(biasptr);
                __m256 _sum1 = _mm256_broadcast_ss(biasptr + 1);
                __m256 _sum2 = _mm256_broadcast_ss(biasptr + 2);
                __m256 _sum3 = _mm256_broadcast_ss(biasptr + 3);

                int k = 0;
                for (; k + 3 < L; k = k + 4)
                {
                    // k0
                    __m256 _va0 = _mm256_broadcast_ss(va);
                    __m256 _va1 = _mm256_broadcast_ss(va + 1);
                    __m256 _va2 = _mm256_broadcast_ss(va + 2);
                    __m256 _va3 = _mm256_broadcast_ss(va + 3);
                    __m256 _vb0 = _mm256_loadu_ps(vb);
                    __m256 _vb1 = _mm256_loadu_ps(vb + 8);
                    __m256 _vb2 = _mm256_loadu_ps(vb + 16);
                    __m256 _vb3 = _mm256_loadu_ps(vb + 24);
                    _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0); // sum0 = (a00-a07) * k00
                    _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1); // sum1 = (a00-a07) * k10
                    _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2); // sum2 = (a00-a07) * k20
                    _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3); // sum3 = (a00-a07) * k30

                    va += 4;

                    // k1
                    _va0 = _mm256_broadcast_ss(va);
                    _va1 = _mm256_broadcast_ss(va + 1);
                    _va2 = _mm256_broadcast_ss(va + 2);
                    _va3 = _mm256_broadcast_ss(va + 3);
                    _sum0 = _mm256_fmadd_ps(_vb1, _va0, _sum0); // sum0 += (a10-a17) * k01
                    _sum1 = _mm256_fmadd_ps(_vb1, _va1, _sum1); // sum1 += (a10-a17) * k11
                    _sum2 = _mm256_fmadd_ps(_vb1, _va2, _sum2); // sum2 += (a10-a17) * k21
                    _sum3 = _mm256_fmadd_ps(_vb1, _va3, _sum3); // sum3 += (a10-a17) * k31

                    va += 4;

                    // k2
                    _va0 = _mm256_broadcast_ss(va);
                    _va1 = _mm256_broadcast_ss(va + 1);
                    _va2 = _mm256_broadcast_ss(va + 2);
                    _va3 = _mm256_broadcast_ss(va + 3);
                    _sum0 = _mm256_fmadd_ps(_vb2, _va0, _sum0); // sum0 += (a20-a27) * k02
                    _sum1 = _mm256_fmadd_ps(_vb2, _va1, _sum1); // sum1 += (a20-a27) * k12
                    _sum2 = _mm256_fmadd_ps(_vb2, _va2, _sum2); // sum2 += (a20-a27) * k22
                    _sum3 = _mm256_fmadd_ps(_vb2, _va3, _sum3); // sum3 += (a20-a27) * k32

                    va += 4;

                    // k3
                    _va0 = _mm256_broadcast_ss(va);
                    _va1 = _mm256_broadcast_ss(va + 1);
                    _va2 = _mm256_broadcast_ss(va + 2);
                    _va3 = _mm256_broadcast_ss(va + 3);
                    _sum0 = _mm256_fmadd_ps(_vb3, _va0, _sum0); // sum0 += (a30-a37) * k03
                    _sum1 = _mm256_fmadd_ps(_vb3, _va1, _sum1); // sum1 += (a30-a37) * k13
                    _sum2 = _mm256_fmadd_ps(_vb3, _va2, _sum2); // sum2 += (a30-a37) * k23
                    _sum3 = _mm256_fmadd_ps(_vb3, _va3, _sum3); // sum3 += (a30-a37) * k33

                    va += 4;
                    vb += 32;
                }

                for (; k < L; k++)
                {
                    // k0
                    __m256 _va0 = _mm256_broadcast_ss(va);
                    __m256 _va1 = _mm256_broadcast_ss(va + 1);
                    __m256 _va2 = _mm256_broadcast_ss(va + 2);
                    __m256 _va3 = _mm256_broadcast_ss(va + 3);
                    __m256 _vb0 = _mm256_loadu_ps(vb);
                    _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0); // sum0 = (a00-a07) * k00
                    _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1); // sum1 = (a00-a07) * k10
                    _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2); // sum2 = (a00-a07) * k20
                    _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3); // sum3 = (a00-a07) * k30

                    va += 4;
                    vb += 8;
                }

                _mm256_storeu_ps(output0, _sum0);
                _mm256_storeu_ps(output1, _sum1);
                _mm256_storeu_ps(output2, _sum2);
                _mm256_storeu_ps(output3, _sum3);
#else
                float sum0[8] = {0};
                float sum1[8] = {0};
                float sum2[8] = {0};
                float sum3[8] = {0};

                int k = 0;
                for (; k + 7 < L; k = k + 8)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        sum0[n] += va[0] * vb[n];
                        sum1[n] += va[1] * vb[n];
                        sum2[n] += va[2] * vb[n];
                        sum3[n] += va[3] * vb[n];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 8];
                        sum1[n] += va[1] * vb[n + 8];
                        sum2[n] += va[2] * vb[n + 8];
                        sum3[n] += va[3] * vb[n + 8];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 16];
                        sum1[n] += va[1] * vb[n + 16];
                        sum2[n] += va[2] * vb[n + 16];
                        sum3[n] += va[3] * vb[n + 16];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 24];
                        sum1[n] += va[1] * vb[n + 24];
                        sum2[n] += va[2] * vb[n + 24];
                        sum3[n] += va[3] * vb[n + 24];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 32];
                        sum1[n] += va[1] * vb[n + 32];
                        sum2[n] += va[2] * vb[n + 32];
                        sum3[n] += va[3] * vb[n + 32];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 40];
                        sum1[n] += va[1] * vb[n + 40];
                        sum2[n] += va[2] * vb[n + 40];
                        sum3[n] += va[3] * vb[n + 40];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 48];
                        sum1[n] += va[1] * vb[n + 48];
                        sum2[n] += va[2] * vb[n + 48];
                        sum3[n] += va[3] * vb[n + 48];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 56];
                        sum1[n] += va[1] * vb[n + 56];
                        sum2[n] += va[2] * vb[n + 56];
                        sum3[n] += va[3] * vb[n + 56];
                        va -= 28;
                    }

                    va += 32;
                    vb += 64;
                }

                for (; k < L; k++)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        sum0[n] += va[0] * vb[n];
                        sum1[n] += va[1] * vb[n];
                        sum2[n] += va[2] * vb[n];
                        sum3[n] += va[3] * vb[n];
                    }

                    va += 4;
                    vb += 8;
                }

                for (int n = 0; n < 8; n++)
                {
                    output0[n] = sum0[n] + biasptr[0];
                    output1[n] = sum1[n] + biasptr[1];
                    output2[n] = sum2[n] + biasptr[2];
                    output3[n] = sum3[n] + biasptr[3];
                }
#endif // __AVX__
                output0 += 8;
                output1 += 8;
                output2 += 8;
                output3 += 8;
            }

            for (; j < N; j++)
            {
                const float* vb = bottom_tm.channel(j / 8 + j % 8);
                const float* va = kernel_tm.channel(i / 8 + (i % 8) / 4);
#if __AVX__
                __m128 _sum0_3 = _mm_loadu_ps(biasptr);
                __m128 _sum0 = _mm_set1_ps(0.0);
                __m128 _sum1 = _mm_set1_ps(0.0);
                __m128 _sum2 = _mm_set1_ps(0.0);
                __m128 _sum3 = _mm_set1_ps(0.0);

                int k = 0;
                for (; k + 3 < L; k = k + 4)
                {
                    __m128 _vb0 = _mm_set1_ps(vb[0]);
                    __m128 _vb1 = _mm_set1_ps(vb[1]);
                    __m128 _vb2 = _mm_set1_ps(vb[2]);
                    __m128 _vb3 = _mm_set1_ps(vb[3]);
                    __m128 _va0 = _mm_loadu_ps(va);
                    __m128 _va1 = _mm_loadu_ps(va + 4);
                    __m128 _va2 = _mm_loadu_ps(va + 8);
                    __m128 _va3 = _mm_loadu_ps(va + 12);

                    _sum0 = _mm_fmadd_ps(_va0, _vb0, _sum0); // sum0 += (k00-k30) * a00
                    _sum1 = _mm_fmadd_ps(_va1, _vb1, _sum1); // sum1 += (k01-k31) * a10
                    _sum2 = _mm_fmadd_ps(_va2, _vb2, _sum2); // sum2 += (k02-k32) * a20
                    _sum3 = _mm_fmadd_ps(_va3, _vb3, _sum3); // sum3 += (k03-k33) * a30

                    va += 16;
                    vb += 4;
                }

                _sum0 = _mm_add_ps(_sum0, _sum1);
                _sum2 = _mm_add_ps(_sum2, _sum3);
                _sum0_3 = _mm_add_ps(_sum0_3, _sum0);
                _sum0_3 = _mm_add_ps(_sum0_3, _sum2);

                for (; k < L; k++)
                {
                    __m128 _vb0 = _mm_set1_ps(vb[0]);
                    __m128 _va = _mm_loadu_ps(va);

                    _sum0_3 = _mm_fmadd_ps(_va, _vb0, _sum0_3); // sum0 += (k00-k30) * a00

                    va += 4;
                    vb += 1;
                }

                float output_sum0_3[4] = {0.f};
                _mm_storeu_ps(output_sum0_3, _sum0_3);
                output0[0] = output_sum0_3[0];
                output1[0] = output_sum0_3[1];
                output2[0] = output_sum0_3[2];
                output3[0] = output_sum0_3[3];
#else
                float sum0 = biasptr[0];
                float sum1 = biasptr[1];
                float sum2 = biasptr[2];
                float sum3 = biasptr[3];

                for (int k = 0; k < L; k++)
                {
                    sum0 += va[0] * vb[0];
                    sum1 += va[1] * vb[0];
                    sum2 += va[2] * vb[0];
                    sum3 += va[3] * vb[0];

                    va += 4;
                    vb += 1;
                }

                output0[0] = sum0;
                output1[0] = sum1;
                output2[0] = sum2;
                output3[0] = sum3;
#endif // __AVX__
                output0++;
                output1++;
                output2++;
                output3++;
            }
        }

        remain_outch_start += nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_outch_start; i < outch; i++)
        {
            float* output = top_blob.channel(i);

            const float bias0 = bias ? bias[i] : 0.f;

            int j = 0;
            for (; j + 7 < N; j = j + 8)
            {
                const float* vb = bottom_tm.channel(j / 8);
                const float* va = kernel_tm.channel(i / 8 + (i % 8) / 4 + i % 4);
#if __AVX__
                __m256 _sum0 = _mm256_broadcast_ss(&bias0);

                int k = 0;
                for (; k + 3 < L; k = k + 4)
                {
                    // k0
                    __m256 _va0 = _mm256_broadcast_ss(va);
                    __m256 _va1 = _mm256_broadcast_ss(va + 1);
                    __m256 _va2 = _mm256_broadcast_ss(va + 2);
                    __m256 _va3 = _mm256_broadcast_ss(va + 3);
                    __m256 _vb0 = _mm256_loadu_ps(vb);
                    __m256 _vb1 = _mm256_loadu_ps(vb + 8);
                    __m256 _vb2 = _mm256_loadu_ps(vb + 16);
                    __m256 _vb3 = _mm256_loadu_ps(vb + 24);

                    _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0); // sum0 = (a00-a07) * k00
                    _sum0 = _mm256_fmadd_ps(_vb1, _va1, _sum0); // sum0 += (a10-a17) * k01
                    _sum0 = _mm256_fmadd_ps(_vb2, _va2, _sum0); // sum0 += (a20-a27) * k02
                    _sum0 = _mm256_fmadd_ps(_vb3, _va3, _sum0); // sum0 += (a30-a37) * k03

                    va += 4;
                    vb += 32;
                }

                for (; k < L; k++)
                {
                    // k0
                    __m256 _va0 = _mm256_broadcast_ss(va);
                    __m256 _vb0 = _mm256_loadu_ps(vb);

                    _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0); // sum0 = (a00-a07) * k00

                    va += 1;
                    vb += 8;
                }

                _mm256_storeu_ps(output, _sum0);
#else
                float sum[8] = {0};

                int k = 0;
                for (; k + 7 < L; k = k + 8)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        sum[n] += va[0] * vb[n];
                        sum[n] += va[1] * vb[n + 8];
                        sum[n] += va[2] * vb[n + 16];
                        sum[n] += va[3] * vb[n + 24];
                        sum[n] += va[4] * vb[n + 32];
                        sum[n] += va[5] * vb[n + 40];
                        sum[n] += va[6] * vb[n + 48];
                        sum[n] += va[7] * vb[n + 56];
                    }

                    va += 8;
                    vb += 64;
                }

                for (; k < L; k++)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        sum[n] += va[0] * vb[n];
                    }

                    va += 1;
                    vb += 8;
                }

                for (int n = 0; n < 8; n++)
                {
                    output[n] = sum[n] + bias0;
                }
#endif // __AVX__
                output += 8;
            }

            for (; j < N; j++)
            {
                const float* vb = bottom_tm.channel(j / 8 + j % 8);
                const float* va = kernel_tm.channel(i / 8 + (i % 8) / 4 + i % 4);

                int k = 0;
#if __AVX__
                __m128 _sum0 = _mm_set1_ps(0.f);

                for (; k + 3 < L; k += 4)
                {
                    __m128 _p0 = _mm_loadu_ps(vb);
                    vb += 4;

                    __m128 _k0 = _mm_loadu_ps(va);
                    va += 4;

                    _sum0 = _mm_fmadd_ps(_p0, _k0, _sum0);
                }

                float output_sum0[4] = {0.f};
                _mm_storeu_ps(output_sum0, _sum0);

                float sum0 = bias0 + output_sum0[0] + output_sum0[1] + output_sum0[2] + output_sum0[3];

#else
                float sum0 = bias0;
#endif // __AVX__
                for (; k < L; k++)
                {
                    sum0 += va[0] * vb[0];

                    va += 1;
                    vb += 1;
                }
                output[0] = sum0;

                output++;
            }
        }
    }
}
#else
static void conv_im2col_sgemm_transform_kernel_sse(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_size)
{
    const float* kernel = _kernel;

    // kernel memory packed 4 x 4
    kernel_tm.create(4 * kernel_size, inch, outch / 4 + outch % 4);

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 2;
    remain_outch_start = nn_outch << 2;

    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        const float* k0 = kernel + (p + 0) * inch * kernel_size;
        const float* k1 = kernel + (p + 1) * inch * kernel_size;
        const float* k2 = kernel + (p + 2) * inch * kernel_size;
        const float* k3 = kernel + (p + 3) * inch * kernel_size;

        float* ktmp = kernel_tm.channel(p / 4);

        for (int q = 0; q < inch * kernel_size; q++)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];
            ktmp += 4;

            k0 += 1;
            k1 += 1;
            k2 += 1;
            k3 += 1;
        }
    }

    for (int p = remain_outch_start; p < outch; p++)
    {
        const float* k0 = kernel + (p + 0) * inch * kernel_size;

        float* ktmp = kernel_tm.channel(p / 4 + p % 4);

        for (int q = 0; q < inch * kernel_size; q++)
        {
            ktmp[0] = k0[0];
            ktmp++;
            k0++;
        }
    }
}

static void conv_im2col_sgemm_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias,
                                  const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

    // im2col
    Mat bottom_im2col(outw * outh, kernel_h * kernel_w * inch, elemsize, opt.workspace_allocator);
    {
        const int stride = kernel_h * kernel_w * outw * outh;
        float* ret = (float*)bottom_im2col;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const float* input = bottom_blob.channel(p);
            int retID = stride * p;
            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            int row = u + i * stride_h;
                            int col = v + j * stride_w;
                            int index = row * w + col;
                            ret[retID] = input[index];
                            retID++;
                        }
                    }
                }
            }
        }
    }

    int kernel_size = kernel_w * kernel_h;
    int out_size = outw * outh;

    // bottom_im2col memory packed 4 x 4
    Mat bottom_tm(4 * kernel_size, inch, out_size / 4 + out_size % 4, elemsize, opt.workspace_allocator);
    {
        int nn_size = out_size >> 2;
        int remain_size_start = nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 4;

            const float* img0 = bottom_im2col.channel(0);
            img0 += i;

            float* tmpptr = bottom_tm.channel(i / 4);

            for (int q = 0; q < inch * kernel_size; q++)
            {
#if __SSE__
                _mm_storeu_ps(tmpptr, _mm_loadu_ps(img0));
#else
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];
#endif // __SSE__
                tmpptr += 4;
                img0 += out_size;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < out_size; i++)
        {
            const float* img0 = bottom_im2col.channel(0);
            img0 += i;

            float* tmpptr = bottom_tm.channel(i / 4 + i % 4);

            for (int q = 0; q < inch * kernel_size; q++)
            {
                tmpptr[0] = img0[0];

                tmpptr += 1;
                img0 += out_size;
            }
        }
    }

    // sgemm(int M, int N, int L, float* A, float* B, float* C)
    {
        //int M = outch;                    // outch
        int N = outw * outh;                // outsize or out stride
        int L = kernel_w * kernel_h * inch; // ksize * inch

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;
        remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int i = pp * 4;

            float* output0 = top_blob.channel(i);
            float* output1 = top_blob.channel(i + 1);
            float* output2 = top_blob.channel(i + 2);
            float* output3 = top_blob.channel(i + 3);

            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + i : zeros;

            int j = 0;
            for (; j + 3 < N; j = j + 4)
            {
                const float* vb = bottom_tm.channel(j / 4);
                const float* va = kernel_tm.channel(i / 4);
#if __SSE__
                __m128 _sum0 = _mm_set1_ps(biasptr[0]);
                __m128 _sum1 = _mm_set1_ps(biasptr[1]);
                __m128 _sum2 = _mm_set1_ps(biasptr[2]);
                __m128 _sum3 = _mm_set1_ps(biasptr[3]);

                int k = 0;
                for (; k + 3 < L; k = k + 4)
                {
                    // k0
                    __m128 _vb = _mm_loadu_ps(vb);
                    __m128 _va0 = _mm_set1_ps(va[0]);
                    __m128 _va1 = _mm_set1_ps(va[1]);
                    __m128 _va2 = _mm_set1_ps(va[2]);
                    __m128 _va3 = _mm_set1_ps(va[3]);
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0)); // sum0 = (a00-a03) * k00
                    _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1)); // sum1 = (a00-a03) * k10
                    _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2)); // sum2 = (a00-a03) * k20
                    _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3)); // sum3 = (a00-a03) * k30

                    // k1
                    _vb = _mm_loadu_ps(vb + 4);
                    _va0 = _mm_set1_ps(va[4]);
                    _va1 = _mm_set1_ps(va[5]);
                    _va2 = _mm_set1_ps(va[6]);
                    _va3 = _mm_set1_ps(va[7]);
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0)); // sum0 = (a10-a13) * k01
                    _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1)); // sum1 = (a10-a13) * k11
                    _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2)); // sum2 = (a10-a13) * k21
                    _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3)); // sum3 = (a10-a13) * k31

                    // k2
                    _vb = _mm_loadu_ps(vb + 8);
                    _va0 = _mm_set1_ps(va[8]);
                    _va1 = _mm_set1_ps(va[9]);
                    _va2 = _mm_set1_ps(va[10]);
                    _va3 = _mm_set1_ps(va[11]);
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0)); // sum0 = (a20-a23) * k02
                    _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1)); // sum1 = (a20-a23) * k12
                    _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2)); // sum2 = (a20-a23) * k22
                    _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3)); // sum3 = (a20-a23) * k32

                    // k3
                    _vb = _mm_loadu_ps(vb + 12);
                    _va0 = _mm_set1_ps(va[12]);
                    _va1 = _mm_set1_ps(va[13]);
                    _va2 = _mm_set1_ps(va[14]);
                    _va3 = _mm_set1_ps(va[15]);
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0)); // sum0 = (a30-a33) * k03
                    _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1)); // sum1 = (a30-a33) * k13
                    _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2)); // sum2 = (a30-a33) * k23
                    _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3)); // sum3 = (a30-a33) * k33

                    va += 16;
                    vb += 16;
                }

                for (; k < L; k++)
                {
                    // k0
                    __m128 _vb = _mm_loadu_ps(vb);
                    __m128 _va0 = _mm_set1_ps(va[0]);
                    __m128 _va1 = _mm_set1_ps(va[1]);
                    __m128 _va2 = _mm_set1_ps(va[2]);
                    __m128 _va3 = _mm_set1_ps(va[3]);
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb, _va0)); // sum0 = (a00-a03) * k00
                    _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_vb, _va1)); // sum1 = (a00-a03) * k10
                    _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_vb, _va2)); // sum2 = (a00-a03) * k20
                    _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_vb, _va3)); // sum3 = (a00-a03) * k30

                    va += 4;
                    vb += 4;
                }
                _mm_storeu_ps(output0, _sum0);
                _mm_storeu_ps(output1, _sum1);
                _mm_storeu_ps(output2, _sum2);
                _mm_storeu_ps(output3, _sum3);
#else
                float sum0[4] = {0};
                float sum1[4] = {0};
                float sum2[4] = {0};
                float sum3[4] = {0};

                int k = 0;
                for (; k + 7 < L; k = k + 8)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum0[n] += va[0] * vb[n];
                        sum1[n] += va[1] * vb[n];
                        sum2[n] += va[2] * vb[n];
                        sum3[n] += va[3] * vb[n];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 4];
                        sum1[n] += va[1] * vb[n + 4];
                        sum2[n] += va[2] * vb[n + 4];
                        sum3[n] += va[3] * vb[n + 4];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 8];
                        sum1[n] += va[1] * vb[n + 8];
                        sum2[n] += va[2] * vb[n + 8];
                        sum3[n] += va[3] * vb[n + 8];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 12];
                        sum1[n] += va[1] * vb[n + 12];
                        sum2[n] += va[2] * vb[n + 12];
                        sum3[n] += va[3] * vb[n + 12];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 16];
                        sum1[n] += va[1] * vb[n + 16];
                        sum2[n] += va[2] * vb[n + 16];
                        sum3[n] += va[3] * vb[n + 16];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 20];
                        sum1[n] += va[1] * vb[n + 20];
                        sum2[n] += va[2] * vb[n + 20];
                        sum3[n] += va[3] * vb[n + 20];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 24];
                        sum1[n] += va[1] * vb[n + 24];
                        sum2[n] += va[2] * vb[n + 24];
                        sum3[n] += va[3] * vb[n + 24];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 28];
                        sum1[n] += va[1] * vb[n + 28];
                        sum2[n] += va[2] * vb[n + 28];
                        sum3[n] += va[3] * vb[n + 28];
                        va -= 28;
                    }

                    va += 32;
                    vb += 32;
                }

                for (; k < L; k++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum0[n] += va[0] * vb[n];
                        sum1[n] += va[1] * vb[n];
                        sum2[n] += va[2] * vb[n];
                        sum3[n] += va[3] * vb[n];
                    }

                    va += 4;
                    vb += 4;
                }

                for (int n = 0; n < 4; n++)
                {
                    output0[n] = sum0[n] + biasptr[0];
                    output1[n] = sum1[n] + biasptr[1];
                    output2[n] = sum2[n] + biasptr[2];
                    output3[n] = sum3[n] + biasptr[3];
                }
#endif // __SSE__
                output0 += 4;
                output1 += 4;
                output2 += 4;
                output3 += 4;
            }

            for (; j < N; j++)
            {
                const float* vb = bottom_tm.channel(j / 4 + j % 4);
                const float* va = kernel_tm.channel(i / 4);
#if __SSE__
                __m128 _sum0_3 = _mm_loadu_ps(biasptr);
                __m128 _sum0 = _mm_set1_ps(0.0);
                __m128 _sum1 = _mm_set1_ps(0.0);
                __m128 _sum2 = _mm_set1_ps(0.0);
                __m128 _sum3 = _mm_set1_ps(0.0);

                int k = 0;
                for (; k + 3 < L; k = k + 4)
                {
                    __m128 _vb0 = _mm_set1_ps(vb[0]);
                    __m128 _vb1 = _mm_set1_ps(vb[1]);
                    __m128 _vb2 = _mm_set1_ps(vb[2]);
                    __m128 _vb3 = _mm_set1_ps(vb[3]);
                    __m128 _va0 = _mm_loadu_ps(va);
                    __m128 _va1 = _mm_loadu_ps(va + 4);
                    __m128 _va2 = _mm_loadu_ps(va + 8);
                    __m128 _va3 = _mm_loadu_ps(va + 12);

                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va0, _vb0)); // sum0 += (k00-k30) * a00
                    _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_va1, _vb1)); // sum1 += (k01-k31) * a10
                    _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_va2, _vb2)); // sum2 += (k02-k32) * a20
                    _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_va3, _vb3)); // sum3 += (k03-k33) * a30

                    va += 16;
                    vb += 4;
                }

                _sum0 = _mm_add_ps(_sum0, _sum1);
                _sum2 = _mm_add_ps(_sum2, _sum3);
                _sum0_3 = _mm_add_ps(_sum0_3, _sum0);
                _sum0_3 = _mm_add_ps(_sum0_3, _sum2);

                for (; k < L; k++)
                {
                    __m128 _vb0 = _mm_set1_ps(vb[0]);
                    __m128 _va = _mm_loadu_ps(va);

                    _sum0_3 = _mm_add_ps(_sum0_3, _mm_mul_ps(_va, _vb0)); // sum0 += (k00-k30) * a00

                    va += 4;
                    vb += 1;
                }
                output0[0] = _sum0_3[0];
                output1[0] = _sum0_3[1];
                output2[0] = _sum0_3[2];
                output3[0] = _sum0_3[3];
#else
                float sum0 = biasptr[0];
                float sum1 = biasptr[1];
                float sum2 = biasptr[2];
                float sum3 = biasptr[3];

                for (int k = 0; k < L; k++)
                {
                    sum0 += va[0] * vb[0];
                    sum1 += va[1] * vb[0];
                    sum2 += va[2] * vb[0];
                    sum3 += va[3] * vb[0];

                    va += 4;
                    vb += 1;
                }

                output0[0] = sum0;
                output1[0] = sum1;
                output2[0] = sum2;
                output3[0] = sum3;
#endif // __SSE__
                output0++;
                output1++;
                output2++;
                output3++;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_outch_start; i < outch; i++)
        {
            float* output = top_blob.channel(i);

            const float bias0 = bias ? bias[i] : 0.f;

            int j = 0;
            for (; j + 3 < N; j = j + 4)
            {
                const float* vb = bottom_tm.channel(j / 4);
                const float* va = kernel_tm.channel(i / 4 + i % 4);
#if __SSE__
                __m128 _sum0 = _mm_set1_ps(bias0);

                int k = 0;
                for (; k + 3 < L; k = k + 4)
                {
                    // k0
                    __m128 _va0 = _mm_set1_ps(va[0]);
                    __m128 _va1 = _mm_set1_ps(va[1]);
                    __m128 _va2 = _mm_set1_ps(va[2]);
                    __m128 _va3 = _mm_set1_ps(va[3]);
                    __m128 _vb0 = _mm_loadu_ps(vb);
                    __m128 _vb1 = _mm_loadu_ps(vb + 4);
                    __m128 _vb2 = _mm_loadu_ps(vb + 8);
                    __m128 _vb3 = _mm_loadu_ps(vb + 12);

                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb0, _va0)); // sum0 = (a00-a03) * k00
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb1, _va1)); // sum0 += (a10-a13) * k01
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb2, _va2)); // sum0 += (a20-a23) * k02
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb3, _va3)); // sum0 += (a30-a33) * k03

                    va += 4;
                    vb += 16;
                }

                for (; k < L; k++)
                {
                    // k0
                    __m128 _va0 = _mm_set1_ps(va[0]);
                    __m128 _vb0 = _mm_loadu_ps(vb);

                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_vb0, _va0)); // sum0 = (a00-a03) * k00

                    va += 1;
                    vb += 4;
                }
                _mm_storeu_ps(output, _sum0);
#else
                float sum[4] = {0};

                int k = 0;
                for (; k + 3 < L; k = k + 4)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum[n] += va[0] * vb[n];
                        sum[n] += va[1] * vb[n + 4];
                        sum[n] += va[2] * vb[n + 8];
                        sum[n] += va[3] * vb[n + 12];
                        //sum[n] += va[4] * vb[n+16];
                        //sum[n] += va[5] * vb[n+20];
                        //sum[n] += va[6] * vb[n+24];
                        //sum[n] += va[7] * vb[n+28];
                    }

                    va += 4;
                    vb += 16;
                }

                for (; k < L; k++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum[n] += va[0] * vb[n];
                    }

                    va += 1;
                    vb += 4;
                }

                for (int n = 0; n < 4; n++)
                {
                    output[n] = sum[n] + bias0;
                }
#endif // __SSE__
                output += 4;
            }

            for (; j < N; j++)
            {
                const float* vb = bottom_tm.channel(j / 4 + j % 4);
                const float* va = kernel_tm.channel(i / 4 + i % 4);

                int k = 0;
#if __SSE__
                __m128 _sum0 = _mm_set1_ps(0.f);

                for (; k + 3 < L; k += 4)
                {
                    __m128 _p0 = _mm_loadu_ps(vb);
                    __m128 _k0 = _mm_loadu_ps(va);
                    _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_p0, _k0));

                    va += 4;
                    vb += 4;
                }
                float sum0 = bias0 + _sum0[0] + _sum0[1] + _sum0[2] + _sum0[3];
#else
                float sum0 = bias0;
#endif // __SSE__
                for (; k < L; k++)
                {
                    sum0 += va[0] * vb[0];

                    va += 1;
                    vb += 1;
                }
                output[0] = sum0;

                output++;
            }
        }
    }
}
#endif
